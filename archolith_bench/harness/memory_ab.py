"""Mode-B driver: persistent-memory benchmarks via ingest-then-recall.

Unlike `run_ab` (history in-context) this exercises a real memory system. Per
question item, for each memory arm: isolate a fresh `group_id`, ingest the
haystack sessions, recall against the question, feed the recalled memory (not the
raw history) to the chat model, and score. The advertisable claim is the
memory-QA accuracy lift from the no-memory baseline to memory-recall.

The memory backend is abstracted behind `MenhirClient` so offline tests use a
deterministic in-memory stub (no Neo4j); real runs point at a throwaway
menhir+Neo4j (never production — `assert_not_production` guards the target).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from contextlib import nullcontext
from statistics import mean
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from .checkpoint import MemoryCheckpoint

from ..core.api import API_KEY, MODEL, PROXY_URL, send_chat
from ..core.metrics import PricingModel, compute_arm_cost
from .base import ABResult, ArmResult, TaskResult, _compute_deltas, _pick_pricing, _usage_tokens
from .menhir_client import HttpMenhirClient
from .value_nodes import ValueGraph
from .value_nodes_v2 import SupersededValueGraph

# Arms: a no-memory floor vs memory-recall arms.
NO_MEMORY = "no_memory"
SINGLE_RECALL = "menhir_recall"
# Agentic recall: the answer LLM first decomposes the question into focused
# entity/keyword sub-queries and recalls each, instead of embedding the whole
# question as one query. menhir's recall is semantic vector search, so a verbose
# multi-entity question yields a blended embedding that often retrieves one event
# but not the other; per-entity queries retrieve both.
AGENTIC_RECALL = "menhir_agentic_recall"
VALUE_RECALL = "menhir_value_recall"
# Supersession-aware value arms (v2). EXPERIMENTAL - REJECTED (2026-07-18): did not beat
# v1 (0/5 targeted misses recovered); kept only as documented negative evidence and never
# included in DEFAULT_MEMORY_ARMS or any default config. Opt-in only via explicit --arms.
# Pre-registered as two separate arms so the current-only vs current+history emission choice
# was reported across all items, not selected post-hoc. See value_nodes_v2.SupersededValueGraph.
VALUE_RECALL_V2_CURRENT = "menhir_value_recall_v2_current"
VALUE_RECALL_V2_HISTORY = "menhir_value_recall_v2_history"
# v3 authoritative-composition experiment (EXPERIMENTAL; opt-in only). Simulates what an
# authoritative typed View would do during recall composition, in the bench:
#   _v3_coarse        = coarse (near-oracle) grouping + current-only typed snippets. Isolates
#                       the grouping ceiling (if clustering were perfect, does supersession help?).
#   _v3_authoritative = coarse grouping + current-only + SUPPRESS untyped Menhir snippets that
#                       carry a superseded value (the full "typed value is authoritative" test).
VALUE_RECALL_V3_COARSE = "menhir_value_recall_v3_coarse"
VALUE_RECALL_V3_AUTHORITATIVE = "menhir_value_recall_v3_authoritative"
# v4 advisory composition (EXPERIMENTAL; opt-in only). The v3 result showed authoritative
# single-pick + suppression nets negative because every regression came from DELETING the
# correct candidate. v4 instead ADVISES: coarse grouping to find the same candidate set, but
# keep every candidate (additive, so answer-support >= v1), annotate each with its cluster
# role (current/earlier/superseded when an explicit correction marker is present; otherwise
# "candidate"), and hard-drop ONLY an author-rejected value. The answer LLM makes the final
# pick. Determinism is reserved for author-stated corrections; everything else is a hint.
VALUE_RECALL_V4_ADVISORY = "menhir_value_recall_v4_advisory"
# v5 derivation / "assumptions" arm (EXPERIMENTAL; opt-in only). Advisory base (v4) plus a
# deterministic delta fold: when a single stated count anchor and signed increment/decrement
# events resolve to an integer-consistent value that is never stated literally (69fee5aa:
# "37 coins" + "added a new coin" -> 38), surface it as a labeled DERIVED hint. Never
# authoritative, never deletes a stated value; gated hard toward silence to avoid confident-
# wrong arithmetic. See value_nodes_v2.SupersededValueGraph.derived_recall.
VALUE_RECALL_V5_DERIVED = "menhir_value_recall_v5_derived"
DEFAULT_MEMORY_ARMS = (NO_MEMORY, SINGLE_RECALL)  # v2/v3/v4/v5 arms intentionally excluded

# Every arm the dispatcher actually understands. `_value_context_for_arm` ends in
# a plain-recall fallback, so an unrecognised arm name does not fail -- it runs
# single recall and labels the result with whatever string was passed, which then
# reaches an evidence artifact as a measurement of an arm that never ran.
# Validated up front so a typo is a loud error instead of quiet fiction.
VALID_MEMORY_ARMS: frozenset[str] = frozenset({
    NO_MEMORY,
    SINGLE_RECALL,
    AGENTIC_RECALL,
    VALUE_RECALL,
    VALUE_RECALL_V2_CURRENT,
    VALUE_RECALL_V2_HISTORY,
    VALUE_RECALL_V3_COARSE,
    VALUE_RECALL_V3_AUTHORITATIVE,
    VALUE_RECALL_V4_ADVISORY,
    VALUE_RECALL_V5_DERIVED,
})


def assert_known_arms(arms: Sequence[str]) -> None:
    """Reject arm names the dispatcher does not implement."""
    unknown = [a for a in arms if a not in VALID_MEMORY_ARMS]
    if unknown:
        raise ValueError(
            f"unknown memory arm(s): {', '.join(sorted(unknown))}. "
            f"Valid arms: {', '.join(sorted(VALID_MEMORY_ARMS))}"
        )

_PLANNER_SYSTEM = (
    "You turn a user's question into focused memory-search queries. The memory system is a "
    "semantic graph that retrieves best from SHORT entity/keyword queries, not full sentences. "
    "For questions that compare or relate multiple things (e.g. 'which came first, X or Y', "
    "'how many days between A and B'), output ONE separate query per thing/event so each is "
    "retrieved independently. Return ONLY a JSON array of 1-5 short query strings, most "
    "important first. Example: [\"Samsung Galaxy S22 purchase\", \"Dell XPS 13 purchase\"]"
)


def _parse_query_list(text: str) -> list[str]:
    """Extract a JSON array of query strings from the planner reply (lenient)."""
    if not text:
        return []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    raw = match.group(0) if match else text
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, list):
        return [str(q).strip() for q in data if str(q).strip()]
    return []


def _plan_recall_queries(
    question: str, *, send_fn, chat_client, chat_base_url: str, api_key: str, model: str  # noqa: ANN001
) -> list[str]:
    """Ask the answer LLM for focused sub-queries; always keep the raw question as a
    fallback so agentic recall never does strictly worse than single-query recall."""
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        text, _latency, _usage = send_fn(chat_client, chat_base_url, api_key, messages, model)
        queries = _parse_query_list(text)
    except Exception:
        queries = []
    if question and question not in queries:
        queries.append(question)
    return queries[:6]


def _agentic_recall(client: "MenhirClient", group_id: str, subqueries: list[str], recall_limit: int) -> str:
    """Recall each sub-query and union the snippets (dedup, order-preserving), capped at
    recall_limit total so the answer context stays size-comparable to single-query recall."""
    per_q = max(3, recall_limit // max(1, len(subqueries)))
    seen: set[str] = set()
    merged: list[str] = []
    for sq in subqueries:
        try:
            snippets = client.recall(group_id, sq, limit=per_q)
        except Exception:
            snippets = []
        for snip in snippets:
            if snip not in seen:
                seen.add(snip)
                merged.append(snip)
                if len(merged) >= recall_limit:
                    return "\n".join(merged)
    return "\n".join(merged)


def _value_augmented_recall(
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
) -> str:
    """Combine typed value assertions with ordinary recall at equal total top-k.

    The sidecar graph is built from user haystack turns only. Value snippets are
    placed first and ordinary recall backfills the remaining slots. The final
    context never exceeds recall_limit snippets.
    """
    graph = ValueGraph.from_item(f"value-{task_id}", item, adapter.sessions(item))
    value_limit = min(4, max(1, recall_limit // 3))
    value_snippets = graph.recall(question, limit=value_limit)
    merged: list[str] = []
    seen: set[str] = set()
    for snippet in [*value_snippets, *recalled]:
        if snippet in seen:
            continue
        seen.add(snippet)
        merged.append(snippet)
        if len(merged) >= recall_limit:
            break
    return "\n".join(merged)


def _value_augmented_recall_v2(
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
    *,
    emit_history: bool,
) -> str:
    """v2 value augmentation: supersession-aware CURRENT selection, same total top-k.

    Identical merge/cap contract as ``_value_augmented_recall`` (value snippets first,
    ordinary recall backfills, never exceeding recall_limit). The only difference is the
    sidecar graph type (SupersededValueGraph) and the ``emit_history`` emission mode.
    """
    graph = SupersededValueGraph.from_item(f"value2-{task_id}", item, adapter.sessions(item))
    value_limit = min(4, max(1, recall_limit // 3))
    value_snippets = graph.recall(question, limit=value_limit, emit_history=emit_history)
    merged: list[str] = []
    seen: set[str] = set()
    for snippet in [*value_snippets, *recalled]:
        if snippet in seen:
            continue
        seen.add(snippet)
        merged.append(snippet)
        if len(merged) >= recall_limit:
            break
    return "\n".join(merged)


def _snippet_has_stale(snippet: str, stale: set[str]) -> bool:
    """True if the untyped snippet contains a superseded value token (word-bounded)."""
    low = snippet.lower()
    for value in stale:
        if value and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", low):
            return True
    return False


def _value_augmented_recall_v3(
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
    *,
    suppress: bool,
) -> str:
    """v3 authoritative composition: coarse (near-oracle) grouping, current-only typed
    snippets, and (when ``suppress``) drop untyped Menhir snippets that reintroduce a
    superseded value. Same total top-k cap as v1/v2."""
    graph = SupersededValueGraph.from_item(
        f"value3-{task_id}", item, adapter.sessions(item), grouping="coarse"
    )
    value_limit = min(4, max(1, recall_limit // 3))
    value_snippets = graph.recall(question, limit=value_limit, emit_history=False)
    backfill = recalled
    if suppress:
        stale = graph.stale_value_strings(question)
        backfill = [s for s in recalled if not _snippet_has_stale(s, stale)]
    merged: list[str] = []
    seen: set[str] = set()
    for snippet in [*value_snippets, *backfill]:
        if snippet in seen:
            continue
        seen.add(snippet)
        merged.append(snippet)
        if len(merged) >= recall_limit:
            break
    return "\n".join(merged)


def _value_augmented_recall_v4(
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
) -> str:
    """v4 advisory composition: coarse grouping to find the candidate set, then additive
    (never drops a candidate) recall with per-cluster advisory role labels; only an
    author-rejected value (explicit correction marker) is suppressed. The untyped Menhir
    backfill is left intact - advisory never suppresses untyped recall. Same total top-k."""
    graph = SupersededValueGraph.from_item(
        f"value4-{task_id}", item, adapter.sessions(item), grouping="coarse"
    )
    value_limit = min(4, max(1, recall_limit // 3))
    value_snippets = graph.advisory_recall(question, limit=value_limit)
    merged: list[str] = []
    seen: set[str] = set()
    for snippet in [*value_snippets, *recalled]:
        if snippet in seen:
            continue
        seen.add(snippet)
        merged.append(snippet)
        if len(merged) >= recall_limit:
            break
    return "\n".join(merged)


def _value_augmented_recall_v5(
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
) -> str:
    """v5 derivation: v4 advisory typed context plus a deterministic delta-fold derived hint,
    then untyped backfill. Same merge/cap contract as v1-v4; untyped recall left intact."""
    graph = SupersededValueGraph.from_item(
        f"value5-{task_id}", item, adapter.sessions(item), grouping="coarse"
    )
    value_limit = min(4, max(1, recall_limit // 3))
    value_snippets = graph.derived_recall(question, limit=value_limit)
    merged: list[str] = []
    seen: set[str] = set()
    for snippet in [*value_snippets, *recalled]:
        if snippet in seen:
            continue
        seen.add(snippet)
        merged.append(snippet)
        if len(merged) >= recall_limit:
            break
    return "\n".join(merged)


def _value_context_for_arm(
    arm: str,
    adapter: "MemoryQAAdapter",
    item: dict,
    recalled: list[str],
    question: str,
    recall_limit: int,
    task_id: str,
) -> str:
    """Dispatch the memory context for value/plain arms (shared by both recall paths)."""
    if arm == VALUE_RECALL:
        return _value_augmented_recall(adapter, item, recalled, question, recall_limit, task_id)
    if arm == VALUE_RECALL_V2_CURRENT:
        return _value_augmented_recall_v2(
            adapter, item, recalled, question, recall_limit, task_id, emit_history=False
        )
    if arm == VALUE_RECALL_V2_HISTORY:
        return _value_augmented_recall_v2(
            adapter, item, recalled, question, recall_limit, task_id, emit_history=True
        )
    if arm == VALUE_RECALL_V3_COARSE:
        return _value_augmented_recall_v3(
            adapter, item, recalled, question, recall_limit, task_id, suppress=False
        )
    if arm == VALUE_RECALL_V3_AUTHORITATIVE:
        return _value_augmented_recall_v3(
            adapter, item, recalled, question, recall_limit, task_id, suppress=True
        )
    if arm == VALUE_RECALL_V4_ADVISORY:
        return _value_augmented_recall_v4(
            adapter, item, recalled, question, recall_limit, task_id
        )
    if arm == VALUE_RECALL_V5_DERIVED:
        return _value_augmented_recall_v5(
            adapter, item, recalled, question, recall_limit, task_id
        )
    return "\n".join(recalled)


_PROD_MARKERS = ("prod", "production", "menhir.", "staging.", "preprod", "preview", "release")


@runtime_checkable
class MenhirClient(Protocol):
    """Minimal memory backend the Mode-B driver needs (stub or HTTP)."""

    def new_group(self) -> str:
        """Return a fresh isolated namespace id for one benchmark item."""
        ...

    def ingest(self, group_id: str, role: str, content: str) -> None:
        """Ingest one conversation turn into the memory store under group_id."""
        ...

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        """Return recalled memory snippets for the query, scoped to group_id."""
        ...

    def reset(self, group_id: str) -> None:
        """Clear all memory for a group (best-effort cleanup)."""
        ...


@runtime_checkable
class MemoryQAAdapter(Protocol):
    """Adapter contract for an ingest-then-recall memory QA benchmark."""

    benchmark_id: str
    name: str

    def load_items(self, subset, limit, fixture_path) -> list[dict]: ...  # noqa: ANN001
    def sessions(self, item: dict) -> list[list[dict]]: ...
    def question(self, item: dict) -> str:  ...
    def build_messages(self, memory_context: str, question: str) -> list[dict]: ...
    def score(self, item: dict, response_text: str) -> bool: ...


def assert_not_production(target: str) -> None:
    """Refuse to run a write-heavy benchmark against anything that looks prod."""
    low = (target or "").lower()
    if any(m in low for m in _PROD_MARKERS):
        raise SystemExit(
            f"REFUSING: memory benchmark target {target!r} looks like production. "
            "Point --menhir-url at a throwaway instance."
        )


def _run_memory_arm(
    adapter: MemoryQAAdapter,
    arm: str,
    items: Sequence[dict],
    *,
    client: MenhirClient | None,
    chat_client: httpx.Client,
    send_fn,
    chat_base_url: str,
    api_key: str,
    model: str,
    recall_limit: int,
    pricing: PricingModel,
    reset_memory: bool,
    reset_confirmed: bool,
    dry_run_reset: bool,
    recall_only: bool = False,
    namespace_template: str = "lme-{question_id}",
    checkpoint: "MemoryCheckpoint | None" = None,
    score_fn=None,  # noqa: ANN001
) -> ArmResult:
    results: list[TaskResult] = []
    turn_dicts: list[dict] = []
    for i, item in enumerate(items):
        task_id = str(item.get("question_id") or f"item-{i}")
        # Resume: if this (arm, item) was already scored in a prior run, reuse it and
        # skip re-ingesting its haystack -- the expensive, abort-prone part.
        if checkpoint is not None:
            cached = checkpoint.get(arm, task_id)
            if cached is not None:
                results.append(cached)
                turn_dicts.append({"input_tokens": cached.input_tokens, "output_tokens": cached.output_tokens})
                continue
        question = adapter.question(item)
        if arm == NO_MEMORY or client is None:
            memory_context = ""
        elif recall_only:
            # Recall-only A/B: the graph is pre-built once in stable per-question
            # namespaces (e.g. lme-<question_id>), so skip ingest AND reset entirely and
            # recall in place. No new_group, no mutation -> needs no --confirm-menhir-reset.
            group_id = namespace_template.format(question_id=item.get("question_id") or task_id)
            recalled = client.recall(group_id, question, limit=recall_limit)
            memory_context = _value_context_for_arm(
                arm, adapter, item, recalled, question, recall_limit, task_id
            )
        else:
            group_id = client.new_group()
            try:
                for session in adapter.sessions(item):
                    for turn in session:
                        client.ingest(group_id, turn.get("role", "user"), turn.get("content", ""))
                if arm == AGENTIC_RECALL:
                    subqueries = _plan_recall_queries(
                        question, send_fn=send_fn, chat_client=chat_client,
                        chat_base_url=chat_base_url, api_key=api_key, model=model,
                    )
                    memory_context = _agentic_recall(client, group_id, subqueries, recall_limit)
                else:
                    recalled = client.recall(group_id, question, limit=recall_limit)
                    memory_context = _value_context_for_arm(
                        arm, adapter, item, recalled, question, recall_limit, task_id
                    )
            finally:
                if reset_memory:
                    _reset_memory_group(
                        client,
                        group_id,
                        reset_confirmed=reset_confirmed,
                        dry_run_reset=dry_run_reset,
                    )
        messages = adapter.build_messages(memory_context, question)
        text, latency_ms, usage = send_fn(chat_client, chat_base_url, api_key, messages, model)
        inp, out = _usage_tokens(usage)
        correct = score_fn(item, text) if score_fn is not None else adapter.score(item, text)
        scorer_usage = (
            dict(getattr(score_fn, "last_usage", {}) or {}) if score_fn is not None else {}
        )
        scorer_inp, scorer_out = _usage_tokens(scorer_usage)
        tr = TaskResult(
            task_id=task_id,
            response_text=text,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            correct=correct,
            raw_usage=usage,
            question=question,
            recalled=memory_context,
            gold=str(item.get("answer", "")),
            scorer_input_tokens=scorer_inp,
            scorer_output_tokens=scorer_out,
            scorer_raw_usage=scorer_usage,
        )
        results.append(tr)
        turn_dicts.append({"input_tokens": inp, "output_tokens": out})
        # Persist immediately so a later crash/rate-limit abort keeps this item.
        if checkpoint is not None:
            checkpoint.record(arm, task_id, tr)

    score = mean(1.0 if r.correct else 0.0 for r in results) if results else 0.0
    arm_cost = compute_arm_cost(turn_dicts, pricing)
    return ArmResult(
        arm=arm,
        n=len(results),
        score=round(score, 4),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cost_usd=round(arm_cost.total_effective_cost_usd, 6),
        results=results,
    )


def run_memory_ab(
    adapter: MemoryQAAdapter,
    *,
    arms: Sequence[str] = DEFAULT_MEMORY_ARMS,
    subset: str | None = None,
    limit: int | None = None,
    fixture_path=None,  # noqa: ANN001
    client: MenhirClient | None = None,
    send_fn=send_chat,  # noqa: ANN001
    chat_base_url: str = PROXY_URL,
    api_key: str = API_KEY,
    model: str = MODEL,
    # MSC sweep knob: override the recall top-k via env so accuracy-vs-context can be
    # measured without a CLI change. Default 10 preserves prior behavior when unset.
    recall_limit: int = int(os.getenv("LME_RECALL_LIMIT", "10")),
    pricing: PricingModel | None = None,
    reset_memory: bool = True,
    reset_confirmed: bool = False,
    dry_run_reset: bool = False,
    recall_only: bool = False,
    namespace_template: str = "lme-{question_id}",
    checkpoint: "MemoryCheckpoint | None" = None,
    score_fn=None,  # noqa: ANN001
) -> ABResult:
    """Run an ingest-then-recall memory benchmark across arms.

    `no_memory` answers with an empty memory context (the floor); memory arms
    ingest+recall via `client`. Offline: pass `fixture_path` and a stub `client`
    + deterministic `send_fn`. Real runs require a throwaway menhir client.

    `checkpoint`: optional MemoryCheckpoint. When given, each item's result is
    persisted as it completes and already-recorded items are skipped, so a long run
    survives crashes and rate-limit aborts (rerun the same command to resume).

    `score_fn`: optional `(item, response_text) -> bool` override for grading (e.g. an
    LLMJudgeScorer for LongMemEval-comparable accuracy). Defaults to `adapter.score`
    (the offline containment scorer). If it exposes `.close()`, it is closed on exit.
    """
    assert_known_arms(arms)
    items = adapter.load_items(subset, limit, fixture_path)
    pricing = _pick_pricing(model, pricing)
    needs_client = any(a != NO_MEMORY for a in arms)
    if needs_client and client is None:
        raise ValueError("memory arms require a MenhirClient (got None)")
    if recall_only:
        # Recall-only A/B reads a pre-built graph in place: never ingest, never reset.
        # This also makes --confirm-menhir-reset unnecessary (the guard below keys on
        # reset_memory, which we force off here).
        reset_memory = False
    if (
        needs_client
        and isinstance(client, HttpMenhirClient)
        and reset_memory
        and not reset_confirmed
        and not dry_run_reset
    ):
        raise ValueError(
            "memory benchmarks with a real Menhir client require --confirm-menhir-reset "
            "or --dry-run-menhir-reset before any ingest occurs"
        )

    arm_results: dict[str, ArmResult] = {}
    memory_client_cm = client if hasattr(client, "__enter__") and hasattr(client, "__exit__") else nullcontext(client)
    try:
        with httpx.Client(timeout=300) as chat_client:
            with memory_client_cm:
                for arm in arms:
                    arm_client = None if arm == NO_MEMORY else client
                    arm_results[arm] = _run_memory_arm(
                        adapter,
                        arm,
                        items,
                        client=arm_client,
                        chat_client=chat_client,
                        send_fn=send_fn,
                        chat_base_url=chat_base_url,
                        api_key=api_key,
                        model=model,
                        recall_limit=recall_limit,
                        pricing=pricing,
                        reset_memory=reset_memory,
                        reset_confirmed=reset_confirmed,
                        dry_run_reset=dry_run_reset,
                        recall_only=recall_only,
                        namespace_template=namespace_template,
                        checkpoint=checkpoint,
                        score_fn=score_fn,
                    )
    finally:
        if score_fn is not None and hasattr(score_fn, "close"):
            score_fn.close()

    # Deltas vs the no_memory floor (memory lift), reusing the shared helper by
    # aliasing no_memory as the baseline.
    baseline = arm_results.get(NO_MEMORY)
    deltas: dict[str, dict] = {}
    if baseline is not None:
        aliased = {"direct" if k == NO_MEMORY else k: v for k, v in arm_results.items()}
        deltas = _compute_deltas(aliased)

    return ABResult(
        benchmark_id=adapter.benchmark_id,
        name=adapter.name,
        model=model,
        subset=subset,
        arms=arm_results,
        deltas=deltas,
    )


def _reset_memory_group(
    client: MenhirClient,
    group_id: str,
    *,
    reset_confirmed: bool,
    dry_run_reset: bool,
) -> None:
    """Reset a memory group with explicit safety for real HTTP Menhir clients."""
    if dry_run_reset:
        print(f"  [memory] dry-run: would reset group {group_id}")
        return
    if isinstance(client, HttpMenhirClient) and not reset_confirmed:
        raise RuntimeError(
            "Refusing to reset a real Menhir group without confirmation. "
            "Pass --confirm-menhir-reset for cleanup or --dry-run-menhir-reset to inspect."
        )
    client.reset(group_id)
