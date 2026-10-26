"""LongMemEval adapter — official long-term memory QA benchmark.

Dataset: LongMemEval (Wu et al., ICLR 2025) — chat assistants answering questions
that require recalling facts across long, multi-session histories. Measures the
five memory abilities: information extraction, multi-session reasoning, temporal
reasoning, knowledge updates, and abstention.

This is the capability benchmark for a memory system (menhir is built on Graphiti,
the same temporal-KG engine Zep reports on LongMemEval/DMR). Run as a direct-vs-
proxy A/B: the proxy assembles relevant memory from the history, so the claim is
memory-QA accuracy preserved/improved while tokens drop.

Online use needs the `longmemeval` extra (`datasets`); offline reads a fixture.

Scoring note: the official benchmark uses a GPT-4 judge. This adapter ships a
deterministic normalized scorer (good for factual answers, offline, no judge
spend). It rejects obvious negated-answer false positives; an LLM-judge scorer
can still be added behind a flag later for official evidence runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
from pathlib import Path

from .base import Task
from .scoring import is_lme_abstention

_SYSTEM_PROMPT = (
    "You are a helpful assistant with long-term memory of the user's prior conversations. "
    "Use the conversation history to answer the final question accurately and concisely. "
    "If the history does not contain the answer, say you don't know."
)

_PUNCT = str.maketrans("", "", string.punctuation)


def _flatten_history(item: dict) -> list[dict]:
    """Flatten LongMemEval haystack sessions into chat turns."""
    msgs: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    sessions = item.get("haystack_sessions") or item.get("sessions") or []
    for session in sessions:
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                msgs.append({"role": "user" if role == "user" else "assistant", "content": content})
    msgs.append({"role": "user", "content": item.get("question", "")})
    return msgs


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def _has_negated_gold(gold: str, resp: str) -> bool:
    patterns = (
        rf"\bnot\s+{re.escape(gold)}\b",
        rf"\bnot\s+(?:the\s+)?(?:answer|name|place|person|city)\s+{re.escape(gold)}\b",
        rf"\b{re.escape(gold)}\s+(?:is|was)\s+(?:not|wrong|incorrect)\b",
    )
    return any(re.search(pattern, resp) for pattern in patterns)


def _load_lme_items(subset: str | None, fixture_path: str | Path | None) -> list[dict]:
    """Load raw LongMemEval items (shared by the in-context and memory adapters)."""
    if fixture_path is not None:
        with open(fixture_path, encoding="utf-8") as f:
            items = json.load(f)
    else:
        # The xiaowu0162/longmemeval HF repo stores raw JSON arrays as extension-less
        # files (longmemeval_s / longmemeval_m / longmemeval_oracle), so the dataset
        # auto-loader cannot map them to a split. Download the chosen variant directly
        # and parse it. Variant selectable via LONGMEMEVAL_VARIANT (s|m|oracle);
        # default "s" is the canonical reported benchmark (~115k-token haystacks).
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:  # pragma: no cover - online only
            raise RuntimeError(
                "LongMemEval requires the 'datasets' extra (huggingface_hub). "
                "Install with: pip install '.[longmemeval]', or pass fixture_path for offline use."
            ) from e
        variant = (os.getenv("LONGMEMEVAL_VARIANT", "s") or "s").strip().lower()
        filename = {"s": "longmemeval_s", "m": "longmemeval_m", "oracle": "longmemeval_oracle"}.get(
            variant, "longmemeval_s"
        )
        path = hf_hub_download(  # pragma: no cover - network
            repo_id="xiaowu0162/longmemeval", filename=filename, repo_type="dataset"
        )
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
    if subset:
        items = [it for it in items if it.get("question_type") == subset]
    return items


def _score_answer(answer: str, question_type: str | None, response_text: str,
                  *, abstention: bool = False) -> bool:
    """Normalized-containment scorer with abstention handling (shared by both modes).

    ``abstention``: the item is unanswerable (LongMemEval ``_abs`` question id); its gold is an
    explanation, not an answer to contain."""
    gold = _normalize(answer)
    resp = _normalize(response_text)
    if not resp:
        return False
    if abstention or question_type == "abstention" or gold in {"", "no answer", "unknown"}:
        return any(p in resp for p in ("dont know", "not sure", "no information", "cannot", "unable"))
    if not gold or _has_negated_gold(gold, resp):
        return False

    gold_tokens = _tokens(answer)
    resp_tokens = _tokens(response_text)
    if len(gold_tokens) == 1:
        return gold_tokens[0] in set(resp_tokens)

    return gold in resp


class LongMemEvalAdapter:
    """Official LongMemEval memory-QA accuracy, run as a direct-vs-proxy A/B."""

    benchmark_id = "longmemeval"
    name = "LongMemEval"

    def load_tasks(
        self,
        subset: str | None = None,
        limit: int | None = None,
        fixture_path: str | Path | None = None,
    ) -> list[Task]:
        items = _load_lme_items(subset, fixture_path)
        if limit is not None:
            items = items[:limit]
        tasks: list[Task] = []
        for i, item in enumerate(items):
            task_id = str(item.get("question_id") or item.get("id") or f"lme-{i}")
            tasks.append(
                Task(
                    task_id=task_id,
                    prompt_messages=_flatten_history(item),
                    answer=str(item.get("answer", "")),
                    meta={"question_type": item.get("question_type")},
                )
            )
        return tasks

    def score(self, task: Task, response_text: str) -> bool:
        return _score_answer(task.answer, task.meta.get("question_type"), response_text,
                             abstention="_abs" in task.task_id)


_MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use ONLY the retrieved memory below to answer the question "
    "accurately and concisely.\n\n"
    "How to use this memory system:\n"
    "- [AUTHORITATIVE CURRENT MEMORY] is the canonical current value for its subject, attribute, and scope. "
    "A valid record with status 'leads' outranks conflicting related memories.\n"
    "- [SUPERSEDED ... MEMORY] is historical context. Use it only when the question asks about the past.\n"
    "- [RELATED ... MEMORY | non-authoritative] may add context, but it cannot override current authority.\n"
    "- Memories marked \"(agent-recorded)\" record what the assistant told the user earlier, not something the "
    "user said; \"(user + agent)\" mixes both. Use them for questions about what the assistant said or "
    "recommended, and never let them override the user's own statements.\n"
    "- 'valid at' is the source/world time. A provenance quote shows the statement supporting the value. "
    "'absolute' means directly stated; 'delta' means a change applied to an earlier supported value.\n"
    "- Each ordinary memory's source-time evidence says when its fact was true. For questions about "
    "previous, later, changed, newest, or oldest facts, compare those source times; never infer chronology "
    "from retrieval/list order or from when Menhir learned the memory. A 'superseded belief' label means "
    "the fact is historical; a 'current belief' label means it remains current. If source time is unknown, "
    "do not invent an ordering; instead prefer a value the user states as a change or update (\"now\", "
    "\"switched to\", \"just\", \"recently\") over one stated without such wording.\n\n"
    "Answer whenever the retrieved memory addresses the question. When it holds two or more candidate "
    "answers, do not stop at \"I don't know\": give the best-supported candidate as your answer, then briefly "
    "name the other candidate and why it ranks lower. Say you don't know only when no retrieved memory "
    "addresses what the question asks about.\n\n"
    "For questions that count distinct things (\"how many\" items, events, times or places) or ask for a "
    "list: list every candidate item from the retrieved memory and rate how well it fits the category the "
    "question names: high (clearly fits), medium (fits under a reasonable everyday reading of the "
    "question's wording), low (only a stretch). Count high and medium, each distinct item once. Give one "
    "final number; you may then briefly mention low items you left out. When \"how many\" instead asks "
    "for a single current quantity (an amount the user has or has reached now), do not count mentions: "
    "answer with the latest supported value, following the rules above.\n\n"
    "When the question asks for suggestions, tips, advice or recommendations, answer with helpful, "
    "concrete suggestions tailored to the preferences, plans, possessions and situation the retrieved "
    "memory shows; general knowledge is allowed for this. Never answer such a question by saying the "
    "memory holds no suggestions.\n\n"
    "Safe veto policy (structured verdict and authoritative records only; never use it to drop an ordinary "
    "memory that answers the question): ignore such a record only when it does not match the question's "
    "subject, attribute, "
    "scope, unit, or time; when it is superseded for a current-value question; when its provenance clearly "
    "does not support its claimed value. If multiple authoritative current records conflict, veto the "
    "conflicting set's authority and pick the best-supported candidate as above, naming the conflict. "
    "Never veto a matching, supported "
    "authoritative record merely because related memories disagree, because a value appears more often, or "
    "because outside knowledge suggests something else. If no consistent, directly relevant supported memory "
    "remains after those checks, say you don't know rather than inventing an answer."
)


class LongMemEvalMemoryAdapter:
    """Mode B: LongMemEval via persistent ingest-then-recall (menhir memory).

    Per item: ingest the haystack sessions into the memory store, recall against
    the question, and answer from the recalled memory (not the raw history).
    Implements the `memory_ab.MemoryQAAdapter` shape; driven by `run_memory_ab`.
    """

    benchmark_id = "longmemeval-menhir"
    name = "LongMemEval (menhir memory)"

    def load_items(
        self,
        subset: str | None = None,
        limit: int | None = None,
        fixture_path: str | Path | None = None,
    ) -> list[dict]:
        items = _load_lme_items(subset, fixture_path)
        if limit is not None:
            items = items[:limit]
        return items

    def sessions(self, item: dict) -> list[list[dict]]:
        return item.get("haystack_sessions") or item.get("sessions") or []

    def question(self, item: dict) -> str:
        return item.get("question", "")

    # Checkpoint key: a resume never mixes answers from different answer prompts.
    prompt_fingerprint = hashlib.sha256(
        (_MEMORY_SYSTEM_PROMPT + "\x00dated-user-message-v1").encode("utf-8")).hexdigest()[:10]

    def question_date(self, item: dict) -> str | None:
        return str(item.get("question_date") or "").strip() or None

    def build_messages(
        self, memory_context: str, question: str, question_date: str | None = None
    ) -> list[dict]:
        mem = memory_context.strip() or "(no relevant memory found)"
        dated = f"Current date: {question_date}\n\n" if question_date else ""
        return [
            {"role": "system", "content": _MEMORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"{dated}Retrieved memory:\n{mem}\n\nQuestion: {question}"},
        ]

    def score(self, item: dict, response_text: str) -> bool:
        return _score_answer(str(item.get("answer", "")), item.get("question_type"), response_text,
                             abstention=is_lme_abstention(item))
