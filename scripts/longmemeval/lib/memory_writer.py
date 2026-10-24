"""Agent memory writer for LongMemEval ingest (``--segmentation adaptive-writer``, #168 P3).

An LLM reads assistant turns (with the user turn that prompted each) and writes 0..N standalone,
third-person, dated memory items about what the assistant said. Items are ingested as agent-tier
memories (``role="agent"``, ``source="agent-writer"``) grounded on the assistant turn's
TurnEvidence, so they can never found a View or supersede a user statement in Menhir.

Granularities:
* ``turn``: one request per assistant turn, with the preceding user turn.
* ``session``: one request per session with all its turns; every item cites the assistant turn
  index it came from, so it links to that turn's TurnEvidence.

Writer output is produced in a prepass and persisted in a JSONL checkpoint before the ingest
plan is built, so a resumed build replays the same items (the plan digest hashes them). The
``custom_id`` carries granularity, model, prompt sha and a digest of the session text, so a changed
prompt, model or input never reuses stale rows. A request that failed is dropped from the checkpoint and the prepass stops; rerunning
retries only those requests. A failure is never treated as "no items".
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRANULARITIES = ("turn", "session")
WRITER_SOURCE = "agent-writer"
WRITER_ROLE = "agent"
MAX_COMPLETION_TOKENS = 16384
MAX_ITEMS_PER_TURN = 20

WRITER_PROMPT = """You record what an AI assistant told a user, as memory items for later recall.

You get a conversation excerpt with its date. Write standalone memory items about what the
ASSISTANT said: answers, recommendations, lists, explanations, plans, numbers, names, choices.

Rules for every item:
- Third person and attributed: "On <date>, the assistant told/recommended/gave the user ...".
- Standalone: readable without the conversation. Name the topic the user asked about.
- Keep concrete specifics exactly: names, titles, numbers, quantities, list entries (all of them,
  in order), colours, places, times.
- Only what the assistant said. Never present the assistant's words as the user's, and do not
  record what the user said except as brief context ("when the user asked about X").
- No item for greetings, filler, refusals or questions back to the user.
- Zero items is a valid answer.

Reply with JSON only: {"items": [ITEM, ...]}
{item_shape}"""

_TURN_ITEM_SHAPE = 'ITEM is {"text": "<memory item>"}.'
_SESSION_ITEM_SHAPE = (
    'ITEM is {"turn": <the [turn N] number of the assistant turn it came from>, '
    '"text": "<memory item>"}. Every item cites exactly one assistant turn.'
)


def prompt_for(granularity: str) -> str:
    _check_granularity(granularity)
    shape = _TURN_ITEM_SHAPE if granularity == "turn" else _SESSION_ITEM_SHAPE
    return WRITER_PROMPT.replace("{item_shape}", shape)


def prompt_sha(granularity: str) -> str:
    """sha256 of the exact system prompt for ``granularity`` (lineage and cache key)."""
    return hashlib.sha256(prompt_for(granularity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SessionTurn:
    """One non-empty raw turn of a session; ``index`` is its position in the raw session."""

    index: int
    role: str
    content: str


@dataclass(frozen=True)
class WriterSession:
    namespace: str
    session_id: str
    date: str
    turns: tuple[SessionTurn, ...]


@dataclass(frozen=True)
class WriterItem:
    turn_index: int
    text: str


class WriterError(RuntimeError):
    """The writer prepass could not produce trustworthy output; the build must stop."""


def _check_granularity(granularity: str) -> None:
    if granularity not in GRANULARITIES:
        raise ValueError(f"writer granularity must be one of {GRANULARITIES}, got {granularity!r}")


def _is_assistant(turn: SessionTurn) -> bool:
    return turn.role.strip().lower() == "assistant"


def _preceding_user(session: WriterSession, position: int) -> SessionTurn | None:
    for turn in reversed(session.turns[:position]):
        if turn.role.strip().lower() == "user":
            return turn
        if _is_assistant(turn):
            return None
    return None


def _session_digest(session: WriterSession) -> str:
    raw = json.dumps([session.date, [[t.index, t.role, t.content] for t in session.turns]],
                     ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def request_key(session: WriterSession, granularity: str, turn_index: int | None = None) -> str:
    """Per-request key, independent of model and prompt.

    Includes a digest of the session text: a namespace/session id reused with different content
    (fixture vs dataset, a shared results dir) must never replay another text's checkpoint row.
    """
    base = f"{session.namespace}|{session.session_id}|{_session_digest(session)}"
    return base if granularity == "session" else f"{base}|t{turn_index}"


def custom_id(key: str, *, granularity: str, model: str) -> str:
    return f"{granularity}|{model}|{prompt_sha(granularity)[:16]}|{key}"


def _planned(
    sessions: Iterable[WriterSession], granularity: str, model: str
) -> list[tuple[str, WriterSession, int | None, str]]:
    """``(custom_id, session, turn index or None, user message)`` per writer request."""
    planned: list[tuple[str, WriterSession, int | None, str]] = []
    for session in sessions:
        if granularity == "turn":
            for position, turn in enumerate(session.turns):
                if not _is_assistant(turn):
                    continue
                user = _preceding_user(session, position)
                lines = [f"Date: {session.date}"]
                if user is not None:
                    lines.append(f"[user]\n{user.content}")
                lines.append(f"[assistant]\n{turn.content}")
                cid = custom_id(request_key(session, granularity, turn.index), granularity=granularity, model=model)
                planned.append((cid, session, turn.index, "\n\n".join(lines)))
        else:
            if not any(_is_assistant(t) for t in session.turns):
                continue
            lines = [f"Date: {session.date}"]
            lines.extend(f"[turn {t.index}] [{t.role.strip().lower()}]\n{t.content}" for t in session.turns)
            cid = custom_id(request_key(session, granularity), granularity=granularity, model=model)
            planned.append((cid, session, None, "\n\n".join(lines)))
    return planned


def build_requests(
    sessions: Iterable[WriterSession], *, granularity: str, model: str
) -> list[dict[str, Any]]:
    """Flex requests for every assistant turn (turn mode) or session with one (session mode)."""
    _check_granularity(granularity)
    system = prompt_for(granularity)
    return [_request(cid, system, user) for cid, _s, _t, user in _planned(sessions, granularity, model)]


def _request(cid: str, system: str, user: str) -> dict[str, Any]:
    return {
        "custom_id": cid,
        "body": {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
    }


def parse_items(
    content: str | None, *, granularity: str, assistant_indices: set[int], default_turn: int | None = None
) -> tuple[list[WriterItem], int]:
    """Items from one writer reply, and the count of items dropped as malformed or mis-cited.

    Raises ``WriterError`` when the reply is not the JSON object the prompt asks for: a broken
    reply must stop the build, never pass as "no items".
    """
    _check_granularity(granularity)
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as exc:
        raise WriterError(f"writer reply is not JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise WriterError("writer reply has no 'items' list")
    items: list[WriterItem] = []
    dropped = 0
    for raw in data["items"]:
        text = raw.get("text") if isinstance(raw, dict) else None
        if not isinstance(text, str) or not text.strip():
            dropped += 1
            continue
        if granularity == "turn":
            turn_index = default_turn
        else:
            turn_index = raw.get("turn")
            if isinstance(turn_index, str) and turn_index.strip().isdigit():
                turn_index = int(turn_index)
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index not in assistant_indices:
            dropped += 1
            continue
        items.append(WriterItem(turn_index=turn_index, text=" ".join(text.split())))
    return items, dropped


@dataclass
class WriterReport:
    requests: int = 0
    items: int = 0
    dropped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def _drop_failed_rows(checkpoint: Path, also_ids: frozenset[str] | set[str] = frozenset()) -> int:
    """Rewrite the checkpoint without error rows (and ``also_ids``) so a rerun retries them."""
    if not checkpoint.exists():
        return 0
    kept: list[str] = []
    removed = 0
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue
        if (isinstance(row, dict) and row.get("error") is None and row.get("content") is not None
                and row.get("custom_id") not in also_ids):
            kept.append(line)
        else:
            removed += 1
    if removed:
        # Atomic: a crash mid-rewrite must not truncate the paid rows.
        tmp = checkpoint.with_name(checkpoint.name + ".tmp")
        tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
        os.replace(tmp, checkpoint)
    return removed


def write_memories(
    sessions: list[WriterSession],
    *,
    granularity: str,
    model: str,
    checkpoint: Path,
    run_flex: Callable[..., tuple[dict[str, Any], dict]],
    flex_kwargs: dict[str, Any],
) -> tuple[dict[tuple[str, str, int], list[WriterItem]], WriterReport]:
    """Run (or replay) the writer for ``sessions``.

    Returns items keyed by ``(namespace, session_id, assistant turn index)`` in reply order.
    ``run_flex`` is ``archolith_bench.core.openai_flex.run_flex`` (injected for tests);
    ``flex_kwargs`` carries api_key, max_usd, rates and workers.
    """
    _check_granularity(granularity)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _drop_failed_rows(checkpoint)
    requests = build_requests(sessions, granularity=granularity, model=model)
    report = WriterReport(requests=len(requests))
    if not requests:
        return {}, report
    results, _summary = run_flex(requests, model=model, checkpoint=checkpoint, **flex_kwargs)

    failed = [cid for cid in (r["custom_id"] for r in requests)
              if cid not in results or results[cid].error is not None or results[cid].content is None]
    if failed:
        _drop_failed_rows(checkpoint)
        raise WriterError(
            f"{len(failed)} of {len(requests)} writer requests failed (first: {failed[0]}); "
            "rerun to retry only those"
        )

    items: dict[tuple[str, str, int], list[WriterItem]] = {}
    for cid, session, default_turn, _user in _planned(sessions, granularity, model):
        result = results[cid]
        report.prompt_tokens += result.prompt_tokens
        report.completion_tokens += result.completion_tokens
        report.cost_usd += result.cost_usd
        assistant_indices = {t.index for t in session.turns if _is_assistant(t)}
        try:
            parsed, dropped = parse_items(result.content, granularity=granularity,
                                          assistant_indices=assistant_indices, default_turn=default_turn)
            if not parsed and dropped:
                # Items were written but none is usable: a broken reply, not "nothing to record".
                raise WriterError(f"all {dropped} items malformed or mis-cited")
        except WriterError as exc:
            _drop_failed_rows(checkpoint, also_ids={cid})  # a rerun asks again, never replays it
            raise WriterError(f"{cid}: {exc}; rerun to retry it") from exc
        report.dropped += dropped
        for item in parsed:
            bucket = items.setdefault((session.namespace, session.session_id, item.turn_index), [])
            if len(bucket) < MAX_ITEMS_PER_TURN:
                bucket.append(item)
                report.items += 1
            else:
                report.dropped += 1
    return items, report


if __name__ == "__main__":  # pragma: no cover - lineage helper for build_graph.sh
    import sys

    if len(sys.argv) != 3 or sys.argv[1] != "prompt-sha":
        print("usage: memory_writer.py prompt-sha turn|session", file=sys.stderr)
        raise SystemExit(2)
    print(prompt_sha(sys.argv[2]))
