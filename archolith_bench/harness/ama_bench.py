"""AMA-Bench State Updating as an ingest-then-recall memory benchmark.

AMA-Bench (MIT, huggingface.co/datasets/AMA-bench/AMA-bench) pairs long agent trajectories with
expert-written questions. Type ``C`` questions test *state updating*: whether memory returns the
latest state after it changed. Each episode is ingested ONCE (``scripts/ama/ingest_ama.py``);
questions then run recall-only against that episode's namespace. Two ingest modes:

* raw: one memory per trajectory step (a log, not how Menhir is used);
* crafted: a memory agent reads each step and writes add_memory-style memories only where
  appropriate (``scripts/ama/craft_memories.py``), and only those reach Menhir.

Trajectories are agent actions and environment observations; they contain no user turns, so they
never reach Menhir's scalar lane (user TurnEvidence only). This is a recall/supersession check.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATASET_ENV = "AMA_BENCH_DATASET"
DEFAULT_QA_TYPES = ("C",)
DEFAULT_PER_DOMAIN = 5
DEFAULT_MAX_TOKENS = 60_000
DEFAULT_EXCLUDE_DOMAINS = ("OPENWORLD_QA",)
# Synthetic, strictly increasing source times so step order is also time order in the graph.
STEP_TIME_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


RAW_PREFIX = "ama-ep"
CRAFTED_PREFIX = "ama-crafted-ep"


def namespace_for(episode_id: int, prefix: str | None = None) -> str:
    return f"{prefix or os.getenv('AMA_NAMESPACE_PREFIX') or RAW_PREFIX}{episode_id}"


# ---- Memory agent: an agent reading its own trajectory and choosing what to remember ----

MEMORY_AGENT_SYSTEM = (
    "You are an AI agent working on the task below, and you have a long-term memory tool "
    "(add_memory). After each step you decide what, if anything, is worth saving so that you or "
    "another agent could pick this work up later without the transcript. Save what matters: "
    "decisions, findings, errors and their causes, what you changed, test or tool results, and the "
    "current state of things. Skip routine output with nothing new. When something you saved "
    "earlier is no longer true, save the new state and say what it replaces. Each memory must be "
    "self-contained and start with the step number, like 'Step 12: ...'. Reply with JSON only: "
    '{"memories": ["...", "..."]}, or {"memories": []} when nothing is worth saving.'
)
MAX_TASK_CHARS = 4_000
MAX_STEP_CHARS = 12_000
RECENT_MEMORIES = 20
MAX_MEMORIES_PER_STEP = 5


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} more characters]"


def memory_agent_messages(episode: dict, step: dict, recent: list[str]) -> list[dict]:
    idx = int(step.get("turn_idx", 0))
    saved = "\n".join(f"- {m}" for m in recent[-RECENT_MEMORIES:]) or "(none yet)"
    user = (
        f"Task:\n{_clip(episode.get('task', ''), MAX_TASK_CHARS)}\n\n"
        f"Memories you have saved so far (most recent last):\n{saved}\n\n"
        f"Step {idx}\nAction: {_clip(step.get('action', ''), MAX_STEP_CHARS // 4)}\n"
        f"Observation: {_clip(step.get('observation', ''), MAX_STEP_CHARS)}\n\n"
        f"What, if anything, do you save to memory after step {idx}?"
    )
    return [{"role": "system", "content": MEMORY_AGENT_SYSTEM}, {"role": "user", "content": user}]


def parse_memories(text: str) -> list[str]:
    """The agent's memories for one step; malformed output saves nothing (never guessed)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    return [m.strip() for m in items if isinstance(m, str) and m.strip()][:MAX_MEMORIES_PER_STEP]


def crafted_turns(crafted: list[dict]) -> list[dict]:
    """Memories written by the agent, in step order, each at its step's source time."""
    turns = []
    for entry in sorted(crafted, key=lambda e: int(e["step"])):
        for memory in entry.get("memories", []):
            turns.append({"role": "assistant", "content": memory, "occurred_at": step_time(int(entry["step"]) + 1)})
    return turns


def step_time(step_idx: int) -> str:
    return (STEP_TIME_BASE + timedelta(minutes=step_idx)).isoformat()


# ---- Supersession gold: verified state timelines -> templated questions ----

TIMELINE_SYSTEM = (
    "You read an AI agent's complete task trajectory and extract STATE TIMELINES: things whose "
    "state changed during the task (for example a test result, a file's contents, a form field, "
    "the agent's location, an error, a count, which approach the agent is using). For each, list "
    "every distinct state it took, in step order. Every state needs the step it was observed in "
    "and an evidence quote copied EXACTLY, character for character, from that step's action or "
    "observation (a short span, 3 to 20 words). Only include timelines with at least two different "
    "states. Values must be short and concrete. Reply with JSON only: "
    '{"timelines": [{"subject": "...", "attribute": "...", "states": '
    '[{"step": 3, "value": "...", "evidence": "..."}]}]}'
)
TIMELINE_STEP_CHARS = 4_000
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", str(text or "")).strip().lower()


def timeline_messages(episode: dict) -> list[dict]:
    steps = "\n\n".join(
        f"Step {s.get('turn_idx')}\nAction: {_clip(s.get('action', ''), TIMELINE_STEP_CHARS // 4)}\n"
        f"Observation: {_clip(s.get('observation', ''), TIMELINE_STEP_CHARS)}"
        for s in episode.get("trajectory", [])
    )
    user = f"Task:\n{_clip(episode.get('task', ''), MAX_TASK_CHARS)}\n\nTrajectory:\n{steps}"
    return [{"role": "system", "content": TIMELINE_SYSTEM}, {"role": "user", "content": user}]


def verify_timelines(episode: dict, raw: str) -> tuple[list[dict], dict]:
    """Keep only states whose quote appears verbatim (whitespace/case-folded) in their own step,
    in strictly increasing step order; keep timelines with >= 2 distinct verified values."""
    stats = {"timelines_proposed": 0, "states_proposed": 0, "states_verified": 0, "timelines_kept": 0}
    try:
        data = json.loads(raw.strip().strip("`").removeprefix("json").strip())
    except (json.JSONDecodeError, AttributeError):
        return [], stats
    proposed = data.get("timelines") if isinstance(data, dict) else None
    if not isinstance(proposed, list):
        return [], stats
    text_by_step = {
        int(s.get("turn_idx")): _norm(f"{s.get('action', '')} {s.get('observation', '')}")
        for s in episode.get("trajectory", [])
    }
    kept: list[dict] = []
    for tl in proposed:
        if not isinstance(tl, dict) or not isinstance(tl.get("states"), list):
            continue
        stats["timelines_proposed"] += 1
        states: list[dict] = []
        for st in tl["states"]:
            stats["states_proposed"] += 1
            try:
                step = int(st["step"])
                value, evidence = str(st["value"]).strip(), str(st["evidence"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not value or len(_norm(evidence)) < 8 or _norm(evidence) not in text_by_step.get(step, ""):
                continue
            if states and step <= states[-1]["step"]:
                continue
            if states and _norm(value) == _norm(states[-1]["value"]):
                continue  # same state observed again, not a change
            states.append({"step": step, "value": value, "evidence": evidence})
        stats["states_verified"] += len(states)
        if len({_norm(s["value"]) for s in states}) >= 2:
            kept.append({"subject": str(tl.get("subject", "")).strip(),
                         "attribute": str(tl.get("attribute", "")).strip(), "states": states})
    stats["timelines_kept"] = len(kept)
    return kept, stats


CHECK_SYSTEM = (
    "You check a proposed state timeline extracted from an AI agent's trajectory. For each state "
    "you get the step, the proposed value, the quoted evidence and the text around it. Decide two "
    "things per state: about_subject = the evidence is really about the given subject and "
    "attribute (not a different item, size, row or case); real_change = the value is a genuinely "
    "different state from the previous kept state, not a rewording, a refinement that adds detail "
    "to the same state, or a partial view of it (always true for the first state). Reply with JSON "
    'only: {"states": [{"step": 3, "about_subject": true, "real_change": true}]}'
)
CHECK_CONTEXT_CHARS = 400
MAX_TIMELINE_STATES = 8


def _context(episode: dict, step: int, evidence: str) -> str:
    raw = next((f"{s.get('action', '')} {s.get('observation', '')}" for s in episode.get("trajectory", [])
                if int(s.get("turn_idx", -1)) == step), "")
    flat = _WS.sub(" ", str(raw))
    at = flat.lower().find(_norm(evidence)[:40])
    start = max(0, at - CHECK_CONTEXT_CHARS) if at >= 0 else 0
    return flat[start:start + 2 * CHECK_CONTEXT_CHARS + len(evidence)]


def check_messages(episode: dict, timeline: dict) -> list[dict]:
    states = "\n\n".join(
        f"State at step {s['step']}: value = {s['value']!r}\nEvidence: {s['evidence']!r}\n"
        f"Text around it: {_context(episode, s['step'], s['evidence'])}"
        for s in timeline["states"]
    )
    user = f"Subject: {timeline['subject']}\nAttribute: {timeline['attribute']}\n\n{states}"
    return [{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}]


def apply_check(timeline: dict, raw: str) -> dict | None:
    """Keep states judged about the subject; drop non-changes; None unless >= 2 distinct values
    remain. Fail closed: a state with no verdict, or unparseable output, is dropped."""
    try:
        data = json.loads(raw.strip().strip("`").removeprefix("json").strip())
        verdicts = {int(v["step"]): v for v in data["states"] if isinstance(v, dict)}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return None
    kept: list[dict] = []
    for st in timeline["states"]:
        v = verdicts.get(st["step"])
        if not v or v.get("about_subject") is not True:
            continue
        if kept and (v.get("real_change") is not True or _norm(st["value"]) == _norm(kept[-1]["value"])):
            continue
        kept.append(st)
    if len({_norm(s["value"]) for s in kept}) < 2:
        return None
    return {**timeline, "states": kept}


def _distinct_except(values: list[str], *exclude: str) -> list[str]:
    skip = {_norm(v) for v in exclude}
    out: list[str] = []
    for v in values:
        if _norm(v) not in skip and _norm(v) not in {_norm(o) for o in out}:
            out.append(v)
    return out


def gold_questions(episode: dict, timelines: list[dict]) -> list[dict]:
    """Templated current / previous / timeline questions from verified timelines.

    Stale answers are earlier values that DIFFER from the gold answer (a value that returns later,
    e.g. True -> False -> True, is not stale for "current"). Timelines longer than
    MAX_TIMELINE_STATES get no timeline question (flip-flop lists are noise, not memory).
    """
    items: list[dict] = []
    for n, tl in enumerate(timelines):
        subject, attribute, states = tl["subject"], tl["attribute"], tl["states"]
        if not subject or not attribute:
            continue
        last, prev = states[-1], states[-2]
        base = {"episode_id": episode["episode_id"], "domain": episode["domain"],
                "subject": subject, "attribute": attribute}
        values = [s["value"] for s in states]
        items += [
            {**base, "question_id": f"ep{episode['episode_id']}-t{n}-current", "question_type": "current",
             "question": f"By the end of the task, what is the latest {attribute} of {subject}?",
             "answer": last["value"], "stale_answers": _distinct_except(values[:-1], last["value"]),
             "states": states},
            {**base, "question_id": f"ep{episode['episode_id']}-t{n}-previous", "question_type": "previous",
             "question": f"Before the {attribute} of {subject} became \"{last['value']}\", what was it?",
             "answer": prev["value"], "stale_answers": _distinct_except(values[:-2], prev["value"], last["value"]),
             "states": states},
        ]
        if len(states) <= MAX_TIMELINE_STATES:
            items.append(
                {**base, "question_id": f"ep{episode['episode_id']}-t{n}-timeline", "question_type": "timeline",
                 "question": f"List, in order, each {attribute} that {subject} had during the task.",
                 "answer": " -> ".join(values), "stale_answers": [], "states": states})
    return items


def load_episodes(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_episodes(
    episodes: list[dict],
    *,
    qa_types: tuple[str, ...] = DEFAULT_QA_TYPES,
    per_domain: int = DEFAULT_PER_DOMAIN,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    exclude_domains: tuple[str, ...] = DEFAULT_EXCLUDE_DOMAINS,
    episode_ids: tuple[int, ...] | None = None,
) -> list[dict]:
    """Deterministic subset: explicit ids, else the lowest episode ids per domain that have a
    question of a wanted type and fit under ``max_tokens``."""
    if episode_ids:
        wanted = set(episode_ids)
        return sorted((e for e in episodes if e["episode_id"] in wanted), key=lambda e: e["episode_id"])
    selected: list[dict] = []
    for domain in sorted({e["domain"] for e in episodes} - set(exclude_domains)):
        pool = sorted(
            (
                e for e in episodes
                if e["domain"] == domain
                and e.get("total_tokens", 0) <= max_tokens
                and any(q.get("type") in qa_types for q in e.get("qa_pairs", []))
            ),
            key=lambda e: e["episode_id"],
        )
        selected.extend(pool[:per_domain])
    return selected


def render_steps(episode: dict) -> list[dict]:
    """One memory per step, in order: the task first, then each action and its observation."""
    turns = [{
        "role": "assistant",
        "content": f"Task: {episode.get('task', '').strip()}",
        "occurred_at": step_time(0),
    }]
    for step in episode.get("trajectory", []):
        idx = int(step.get("turn_idx", len(turns)))
        content = (
            f"Step {idx}\nAction: {str(step.get('action', '')).strip()}\n"
            f"Observation: {str(step.get('observation', '')).strip()}"
        )
        turns.append({"role": "assistant", "content": content, "occurred_at": step_time(idx + 1)})
    return turns


def question_items(episodes: list[dict], qa_types: tuple[str, ...] = DEFAULT_QA_TYPES) -> list[dict]:
    items: list[dict] = []
    for e in episodes:
        for q in e.get("qa_pairs", []):
            if q.get("type") not in qa_types:
                continue
            items.append({
                "question_id": q["question_uuid"],
                "namespace": namespace_for(e["episode_id"]),
                "episode_id": e["episode_id"],
                "domain": e["domain"],
                "question_type": f"ama-{q['type']}",
                "question": q["question"],
                "answer": q["answer"],
                "_episode": e,
            })
    return items


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    return default if raw is None else tuple(s.strip() for s in raw.split(",") if s.strip())


def selection_from_env() -> dict:
    ids = _env_tuple("AMA_EPISODE_IDS", ())
    return {
        "qa_types": _env_tuple("AMA_QA_TYPES", DEFAULT_QA_TYPES),
        "per_domain": int(os.getenv("AMA_PER_DOMAIN", DEFAULT_PER_DOMAIN)),
        "max_tokens": int(os.getenv("AMA_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        "exclude_domains": _env_tuple("AMA_EXCLUDE_DOMAINS", DEFAULT_EXCLUDE_DOMAINS),
        "episode_ids": tuple(int(i) for i in ids) or None,
    }


_SYSTEM_PROMPT = (
    "You answer questions about an AI agent's past task trajectory using only the retrieved "
    "memory of its steps (actions and observations). When the state changed during the "
    "trajectory, answer with the state at the point the question asks about. If the memory does "
    "not contain the answer, say you don't know."
)


class AmaBenchStateAdapter:
    """AMA-Bench State Updating (type C) over Menhir memory; run with ``--recall-only``."""

    benchmark_id = "ama-bench-state"
    name = "AMA-Bench State Updating (menhir memory)"

    def load_items(self, subset=None, limit=None, fixture_path=None) -> list[dict]:  # noqa: ANN001
        path = fixture_path or os.getenv(DATASET_ENV)
        if not path:
            raise ValueError(f"set {DATASET_ENV} to AMA-Bench's open_end_qa_set.jsonl")
        sel = selection_from_env()
        if subset:
            sel["qa_types"] = tuple(s.strip() for s in subset.split(",") if s.strip())
        items = question_items(select_episodes(load_episodes(path), **sel), sel["qa_types"])
        return items[:limit] if limit is not None else items

    def sessions(self, item: dict) -> list[list[dict]]:
        return [render_steps(item["_episode"])]

    def question(self, item: dict) -> str:
        return item.get("question", "")

    def build_messages(self, memory_context: str, question: str) -> list[dict]:
        mem = memory_context.strip() or "(no relevant memory found)"
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Retrieved memory:\n{mem}\n\nQuestion: {question}"},
        ]

    def score(self, item: dict, response_text: str) -> bool:
        # Offline placeholder only: AMA answers are free text, so real runs use --scorer llm-judge.
        gold = str(item.get("answer", "")).strip().lower()
        return bool(gold) and gold in (response_text or "").lower()
