"""AMA-Bench State Updating as an ingest-then-recall memory benchmark.

AMA-Bench (MIT, huggingface.co/datasets/AMA-bench/AMA-bench) pairs long agent trajectories with
expert-written questions. Type ``C`` questions test *state updating*: whether memory returns the
latest state after it changed. Each episode is ingested ONCE, one memory per trajectory step
(``scripts/ama/ingest_ama.py``); questions then run recall-only against that episode's namespace.

Trajectories are agent actions and environment observations; they contain no user turns, so they
never reach Menhir's scalar lane (user TurnEvidence only). This is a recall/supersession check.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATASET_ENV = "AMA_BENCH_DATASET"
DEFAULT_QA_TYPES = ("C",)
DEFAULT_PER_DOMAIN = 5
DEFAULT_MAX_TOKENS = 60_000
DEFAULT_EXCLUDE_DOMAINS = ("OPENWORLD_QA",)
# Synthetic, strictly increasing source times so step order is also time order in the graph.
STEP_TIME_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def namespace_for(episode_id: int) -> str:
    return f"ama-ep{episode_id}"


def step_time(step_idx: int) -> str:
    return (STEP_TIME_BASE + timedelta(minutes=step_idx)).isoformat()


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
