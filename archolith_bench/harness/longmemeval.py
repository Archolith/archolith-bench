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

from .. import prompts as _prompts
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


# Text lives in archolith_bench/prompts/answer/longmemeval-memory.system.txt.
_MEMORY_SYSTEM_PROMPT = _prompts.load(_prompts.ANSWER_LONGMEMEVAL)


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
