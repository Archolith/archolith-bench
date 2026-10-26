"""Resumable checkpoint for the Mode-B memory harness.

A full LongMemEval Mode B run ingests hundreds of thousands of episodes and takes
hours; the memory harness has no native resume, so a crash or rate-limit abort loses
everything. This persists each per-item TaskResult to an append-only JSONL file keyed
by (arm, task_id). On restart the driver skips items already recorded and continues,
so the run survives interruptions and OpenAI 429s (rerun the same command).

The checkpoint stores ANSWER-level results (score/tokens), not the menhir graph: each
benchmark item uses a fresh, immediately-reset namespace, so completed items need no
graph state and the throwaway menhir/Neo4j can be torn down between runs safely.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

from .base import TaskResult


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip()).strip("-") or "default"


def checkpoint_path_for(
    output_dir: str | Path,
    benchmark_id: str,
    model: str,
    *,
    variant: str | None = None,
    prompt_tag: str | None = None,
) -> Path:
    """Deterministic checkpoint file for a (benchmark, variant, answer-model[, prompt]) run.

    Keyed on the answer model and dataset variant because both change the scored
    outcome; mixing them in one checkpoint would corrupt the aggregate. An adapter whose
    answer prompt changes passes ``prompt_tag`` so a resume never mixes prompt versions.
    """
    variant = variant if variant is not None else (os.getenv("LONGMEMEVAL_VARIANT", "s") or "s")
    tag = f"_p{_slug(prompt_tag)}" if prompt_tag else ""
    name = f".checkpoint_{_slug(benchmark_id)}_{_slug(variant)}_{_slug(model)}{tag}.jsonl"
    return Path(output_dir) / name


def checkpoint_prompt_tag(
    adapter: object, *, drill_down: bool = False, counted_authority_only: bool = False,
    source_memories: bool = False, official_judge: bool = False,
) -> str | None:
    """The ``prompt_tag`` for a run: the adapter's prompt fingerprint, marked for drill-down,
    counted-authority-only and source-memory rendering, and for grading with LongMemEval's
    official judge prompts.

    Each changes how answers are produced or graded but not the arm names, so without the mark a
    resume would mix them with plain answers (or older grades) in one checkpoint.
    """
    tag = getattr(adapter, "prompt_fingerprint", None)
    if drill_down:
        tag = f"{tag or ''}dd"
    if counted_authority_only:
        tag = f"{tag or ''}ca"
    if source_memories:
        tag = f"{tag or ''}sm"
    if official_judge:
        tag = f"{tag or ''}oj"
    return tag


class MemoryCheckpoint:
    """Append-only JSONL checkpoint of completed (arm, task_id) TaskResults."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._done: dict[tuple[str, str], TaskResult] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    arm = str(rec["arm"])
                    res = rec["result"]
                    tr = TaskResult(
                        task_id=str(res["task_id"]),
                        response_text=str(res.get("response_text") or ""),
                        input_tokens=int(res.get("input_tokens") or 0),
                        output_tokens=int(res.get("output_tokens") or 0),
                        latency_ms=float(res.get("latency_ms") or 0.0),
                        correct=bool(res.get("correct")),
                        raw_usage=dict(res.get("raw_usage") or {}),
                        scorer_input_tokens=int(res.get("scorer_input_tokens") or 0),
                        scorer_output_tokens=int(res.get("scorer_output_tokens") or 0),
                        scorer_raw_usage=dict(res.get("scorer_raw_usage") or {}),
                        unofficial_correct=(
                            bool(res["unofficial_correct"])
                            if res.get("unofficial_correct") is not None
                            else None
                        ),
                        unofficial_scorer_raw_usage=dict(
                            res.get("unofficial_scorer_raw_usage") or {}
                        ),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # Tolerate a torn final line from a hard kill; skip it.
                    continue
                self._done[(arm, tr.task_id)] = tr

    def get(self, arm: str, task_id: str) -> TaskResult | None:
        """Return the recorded result for (arm, task_id), or None if not yet done."""
        return self._done.get((arm, task_id))

    def record(self, arm: str, task_id: str, result: TaskResult) -> None:
        """Append a completed result and flush so a later crash keeps it."""
        self._done[(arm, task_id)] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            rec = {"arm": arm, "task_id": task_id, "ts": time.time(), "result": asdict(result)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def done_count(self, arm: str | None = None) -> int:
        """Number of completed items, optionally for one arm."""
        if arm is None:
            return len(self._done)
        return sum(1 for (a, _t) in self._done if a == arm)
