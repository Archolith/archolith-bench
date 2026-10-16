"""Run a stratified LongMemEval slice: N questions per question type.

Calls the existing ``archolith-bench harness longmemeval-menhir`` CLI once per
question type (e.g. ``--subset temporal-reasoning --limit 20``), then aggregates
the 6 result files into one ``results.json`` and computes the overall + per-type
scores.

The locally cached official LongMemEval oracle dataset contains exactly these
six question types. Keep this tuple aligned with ``answer_matrix.sh`` and fail
closed if any requested slice is incomplete.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The 6 stratified types for the CI slice. Override via env BENCH_SLICE_TYPES.
DEFAULT_SLICE_TYPES = (
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
)

DEFAULT_QUESTIONS_PER_TYPE = 20


@dataclass
class TypeResult:
    type: str
    score: float
    n: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    per_question: list[dict] = field(default_factory=list)
    error: str | None = None


@dataclass
class StratifiedResult:
    overall: float
    by_type: dict[str, float]
    n_total: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    per_question: dict[str, dict]  # id -> {type, score, passed}
    type_results: list[TypeResult]
    llm_calls_used: int = 0


def run_stratified_slice(
    *,
    menhir_url: str,
    output_dir: str | Path,
    judge_model: str = "gpt-6-luna",
    questions_per_type: int = DEFAULT_QUESTIONS_PER_TYPE,
    slice_types: tuple[str, ...] = DEFAULT_SLICE_TYPES,
    scorer: str = "llm-judge",
    recall_limit: int = 10,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> StratifiedResult:
    """Invoke the harness once per type, then aggregate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    type_results: list[TypeResult] = []
    for q_type in slice_types:
        out_file = output_dir / f"harness_longmemeval-menhir_{q_type}.json"
        result = _run_one_type(
            menhir_url=menhir_url,
            q_type=q_type,
            limit=questions_per_type,
            out_file=out_file,
            judge_model=judge_model,
            scorer=scorer,
            recall_limit=recall_limit,
            extra_args=extra_args,
            dry_run=dry_run,
        )
        type_results.append(result)

    return aggregate_results(type_results)


def _run_one_type(
    *,
    menhir_url: str,
    q_type: str,
    limit: int,
    out_file: Path,
    judge_model: str,
    scorer: str,
    recall_limit: int,
    extra_args: list[str] | None,
    dry_run: bool,
) -> TypeResult:
    cmd = [
        sys.executable, "-m", "archolith_bench",
        "harness", "longmemeval-menhir",
        "--menhir-url", menhir_url,
        "--subset", q_type,
        "--limit", str(limit),
        "--recall-only",
        "--scorer", scorer,
        "--judge-model", judge_model,
        "--out", str(out_file),
        "--format", "json",
        "--recall-limit", str(recall_limit),
        "--confirm-menhir-reset",  # recall-only doesn't actually reset, but the flag suppresses the guard
    ]
    if extra_args:
        cmd.extend(extra_args)

    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return TypeResult(type=q_type, score=0.0, n=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                          error="dry-run")

    print(f"  running: {q_type} (limit={limit})")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        return TypeResult(
            type=q_type, score=0.0, n=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=f"exit {proc.returncode}: {proc.stderr[:500]}",
        )

    if not out_file.exists():
        return TypeResult(
            type=q_type, score=0.0, n=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=f"output file not written: {out_file}",
        )

    result = _parse_type_result(q_type, out_file)
    if result.error is None and result.n != limit:
        result.error = f"incomplete slice: expected {limit} questions, got {result.n}"
    return result


def _parse_type_result(q_type: str, path: Path) -> TypeResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    arms = raw.get("arms", {})
    # Recall-only Mode-B runs a single arm (the first one in the dict)
    if not arms:
        return TypeResult(type=q_type, score=0.0, n=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                          error="no arms in result")
    arm_name = next(iter(arms))
    arm = arms[arm_name]
    per_q = [
        {
            "id": t.get("task_id", ""),
            "type": q_type,
            "score": 1.0 if t.get("passed") else 0.0,
            "passed": bool(t.get("passed")),
        }
        for t in arm.get("results", [])
    ]
    return TypeResult(
        type=q_type,
        score=float(arm.get("score", 0.0)),
        n=int(arm.get("n", 0)),
        input_tokens=int(arm.get("input_tokens", 0)),
        output_tokens=int(arm.get("output_tokens", 0)),
        cost_usd=float(arm.get("cost_usd", 0.0)),
        per_question=per_q,
    )


def aggregate_results(type_results: list[TypeResult]) -> StratifiedResult:
    """Aggregate per-type results into one StratifiedResult."""
    total_n = sum(t.n for t in type_results)
    total_in = sum(t.input_tokens for t in type_results)
    total_out = sum(t.output_tokens for t in type_results)
    total_cost = sum(t.cost_usd for t in type_results)

    # Overall score = weighted average (by n) of type scores
    if total_n > 0:
        overall = sum(t.score * t.n for t in type_results) / total_n
    else:
        overall = 0.0

    by_type = {t.type: t.score for t in type_results}

    per_question: dict[str, dict] = {}
    for t in type_results:
        for q in t.per_question:
            if q["id"]:
                per_question[q["id"]] = q

    return StratifiedResult(
        overall=overall,
        by_type=by_type,
        n_total=total_n,
        input_tokens=total_in,
        output_tokens=total_out,
        cost_usd=total_cost,
        per_question=per_question,
        type_results=type_results,
    )
