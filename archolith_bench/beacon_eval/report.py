"""Markdown report: per-condition means, median tokens, and every per-run row."""

from __future__ import annotations

import statistics
from collections import defaultdict

from archolith_bench.beacon_eval.models import RunResult

METRICS = (
    "answered",
    "doc_recall",
    "file_recall",
    "file_precision",
    "command_recall",
    "guardrail_recall",
    "point_recall",
    "verdict_correct",
    "risky_false_positive",
    "citation_location_validity",
    "evidence_recall",
)


def _mean(values: list[float]) -> str:
    return f"{statistics.fmean(values):.2f}" if values else "-"


def render(results: list[RunResult], header: dict[str, str], stopped: str = "") -> str:
    by_condition: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_condition[result.condition].append(result)
    lines = ["# Beacon agent-task evaluation", ""]
    lines += [f"- **{key}:** {value}" for key, value in header.items()]
    lines += [f"- **Runs completed:** {len(results)}"]
    if stopped:
        lines += [f"- **Stopped early:** {stopped}"]
    lines += ["", "## By condition", ""]
    lines += ["| Condition | Runs | " + " | ".join(METRICS) + " | median tokens |"]
    lines += ["|---" * (len(METRICS) + 3) + "|"]
    for condition in sorted(by_condition):
        runs = by_condition[condition]
        means = [
            _mean([r.scores[m] for r in runs if m in r.scores]) for m in METRICS
        ]
        tokens = statistics.median([r.total_tokens for r in runs]) if runs else 0
        lines += [f"| {condition} | {len(runs)} | " + " | ".join(means) + f" | {tokens:,.0f} |"]
    lines += ["", "## Runs", "", "| Repo | Task | Cond | Rep | Tokens | Tools | s | Scores | Error |"]
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        scores = ", ".join(f"{k}={v:.2f}" for k, v in sorted(r.scores.items()))
        lines += [
            f"| {r.repo} | {r.task_id} | {r.condition} | {r.repeat} | {r.total_tokens:,} | "
            f"{r.tool_calls} | {r.seconds} | {scores} | {r.error[:60]} |"
        ]
    return "\n".join(lines) + "\n"
