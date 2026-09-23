"""Deterministic scoring of an agent's final answer against the gold answer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from archolith_bench.beacon_eval.models import Gold

_FENCED_JSON = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_answer(text: str) -> dict[str, Any] | None:
    """The last fenced ```json block in *text* that parses to an object."""
    for raw in reversed(_FENCED_JSON.findall(text or "")):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _norm_path(value: str) -> str:
    return value.strip().strip("`").replace("\\", "/").lstrip("./").lower()


def _norm_command(value: str) -> str:
    return " ".join(value.strip().strip("`").split()).lower()


def _as_list(answer: dict[str, Any], key: str) -> list[str]:
    value = answer.get(key) or []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if isinstance(item, str | int | float)]


def _recall(expected: tuple[str, ...], given: list[str], norm: Any) -> float | None:
    if not expected:
        return None
    have = {norm(item) for item in given}
    return sum(1 for item in expected if norm(item) in have) / len(expected)


def _precision(expected: tuple[str, ...], given: list[str], norm: Any) -> float | None:
    if not expected or not given:
        return None if not expected else 0.0
    want = {norm(item) for item in expected}
    return sum(1 for item in given if norm(item) in want) / len(given)


def citation_validity(answer: dict[str, Any], repo_root: Path) -> float | None:
    """Share of cited ``{path, line_start?, line_end?}`` that exist in the checkout."""
    citations = answer.get("citations") or []
    if not isinstance(citations, list) or not citations:
        return None
    valid = 0
    for item in citations:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        target = (repo_root / _norm_path(item["path"])).resolve()
        try:
            if not target.is_relative_to(repo_root.resolve()) or not target.is_file():
                continue
            lines = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        except OSError:
            continue
        start = item.get("line_start")
        end = item.get("line_end", start)
        if start is None or (
            isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= lines
        ):
            valid += 1
    return valid / len(citations)


def score(answer: dict[str, Any] | None, gold: Gold, repo_root: Path) -> dict[str, float]:
    """Metrics in [0, 1]; a metric the gold does not define is omitted."""
    if answer is None:
        return {"answered": 0.0}
    metrics: dict[str, float | None] = {
        "answered": 1.0,
        "doc_recall": _recall(gold.docs, _as_list(answer, "docs"), _norm_path),
        "file_recall": _recall(gold.files, _as_list(answer, "files"), _norm_path),
        "file_precision": _precision(gold.files, _as_list(answer, "files"), _norm_path),
        # A gold command may be a prefix (the repo's example continues with paths).
        "command_recall": (
            None
            if not gold.commands
            else sum(
                1
                for expected in gold.commands
                if any(
                    _norm_command(given).startswith(_norm_command(expected))
                    for given in _as_list(answer, "commands")
                )
            )
            / len(gold.commands)
        ),
        "guardrail_recall": (
            None
            if not gold.guardrails
            else sum(
                1
                for keyword in gold.guardrails
                if keyword.lower() in " ".join(_as_list(answer, "guardrails")).lower()
            )
            / len(gold.guardrails)
        ),
        "verdict_correct": (
            None
            if not gold.verdict
            else float(str(answer.get("verdict", "")).strip().lower() == gold.verdict.lower())
        ),
        # Only what the answer tells the agent to run counts: a plan that warns
        # "never use git add -A" must not be scored as risky.
        "risky_false_positive": (
            None
            if not gold.risky
            else float(
                any(
                    item.lower() in command.lower()
                    for item in gold.risky
                    for command in _as_list(answer, "commands")
                )
            )
        ),
        "citation_validity": citation_validity(answer, repo_root),
    }
    return {key: value for key, value in metrics.items() if value is not None}
