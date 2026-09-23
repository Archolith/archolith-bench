"""Deterministic scoring of an agent's final answer against the gold answer."""

from __future__ import annotations

import json
import re
import shlex
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


def _command_tokens(value: str) -> list[str]:
    text = _norm_command(value)
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def command_matches(given: str, expected: str, allowed_flags: tuple[str, ...] = ()) -> bool:
    """*given* is *expected* followed only by plain arguments (paths) or allowed flags.

    The repository's examples often continue with paths, so extra arguments pass; an
    added flag such as ``--fix`` changes what the command does and fails unless the
    task's gold allows it.
    """
    want, have = _command_tokens(expected), _command_tokens(given)
    if have[: len(want)] != want:
        return False
    allowed = {flag.lower() for flag in allowed_flags}
    return all(
        not token.startswith("-") or token in allowed or token.split("=", 1)[0] in allowed
        for token in have[len(want):]
    )


_WORD = re.compile(r"[a-z0-9_][a-z0-9_./-]*")


def _words(text: str) -> set[str]:
    # Paths and flags keep inner "." "/" "-"; sentence punctuation at the end is dropped.
    return {word.rstrip("./-") for word in _WORD.findall(text.lower().replace("`", ""))} - {""}


def guardrail_met(accepted: str | tuple[str, ...], entries: list[str]) -> bool:
    """One answer entry holds every word of one accepted wording (any order).

    Words are matched within a single entry so that pieces spread across unrelated
    guardrails do not add up. Negation is not detected: rules are often phrased as
    prohibitions, so "not" or "never" cannot count against an entry.
    """
    wordings = (accepted,) if isinstance(accepted, str) else accepted
    wanted = [_words(wording) for wording in wordings if _words(wording)]
    return any(want <= _words(entry) for entry in entries for want in wanted)


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


_PROHIBITION = re.compile(r"\b(?:never|don't|do not|avoid|not|instead of)\b")


def risky_instructed(item: str, commands: list[str], plan: list[str]) -> bool:
    """The answer tells the agent to do *item*: in a command, or in a plan step
    where no prohibition word comes before it (a warning such as "never use
    git add -A" does not count).
    """
    needle = item.lower()
    if any(needle in command.lower() for command in commands):
        return True
    for step in plan:
        text = step.lower().replace("\u2019", "'")
        start = text.find(needle)
        while start != -1:
            if not _PROHIBITION.search(text[:start]):
                return True
            start = text.find(needle, start + 1)
    return False


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
        "command_recall": (
            None
            if not gold.commands
            else sum(
                1
                for expected in gold.commands
                if any(
                    command_matches(given, expected, gold.allowed_flags)
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
                for accepted in gold.guardrails
                if guardrail_met(accepted, _as_list(answer, "guardrails"))
            )
            / len(gold.guardrails)
        ),
        "verdict_correct": (
            None
            if not gold.verdict
            else float(str(answer.get("verdict", "")).strip().lower() == gold.verdict.lower())
        ),
        "risky_false_positive": (
            None
            if not gold.risky
            else float(
                any(
                    risky_instructed(item, _as_list(answer, "commands"), _as_list(answer, "plan"))
                    for item in gold.risky
                )
            )
        ),
        "citation_validity": citation_validity(answer, repo_root),
    }
    return {key: value for key, value in metrics.items() if value is not None}
