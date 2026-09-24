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


_ANNOTATION = re.compile(r"\s+\(.*\)\s*$")
# A short label, a colon and a space, then something that looks like a path (has "/" or ".").
_LABEL = re.compile(r"^[^:/`]{1,60}:\s+(`?[^\s`]*[/.][^\s`]*`?)$")


def _norm_path(value: str) -> str:
    """Comparable path: drops a trailing "(note)", backticks and leading "./" only.

    Dot-directories such as ``.agent/`` must survive (``lstrip("./")`` used to eat them).
    """
    path = _ANNOTATION.sub("", value.strip()).strip().strip("`").replace("\\", "/")
    # "Implementation: scripts/x.py" -> "scripts/x.py" (a label before a path-like value).
    labelled = _LABEL.match(path)
    if labelled:
        path = labelled.group(1).strip().strip("`")
    while path.startswith("./"):
        path = path[2:]
    return path.lower()


def _norm_command(value: str) -> str:
    return " ".join(value.strip().strip("`").split()).lower()


def _flatten(value: Any) -> list[str]:
    if isinstance(value, str | int | float) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, dict):
        return [item for inner in value.values() for item in _flatten(inner)]
    if isinstance(value, list | tuple):
        return [item for inner in value for item in _flatten(inner)]
    return []


def _as_list(answer: dict[str, Any], key: str) -> list[str]:
    """Answer entries as strings; grouped answers ({"source": [...], "tests": [...]}) are flattened."""
    return _flatten(answer.get(key) or [])


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


def _precision(
    expected: tuple[str, ...], given: list[str], norm: Any, acceptable: tuple[str, ...] = ()
) -> float | None:
    """Share of *given* that is required or acceptable (None when nothing is required)."""
    if not expected or not given:
        return None if not expected else 0.0
    want = {norm(item) for item in (*expected, *acceptable)}
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


def _answer_spans(answer: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Answer citations with an integer line range (``line_end`` defaults to ``line_start``)."""
    citations = answer.get("citations") or []
    spans = []
    for item in citations if isinstance(citations, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        start = item.get("line_start")
        end = item.get("line_end", start)
        if isinstance(start, int) and isinstance(end, int):
            spans.append((_norm_path(item["path"]), start, end))
    return spans


def citation_location_validity(answer: dict[str, Any], repo_root: Path) -> float | None:
    """Share of citations whose path exists in the checkout and whose line range fits.

    A citation without a line range is invalid. This checks location only, not
    whether the lines support the claim (see :func:`evidence_recall`).
    """
    citations = answer.get("citations") or []
    if not isinstance(citations, list) or not citations:
        return None
    root = repo_root.resolve()
    valid = 0
    for path, start, end in _answer_spans(answer):
        target = (root / path).resolve()
        try:
            if not target.is_relative_to(root) or not target.is_file():
                continue
            lines = target.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        except OSError:
            continue
        if 1 <= start <= end <= lines:
            valid += 1
    return valid / len(citations)


def evidence_recall(answer: dict[str, Any], gold: Gold) -> float | None:
    """Share of distinct gold citation spans overlapped by an answer citation in the same file.

    A lower bound: correct evidence cited from elsewhere does not count.
    """
    if not gold.evidence:
        return None
    spans = _answer_spans(answer)
    hit = sum(
        1
        for path, start, end in gold.evidence
        if any(
            given == _norm_path(path) and g_start <= end and start <= g_end
            for given, g_start, g_end in spans
        )
    )
    return hit / len(gold.evidence)


def score(answer: dict[str, Any] | None, gold: Gold, repo_root: Path) -> dict[str, float]:
    """Metrics in [0, 1]; a metric the gold does not define is omitted."""
    if answer is None:
        return {"answered": 0.0}
    metrics: dict[str, float | None] = {
        "answered": 1.0,
        "doc_recall": _recall(gold.docs, _as_list(answer, "docs"), _norm_path),
        "file_recall": _recall(gold.files, _as_list(answer, "files"), _norm_path),
        "file_precision": _precision(
            gold.files, _as_list(answer, "files"), _norm_path, gold.acceptable_files
        ),
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
        "point_recall": (
            None
            if not gold.points
            else sum(
                1
                for accepted in gold.points
                if guardrail_met(
                    accepted, _as_list(answer, "findings") + _as_list(answer, "plan")
                )
            )
            / len(gold.points)
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
        "citation_location_validity": citation_location_validity(answer, repo_root),
        "evidence_recall": evidence_recall(answer, gold),
    }
    return {key: value for key, value in metrics.items() if value is not None}
