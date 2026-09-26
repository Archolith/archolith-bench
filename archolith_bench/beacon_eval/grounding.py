"""Mechanical grounding check for gold answers: every citation resolves and quotes its lines.

A gold answer is only usable if each expected item is backed by a citation whose path
exists in the pinned checkout, whose line range fits the file, and whose ``quote``
appears verbatim (whitespace-normalized) in those lines.
"""

from __future__ import annotations

import json
from pathlib import Path

_GOLD_LISTS = ("docs", "files", "commands", "guardrails", "points", "risky")


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def check_task_file(task_path: Path, checkout: Path) -> list[str]:
    """Problems with one task file's gold citations (empty list = grounded)."""
    data = json.loads(task_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    citations = data.get("gold_citations") or []
    cited_items = {str(c.get("item", "")) for c in citations if isinstance(c, dict)}
    gold = data.get("gold") or {}
    for key in _GOLD_LISTS:
        if key == "risky":
            continue  # risky items are prohibitions, not claims about the repository
        for entry in gold.get(key) or []:
            # A guardrail or point may list accepted wordings; its first wording is the cited id.
            item = entry[0] if isinstance(entry, list) and entry else entry
            if item not in cited_items:
                problems.append(f"{key} item has no citation: {item}")
    if gold.get("verdict") and gold["verdict"] not in cited_items:
        problems.append(f"verdict has no citation: {gold['verdict']}")
    root = checkout.resolve()
    for path in gold.get("acceptable_files") or []:
        # Acceptable extras are not claims, so they need no citation, only a real path, or a
        # new file (e.g. an archive destination) whose folder exists.
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not (target.is_file() or target.parent.is_dir()):
            problems.append(f"acceptable file missing: {path}")
    for citation in citations:
        path = str(citation.get("path", ""))
        target = (root / path).resolve()
        if not path or not target.is_relative_to(root) or not target.is_file():
            problems.append(f"cited path missing: {path}")
            continue
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start, end = citation.get("line_start"), citation.get("line_end")
        if not (isinstance(start, int) and isinstance(end, int) and 1 <= start <= end <= len(lines)):
            problems.append(f"bad line range {start}-{end} in {path} ({len(lines)} lines)")
            continue
        quote = _norm(str(citation.get("quote", "")))
        span = _norm(" ".join(lines[start - 1 : end]))
        if not quote or quote not in span:
            problems.append(f"quote not in {path}:{start}-{end}: {citation.get('quote', '')[:60]}")
    return problems


def check_tree(tasks_dir: Path, checkout: Path) -> dict[str, list[str]]:
    return {path.name: check_task_file(path, checkout) for path in sorted(tasks_dir.glob("*.json"))}
