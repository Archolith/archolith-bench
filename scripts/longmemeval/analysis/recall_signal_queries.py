#!/usr/bin/env python
"""Build the seeded query files for the recall signal test. No graph, no network.

Plan: menhir-recall-real-similarity-test-plan-2026-11-05, Data section:

    python recall_signal_queries.py --dataset o500.json --output-dir inputs/

Writes three files:

* ``queries.jsonl``: every question in its own namespace, plus one swapped null per namespace.
  A swapped null asks a question of a different type, drawn with the fixed seed, and its id is
  ``<question_id>@<namespace>``.
* ``distractor-groups.json``: namespaces shuffled into groups of 10, as old to merged namespace.
* ``distractor-queries.jsonl``: every question, searching its merged group.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_labels import load_items  # noqa: E402

SEED = 20261105
GROUP_SIZE = 10


def namespace_of(item: dict[str, Any]) -> str:
    return f"lme-{item['question_id']}"


def own_rows(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"id": str(item["question_id"]), "namespace": namespace_of(item),
         "query": item["question"]}
        for item in items
    ]


def swapped_rows(items: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, str]]:
    """One null per namespace: a question of another type, so the answer is absent."""
    rng = random.Random(seed)
    rows = []
    for item in items:
        pool = [o for o in items if o["question_type"] != item["question_type"]]
        other = rng.choice(pool)
        rows.append({
            "id": f"{other['question_id']}@{namespace_of(item)}",
            "namespace": namespace_of(item),
            "query": other["question"],
        })
    return rows


def merged_groups(
    items: list[dict[str, Any]], seed: int = SEED, size: int = GROUP_SIZE
) -> dict[str, str]:
    """Old namespace to merged namespace."""
    namespaces = sorted(namespace_of(item) for item in items)
    random.Random(seed).shuffle(namespaces)
    return {ns: f"lme-mix-{i // size:03d}" for i, ns in enumerate(namespaces)}


def distractor_rows(
    items: list[dict[str, Any]], groups: dict[str, str]
) -> list[dict[str, str]]:
    return [
        {"id": str(item["question_id"]), "namespace": groups[namespace_of(item)],
         "query": item["question"]}
        for item in items
    ]


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    groups = merged_groups(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "queries.jsonl", own_rows(items) + swapped_rows(items))
    _write_jsonl(args.output_dir / "distractor-queries.jsonl", distractor_rows(items, groups))
    (args.output_dir / "distractor-groups.json").write_text(json.dumps(groups), encoding="utf-8")
    print(f"questions={len(items)} groups={len(set(groups.values()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
