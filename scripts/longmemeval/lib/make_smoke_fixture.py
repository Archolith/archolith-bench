"""Generate the smoke fixtures used by run_date_smoke.sh from the local LongMemEval cache.

These are verbatim slices of the benchmark, not hand-authored regression fixtures like the ones
described in fixtures/README.md, so they are GENERATED rather than committed -- 12 items is ~360 KB
of dataset text that is already on disk in the HuggingFace cache.

  date   : the single knowledge-update item whose answer depends entirely on valid_at ordering
           (session 0 "about an hour" -> session 1 "about two hours"). Used to prove menhir's
           ingest backdating works with LME_BACKFILL_DATES=0.
  multi  : 2 items x each of the 6 question types, to check a fix across categories rather than
           on one item. Includes single-session-assistant, which is the category that depends on
           assistant-turn content surviving ingestion. `--per-type N` takes N per type instead
           (N=1 writes multi-smoke-6.json, the fixture builds/per-type-1-acceptance.yaml pins).

Usage:
  python scripts/longmemeval/lib/make_smoke_fixture.py date
  python scripts/longmemeval/lib/make_smoke_fixture.py multi [--per-type N]
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

#: The date-ordering item. Its two sessions state different values for the same fact.
DATE_ITEM = "cc5ded98"

#: Items per question type for the multi-type smoke.
PER_TYPE = 2

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def load_oracle() -> list[dict]:
    hits = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not hits:
        raise SystemExit("LongMemEval oracle dataset not found in the HuggingFace cache")
    with open(hits[0], encoding="utf-8") as handle:
        return json.load(handle)


def write(name: str, items: list[dict]) -> str:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=1)
    turns = sum(sum(len(s) for s in i["haystack_sessions"]) for i in items)
    print(f"wrote {path}  items={len(items)} turns={turns}")
    return path


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "date"
    per_type = PER_TYPE
    if argv[2:3] == ["--per-type"] and len(argv) == 4 and argv[3].isdigit() and int(argv[3]) > 0:
        per_type = int(argv[3])
    elif len(argv) > 2:
        raise SystemExit("usage: make_smoke_fixture.py date | multi [--per-type N]")
    items = load_oracle()

    if which == "date":
        picked = [i for i in items if i.get("question_id") == DATE_ITEM]
        if not picked:
            raise SystemExit(f"{DATE_ITEM} not present in the cached dataset")
        write("date-smoke-cc5ded98.json", picked)
        return 0

    if which == "multi":
        by_type: dict[str, list[dict]] = collections.defaultdict(list)
        for item in items:
            by_type[str(item.get("question_type"))].append(item)
        picked: list[dict] = []
        for question_type in sorted(by_type):
            picked.extend(by_type[question_type][:per_type])
        write(f"multi-smoke-{len(picked)}.json", picked)
        for item in picked:
            turns = sum(len(s) for s in item["haystack_sessions"])
            print(f"  {item['question_id']:<14} {str(item.get('question_type')):<28} turns={turns}")
        return 0

    raise SystemExit(f"unknown fixture {which!r}; expected 'date' or 'multi'")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
