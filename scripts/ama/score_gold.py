"""Score AMA answers against our supersession gold set: correct / stale / wrong / unknown.

Reads the harness evidence JSON (``--answers``: per-question ``task_id`` and ``response_text``)
and the gold set (``--gold``: ``gold_questions.jsonl``). A judge model labels every answer:

* current / previous: ``correct`` (conveys the gold value), ``stale`` (gives one of the earlier,
  superseded values instead -- the supersession failure), ``wrong`` (another value), ``unknown``
  (declines or says it cannot tell);
* timeline: ``correct`` (all states, in order), ``partial`` (some, or out of order), ``wrong``,
  ``unknown``.

Labels are cached per question in ``<out>/labels.jsonl``; the report goes to ``<out>/scores.json``.
Unparseable judge output is recorded as ``unscored`` and counted, never guessed. A 429 stops all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from craft_memories import STOP, RateLimited, ask  # noqa: E402

STATE_LABELS = ("correct", "stale", "wrong", "unknown")
TIMELINE_LABELS = ("correct", "partial", "wrong", "unknown")

JUDGE_SYSTEM = (
    "You grade one answer from a memory benchmark about an AI agent's task. Compare the answer "
    "to the gold answer and to the stale answers (values that were true EARLIER but were later "
    "replaced). Reply with JSON only: {\"label\": \"...\"}. For a state question use: correct = "
    "the answer conveys the gold value (wording may differ); stale = it gives a stale value "
    "instead of the gold value; wrong = it gives some other value; unknown = it declines or says "
    "it cannot tell. For a timeline question use: correct = every gold state appears in the gold "
    "order; partial = some gold states, or the right states in the wrong order; wrong = none of "
    "the gold states; unknown = it declines."
)


def judge_messages(gold: dict, response: str) -> list[dict]:
    kind = "timeline" if gold["question_type"] == "timeline" else "state"
    user = (
        f"Question type: {kind}\nQuestion: {gold['question']}\nGold answer: {gold['answer']}\n"
        f"Stale answers: {json.dumps(gold.get('stale_answers') or [])}\n\nAnswer to grade:\n{response}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_label(text: str, question_type: str) -> str:
    allowed = TIMELINE_LABELS if question_type == "timeline" else STATE_LABELS
    try:
        label = json.loads((text or "").strip().strip("`").removeprefix("json").strip()).get("label")
    except (json.JSONDecodeError, AttributeError):
        return "unscored"
    return label if label in allowed else "unscored"


def summarize(rows: list[dict]) -> dict:
    """Label rates overall, by question type, and by domain x type."""
    def rates(group: list[dict]) -> dict:
        counts = Counter(r["label"] for r in group)
        n = len(group)
        return {"n": n, **{k: counts[k] for k in sorted(counts)},
                **{f"{k}_rate": round(counts[k] / n, 3) for k in sorted(counts)}} if n else {"n": 0}

    by_type: dict[str, list] = defaultdict(list)
    by_domain_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
        by_domain_type[f"{r['domain']}/{r['question_type']}"].append(r)
    return {
        "overall": rates(rows),
        "by_type": {k: rates(v) for k, v in sorted(by_type.items())},
        "by_domain_type": {k: rates(v) for k, v in sorted(by_domain_type.items())},
    }


def load_answers(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    arms = data.get("arms") or {}
    results = next(iter(arms.values()))["results"] if arms else data.get("results", [])
    return {str(r["task_id"]): str(r.get("response_text") or "") for r in results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.getenv("UPSTREAM_API_KEY", "unused"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=25)
    args = ap.parse_args(argv)

    gold = {g["question_id"]: g for g in (json.loads(line) for line in open(args.gold, encoding="utf-8") if line.strip())}
    answers = load_answers(args.answers)
    missing = sorted(set(gold) - set(answers))
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "labels.jsonl"
    done = {json.loads(line)["question_id"]: json.loads(line) for line in open(cache_path, encoding="utf-8")} if cache_path.exists() else {}
    lock = threading.Lock()

    def grade(qid: str) -> dict:
        if qid in done:
            return done[qid]
        if STOP.is_set():
            raise RuntimeError("stopped")
        g = gold[qid]
        with httpx.Client() as client:
            raw = ask(client, args.base_url, args.api_key, args.model, judge_messages(g, answers[qid]))
        row = {"question_id": qid, "domain": g["domain"], "question_type": g["question_type"],
               "label": parse_label(raw, g["question_type"]), "answer": answers[qid][:500], "gold": g["answer"]}
        with lock, open(cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(grade, qid): qid for qid in sorted(set(gold) & set(answers))}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except RateLimited as exc:
                STOP.set()
                failed += 1
                print(f"STOPPING: 429: {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report every question
                failed += 1
                print(f"  {futures[fut]} failed: {exc.__class__.__name__}: {exc}", flush=True)
    report = summarize(rows) | {"missing_answers": missing, "failed": failed}
    (args.out / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"]), flush=True)
    print(json.dumps(report["by_type"], indent=1), flush=True)
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
