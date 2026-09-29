"""Build our own supersession gold set for AMA-Bench episodes.

Per episode, one model call reads the RAW trajectory and proposes state timelines, each state
with a verbatim evidence quote. Code keeps only states whose quote appears in its own step
(``verify_timelines``), then generates current / previous / timeline questions from the kept
timelines by fixed templates (``gold_questions``). Cached per episode in ``<out>/gold/``;
the combined set is ``<out>/gold_questions.jsonl``. Episodes run in parallel; a 429 stops all.

    python scripts/ama/build_gold.py --dataset .../open_end_qa_set.jsonl \
        --base-url http://127.0.0.1:8765/v1 --model openai/gpt-6-luna --out results/ama-gold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from archolith_bench.harness.ama_bench import (
    apply_check,
    check_messages,
    DATASET_ENV,
    DEFAULT_EXCLUDE_DOMAINS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PER_DOMAIN,
    gold_questions,
    load_episodes,
    select_episodes,
    timeline_messages,
    verify_timelines,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from craft_memories import STOP, RateLimited, ask  # noqa: E402


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _save(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def build_episode(episode: dict, out_dir: Path, *, base_url: str, api_key: str, model: str) -> dict:
    """Extract (cached) -> quote-verify -> subject/change check (cached) -> templated questions."""
    path = out_dir / f"ep{episode['episode_id']}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    with httpx.Client() as client:
        if "raw" not in record:
            if STOP.is_set():
                raise RuntimeError("stopped")
            record["raw"] = ask(client, base_url, api_key, model, timeline_messages(episode))
            _save(path, record)
        timelines, stats = verify_timelines(episode, record["raw"])
        checks: dict[str, str] = record.get("checks", {})
        for n, tl in enumerate(timelines):
            if str(n) not in checks:
                if STOP.is_set():
                    raise RuntimeError("stopped")
                checks[str(n)] = ask(client, base_url, api_key, model, check_messages(episode, tl))
                record["checks"] = checks
                _save(path, record)
    checked = [c for n, tl in enumerate(timelines) if (c := apply_check(tl, checks[str(n)])) is not None]
    stats = stats | {"timelines_after_check": len(checked),
                     "states_after_check": sum(len(t["states"]) for t in checked)}
    questions = gold_questions(episode, checked)
    record |= {"timelines": checked, "stats": stats, "questions": questions}
    _save(path, record)
    return stats | {"episode_id": episode["episode_id"], "questions": len(questions)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.getenv(DATASET_ENV))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.getenv("UPSTREAM_API_KEY", "unused"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--episode-ids", type=lambda v: tuple(int(i) for i in _csv(v)), default=None)
    ap.add_argument("--qa-types", type=_csv, default=("C",))
    ap.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--exclude-domains", type=_csv, default=DEFAULT_EXCLUDE_DOMAINS)
    ap.add_argument("--workers", type=int, default=25)
    args = ap.parse_args(argv)

    if not args.dataset:
        print(f"ERROR: pass --dataset or set {DATASET_ENV}", file=sys.stderr)
        return 2
    episodes = select_episodes(
        load_episodes(args.dataset),
        qa_types=args.qa_types,
        per_domain=args.per_domain,
        max_tokens=args.max_tokens,
        exclude_domains=args.exclude_domains,
        episode_ids=args.episode_ids,
    )
    out_dir = args.out / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(build_episode, e, out_dir, base_url=args.base_url, api_key=args.api_key, model=args.model): e
            for e in episodes
        }
        for fut in as_completed(futures):
            try:
                print(f"  {fut.result()}", flush=True)
            except RateLimited as exc:
                STOP.set()
                failed += 1
                print(f"STOPPING: 429: {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report every episode
                failed += 1
                print(f"  episode {futures[fut]['episode_id']} failed: {exc.__class__.__name__}: {exc}", flush=True)

    rows = []
    for e in episodes:
        path = out_dir / f"ep{e['episode_id']}.json"
        if path.exists():
            rows += json.loads(path.read_text(encoding="utf-8"))["questions"]
    with open(args.out / "gold_questions.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    by_type: dict[str, int] = {}
    for row in rows:
        by_type[row["question_type"]] = by_type.get(row["question_type"], 0) + 1
    print(f"gold questions: {len(rows)} {by_type}; failed episodes: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
