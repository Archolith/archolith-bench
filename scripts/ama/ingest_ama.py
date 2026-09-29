"""Ingest selected AMA-Bench episodes into a throwaway Menhir, once per episode.

Each episode goes to its own namespace (``ama-ep<id>``): the task, then one memory per trajectory
step, strictly in step order with ``wait=true`` and step-ordered synthetic source times. Episodes
run in parallel (``--workers``); steps within an episode never do. Questions are then answered
recall-only by the ``ama-bench-state`` harness adapter against these namespaces.

    AMA_BENCH_DATASET=.../open_end_qa_set.jsonl python scripts/ama/ingest_ama.py \
        --menhir-url http://127.0.0.1:8102 --out results/ama-smoke --episode-ids 3,7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from archolith_bench.harness.ama_bench import (
    DATASET_ENV,
    DEFAULT_EXCLUDE_DOMAINS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PER_DOMAIN,
    load_episodes,
    namespace_for,
    render_steps,
    select_episodes,
)
from archolith_bench.harness.memory_ab import assert_not_production
from archolith_bench.harness.menhir_client import HttpMenhirClient


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _reset_namespace(base_url: str, namespace: str) -> None:
    with httpx.Client(timeout=300.0) as admin:
        admin.post(
            base_url.rstrip("/") + "/api/internal/backend/delete_namespace",
            json={"namespace": namespace, "force": True},
        ).raise_for_status()
        admin.post(
            base_url.rstrip("/") + "/api/phase3/reset", params={"namespace": namespace}
        ).raise_for_status()


def ingest_episode(base_url: str, episode: dict, *, reset: bool, tries: int = 3) -> dict:
    namespace = namespace_for(episode["episode_id"])
    if reset:
        _reset_namespace(base_url, namespace)
    steps = render_steps(episode)
    failed: list[int] = []
    t0 = time.time()
    with HttpMenhirClient(base_url, timeout=300.0) as client:
        for index, turn in enumerate(steps):
            for attempt in range(tries):
                try:
                    client.ingest(
                        namespace,
                        turn["role"],
                        turn["content"],
                        occurred_at=turn["occurred_at"],
                        session_id=namespace,
                        wait=True,
                    )
                    break
                except httpx.HTTPError as exc:
                    if attempt == tries - 1:
                        failed.append(index)
                        print(f"  {namespace} step {index} failed: {exc.__class__.__name__}", flush=True)
                    else:
                        time.sleep(2 ** (attempt + 1))
    return {
        "episode_id": episode["episode_id"],
        "namespace": namespace,
        "domain": episode["domain"],
        "steps": len(steps),
        "failed_steps": failed,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menhir-url", required=True)
    ap.add_argument("--dataset", default=os.getenv(DATASET_ENV))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episode-ids", type=lambda v: tuple(int(i) for i in _csv(v)), default=None)
    ap.add_argument("--qa-types", type=_csv, default=("C",))
    ap.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--exclude-domains", type=_csv, default=DEFAULT_EXCLUDE_DOMAINS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reset", action="store_true", help="force-clear each episode namespace first")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; ingest nothing")
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
    plan = [
        {
            "episode_id": e["episode_id"],
            "domain": e["domain"],
            "steps": len(e.get("trajectory", [])) + 1,
            "tokens": e.get("total_tokens"),
            "questions": sum(q.get("type") in args.qa_types for q in e.get("qa_pairs", [])),
        }
        for e in episodes
    ]
    print(
        f"plan: {len(plan)} episodes, {sum(p['steps'] for p in plan)} steps, "
        f"{sum(p['tokens'] or 0 for p in plan):,} tokens, {sum(p['questions'] for p in plan)} questions",
        flush=True,
    )
    if args.dry_run:
        for p in plan:
            print(f"  {p}")
        return 0

    assert_not_production(args.menhir_url)
    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(ingest_episode, args.menhir_url, e, reset=args.reset): e for e in episodes}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"  done {result}", flush=True)
    results.sort(key=lambda r: r["episode_id"])
    (args.out / "ama_ingest_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = sum(len(r["failed_steps"]) for r in results)
    print(f"ingested {len(results)} episodes; failed steps: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
