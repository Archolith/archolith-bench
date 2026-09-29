"""Write live per-episode AMA ingest progress for the dashboard, read-only from the graph.

Counts each selected episode's step states in the throwaway graph and writes
``<out>/ama_ingest_progress.json``, which ``archolith-bench dashboard`` renders. It reads
only, so it can run beside an ingest already in progress.

    python scripts/ama/progress_ama.py --dataset .../open_end_qa_set.jsonl \
        --neo4j-uri bolt://127.0.0.1:7688 --out results/ama-subset --watch 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from archolith_bench.dashboard import EPISODE_PROGRESS_FILE
from archolith_bench.harness.ama_bench import (
    CRAFTED_PREFIX,
    RAW_PREFIX,
    crafted_turns,
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

_STATE_COUNTS = (
    "MATCH (e:Episodic) WHERE e.namespace IN $namespaces AND e.processing_state IS NOT NULL "
    "RETURN e.namespace AS ns, e.processing_state AS state, count(*) AS c"
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _crafted_total(crafted_dir: Path, episode: dict) -> int:
    path = crafted_dir / f"ep{episode['episode_id']}.json"
    if not path.exists():
        return 0
    return len(crafted_turns(json.loads(path.read_text(encoding="utf-8"))))


def build_progress(
    episodes: list[dict], counts: dict[str, dict[str, int]], run: str, crafted_dir: Path | None = None,
) -> dict:
    rows = []
    for e in episodes:
        if crafted_dir is not None:
            ns, total = namespace_for(e["episode_id"], CRAFTED_PREFIX), _crafted_total(crafted_dir, e)
        else:
            ns, total = namespace_for(e["episode_id"], RAW_PREFIX), len(render_steps(e))
        states = counts.get(ns, {})
        rows.append({
            "namespace": ns,
            "domain": e["domain"],
            "steps_total": total,
            "ready": states.get("READY", 0),
            "failed": states.get("FAILED", 0),
            "in_flight": sum(v for k, v in states.items() if k not in ("READY", "FAILED")),
        })
    return {
        "run": run,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": rows,
    }


def read_counts(driver, namespaces: list[str]) -> dict[str, dict[str, int]]:  # noqa: ANN001
    counts: dict[str, dict[str, int]] = {}
    for record in driver.execute_query(_STATE_COUNTS, namespaces=namespaces).records:
        counts.setdefault(record["ns"], {})[record["state"]] = int(record["c"])
    return counts


def write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.getenv(DATASET_ENV))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neo4j-uri", default=os.getenv("AMA_NEO4J_URI"))
    ap.add_argument("--neo4j-user", default=os.getenv("AMA_NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password-env", default="AMA_NEO4J_PASSWORD")
    ap.add_argument("--episode-ids", type=lambda v: tuple(int(i) for i in _csv(v)), default=None)
    ap.add_argument("--qa-types", type=_csv, default=("C",))
    ap.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--exclude-domains", type=_csv, default=DEFAULT_EXCLUDE_DOMAINS)
    ap.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds; 0 = once")
    ap.add_argument("--crafted-dir", type=Path, default=None, help="track a crafted-memory ingest")
    args = ap.parse_args(argv)
    prefix = CRAFTED_PREFIX if args.crafted_dir is not None else RAW_PREFIX

    if not args.dataset or not args.neo4j_uri:
        print(f"ERROR: need --dataset (or {DATASET_ENV}) and --neo4j-uri (or AMA_NEO4J_URI)", file=sys.stderr)
        return 2
    assert_not_production(args.neo4j_uri)
    episodes = select_episodes(
        load_episodes(args.dataset),
        qa_types=args.qa_types,
        per_domain=args.per_domain,
        max_tokens=args.max_tokens,
        exclude_domains=args.exclude_domains,
        episode_ids=args.episode_ids,
    )
    namespaces = [namespace_for(e["episode_id"], prefix) for e in episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / EPISODE_PROGRESS_FILE

    from neo4j import GraphDatabase

    with GraphDatabase.driver(
        args.neo4j_uri, auth=(args.neo4j_user, os.getenv(args.neo4j_password_env, ""))
    ) as driver:
        while True:
            payload = build_progress(episodes, read_counts(driver, namespaces), args.out.name, args.crafted_dir)
            write_atomic(target, payload)
            done = sum(r["ready"] + r["failed"] for r in payload["episodes"])
            total = sum(r["steps_total"] for r in payload["episodes"])
            print(f"{payload['generated_at']} {done}/{total} steps", flush=True)
            if args.watch <= 0 or done >= total:
                return 0
            time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
