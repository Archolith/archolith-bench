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
    CRAFTED_PREFIX,
    DATASET_ENV,
    DEFAULT_EXCLUDE_DOMAINS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PER_DOMAIN,
    RAW_PREFIX,
    crafted_turns,
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


_PROMOTE_ENTITIES = (
    "MATCH (n:Entity) WHERE n.group_id = $ns AND coalesce(n.scope, 'SESSION') = 'SESSION' "
    "SET n.scope = 'PERSISTENT' RETURN count(*) AS c"
)
_PROMOTE_EDGES = (
    "MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $ns AND coalesce(r.scope, 'SESSION') = 'SESSION' "
    "SET r.scope = 'PERSISTENT' RETURN count(*) AS c"
)


_UNSETTLED = (
    "MATCH (e:Episodic {namespace: $ns}) WHERE e.processing_state IS NOT NULL "
    "AND NOT e.processing_state IN ['READY', 'FAILED'] RETURN count(e) AS c"
)


def wait_until_settled(count_unsettled, *, poll_s: float = 10.0, timeout_s: float = 6 * 3600,  # noqa: ANN001
                       sleep=time.sleep, clock=time.monotonic) -> bool:
    """Block until the namespace has no episode still pending or enriching (twice in a row).

    Posting returns before enrichment finishes, so promoting at that point left everything
    extracted afterwards SESSION-scoped and invisible to recall.
    """
    deadline, settled = clock() + timeout_s, 0
    while clock() < deadline:
        settled = settled + 1 if count_unsettled() == 0 else 0
        if settled >= 2:
            return True
        sleep(poll_s)
    return False


def promote_namespace(bolt_uri: str, user: str, password: str, namespace: str) -> dict:
    """SESSION -> PERSISTENT for one episode namespace, as LME's promote_persistent.sh does.

    Ingest stamps every memory with the episode's session; REST recall without that session
    admits none of them, so unpromoted episodes recall nothing.
    """
    from neo4j import GraphDatabase

    with GraphDatabase.driver(bolt_uri, auth=(user, password)) as driver:
        entities = driver.execute_query(_PROMOTE_ENTITIES, ns=namespace).records[0]["c"]
        edges = driver.execute_query(_PROMOTE_EDGES, ns=namespace).records[0]["c"]
    return {"entities": entities, "edges": edges}


def load_crafted(crafted_dir: Path, episode: dict) -> list[dict]:
    path = crafted_dir / f"ep{episode['episode_id']}.json"
    if not path.exists():
        raise FileNotFoundError(f"no crafted memories for episode {episode['episode_id']}: {path}")
    crafted = json.loads(path.read_text(encoding="utf-8"))
    if len(crafted) < len(episode.get("trajectory", [])):
        raise ValueError(f"crafted memories for episode {episode['episode_id']} are incomplete")
    return crafted


def ingest_episode(
    base_url: str, episode: dict, *, reset: bool, tries: int = 3, crafted_dir: Path | None = None,
) -> dict:
    if crafted_dir is not None:
        namespace = namespace_for(episode["episode_id"], CRAFTED_PREFIX)
        steps = crafted_turns(load_crafted(crafted_dir, episode))
    else:
        namespace = namespace_for(episode["episode_id"], RAW_PREFIX)
        steps = render_steps(episode)
    if reset:
        _reset_namespace(base_url, namespace)
    failed: list[int] = []
    t0 = time.time()
    with HttpMenhirClient(base_url, timeout=900.0) as client, httpx.Client(timeout=900.0) as http:
        for index, turn in enumerate(steps):
            for attempt in range(tries):
                try:
                    if crafted_dir is not None:
                        # add_memory text as the agent wrote it: no "role: " speaker prefix.
                        http.post(
                            base_url.rstrip("/") + "/api/memory",
                            params={"wait": "true"},
                            json={
                                "episode": turn["content"],
                                "namespace": namespace,
                                "occurred_at": turn["occurred_at"],
                                "session_id": namespace,
                            },
                        ).raise_for_status()
                    else:
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
    ap.add_argument("--neo4j-uri", default=os.getenv("AMA_NEO4J_URI"),
                    help="bolt URI of the throwaway graph; required to promote SESSION memories")
    ap.add_argument("--neo4j-user", default=os.getenv("AMA_NEO4J_USER", "neo4j"))
    ap.add_argument("--neo4j-password-env", default="AMA_NEO4J_PASSWORD")
    ap.add_argument("--promote-only", action="store_true",
                    help="promote the selected namespaces; ingest nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; ingest nothing")
    ap.add_argument("--crafted-dir", type=Path, default=None,
                    help="ingest memory-agent memories from this dir (craft_memories.py) instead of raw steps")
    args = ap.parse_args(argv)
    prefix = CRAFTED_PREFIX if args.crafted_dir is not None else RAW_PREFIX

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
    if not args.neo4j_uri:
        print("ERROR: pass --neo4j-uri (or AMA_NEO4J_URI) so ingested memories can be promoted", file=sys.stderr)
        return 2
    assert_not_production(args.neo4j_uri)
    password = os.getenv(args.neo4j_password_env, "")

    def promote(namespace: str) -> dict:
        return promote_namespace(args.neo4j_uri, args.neo4j_user, password, namespace)

    if args.promote_only:
        for e in episodes:
            ns = namespace_for(e["episode_id"], prefix)
            print(f"  promoted {ns}: {promote(ns)}", flush=True)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def run(episode: dict) -> dict:
        from neo4j import GraphDatabase

        result = ingest_episode(args.menhir_url, episode, reset=args.reset, crafted_dir=args.crafted_dir)
        with GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password)) as driver:
            result["settled"] = wait_until_settled(
                lambda: driver.execute_query(_UNSETTLED, ns=result["namespace"]).records[0]["c"]
            )
        result["promoted"] = promote(result["namespace"])
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, e): e for e in episodes}
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
