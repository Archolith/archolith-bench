"""Re-enrich FAILED Episodic nodes in the LongMemEval oracle Neo4j.

Run with menhir-frontier already serving on --menhir-url (started by
retry.sh). Queries Neo4j for all FAILED episodes, calls
force_reset_failed_episode for each, then drains globally.

Usage:
    python lib/retry.py [--menhir-url http://localhost:8102]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

import httpx

NEO4J_CONTAINER = os.getenv("LME_NEO4J_CONTAINER", "menhir-lme-neo4j")
NEO4J_PW = os.getenv("LME_NEO4J_PW", "lmedata123")


def _cypher(query: str, timeout: int = 300) -> list[list[str]]:
    try:
        proc = subprocess.run(
            ["docker", "exec", NEO4J_CONTAINER, "cypher-shell",
             "-u", "neo4j", "-p", NEO4J_PW, "--format", "plain", query],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return []
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        rows = []
        for ln in lines[1:]:  # drop header
            rows.append([c.strip().strip('"') for c in ln.split(", ")])
        return rows
    except Exception:
        return []


def _ns_prefix() -> str:
    # Spliced into Cypher text the same way ingest.py does it, so refuse quoting characters.
    prefix = os.getenv("LME_NS_PREFIX", "lme-")
    if "'" in prefix or "\\" in prefix:
        raise SystemExit("LME_NS_PREFIX must not contain a quote or backslash")
    return prefix


def all_failed_uuids(limit: int = 25_000) -> list[str]:
    rows = _cypher(
        f"MATCH (e:Episodic) WHERE e.namespace STARTS WITH '{_ns_prefix()}' "
        f"AND e.processing_state='FAILED' "
        f"RETURN e.uuid AS uuid LIMIT {limit};"
    )
    return [r[0] for r in rows if r and r[0]]


def global_counts() -> dict[str, int]:
    rows = _cypher(
        f"MATCH (e:Episodic) WHERE e.namespace STARTS WITH '{_ns_prefix()}' RETURN "
        "sum(CASE WHEN e.processing_state='READY'     THEN 1 ELSE 0 END) AS ready, "
        "sum(CASE WHEN e.processing_state='ENRICHING' THEN 1 ELSE 0 END) AS enriching, "
        "sum(CASE WHEN e.processing_state='FAILED'    THEN 1 ELSE 0 END) AS failed, "
        "count(*) AS total;"
    )
    if not rows or len(rows[0]) < 4:
        return {"ready": -1, "enriching": -1, "failed": -1, "total": -1}
    r = rows[0]
    return {
        "ready":     int(r[0] or 0),
        "enriching": int(r[1] or 0),
        "failed":    int(r[2] or 0),
        "total":     int(r[3] or 0),
    }


def queue_depth(admin: httpx.Client, menhir_url: str) -> int:
    try:
        resp = admin.get(menhir_url.rstrip("/") + "/api/stats")
        resp.raise_for_status()
        return int(resp.json().get("queue_depth", -1))
    except Exception:
        return -1


def backend(admin: httpx.Client, menhir_url: str, op: str, body: dict | None = None):
    url = menhir_url.rstrip("/") + "/api/internal/backend/" + op
    resp = admin.post(url, json=(body or {}))
    resp.raise_for_status()
    return resp.json()


def drain(
    admin: httpx.Client,
    menhir_url: str,
    *,
    idle_polls: int = 3,
    poll_s: float = 10.0,
    timeout_s: float = 86_400.0,
) -> dict:
    """Block until queue_depth==0 and no ENRICHING episodes, stable for idle_polls checks."""
    t0 = time.time()
    settled = 0
    last: dict = {}
    while time.time() - t0 < timeout_s:
        qd = queue_depth(admin, menhir_url)
        counts = global_counts()
        enriching = counts["enriching"] if counts["enriching"] >= 0 else 1
        elapsed = time.time() - t0
        print(
            f"  [{elapsed/3600:.1f}h] queue={qd} ready={counts['ready']} "
            f"enriching={enriching} failed={counts['failed']} total={counts['total']}",
            flush=True,
        )
        last = dict(counts)
        if qd == 0 and enriching == 0:
            settled += 1
            if settled >= idle_polls:
                last["timed_out"] = False
                return last
        else:
            settled = 0
        time.sleep(poll_s)
    last["timed_out"] = True
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Retry FAILED episodes in the LME oracle Neo4j.")
    ap.add_argument("--menhir-url", default="http://localhost:8102")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="UUIDs to reset per batch before a brief pause")
    ap.add_argument("--drain-timeout", type=float, default=86_400.0,
                    help="max seconds to wait for enrichment to settle (default 24h)")
    args = ap.parse_args()

    admin = httpx.Client(timeout=60.0)

    print("querying FAILED episodes from Neo4j...", flush=True)
    uuids = all_failed_uuids()
    if not uuids:
        print("no FAILED episodes found — nothing to do")
        counts = global_counts()
        print(f"current state: {counts}")
        return 0

    print(f"found {len(uuids)} FAILED episodes; resetting to QUEUED...", flush=True)
    requeued = 0
    errors = 0
    t_reset = time.time()
    for i, u in enumerate(uuids):
        try:
            backend(admin, args.menhir_url, "force_reset_failed_episode", {"episode_uuid": u})
            requeued += 1
        except Exception as exc:
            errors += 1
            if errors <= 5:  # only log the first few to avoid spam
                print(f"  reset error uuid={u}: {exc}", flush=True)
        if (i + 1) % args.batch_size == 0:
            pct = 100 * (i + 1) / len(uuids)
            print(f"  reset {i+1}/{len(uuids)} ({pct:.0f}%) errors={errors}", flush=True)
            time.sleep(0.5)

    print(
        f"reset pass done in {time.time()-t_reset:.0f}s: "
        f"{requeued} requeued, {errors} errors",
        flush=True,
    )
    print(f"draining (timeout {args.drain_timeout/3600:.0f}h)...", flush=True)
    final = drain(admin, args.menhir_url, timeout_s=args.drain_timeout)

    print(
        f"\nfinal state: ready={final.get('ready')} enriching={final.get('enriching')} "
        f"failed={final.get('failed')} total={final.get('total')}",
        flush=True,
    )
    if final.get("timed_out"):
        print("WARNING: drain timed out — some episodes may still be ENRICHING/FAILED", flush=True)
        return 1
    residual = final.get("failed", 0)
    if residual and residual > 0:
        print(f"WARNING: {residual} episodes still FAILED after retry", flush=True)
        return 1
    print("done — all episodes enriched.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
