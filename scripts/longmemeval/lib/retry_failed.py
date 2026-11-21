"""Re-enrich one build's FAILED episodes through Menhir backend ops (replaces retry.py).

Usage: retry_failed.py <manifest.json> <menhir_url> [--drain-timeout SECONDS]

Run by the build pipeline's retry mode while the build's Menhir serves. For each manifest
namespace it lists FAILED episodes (``list_episode_processing``) and resets each once
(``force_reset_failed_episode``), then drains: the queue is empty and the reset namespaces hold
nothing PENDING or ENRICHING, three polls in a row. Exits non-zero on a drain timeout or when any
manifest namespace still holds a FAILED episode.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from promote import manifest_namespaces

Call = Callable[..., Any]

# list_episode_processing caps a page at 200; a namespace is re-listed until nothing new appears.
_PAGE = 200


def http_call(client: httpx.Client, menhir_url: str) -> Call:
    def call(op: str, body: dict | None = None) -> Any:
        resp = client.post(f"{menhir_url.rstrip('/')}/api/internal/backend/{op}", json=body or {})
        resp.raise_for_status()
        return resp.json()

    return call


def failed_uuids(call: Call, namespace: str) -> list[str]:
    rows = call("list_episode_processing", {"states": ["FAILED"], "limit": _PAGE, "namespace": namespace})
    return [str(row["uuid"]) for row in rows or [] if row.get("uuid")]


def reset_namespace(call: Call, namespace: str) -> tuple[int, int]:
    """Reset every FAILED episode in ``namespace`` once; returns (reset, refused).

    An episode is reset at most once per pass: one that fails again while the pass runs stays
    FAILED for the residual check instead of looping.
    """
    tried: set[str] = set()
    reset = refused = 0
    while True:
        fresh = [u for u in failed_uuids(call, namespace) if u not in tried]
        if not fresh:
            return reset, refused
        for uuid in fresh:
            tried.add(uuid)
            if call("force_reset_failed_episode", {"episode_uuid": uuid}):
                reset += 1
            else:
                refused += 1


def state(call: Call, namespace: str) -> dict[str, int]:
    return call("fetch_namespace_build_counts", {"namespace": namespace, "sections": ["state"]})["state"]


def drain(
    call: Call,
    namespaces: list[str],
    *,
    timeout_s: float,
    poll_s: float = 10.0,
    idle_polls: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """True once the queue is empty and ``namespaces`` hold nothing in flight ``idle_polls`` times."""
    start = clock()
    settled = 0
    while clock() - start < timeout_s:
        depth = int(call("get_queue_depth"))
        totals = {"pending": 0, "enriching": 0, "failed": 0, "ready": 0}
        for namespace in namespaces:
            counts = state(call, namespace)
            for key in totals:
                totals[key] += int(counts.get(key, 0))
        print(f"  [{(clock() - start) / 3600:.1f}h] queue={depth} " +
              " ".join(f"{k}={v}" for k, v in totals.items()), flush=True)
        if depth == 0 and totals["pending"] == 0 and totals["enriching"] == 0:
            settled += 1
            if settled >= idle_polls:
                return True
        else:
            settled = 0
        sleep(poll_s)
    return False


def retry(call: Call, namespaces: list[str], *, timeout_s: float, **drain_kwargs: Any) -> int:
    touched: list[str] = []
    reset_total = refused_total = 0
    for namespace in namespaces:
        reset, refused = reset_namespace(call, namespace)
        if reset or refused:
            touched.append(namespace)
            print(f"  {namespace}: reset={reset} refused={refused}", flush=True)
        reset_total += reset
        refused_total += refused
    print(f"[lme-retry] {len(namespaces)} namespaces: reset={reset_total} refused={refused_total}", flush=True)
    if reset_total and not drain(call, touched, timeout_s=timeout_s, **drain_kwargs):
        print("[lme-retry] ERROR: drain timed out; episodes may still be PENDING/ENRICHING", flush=True)
        return 1
    residual = {ns: n for ns in namespaces if (n := int(state(call, ns).get("failed", 0)))}
    if residual:
        print(f"[lme-retry] ERROR: {sum(residual.values())} episodes still FAILED in "
              f"{len(residual)} namespaces: {sorted(residual)[:10]}", flush=True)
        return 1
    print("[lme-retry] done: no FAILED episodes remain in this build's namespaces.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("menhir_url")
    ap.add_argument("--drain-timeout", type=float, default=86_400.0)
    args = ap.parse_args(argv)
    namespaces = manifest_namespaces(args.manifest)
    with httpx.Client(timeout=120.0) as client:
        return retry(http_call(client, args.menhir_url), namespaces, timeout_s=args.drain_timeout)


if __name__ == "__main__":
    sys.exit(main())
