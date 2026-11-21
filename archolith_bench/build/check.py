"""build --check: one build's progress and acceptance (replaces status.sh, expected.py, lme.sh validate).

Progress comes from the manifest (one row per finished item, counts Menhir reported at drain) and
the ingest state log (namespaces started). Live state comes only from a Menhir that already serves
the graph (the build's own, mid-build): ``get_queue_depth`` and ``fetch_namespace_build_counts``.
Check never starts Menhir and never writes the graph. With ``expected_items`` it also runs the
acceptance validator (validate_run.py) and returns its verdict.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB = REPO_ROOT / "scripts" / "longmemeval" / "lib"
CANONICAL_CONTAINER = "menhir-lme-neo4j"

Call = Callable[..., Any]


def check_paths(config: dict[str, dict[str, Any]], environ: dict[str, str], results_dir: str | None) -> dict[str, Path]:
    """config.sh's results/manifest derivation for this build's container."""
    name = str(config["container"]["neo4j_name"])
    results = Path(results_dir or environ.get("LME_RESULTS_DIR") or REPO_ROOT / "results" / "lme-ingest")
    default_manifest = "manifest.json" if name == CANONICAL_CONTAINER else f"manifest-{name}.json"
    manifest = Path(environ.get("LME_MANIFEST_PATH") or results / default_manifest)
    return {
        "results": results,
        "manifest": manifest,
        "state_log": manifest.parent / f"ingest-state-{manifest.stem}.jsonl",
        "provenance": results / f"graph-provenance-{name}.json",
        "telemetry": results / "mcp_telemetry.db",
        "report": results / "acceptance-report.json",
        "progress_log": results / "progress.log",
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) else []


def started_namespaces(state_log: Path) -> list[str]:
    """Namespaces the ingest state log has a plan for (read-only)."""
    if not state_log.is_file():
        return []
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    from ingest_state import IngestStateLog

    return IngestStateLog(state_log).namespaces()


def menhir_call(client: httpx.Client, url: str) -> Call | None:
    """An op caller when a Menhir already serves ``url``; None otherwise (check never serves one)."""
    try:
        if client.get(f"{url.rstrip('/')}/api/health").status_code != 200:
            return None
    except httpx.HTTPError:
        return None

    def call(op: str, body: dict | None = None) -> Any:
        resp = client.post(f"{url.rstrip('/')}/api/internal/backend/{op}", json=body or {})
        resp.raise_for_status()
        return resp.json()

    return call


def snapshot(rows: list[dict[str, Any]], started: list[str], limit: int, call: Call | None,
             stamp: str) -> list[str]:
    """Summary line first, then one line per namespace."""
    done = {str(r.get("namespace")): r for r in rows}
    turns = sum(int(r.get("turns") or 0) for r in rows)
    failed = sum(int(r.get("failed_remaining") or 0) for r in rows)
    summary = (f"[{stamp}] items_done={len(rows)}/{limit} started={len(set(started) | set(done))} "
               f"turns_done={turns} failed_remaining={failed}")
    lines: list[str] = []
    if call is None:
        lines.append(summary + " | live: no Menhir serving (manifest counts only)")
        for ns, r in done.items():
            lines.append(f"    {ns:<24} done episodes={r.get('episodes')} ready={r.get('ready')} "
                         f"failed={r.get('failed_remaining')}")
        return lines
    totals = {k: 0 for k in ("pending", "enriching", "ready", "failed")}
    detail = []
    for ns in list(dict.fromkeys([*done, *started])):
        state = call("fetch_namespace_build_counts", {"namespace": ns, "sections": ["state"]})["state"]
        for key in totals:
            totals[key] += int(state.get(key, 0))
        mark = "done" if ns in done else "in-flight"
        planned = done[ns].get("turns") if ns in done else None
        of = f"/{planned}" if planned else ""
        detail.append(f"    {ns:<24} {mark:<9} episodes={state.get('total', 0)}{of} ready={state.get('ready', 0)} "
                      f"pending={state.get('pending', 0)} enriching={state.get('enriching', 0)} "
                      f"failed={state.get('failed', 0)}")
    live = " ".join(f"{k}={v}" for k, v in totals.items())
    lines.append(f"{summary} | live: queue={int(call('get_queue_depth'))} {live}")
    return lines + detail


def validate_args(paths: dict[str, Path], expected_items: int) -> list[str]:
    """lme.sh validate's canonical acceptance arguments."""
    return [
        str(LIB / "validate_run.py"), str(paths["provenance"]), str(paths["manifest"]),
        "--telemetry-db", str(paths["telemetry"]), "--require-fresh-clean",
        "--expected-items", str(expected_items), "--output", str(paths["report"]),
    ]


def run_check(
    paths: dict[str, Path],
    *,
    limit: int,
    menhir_url: str,
    expected_items: int | None,
    watch: int | None = None,
    python: str = sys.executable,
    sleep: Callable[[float], None] = time.sleep,
    out: Callable[[str], None] = lambda line: print(line, flush=True),  # before the validator's output
    max_rounds: int | None = None,
) -> int:
    with httpx.Client(timeout=60.0) as client:
        rounds = 0
        while True:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            lines = snapshot(read_manifest(paths["manifest"]), started_namespaces(paths["state_log"]), limit,
                             menhir_call(client, menhir_url), stamp)
            for line in lines:
                out(line)
            rounds += 1
            if watch is None:
                break
            paths["progress_log"].parent.mkdir(parents=True, exist_ok=True)
            with paths["progress_log"].open("a", encoding="utf-8") as log:
                log.write(lines[0] + "\n")
            if max_rounds is not None and rounds >= max_rounds:
                return 0
            sleep(watch)
    if expected_items is None:
        return 0
    return subprocess.run([python, *validate_args(paths, expected_items)]).returncode
