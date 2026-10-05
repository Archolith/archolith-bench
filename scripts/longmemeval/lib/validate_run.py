"""Final acceptance report for a LongMemEval buildout.

Reads the graph provenance file, manifest, and optional telemetry DB, and emits a
machine-readable JSON report covering the contract from the scalar-history plan:

- manifest cardinality (expected vs actual items)
- failed episodes (zero-tolerance policy)
- projection counts (scalar_state, scalar_history Views)
- source-time integrity (valid_at present and plausible)
- provenance-chain completeness (TurnEvidence, assertions, FOUNDS, ADMITTED_ON)
- namespace isolation (no cross-namespace leakage)
- commit immutability (all attempts ran the same code)
- telemetry presence (vote receipt DB exists and has rows)

Usage::

    validate_run.py <provenance.json> <manifest.json> [--telemetry-db <path>]
                    [--expected-items N] [--require-fresh-clean] [--output <report.json>]

``--require-fresh-clean`` (canonical acceptance) requires ``--expected-items``.

Exit 0 when all checks pass; exit 1 with the report on any failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check(
    name: str,
    passed: bool,
    detail: str,
    *,
    severity: str = "FAIL",
) -> dict[str, Any]:
    return {
        "check": name,
        "status": "PASS" if passed else severity,
        "detail": detail,
    }


def consolidation_errors(
    result: dict[str, Any], requested: dict[str, bool], *, namespace: str, k: int,
) -> list[str]:
    """Check lane completion, including untouched or partly processed event work."""
    errors = []
    if result.get("namespace") != namespace:
        errors.append(f"{namespace}: missing or mismatched consolidation namespace")
    if requested.get("scalar"):
        if result.get("scalar_enabled") is not True:
            errors.append(f"{namespace}: scalar consolidation is disabled")
        if result.get("scalar_namespaces_processed") != 1 or int(result.get("scalar_llm_calls", 0)) < k:
            errors.append(f"{namespace}: scalar consolidation did not finish its perception samples")
    if requested.get("counter"):
        if result.get("counter_enabled") is not True:
            errors.append(f"{namespace}: counter consolidation is disabled")
        if result.get("namespaces_processed") != 1 or result.get("dirty_after") is not False:
            errors.append(f"{namespace}: counter consolidation did not finish")
    if requested.get("event"):
        if result.get("event_history_enabled") is not True:
            errors.append(f"{namespace}: event history is disabled")
        if result.get("event_namespaces_failed") != 0:
            errors.append(f"{namespace}: event namespace failed or failure count is missing")
        if result.get("event_dirty_after") is not False:
            errors.append(f"{namespace}: event evidence remains pending or completion is unknown")
    return errors


def validate(
    provenance_path: Path,
    manifest_path: Path,
    *,
    telemetry_db: Path | None = None,
    expected_items: int | None = None,
    require_fresh_clean: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # ---- Provenance file ----
    if not provenance_path.exists():
        checks.append(_check(
            "provenance_exists", False, f"provenance file not found: {provenance_path}",
        ))
        return _report(checks)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    # ---- Commit immutability ----
    attempts = provenance.get("attempts") or []
    commit_errors = []
    for key in ("menhir_commit", "bench_commit"):
        expected = provenance.get(key)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{7,40}", expected):
            commit_errors.append(f"{key} missing or invalid")
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict) or attempt.get(key) != expected:
                commit_errors.append(f"attempt {index + 1} {key} differs")
    if not attempts:
        commit_errors.append("no attempts recorded")
    checks.append(_check(
        "commit_immutability", not commit_errors,
        "; ".join(commit_errors) if commit_errors else
        f"{len(attempts)} attempt(s) match both recorded commits",
    ))

    if require_fresh_clean:
        phase_settings = [p.get("effective_settings") for p in provenance.get("phases", [])
                          if isinstance(p, dict) and p.get("phase") == "ingest-graph"]
        clean = bool(phase_settings) and all(
            isinstance(settings, dict)
            and settings.get("menhir_dirty") is False
            and settings.get("bench_dirty") is False
            and settings.get("menhir_untracked") == 0
            and settings.get("bench_untracked") == 0
            for settings in phase_settings
        )
        fresh = (provenance.get("graph_fresh") is True
                 and provenance.get("volume_pre_existed") is False
                 and provenance.get("require_fresh") == 1)
        digest = provenance.get("surface_digest")
        fingerprinted = isinstance(digest, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", digest))
        checks.append(_check(
            "fresh_clean_provenance",
            fresh and clean and fingerprinted and provenance.get("noncanonical") is not True,
            f"fresh={fresh}, clean={clean}, surface_fingerprinted={fingerprinted}, "
            f"canonical={provenance.get('noncanonical') is not True}",
        ))

    # Noncanonical label
    checks.append(_check(
        "canonical_label",
        not provenance.get("noncanonical", False),
        "noncanonical" if provenance.get("noncanonical") else "canonical",
        severity="WARN",
    ))

    # ---- Manifest cardinality ----
    if not manifest_path.exists():
        checks.append(_check(
            "manifest_exists", False, f"manifest not found: {manifest_path}",
        ))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            checks.append(_check(
                "manifest_format", False, "manifest is not a JSON list",
            ))
        else:
            actual = len(manifest)
            if expected_items is not None:
                checks.append(_check(
                    "manifest_cardinality",
                    actual == expected_items,
                    f"expected {expected_items}, got {actual}",
                ))
            elif require_fresh_clean:
                # Any count > 0 would pass otherwise, so a short canonical run could be accepted.
                checks.append(_check(
                    "manifest_cardinality", False,
                    f"{actual} items, but canonical (fresh-clean) validation requires an "
                    "expected item count",
                ))
            else:
                checks.append(_check(
                    "manifest_cardinality", actual > 0,
                    f"{actual} items (no expected count specified)",
                ))

            # Requested consolidation lanes
            lane_errors = []
            declared_lanes = {
                "scalar": bool(provenance.get("scalar_state_enabled", False)),
                "counter": bool(provenance.get("counter_state_enabled", False)),
                "event": bool(provenance.get("event_history_enabled", False)),
            }
            for row in manifest:
                if not isinstance(row, dict):
                    lane_errors.append("manifest contains a non-object row")
                    continue
                requested = row.get("consolidation_requested", declared_lanes)
                if any(declared_lanes.values()) and requested != declared_lanes:
                    lane_errors.append(f"{row.get('namespace')}: requested lanes differ from provenance")
                if any(requested.values()):
                    lane_errors.extend(consolidation_errors(
                        row.get("consolidation_result", {}), requested,
                        namespace=str(row.get("namespace", "")),
                        k=int(row.get("consolidation_k", 3)),
                    ))
            checks.append(_check(
                "consolidation_lanes", not lane_errors,
                "; ".join(lane_errors[:5]) if lane_errors else "all requested lanes completed",
            ))

            # Ingest writes per-namespace counts, not an episode-level status.
            # Missing or malformed counts are unknown, never evidence of success.
            failed = []
            unknown = []
            timed_out = []
            for index, row in enumerate(manifest):
                if not isinstance(row, dict):
                    unknown.append(index)
                    continue
                count = row.get("failed_remaining")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    unknown.append(index)
                elif count > 0:
                    failed.append((index, count))
                if row.get("drain_timed_out") is True:
                    timed_out.append(index)
            checks.append(_check(
                "zero_failed_episodes",
                not failed and not unknown and not timed_out,
                f"failed={failed[:5]}, unknown_counts={unknown[:5]}, "
                f"drain_timed_out={timed_out[:5]}" if failed or unknown or timed_out
                else f"all {actual} manifest items have zero failed episodes",
            ))

            # Namespace isolation: every namespace starts with the configured prefix
            ns_prefix = provenance.get("namespace_prefix", "lme-")
            bad_ns = [
                str(row.get("namespace", ""))
                for row in manifest if isinstance(row, dict)
                and not str(row.get("namespace", "")).startswith(ns_prefix)
            ]
            checks.append(_check(
                "namespace_isolation",
                len(bad_ns) == 0,
                f"{len(bad_ns)} namespace(s) outside prefix '{ns_prefix}': {bad_ns[:5]}"
                if bad_ns else f"all namespaces start with '{ns_prefix}'",
            ))

            # Projection counts
            total_assertions = sum(
                int(row.get("typed_assertions", 0))
                for row in manifest if isinstance(row, dict)
            )
            total_views = sum(
                int(row.get("scalar_views", 0))
                for row in manifest if isinstance(row, dict)
            )
            total_history_views = sum(
                int(row.get("scalar_history_views", 0))
                for row in manifest if isinstance(row, dict)
            )
            checks.append(_check(
                "projection_counts",
                True,
                f"{total_assertions} assertions, {total_views} scalar_state views, "
                f"{total_history_views} scalar_history views",
                severity="INFO",
            ))

    # ---- Telemetry presence ----
    if telemetry_db is None:
        checks.append(_check(
            "telemetry_presence", False,
            "no telemetry DB path specified",
            severity="WARN",
        ))
    elif not telemetry_db.exists():
        checks.append(_check(
            "telemetry_presence", False,
            f"telemetry DB not found: {telemetry_db}",
        ))
    else:
        try:
            uri = f"{telemetry_db.resolve().as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=2.0) as conn:
                row_count = conn.execute(
                    "SELECT count(*) FROM lifecycle_events"
                ).fetchone()[0]
            checks.append(_check(
                "telemetry_presence",
                row_count > 0,
                f"{row_count} lifecycle events" if row_count
                else "telemetry DB exists but has no events",
            ))
        except (sqlite3.Error, OSError) as exc:
            checks.append(_check(
                "telemetry_presence", False,
                f"could not read telemetry DB: {exc}",
            ))

    # ---- Source-time integrity (from provenance phases) ----
    phases = provenance.get("phases") or []
    unfinished = [p for p in phases if not isinstance(p, dict)
                  or p.get("status") != "completed"]
    checks.append(_check(
        "no_interrupted_phases",
        bool(phases) and not unfinished,
        f"{len(unfinished)} non-completed phase(s)" if unfinished
        else f"all {len(phases)} phases completed" if phases
        else "no phases recorded",
    ))

    return _report(checks)


def _report(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [c for c in checks if c["status"] == "FAIL"]
    warnings = [c for c in checks if c["status"] == "WARN"]
    return {
        "validated_at": _now(),
        "verdict": "PASS" if not failures else "FAIL",
        "failures": len(failures),
        "warnings": len(warnings),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance", type=Path, help="graph-provenance-*.json")
    parser.add_argument("manifest", type=Path, help="manifest.json")
    parser.add_argument("--telemetry-db", type=Path, default=None)
    parser.add_argument("--expected-items", type=int, default=None)
    parser.add_argument("--require-fresh-clean", action="store_true")
    parser.add_argument("--output", type=Path, default=None,
                        help="write report JSON here (also printed to stdout)")
    args = parser.parse_args(argv)
    if args.require_fresh_clean and args.expected_items is None:
        parser.error("--require-fresh-clean requires --expected-items N")

    report = validate(
        args.provenance,
        args.manifest,
        telemetry_db=args.telemetry_db,
        expected_items=args.expected_items,
        require_fresh_clean=args.require_fresh_clean,
    )

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
