#!/usr/bin/env python3
"""CLI entrypoint for the offline evidence policy validator.

Usage:
    python scripts/check_evidence_policy.py
    python scripts/check_evidence_policy.py --headline HEADLINE-NUMBERS.md
    python scripts/check_evidence_policy.py --headline HEADLINE-NUMBERS.md --evidence ev.json
    python scripts/check_evidence_policy.py --headline HEADLINE-NUMBERS.md --evidence-dir benchmarks/
    python scripts/check_evidence_policy.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from archolith_bench.core.evidence_policy import PolicyResult, validate_policy


def _resolve_headline() -> Path:
    """Walk up from cwd or script dir to find HEADLINE-NUMBERS.md."""
    candidates = [
        Path.cwd() / "HEADLINE-NUMBERS.md",
        Path(__file__).resolve().parent.parent / "HEADLINE-NUMBERS.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _detect_evidence_files(directory: Path, *, include_markdown: bool = False) -> list[Path]:
    """Discover evidence artifacts in *directory*.

    Markdown counts as evidence only inside a directory that holds evidence.
    Scanning the repo root for ``*.md`` would sweep in README, BENCHMARKS,
    METHODOLOGY and HEADLINE-NUMBERS itself, so callers opt in per directory.
    validate_policy still screens out README/RUNBOOK inside those directories.
    """
    if not directory.is_dir():
        return []
    paths = list(directory.glob("*.json"))
    if include_markdown:
        paths.extend(directory.glob("*.md"))
    return sorted(paths, key=lambda p: p.name)


def _print_human(result: PolicyResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"Evidence policy: {status}")
    for key, val in result.summary.items():
        print(f"  {key}={val}")

    if result.errors:
        print("\nErrors:")
        for e in result.errors:
            print(f"  - {e.path}: {e.message}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w.path}: {w.message}")


def _print_json(result: PolicyResult) -> None:
    print(json.dumps({
        "ok": result.ok,
        "errors": [
            {"code": e.code, "path": e.path, "message": e.message}
            for e in result.errors
        ],
        "warnings": [
            {"code": w.code, "path": w.path, "message": w.message}
            for w in result.warnings
        ],
        "summary": result.summary,
    }, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark evidence against public-claim policy",
    )
    parser.add_argument("--headline", type=Path, default=None,
                        help="Path to HEADLINE-NUMBERS.md (auto-detected if omitted)")
    parser.add_argument("--evidence", type=Path, action="append", default=None,
                        help="Individual evidence JSON file(s) to validate")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="Directory of evidence JSON files to validate")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON output")

    args = parser.parse_args(argv)

    headline_path = args.headline or _resolve_headline()
    evidence_paths: list[Path] = []

    if args.evidence:
        evidence_paths.extend(args.evidence)
    if args.evidence_dir:
        # An explicitly named directory is an evidence directory by definition.
        evidence_paths.extend(_detect_evidence_files(args.evidence_dir, include_markdown=True))
    if not args.evidence and not args.evidence_dir:
        evidence_paths.extend(_detect_evidence_files(Path.cwd()))
        evidence_paths.extend(_detect_evidence_files(
            headline_path.parent / "benchmarks", include_markdown=True))
        evidence_paths.extend(_detect_evidence_files(headline_path.parent / "results"))

    # Check all input files exist before validating.
    if not headline_path.exists():
        print(f"ERROR: headline file not found: {headline_path}", file=sys.stderr)
        sys.exit(2)

    for ep in evidence_paths[:]:
        if not ep.exists():
            print(f"ERROR: evidence file not found: {ep}", file=sys.stderr)
            sys.exit(2)

    result = validate_policy(headline_path, evidence_paths)

    if args.json:
        _print_json(result)
    else:
        _print_human(result)

    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
