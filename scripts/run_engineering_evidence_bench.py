"""Run the S4 + S5 Engineering Evidence Graph benchmark and print the comparison table.

Zero LLM calls, zero external Neo4j required. Exercises real `menhir` modules across:
  1. Coverage-aware blast_radius / affected_tests refusal (S5)
  2. Stale code-anchor advisory labeling (S1 + S5)
  3. L4 WorkArtifact trust invariants & supersession (S4)
  4. Git-backed corpus reconciliation & fail-closed ambiguity (S4)

Usage:
    python scripts/run_engineering_evidence_bench.py [--out results/engineering_evidence_s4_s5.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archolith_bench.engineering_evidence.runner import (  # noqa: E402
    CONDITIONS,
    run_engineering_evidence_benchmark,
)


def _print_table(report: dict) -> None:
    print("\n=== Menhir S4 + S5 Engineering Evidence Graph Benchmark (zero-LLM) ===")
    print(f"{'Pillar':32s} {'naive_baseline':>16s} {'menhir_s4_s5':>16s} {'delta':>10s}")
    print("-" * 78)
    for pillar in report["pillars"]:
        nb = pillar["conditions"]["naive_baseline"]["composite"]
        mh = pillar["conditions"]["menhir_s4_s5"]["composite"]
        print(f"{pillar['pillar']:32s} {nb:>16.4f} {mh:>16.4f} {mh - nb:>+10.4f}")
    print("-" * 78)
    s = report["summary"]
    print(
        f"{'OVERALL':32s} {s['naive_baseline']:>16.4f} "
        f"{s['menhir_s4_s5']:>16.4f} {report['delta']:>+10.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic S4 + S5 Engineering Evidence Graph benchmark."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON benchmark report.",
    )
    args = parser.parse_args(argv)

    report = run_engineering_evidence_benchmark()
    _print_table(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
