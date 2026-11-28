"""Compare sprint v2 run dirs: build/setting mismatches first, then per-arm means.

    python scripts/compare_sprint_runs.py RUN_DIR RUN_DIR [...] [--strict]

--strict exits 3 when the runs served different Menhir builds.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

# Loaded by file: the engineering_evidence package imports Menhir, and comparing reports needs none.
_SPEC = importlib.util.spec_from_file_location(
    "sprint_compare",
    Path(__file__).resolve().parent.parent / "archolith_bench" / "engineering_evidence" / "sprint_compare.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
MENHIR_KEYS, render = _MOD.MENHIR_KEYS, _MOD.render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="sprint v2 run dirs (or their report.json)")
    parser.add_argument("--strict", action="store_true", help="exit 3 if the Menhir builds differ")
    args = parser.parse_args(argv)
    if len(args.run_dirs) < 2:
        parser.error("give at least two runs")
    text, diff = render(args.run_dirs)
    print(text)
    return 3 if args.strict and any(k in MENHIR_KEYS for k in diff) else 0


if __name__ == "__main__":
    raise SystemExit(main())
