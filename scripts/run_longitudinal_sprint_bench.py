"""Run the 3-arm x 4-session Longitudinal Frozen-Repo Coding-Sprint benchmark.

Defaults to `--dry-run` (zero LLM spend) to validate the fixture, deterministic
`beacon_eval` scoring, and token/redundant-read accounting before any live run.

Usage:
    python scripts/run_longitudinal_sprint_bench.py --dry-run
    python scripts/run_longitudinal_sprint_bench.py --fixture fixtures/sprint_simonw_llm_764dc386.json --dry-run --out results/sprint_dry_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archolith_bench.beacon_eval.runner import DEFAULT_MODEL, DEFAULT_RUN_RESERVE  # noqa: E402
from archolith_bench.engineering_evidence.live_executor import (  # noqa: E402
    LiveSprintConfig,
    OpenCodeSprintExecutor,
)
from archolith_bench.engineering_evidence.sprint import (  # noqa: E402
    SPRINT_ARMS,
    SprintFixture,
    run_longitudinal_sprint,
)

DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json"


def _print_table(report: dict) -> None:
    mode = "DRY-RUN (0 LLM calls)" if report["dry_run"] else "LIVE"
    print(f"\n=== Longitudinal Frozen-Repo Sprint: {report['fixture']} [{mode}] ===")
    print(
        f"Repo: {report['repo']['name']} @ {report['repo']['commit'][:12]} "
        f"({report['repo']['url']})"
    )
    print(
        f"\n{'Arm':22s} {'MeanAcc':>9s} {'TotalTok':>10s} {'S2-S4Tok':>10s} "
        f"{'Tools':>7s} {'RedundRd':>9s} {'MemSaved':>9s}"
    )
    print("-" * 82)
    for arm in SPRINT_ARMS:
        if arm not in report["arms"]:
            continue
        r = report["arms"][arm]
        print(
            f"{arm:22s} {r['mean_accuracy']:>9.4f} {r['total_tokens']:>10d} "
            f"{r['downstream_tokens_s2_to_end']:>10d} {r['total_tool_calls']:>7d} "
            f"{r['total_redundant_file_reads']:>9d} {r['total_memories_saved']:>9d}"
        )
    if report["comparisons_vs_no_menhir"]:
        print("\nDeltas vs no_menhir:")
        for arm, cmp in report["comparisons_vs_no_menhir"].items():
            print(
                f"  {arm:20s}: acc={cmp['accuracy_delta']:+.4f}, "
                f"total_tok={cmp['total_tokens_delta_pct']:+.2f}%, "
                f"s2_s4_tok={cmp['downstream_tokens_delta_pct']:+.2f}%, "
                f"redundant_reads={cmp['redundant_reads_delta']:+d}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the 3-arm x 4-session longitudinal frozen-repo sprint benchmark."
    )
    parser.add_argument(
        "--fixture",
        default=str(DEFAULT_FIXTURE),
        help="Path to sprint fixture JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run with deterministic zero-LLM DryRunExecutor (default when --live is not set).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run live sessions via isolated OpenCode + Menhir MCP (spends LLM tokens).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenCode model id for --live.")
    parser.add_argument(
        "--menhir-mcp-url",
        default="",
        help="Menhir Streamable HTTP MCP URL for menhir_structure/menhir_longitudinal arms.",
    )
    parser.add_argument(
        "--reserve-tokens",
        type=int,
        default=DEFAULT_RUN_RESERVE,
        help="Per-session token reserve kill-switch for --live.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=300.0,
        help="Per-session timeout in seconds for --live.",
    )
    parser.add_argument(
        "--work-dir",
        default="results/longitudinal-sprint-work",
        help="Working directory for isolated checkouts and session logs in --live mode.",
    )
    parser.add_argument(
        "--cache-dir",
        default="results/repo-cache",
        help="Git clone cache directory for pinned frozen-repo exports in --live mode.",
    )
    parser.add_argument(
        "--arms",
        default=",".join(SPRINT_ARMS),
        help="Comma-separated arms to run (subset of: no_menhir,menhir_structure,menhir_longitudinal).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Optional cap on how many sessions to run from the fixture (e.g. 1 for Phase 1 smoke).",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Optional .env file supplying provider API keys for --live.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the JSON sprint report.",
    )
    args = parser.parse_args(argv)

    fixture = SprintFixture.from_file(args.fixture)
    if args.max_sessions is not None:
        from dataclasses import replace

        fixture = replace(fixture, sessions=fixture.sessions[: args.max_sessions])
    selected_arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    if args.live and not args.dry_run:
        env_path = Path(args.env_file) if args.env_file else None
        executor = OpenCodeSprintExecutor(
            LiveSprintConfig(
                work_dir=Path(args.work_dir),
                cache_dir=Path(args.cache_dir),
                model=args.model,
                menhir_mcp_url=args.menhir_mcp_url,
                env_file=env_path,
                reserve_tokens=args.reserve_tokens,
                timeout_s=args.timeout_s,
            )
        )
        report = run_longitudinal_sprint(
            fixture, executor=executor, arms=selected_arms, dry_run=False
        )
    else:
        report = run_longitudinal_sprint(fixture, arms=selected_arms, dry_run=True)
    _print_table(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
