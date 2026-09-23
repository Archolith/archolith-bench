"""``archolith-bench beacon-eval plan|run``."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from archolith_bench.beacon_eval import CONDITIONS
from archolith_bench.beacon_eval.models import load_repos, load_tasks
from archolith_bench.beacon_eval.report import render
from archolith_bench.beacon_eval.runner import (
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RUN_RESERVE,
    RunnerConfig,
    run_matrix,
)

HERE = Path(__file__).parent


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "beacon-eval", help="Beacon agent-task evaluation (A: docs, B: +Beacon MCP, C: +pasted)"
    )
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--repos", default="", help="Comma-separated repo names (default: all)")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids (default: all)")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET_TOKENS)
    parser.add_argument(
        "--run-reserve-tokens",
        type=int,
        default=DEFAULT_RUN_RESERVE,
        help="Tokens set aside per run; a run past it is killed and the matrix stops",
    )
    parser.add_argument("--workdir", default="results/beacon-eval")
    parser.add_argument("--beacon-python", default=sys.executable)
    parser.add_argument("--beacon-src", default=None, help="PYTHONPATH for a Beacon source tree")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Also run tasks whose gold answer the owner has not approved (dry runs only)",
    )
    parser.add_argument("--report", default="", help="Markdown report path")


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def run(args: argparse.Namespace) -> int:
    pins = load_repos(HERE / "repos.json")
    tasks = load_tasks(HERE / "tasks", _split(args.repos), _split(args.tasks))
    if not args.include_unreviewed:
        tasks = [task for task in tasks if task.reviewed]
    conditions = _split(args.conditions)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        print(f"unknown conditions: {unknown}", file=sys.stderr)
        return 2
    runs = len(tasks) * len(conditions) * args.repeats
    print(f"{len(tasks)} task(s) x {len(conditions)} condition(s) x {args.repeats} repeat(s) = {runs} run(s)")
    for task in tasks:
        print(f"  {task.repo}/{task.task_id} ({task.kind}){'' if task.reviewed else ' [unreviewed]'}")
    if args.action == "plan" or not tasks:
        return 0
    config = RunnerConfig(
        workdir=Path(args.workdir),
        beacon_python=args.beacon_python,
        beacon_src=args.beacon_src,
        model=args.model,
        budget_tokens=args.budget_tokens,
        run_reserve_tokens=args.run_reserve_tokens,
    )
    results, stopped = run_matrix(config, pins, tasks, conditions, args.repeats)
    report = render(
        results,
        {
            "Date": date.today().isoformat(),
            "Model": args.model,
            "Conditions": ", ".join(conditions),
            "Repeats": str(args.repeats),
            "Budget": f"{args.budget_tokens:,} tokens ({args.run_reserve_tokens:,} reserved per run)",
            "Pins": "; ".join(f"{name} {pin.commit[:10]}" for name, pin in pins.items()),
        },
        stopped,
    )
    target = Path(args.report) if args.report else config.workdir / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"report: {target}" + (f" (stopped: {stopped})" if stopped else ""))
    return 1 if stopped else 0
