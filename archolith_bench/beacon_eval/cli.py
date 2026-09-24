"""``archolith-bench beacon-eval plan|run|rescore|judge``."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from archolith_bench.beacon_eval import CONDITIONS, DEFAULT_CONDITIONS
from archolith_bench.beacon_eval.isolation import load_api_keys
from archolith_bench.beacon_eval.judge import (
    DEFAULT_JUDGE_MODEL,
    JudgeBudgetExhausted,
    JudgeRateLimited,
    judge_workdir,
    openai_call,
)
from archolith_bench.beacon_eval.models import RunResult, load_repos, load_tasks
from archolith_bench.beacon_eval.report import METRICS, render
from archolith_bench.beacon_eval.runner import (
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RUN_RESERVE,
    RunnerConfig,
    rescore,
    run_matrix,
)

HERE = Path(__file__).parent


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "beacon-eval", help="Beacon agent-task evaluation (A: docs, B: +Beacon MCP, C: +pasted)"
    )
    parser.add_argument("action", choices=("plan", "run", "rescore", "judge"))
    parser.add_argument("--repos", default="", help="Comma-separated repo names (default: all)")
    parser.add_argument("--tasks", default="", help="Comma-separated task ids (default: all)")
    parser.add_argument(
        "--task-set", choices=("main", "why"), default="main",
        help="main: the 20 orientation tasks (tasks/); why: memory-only \"why\" tasks (why_tasks/)",
    )
    parser.add_argument(
        "--conditions", default=",".join(DEFAULT_CONDITIONS),
        help="A, B, C by default; D (Beacon MCP only, built-in tools off) is opt-in",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--budget-tokens", type=int, default=None,
        help=f"Token cap (default {DEFAULT_BUDGET_TOKENS:,}; none when --budget-usd is set)",
    )
    parser.add_argument(
        "--run-reserve-tokens", type=int, default=None,
        help=f"Tokens set aside per run; a run past it is killed and the matrix stops "
        f"(default {DEFAULT_RUN_RESERVE:,}; none when --budget-usd is set)",
    )
    parser.add_argument(
        "--budget-usd", type=float, default=None,
        help="Dollar cap from OpenCode's per-step cost; runs with no cost data stop the matrix",
    )
    parser.add_argument(
        "--run-reserve-usd", type=float, default=0.10,
        help="Dollars set aside per run under --budget-usd; a run past it is killed",
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
    parser.add_argument(
        "--env-file",
        default="",
        help=".env whose *_API_KEY values are passed to OpenCode only (for built-in providers)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="run: reuse saved runs that have an answer and no error (after a killed matrix)",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help="judge: OpenAI model grading each gold point of saved answers (key from --env-file)",
    )
    parser.add_argument(
        "--judge-budget-usd", type=float, default=0.10,
        help="judge: dollar cap for judge calls (cached verdicts cost nothing)",
    )
    parser.add_argument(
        "--memory-url", default="",
        help="Condition M: Menhir remote MCP URL (e.g. http://127.0.0.1:8795/mcp-http)",
    )
    parser.add_argument(
        "--memory-key-file", default="",
        help="Condition M: file holding the Menhir key (read-only tier); passed to OpenCode only",
    )


def _split(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def run(args: argparse.Namespace) -> int:
    pins = load_repos(HERE / "repos.json")
    task_root = HERE / ("why_tasks" if args.task_set == "why" else "tasks")
    tasks = load_tasks(task_root, _split(args.repos), _split(args.tasks))
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
    if args.action == "rescore":
        rescored = rescore(Path(args.workdir), tasks)
        target = Path(args.report) if args.report else Path(args.workdir) / "report.md"
        target.write_text(
            render(rescored, {"Date": date.today().isoformat(), "Rescored": "from saved answers"}),
            encoding="utf-8",
        )
        print(f"rescored {len(rescored)} run(s); report: {target}")
        return 0
    if args.action == "judge":
        return _judge(args, task_root)
    dollars = args.budget_usd is not None
    # In dollar mode token limits are off (None) unless given; placeholders would trip the check.
    budget_tokens = args.budget_tokens or (None if dollars else DEFAULT_BUDGET_TOKENS)
    reserve_tokens = args.run_reserve_tokens or (None if dollars else DEFAULT_RUN_RESERVE)
    config = RunnerConfig(
        workdir=Path(args.workdir),
        beacon_python=args.beacon_python,
        beacon_src=args.beacon_src,
        model=args.model,
        budget_tokens=budget_tokens,
        run_reserve_tokens=reserve_tokens,
        budget_usd=args.budget_usd,
        run_reserve_usd=args.run_reserve_usd,
        env_file=Path(args.env_file) if args.env_file else None,
        memory_url=args.memory_url or None,
        memory_key=(
            Path(args.memory_key_file).read_text(encoding="utf-8").strip()
            if args.memory_key_file
            else None
        ),
    )
    if "M" in conditions and not (config.memory_url and config.memory_key):
        print("condition M needs --memory-url and --memory-key-file", file=sys.stderr)
        return 2
    results, stopped = run_matrix(config, pins, tasks, conditions, args.repeats, resume=args.resume)
    report = render(
        results,
        {
            "Date": date.today().isoformat(),
            "Model": args.model,
            "Conditions": ", ".join(conditions),
            "Repeats": str(args.repeats),
            "Budget": (
                f"${args.budget_usd:.2f} (${args.run_reserve_usd:.2f} reserved per run)"
                if dollars
                else f"{budget_tokens or 0:,} tokens ({reserve_tokens or 0:,} reserved per run)"
            ),
            "Pins": "; ".join(f"{name} {pin.commit[:10]}" for name, pin in pins.items()),
        },
        stopped,
    )
    target = Path(args.report) if args.report else config.workdir / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(f"report: {target}" + (f" (stopped: {stopped})" if stopped else ""))
    return 1 if stopped else 0


def _judge(args: argparse.Namespace, task_root: Path) -> int:
    """Add ``point_recall_judged`` to saved runs (no agent runs) and write report-judged.md."""
    if not args.env_file:
        print("judge needs --env-file with OPENAI_API_KEY", file=sys.stderr)
        return 2
    key = load_api_keys(Path(args.env_file)).get("OPENAI_API_KEY")
    if not key:
        print("no OPENAI_API_KEY in --env-file", file=sys.stderr)
        return 2
    workdir = Path(args.workdir)
    stopped = ""
    try:
        judged, spent = judge_workdir(
            workdir, task_root, openai_call(key, args.judge_model), args.judge_model,
            args.judge_budget_usd,
        )
    except (JudgeRateLimited, JudgeBudgetExhausted) as exc:
        judged, spent, stopped = {}, float(getattr(exc, "spent", 0.0)), str(exc)
    results = []
    for path in sorted((workdir / "runs").glob("*/result.json")):
        result = RunResult(**json.loads(path.read_text(encoding="utf-8")))
        cache = path.parent / "judged.json"
        if cache.is_file():
            result.scores["point_recall_judged"] = json.loads(cache.read_text(encoding="utf-8"))[
                "point_recall_judged"
            ]
        results.append(result)
    metrics = tuple(
        m for pair in ((m, "point_recall_judged") if m == "point_recall" else (m,) for m in METRICS)
        for m in pair
    )
    target = Path(args.report) if args.report else workdir / "report-judged.md"
    target.write_text(
        render(
            results,
            {"Date": date.today().isoformat(), "Judge": f"{args.judge_model} (${spent:.4f} this call)"},
            stopped,
            metrics,
        ),
        encoding="utf-8",
    )
    print(f"judged {len(judged)} run(s) for ${spent:.4f}; report: {target}"
          + (f" (stopped: {stopped})" if stopped else ""))
    return 0 if not stopped else 1
