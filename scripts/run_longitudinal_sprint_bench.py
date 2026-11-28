"""Run the 3-arm x 4-session Longitudinal Frozen-Repo Coding-Sprint benchmark.

Defaults to `--dry-run` (zero LLM spend) to validate the fixture, deterministic
`beacon_eval` scoring, and token/redundant-read accounting before any live run.

Usage:
    python scripts/run_longitudinal_sprint_bench.py --dry-run
    python scripts/run_longitudinal_sprint_bench.py --fixture fixtures/sprint_simonw_llm_764dc386.json --dry-run --out results/sprint_dry_run.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The Menhir checkout releases are resolved from: $MENHIR_REPO, else the sibling ../menhir.
DEFAULT_MENHIR_REPO = os.environ.get("MENHIR_REPO") or str(REPO_ROOT.parent / "menhir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archolith_bench.beacon_eval.runner import DEFAULT_MODEL, DEFAULT_RUN_RESERVE  # noqa: E402
from archolith_bench.engineering_evidence.live_executor import (  # noqa: E402
    LiveSprintConfig,
    OpenCodeSprintExecutor,
)
from archolith_bench.beacon_eval.runner import Budget  # noqa: E402
from archolith_bench.engineering_evidence.campaign import (  # noqa: E402
    Campaign,
    ProgressFeed,
    RunDirInUse,
    new_run_id,
    prepare_run_dir,
    prompt_hash,
    prompt_manifest,
)
from archolith_bench.engineering_evidence.server_usage import ServerRates  # noqa: E402
from archolith_bench.engineering_evidence.sprint import (  # noqa: E402
    SPRINT_ARMS,
    V2_ARMS,
    PROMPT_SET,
    DryRunExecutor,
    SprintFixture,
    run_longitudinal_sprint,
)
from archolith_bench.engineering_evidence.sprint_rubric import RUBRIC_VERSION  # noqa: E402
from archolith_bench.staged.rundir import RunConfigMismatch  # noqa: E402
from archolith_bench.staged.serve import graph_settings  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json"


def _print_table(report: dict) -> None:
    mode = "DRY-RUN (0 LLM calls)" if report["dry_run"] else "LIVE"
    print(f"\n=== Longitudinal Frozen-Repo Sprint: {report['fixture']} [{mode}] ===")
    print(
        f"Repo: {report['repo']['name']} @ {report['repo']['commit'][:12]} "
        f"({report['repo']['url']})"
    )
    print(
        f"\n{'Arm':22s} {'MeanAcc':>8s} {'TotalTok':>10s} {'S2-S4Tok':>10s} "
        f"{'Turns':>6s} {'Tools':>6s} {'RedundRd':>8s} {'MemSaved':>8s}"
    )
    print("-" * 86)
    for arm in SPRINT_ARMS:
        if arm not in report["arms"]:
            continue
        r = report["arms"][arm]
        print(
            f"{arm:22s} {r['mean_accuracy']:>8.4f} {r['total_tokens']:>10d} "
            f"{r['downstream_tokens_s2_to_end']:>10d} {r['total_turns']:>6d} "
            f"{r['total_tool_calls']:>6d} {r['total_redundant_file_reads']:>8d} "
            f"{r['total_memories_saved']:>8d}"
        )
    if report["comparisons_vs_no_menhir"]:
        print("\nRubric vs no_menhir (accuracy-gated tokens + turns):")
        for arm, cmp in report["comparisons_vs_no_menhir"].items():
            print(
                f"  {arm:20s}: [{cmp['verdict']}] acc={cmp['accuracy_delta']:+.4f}, "
                f"total_tok={cmp['total_tokens_delta_pct']:+.2f}%, "
                f"s2_s4_tok={cmp['downstream_tokens_delta_pct']:+.2f}%, "
                f"turns={cmp['total_turns_delta_pct']:+.2f}%, "
                f"tools={cmp['total_tool_calls_delta_pct']:+.2f}%, "
                f"redundant_reads={cmp['redundant_reads_delta']:+d}"
            )


def _print_v2(report: dict) -> None:
    mode = "DRY-RUN (0 LLM calls)" if report["dry_run"] else "LIVE"
    print(f"\n=== Sprint v2 [{report['rubric_version']}] {report['fixture']} run {report['run_id']} [{mode}] ===")
    camp = report["campaign"]
    print(f"spent ${camp['spent_usd']:.4f} (cap {camp['max_usd']}); stopped_early={camp['stopped_early']} {camp['stop_reason']}")
    for arm, rows in report["analysis"]["per_chain"].items():
        for r in rows:
            print(
                f"  {arm:20s} r{r['repeat']} acc={r['accuracy']:.4f} complete={r['complete']} "
                f"uncached(client+server)={r['primary_efficiency_tokens']} "
                f"cost=${r['client_cost_usd'] + r['server_cost_usd']:.4f}"
            )
    for label, v in report["analysis"]["verdicts"].items():
        print(
            f"  {label}: {v['combined_verdict']} "
            f"(acc d={v['accuracy']['mean_delta']}, eff d%={v['efficiency']['mean_delta_pct']})"
        )


def _git_state(root: str | None) -> str | None:
    from archolith_bench.staged.__main__ import _menhir_commit

    return _menhir_commit(root)


def _agent_version(agent: str) -> str | None:
    """The harness's git state (an editable install); OpenCode's version is in its own logs."""
    if agent != "archolith-harness":
        return None
    import archolith_harness

    return _git_state(str(Path(archolith_harness.__file__).resolve().parents[2]))


def sprint_serve_overrides(chat_model: str) -> dict[str, str]:
    """Write-path env for the sprint's served Menhir (served_menhir defaults to recall-only).

    Benchmark mode is off so the maintenance scheduler (failed-enrichment retry, stale-lease
    recovery, conflict resolution) runs as in a real deployment and the 2026-11-27 run. The chat
    model is pinned so server usage matches the --server-rates table.
    """
    return {
        "MENHIR_BENCHMARK_MODE": "0",
        "MENHIR_LOOPBACK_PROFILE": "lean",
        "LLM_CHAT_PROVIDER": "openai",
        "GRAPHITI_LLM_PROVIDER": "openai",
        "OPENAI_CHAT_MODEL": chat_model,
        "LLM_MAX_TOKENS": "3000",
        "GRAPHITI_ADD_EPISODE_TIMEOUT_SECONDS": "1800",
    }


def preflight_cap_usd(*, sessions: int, est_usd_per_session: float | None, parallel: int,
                      reserve_usd: float) -> float | None:
    """Smallest --max-usd that lets every session start: Budget.check refuses a new session once
    spent + reserve for each session in flight would pass the cap (run v2r2-20261127 stopped
    early on exactly that). The estimate gets a 25% margin. None when no estimate is given."""
    if not est_usd_per_session or est_usd_per_session <= 0:
        return None
    return round(sessions * est_usd_per_session * 1.25 + parallel * reserve_usd, 2)


def _run_v2(args: argparse.Namespace, fixture: SprintFixture) -> int:
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip()) if args.arms else V2_ARMS
    live = bool(args.live and not args.dry_run)
    if live and "menhir_longitudinal" in arms and not args.server_rates:
        print("ERROR: --server-rates input,cached_input,output ($/1M) is required for the Menhir arm", file=sys.stderr)
        return 2
    try:
        rates = ServerRates.parse(args.server_rates) if args.server_rates else None
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if live and not args.max_usd:
        print("ERROR: a live v2 campaign needs --max-usd", file=sys.stderr)
        return 2
    if live:
        need = preflight_cap_usd(
            sessions=args.repeats * len(arms) * len(fixture.sessions),
            est_usd_per_session=args.est_usd_per_session, parallel=args.parallel,
            reserve_usd=args.reserve_usd,
        )
        if need is None or args.max_usd + 1e-9 < need:
            print(f"ERROR: --max-usd {args.max_usd} cannot finish this campaign: it needs "
                  f"{'--est-usd-per-session' if need is None else f'at least ${need:.2f}'} "
                  "(sessions x estimate x 1.25, plus parallel x reserve held in flight)", file=sys.stderr)
            return 2
    if live and args.serve and not args.graph:
        print("ERROR: --serve needs --graph <build yaml> (e.g. builds/sprint-v2.yaml)", file=sys.stderr)
        return 2
    menhir_release = None
    if live and args.graph and not args.serve and not args.menhir_mcp_url:
        # No tree named: serve a release, resolved once here and recorded for the whole run.
        from archolith_bench.staged.release_tree import ReleaseError, release_tree

        try:
            rel = release_tree(Path(args.menhir_repo or DEFAULT_MENHIR_REPO), args.menhir_release)
        except ReleaseError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        args.serve, menhir_release = str(rel.path), rel.tag
        print(f"serving Menhir {rel.tag} ({rel.commit[:12]}) from {rel.path}")
    if live and args.serve:
        from archolith_bench.staged.release_tree import has_lean_profile

        if not has_lean_profile(Path(args.serve)):
            print(f"ERROR: {args.serve} has no MENHIR_LOOPBACK_PROFILE=lean tool set "
                  f"({'release ' + menhir_release if menhir_release else 'served tree'}); the Menhir "
                  "arm would see every tool. Pass --serve <tree with the lean profile>.", file=sys.stderr)
            return 2
    run_id = args.run_id or new_run_id()
    manifest = prompt_manifest(fixture, arms, args.cache_shaping)
    fixture_sha = hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest()
    resolved = {
        "rubric_version": RUBRIC_VERSION, "prompt_set": PROMPT_SET,
        "fixture": fixture.name, "fixture_sha256": fixture_sha,
        "arms": list(arms), "repeats": args.repeats, "run_id": run_id, "model": args.model,
        "reserve_tokens": args.reserve_tokens, "timeout_s": args.timeout_s,
        "prompt_sha256": prompt_hash(manifest), "dry_run": not live,
    }
    if args.agent != "opencode":
        # A resume never mixes agents in one run dir; OpenCode dirs predate the key and still resume.
        resolved["agent"] = args.agent
        resolved["harness_flex"] = args.harness_flex  # open and fallback censor differently
    if menhir_release:
        # A resume after a newer release is refused rather than mixing two Menhirs in one run dir.
        resolved["menhir_release"] = menhir_release
    try:
        rd = prepare_run_dir(Path(args.out_root), f"{fixture.name}-{run_id}", resolved, resume=args.resume)
    except (RunDirInUse, RunConfigMismatch) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    (rd.path / "prompts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    menhir_tree = args.serve or args.menhir_repo
    provenance = {
        "fixture_path": str(Path(args.fixture)), "fixture_sha256": fixture_sha,
        "bench_commit": _git_state(str(REPO_ROOT)),
        "menhir_commit": _git_state(menhir_tree) if menhir_tree else None,
        "menhir_release": menhir_release,  # None: an explicit --serve tree or an external server
        "model": args.model, "reserve_tokens": args.reserve_tokens, "timeout_s": args.timeout_s,
        "repeats": args.repeats, "parallel": args.parallel, "run_id": run_id,
        "prompt_sha256": resolved["prompt_sha256"], "loopback_profile": "lean" if args.serve else None,
        "server_chat_model": args.server_chat_model if args.serve else None,
        # OpenCode's serviceTier (beacon_eval.isolation.SERVICE_TIER_ENV); its step cost is still
        # priced at the Standard catalog rate, so under flex client_cost_usd overstates the bill.
        "cache_shaping": args.cache_shaping, "prompt_set": PROMPT_SET,
        "agent": args.agent, "agent_version": _agent_version(args.agent),
        "harness_flex": args.harness_flex if args.agent == "archolith-harness" else None,
        # archolith-harness requests Flex on every call and prices each step at its served tier
        # (fallback mode may resend on Standard); only OpenCode's tier comes from the environment.
        "client_service_tier": ("flex" if args.harness_flex == "open" else "flex+standard-fallback")
        if args.agent == "archolith-harness"
        else os.environ.get("BEACON_EVAL_SERVICE_TIER", "").strip() or "default",
        "server_service_tier": graph_settings(args.graph, dict(os.environ))["service_tier"]
        if args.serve and args.graph else None,
        "server_benchmark_mode": False if args.serve else None,
        "server_rates_per_1m": None if rates is None else [rates.input, rates.cached_input, rates.output],
    }
    budget = Budget(None, None, cap_usd=args.max_usd, reserve_usd=args.reserve_usd)
    progress = ProgressFeed(
        f"sprint-v2-{run_id}", f"sprint v2 {fixture.name}",
        args.repeats * len(arms) * len(fixture.sessions), ceiling_usd=args.max_usd,
    )
    structure_index = None
    stack = contextlib.ExitStack()
    failed = True
    with stack:
        if live:
            env_path = Path(args.env_file) if args.env_file else None
            mcp_url = args.menhir_mcp_url
            telemetry_db = Path(args.telemetry_db) if args.telemetry_db else None
            if args.serve:
                from archolith_bench.staged.serve import served_menhir

                url = stack.enter_context(
                    served_menhir(
                        Path(args.serve).resolve(), port=args.serve_port,
                        graph=graph_settings(args.graph, dict(os.environ)),
                        openai_key=os.getenv("OPENAI_API_KEY", ""), out_dir=rd.path,
                        menhir_bin=args.menhir_bin,
                        env_overrides=sprint_serve_overrides(args.server_chat_model),
                    )
                )
                mcp_url, telemetry_db = f"{url}/mcp-http", rd.path / "mcp_telemetry.db"
            work = Path(args.work_dir) / f"{fixture.repo.name}-{run_id}"
            executor_cls = OpenCodeSprintExecutor
            if args.agent == "archolith-harness":
                from functools import partial

                from archolith_bench.engineering_evidence.harness_executor import HarnessSprintExecutor

                executor_cls = partial(HarnessSprintExecutor, flex_mode=args.harness_flex)
            executor = executor_cls(
                LiveSprintConfig(
                    work_dir=work, cache_dir=Path(args.cache_dir), model=args.model,
                    menhir_mcp_url=mcp_url, env_file=env_path, reserve_tokens=args.reserve_tokens,
                    timeout_s=args.timeout_s, runs_dir=rd.path / "runs", checkout_dir=work / "co",
                    telemetry_db=telemetry_db, gzip_events=True, cache_shaping=args.cache_shaping,
                )
            )
            if any(a.startswith("menhir") for a in arms) and not args.resume:
                structure_index = executor.index_pristine(fixture, run_id)
        else:
            executor = DryRunExecutor()
        campaign = Campaign(
            fixture, executor=executor, run_dir=rd, run_id=run_id, arms=arms, repeats=args.repeats,
            parallel=args.parallel, budget=budget, rates=rates, progress=progress,
            resume=args.resume, dry_run=not live,
        )
        try:
            chains, stop = campaign.run()
            report = campaign.build_report(chains, stop, provenance=provenance, structure_index=structure_index)
            failed = bool(stop)
        finally:
            progress.finish(failed=failed)
    (rd.path / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_v2(report)
    print(f"\nwrote run dir: {rd.path}")
    return 3 if report["campaign"]["stopped_early"] else 0


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
    v2 = parser.add_argument_group("v2 campaign (selected by --repeats)")
    v2.add_argument("--repeats", type=int, default=None, help="Chains per arm (turns on the v2 campaign).")
    v2.add_argument("--parallel", type=int, default=1, help="Chains run at once.")
    v2.add_argument("--run-id", default=None, help="Run id (default: UTC timestamp).")
    v2.add_argument("--out-root", default=str(REPO_ROOT / "results" / "sprint-v2"),
                    help="Parent of <fixture>-<run-id>/ (an existing run dir is refused unless --resume).")
    v2.add_argument("--resume", action="store_true", help="Reuse completed chains of an existing run dir.")
    v2.add_argument("--max-usd", type=float, default=None, help="Shared dollar cap (client + server).")
    v2.add_argument("--cache-shaping", action="store_true",
                    help="Shared per-arm prompt_cache_key and arm-wide prompt text first (cache reuse).")
    v2.add_argument("--agent", choices=("opencode", "archolith-harness"), default="opencode",
                    help="Agent that runs each session (results are never pooled across agents).")
    v2.add_argument("--harness-flex", choices=("open", "fallback"), default="fallback",
                    help="archolith-harness timeouts: open = no session clock, Flex only; fallback = "
                         "--timeout-s per session, resending on Standard rather than missing it.")
    v2.add_argument("--est-usd-per-session", type=float, default=None,
                    help="Expected client+server USD per session (required for --live; from a prior run).")
    v2.add_argument("--reserve-usd", type=float, default=0.05,
                    help="Per-session reserve held by Budget.check for every session in flight.")
    v2.add_argument("--server-rates", default="", help="input,cached_input,output in $/1M (Menhir server LLM).")
    v2.add_argument("--telemetry-db", default="", help="MENHIR_MCP_TELEMETRY_DB sqlite (when not using --serve).")
    v2.add_argument("--serve", default=None, metavar="MENHIR_TREE",
                    help="serve this Menhir tree (staged.serve.served_menhir) over --graph's Neo4j; "
                         "overrides --menhir-release (for unreleased branches).")
    v2.add_argument("--menhir-release", default="latest", metavar="latest|vX.Y.Z",
                    help="with --graph and no --serve/--menhir-mcp-url, serve this Menhir release from "
                         "--menhir-repo (a cached worktree per tag; recorded as menhir_release).")
    v2.add_argument("--graph", default=None, help="build YAML naming the graph, e.g. builds/sprint-v2.yaml.")
    v2.add_argument("--serve-port", type=int, default=18096)
    v2.add_argument("--server-chat-model", default="gpt-6-luna",
                    help="OPENAI_CHAT_MODEL for the served Menhir; must match --server-rates.")
    v2.add_argument("--menhir-bin", default=os.environ.get("MENHIR_MAIN_BIN"))
    v2.add_argument("--menhir-repo", default=None,
                    help="Menhir checkout: provenance when not serving, and where --menhir-release "
                         f"tags are resolved (default {DEFAULT_MENHIR_REPO}).")
    args = parser.parse_args(argv)

    fixture = SprintFixture.from_file(args.fixture)
    if args.max_sessions is not None:
        from dataclasses import replace

        fixture = replace(fixture, sessions=fixture.sessions[: args.max_sessions])
    if args.repeats is not None:
        if args.arms == ",".join(SPRINT_ARMS):
            args.arms = ""
        return _run_v2(args, fixture)
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
