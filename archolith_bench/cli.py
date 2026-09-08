"""CLI entrypoint for archolith-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .arms import ARMS
from .harness.presets import (
    PresetConflict,
    apply_preset,
    format_preset_list,
    get_preset,
)
from .core.api import API_KEY, DIRECT_URL, MODEL, PROXY_URL, check_proxy_health, send_chat
from .core.display import print_cross_scenario_summary, print_four_way_table, print_summary
from .core.metrics import PRICING_DEFAULTS, PricingModel
from .core.report import save_results
from .core.scenario import Scenario, list_scenarios


def _add_common_proxy_args(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments for proxy and stack subcommands.

    These args are common to both proxy and stack suites:
    scenario selection, budget, turns, endpoints, model, output, api-key,
    fact probes, restart scoring, and pricing configuration.
    """
    parser.add_argument("--scenario", type=Path, help="Path to scenario JSON file")
    parser.add_argument("--all", action="store_true", help="Run all scenarios in scenarios/")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit")
    parser.add_argument("--budget", type=int, default=None, help="Token budget")
    parser.add_argument("--turns", type=int, default=None, help="Limit number of turns")
    parser.add_argument("--proxy", default=PROXY_URL, help="Proxy URL")
    parser.add_argument("--direct", default=DIRECT_URL, help="Direct upstream URL")
    parser.add_argument("--model", default=MODEL, help="Model to use")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory for results")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--no-probes", action="store_true", help="Skip fact probes")
    parser.add_argument("--no-restart", action="store_true", help="Skip restart/bootstrap scoring")
    parser.add_argument("--no-collapse-abort", action="store_true",
                        help="Don't abort an arm on consecutive short-output turns "
                             "(the collapse guard mis-fires on agentic scenarios with "
                             "legitimately terse tool-continuation responses)")
    parser.add_argument("--poll-interval", type=float, default=3.0,
                        help="Seconds to wait between proxy turn completion and trace fetch")
    parser.add_argument("--provider", default=None,
                        help=f"Pricing provider. Available: {', '.join(PRICING_DEFAULTS)}")
    parser.add_argument("--pricing-file", type=Path, default=None,
                        help="Path to a JSON file overriding provider pricing rates")


def _add_evidence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-evidence", type=Path, default=None,
                        help="Optional shared evidence artifact output (.md or .json)")
    parser.add_argument("--public-copy", action="store_true",
                        help="Mark the shared evidence artifact as allowed for public copy")


def _load_pricing(args: argparse.Namespace) -> PricingModel | None:
    """Resolve pricing model from --provider, --pricing-file, or neither.

    Returns None when the user supplied no pricing flags (cost columns are
    omitted and the report caveats accordingly).
    """
    if args.pricing_file:
        with open(args.pricing_file, encoding="utf-8") as f:
            raw = json.load(f)
        return PricingModel(**raw)

    if args.provider:
        provider = args.provider.lower()
        if provider not in PRICING_DEFAULTS:
            print(f"ERROR: Unknown provider '{args.provider}'. "
                  f"Available: {', '.join(PRICING_DEFAULTS)}", file=sys.stderr)
            sys.exit(1)
        return PRICING_DEFAULTS[provider]

    return None


def _resolve_menhir_bearer_key() -> str:
    """Resolve the Menhir bearer token with priority: MENHIR_AGENT_KEY > MENHIR_API_KEY > API_KEY."""
    return os.getenv("MENHIR_AGENT_KEY") or os.getenv("MENHIR_API_KEY") or API_KEY or ""


def _publish_cli_evidence(
    args: argparse.Namespace,
    *,
    title: str,
    product: str,
    ability: str,
    fixture_or_live_source: str,
    model_provider: str,
    environment_caveats: list[str],
    metric_rows: list[dict],
    artifact: dict,
) -> None:
    out_path = getattr(args, "publish_evidence", None)
    if not out_path:
        return

    from .core.evidence import EvidenceRecord, current_commit, publish_evidence

    repo_root = Path(__file__).resolve().parents[1]
    record = EvidenceRecord(
        title=title,
        command=getattr(args, "command_text", "archolith-bench"),
        commit=current_commit(repo_root),
        product=product,
        ability=ability,
        fixture_or_live_source=fixture_or_live_source,
        model_provider=model_provider,
        environment_caveats=environment_caveats,
        public_copy_allowed=bool(getattr(args, "public_copy", False)),
        metric_rows=metric_rows,
        artifact=artifact,
    )
    path = publish_evidence(record, out_path)
    print(f"  Shared evidence published to {path}")


def _proxy_metric_rows(results: list[dict]) -> list[dict]:
    rows = []
    for item in results:
        summary = item.get("summary", {})
        quality = item.get("quality", {})
        rows.append({
            "scenario": item.get("scenario"),
            "arm": item.get("arm") or item.get("stack_arm"),
            "turns": item.get("turns_run"),
            "savings_ratio": summary.get("overall_savings_ratio"),
            "direct_tokens": summary.get("total_direct_input_tokens"),
            "arm_tokens": summary.get("total_proxy_input_tokens"),
            "quality_score": quality.get("score"),
        })
    return rows


def _filter_metric_rows(data: dict) -> list[dict]:
    return [
        {
            "category": row.get("category"),
            "samples": row.get("samples"),
            "raw_tokens": row.get("total_raw_tokens"),
            "filtered_tokens": row.get("total_filtered_tokens"),
            "savings_ratio": row.get("savings_ratio"),
        }
        for row in data.get("per_category", [])
    ]


def _audit_metric_rows(data: dict) -> list[dict]:
    return [
        {
            "server": row.get("server"),
            "before_tokens": row.get("before_tokens"),
            "after_tokens": row.get("after_tokens"),
            "token_change_pct": row.get("token_change_pct"),
            "status": row.get("status"),
        }
        for row in data.get("per_server", [])
    ]


def _harness_metric_rows(data: dict) -> list[dict]:
    rows = [
        {
            "arm": arm,
            "n": result.get("n"),
            "score": result.get("score"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
        }
        for arm, result in data.get("arms", {}).items()
    ]
    for arm, delta in data.get("deltas", {}).items():
        rows.append({
            "arm": f"delta:{arm}",
            "score": delta.get("score_delta"),
            "input_tokens": delta.get("input_token_reduction_pct"),
            "cost_usd": delta.get("cost_reduction_pct"),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="archolith-bench",
        description="Unified benchmark suite for the archolith product family",
    )
    subparsers = parser.add_subparsers(dest="suite", help="Benchmark suite to run")

    # ---- proxy subcommand ----
    proxy_p = subparsers.add_parser("proxy", help="Proxy suite: multi-turn token savings + continuity")
    _add_common_proxy_args(proxy_p)
    proxy_p.add_argument("--arms", type=str, default="proxy_plus_filter",
                         help=f"Comma-separated arms to run. Available: {', '.join(ARMS)}")
    proxy_p.add_argument("--budgets", type=str, default=None,
                         help="Comma-separated budgets for matrix run (e.g., 4000,8000,15000)")
    proxy_p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    proxy_p.add_argument("--experiment", type=str, default=None,
                         help="Named experiment -- snapshots proxy config, saves to experiments/<name>/")
    proxy_p.add_argument("--config", type=str, default=None,
                         help="JSON proxy config overrides")
    _add_evidence_args(proxy_p)

    # ---- filter subcommand ----
    filter_p = subparsers.add_parser("filter", help="Filter suite: compression-ratio measurement via archolith-filter")
    filter_p.add_argument("--corpora", type=Path, default=Path("corpora"),
                          help="Path to corpora directory (default: corpora/)")
    filter_p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                          help="Output format (default: markdown)")
    filter_p.add_argument("--output-dir", type=Path, default=Path("results"),
                          help="Output directory for results")
    _add_evidence_args(filter_p)

    # ---- stack subcommand ----
    stack_p = subparsers.add_parser("stack", help="Stack suite: four-way comparison (direct/filter/proxy/proxy+filter)")
    _add_common_proxy_args(stack_p)
    _add_evidence_args(stack_p)

    # ---- audit subcommand ----
    audit_p = subparsers.add_parser("audit", help="Audit suite: MCP token-waste before/after comparison")
    audit_p.add_argument("--before", type=Path, required=True,
                         help="Path to 'before' JSON audit report")
    audit_p.add_argument("--after", type=Path, required=True,
                         help="Path to 'after' JSON audit report")
    audit_p.add_argument("--format", choices=["markdown", "json", "report"], default="report",
                         help="Output format (default: report)")
    audit_p.add_argument("--output-dir", type=Path, default=Path("results"),
                         help="Output directory for results")
    _add_evidence_args(audit_p)

    # ---- industry subcommand ----
    industry_p = subparsers.add_parser(
        "industry",
        help="Industry benchmark coverage matrix for launch-readiness claims",
    )
    industry_p.add_argument("--product", default=None,
                            help="Filter by product, e.g. archolith-context")
    industry_p.add_argument("--suite", dest="benchmark_suite", default=None,
                            help="Filter by suite, e.g. proxy/filter/audit")
    industry_p.add_argument("--launch-only", action="store_true",
                            help="Only include launch-relevant implemented/candidate benchmarks")
    industry_p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                            help="Output format (default: markdown)")
    industry_p.add_argument("--output-dir", type=Path, default=Path("results"),
                            help="Output directory for generated registry files")
    industry_p.add_argument("--out", type=Path, default=None,
                            help="Optional explicit output file")
    _add_evidence_args(industry_p)

    # ---- harness subcommand ----
    harness_p = subparsers.add_parser(
        "harness",
        help="Run an official external benchmark as a direct-vs-proxy A/B",
    )
    harness_p.add_argument("benchmark_id", nargs="?", default=None,
                           help="Benchmark id, e.g. longbench-v2 (omit with --list)")
    harness_p.add_argument("--list", action="store_true", dest="list_adapters",
                           help="List available harness adapters and exit")
    harness_p.add_argument("--preset", default=None,
                           help="Named smoke preset (see --list-presets). Sets benchmark id, "
                                "subset, limit and arms; explicit flags still win.")
    harness_p.add_argument("--list-presets", action="store_true", dest="list_presets",
                           help="List available smoke presets and exit")
    harness_p.add_argument("--arms", default="direct,proxy_only,proxy_plus_filter",
                           help="Comma-separated experiment arms")
    harness_p.add_argument("--subset", default=None, help="Benchmark subset/domain filter")
    harness_p.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    harness_p.add_argument("--model", default=MODEL, help="Model id for inference")
    harness_p.add_argument("--offline-fixture", type=Path, default=None,
                           help="Run offline against a bundled fixture JSON (no API calls)")
    harness_p.add_argument("--menhir-url", default=None,
                           help="Memory benchmarks: throwaway menhir base URL (refuses prod-looking targets)")
    harness_p.add_argument("--neo4j-uri", default="bolt://localhost:7688",
                           help="menhir-scalar-state: throwaway Neo4j bolt URI to verify shadow Views "
                                "over (refuses prod port 7687; default bolt://localhost:7688)")
    harness_p.add_argument("--neo4j-user", default="neo4j",
                           help="menhir-scalar-state: throwaway Neo4j user (default neo4j)")
    harness_p.add_argument("--neo4j-password", default="benchthrowaway",
                           help="menhir-scalar-state: throwaway Neo4j password (default benchthrowaway)")
    harness_p.add_argument("--scalar-max-wait-s", type=float, default=180.0,
                           help="menhir-scalar-state: max seconds to wait for the background scheduler "
                                "to materialize scalar_state Views (default 180)")
    harness_p.add_argument("--scalar-no-cleanup", action="store_true",
                           help="menhir-scalar-state: do NOT tear down the seeded namespace after the "
                                "run, so the graph can be inspected over bolt (debugging)")
    harness_p.add_argument("--scalar-fixture", choices=["default", "third-party"], default="default",
                           help="menhir-scalar-state: fixture set -- 'default' (first-person, 9 kinds + "
                                "advisory/control) or 'third-party' (named 'Alice' subject, Lever 3 "
                                "binder-isolation experiment)")
    harness_p.add_argument("--recall-limit", type=int, default=10,
                           help="Memory benchmarks: how many memory snippets to recall per question "
                                "(default 10; raise for multi-fact temporal-reasoning questions)")
    harness_p.add_argument("--resume", action="store_true",
                           help="Memory benchmarks: checkpoint each item and skip already-completed "
                                "items on rerun (survives crashes/rate-limit aborts). Rerun the same command.")
    harness_p.add_argument("--scorer", choices=["containment", "llm-judge"], default="containment",
                           help="Memory benchmarks: 'containment' (fast, offline) or 'llm-judge' "
                                "(LongMemEval protocol; comparable to published Mem0/Zep numbers).")
    harness_p.add_argument("--judge-model", default="gpt-4o-mini",
                           help="llm-judge: grader model (default gpt-4o-mini; use gpt-4o for paper-fidelity)")
    harness_p.add_argument("--judge-url", default=None,
                           help="llm-judge: grader base URL (default: UPSTREAM_BASE_URL/OpenAI)")
    harness_p.add_argument("--judge-api-key", default=None,
                           help="llm-judge: grader API key (default: OPENAI_API_KEY env)")
    harness_p.add_argument("--confirm-menhir-reset", action="store_true",
                           help="Allow memory benchmarks to reset throwaway Menhir groups after each item")
    harness_p.add_argument("--dry-run-menhir-reset", action="store_true",
                           help="Print Menhir group resets without performing them")
    harness_p.add_argument("--recall-only", action="store_true",
                           help="Memory benchmarks: recall against a PRE-BUILT graph in stable "
                                "per-question namespaces (--namespace-template). Skips ingest AND "
                                "reset, so no --confirm-menhir-reset is needed. Use after the graph "
                                "is ingested+enriched once (the LongMemEval Mode-B recall-only A/B).")
    harness_p.add_argument("--namespace-template", default="lme-{question_id}",
                           help="Recall-only: namespace per item, formatted with {question_id} "
                                "(default: lme-{question_id}, matching _ingest_lme.py).")
    harness_p.add_argument("--format", choices=["markdown", "json"], default="markdown",
                           help="Evidence output format (default: markdown)")
    harness_p.add_argument("--output-dir", type=Path, default=Path("results"),
                           help="Output directory for evidence files")
    harness_p.add_argument("--out", type=Path, default=None,
                           help="Optional explicit evidence output file")
    _add_evidence_args(harness_p)

    # ---- dashboard subcommand ----
    dash_p = subparsers.add_parser("dashboard", help="Live view of in-progress memory runs (reads checkpoints)")
    dash_p.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Directory holding .checkpoint_*.jsonl files (default: results)")
    dash_p.add_argument("--menhir-url", default=None,
                        help="Throwaway menhir base URL to probe for live activity (e.g. http://localhost:8101)")
    dash_p.add_argument("--interval", type=float, default=5.0,
                        help="Refresh interval seconds (default: 5)")
    dash_p.add_argument("--once", action="store_true", help="Print a single snapshot and exit")
    dash_p.add_argument("--total-items", type=int, default=None,
                        help="Per-arm item total for the progress bar (default: inferred from variant)")
    dash_p.add_argument("--serve", action="store_true",
                        help="Serve an auto-refreshing web dashboard instead of the terminal view")
    dash_p.add_argument("--host", default="127.0.0.1", help="Web dashboard bind host (default: 127.0.0.1)")
    dash_p.add_argument("--port", type=int, default=8200, help="Web dashboard port (default: 8200)")
    dash_p.add_argument("--items", type=int, default=20,
                        help="Turn-by-turn feed length: latest N item results per run (default: 20)")
    dash_p.add_argument(
        "--scalar-neo4j-uri",
        default=os.getenv("LME_DASHBOARD_NEO4J_URI"),
        help="Enable the read-only one-task scalar viewer against this Bolt URI "
             "(env: LME_DASHBOARD_NEO4J_URI)",
    )
    dash_p.add_argument(
        "--scalar-neo4j-user",
        default=os.getenv("LME_DASHBOARD_NEO4J_USER", "neo4j"),
        help="Scalar viewer Neo4j user (env: LME_DASHBOARD_NEO4J_USER; default: neo4j)",
    )
    dash_p.add_argument(
        "--scalar-neo4j-password",
        default=os.getenv("LME_DASHBOARD_NEO4J_PASSWORD"),
        help="Scalar viewer Neo4j password (prefer env: LME_DASHBOARD_NEO4J_PASSWORD)",
    )
    dash_p.add_argument(
        "--scalar-telemetry-db",
        type=Path,
        default=(
            Path(os.environ["LME_DASHBOARD_TELEMETRY_DB"])
            if os.getenv("LME_DASHBOARD_TELEMETRY_DB")
            else None
        ),
        help="Optional Menhir telemetry DB for exact k-sample vote receipts "
             "(env: LME_DASHBOARD_TELEMETRY_DB)",
    )
    dash_p.add_argument(
        "--scalar-task",
        default=os.getenv("LME_DASHBOARD_SCALAR_TASK"),
        help="Namespace selected when the scalar viewer opens (e.g. lme-01493427)",
    )
    dash_p.add_argument("--active-within", type=float, default=300.0,
                        help="Only show runs whose checkpoint was written within N seconds "
                             "(active sessions). 0 = show all runs including finished ones (default: 300)")

    # ---- extraction-bench subcommand ----
    ext_p = subparsers.add_parser("extraction-bench",
                                  help="Simulate menhir's backend extraction pipeline to compare models on speed+quality")
    ext_p.add_argument("--repeats", type=int, default=1,
                       help="Replay the corpus N times to amplify sustained load (default 1)")
    ext_p.add_argument("--targets-file", type=Path, default=None,
                       help="JSON list of {label,base_url,api_key_env,model} to add to the defaults")
    ext_p.add_argument("--exclude", default=None,
                       help="Comma-separated substrings; skip any target whose label/model matches "
                            "(e.g. 'gpt-5' to drop slow reasoning models)")
    ext_p.add_argument("--all", action="store_true",
                       help="Benchmark the full candidate sweep instead of just the blessed "
                            "keepers (gpt-4.1-nano + qwen3-next-80b)")
    ext_p.add_argument("--out", type=Path, default=None, help="Optional evidence output file")
    _add_evidence_args(ext_p)

    # ---- ports subcommand ----
    ports_p = subparsers.add_parser("ports", help="Index running stack processes by label + port")
    ports_p.add_argument("--all", action="store_true", dest="show_all",
                         help="Show every listening port, not just menhir/neo4j/bench/etc.")
    ports_p.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON")

    # ---- menhir subcommand group ----
    menhir_p = subparsers.add_parser("menhir", help="Menhir capability evidence runners")
    menhir_sub = menhir_p.add_subparsers(dest="menhir_command", help="Menhir command")
    menhir_sub.add_parser("list", help="List Menhir capabilities and evidence gates")

    smoke_p = menhir_sub.add_parser("smoke", help="Run offline-safe Menhir smoke ladders")
    smoke_p.add_argument("--output-dir", type=Path, default=Path("results/menhir-smoke"))
    smoke_p.add_argument("--publish-dir", type=Path, default=None,
                         help="Optional directory for tracked evidence markdown")

    hygiene_p = menhir_sub.add_parser(
        "bootstrap-hygiene", help="Gate recent/bootstrap scope hygiene"
    )
    hygiene_p.add_argument(
        "--fixture", type=Path, default=Path("fixtures/menhir_bootstrap_hygiene.json")
    )
    hygiene_p.add_argument("--offline", action="store_true", help="Run deterministic fixture policy only")
    hygiene_p.add_argument("--menhir-url", default=None, help="Throwaway Menhir URL for live black-box mode")
    hygiene_p.add_argument("--confirm-menhir-reset", action="store_true")
    hygiene_p.add_argument("--out", type=Path, default=Path("results/menhir-bootstrap-hygiene.json"))

    def add_menhir_run_args(p: argparse.ArgumentParser, default_out: str) -> None:
        p.add_argument("fixture", nargs="?", default=None, help="Optional fixture JSON")
        p.add_argument("--out", type=Path, default=Path(default_out), help="JSON artifact output")
        p.add_argument("--publish-evidence", type=Path, default=None,
                       help="Optional tracked evidence output (.md or .json)")
        p.add_argument("--public-copy", action="store_true",
                       help="Mark the evidence as allowed for public copy")

    r1_p = menhir_sub.add_parser("r1", help="Run Menhir R1 hybrid retrieval ladder")
    add_menhir_run_args(r1_p, "results/r1_run.json")
    r1_p.add_argument("--k", type=int, default=5)

    r2_p = menhir_sub.add_parser("r2-facet", help="Run Menhir R2 facet retrieval ladder")
    add_menhir_run_args(r2_p, "results/facet_run.json")
    r2_p.add_argument("--no-traces", action="store_true")
    r2_p.add_argument("--facet-scope", choices=["all", "regular"], default="all")

    r3_p = menhir_sub.add_parser("r3-belief", help="Run Menhir R3 belief/currentness ladder")
    add_menhir_run_args(r3_p, "results/r3_run.json")

    oracle_p = menhir_sub.add_parser("oracle", help="Run Menhir oracle combiner ladder")
    add_menhir_run_args(oracle_p, "results/oracle_run.json")
    oracle_p.add_argument("--no-traces", action="store_true")

    intent_p = menhir_sub.add_parser("intent", help="Run Menhir intent routing ladder")
    add_menhir_run_args(intent_p, "results/intent_run.json")

    l4_p = menhir_sub.add_parser("l4-artifacts", help="Run Menhir L4 institutional artifact ladder")
    add_menhir_run_args(l4_p, "results/l4_run.json")

    r5_p = menhir_sub.add_parser("r5-structure-temporal", help="Run Menhir R5 structure-temporal ladder")
    add_menhir_run_args(r5_p, "results/r5_run.json")
    r5_p.add_argument("--k", type=int, default=3)

    lme_p = menhir_sub.add_parser("longmemeval", help="Run LongMemEval Menhir memory benchmark")
    lme_p.add_argument("--offline-fixture", type=Path, default=Path("fixtures/longmemeval_sample.json"))
    lme_p.add_argument("--menhir-url", default=None)
    lme_p.add_argument("--limit", type=int, default=None)
    lme_p.add_argument("--subset", default=None)
    lme_p.add_argument("--model", default=MODEL)
    lme_p.add_argument("--output-dir", type=Path, default=Path("results"))
    lme_p.add_argument("--out", type=Path, default=None)
    lme_p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    _add_evidence_args(lme_p)

    extract_p = menhir_sub.add_parser("extraction-models", help="Run Menhir extraction model benchmark")
    extract_p.add_argument("--repeats", type=int, default=1)
    extract_p.add_argument("--targets-file", type=Path, default=None)
    extract_p.add_argument("--exclude", default=None)
    extract_p.add_argument("--all", action="store_true")
    extract_p.add_argument("--out", type=Path, default=None)
    _add_evidence_args(extract_p)

    # ---- report subcommand ----
    report_p = subparsers.add_parser("report", help="Generate BENCHMARKS.md from results/")
    report_p.add_argument("--out", type=Path, default=Path("BENCHMARKS.md"),
                          help="Output file path (default: BENCHMARKS.md)")
    report_p.add_argument("--results-dir", type=Path, default=Path("results"),
                          help="Directory containing results/ JSON files")
    report_p.add_argument("--format", choices=["markdown"], default="markdown",
                          help="Output format (default: markdown)")

    args = parser.parse_args(argv)
    args.command_text = "archolith-bench" + (f" {' '.join(raw_argv)}" if raw_argv else "")

    if not args.suite:
        parser.print_help()
        sys.exit(1)

    # Presets are resolved here rather than in _run_harness because deciding
    # whether a flag was passed explicitly needs the parser's defaults.
    if args.suite == "harness":
        if getattr(args, "list_presets", False):
            print(format_preset_list())
            return
        if getattr(args, "preset", None):
            try:
                preset = get_preset(args.preset)
            except KeyError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
            try:
                applied = apply_preset(preset, args, harness_p)
            except PresetConflict as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)
            print(f"Preset {preset.name}: {preset.description}")
            if applied:
                print(f"  set: {', '.join(applied)}")
            if preset.requires:
                print(f"  this preset still needs: {' '.join(preset.requires)}")

    if args.suite == "proxy":
        _run_proxy(args)
    elif args.suite == "filter":
        _run_filter(args)
    elif args.suite == "stack":
        _run_stack(args)
    elif args.suite == "audit":
        _run_audit(args)
    elif args.suite == "industry":
        _run_industry(args)
    elif args.suite == "harness":
        _run_harness(args)
    elif args.suite == "dashboard":
        _run_dashboard(args)
    elif args.suite == "extraction-bench":
        _run_extraction_bench(args)
    elif args.suite == "ports":
        _run_ports(args)
    elif args.suite == "menhir":
        _run_menhir(args)
    elif args.suite == "report":
        _run_report(args)
    else:
        print(f"Unknown suite: {args.suite}")
        sys.exit(1)


def _run_proxy(args: argparse.Namespace) -> None:
    from .suites.proxy import run_benchmark, run_experiment

    if args.list:
        print("Available scenarios:")
        for p in list_scenarios():
            s = Scenario.from_file(p)
            print(f"  {p.name:<25} {s.name:<15} {len(s.turns)} turns  "
                  f"{len(s.fact_probes)} probes  {s.description[:60]}")
        return

    api_key = args.api_key or API_KEY
    pricing = _load_pricing(args)

    arm_names = [a.strip() for a in args.arms.split(",")]
    for arm_name in arm_names:
        if arm_name not in ARMS:
            print(f"ERROR: Unknown arm '{arm_name}'. Available: {', '.join(ARMS)}", file=sys.stderr)
            sys.exit(1)

    scenarios: list[Scenario] = []
    if args.all:
        scenarios = [Scenario.from_file(p) for p in list_scenarios()]
    elif args.scenario:
        scenarios = [Scenario.from_file(args.scenario)]
    else:
        print("ERROR: Specify --scenario <file> or --all", file=sys.stderr)
        sys.exit(1)

    budgets: list[int | None] = [args.budget]
    if args.budgets:
        budgets = [int(b.strip()) for b in args.budgets.split(",")]

    print(f"Proxy suite: {len(scenarios)} scenario(s) x {len(arm_names)} arm(s) x {len(budgets)} budget(s)")
    print(f"  Proxy:  {args.proxy}")
    print(f"  Direct: {args.direct}")
    print(f"  Model:  {args.model}")
    print(f"  Arms:   {', '.join(arm_names)}")
    if pricing:
        print(f"  Pricing: {pricing.provider} (input={pricing.input_full}/M, "
              f"cache_hit={pricing.input_cache_hit}/M, output={pricing.output}/M)")

    needs_proxy = any(ARMS[a]["proxy_enabled"] for a in arm_names)
    if needs_proxy:
        health = check_proxy_health(args.proxy)
        if health is None:
            print(f"ERROR: Cannot reach proxy at {args.proxy}. Is it running?", file=sys.stderr)
            sys.exit(1)
        print(f"  Proxy health: {health}")
        if health.get("graph") != "connected":
            print("WARNING: Proxy graph not connected -- assembly won't fire")

    if not api_key:
        print("ERROR: Set UPSTREAM_API_KEY in .env or pass --api-key", file=sys.stderr)
        sys.exit(1)
    print(f"  API key: ...{api_key[-8:]}")

    config_overrides = None
    if args.config:
        try:
            config_overrides = json.loads(args.config)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid --config JSON: {e}", file=sys.stderr)
            sys.exit(1)

    if args.experiment:
        run_experiment(
            experiment_name=args.experiment,
            scenarios=scenarios,
            arms=arm_names,
            budgets=budgets,
            proxy_url=args.proxy,
            direct_url=args.direct,
            model=args.model,
            output_dir=args.output_dir,
            resume=args.resume,
            api_key=api_key,
            max_turns=args.turns,
            pricing=pricing,
            poll_interval_s=args.poll_interval,
        )
        _publish_cli_evidence(
            args,
            title=f"Proxy experiment evidence: {args.experiment}",
            product="archolith-context",
            ability="context curation and continuity",
            fixture_or_live_source=str(args.output_dir / "experiments" / args.experiment),
            model_provider=args.model,
            environment_caveats=[
                "Experiment mode writes detailed per-run artifacts under the experiment directory.",
                "Public launch copy requires current tracked launch configuration evidence.",
            ],
            metric_rows=[],
            artifact={
                "experiment": args.experiment,
                "scenarios": [s.name for s in scenarios],
                "arms": arm_names,
                "budgets": budgets,
                "output_dir": str(args.output_dir),
            },
        )
        return

    if config_overrides:
        from .core.api import apply_arm_config
        import httpx
        with httpx.Client() as client:
            apply_arm_config(client, args.proxy, config_overrides)

    all_results = []
    for arm_name in arm_names:
        for scenario in scenarios:
            for budget in budgets:
                print(f"\n{'#'*70}")
                print(f"  Running: {scenario.name} / {arm_name} @ budget={budget or 'default'}")
                print(f"  {scenario.description}")
                print(f"{'#'*70}")

                data = run_benchmark(
                    scenario, arm_name, args.proxy, args.direct, args.model,
                    budget, args.output_dir, args.resume, api_key=api_key,
                    max_turns=args.turns,
                    run_probes=not args.no_probes,
                    run_restart=not args.no_restart,
                    pricing=pricing,
                    collapse_abort=not args.no_collapse_abort,
                    poll_interval_s=args.poll_interval,
                )
                print_summary(data)
                save_results(data, args.output_dir)
                all_results.append(data)

    if len(all_results) > 1:
        print_cross_scenario_summary(all_results)

    _publish_cli_evidence(
        args,
        title="Proxy suite evidence",
        product="archolith-context",
        ability="context curation and continuity",
        fixture_or_live_source=str(args.scenario or "scenarios/"),
        model_provider=args.model,
        environment_caveats=[
            "Requires a running archolith-context proxy and upstream provider.",
            "Public launch copy requires current launch config and HEADLINE-NUMBERS.md approval.",
        ],
        metric_rows=_proxy_metric_rows(all_results),
        artifact={"runs": all_results},
    )


def _run_filter(args: argparse.Namespace) -> None:
    from .suites.filter import print_filter_summary, run_filter_suite

    corpora_dir = args.corpora
    if not corpora_dir.exists():
        print(f"ERROR: Corpora directory not found: {corpora_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Filter suite: scanning {corpora_dir}")
    _, data = run_filter_suite(corpora_dir=corpora_dir, output_dir=args.output_dir)
    print_filter_summary(data)

    if args.format == "markdown":
        md_path = args.output_dir / "filter_results.md"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Filter Suite Results\n\n")
            f.write("| Category | Samples | Raw Tokens | Filtered | Savings |\n")
            f.write("|----------|---------|------------|----------|---------|\n")
            for cat in data.get("per_category", []):
                f.write(f"| {cat['category']} | {cat['samples']} | {cat['total_raw_tokens']:,} "
                        f"| {cat['total_filtered_tokens']:,} | {cat['savings_ratio']:.1%} |\n")
            f.write(f"| **Total** | {data['total_samples']} | {data['total_raw_tokens']:,} "
                    f"| {data['total_filtered_tokens']:,} | **{data['aggregate_savings_ratio']:.1%}** |\n")
        print(f"  Markdown report saved to {md_path}")

    _publish_cli_evidence(
        args,
        title="Filter suite evidence",
        product="archolith-filter",
        ability="tool-output compression",
        fixture_or_live_source=str(corpora_dir),
        model_provider="offline archolith-filter corpus",
        environment_caveats=[
            "Corpus provenance must be reviewed before using compression figures in public copy.",
            "Launch copy requires current tracked aggregate evidence.",
        ],
        metric_rows=_filter_metric_rows(data),
        artifact=data,
    )


def _run_stack(args: argparse.Namespace) -> None:
    from .suites.stack import run_stack_suite

    if args.list:
        print("Available scenarios:")
        for p in list_scenarios():
            s = Scenario.from_file(p)
            print(f"  {p.name:<25} {s.name:<15} {len(s.turns)} turns  "
                  f"{len(s.fact_probes)} probes  {s.description[:60]}")
        return

    api_key = args.api_key or API_KEY
    if not api_key:
        print("ERROR: Set UPSTREAM_API_KEY in .env or pass --api-key", file=sys.stderr)
        sys.exit(1)

    pricing = _load_pricing(args)

    scenarios: list[Scenario] = []
    if args.all:
        scenarios = [Scenario.from_file(p) for p in list_scenarios()]
    elif args.scenario:
        scenarios = [Scenario.from_file(args.scenario)]
    else:
        print("ERROR: Specify --scenario <file> or --all", file=sys.stderr)
        sys.exit(1)

    health = check_proxy_health(args.proxy)
    if health is None:
        print(f"ERROR: Cannot reach proxy at {args.proxy}. Is it running?", file=sys.stderr)
        sys.exit(1)
    print(f"  Proxy health: {health}")

    print(f"Stack suite: {len(scenarios)} scenario(s) x 4 arms (direct/filter_only/proxy_only/proxy_plus_filter)")
    if pricing:
        print(f"  Pricing: {pricing.provider} (input={pricing.input_full}/M, "
              f"cache_hit={pricing.input_cache_hit}/M, output={pricing.output}/M)")

    all_arm_results = run_stack_suite(
        scenarios=scenarios,
        proxy_url=args.proxy,
        direct_url=args.direct,
        model=args.model,
        budget=args.budget,
        output_dir=args.output_dir,
        api_key=api_key,
        max_turns=args.turns,
        run_probes=not args.no_probes,
        run_restart=not args.no_restart,
        pricing=pricing,
        poll_interval_s=args.poll_interval,
    )

    print_four_way_table(all_arm_results)

    _publish_cli_evidence(
        args,
        title="Stack suite evidence",
        product="archolith-context",
        ability="context curation and tool-output compression",
        fixture_or_live_source=str(args.scenario or "scenarios/"),
        model_provider=args.model,
        environment_caveats=[
            "Requires running proxy/filter stack and upstream provider.",
            "Public launch copy requires current launch config and HEADLINE-NUMBERS.md approval.",
        ],
        metric_rows=_proxy_metric_rows(all_arm_results),
        artifact={"runs": all_arm_results},
    )


def _run_audit(args: argparse.Namespace) -> None:
    from .suites.audit import print_audit_summary, run_audit_comparison

    before_path = args.before
    after_path = args.after

    if not before_path.exists():
        print(f"ERROR: Before report not found: {before_path}", file=sys.stderr)
        sys.exit(1)
    if not after_path.exists():
        print(f"ERROR: After report not found: {after_path}", file=sys.stderr)
        sys.exit(1)

    print("Audit suite: comparing before vs after")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")

    result = run_audit_comparison(before_path, after_path, output_dir=args.output_dir)
    print_audit_summary(result)

    if args.format == "json":
        pass
    elif args.format == "markdown":
        md_path = args.output_dir / "audit_comparison.md"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Audit Comparison: Before vs After\n\n")
            f.write("| Server | Before | After | Change | Pct | Status |\n")
            f.write("|--------|--------|-------|--------|-----|--------|\n")
            for s in result.get("per_server", []):
                f.write(f"| {s['server']} | {s['before_tokens']:,} | {s['after_tokens']:,} "
                        f"| {s['token_change']:+,} | {s['token_change_pct']:+.1f}% | {s['status']} |\n")
            f.write(f"| **Total** | {result['before_total_tokens']:,} | {result['after_total_tokens']:,} "
                    f"| {result['token_reduction']:+,} | {result['token_reduction_pct']:+.1f}% | - |\n")
            f.write(f"\n**Waste reduction:** {result['waste_reduction']:,} tokens "
                    f"({result['waste_reduction_pct']:.1f}%)\n")
        print(f"  Markdown report saved to {md_path}")

    _publish_cli_evidence(
        args,
        title="MCP audit evidence",
        product="archolith-skree",
        ability="MCP token-waste reduction",
        fixture_or_live_source=f"before={before_path}; after={after_path}",
        model_provider="offline audit logs",
        environment_caveats=[
            "Fixture audit numbers are not public launch claims.",
            "Public copy requires real before/after logs from the target environment.",
        ],
        metric_rows=_audit_metric_rows(result),
        artifact=result,
    )


def _run_industry(args: argparse.Namespace) -> None:
    from .core.industry import write_industry_benchmarks
    from .suites.industry import print_industry_summary, run_industry_suite

    data = run_industry_suite(
        output_dir=args.output_dir,
        product=args.product,
        suite=args.benchmark_suite,
        launch_only=args.launch_only,
    )
    print_industry_summary(data)

    if args.out:
        write_industry_benchmarks(
            args.out,
            product=args.product,
            suite=args.benchmark_suite,
            launch_only=args.launch_only,
            output_format=args.format,
        )
        print(f"  Written to {args.out}")
    else:
        suffix = "json" if args.format == "json" else "md"
        default_out = args.output_dir / f"industry_benchmarks.{suffix}"
        print(f"  Written to {default_out}")

    _publish_cli_evidence(
        args,
        title="Industry benchmark registry evidence",
        product=args.product or "archolith-family",
        ability="launch benchmark coverage",
        fixture_or_live_source=str(args.output_dir),
        model_provider="offline registry",
        environment_caveats=[
            "Registry evidence documents coverage state; it is not a performance result.",
            "Public copy depends on each benchmark row's launch claim rule.",
        ],
        metric_rows=[
            {
                "product": row.get("product"),
                "suite": row.get("suite"),
                "benchmark": row.get("name"),
                "status": row.get("status"),
            }
            for row in data.get("benchmarks", [])
        ],
        artifact=data,
    )


def _run_phase3_harness(args: argparse.Namespace, adapter) -> None:  # noqa: ANN001
    """Drive the Menhir Phase 3 View-consolidation benchmark (live, black-box, throwaway menhir).

    Not an A/B: posts :TurnEvidence, triggers consolidation, and validates Views / abstentions /
    supersession / idempotence. Requires --menhir-url (throwaway) and --confirm-menhir-reset.
    """
    from .harness import (
        Phase3MenhirClient,
        StubPhase3Client,
        assert_not_production,
        phase3_result_to_dict,
        run_phase3,
        run_scenario_suite,
        scenario_result_to_dict,
        suite_verdict,
        write_phase3_evidence,
    )

    # Offline smoke: --offline-fixture (any value) runs the FULL driver + scenario suite + report
    # against a deterministic in-memory stub — no menhir, no Neo4j, no network. CI-safe; it guards
    # the harness itself, NOT menhir (it cannot reveal server-side extraction defects/stochasticity).
    offline = args.offline_fixture is not None
    if offline:
        print(f"Harness: {adapter.name} — OFFLINE smoke (deterministic stub; not a live menhir run)")
        client = StubPhase3Client()
        source_label = "offline-stub"
    else:
        if not args.menhir_url:
            print("ERROR: menhir-phase3 needs --menhir-url (a throwaway menhir instance), "
                  "or --offline-fixture for a network-free smoke", file=sys.stderr)
            sys.exit(1)
        assert_not_production(args.menhir_url)
        if not args.confirm_menhir_reset:
            print("ERROR: menhir-phase3 writes to and tears down a throwaway namespace; "
                  "pass --confirm-menhir-reset", file=sys.stderr)
            sys.exit(1)
        print(f"Harness: {adapter.name} — live Phase 3 consolidation against {args.menhir_url}")
        # Menhir's bearer (agent tier) is NOT the upstream LLM key. Prefer the menhir key envs so the
        # run works against an auth-enabled instance; fall back to API_KEY (empty => auth-disabled OK).
        menhir_key = os.getenv("MENHIR_AGENT_KEY") or os.getenv("MENHIR_API_KEY") or API_KEY
        print(f"  auth: {'bearer set' if menhir_key else 'no key (expects an auth-disabled instance)'}")
        client = Phase3MenhirClient(args.menhir_url, api_key=menhir_key)
        source_label = args.menhir_url
    try:
        result = run_phase3(
            client,
            cases=adapter.cases(),
            k=3,
            # Offline stub has no real teardown, so reset is implicitly safe; live requires the flag.
            reset_confirmed=True if offline else args.confirm_menhir_reset,
            menhir_url=source_label,
            # Phase 3 runs the consolidation LLM SERVER-SIDE (menhir's personal-memory model);
            # the harness --model (an upstream chat model) is not the model under test here.
            model="offline-stub" if offline else "menhir-personal-memory (server-side)",
        )
        # Expanded consumer scenarios (ambiguous/negative correction, currency SUM, count-vs-spend,
        # multi-namespace). Gate scenarios fold into the exit code; characterization ones do not.
        scenario_results = run_scenario_suite(
            client, base_namespace=f"phase3-scn-{result.namespace[-8:]}", k=3,
        )
    finally:
        client.close()

    print(f"\n{adapter.name}: verdict={'PASS' if result.verdict else 'FAIL'}  "
          f"namespace={result.namespace}")
    for cr in result.cases:
        tail = f" — {cr.notes}" if cr.notes else ""
        print(f"  [{'PASS' if cr.passed else 'FAIL'}] {cr.case_id} ({cr.kind}){tail}")
    m = result.metrics
    print(f"  views_current={m['views_current']} superseded={m['views_superseded']} "
          f"abstentions={m['abstentions']} supersessions={m['supersessions_applied']} "
          f"wrong_view_writes={m['wrong_view_writes']} silent_abstentions={m['silent_abstentions']} "
          f"duplicate_writes={m['duplicate_writes_on_rerun']}")

    scenarios_gate_pass = suite_verdict(scenario_results)
    print(f"\n  scenario suite ({len(scenario_results)}): "
          f"gate={'PASS' if scenarios_gate_pass else 'FAIL'}")
    for sr in scenario_results:
        tag = "gate" if sr.gate else "characterization"
        status = "PASS" if sr.passed else ("FAIL" if sr.gate else "DIVERGES")
        print(f"  [{status}] {sr.scenario_id} ({tag}) — {sr.description}")
        for ar in sr.assertions:
            if not ar.passed:
                print(f"        - {ar.label}: {ar.detail}")

    scenario_suite_data = {
        "gate_passed": scenarios_gate_pass,
        "scenarios": [scenario_result_to_dict(sr) for sr in scenario_results],
    }
    suffix = "json" if args.format == "json" else "md"
    out_path = args.out or (args.output_dir / f"harness_{adapter.benchmark_id}.{suffix}")
    write_phase3_evidence(result, out_path, output_format=args.format,
                          scenario_suite=scenario_suite_data)
    print(f"  Evidence written to {out_path}")

    result_data = phase3_result_to_dict(result, scenario_suite_data)
    _publish_cli_evidence(
        args,
        title=f"Harness evidence: {adapter.name}",
        product="menhir",
        ability="Phase 3 personal-memory View consolidation",
        fixture_or_live_source=source_label,
        model_provider=args.model,
        environment_caveats=[
            "OFFLINE stub smoke exercises the harness only — not a live menhir result."
            if offline else
            "Phase 3 is a live black-box integration test against a throwaway menhir + Neo4j.",
            "The producer (host lifecycle triage) is frozen; fixtures encode its verdicts.",
            "Characterization scenarios document current behavior and do not gate.",
            "Public launch copy requires current tracked launch evidence.",
        ],
        metric_rows=[
            {"metric": key, "value": value}
            for key, value in m.items()
            if not isinstance(value, (list, dict))
        ],
        artifact=result_data,
    )

    if not result.verdict or not scenarios_gate_pass:
        sys.exit(1)


def _run_scalar_state_harness(args: argparse.Namespace, adapter) -> None:  # noqa: ANN001
    """Drive the Menhir ScalarStateView e2e benchmark (live, throwaway menhir + scalar scheduler ON).

    Ingests known typed-scalar episodes over HTTP, waits for the BACKGROUND consolidation scheduler
    to materialize `scalar_state` Views, and verifies invariants over bolt (shadow Views are not
    recall-visible). Requires --menhir-url (throwaway, scheduler on) + --neo4j-uri (same throwaway
    Neo4j) + --confirm-menhir-reset. See benchmarks/RUNBOOK-scalar-state-e2e.md for the launch profile.
    """
    from .harness import (
        SCALAR_FIXTURE_SETS,
        HttpMenhirClient,
        ScalarBoltReader,
        StubScalarStateClient,
        assert_not_production,
        run_scalar_state,
        scalar_state_result_to_dict,
        write_scalar_state_evidence,
    )

    fixture_name = getattr(args, "scalar_fixture", "default")
    cases = SCALAR_FIXTURE_SETS[fixture_name]()

    # Offline smoke: --offline-fixture runs the FULL driver + invariants + report against a
    # deterministic in-memory stub (both ingest client and bolt reader) — no menhir, no Neo4j, no
    # network. CI-safe; guards the harness itself, NOT menhir (it cannot reveal real perception defects).
    offline = args.offline_fixture is not None
    if offline:
        print(f"Harness: {adapter.name} — OFFLINE smoke (deterministic stub; not a live menhir run)")
        stub = StubScalarStateClient()
        client = stub
        bolt = stub
        source_label = "offline-stub"
        neo4j_label = "offline-stub"
    else:
        if not args.menhir_url:
            print("ERROR: menhir-scalar-state needs --menhir-url (a throwaway menhir with the scalar "
                  "scheduler ON), or --offline-fixture for a network-free smoke", file=sys.stderr)
            sys.exit(1)
        assert_not_production(args.menhir_url)
        if not args.confirm_menhir_reset:
            print("ERROR: menhir-scalar-state writes to and tears down a throwaway namespace; "
                  "pass --confirm-menhir-reset", file=sys.stderr)
            sys.exit(1)
        print(f"Harness: {adapter.name} — live scalar-state e2e against {args.menhir_url}")
        # A throwaway scalar-state menhir is keyless (or uses a real MENHIR_*_KEY). Do NOT fall back
        # to API_KEY here: that is the upstream LLM proxy key (UPSTREAM_API_KEY), never a menhir
        # bearer, so sending it makes a keyless throwaway reject the ingest with 401.
        menhir_key = os.getenv("MENHIR_AGENT_KEY") or os.getenv("MENHIR_API_KEY") or ""
        print(f"  auth: {'bearer set' if menhir_key else 'no key (expects an auth-disabled instance)'}")
        print(f"  bolt: {args.neo4j_uri} (throwaway; refuses prod port 7687)")
        client = HttpMenhirClient(args.menhir_url, api_key=menhir_key)
        # ScalarBoltReader.__post_init__ refuses a prod-looking bolt URI before connecting.
        bolt = ScalarBoltReader(
            args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password
        )
        source_label = args.menhir_url
        neo4j_label = args.neo4j_uri

    try:
        print(f"  fixture: {fixture_name} ({len(cases)} cases)")
        result = run_scalar_state(
            client,
            bolt,
            cases=cases,
            reset_confirmed=True,
            cleanup=not getattr(args, "scalar_no_cleanup", False),
            max_wait_s=0.0 if offline else args.scalar_max_wait_s,
            poll_interval_s=0.0 if offline else 3.0,
            menhir_url=source_label,
            neo4j_uri=neo4j_label,
            # Perception runs the consolidation LLM SERVER-SIDE (menhir's personal-memory model);
            # the harness --model is not the model under test here.
            model="offline-stub" if offline else "menhir-personal-memory (server-side)",
        )
    finally:
        client.close()
        if not offline:
            bolt.close()

    print(f"\n{adapter.name}: verdict={'PASS' if result.verdict else 'FAIL'}  "
          f"namespace={result.namespace}")
    for cr in result.cases:
        status = "MATCH" if cr.matched else ("MISS" if cr.outcome == "view" else "FAIL")
        tail = f" — {cr.notes}" if cr.notes else ""
        print(f"  [{status}] {cr.case_id} ({cr.outcome}){tail}")
    inv = result.invariants
    m = result.metrics
    print(f"  views_current={inv['views_current']} assertions={inv['assertions_current']} "
          f"dup_keys={inv['duplicate_current_keys']} dup_slots={inv['duplicate_slots']} "
          f"non_agent_tiers={inv['non_agent_tiers'] or 'none'} default_leak={inv['default_silo_leak']}")
    print(f"  view_slots_committed={m['view_slots_committed']}/{m['expected_view_slots']} "
          f"advisories_pending={m['advisories_pending']} controls_clean={m['controls_clean']} "
          f"waited_s={m['waited_s']}")
    for w in result.warnings:
        print(f"  ! {w}")

    suffix = "json" if args.format == "json" else "md"
    out_path = args.out or (args.output_dir / f"harness_{adapter.benchmark_id}.{suffix}")
    write_scalar_state_evidence(result, out_path, output_format=args.format)
    print(f"  Evidence written to {out_path}")

    result_data = scalar_state_result_to_dict(result)
    _publish_cli_evidence(
        args,
        title=f"Harness evidence: {adapter.name}",
        product="menhir",
        ability="ScalarStateView typed-scalar personal-memory consolidation",
        fixture_or_live_source=source_label,
        model_provider=args.model,
        environment_caveats=[
            "OFFLINE stub smoke exercises the harness only — not a live menhir result."
            if offline else
            "ScalarStateView is a live e2e integration test against a throwaway menhir + Neo4j; "
            "perception is LLM-driven and stochastic, so the verdict gates on structural invariants, "
            "not exact values.",
            "Shadow build: scalar_state Views are not recall-visible and are verified over bolt.",
            "Requires the throwaway menhir launched with the scalar scheduler ON "
            "(MENHIR_BENCHMARK_MODE=0, CONSOLIDATION_ENABLED=1, SCALAR_STATE_ENABLED=1).",
            "Public launch copy requires current tracked launch evidence.",
        ],
        metric_rows=[
            {"metric": key, "value": value}
            for key, value in {**inv, **m}.items()
            if not isinstance(value, (list, dict))
        ],
        artifact=result_data,
    )

    if not result.verdict:
        sys.exit(1)


def _run_harness(args: argparse.Namespace) -> None:
    from .harness import (
        ADAPTERS,
        DEFAULT_MEMORY_ARMS,
        HttpMenhirClient,
        LLMJudgeScorer,
        MemoryCheckpoint,
        StubMenhirClient,
        ab_result_to_dict,
        assert_not_production,
        checkpoint_path_for,
        get_adapter,
        is_external,
        is_memory,
        is_phase3,
        is_scalar_state,
        run_ab,
        run_external_ab,
        run_memory_ab,
        write_harness_evidence,
    )

    if args.list_adapters:
        print("Available harness adapters:")
        for bid, adapter in sorted(ADAPTERS.items()):
            if is_phase3(adapter):
                kind = "phase3"
            elif is_scalar_state(adapter):
                kind = "scalar-state"
            elif is_memory(adapter):
                kind = "memory"
            elif is_external(adapter):
                kind = "external-cli"
            else:
                kind = "in-process"
            print(f"  {bid}  ({adapter.name}) [{kind}]")
        return

    if not args.benchmark_id:
        print("ERROR: benchmark_id is required (or use --list)", file=sys.stderr)
        sys.exit(1)

    try:
        adapter = get_adapter(args.benchmark_id)
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if is_phase3(adapter):
        _run_phase3_harness(args, adapter)
        return

    if is_scalar_state(adapter):
        _run_scalar_state_harness(args, adapter)
        return

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    offline = args.offline_fixture is not None
    if offline:
        print(f"Harness: {adapter.name} (OFFLINE fixture {args.offline_fixture})")
    else:
        print(f"Harness: {adapter.name} — running official A/B (arms: {', '.join(arms)})")

    if is_memory(adapter):
        mem_arms = list(DEFAULT_MEMORY_ARMS) if args.arms == "direct,proxy_only,proxy_plus_filter" else arms
        if offline:
            client = StubMenhirClient()
            send_fn = _offline_send_fn
        else:
            if not args.menhir_url:
                print("ERROR: memory benchmarks need --menhir-url (a throwaway menhir instance)",
                      file=sys.stderr)
                sys.exit(1)
            assert_not_production(args.menhir_url)
            menhir_key = _resolve_menhir_bearer_key()
            print(f"  auth: {'bearer set' if menhir_key else 'no key (expects an auth-disabled instance)'}")
            client = HttpMenhirClient(args.menhir_url, api_key=menhir_key)
            send_fn = send_chat
        checkpoint = None
        if getattr(args, "resume", False):
            ckpt_path = checkpoint_path_for(args.output_dir, adapter.benchmark_id, args.model)
            checkpoint = MemoryCheckpoint(ckpt_path)
            done = checkpoint.done_count()
            print(f"  [resume] checkpoint {ckpt_path} ({done} item-results already recorded)")
        score_fn = None
        if getattr(args, "scorer", "containment") == "llm-judge":
            if offline:
                print("ERROR: --scorer llm-judge needs network; not valid with --offline-fixture", file=sys.stderr)
                sys.exit(1)
            judge_key = args.judge_api_key or os.getenv("OPENAI_API_KEY", "")
            if not judge_key:
                print("ERROR: --scorer llm-judge needs an API key (--judge-api-key or OPENAI_API_KEY)", file=sys.stderr)
                sys.exit(1)
            score_fn = LLMJudgeScorer(
                base_url=args.judge_url or os.getenv("UPSTREAM_BASE_URL") or "https://api.openai.com/v1",
                api_key=judge_key,
                model=args.judge_model,
            )
            print(f"  [scorer] llm-judge model={args.judge_model} (LongMemEval-comparable)")
        ab = run_memory_ab(
            adapter,
            arms=mem_arms,
            subset=args.subset,
            limit=args.limit,
            model=args.model,
            client=client,
            send_fn=send_fn,
            recall_limit=args.recall_limit,
            # Mode B answers the question directly from recalled memory; the answer
            # model talks straight to the upstream, NOT the archolith-context proxy
            # (which need not be running for memory benchmarks). Default chat_base_url
            # is PROXY_URL, which would send every answer call to a dead port.
            chat_base_url=DIRECT_URL,
            api_key=API_KEY,
            fixture_path=args.offline_fixture,
            reset_confirmed=args.confirm_menhir_reset,
            dry_run_reset=args.dry_run_menhir_reset,
            recall_only=getattr(args, "recall_only", False),
            namespace_template=getattr(args, "namespace_template", "lme-{question_id}"),
            checkpoint=checkpoint,
            score_fn=score_fn,
        )
    elif is_external(adapter):
        results_fixtures = (
            {arm: args.offline_fixture for arm in arms} if offline else None
        )
        ab = run_external_ab(
            adapter,
            arms=arms,
            subset=args.subset,
            limit=args.limit,
            model=args.model,
            configure_proxy=not offline,
            results_fixtures=results_fixtures,
        )
    else:
        ab = run_ab(
            adapter,
            arms=arms,
            subset=args.subset,
            limit=args.limit,
            model=args.model,
            send_fn=_offline_send_fn if offline else send_chat,
            configure_proxy=not offline,
            fixture_path=args.offline_fixture,
        )

    print(f"\n{adapter.name} A/B ({ab.model}):")
    for arm, r in ab.arms.items():
        print(f"  {arm}: score={r.score:.3f} n={r.n} "
              f"in={r.input_tokens:,} out={r.output_tokens:,} cost=${r.cost_usd:.6f}")
    for arm, d in ab.deltas.items():
        print(f"  delta {arm}: score {d['score_delta']:+.3f}, "
              f"input reduction {d['input_token_reduction_pct']:+.1f}%, "
              f"cost reduction {d['cost_reduction_pct']:+.1f}%")

    suffix = "json" if args.format == "json" else "md"
    out_path = args.out or (args.output_dir / f"harness_{adapter.benchmark_id}.{suffix}")
    write_harness_evidence(ab, out_path, output_format=args.format)
    print(f"  Evidence written to {out_path}")
    ab_data = ab_result_to_dict(ab)
    _publish_cli_evidence(
        args,
        title=f"Harness evidence: {adapter.name}",
        product="menhir" if is_memory(adapter) else "archolith-family",
        ability="persistent memory QA" if is_memory(adapter) else "external benchmark A/B",
        fixture_or_live_source=str(args.offline_fixture) if offline else args.menhir_url or "official benchmark source",
        model_provider=args.model,
        environment_caveats=[
            "Offline fixtures are smoke evidence only.",
            "External benchmark results are middleware deltas, not standalone model scores.",
            "Public launch copy requires current tracked launch evidence.",
        ],
        metric_rows=_harness_metric_rows(ab_data),
        artifact=ab_data,
    )


def _offline_send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
    """Deterministic offline stub for --offline-fixture pipeline smoke tests.

    Always answers "A" with an estimated token count, so the A/B plumbing (load ->
    run -> score -> aggregate -> deltas -> evidence) runs without network access.
    This is a smoke path, not real evidence; real runs use core.api.send_chat.
    """
    prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) // 4
    usage = {"prompt_tokens": max(1, prompt_tokens), "completion_tokens": 1}
    return "A", 0.0, usage


def _run_dashboard(args: argparse.Namespace) -> None:
    from .dashboard import run_dashboard, serve_dashboard

    try:
        if args.serve:
            scalar_reader = None
            if args.scalar_neo4j_uri:
                if not args.scalar_neo4j_password:
                    print(
                        "ERROR: --scalar-neo4j-uri needs --scalar-neo4j-password "
                        "(prefer LME_DASHBOARD_NEO4J_PASSWORD)",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                from .scalar_viewer import ScalarTaskReader

                scalar_reader = ScalarTaskReader(
                    neo4j_uri=args.scalar_neo4j_uri,
                    neo4j_user=args.scalar_neo4j_user,
                    neo4j_password=args.scalar_neo4j_password,
                    telemetry_db=args.scalar_telemetry_db,
                )
            serve_dashboard(
                args.results_dir,
                menhir_url=args.menhir_url,
                host=args.host,
                port=args.port,
                total_items=args.total_items,
                refresh_s=int(args.interval),
                items_n=args.items,
                active_within_s=(args.active_within or None),
                scalar_reader=scalar_reader,
                scalar_default_namespace=args.scalar_task,
            )
        else:
            run_dashboard(
                args.results_dir,
                menhir_url=args.menhir_url,
                interval=args.interval,
                once=args.once,
                total_items=args.total_items,
                active_within_s=(args.active_within or None),
            )
    except KeyboardInterrupt:
        print("\n(dashboard stopped)")


def _run_extraction_bench(args: argparse.Namespace) -> None:
    from .extraction_sim import default_targets, render_results, simulate_model

    targets = default_targets(full=args.all)
    if args.targets_file:
        extra = json.loads(Path(args.targets_file).read_text(encoding="utf-8"))
        for t in extra:
            key = os.getenv(t.get("api_key_env", "")) if t.get("api_key_env") else t.get("api_key")
            if key:
                targets.append({"label": t["label"], "base_url": t["base_url"], "api_key": key, "model": t["model"]})
    if args.exclude:
        pats = [p.strip().lower() for p in args.exclude.split(",") if p.strip()]
        targets = [t for t in targets if not any(p in (t["label"] + t["model"]).lower() for p in pats)]
    if not targets:
        print("ERROR: no targets with available keys. Set OPENAI_API_KEY / GROQ_API_KEY / GEMINI_API_KEY / "
              "CEREBRAS_API_KEY (fast providers auto-enable when their key is present).", file=sys.stderr)
        sys.exit(1)

    print(f"Simulating backend extraction over {len(targets)} model(s), repeats={args.repeats} "
          f"(~{3 * 5 * args.repeats} calls each)...")
    results = []
    for t in targets:
        print(f"  probing {t['label']} ...", flush=True)
        results.append(simulate_model(t["label"], t["base_url"], t["api_key"], t["model"], repeats=args.repeats))

    report = render_results(results)
    print("\n" + report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nEvidence written to {args.out}")
    _publish_cli_evidence(
        args,
        title="Menhir extraction model evidence",
        product="menhir",
        ability="extraction model selection",
        fixture_or_live_source="built-in extraction corpus",
        model_provider="provider API targets",
        environment_caveats=[
            "Requires provider API keys and current provider availability.",
            "Provider/model recommendations require fresh tracked evidence before public copy.",
        ],
        metric_rows=[
            {
                "model": r.label,
                "mode": r.mode,
                "episodes_per_s": r.throughput_eps,
                "call_p50_s": r.call_p50,
                "call_p95_s": r.call_p95,
                "valid_json_rate": r.valid_json_rate,
                "entity_recall": r.mean_entity_recall,
                "fact_recall": r.mean_fact_recall,
                "cost_per_1k_episodes": r.cost_per_1k_episodes(),
                "error": r.error,
            }
            for r in results
        ],
        artifact={
            "report": report,
            "results": [
                {
                    "label": r.label,
                    "model": r.model,
                    "mode": r.mode,
                    "base_url": r.base_url,
                    "episodes": r.episodes,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_input_tokens": r.cached_input_tokens,
                    "wall_clock_s": r.wall_clock_s,
                    "rate_limit_hits": r.rate_limit_hits,
                    "rate_limit_wait_s": r.rate_limit_wait_s,
                    "error": r.error,
                    "last_call_error": r.last_call_error,
                }
                for r in results
            ],
        },
    )


def _run_ports(args: argparse.Namespace) -> None:
    from .procindex import discover, filter_relevant, render

    entries = discover()
    if args.as_json:
        rows = filter_relevant(entries, show_all=args.show_all)
        print(json.dumps([{"port": e.port, "pid": e.pid, "label": e.label, "cmd": e.cmd} for e in rows], indent=2))
    else:
        print(render(entries, show_all=args.show_all))


def _run_menhir(args: argparse.Namespace) -> None:
    from .core.capabilities import render_capabilities_markdown
    from .harness import assert_not_production
    from .suites import menhir as menhir_suite

    cmd = getattr(args, "menhir_command", None)
    if not cmd:
        print("ERROR: specify a menhir command (use `archolith-bench menhir --help`)", file=sys.stderr)
        sys.exit(1)

    if cmd == "list":
        print(render_capabilities_markdown(product="menhir"))
        return
    if cmd == "smoke":
        menhir_suite.run_smoke(args.output_dir, publish_dir=args.publish_dir)
        return
    if cmd == "bootstrap-hygiene":
        if not args.offline:
            if not args.menhir_url:
                print("ERROR: live bootstrap-hygiene requires --menhir-url", file=sys.stderr)
                sys.exit(1)
            assert_not_production(args.menhir_url)
            if not args.confirm_menhir_reset:
                print(
                    "ERROR: live bootstrap-hygiene requires --confirm-menhir-reset",
                    file=sys.stderr,
                )
                sys.exit(1)
        artifact = menhir_suite.run_bootstrap_hygiene(
            args.fixture,
            menhir_url=None if args.offline else args.menhir_url,
            api_key="" if args.offline else _resolve_menhir_bearer_key(),
        )
        menhir_suite.print_summary(cmd, artifact)
        menhir_suite.write_json_artifact(artifact, args.out)
        print(f"\nwrote artifact: {args.out}")
        if not artifact.get("passed"):
            sys.exit(2)
        return
    if cmd == "longmemeval":
        harness_args = argparse.Namespace(
            benchmark_id="longmemeval-menhir",
            list_adapters=False,
            arms="direct,proxy_only,proxy_plus_filter",
            subset=args.subset,
            limit=args.limit,
            model=args.model,
            offline_fixture=None if args.menhir_url else args.offline_fixture,
            menhir_url=args.menhir_url,
            resume=False,
            scorer="containment",
            judge_model="gpt-4o-mini",
            judge_url=None,
            judge_api_key=None,
            confirm_menhir_reset=False,
            dry_run_menhir_reset=False,
            recall_only=bool(args.menhir_url),
            namespace_template="lme-{question_id}",
            format=args.format,
            output_dir=args.output_dir,
            out=args.out,
            publish_evidence=args.publish_evidence,
            public_copy=args.public_copy,
            command_text=getattr(args, "command_text", "archolith-bench menhir longmemeval"),
        )
        _run_harness(harness_args)
        return
    if cmd == "extraction-models":
        _run_extraction_bench(args)
        return

    runners = {
        "r1": ("Menhir R1 hybrid retrieval", "hybrid retrieval tuning", menhir_suite.run_r1),
        "r2-facet": ("Menhir R2 facet retrieval", "facet retrieval", menhir_suite.run_r2_facet),
        "r3-belief": ("Menhir R3 belief/currentness", "belief/currentness", menhir_suite.run_r3_belief),
        "oracle": ("Menhir oracle combiner", "oracle combiner", menhir_suite.run_oracle),
        "intent": ("Menhir intent routing", "intent-aware retrieval", menhir_suite.run_intent),
        "l4-artifacts": ("Menhir L4 institutional artifacts", "institutional artifact memory",
                         menhir_suite.run_l4_artifacts),
        "r5-structure-temporal": ("Menhir R5 structure-temporal", "structure-temporal blast radius",
                                  menhir_suite.run_r5_structure_temporal),
    }
    if cmd not in runners:
        print(f"ERROR: unknown menhir command: {cmd}", file=sys.stderr)
        sys.exit(1)

    title, ability, runner = runners[cmd]
    fixture = Path(args.fixture) if getattr(args, "fixture", None) else None
    kwargs: dict[str, object] = {}
    if cmd == "r1":
        kwargs["k"] = args.k
    elif cmd == "r2-facet":
        kwargs["no_traces"] = args.no_traces
        kwargs["facet_scope"] = args.facet_scope
    elif cmd == "oracle":
        kwargs["no_traces"] = args.no_traces
    elif cmd == "r5-structure-temporal":
        kwargs["k"] = args.k

    artifact = runner(fixture, **kwargs)
    menhir_suite.print_summary(cmd, artifact)
    menhir_suite.write_json_artifact(artifact, args.out)
    print(f"\nwrote artifact: {args.out}")

    if args.publish_evidence:
        path = menhir_suite.publish_menhir_evidence(
            title=title,
            command=getattr(args, "command_text", f"archolith-bench menhir {cmd}"),
            ability=ability,
            fixture_or_live_source=str(fixture or artifact.get("fixture", "default fixture")),
            artifact=artifact,
            out_path=args.publish_evidence,
            public_copy_allowed=args.public_copy,
        )
        print(f"published evidence: {path}")


def _run_report(args: argparse.Namespace) -> None:
    from .core.report import write_benchmarks_md

    results_dir = args.results_dir
    out_path = args.out

    print(f"Generating BENCHMARKS.md from {results_dir}")
    write_benchmarks_md(results_dir, out_path)
    print(f"  Written to {out_path}")


if __name__ == "__main__":
    main()
