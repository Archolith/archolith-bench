"""CLI entry for the orchestrator so it can be called as ``python -m
archolith_bench.ci.orchestrator --pr 123 ...``"""

from __future__ import annotations

import argparse
import sys

from .orchestrator import OrchestratorConfig, run_bench_for_pr


def main() -> int:
    p = argparse.ArgumentParser(prog="archolith-bench-ci", description="Run the recall benchmark for a PR")
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--pr-author", default="unknown")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--menhir-port", type=int, default=8090)
    p.add_argument("--proxy-port", type=int, default=8765)
    p.add_argument("--neo4j-uri", default="bolt://localhost:7689")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default="lmedata123")
    p.add_argument("--upstream", default="https://api.openai.com")
    p.add_argument("--max-calls", type=int, default=200)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--max-seconds", type=float, default=900.0)
    p.add_argument("--judge-model", default="gpt-6-luna")
    p.add_argument("--questions-per-type", type=int, default=20)
    p.add_argument("--baseline-file", default="benchmarks/longmemeval-baseline.json")
    p.add_argument("--runs-dir", default=".bench/runs")
    p.add_argument("--menhir-dir")
    p.add_argument("--menhir-python")
    p.add_argument("--menhir-startup-seconds", type=float, default=60.0)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-menhir-start", action="store_true")
    args = p.parse_args()

    config = OrchestratorConfig(
        pr_number=args.pr,
        head_sha=args.head_sha,
        pr_author=args.pr_author,
        repo_root=args.repo_root,
        menhir_port=args.menhir_port,
        proxy_port=args.proxy_port,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        upstream=args.upstream,
        max_calls=args.max_calls,
        max_usd=args.max_usd,
        max_seconds=args.max_seconds,
        judge_model=args.judge_model,
        questions_per_type=args.questions_per_type,
        baseline_file=args.baseline_file,
        runs_dir=args.runs_dir,
        menhir_dir=args.menhir_dir,
        menhir_python=args.menhir_python,
        confirm=args.confirm,
        dry_run=args.dry_run,
        skip_menhir_start=args.skip_menhir_start,
        menhir_startup_seconds=args.menhir_startup_seconds,
    )

    result = run_bench_for_pr(config)
    if result.killed:
        print(f"\n[bench] ABORTED: {result.kill_reason}", file=sys.stderr)
        return 1
    # Console may not support emoji (Windows cp1252) — encode safely
    try:
        print(f"\n[bench] card written to: {result.card_md[:100] if result.card_md else '(empty)'}...")
    except UnicodeEncodeError:
        print("\n[bench] card written (see card.md for full content)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
