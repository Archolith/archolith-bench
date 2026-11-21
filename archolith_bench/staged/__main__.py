"""Staged CLI.

  python -m archolith_bench.staged run --arm arms/x.yaml --fixture f.json
  python -m archolith_bench.staged run --arm arms/x.yaml --fixture f.json --serve ../menhir --graph builds/x.yaml
  python -m archolith_bench.staged rerun --run runs/<id> --from answer|judge
  python -m archolith_bench.staged compare runs/<a> runs/<b>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

from .. import prompts
from ..core.api import API_KEY, DIRECT_URL, MODEL, send_chat
from ..harness.longmemeval import LongMemEvalMemoryAdapter
from ..harness.memory_ab import assert_not_production
from ..harness.scoring import LLMJudgeScorer, UnofficialJudgeScorer
from .config import arm_from_resolved, load_arm, renderer_sha256, resolve, run_id
from .menhir_system import MenhirSystem, load_backend_client
from .ops import RERUN_STAGES, compare_runs, derive_run, format_compare, load_resolved
from .pipeline import build_report, run_answer, run_judge, run_search
from .rundir import RunDir
from .serve import ServeError, graph_record, graph_settings, served_menhir


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default="gpt-6-luna")
    p.add_argument("--no-unofficial-judge", action="store_true",
                   help="skip the unofficial (substance) judge on official-wrong items")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="archolith_bench.staged")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run staged search/answer/judge for one arm")
    r.add_argument("--arm", required=True)
    r.add_argument("--fixture", required=True)
    r.add_argument("--subset", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--stages", default="search,answer,judge")
    r.add_argument("--menhir-url", default=None)
    r.add_argument("--menhir-root", default=None)
    r.add_argument("--namespace-template", default="lme-{question_id}")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--menhir-commit", default=None)
    r.add_argument("--serve", default=None, metavar="MENHIR_TREE",
                   help="serve this Menhir checkout over the --graph build's Neo4j for the search stage "
                        "(replaces --menhir-url/--menhir-root)")
    r.add_argument("--graph", default=None, metavar="BUILD_YAML",
                   help="build YAML whose graph --serve reads (bolt port, embedder)")
    r.add_argument("--serve-port", type=int, default=8098, help="loopback port; must be on the throwaway allow-list")
    r.add_argument("--graph-results-dir", default=os.environ.get("LME_RESULTS_DIR"),
                   help="the --graph build's --results-dir, for its graph-provenance file (default: $LME_RESULTS_DIR); "
                        "the run records that build's commits beside the served tree's")
    r.add_argument("--menhir-bin", default=os.environ.get("MENHIR_MAIN_BIN"),
                   help="menhir executable when the tree has no .venv (default: $MENHIR_MAIN_BIN)")
    _add_judge_args(r)

    rr = sub.add_parser("rerun", help="rerun a saved run from answer or judge; never calls Menhir")
    rr.add_argument("--run", required=True, help="source run directory")
    rr.add_argument("--from", dest="from_stage", required=True, choices=RERUN_STAGES)
    rr.add_argument("--runs-dir", default=None, help="default: the source run's parent directory")
    rr.add_argument("--fixture", default=None, help="default: the fixture path recorded in the run")
    rr.add_argument("--model", default=None, help="answer model (from answer only; default: the run's)")
    rr.add_argument("--tag", default=None, help="force a fresh derived run when settings are unchanged")
    _add_judge_args(rr)

    c = sub.add_parser("compare", help="accuracy and per-item flips between two runs")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--json", dest="json_out", default=None, help="also write the comparison as JSON")
    return p


def _menhir_commit(root: str | None) -> str | None:
    """HEAD of the Menhir checkout serving the run, with "+dirty" for uncommitted changes."""
    if not root:
        return None
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}+dirty" if dirty else head


def _judge_config(args: argparse.Namespace) -> dict:
    return {"scorer": "llm-judge", "model": args.judge_model, "unofficial": not args.no_unofficial_judge}


def _judge(rundir: RunDir, items: list[dict], args: argparse.Namespace) -> int:
    judge_key = os.getenv("OPENAI_API_KEY", "")
    if not judge_key:
        print("ERROR: judge stage needs OPENAI_API_KEY", file=sys.stderr)
        return 2
    base_url = os.getenv("UPSTREAM_BASE_URL") or "https://api.openai.com/v1"
    scorer = LLMJudgeScorer(base_url=base_url, api_key=judge_key, model=args.judge_model)
    unofficial = None
    if not args.no_unofficial_judge:
        unofficial = UnofficialJudgeScorer(base_url=base_url, api_key=judge_key, model=args.judge_model)
    try:
        by_id = {str(it.get("question_id") or f"item-{i}"): it for i, it in enumerate(items)}
        run_judge(rundir, by_id, score_fn=scorer, unofficial_fn=unofficial)
    finally:
        scorer.close()
        if unofficial is not None:
            unofficial.close()
    build_report(rundir)
    return 0


def _answer(rundir, arm, adapter, model: str) -> None:  # noqa: ANN001
    with httpx.Client(timeout=180) as chat_client:
        run_answer(
            rundir, arm, adapter,
            send_fn=send_chat, chat_client=chat_client,
            # Same endpoint as the memory arms in cli.py: upstream directly, not the proxy.
            chat_base_url=DIRECT_URL, api_key=API_KEY, model=model,
        )


def _search(rundir, arm, items, adapter, args, menhir_url) -> None:  # noqa: ANN001
    system = None
    if arm.context != "none":
        system = MenhirSystem(load_backend_client(Path(args.menhir_root), menhir_url))
    asyncio.run(run_search(rundir, arm, items, adapter, system, namespace_template=args.namespace_template))


def _cmd_run(args: argparse.Namespace) -> int:
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if args.serve:
        if args.menhir_url or args.menhir_root:
            print("ERROR: --serve replaces --menhir-url/--menhir-root; pass one or the other", file=sys.stderr)
            return 2
        if not args.graph:
            print("ERROR: --serve needs --graph <build yaml> naming the built graph", file=sys.stderr)
            return 2
        # The client refuses a non-allow-listed port; refuse before serving, not after.
        assert_not_production(f"http://127.0.0.1:{args.serve_port}")
        args.menhir_root = str(Path(args.serve).resolve())
    graph_rec = None
    if args.serve:
        try:
            graph_rec = graph_record(args.graph, dict(os.environ), args.graph_results_dir)
        except (ValueError, OSError, yaml.YAMLError) as exc:
            print(f"ERROR: --graph {args.graph}: {exc}", file=sys.stderr)
            return 2
        if graph_rec["provenance"] is None:
            print(f"WARNING: no graph-provenance-{graph_rec['neo4j_name']}.json under --graph-results-dir "
                  f"{args.graph_results_dir!r}; the run records the graph's build commits as unknown", file=sys.stderr)
    arm = load_arm(args.arm)
    adapter = LongMemEvalMemoryAdapter()
    items = adapter.load_items(args.subset, args.limit, args.fixture)
    fixture_sha = hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest()
    model = arm.answer_model or args.model
    resolved = resolve(
        arm,
        benchmark_id=adapter.benchmark_id,
        fixture_path=str(args.fixture),
        fixture_sha256=fixture_sha,
        prompt_fingerprint=adapter.prompt_fingerprint,
        renderer_sha256=renderer_sha256(),
        menhir_commit=args.menhir_commit or _menhir_commit(args.menhir_root),
        namespace_template=args.namespace_template,
        model=model,
        judge=_judge_config(args),
        prompt_files=prompts.hashes(prompts.LONGMEMEVAL_RUN_PROMPTS),
        graph=graph_rec,
    )
    rundir = RunDir(Path(args.runs_dir), run_id(resolved))
    rundir.write_config(resolved)

    if "search" in stages:
        if arm.context != "none" and args.serve:
            try:
                graph = graph_settings(args.graph, dict(os.environ))
            except (ValueError, OSError, yaml.YAMLError) as exc:
                print(f"ERROR: --graph {args.graph}: {exc}", file=sys.stderr)
                return 2
            try:
                with served_menhir(Path(args.menhir_root), port=args.serve_port, graph=graph,
                                   openai_key=os.getenv("OPENAI_API_KEY", ""), out_dir=rundir.path,
                                   menhir_bin=args.menhir_bin) as url:
                    _search(rundir, arm, items, adapter, args, url)
            except ServeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
        else:
            if arm.context != "none" and (not args.menhir_url or not args.menhir_root):
                print("ERROR: --menhir-url and --menhir-root (or --serve) are required for this arm",
                      file=sys.stderr)
                return 2
            _search(rundir, arm, items, adapter, args, args.menhir_url)
    if "answer" in stages:
        _answer(rundir, arm, adapter, model)
    if "judge" in stages:
        rc = _judge(rundir, items, args)
        if rc:
            return rc
    print(rundir.path)
    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    src = Path(args.run)
    resolved = load_resolved(src)
    adapter = LongMemEvalMemoryAdapter()
    fixture = args.fixture or resolved["fixture"]["path"]
    fixture_sha = hashlib.sha256(Path(fixture).read_bytes()).hexdigest()
    if fixture_sha != resolved["fixture"]["sha256"]:
        print(f"ERROR: fixture {fixture} does not match the run's fixture sha256", file=sys.stderr)
        return 2
    model = args.model or resolved.get("model")
    rundir = derive_run(
        src, Path(args.runs_dir) if args.runs_dir else src.parent, args.from_stage,
        prompt_files=prompts.hashes(prompts.LONGMEMEVAL_RUN_PROMPTS),
        judge=_judge_config(args),
        model=model,
        answer_prompt_fingerprint=adapter.prompt_fingerprint,
        tag=args.tag,
    )
    if args.from_stage == "answer":
        _answer(rundir, arm_from_resolved(resolved), adapter, model)
    items = adapter.load_items(None, None, fixture)
    rc = _judge(rundir, items, args)
    if rc:
        return rc
    print(rundir.path)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    result = compare_runs(Path(args.run_a), Path(args.run_b))
    print(format_compare(result))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "rerun":
        return _cmd_rerun(args)
    return _cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
