"""Minimal CLI: python -m archolith_bench.staged run --arm arms/x.yaml --fixture f.json."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import httpx

from ..core.api import API_KEY, MODEL, PROXY_URL, send_chat
from ..harness.longmemeval import LongMemEvalMemoryAdapter
from ..harness.scoring import LLMJudgeScorer
from .config import load_arm, renderer_sha256, resolve, run_id
from .menhir_system import MenhirSystem, load_backend_client
from .pipeline import build_report, run_answer, run_judge, run_search
from .rundir import RunDir


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
    r.add_argument("--judge-model", default="gpt-6-luna")
    r.add_argument("--menhir-commit", default=None)
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
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
        judge={"scorer": "llm-judge", "model": args.judge_model},
    )
    rundir = RunDir(Path(args.runs_dir), run_id(resolved))
    rundir.write_config(resolved)

    if "search" in stages:
        system = None
        if arm.context != "none":
            if not args.menhir_url or not args.menhir_root:
                print("ERROR: --menhir-url and --menhir-root are required for this arm", file=sys.stderr)
                return 2
            system = MenhirSystem(load_backend_client(Path(args.menhir_root), args.menhir_url))
        asyncio.run(
            run_search(rundir, arm, items, adapter, system, namespace_template=args.namespace_template)
        )
    if "answer" in stages:
        with httpx.Client(timeout=180) as chat_client:
            run_answer(
                rundir, arm, adapter,
                send_fn=send_chat, chat_client=chat_client,
                chat_base_url=PROXY_URL, api_key=API_KEY, model=model,
            )
    if "judge" in stages:
        judge_key = os.getenv("OPENAI_API_KEY", "")
        if not judge_key:
            print("ERROR: judge stage needs OPENAI_API_KEY", file=sys.stderr)
            return 2
        scorer = LLMJudgeScorer(
            base_url=os.getenv("UPSTREAM_BASE_URL") or "https://api.openai.com/v1",
            api_key=judge_key,
            model=args.judge_model,
        )
        try:
            by_id = {str(it.get("question_id") or f"item-{i}"): it for i, it in enumerate(items)}
            run_judge(rundir, by_id, score_fn=scorer)
        finally:
            scorer.close()
        build_report(rundir)
    print(rundir.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
