"""CLI: python -m archolith_bench.build --build builds/x.yaml [run-control flags] [--dry-run].

--retry-failed re-enriches the built graph's FAILED episodes (LME_RETRY_FAILED=1) instead of building.
--check reports the build's progress (and, with --expected-items, its acceptance verdict); it builds nothing.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .config import (
    API_KEY_VAR,
    apply_env_overrides,
    child_env,
    env_assignments,
    resolve_yaml,
    resolved_record,
)

log = logging.getLogger("archolith_bench.build")

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "longmemeval" / "build_graph.sh"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m archolith_bench.build")
    p.add_argument("--build", required=True, help="build YAML (e.g. builds/default.yaml)")
    p.add_argument("--resume", action="store_true", help="LME_RESUME=1")
    p.add_argument("--fresh", action="store_true", help="LME_REQUIRE_FRESH=1")
    p.add_argument("--results-dir", help="LME_RESULTS_DIR; build.resolved.yaml is written here")
    p.add_argument("--reset-file", help="LME_RESET_FILE")
    p.add_argument("--stop-after-items", type=int, help="LME_INGEST_STOP_AFTER_ITEMS")
    p.add_argument("--retry-failed", action="store_true",
                   help="LME_RETRY_FAILED=1: reset and drain this build's FAILED episodes on the existing graph")
    p.add_argument("--drain-timeout", type=int, help="LME_RETRY_DRAIN_TIMEOUT seconds (--retry-failed; default 86400)")
    p.add_argument("--check", action="store_true",
                   help="report progress (manifest, state log, live Menhir ops if one serves); never builds")
    p.add_argument("--expected-items", type=int, help="--check: also run canonical acceptance for N items")
    p.add_argument("--watch", type=int, metavar="SECS", help="--check: repeat every SECS, logging to progress.log")
    p.add_argument("--menhir-url", help="--check: Menhir to read live state from (default: the build port)")
    p.add_argument("--dry-run", action="store_true", help="print the resolution and run nothing")
    return p


def _mask(var: str, value: str) -> str:
    return "***" if var == API_KEY_VAR else value


def _check(args: argparse.Namespace, config: dict, environ: dict[str, str]) -> int:
    from ..harness.targets import TargetRefused, assert_allowed_target
    from .check import check_paths, run_check

    url = args.menhir_url or f"http://127.0.0.1:{config['container']['menhir_port']}"
    try:
        assert_allowed_target(url, what="--check Menhir")
    except TargetRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    paths = check_paths(config, environ, args.results_dir)
    return run_check(paths, limit=int(config["dataset"]["limit"]), menhir_url=url,
                     expected_items=args.expected_items, watch=args.watch)


def main(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[build] %(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    environ = dict(os.environ if environ is None else environ)
    if args.resume and args.fresh:
        print("error: --resume and --fresh are mutually exclusive", file=sys.stderr)
        return 2
    if args.retry_failed and (args.resume or args.fresh or args.reset_file or args.stop_after_items is not None):
        print("error: --retry-failed takes no build run-control flag (--resume/--fresh/--reset-file/--stop-after-items)",
              file=sys.stderr)
        return 2
    if args.drain_timeout is not None and not args.retry_failed:
        print("error: --drain-timeout applies only to --retry-failed", file=sys.stderr)
        return 2
    check_only = [f for f, v in (("--expected-items", args.expected_items), ("--watch", args.watch),
                                 ("--menhir-url", args.menhir_url)) if v is not None]
    if check_only and not args.check:
        print(f"error: {', '.join(check_only)} applies only to --check", file=sys.stderr)
        return 2
    if args.check and (args.retry_failed or args.resume or args.fresh or args.reset_file
                       or args.stop_after_items is not None or args.dry_run):
        print("error: --check takes no build, retry or dry-run flag", file=sys.stderr)
        return 2
    if args.watch is not None and (args.watch < 1 or args.expected_items is not None):
        print("error: --watch needs SECS >= 1 and cannot be combined with --expected-items", file=sys.stderr)
        return 2
    if args.expected_items is not None and args.expected_items < 1:
        print("error: --expected-items must be a positive item count", file=sys.stderr)
        return 2
    try:
        config = resolve_yaml(args.build)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        config, overrides, warnings = apply_env_overrides(config, environ)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for w in warnings:
        log.warning(w)
    record = resolved_record(args.build, config, overrides)
    if args.check:
        return _check(args, config, environ)

    try:
        sets, unset = env_assignments(config, environ, overrides)
    except KeyError as exc:
        if not args.dry_run:
            print(f"error: api_key_env {exc.args[0]} is not set in the environment", file=sys.stderr)
            return 2
        log.warning("api_key_env %s is not set (dry run continues)", exc.args[0])
        sets, unset = env_assignments(
            {**config, "extraction": {**config["extraction"], "api_key_env": None}}, environ, overrides
        )

    run_control: dict[str, str] = {}
    if args.resume:
        run_control["LME_RESUME"] = "1"
    if args.fresh:
        run_control["LME_REQUIRE_FRESH"] = "1"
    if args.reset_file:
        run_control["LME_RESET_FILE"] = args.reset_file
    if args.stop_after_items is not None:
        run_control["LME_INGEST_STOP_AFTER_ITEMS"] = str(args.stop_after_items)
    if args.retry_failed:
        run_control["LME_RETRY_FAILED"] = "1"
    if args.drain_timeout is not None:
        run_control["LME_RETRY_DRAIN_TIMEOUT"] = str(args.drain_timeout)
    results_dir = args.results_dir or environ.get("LME_RESULTS_DIR")
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve().as_posix()
        run_control["LME_RESULTS_DIR"] = results_dir

    if args.dry_run:
        print(yaml.safe_dump(record, sort_keys=True, default_flow_style=False).rstrip())
        print("# LME_* assignments")
        for var in sorted(sets):
            print(f"{var}={_mask(var, sets[var])}")
        for var in sorted(run_control):
            print(f"{var}={run_control[var]}")
        for var in sorted(unset):
            print(f"unset {var}")
        return 0

    if not results_dir:
        print("error: --results-dir is required (or set LME_RESULTS_DIR)", file=sys.stderr)
        return 2
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build.resolved.yaml").write_text(
        yaml.safe_dump(record, sort_keys=True, default_flow_style=False), encoding="utf-8", newline="\n"
    )

    env = child_env(environ, sets, unset)
    # Only the flags select retry mode: an inherited export must not turn a build into a retry.
    for var in ("LME_RETRY_FAILED", "LME_RETRY_DRAIN_TIMEOUT"):
        env.pop(var, None)
    env.update(run_control)
    bash = shutil.which("bash")
    if not bash:
        print("error: bash not found on PATH", file=sys.stderr)
        return 2
    limit = str(config["dataset"]["limit"])
    log.info("build_id=%s results_dir=%s", record["build_id"], results_dir)
    return subprocess.run([bash, BUILD_SCRIPT.as_posix(), limit], env=env, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
