"""End-to-end orchestrator: start proxy + menhir, run slice, compare, render card.

This is the Python entry point called by ``scripts/bench-pr.ps1``. It owns:

1. Pre-flight checks (env vars, cooldown, per-PR run count)
2. Start the budget proxy (holds the real OPENAI_API_KEY)
3. Fetch the PR's menhir code into an isolated worktree
4. Start the PR's menhir as a subprocess pointed at the proxy
5. Wait for menhir health
6. Run the stratified LongMemEval slice
7. Compare vs baseline
8. Render the PR comment card
9. Tear down (kill menhir + proxy)

The PR's menhir code never sees the real API key — it only sees
``OPENAI_BASE_URL=http://127.0.0.1:<proxy_port>/v1``.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .budget_proxy import BudgetProxy
from .compare import compare_results, load_baseline
from .card import render_pr_card
from .stratified import run_stratified_slice


@dataclass
class OrchestratorConfig:
    pr_number: int
    head_sha: str
    pr_author: str = "unknown"
    repo_root: str = "."
    menhir_port: int = 8090
    proxy_port: int = 8765
    neo4j_uri: str = "bolt://localhost:7689"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "lmedata123"
    upstream: str = "https://api.openai.com"
    max_calls: int = 200
    max_usd: float = 5.0
    max_seconds: float = 900.0
    judge_model: str = "gpt-6-luna"
    questions_per_type: int = 20
    baseline_file: str = "benchmarks/longmemeval-baseline.json"
    runs_dir: str = ".bench/runs"
    menhir_dir: str | None = None
    menhir_python: str | None = None
    confirm: bool = False
    dry_run: bool = False
    skip_menhir_start: bool = False  # if menhir is already running externally
    menhir_startup_seconds: float = 60.0


@dataclass
class OrchestratorResult:
    killed: bool = False
    kill_reason: str | None = None
    card_md: str = ""
    results_path: str = ""
    budget_path: str = ""


def run_bench_for_pr(config: OrchestratorConfig) -> OrchestratorResult:
    """End-to-end: proxy → menhir → slice → compare → card."""
    runs_dir = Path(config.runs_dir) / str(config.pr_number)
    runs_dir.mkdir(parents=True, exist_ok=True)

    trace_file = runs_dir / "traces.jsonl"
    budget_file = runs_dir / "budget.json"
    results_path = runs_dir / "results.json"
    card_path = runs_dir / "card.md"

    # Dry-run renders the command plan and card without credentials, a proxy,
    # or a Menhir process.
    if config.dry_run:
        return _run_dry_run(config, runs_dir, results_path, card_path)

    baseline_path = Path(config.repo_root) / config.baseline_file
    if not baseline_path.exists():
        return _fail(runs_dir, f"baseline file not found: {baseline_path}")
    try:
        baseline = load_baseline(baseline_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _fail(runs_dir, f"baseline is not gate-ready: {exc}")

    # ─── 1. Resolve API key ───────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return _fail(runs_dir, "OPENAI_API_KEY not in env — refusing to run")

    # ─── 2. Start budget proxy ────────────────────────────────────
    print(f"[bench] starting budget proxy on :{config.proxy_port}", flush=True)
    menhir_proc: subprocess.Popen | None = None

    def kill_menhir_for_budget() -> None:
        if menhir_proc is not None:
            _kill_process_tree(menhir_proc)

    proxy = BudgetProxy(
        api_key=api_key,
        upstream=config.upstream,
        port=config.proxy_port,
        trace_file=trace_file,
        budget_file=budget_file,
        max_calls=config.max_calls,
        max_usd=config.max_usd,
        max_seconds=config.max_seconds,
        on_killed=kill_menhir_for_budget,
    )
    try:
        proxy.start()
    except OSError as e:
        return _fail(runs_dir, f"failed to start budget proxy: {e}")

    watchdog_stop = threading.Event()
    watchdog: threading.Thread | None = None
    try:
        # ─── 3. Start menhir (if not externally provided) ──────────
        if not config.skip_menhir_start:
            if not _is_port_available(config.menhir_port):
                return _fail(runs_dir, f"menhir port {config.menhir_port} is already in use")
            menhir_env = _build_menhir_env(config, proxy.base_url)
            menhir_proc = _start_menhir(config, menhir_env, runs_dir)
            if menhir_proc is None:
                return _fail(runs_dir, "failed to start menhir subprocess")

            watchdog = threading.Thread(
                target=_watch_budget,
                args=(proxy, watchdog_stop),
                name="menhir-budget-watchdog",
                daemon=True,
            )
            watchdog.start()

            # Reject an early-exiting child instead of accepting an unrelated
            # service that happens to be listening on the configured port.
            ok = _wait_for_health(menhir_proc, config.menhir_port, config.menhir_startup_seconds)
            if not ok:
                if menhir_proc.poll() is not None:
                    return _fail(runs_dir, f"menhir subprocess exited early ({menhir_proc.returncode})")
                return _fail(runs_dir, f"menhir did not become healthy within {config.menhir_startup_seconds}s")
            print(f"[bench] menhir healthy on :{config.menhir_port}", flush=True)

        # ─── 4. Run stratified slice ──────────────────────────────
        print("[bench] running stratified slice", flush=True)
        result = run_stratified_slice(
            menhir_url=f"http://127.0.0.1:{config.menhir_port}",
            output_dir=runs_dir,
            judge_model=config.judge_model,
            questions_per_type=config.questions_per_type,
            scorer="llm-judge",
            dry_run=config.dry_run,
        )

        # Check if proxy killed us mid-run
        if proxy.is_killed():
            return _killed(runs_dir, proxy.kill_reason(), result, config, budget_file)

        # ─── 5. Persist aggregated results ────────────────────────
        results_payload = {
            "pr_number": config.pr_number,
            "head_sha": config.head_sha,
            "pr_author": config.pr_author,
            "overall": result.overall,
            "by_type": result.by_type,
            "n_total": result.n_total,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
            "per_question": result.per_question,
            "type_results": [
                {
                    "type": t.type, "score": t.score, "n": t.n,
                    "input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
                    "cost_usd": t.cost_usd, "error": t.error,
                }
                for t in result.type_results
            ],
            "llm_calls_used": proxy.state.calls,
        }
        results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

        slice_errors = [f"{item.type}: {item.error}" for item in result.type_results if item.error]
        expected_total = config.questions_per_type * len(result.type_results)
        if result.n_total != expected_total:
            slice_errors.append(f"aggregate: expected {expected_total} questions, got {result.n_total}")
        if slice_errors:
            return _killed(
                runs_dir,
                "incomplete stratified slice: " + "; ".join(slice_errors),
                result,
                config,
                budget_file,
            )

        # ─── 6. Compare vs baseline ────────────────────────────────
        comparison = compare_results(
            baseline=baseline,
            current_overall=result.overall,
            current_by_type=result.by_type,
            current_per_question=result.per_question,
        )

        # ─── 7. Render PR card ────────────────────────────────────
        card = render_pr_card(
            pr_number=config.pr_number,
            pr_author=config.pr_author,
            head_sha=config.head_sha,
            result=result,
            comparison=comparison,
            max_calls=config.max_calls,
            max_usd=config.max_usd,
            llm_calls_used=proxy.state.calls,
            usd_used=proxy.state.usd,
        )
        card_path.write_text(card, encoding="utf-8")

        print(f"[bench] done. card: {card_path}", flush=True)
        print(f"[bench] results: {results_path}", flush=True)
        return OrchestratorResult(
            killed=False,
            card_md=card,
            results_path=str(results_path),
            budget_path=str(budget_file),
        )

    finally:
        watchdog_stop.set()
        if watchdog is not None:
            watchdog.join(timeout=2.0)
        if menhir_proc is not None:
            _kill_process_tree(menhir_proc)
        proxy.stop()


def _build_menhir_env(config: OrchestratorConfig, proxy_base_url: str) -> dict[str, str]:
    """Build the env for the PR's menhir subprocess. Key insight: OPENAI_API_KEY
    is set to a dummy value — the proxy ignores it and adds the real key."""
    # Do not inherit credentials or user-specific tool configuration from the
    # maintainer shell. This is an environment boundary, not an OS sandbox;
    # run untrusted PR code in a VM/container when adversarial execution is in
    # scope.
    runtime_keys = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "LOCALAPPDATA")
    env = {key: os.environ[key] for key in runtime_keys if key in os.environ}
    env["OPENAI_API_KEY"] = "bench-proxy-holds-the-real-key"
    env["OPENAI_BASE_URL"] = f"{proxy_base_url}/v1"
    env["NEO4J_URI"] = config.neo4j_uri
    env["NEO4J_USER"] = config.neo4j_user
    env["NEO4J_PASSWORD"] = config.neo4j_password
    env["MENHIR_API_HOST"] = "127.0.0.1"
    env["MENHIR_API_PORT"] = str(config.menhir_port)
    env["MENHIR_EXPLORER_ENABLED"] = "0"  # skip explorer for bench
    env["MENHIR_BENCHMARK_MODE"] = "1"
    env["MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED"] = "0"
    if config.menhir_dir:
        env["PYTHONPATH"] = str(Path(config.menhir_dir) / "src")
    # Sandbox HOME so PR code can't read ~/.ssh, ~/.aws, etc.
    fake_home = Path(config.runs_dir) / str(config.pr_number) / "fake-home"
    fake_home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    return env


def _start_menhir(config: OrchestratorConfig, env: dict[str, str], runs_dir: Path) -> subprocess.Popen | None:
    """Start the PR's menhir from the worktree."""
    if not config.menhir_dir:
        print("[bench] no Menhir worktree supplied", file=sys.stderr, flush=True)
        return None
    menhir_dir = config.menhir_dir
    if not config.menhir_python:
        print("[bench] no Menhir Python supplied", file=sys.stderr, flush=True)
        return None
    menhir_python = Path(config.menhir_python)
    if not menhir_python.is_file():
        print(f"[bench] Menhir Python not found: {menhir_python}", file=sys.stderr, flush=True)
        return None
    stdout_path = runs_dir / "menhir-stdout.log"
    stderr_path = runs_dir / "menhir-stderr.log"
    stdout_fp = stdout_path.open("w", encoding="utf-8")
    stderr_fp = stderr_path.open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            [str(menhir_python), "-m", "menhir", "serve"],
            cwd=menhir_dir,
            env=env,
            stdout=stdout_fp,
            stderr=stderr_fp,
        )
    except (OSError, FileNotFoundError) as e:
        print(f"[bench] failed to start menhir: {e}", file=sys.stderr, flush=True)
        return None


def _wait_for_health(proc: subprocess.Popen, port: int, timeout_s: float) -> bool:
    import httpx
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200 and proc.poll() is None:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def _is_port_available(port: int) -> bool:
    """Fail closed when a local service could be mistaken for the PR service."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _watch_budget(proxy: BudgetProxy, stop: threading.Event) -> None:
    """Enforce the elapsed-time limit even while Menhir is idle."""
    while not stop.wait(0.1):
        if proxy.enforce_caps() is not None:
            return


def _run_dry_run(
    config: OrchestratorConfig,
    runs_dir: Path,
    results_path: Path,
    card_path: Path,
) -> OrchestratorResult:
    """Generate dry-run artifacts without starting privileged dependencies."""
    result = run_stratified_slice(
        menhir_url=f"http://127.0.0.1:{config.menhir_port}",
        output_dir=runs_dir,
        judge_model=config.judge_model,
        questions_per_type=config.questions_per_type,
        scorer="llm-judge",
        dry_run=True,
    )
    results_payload = {
        "pr_number": config.pr_number,
        "head_sha": config.head_sha,
        "pr_author": config.pr_author,
        "overall": result.overall,
        "by_type": result.by_type,
        "n_total": result.n_total,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "per_question": result.per_question,
        "type_results": [
            {"type": t.type, "score": t.score, "n": t.n, "error": t.error}
            for t in result.type_results
        ],
        "llm_calls_used": 0,
    }
    results_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    baseline_path = Path(config.repo_root) / config.baseline_file
    if not baseline_path.exists():
        return _fail(runs_dir, f"baseline file not found: {baseline_path}")
    comparison = compare_results(
        baseline=load_baseline(baseline_path, allow_stub=True),
        current_overall=result.overall,
        current_by_type=result.by_type,
        current_per_question=result.per_question,
    )
    card = render_pr_card(
        pr_number=config.pr_number,
        pr_author=config.pr_author,
        head_sha=config.head_sha,
        result=result,
        comparison=comparison,
        max_calls=config.max_calls,
        max_usd=config.max_usd,
        llm_calls_used=0,
        usd_used=0.0,
    )
    card_path.write_text(card, encoding="utf-8")
    return OrchestratorResult(
        killed=False,
        card_md=card,
        results_path=str(results_path),
    )


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            if os.name == "nt":
                # Windows: taskkill the whole tree
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                )
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
        pass


def _fail(runs_dir: Path, reason: str) -> OrchestratorResult:
    print(f"[bench] FAIL: {reason}", file=sys.stderr, flush=True)
    return OrchestratorResult(killed=True, kill_reason=reason)


def _killed(
    runs_dir: Path,
    reason: str,
    result,
    config: OrchestratorConfig,
    budget_file: Path,
) -> OrchestratorResult:
    from .card import _render_killed
    card = _render_killed(
        pr_number=config.pr_number,
        pr_author=config.pr_author,
        head_sha=config.head_sha,
        kill_reason=reason,
        max_calls=config.max_calls,
        max_usd=config.max_usd,
        llm_calls_used=0,
    )
    (runs_dir / "card.md").write_text(card, encoding="utf-8")
    return OrchestratorResult(killed=True, kill_reason=reason, card_md=card, budget_path=str(budget_file))
