"""Run the task x condition x repeat matrix through ``opencode run --format json``.

Every run gets a fresh export of the repository at its pinned commit, sealed as its own
git repository (condition A never sees a beacon), a private OpenCode config home, and a
fixed prompt on stdin. OpenCode's events are streamed: a run is killed as soon as its
tokens pass the per-run reserve or a rate-limit error appears on stdout or stderr. The
matrix admits a run only while the reserve still fits under the cap, and stops at the
first rate limit (never retrying), an over-reserve run, or a run with no usage data.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

from archolith_bench.beacon_eval import CONDITIONS
from archolith_bench.beacon_eval.isolation import (
    beacon_server,
    default_config_source,
    isolated_config_home,
    isolated_env,
    load_api_keys,
)
from archolith_bench.beacon_eval.models import ANSWER_KEYS, RepoPin, RunResult, Task
from archolith_bench.beacon_eval.scoring import extract_answer, score

#: Owner decision 2026-09-23 (GPT-6 Luna via OpenAI; key through --env-file).
DEFAULT_MODEL = "openai/gpt-6-luna"
DEFAULT_BUDGET_TOKENS = 20_000_000
#: Tokens set aside for each run; a run that passes it is killed and stops the matrix.
DEFAULT_RUN_RESERVE = 400_000

#: Same line for every condition, so the only intended difference is the added context.
TOOL_NOTE = "Use whatever tools are available to you."

ANSWER_INSTRUCTIONS = (
    "Work read-only: do not edit, create or delete files, and do not run tests or installs. "
    "When you are done, end your reply with one fenced ```json block containing exactly "
    'these keys: "docs" (paths of documents to read), "files" (source files involved), '
    '"commands" (exact shell commands), "guardrails" (rules that apply, each a short '
    'sentence), "verdict" (a single word when the task asks for one, else ""), "findings" '
    "(short statements that answer the question), \"plan\" "
    '(ordered steps), "citations" (a list of {"path", "line_start", "line_end"} backing '
    "your claims)."
)

_RATE_LIMIT = re.compile(r"\b429\b|rate[ _-]?limit|too many requests", re.IGNORECASE)


class RateLimited(RuntimeError):
    """The provider refused with a rate limit; the matrix stops, nothing is retried."""


class BudgetExhausted(RuntimeError):
    """The next run's reserve would not fit under the cap, or a run passed its reserve."""


class AccountingError(RuntimeError):
    """A run reported no token usage, so the budget can no longer be trusted."""


def resolve_opencode() -> list[str]:
    """The OpenCode executable; on Windows the real ``.exe`` behind the npm shim."""
    found = shutil.which("opencode")
    if not found:
        return ["opencode"]
    if sys.platform == "win32":
        exe = Path(found).parent / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
        if exe.is_file():
            return [str(exe)]
    return [found]


@dataclass
class RunnerConfig:
    workdir: Path
    beacon_python: str
    beacon_src: str | None = None
    opencode_cmd: list[str] = field(default_factory=resolve_opencode)
    model: str = DEFAULT_MODEL
    #: The user's ``opencode.json``; only the model's provider block is taken from it.
    config_source: Path | None = None
    #: None switches a token limit off (dollar mode uses budget_usd instead).
    budget_tokens: int | None = DEFAULT_BUDGET_TOKENS
    run_reserve_tokens: int | None = DEFAULT_RUN_RESERVE
    timeout_s: float = 900.0
    #: ``.env`` whose ``*_API_KEY`` values reach only the OpenCode process (built-in providers).
    env_file: Path | None = None
    #: Dollar cap from OpenCode's per-step cost; when set, a run with no cost data stops the matrix.
    budget_usd: float | None = None
    run_reserve_usd: float = 0.10


@dataclass
class Budget:
    cap: int | None
    reserve: int | None = DEFAULT_RUN_RESERVE
    used: int = 0
    cap_usd: float | None = None
    reserve_usd: float = 0.0
    used_usd: float = 0.0

    def check(self) -> None:
        if self.cap is not None and self.used + (self.reserve or 0) > self.cap:
            raise BudgetExhausted(
                f"the next run's {self.reserve:,}-token reserve would exceed the "
                f"{self.cap:,}-token cap ({self.used:,} used)"
            )
        if self.cap_usd is not None and self.used_usd + self.reserve_usd > self.cap_usd:
            raise BudgetExhausted(
                f"the next run's ${self.reserve_usd:.2f} reserve would exceed the "
                f"${self.cap_usd:.2f} cap (${self.used_usd:.4f} spent)"
            )

    def spend(self, tokens: int, usd: float = 0.0) -> None:
        self.used += tokens
        self.used_usd += usd


# ---------------------------------------------------------------------------
# Checkouts and beacons
# ---------------------------------------------------------------------------


def export_commit(pin: RepoPin, dest: Path, cache: Path) -> Path:
    """Write the tree of *pin* at its commit into *dest* (no .git)."""
    source = Path(pin.local_path) if pin.local_path else cache / pin.name
    if not pin.local_path and not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", pin.url, str(source)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--quiet", "origin", pin.commit],
        check=False,
        capture_output=True,
    )
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", pin.commit],
        check=True,
        capture_output=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def seal_checkout(checkout: Path) -> None:
    """Make *checkout* its own git root with one commit.

    OpenCode searches parent directories for project config and instructions up to the
    git root; without this a checkout under the results tree picks up the enclosing
    repositories' ``AGENTS.md`` and ``opencode.json``.
    """
    git = [
        "git", "-C", str(checkout),
        "-c", "user.name=beacon-eval", "-c", "user.email=beacon-eval@example.invalid",
        "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
    ]
    subprocess.run([*git, "init", "--quiet"], check=True, capture_output=True)
    subprocess.run([*git, "add", "--all"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", "pinned export"],
                   check=True, capture_output=True)


def build_beacon(config: RunnerConfig, pin: RepoPin) -> Path:
    """Build the beacon for *pin* in its own export (never the agent's checkout)."""
    root = config.workdir / "beacons" / pin.name
    manifest = root / "beacon.generated.yaml"
    if manifest.is_file():
        return manifest
    export_commit(pin, root, config.workdir / "cache")
    env = dict(os.environ)
    if config.beacon_src:
        env["PYTHONPATH"] = config.beacon_src
    subprocess.run(
        [config.beacon_python, "-m", "beacon", "build", "--repo", str(root), "--format", "json"],
        check=True,
        capture_output=True,
        env=env,
    )
    return manifest


def build_prompt(task: Task, condition: str, manifest_text: str = "") -> str:
    """Task first, then (C only) the pasted manifest, then the same closing lines."""
    parts = [task.prompt.strip()]
    if condition == "C":
        parts.append(
            "Project knowledge (beacon.generated.yaml) follows.\n```yaml\n"
            + manifest_text.strip()
            + "\n```"
        )
    parts += [TOOL_NOTE, ANSWER_INSTRUCTIONS]
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# OpenCode events
# ---------------------------------------------------------------------------


@dataclass
class EventLog:
    """Accumulates ``--format json`` events from their known locations only.

    One JSON object per line with a top-level ``type``: ``text`` (``part.text``),
    ``tool_use`` (one tool call), ``step_finish`` (``part.tokens``) and ``error``.
    Rate limits are looked for in error events, non-JSON stdout lines and stderr,
    never in text or tool output (which can quote "429" from the repository).
    """

    texts: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reported_total_tokens: int = 0
    usage_events: int = 0
    cost_usd: float = 0.0
    cost_events: int = 0
    tool_calls: int = 0
    errors: list[str] = field(default_factory=list)
    unknown_types: set[str] = field(default_factory=set)
    rate_limited: bool = False

    @property
    def total_tokens(self) -> int:
        computed = (
            self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
        )
        return max(self.reported_total_tokens, computed)

    def feed(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self._scan(line)
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        raw_part = event.get("part")
        part: dict[str, Any] = raw_part if isinstance(raw_part, dict) else {}
        if kind == "error" or "error" in event:
            message = json.dumps(event.get("error", event))[:500]
            self.errors.append(message)
            self._scan(message)
        elif kind == "text":
            if isinstance(part.get("text"), str):
                self.texts.append(part["text"])
        elif kind == "tool_use":
            self.tool_calls += 1
        elif kind == "step_finish":
            self._add_usage(part.get("tokens"))
            if isinstance(part.get("cost"), int | float):
                self.cost_usd += float(part["cost"])
                self.cost_events += 1
        elif kind not in ("step_start", "reasoning"):
            self.unknown_types.add(str(kind))

    def feed_stderr(self, line: str) -> None:
        self._scan(line)

    def _scan(self, text: str) -> None:
        if _RATE_LIMIT.search(text):
            self.rate_limited = True

    def _add_usage(self, tokens: Any) -> None:
        if not isinstance(tokens, dict):
            return
        self.usage_events += 1
        self.input_tokens += int(tokens.get("input") or 0)
        self.output_tokens += int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
        raw_cache = tokens.get("cache")
        cache: dict[str, Any] = raw_cache if isinstance(raw_cache, dict) else {}
        self.cache_read_tokens += int(cache.get("read") or 0)
        self.cache_write_tokens += int(cache.get("write") or 0)
        self.reported_total_tokens += int(tokens.get("total") or 0)


def parse_events(stdout: str, stderr: str = "") -> dict[str, Any]:
    """Summary of a finished run's output (see :class:`EventLog`)."""
    log = EventLog()
    for line in stdout.splitlines():
        log.feed(line)
    for line in stderr.splitlines():
        log.feed_stderr(line)
    return {
        "text": "\n".join(log.texts),
        "input_tokens": log.input_tokens,
        "output_tokens": log.output_tokens,
        "cache_read_tokens": log.cache_read_tokens,
        "cache_write_tokens": log.cache_write_tokens,
        "total_tokens": log.total_tokens,
        "usage_events": log.usage_events,
        "tool_calls": log.tool_calls,
        "rate_limited": log.rate_limited,
    }


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def _pump(stream: IO[str], tag: str, sink: queue.Queue[tuple[str, str | None]]) -> None:
    for line in stream:
        sink.put((tag, line))
    sink.put((tag, None))


def stream_opencode(
    cmd: list[str], prompt: str, cwd: Path, env: dict[str, str], run_dir: Path,
    reserve: int | None, timeout_s: float, reserve_usd: float | None = None,
) -> tuple[EventLog, str, int | None]:
    """Run OpenCode, feeding events as they arrive; returns (log, stop reason, exit code).

    The raw streams are kept in ``events.jsonl`` and ``stderr.log``. Stop reasons:
    "" (finished), "rate_limited", "over_reserve", "timeout".
    """
    log = EventLog()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdin and proc.stdout and proc.stderr
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
    except OSError:
        pass  # the process exited early; its output says why
    lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
    for stream, tag in ((proc.stdout, "out"), (proc.stderr, "err")):
        threading.Thread(target=_pump, args=(stream, tag, lines), daemon=True).start()
    reason, open_streams = "", 2
    deadline = time.monotonic() + timeout_s
    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as events, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as errors:
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timeout"
                break
            try:
                tag, line = lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                open_streams -= 1
                continue
            if tag == "out":
                events.write(line)
                log.feed(line)
            else:
                errors.write(line)
                log.feed_stderr(line)
            if log.rate_limited:
                reason = "rate_limited"
                break
            over_tokens = reserve is not None and log.total_tokens > reserve
            if over_tokens or (reserve_usd is not None and log.cost_usd > reserve_usd):
                reason = "over_reserve"
                break
    _kill_tree(proc)
    try:
        code = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        code = None
    return log, reason, code


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def run_one(
    config: RunnerConfig, pin: RepoPin, task: Task, condition: str, repeat: int
) -> RunResult:
    """One run. Raises RateLimited, BudgetExhausted or AccountingError after saving it."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    # Absolute paths: PWD and B's --manifest are resolved by processes running in the checkout.
    run_dir = (config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}").resolve()
    checkout = export_commit(pin, run_dir / "checkout", config.workdir / "cache")
    seal_checkout(checkout)
    manifest = build_beacon(config, pin).resolve() if condition in ("B", "C") else None
    prompt = build_prompt(
        task,
        condition,
        manifest.read_text(encoding="utf-8") if manifest and condition == "C" else "",
    )
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    mcp = (
        beacon_server(config.beacon_python, manifest, config.beacon_src)
        if condition == "B" and manifest is not None
        else None
    )
    # --title skips OpenCode's title-generation request, whose tokens no event reports.
    cmd = [*config.opencode_cmd, "run", "--pure", "--print-logs", "--title", "beacon-eval",
           "-m", config.model, "--format", "json"]
    started = time.monotonic()
    keys = load_api_keys(config.env_file) if config.env_file else {}
    with isolated_config_home(
        config.config_source or default_config_source(), config.model, mcp, builtin_provider=bool(keys)
    ) as home:
        env = isolated_env(os.environ, home)
        env.update(keys)
        # An inherited PWD (Git Bash, MSYS, most shells) may root OpenCode in the
        # caller's repo instead of the checkout; the fake-provider check exercises this.
        env["PWD"] = str(checkout)
        log, reason, code = stream_opencode(
            cmd, prompt, checkout, env, run_dir, config.run_reserve_tokens, config.timeout_s,
            config.run_reserve_usd if config.budget_usd is not None else None,
        )
    text = "\n".join(log.texts)
    answer = extract_answer(text)
    error = reason or ("; ".join(log.errors) if log.errors else "")
    if not error and code not in (0, None):
        error = f"exit code {code}"
    result = RunResult(
        repo=task.repo,
        task_id=task.task_id,
        condition=condition,
        repeat=repeat,
        answer=answer,
        final_text=text,
        input_tokens=log.input_tokens,
        output_tokens=log.output_tokens,
        cache_read_tokens=log.cache_read_tokens,
        cache_write_tokens=log.cache_write_tokens,
        reported_total_tokens=log.reported_total_tokens,
        cost_usd=round(log.cost_usd, 6),
        tool_calls=log.tool_calls,
        seconds=round(time.monotonic() - started, 1),
        error=error[:500],
    )
    result.scores = score(answer, task.gold, checkout)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    _redact(run_dir, keys.values())
    if reason == "rate_limited":
        raise RateLimited(f"rate limited during {run_dir.name}; stopping, not retrying")
    if reason == "over_reserve":
        raise BudgetExhausted(
            f"{run_dir.name} passed its reserve ("
            + " / ".join(
                part
                for part in (
                    f"{config.run_reserve_tokens:,} tokens" if config.run_reserve_tokens else "",
                    f"${config.run_reserve_usd:.2f}" if config.budget_usd is not None else "",
                )
                if part
            )
            + ") and was killed"
        )
    if log.usage_events == 0:
        raise AccountingError(f"{run_dir.name} reported no token usage; stopping")
    if config.budget_usd is not None and log.cost_events == 0:
        raise AccountingError(f"{run_dir.name} reported no cost; the dollar cap cannot be kept")
    return result


def rescore(workdir: Path, tasks: list[Task]) -> list[RunResult]:
    """Recompute scores for every saved run from its answer and checkout (no model calls).

    Rewrites each ``result.json`` and ``results.jsonl``; runs whose task is not in *tasks*
    are left untouched and not returned.
    """
    by_id = {(task.repo, task.task_id): task for task in tasks}
    results: list[RunResult] = []
    for path in sorted((workdir / "runs").glob("*/result.json")):
        result = RunResult(**json.loads(path.read_text(encoding="utf-8")))
        task = by_id.get((result.repo, result.task_id))
        if task is None:
            continue
        result.scores = score(result.answer, task.gold, path.parent / "checkout")
        path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        results.append(result)
    (workdir / "results.jsonl").write_text(
        "".join(json.dumps(asdict(result)) + "\n" for result in results), encoding="utf-8"
    )
    return results


def _redact(run_dir: Path, secrets: Any) -> None:
    """Replace any key value that reached a saved file (logs can echo request errors)."""
    values = [value for value in secrets if len(value) >= 8]
    if not values:
        return
    for name in ("events.jsonl", "stderr.log", "result.json", "prompt.txt"):
        path = run_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        cleaned = text
        for value in values:
            cleaned = cleaned.replace(value, "<redacted>")
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")


def run_matrix(
    config: RunnerConfig,
    pins: dict[str, RepoPin],
    tasks: list[Task],
    conditions: tuple[str, ...] = CONDITIONS,
    repeats: int = 1,
) -> tuple[list[RunResult], str]:
    """Run every task x condition x repeat; returns results and why it stopped ("" = done).

    A run that stops the matrix is still recorded and counted against the budget.
    """
    budget = Budget(
        config.budget_tokens, config.run_reserve_tokens,
        cap_usd=config.budget_usd, reserve_usd=config.run_reserve_usd,
    )
    results: list[RunResult] = []
    log = config.workdir / "results.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(1, repeats + 1):
            for task in tasks:
                for condition in conditions:
                    budget.check()
                    try:
                        result = run_one(config, pins[task.repo], task, condition, repeat)
                    except (RateLimited, BudgetExhausted, AccountingError):
                        _record_stopped(config, task, condition, repeat, budget, results, log)
                        raise
                    budget.spend(result.total_tokens, result.cost_usd)
                    results.append(result)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(result)) + "\n")
    except (BudgetExhausted, RateLimited, AccountingError) as exc:
        return results, str(exc)
    return results, ""


def _record_stopped(
    config: RunnerConfig, task: Task, condition: str, repeat: int, budget: Budget,
    results: list[RunResult], log: Path,
) -> None:
    saved = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}" / "result.json"
    if not saved.is_file():
        return
    data = json.loads(saved.read_text(encoding="utf-8"))
    result = RunResult(**data)
    budget.spend(result.total_tokens, result.cost_usd)
    results.append(result)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data) + "\n")


__all__ = [
    "ANSWER_INSTRUCTIONS",
    "ANSWER_KEYS",
    "AccountingError",
    "Budget",
    "BudgetExhausted",
    "EventLog",
    "RateLimited",
    "RunnerConfig",
    "TOOL_NOTE",
    "build_prompt",
    "parse_events",
    "resolve_opencode",
    "run_matrix",
    "run_one",
    "seal_checkout",
    "stream_opencode",
]
