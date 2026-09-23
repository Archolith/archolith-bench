"""Run the task x condition x repeat matrix through ``opencode run --format json``.

Every run gets a fresh export of the repository at its pinned commit (condition A
never sees a beacon), a private stripped OpenCode config, and a fixed prompt. Token
use is read from OpenCode's JSON events; the matrix stops before the budget cap would
be exceeded and at the first rate-limit error (never retrying).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from archolith_bench.beacon_eval import CONDITIONS
from archolith_bench.beacon_eval.isolation import beacon_overlay, stripped_config
from archolith_bench.beacon_eval.models import ANSWER_KEYS, RepoPin, RunResult, Task
from archolith_bench.beacon_eval.scoring import extract_answer, score

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BUDGET_TOKENS = 20_000_000
#: Assumed cost of a run before any run has been measured.
INITIAL_RUN_ESTIMATE = 80_000

ANSWER_INSTRUCTIONS = (
    "Work read-only: do not edit, create or delete files, and do not run tests or installs. "
    "When you are done, end your reply with one fenced ```json block containing exactly "
    'these keys: "docs" (paths of documents to read), "files" (source files involved), '
    '"commands" (exact shell commands), "guardrails" (rules that apply, each a short '
    'sentence), "verdict" (a single word when the task asks for one, else ""), "plan" '
    '(ordered steps), "citations" (a list of {"path", "line_start", "line_end"} backing '
    "your claims)."
)


class RateLimited(RuntimeError):
    """The provider refused with a rate limit; the matrix stops, nothing is retried."""


class BudgetExhausted(RuntimeError):
    """The next run could exceed the token cap."""


@dataclass
class RunnerConfig:
    workdir: Path
    beacon_python: str
    beacon_src: str | None = None
    opencode_cmd: list[str] = field(default_factory=lambda: ["opencode"])
    model: str = DEFAULT_MODEL
    config_source: Path | None = None
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    timeout_s: float = 900.0


@dataclass
class Budget:
    cap: int
    used: int = 0
    largest_run: int = INITIAL_RUN_ESTIMATE

    def check(self) -> None:
        if self.used + self.largest_run > self.cap:
            raise BudgetExhausted(
                f"next run could exceed the {self.cap:,}-token cap ({self.used:,} used)"
            )

    def spend(self, tokens: int) -> None:
        self.used += tokens
        self.largest_run = max(self.largest_run, tokens)


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
    parts = [task.prompt.strip(), ANSWER_INSTRUCTIONS]
    if condition == "B":
        parts.insert(1, "A Beacon MCP server for this project is available to you.")
    if condition == "C":
        parts.insert(
            0,
            "Project knowledge (beacon.generated.yaml) follows.\n```yaml\n"
            + manifest_text.strip()
            + "\n```",
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# OpenCode events
# ---------------------------------------------------------------------------


def parse_events(stdout: str) -> dict[str, Any]:
    """Final text, token use and tool-call count from ``--format json`` output.

    The event shape is read defensively: text parts, any ``tokens`` object with
    ``input``/``output`` counts, and tool events are collected wherever they appear.
    """
    texts: list[str] = []
    input_tokens = output_tokens = tool_calls = 0
    rate_limited = False

    def walk(node: Any) -> None:
        nonlocal input_tokens, output_tokens, tool_calls
        if isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                texts.append(node["text"])
            if node.get("type") in ("tool", "tool_use", "tool-call", "tool_call"):
                tool_calls += 1
            tokens = node.get("tokens")
            if isinstance(tokens, dict):
                input_tokens += int(tokens.get("input") or 0)
                output_tokens += int(tokens.get("output") or 0) + int(
                    tokens.get("reasoning") or 0
                )
            for value in node.values():
                if isinstance(value, dict | list):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for line in stdout.splitlines():
        lowered = line.lower()
        if '"429"' in lowered or " 429" in lowered or "rate limit" in lowered:
            rate_limited = True
        try:
            walk(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {
        "text": "\n".join(texts),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "rate_limited": rate_limited,
    }


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def run_one(
    config: RunnerConfig, pin: RepoPin, task: Task, condition: str, repeat: int
) -> RunResult:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    run_dir = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}"
    checkout = export_commit(pin, run_dir / "checkout", config.workdir / "cache")
    manifest = build_beacon(config, pin) if condition in ("B", "C") else None
    prompt = build_prompt(
        task,
        condition,
        manifest.read_text(encoding="utf-8") if manifest and condition == "C" else "",
    )
    env = dict(os.environ)
    source = config.config_source or Path.home() / ".config" / "opencode"
    started = time.monotonic()
    with stripped_config(source, run_dir) as config_dir:
        env["OPENCODE_CONFIG_DIR"] = str(config_dir)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        if condition == "B" and manifest is not None:
            env["OPENCODE_CONFIG_CONTENT"] = beacon_overlay(
                config.beacon_python, manifest, config.beacon_src
            )
        try:
            completed = subprocess.run(
                [*config.opencode_cmd, "run", "-m", config.model, "--format", "json", prompt],
                cwd=checkout,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=config.timeout_s,
            )
            stdout, error = completed.stdout, completed.stderr[-500:] if completed.returncode else ""
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            error = "timeout"
    events = parse_events(stdout)
    if events["rate_limited"]:
        raise RateLimited(f"rate limited during {run_dir.name}; stopping, not retrying")
    answer = extract_answer(events["text"])
    result = RunResult(
        repo=task.repo,
        task_id=task.task_id,
        condition=condition,
        repeat=repeat,
        answer=answer,
        final_text=events["text"][-4000:],
        input_tokens=events["input_tokens"],
        output_tokens=events["output_tokens"],
        tool_calls=events["tool_calls"],
        seconds=round(time.monotonic() - started, 1),
        error=error,
    )
    result.scores = score(answer, task.gold, checkout)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def run_matrix(
    config: RunnerConfig,
    pins: dict[str, RepoPin],
    tasks: list[Task],
    conditions: tuple[str, ...] = CONDITIONS,
    repeats: int = 1,
) -> tuple[list[RunResult], str]:
    """Run every task x condition x repeat; returns results and why it stopped ("" = done)."""
    budget = Budget(config.budget_tokens)
    results: list[RunResult] = []
    log = config.workdir / "results.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        for repeat in range(1, repeats + 1):
            for task in tasks:
                for condition in conditions:
                    budget.check()
                    result = run_one(config, pins[task.repo], task, condition, repeat)
                    budget.spend(result.total_tokens)
                    results.append(result)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(result)) + "\n")
    except (BudgetExhausted, RateLimited) as exc:
        return results, str(exc)
    return results, ""


__all__ = [
    "ANSWER_INSTRUCTIONS",
    "ANSWER_KEYS",
    "Budget",
    "BudgetExhausted",
    "RateLimited",
    "RunnerConfig",
    "build_prompt",
    "parse_events",
    "run_matrix",
    "run_one",
]
