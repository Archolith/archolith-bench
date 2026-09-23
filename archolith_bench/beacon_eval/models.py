"""Task, gold-answer and run-result records, loaded from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The answer every agent must end with, so scoring needs no judge.
ANSWER_KEYS = ("docs", "files", "commands", "guardrails", "verdict", "plan", "citations")


@dataclass(frozen=True)
class RepoPin:
    name: str
    url: str
    commit: str
    local_path: str = ""  # a local checkout to export from instead of cloning


@dataclass(frozen=True)
class Gold:
    """What a correct answer contains. Every expected item cites path and lines."""

    docs: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()  # keywords that must appear in a named guardrail
    verdict: str = ""  # e.g. "current" / "superseded" for stale-document tasks
    risky: tuple[str, ...] = ()  # substrings that must NOT appear (destructive or out-of-bounds)
    #: Flags an answer may add after a gold command (e.g. "-x"); any other added flag fails.
    allowed_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    repo: str
    task_id: str
    kind: str  # docs_and_files | commands_and_guardrails | decision | stale_doc | first_plan
    prompt: str
    gold: Gold
    reviewed: bool = False  # the owner approved this gold answer


@dataclass
class RunResult:
    repo: str
    task_id: str
    condition: str
    repeat: int
    answer: dict[str, Any] | None
    final_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    seconds: float = 0.0
    error: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    #: Sum of OpenCode's own ``tokens.total`` per step, when it reports one.
    reported_total_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        computed = (
            self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
        )
        return max(self.reported_total_tokens, computed)


def load_repos(path: Path) -> dict[str, RepoPin]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: RepoPin(**item) for item in data["repos"]}


def load_task(path: Path) -> Task:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold = data.get("gold") or {}
    return Task(
        repo=data["repo"],
        task_id=data["task_id"],
        kind=data["kind"],
        prompt=data["prompt"],
        reviewed=bool(data.get("reviewed", False)),
        gold=Gold(
            docs=tuple(gold.get("docs", ())),
            files=tuple(gold.get("files", ())),
            commands=tuple(gold.get("commands", ())),
            guardrails=tuple(gold.get("guardrails", ())),
            verdict=str(gold.get("verdict", "")),
            risky=tuple(gold.get("risky", ())),
            allowed_flags=tuple(gold.get("allowed_flags", ())),
        ),
    )


def load_tasks(root: Path, repos: tuple[str, ...] = (), task_ids: tuple[str, ...] = ()) -> list[Task]:
    tasks = [load_task(path) for path in sorted(root.glob("*/*.json"))]
    return [
        task
        for task in tasks
        if (not repos or task.repo in repos) and (not task_ids or task.task_id in task_ids)
    ]
