"""Live OpenCode executor for the 3-arm x 4-session Longitudinal Frozen-Repo Sprint.

Reuses `archolith_bench.beacon_eval`'s pinned repo export, per-session isolated
OpenCode config homes, managed Menhir MCP config, streaming token/429 guards,
and deterministic JSON answer contract.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from archolith_bench.beacon_eval.isolation import (
    MEMORY_KEY_ENV,
    default_config_source,
    isolated_config_home,
    isolated_env,
    load_api_keys,
    memory_server,
)
from archolith_bench.beacon_eval.runner import (
    ANSWER_INSTRUCTIONS,
    DEFAULT_MODEL,
    DEFAULT_RUN_RESERVE,
    BudgetExhausted,
    EventLog,
    RateLimited,
    check_memory_ready,
    export_commit,
    resolve_opencode,
    stream_opencode,
)
from archolith_bench.engineering_evidence.sprint import (
    DriftPatch,
    SessionExecution,
    SprintFixture,
    SprintSession,
    build_session_prompt,
)

_READ_TOOL_NAMES = frozenset({"read", "read_file", "view_file", "cat"})
_MENHIR_TOOL_SUBSTRINGS = (
    "query_structure",
    "blast_radius",
    "affected_tests",
    "recall_memories",
    "recall_context_memories",
    "add_memory",
    "add_todo",
    "list_todos",
    "get_provenance",
)


def apply_drift_patch(repo_root: Path, patch: DriftPatch) -> Path:
    """Apply a deterministic single-replacement drift patch inside *repo_root*.

    Fails closed if the target file is outside *repo_root*, does not exist,
    or does not contain the exact *patch.find* text.
    """
    resolved_root = repo_root.resolve()
    target = (resolved_root / patch.path).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"drift patch path escapes repo root: {patch.path}") from exc

    if not target.is_file():
        raise FileNotFoundError(f"drift patch target missing: {patch.path}")

    original = target.read_text(encoding="utf-8")
    if patch.find not in original:
        raise ValueError(
            f"drift patch anchor text not found in {patch.path}: {patch.find[:80]!r}"
        )

    updated = original.replace(patch.find, patch.replace, 1)
    target.write_text(updated, encoding="utf-8")
    return target


def notify_menhir_file_changed(
    mcp_url: str,
    *,
    path: str,
    project: str,
    api_key: str = "",
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Notify Menhir's `/api/tool-events` hook that *path* changed after anchoring."""
    parts = urlsplit(mcp_url)
    origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    url = f"{origin}/api/tool-events"
    payload = json.dumps(
        {
            "event_type": "file_changed",
            "operation": "edit",
            "path": path,
            "project": project,
            "source_client": "archolith-bench-sprint",
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_repo_path(raw_path: str, repo_root: Path | None) -> str:
    cleaned = raw_path.strip().replace("\\", "/").rstrip("/")
    if repo_root is not None:
        root_norm = str(repo_root.resolve()).replace("\\", "/").rstrip("/")
        if cleaned.lower() == root_norm.lower():
            return ""
        if cleaned.lower().startswith(root_norm.lower() + "/"):
            cleaned = cleaned[len(root_norm) + 1 :]
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def parse_sprint_tool_telemetry(
    events_jsonl: str, repo_root: Path | None = None
) -> dict[str, Any]:
    """Extract `files_read`, `memories_saved`, `todos_saved`, and Menhir tool counts
    from an OpenCode `--format json` NDJSON event stream."""
    files_read: list[str] = []
    seen_files: set[str] = set()
    memories_saved = 0
    todos_saved = 0
    menhir_tools: dict[str, int] = {}

    for line in events_jsonl.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(evt, dict) or evt.get("type") != "tool_use":
            continue
        part = evt.get("part") or {}
        if not isinstance(part, dict):
            continue
        tool_name = str(part.get("tool") or part.get("name") or "").strip()
        tool_lower = tool_name.lower()
        state = part.get("state") or {}
        tool_input = state.get("input") if isinstance(state, dict) else None
        if not isinstance(tool_input, dict):
            tool_input = part.get("input") if isinstance(part.get("input"), dict) else {}

        # Track file reads (built-in OpenCode `read` uses `filePath` or `path`)
        if tool_lower in _READ_TOOL_NAMES or tool_lower.endswith("_read"):
            raw_fp = (
                tool_input.get("filePath")
                or tool_input.get("path")
                or tool_input.get("file_path")
                or ""
            )
            if isinstance(raw_fp, str) and raw_fp.strip():
                norm = _normalize_repo_path(raw_fp, repo_root)
                if norm and norm not in seen_files:
                    seen_files.add(norm)
                    files_read.append(norm)

        # Track Menhir MCP tool usage
        for suffix in _MENHIR_TOOL_SUBSTRINGS:
            if tool_lower == suffix or tool_lower.endswith("_" + suffix) or tool_lower.endswith(":" + suffix):
                menhir_tools[suffix] = menhir_tools.get(suffix, 0) + 1
                if suffix == "add_memory":
                    memories_saved += 1
                elif suffix == "add_todo":
                    todos_saved += 1
                break

    return {
        "files_read": files_read,
        "memories_saved": memories_saved,
        "todos_saved": todos_saved,
        "menhir_tools": menhir_tools,
    }


@dataclass
class LiveSprintConfig:
    """Configuration for live OpenCode execution of a longitudinal sprint."""

    work_dir: Path
    cache_dir: Path
    model: str = DEFAULT_MODEL
    menhir_mcp_url: str = ""
    menhir_api_key: str = ""
    env_file: Path | None = None
    source_config: Path = field(default_factory=default_config_source)
    reserve_tokens: int = DEFAULT_RUN_RESERVE
    timeout_s: float = 300.0
    builtin_provider: bool = True


class OpenCodeSprintExecutor:
    """Executes chained sprint sessions through isolated `opencode run --format json`.

    Each arm gets its own exported frozen checkout (persisting `drift_patch` edits
    between sessions) while every individual session gets a fresh `isolated_config_home`
    so the LLM always starts with an empty context window.
    """

    def __init__(
        self,
        config: LiveSprintConfig,
        *,
        stream_fn: Callable[..., tuple[EventLog, str, int | None]] = stream_opencode,
        export_fn: Callable[..., Path] = export_commit,
        resolve_cmd_fn: Callable[[], list[str]] = resolve_opencode,
        check_memory_fn: Callable[[str], None] = check_memory_ready,
        notify_file_dirty_fn: Callable[..., dict[str, Any]] = notify_menhir_file_changed,
    ) -> None:
        self.config = config
        self._stream_fn = stream_fn
        self._export_fn = export_fn
        self._resolve_cmd_fn = resolve_cmd_fn
        self._check_memory_fn = check_memory_fn
        self._notify_file_dirty_fn = notify_file_dirty_fn
        self._arm_checkouts: dict[str, Path] = {}
        self._memory_checked = False

    def _ensure_checkout(self, fixture: SprintFixture, arm: str) -> Path:
        if arm in self._arm_checkouts:
            return self._arm_checkouts[arm]
        dest = self.config.work_dir / "checkouts" / f"{fixture.name}-{arm}"
        checkout = self._export_fn(fixture.repo, dest, self.config.cache_dir)
        self._arm_checkouts[arm] = checkout
        return checkout

    def _mcp_for_arm(self, arm: str) -> dict[str, Any] | None:
        if arm == "no_menhir":
            return None
        if not self.config.menhir_mcp_url:
            raise ValueError(f"arm {arm!r} requires --menhir-mcp-url")
        if not self._memory_checked:
            self._check_memory_fn(self.config.menhir_mcp_url)
            self._memory_checked = True
        return memory_server(self.config.menhir_mcp_url)

    def run_session(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        session_index: int,
        namespace: str,
    ) -> SessionExecution:
        checkout = self._ensure_checkout(fixture, arm)
        if session.drift_patch is not None:
            apply_drift_patch(checkout, session.drift_patch)
            if arm != "no_menhir" and self.config.menhir_mcp_url:
                self._notify_file_dirty_fn(
                    self.config.menhir_mcp_url,
                    path=session.drift_patch.path,
                    project=fixture.repo.name,
                    api_key=self.config.menhir_api_key,
                )

        mcp = self._mcp_for_arm(arm)
        run_dir = self.config.work_dir / "runs" / arm / f"{session_index:02d}-{session.session_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        full_prompt = (
            build_session_prompt(session, arm)
            + (
                f"\n\nUse Menhir namespace `{namespace}` for all memory/TODO operations."
                if arm == "menhir_longitudinal"
                else ""
            )
            + "\n\n"
            + ANSWER_INSTRUCTIONS
        )
        (run_dir / "prompt.txt").write_text(full_prompt, encoding="utf-8")

        cmd = [*self._resolve_cmd_fn(), "run", "--format", "json"]
        extra_env = dict(os.environ)
        if self.config.env_file and self.config.env_file.is_file():
            extra_env.update(load_api_keys(self.config.env_file))
        if self.config.menhir_api_key:
            extra_env[MEMORY_KEY_ENV] = self.config.menhir_api_key

        t0 = time.monotonic()
        with isolated_config_home(
            self.config.source_config,
            self.config.model,
            mcp=mcp,
            builtin_provider=self.config.builtin_provider,
        ) as config_home:
            env = isolated_env(extra_env, config_home)
            log, stop_reason, exit_code = self._stream_fn(
                cmd,
                full_prompt,
                checkout,
                env,
                run_dir,
                self.config.reserve_tokens,
                self.config.timeout_s,
            )
        elapsed = time.monotonic() - t0

        if stop_reason == "rate_limited" or log.rate_limited:
            raise RateLimited(
                f"rate limit hit during {arm}/{session.session_id}; stopping fail-closed"
            )
        # On over_reserve, record the killed session honestly (error='over_reserve', actual
        # tokens/turns burned) instead of aborting the other arms in a multi-arm A/B run.

        events_path = run_dir / "events.jsonl"
        events_text = events_path.read_text(encoding="utf-8") if events_path.is_file() else ""
        telemetry = parse_sprint_tool_telemetry(events_text, repo_root=checkout)

        error = ""
        if stop_reason:
            error = f"stop_reason={stop_reason}"
        elif exit_code not in (0, None):
            error = f"exit_code={exit_code}"

        return SessionExecution(
            response_text="\n".join(log.texts),
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            turns=log.usage_events,
            tool_calls=log.tool_calls,
            files_read=telemetry["files_read"],
            memories_saved=telemetry["memories_saved"],
            todos_saved=telemetry["todos_saved"],
            seconds=elapsed,
            error=error,
            repo_root=checkout,
        )
