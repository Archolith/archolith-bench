"""Live OpenCode executor for the 3-arm x 4-session Longitudinal Frozen-Repo Sprint.

Reuses `archolith_bench.beacon_eval`'s pinned repo export, per-session isolated
OpenCode config homes, managed Menhir MCP config, streaming token/429 guards,
and deterministic JSON answer contract.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
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
from archolith_bench.engineering_evidence.server_usage import (
    ServerRates,
    attribute_server_usage,
    backend_post,
    list_namespace_episodes,
    read_usage_rows,
    wait_for_enrichment_drain,
)
from archolith_bench.engineering_evidence.sprint import (
    DriftPatch,
    SessionExecution,
    SprintFixture,
    SprintSession,
    compose_full_prompt,
    diff_snapshots,
    snapshot_tree,
)

#: Arms that talk to Menhir. `notes_file` and `no_menhir` get no MCP server at all.
MENHIR_ARMS = frozenset({"menhir_structure", "menhir_longitudinal"})

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


#: Scan meta keys `scan_and_write_project` returns (menhir backend_runtime_data_ops).
_SCAN_COUNT_KEYS = (
    "dirs", "files", "deps", "endpoints", "imports", "test_edges", "cross_refs", "symbols", "call_edges",
)


def short_checkout_id(fixture: SprintFixture, arm: str, repeat: int) -> str:
    """``c`` + 6 hex chars, stable per (fixture, arm, repeat); checkouts.json maps it back."""
    return "c" + hashlib.sha256(f"{fixture.name}|{arm}|{repeat}".encode("utf-8")).hexdigest()[:6]


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
    #: Where per-session run dirs go (default: ``work_dir / "runs"``).
    runs_dir: Path | None = None
    #: Where per-chain checkouts go (default: ``work_dir / "checkouts"``).
    checkout_dir: Path | None = None
    #: Menhir's `MENHIR_MCP_TELEMETRY_DB` sqlite file, for server-side usage.
    telemetry_db: Path | None = None
    #: v2 campaigns wait up to this long before each Menhir session after S1 (0 disables).
    enrichment_timeout_s: float = 180.0
    enrichment_poll_s: float = 3.0
    #: How long index_pristine waits for the background structure write to become visible.
    index_timeout_s: float = 600.0
    #: One OpenAI prompt_cache_key per (fixture, arm) instead of OpenCode's per-session key, and
    #: arm-wide prompt text ahead of session text (sprint.compose_full_prompt).
    cache_shaping: bool = False
    #: Replace each session's events.jsonl with events.jsonl.gz once parsed.
    gzip_events: bool = False


def _notes_violations(notes_before: dict[str, str] | None, checkout: Path) -> list[str]:
    """Writes outside the notes file during a notes_file session (empty for other arms)."""
    return diff_snapshots(notes_before, snapshot_tree(checkout)) if notes_before is not None else []


def _with_notes_error(error: str, notes_violations: list[str]) -> str:
    if not notes_violations:
        return error
    return "; ".join(part for part in (error, "notes_file_violation=" + ",".join(notes_violations)) if part)


class OpenCodeSprintExecutor:
    """Executes chained sprint sessions through isolated `opencode run --format json`.

    Each chain gets its own exported frozen checkout (persisting `drift_patch` edits
    between sessions) while every individual session gets a fresh `isolated_config_home`
    so the LLM always starts with an empty context window. A chain is `(arm, repeat)`;
    the old single-run path passes ``repeat=None`` and keeps its old paths. Chains may run
    concurrently from several threads: per-chain state is keyed by `(arm, repeat)` and
    shared state (checkout creation, the memory-readiness check) is locked.
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
        post_fn: Callable[..., Any] = backend_post,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._stream_fn = stream_fn
        self._export_fn = export_fn
        self._resolve_cmd_fn = resolve_cmd_fn
        self._check_memory_fn = check_memory_fn
        self._notify_file_dirty_fn = notify_file_dirty_fn
        self._post_fn = post_fn
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn
        self._checkouts: dict[tuple[str, int | None], Path] = {}
        self._memory_checked = False
        self._lock = threading.Lock()
        self._export_lock = threading.Lock()

    # -- per-chain state -------------------------------------------------------------

    def _ensure_checkout(
        self, fixture: SprintFixture, arm: str, repeat: int | None = None
    ) -> Path:
        key = (arm, repeat)
        with self._lock:
            if key in self._checkouts:
                return self._checkouts[key]
        root = self.config.checkout_dir or (self.config.work_dir / "checkouts")
        if repeat is None:
            dest = root / f"{fixture.name}-{arm}"
        else:
            # Short and non-repeating: run v2r2-20261127 lost sessions in every arm to agents
            # mistyping long checkout paths that repeated the fixture name at two levels.
            dest = root / short_checkout_id(fixture, arm, repeat)
        with self._export_lock:  # the repo cache is shared; clone and archive one at a time
            checkout = self._export_fn(fixture.repo, dest, self.config.cache_dir)
        with self._lock:
            self._checkouts[key] = checkout
            if repeat is not None:
                mapping = {f"{a}/r{r}": str(p) for (a, r), p in self._checkouts.items() if r is not None}
                (root / "checkouts.json").write_text(json.dumps(mapping, indent=2, sort_keys=True),
                                                     encoding="utf-8")
        return checkout

    def _run_dir(self, arm: str, repeat: int | None, session_index: int, session_id: str) -> Path:
        base = self.config.runs_dir or (self.config.work_dir / "runs")
        chain = base / arm / (f"r{repeat}" if repeat is not None else "")
        return chain / f"{session_index:02d}-{session_id}"

    def _mcp_for_arm(self, arm: str) -> dict[str, Any] | None:
        if arm not in MENHIR_ARMS:
            return None
        if not self.config.menhir_mcp_url:
            raise ValueError(f"arm {arm!r} requires --menhir-mcp-url")
        with self._lock:
            checked = self._memory_checked
        if not checked:
            self._check_memory_fn(self.config.menhir_mcp_url)
            with self._lock:
                self._memory_checked = True
        return memory_server(self.config.menhir_mcp_url)

    # -- Menhir server side ----------------------------------------------------------

    def _post(self, operation: str, body: dict[str, Any]) -> Any:
        return self._post_fn(
            self.config.menhir_mcp_url, operation, body, api_key=self.config.menhir_api_key
        )

    def wait_for_enrichment(self, namespace: str) -> dict[str, Any]:
        """Wait (up to `enrichment_timeout_s`) until *namespace* has no pending enrichment."""
        return wait_for_enrichment_drain(
            self._post_fn,
            self.config.menhir_mcp_url,
            namespace,
            api_key=self.config.menhir_api_key,
            timeout_s=self.config.enrichment_timeout_s,
            poll_s=self.config.enrichment_poll_s,
            sleep_fn=self._sleep_fn,
            clock_fn=self._clock_fn,
        )

    def index_pristine(self, fixture: SprintFixture, run_id: str) -> dict[str, Any]:
        """Write the structural index once per fixture from a pristine export.

        Exports the pinned commit to its own directory (never a chain checkout, so drift
        patches cannot touch it) and has the server scan it with `scan_and_write_project`.
        """
        root = (self.config.checkout_dir or (self.config.work_dir / "checkouts")).parent
        dest = root / "p"
        with self._export_lock:
            pristine = self._export_fn(fixture.repo, dest, self.config.cache_dir)
        body = {
            "path": str(Path(pristine).resolve()),
            "name": fixture.repo.name,
            "force": True,
            "session_id": f"sprint-v2-{run_id}",
            "user_id": "archolith-bench-sprint",
            "force_identity": False,
            "identity_action": None,
            "adopt_project_id": None,
        }
        result = self._post("scan_and_write_project", body)
        result = result if isinstance(result, dict) else {}
        identity_action = None
        # A fresh export is bound to no project, so the server asks for an identity decision
        # instead of scanning; on a disposable graph with no candidates the answer is "new".
        if result.get("status") == "needs_decision" and not result.get("candidates"):
            identity_action = "new"
            result = self._post("scan_and_write_project", {**body, "identity_action": "new"})
            result = result if isinstance(result, dict) else {}
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        counts = {k: meta[k] for k in _SCAN_COUNT_KEYS if isinstance(meta.get(k), int)}
        if not result.get("skipped") and not counts.get("files"):
            # Fail closed: a Menhir arm without its structure index is not the arm we measure.
            raise RuntimeError(f"structural index for {fixture.repo.name} was not written: {result!r}")
        # The server writes the graph in the background; S1 must not start before it lands.
        start = self._clock_fn()
        while not self._post("get_scan_fingerprint", {"project_name": fixture.repo.name}):
            if self._clock_fn() - start > self.config.index_timeout_s:
                raise RuntimeError(f"structural index for {fixture.repo.name} not visible after "
                                   f"{self.config.index_timeout_s}s")
            self._sleep_fn(2.0)
        return {
            "path": str(Path(pristine).resolve()),
            "project": fixture.repo.name,
            "identity_action": identity_action,
            "counts": counts,
            "skipped": bool(result.get("skipped", False)),
        }

    def server_usage(
        self,
        namespaces: list[str],
        *,
        since_iso: str | None = None,
        rates: ServerRates | None = None,
    ) -> dict[str, Any] | None:
        """Server LLM usage per Menhir namespace, or None when no telemetry DB is configured."""
        db = self.config.telemetry_db
        if db is None or not Path(db).is_file():
            return None
        rows = read_usage_rows(Path(db), since_iso=since_iso)
        episodes: dict[str, list[str]] | None = {}
        details: dict[str, dict[str, Any]] = {}
        listing_error = ""
        try:
            for ns in namespaces:
                info = list_namespace_episodes(
                    self._post_fn, self.config.menhir_mcp_url, ns, api_key=self.config.menhir_api_key
                )
                episodes[ns] = info["uuids"]
                details[ns] = {
                    "episodes": len(info["uuids"]),
                    "failed_episodes": info["failed_episodes"],
                    "episode_list_truncated": info["truncated"],
                }
        except Exception as exc:  # noqa: BLE001 - fall back to the documented even split
            episodes = None
            details = {}
            listing_error = f"{type(exc).__name__}: {exc}"[:200]
        out = attribute_server_usage(rows, episodes, rates=rates, namespaces=namespaces)
        for ns, extra in details.items():
            out["by_namespace"][ns].update(extra)
        if listing_error:
            out["episode_listing_error"] = listing_error
        return out

    # -- one session -----------------------------------------------------------------

    def run_session(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        session_index: int,
        namespace: str,
        repeat: int | None = None,
    ) -> SessionExecution:
        checkout = self._ensure_checkout(fixture, arm, repeat)
        uses_menhir = arm in MENHIR_ARMS

        wait_info: dict[str, Any] = {"waited_s": 0.0, "drained": True}
        if (
            uses_menhir
            and repeat is not None
            and session_index > 0
            and self.config.menhir_mcp_url
            and self.config.enrichment_timeout_s > 0
        ):
            wait_info = self.wait_for_enrichment(namespace)

        if session.drift_patch is not None:
            apply_drift_patch(checkout, session.drift_patch)
            if uses_menhir and self.config.menhir_mcp_url:
                self._notify_file_dirty_fn(
                    self.config.menhir_mcp_url,
                    path=session.drift_patch.path,
                    project=fixture.repo.name,
                    api_key=self.config.menhir_api_key,
                )

        mcp = self._mcp_for_arm(arm)
        run_dir = self._run_dir(arm, repeat, session_index, session.session_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        full_prompt = compose_full_prompt(session, arm, namespace, self.config.cache_shaping)
        (run_dir / "prompt.txt").write_text(full_prompt, encoding="utf-8")

        notes_before = snapshot_tree(checkout) if arm == "notes_file" else None
        return self._run_agent(
            fixture=fixture, session=session, arm=arm, checkout=checkout, mcp=mcp, run_dir=run_dir,
            full_prompt=full_prompt, notes_before=notes_before, wait_info=wait_info,
        )

    def _run_agent(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        checkout: Path,
        mcp: dict[str, Any] | None,
        run_dir: Path,
        full_prompt: str,
        notes_before: dict[str, str] | None,
        wait_info: dict[str, Any],
    ) -> SessionExecution:
        """Run the agent for one prepared session (OpenCode here; subclasses swap the agent)."""
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
            # One key per arm and fixture: every session of the arm shares the system prompt and
            # tool prefix, which OpenCode's per-session default key never reuses.
            prompt_cache_key=f"sv2-{fixture.repo.name}-{arm}" if self.config.cache_shaping else None,
        ) as config_home:
            env = isolated_env(extra_env, config_home)
            # An inherited PWD (Git Bash, MSYS) roots OpenCode in the caller's repo instead of
            # the checkout (same fix as beacon_eval.runner); test_sprint_v2 pins it.
            env["PWD"] = str(checkout)
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
        if self.config.gzip_events and events_path.is_file():
            with gzip.open(run_dir / "events.jsonl.gz", "wt", encoding="utf-8") as gz:
                gz.write(events_text)
            events_path.unlink()

        notes_violations = _notes_violations(notes_before, checkout)

        stderr_path = run_dir / "stderr.log"
        refused_reads = (
            stderr_path.read_text(encoding="utf-8", errors="replace").count("auto-rejecting")
            if stderr_path.is_file()
            else 0
        )

        error = ""
        if stop_reason:
            error = f"stop_reason={stop_reason}"
        elif exit_code not in (0, None):
            error = f"exit_code={exit_code}"
        error = _with_notes_error(error, notes_violations)

        return SessionExecution(
            response_text="\n".join(log.texts),
            input_tokens=log.input_tokens,
            output_tokens=max(log.output_tokens - log.reasoning_tokens, 0),
            reasoning_tokens=log.reasoning_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            cost_usd=log.cost_usd,
            censored=stop_reason in ("over_reserve", "timeout"),
            turns=log.usage_events,
            tool_calls=log.tool_calls,
            files_read=telemetry["files_read"],
            memories_saved=telemetry["memories_saved"],
            todos_saved=telemetry["todos_saved"],
            seconds=elapsed,
            error=error,
            repo_root=checkout,
            notes_violations=notes_violations,
            refused_reads=refused_reads,
            enrichment_wait_s=float(wait_info.get("waited_s", 0.0)),
            enrichment_drained=bool(wait_info.get("drained", True)),
            run_dir=run_dir,
        )
