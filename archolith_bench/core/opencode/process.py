"""OpenCode process handling: launching, streaming, killing and parsing a run.

Event parsing, process-tree cleanup, the interrupt (cancel) scope, the OpenCode executable
lookup and the memory-backend readiness check. Nothing here knows a benchmark's tasks,
budgets or conditions.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from archolith_bench.core.ratelimit import mentions_capacity

#: Owner decision 2026-09-23 (GPT-6 Luna via OpenAI; key through --env-file).
DEFAULT_MODEL = "openai/gpt-6-luna"

_RATE_LIMIT = re.compile(r"\b429\b|rate[ _-]?limit|too many requests", re.IGNORECASE)


class RateLimited(RuntimeError):
    """The provider refused with a rate limit; the matrix stops, nothing is retried."""


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
    #: Reasoning tokens, already included in ``output_tokens`` (kept for a split view).
    reasoning_tokens: int = 0
    reported_total_tokens: int = 0
    usage_events: int = 0
    cost_usd: float = 0.0
    cost_events: int = 0
    tool_calls: int = 0
    session_id: str = ""
    last_step_reason: str = ""
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
        if isinstance(event.get("sessionID"), str) and not self.session_id:
            self.session_id = event["sessionID"]
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
            self.last_step_reason = str(part.get("reason") or "")
            if isinstance(part.get("cost"), int | float):
                self.cost_usd += float(part["cost"])
                self.cost_events += 1
        elif kind not in ("step_start", "reasoning"):
            self.unknown_types.add(str(kind))

    def feed_stderr(self, line: str) -> None:
        self._scan(line)

    def _scan(self, text: str) -> None:
        # Flex capacity refusals are retried by OpenCode; they are not a quota or rate stop.
        if _RATE_LIMIT.search(text) and not mentions_capacity(text):
            self.rate_limited = True

    def _add_usage(self, tokens: Any) -> None:
        if not isinstance(tokens, dict):
            return
        self.usage_events += 1
        self.input_tokens += int(tokens.get("input") or 0)
        self.output_tokens += int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
        self.reasoning_tokens += int(tokens.get("reasoning") or 0)
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


#: POSIX children get this long to exit on TERM before their whole group gets KILL.
KILL_GRACE_S = 5.0
#: How long _kill_tree waits for a tree to die before giving up on reaping it, and for
#: ``taskkill /T /F`` to come back on Windows.
KILL_WAIT_S = 15.0
#: How long _kill_tree waits for a pump thread to see EOF before leaving it behind.
PUMP_JOIN_S = 5.0
#: After an interrupt, how long the parallel matrix keeps collecting the killed runs'
#: results before it leaves. This bounds only that collection: the executor's shutdown
#: afterwards still waits for the worker threads, so it does not bound the whole shutdown.
DRAIN_TIMEOUT_S = 60.0

#: Processes a run started and has not reaped yet, so an interrupt can end them now
#: instead of waiting for their runs to finish (see _run_parallel).
_live_children: set[subprocess.Popen[str]] = set()
_live_children_lock = threading.Lock()

#: Set while an interrupted parallel matrix is cancelling: checked before a child is
#: registered, before OpenCode or a server is spawned, and before a session resume, so
#: anything that would start after the interrupt dies at once instead of running on.
_cancel_guard = threading.Lock()
_cancel_event: threading.Event | None = None


@contextmanager
def _cancel_scope() -> Iterator[threading.Event]:
    """Publish the matrix's cancel event for the code running in this process."""
    global _cancel_event
    event = threading.Event()
    with _cancel_guard:
        _cancel_event = event
    try:
        yield event
    finally:
        with _cancel_guard:
            _cancel_event = None


def _cancelled() -> bool:
    with _cancel_guard:
        return _cancel_event is not None and _cancel_event.is_set()


def _register_child(proc: subprocess.Popen[str]) -> None:
    """Remember a spawned run process for :func:`_kill_live_children`.

    A child that registers after the matrix was interrupted is killed at once, so a run
    straggling past the interrupt cannot leave a live tree behind (a one-time kill sweep
    would miss it).
    """
    with _live_children_lock:
        _live_children.add(proc)
    if _cancelled():
        _kill_tree(proc)


def _unregister_child(proc: subprocess.Popen[str]) -> None:
    with _live_children_lock:
        _live_children.discard(proc)


def _kill_live_children() -> None:
    """Kill every process tree a run started and has not reaped yet."""
    with _live_children_lock:
        children = list(_live_children)
    for proc in children:
        _kill_tree(proc)


#: Pump threads reading a child's pipes, by the child: _kill_tree joins them (bounded)
#: after killing the tree, so no stream is ever closed under a reader that is still in
#: the middle of a read.
_pumps: dict[subprocess.Popen[str], list[threading.Thread]] = {}
_pumps_guard = threading.Lock()
#: Pids whose pipes a pump still holds because a surviving descendant keeps the write end
#: open; their streams are never closed again (the daemon thread dies with the process).
_pumps_left: set[int] = set()


def _register_pump(proc: subprocess.Popen[str], thread: threading.Thread) -> None:
    with _pumps_guard:
        _pumps.setdefault(proc, []).append(thread)


def _child_kwargs() -> dict[str, Any]:
    """Popen keywords for a managed child: its own process group on POSIX.

    The group lets :func:`_kill_tree` signal the whole tree, leaderless if it must.
    Windows needs nothing here; ``taskkill /T`` covers the tree while its root lives
    (a tree whose root already exited cannot be reached -- the documented limit below).
    """
    return {} if sys.platform == "win32" else {"start_new_session": True}


def _track(proc: subprocess.Popen[str]) -> None:
    """Register a spawned run process for interrupts to kill (see ``_register_child``)."""
    _register_child(proc)


def _group_has_members(pgid: int, grace_s: float) -> bool:
    """Whether the process group still has a member, waiting up to *grace_s* for TERM.

    Membership, not the leader's exit, decides: a live member keeps the group, so a
    grandchild that outlived its parent is still seen here.
    """
    deadline = time.monotonic() + grace_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # a group exists that we may not signal
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.05)


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """End the whole tree *proc* started, reap it, and close its pipes.

    POSIX: the child was started in its own process group (``_child_kwargs``), so the
    group gets TERM and, if it still has members after :data:`KILL_GRACE_S` -- the group
    is probed with ``os.killpg(pgid, 0)``, because a grandchild that ignores TERM keeps
    it alive long after the leader exited -- KILL. The group id stays the parent's pid;
    a reaped parent's pid is only reused after the whole pid space wraps, so the kill
    window here is safe in practice.

    Windows: ``taskkill /T /F`` with a timeout while the process is still live (a dead
    root gives taskkill nothing to walk from), falling back to ``proc.kill()`` when it
    fails. Known limit, accepted by owner decision: a tree whose root already exited
    cannot be reached at all -- taskkill needs the live root to walk the tree, so such
    orphans can survive this call.

    The tree is killed before any pipe is touched (the writers die first, so pumps see
    EOF); each registered pump is then joined with :data:`PUMP_JOIN_S`, and one that
    still runs -- a descendant kept the write end -- is left behind (it is a daemon
    thread) with a warning, rather than closing a stream under a blocked reader or
    hanging here. The reap, the pump join and the pipe close run whatever the kill did.
    """
    alive = proc.poll() is None
    try:
        if sys.platform == "win32":
            if alive:
                try:
                    taskkilled = subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True, timeout=KILL_WAIT_S,
                    )
                except (subprocess.TimeoutExpired, OSError):
                    taskkilled = None
                if taskkilled is None or taskkilled.returncode != 0:
                    try:
                        proc.kill()
                    except OSError:
                        pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                if alive:
                    try:
                        proc.kill()
                    except OSError:
                        pass
            else:
                if _group_has_members(proc.pid, KILL_GRACE_S):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    finally:
        try:
            proc.wait(timeout=KILL_WAIT_S)
        except (subprocess.TimeoutExpired, OSError):
            pass
        with _pumps_guard:
            pumps = _pumps.pop(proc, [])
            stuck = proc.pid in _pumps_left
        for thread in pumps:
            thread.join(timeout=PUMP_JOIN_S)
            if thread.is_alive():
                with _pumps_guard:
                    _pumps_left.add(proc.pid)
                stuck = True
        if stuck:
            print(
                f"warning: pid {proc.pid}: a descendant still holds its output pipe; "
                "leaving its log pump thread running",
                file=sys.stderr,
            )
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stuck and stream is not proc.stdin:
                continue  # a live pump may be blocked reading it; never close under it
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        _unregister_child(proc)


def _pump(stream: IO[str], tag: str, sink: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for line in stream:
            sink.put((tag, line))
    except BaseException as exc:  # noqa: BLE001 - reported here, never re-raised on a thread
        print(f"warning: a run's {tag} pump failed ({exc!r}); its stream ended early",
              file=sys.stderr)
    finally:
        # The consumer waits for this sentinel to count the stream as closed; an
        # iteration error (a pipe closed under us, a decoding blow-up) must not turn
        # that wait into the run's whole timeout.
        sink.put((tag, None))


def stream_opencode(
    cmd: list[str], prompt: str, cwd: Path, env: dict[str, str], run_dir: Path,
    reserve: int | None, timeout_s: float, reserve_usd: float | None = None,
    log: EventLog | None = None,
) -> tuple[EventLog, str, int | None]:
    """Run OpenCode, feeding events as they arrive; returns (log, stop reason, exit code).

    The raw streams are kept in ``events.jsonl`` and ``stderr.log``. Stop reasons:
    "" (finished), "rate_limited", "over_reserve", "timeout". Passing *log* continues it
    (a resumed session): limits apply to the run's running totals and the files are
    appended. Every way out of the body -- a KeyboardInterrupt included -- kills the
    whole process tree first (see :func:`_kill_tree`).
    """
    mode = "a" if log is not None else "w"
    log = log if log is not None else EventLog()
    if _cancelled():
        raise KeyboardInterrupt("the matrix was interrupted; not starting OpenCode")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        **_child_kwargs(),
    )
    assert proc.stdin and proc.stdout and proc.stderr
    _track(proc)
    try:
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except OSError:
            pass  # the process exited early; its output says why
        lines: queue.Queue[tuple[str, str | None]] = queue.Queue()
        for stream, tag in ((proc.stdout, "out"), (proc.stderr, "err")):
            thread = threading.Thread(target=_pump, args=(stream, tag, lines), daemon=True)
            _register_pump(proc, thread)
            thread.start()
        reason, open_streams = "", 2
        deadline = time.monotonic() + timeout_s
        with (run_dir / "events.jsonl").open(mode, encoding="utf-8") as events, (
            run_dir / "stderr.log"
        ).open(mode, encoding="utf-8") as errors:
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
    finally:
        # Every way out -- finished, stopped, timed out, or interrupted (a
        # KeyboardInterrupt included) -- ends the tree first, so no Ctrl+C can orphan
        # OpenCode or the processes it started; _kill_tree then joins the pumps with a
        # bound and never closes a stream under one that is stuck.
        _kill_tree(proc)
    return log, reason, proc.returncode


class MemoryNotReady(RuntimeError):
    """Condition M's memory backend cannot serve reads; nothing is run."""


def check_memory_ready(mcp_url: str, timeout_s: float = 10.0) -> None:
    """Refuse M unless the backend behind *mcp_url* reports ``reads_ready`` at ``/api/ready``.

    A degraded backend (for example no working embedder) would still list its tools, so M
    would silently run as A with failing recalls.
    """
    parts = urlsplit(mcp_url)
    if not parts.scheme or not parts.netloc:
        raise MemoryNotReady(f"condition M needs a memory MCP URL, got {mcp_url!r}")
    ready_url = urlunsplit((parts.scheme, parts.netloc, "/api/ready", "", ""))
    try:
        with urlopen(ready_url, timeout=timeout_s) as response:  # noqa: S310 - operator-given URL
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise MemoryNotReady(f"memory backend not reachable at {ready_url}: {exc}") from exc
    capabilities = body.get("capabilities") or {}
    if capabilities.get("reads_ready") is not True:
        failures = "; ".join(str(f)[:120] for f in body.get("failures") or [])
        raise MemoryNotReady(f"memory backend cannot serve reads ({body.get('status')}): {failures}")
