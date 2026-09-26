"""Run the task x condition x repeat matrix through ``opencode run --format json``.

Every run gets a fresh export of the repository at its pinned commit, sealed as its own
git repository (condition A never sees a beacon), a private OpenCode config home, and a
fixed prompt on stdin. Conditions H and R additionally get a managed Beacon server (HTTP
JSON for H, MCP over Streamable HTTP for R), started before OpenCode and stopped
afterwards whatever the outcome; H and R also run OpenCode from a fresh empty directory
outside every git repository, so the checkout's instructions cannot be auto-loaded. The
server is ready when its own stderr prints the ready line; H lets the OS pick the port
(``--port 0``), R retries a picked port whose bind fails. H's ``webfetch`` calls are
audited afterwards: OpenCode cannot restrict that tool by URL, so a fetch of anything but
the managed origin fails the run. OpenCode's events are streamed: a run is killed as soon
as its tokens pass the per-run reserve or a rate-limit error appears on stdout or stderr.
The matrix admits a run only while the reserve still fits under the cap, and stops at the
first rate limit (never retrying), an over-reserve run, a run with no usage data, or a
server that never became ready. A stop ends the matrix promptly, a KeyboardInterrupt
included: no new runs are admitted, queued runs are cancelled, and the OpenCode and
Beacon processes of runs in flight are killed instead of being waited on. A rerun of a
run (--resume, or a rerun into an existing run dir) first moves the previous attempt's
prompt, logs and result into ``attempts/<n>/``, so attempts never overwrite each other.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from archolith_bench.beacon_eval import CONDITIONS, DEFAULT_CONDITIONS
from archolith_bench.beacon_eval.isolation import (
    MEMORY_KEY_ENV,
    beacon_remote_server,
    beacon_server,
    empty_cwd,
    memory_stdio_server,
    default_config_source,
    deps_template_dir,
    isolated_config_home,
    isolated_env,
    load_api_keys,
    memory_server,
    remove_tree,
)
from archolith_bench.beacon_eval.models import ANSWER_KEYS, RepoPin, RunResult, Task
from archolith_bench.beacon_eval.scoring import extract_answer, score

#: Owner decision 2026-09-23 (GPT-6 Luna via OpenAI; key through --env-file).
DEFAULT_MODEL = "openai/gpt-6-luna"
DEFAULT_BUDGET_TOKENS = 20_000_000
#: Tokens set aside for each run; a run that passes it is killed and stops the matrix.
DEFAULT_RUN_RESERVE = 400_000
#: Owner decision 2026-09-25: no new run below 15 GB free on the workdir's volume.
DEFAULT_MIN_FREE_GB = 15.0

#: OpenCode's built-in tools, switched off in condition D (Beacon MCP only).
BUILTIN_TOOLS = (
    "bash", "codesearch", "edit", "glob", "grep", "list", "lsp", "patch", "read", "skill",
    "task", "todoread", "todowrite", "webfetch", "websearch", "write",
)
#: Resumes of a session that OpenCode ended right after a tool-calls step (see run_one).
MAX_RESUMES = 2
RESUME_PROMPT = "Continue. When you are done, give the final answer in the required JSON block."

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


class ServerNotReady(RuntimeError):
    """A managed Beacon server (H or R) never became ready; the run fails."""


def pick_free_port() -> int:
    """A free loopback port: the OS assigns one to a ``127.0.0.1:0`` socket.

    Only servers that cannot ask the OS at bind time use this (condition R refuses
    port 0); between this probe and the server's bind another process can still take
    the port, so :func:`beacon_http_process` retries on a fresh one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_accepts(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


#: Seconds a managed server has to bind and print its ready line.
SERVER_READY_TIMEOUT_S = 30.0
#: How much of a failed server's stderr the ServerNotReady message carries.
SERVER_TAIL_CHARS = 400
#: Starts a port-picking server (condition R) gets, each on a fresh port, before the run
#: fails: the probe-to-bind window can lose a port to another process under --workers.
SERVER_BIND_ATTEMPTS = 3
#: Beacon's stable failure code when its loopback bind fails; condition R retries on it.
HTTP_BIND_FAILED = "http_bind_failed"
#: The ``url=http://127.0.0.1:<port>`` a server that picked its own port (condition H's
#: ``--port 0``) reports in its ready line, read here so the port need not be guessed.
_READY_URL_PORT = re.compile(r"url=http://127\.0\.0\.1:(\d+)")


def _server_error(what: str, ready_line: str, tail: deque[str]) -> str:
    return (
        f"the beacon server {what} without reporting {ready_line!r}; "
        f"its stderr ends with: {''.join(tail).strip()[-SERVER_TAIL_CHARS:]}"
    )


def _port_from_ready_line(ready_line: str, tail: deque[str]) -> int:
    """The port a server that chose its own (``--port 0``) reported on its ready line."""
    for line in tail:
        if ready_line in line:
            found = _READY_URL_PORT.search(line)
            if found:
                return int(found.group(1))
    raise ServerNotReady(
        f"the beacon server printed {ready_line!r} without a url=http://127.0.0.1:<port> "
        "to read its port from"
    )


@contextmanager
def beacon_http_process(
    cmd: list[str],
    log_path: Path,
    ready_line: str,
    env: dict[str, str] | None = None,
    timeout_s: float = SERVER_READY_TIMEOUT_S,
) -> Iterator[int]:
    """Start *cmd*, yield its loopback port, and always stop it.

    A ``{port}`` placeholder in *cmd*'s arguments is replaced with a port picked here;
    when such a server exits with ``http_bind_failed`` it is started again on a fresh
    port, up to ``SERVER_BIND_ATTEMPTS`` starts, before the run fails. Without the
    placeholder the command asks the OS itself (condition H passes ``--port 0``) and the
    port is read from the ready line's ``url=http://127.0.0.1:<port>``.

    The server is ready only when its own stderr prints *ready_line* while the child is
    still alive -- never because something else holds the port. On early exit (after the
    bind retries), a stderr-pump failure, or *timeout_s*, the run fails with
    :class:`ServerNotReady`, carrying the server's stderr tail. stderr streams into
    *log_path* (``beacon-server.log`` in the run dir); the log file is opened before the
    pump thread starts. The process tree is killed on every way out of the body --
    success, error, timeout, budget stop, rate-limit stop -- the way ``stream_opencode``
    stops OpenCode.
    """
    picked = "{port}" in cmd
    proc: subprocess.Popen[str] | None = None
    pump_thread: threading.Thread | None = None
    try:
        for attempt in range(SERVER_BIND_ATTEMPTS if picked else 1):
            port = pick_free_port()
            argv = [part.replace("{port}", str(port)) for part in cmd] if picked else list(cmd)
            # Opened here, not on the pump thread: a failure to open must fail the run.
            # Bind retries append, so the log keeps the failed attempts' diagnostics.
            sink = log_path.open("w" if attempt == 0 else "a", encoding="utf-8")
            tail: deque[str] = deque(maxlen=20)
            ready = threading.Event()
            pump_done = threading.Event()
            pump_error: list[BaseException] = []

            def pump(sink: IO[str], proc: subprocess.Popen[str]) -> None:
                try:
                    with sink:
                        for line in proc.stderr:
                            sink.write(line)
                            sink.flush()
                            tail.append(line)
                            if ready_line in line:
                                ready.set()
                except BaseException as exc:  # noqa: BLE001 - surfaced below, never lost
                    pump_error.append(exc)
                finally:
                    pump_done.set()

            try:
                proc = subprocess.Popen(
                    argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                    **_child_kwargs(),
                )
            except OSError:
                sink.close()
                raise
            assert proc.stderr is not None
            _track(proc)
            pump_thread = threading.Thread(target=pump, args=(sink, proc), daemon=True)
            pump_thread.start()

            deadline = time.monotonic() + timeout_s
            while not ready.is_set() and not pump_done.is_set() and proc.poll() is None:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
            if ready.is_set() and proc.poll() is None:
                if pump_error:  # readiness held; the log stream broke mid-run
                    print(
                        f"warning: {log_path.name}: the server log pump failed ({pump_error[0]!r})",
                        file=sys.stderr,
                    )
                yield port if picked else _port_from_ready_line(ready_line, tail)
                return
            # Not ready: drain the pump first, so the tail carries the last lines.
            if proc.poll() is not None:
                pump_thread.join(timeout=5)
            if pump_error:
                raise ServerNotReady(f"the beacon server's log pump failed: {pump_error[0]!r}")
            if ready.is_set():
                raise ServerNotReady(_server_error("exited right after its ready line", ready_line, tail))
            if proc.poll() is None:
                raise ServerNotReady(_server_error(f"was not ready within {timeout_s:.0f}s", ready_line, tail))
            if (
                picked and attempt + 1 < SERVER_BIND_ATTEMPTS
                and any(HTTP_BIND_FAILED in line for line in tail)
            ):
                _kill_tree(proc)
                proc = None  # the next start owns the cleanup from here
                continue
            raise ServerNotReady(_server_error("exited", ready_line, tail))
    finally:
        if proc is not None:
            _kill_tree(proc)  # reaps the server and closes its pipes, so the pump ends
        if pump_thread is not None:
            pump_thread.join(timeout=5)


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
    #: Condition M: a Menhir remote MCP URL (e.g. http://127.0.0.1:8795/mcp-http) and its key.
    memory_url: str | None = None
    memory_key: str | None = field(default=None, repr=False)
    #: Condition M over stdio instead of remote MCP: the bridge command, its non-secret
    #: environment, and the variable it reads the backend key from. memory_url still names the
    #: backend, for the readiness check.
    memory_stdio: list[str] | None = None
    memory_stdio_env: dict[str, str] = field(default_factory=dict)
    memory_stdio_key_var: str = "MENHIR_API_KEY"
    #: Keep each run's checkout after scoring (debugging); by default it is deleted and
    #: ``rescore`` rebuilds it from ``pin.json`` and the saved changes.
    keep_checkouts: bool = False
    #: No new run starts while the workdir's volume has less free space (GB); None = off.
    min_free_gb: float | None = DEFAULT_MIN_FREE_GB


class LowDisk(RuntimeError):
    """The workdir's volume is below ``min_free_gb``; no new run starts."""


def check_disk(workdir: Path, min_free_gb: float | None) -> None:
    if not min_free_gb:
        return
    free = shutil.disk_usage(workdir).free / 1e9
    if free < min_free_gb:
        raise LowDisk(f"only {free:.1f} GB free on the workdir's volume (floor {min_free_gb:g} GB)")


@dataclass
class Budget:
    cap: int | None
    reserve: int | None = DEFAULT_RUN_RESERVE
    used: int = 0
    cap_usd: float | None = None
    reserve_usd: float = 0.0
    used_usd: float = 0.0
    #: Runs admitted but not finished; each still holds its reserve (parallel matrix).
    in_flight: int = 0

    def check(self) -> None:
        held = self.in_flight + 1
        flying = f", {self.in_flight} run(s) in flight" if self.in_flight else ""
        if self.cap is not None and self.used + (self.reserve or 0) * held > self.cap:
            raise BudgetExhausted(
                f"the next run's {self.reserve:,}-token reserve would exceed the "
                f"{self.cap:,}-token cap ({self.used:,} used{flying})"
            )
        if self.cap_usd is not None and self.used_usd + self.reserve_usd * held > self.cap_usd:
            raise BudgetExhausted(
                f"the next run's ${self.reserve_usd:.2f} reserve would exceed the "
                f"${self.cap_usd:.2f} cap (${self.used_usd:.4f} spent{flying})"
            )

    def spend(self, tokens: int, usd: float = 0.0) -> None:
        self.used += tokens
        self.used_usd += usd


# ---------------------------------------------------------------------------
# Checkouts and beacons
# ---------------------------------------------------------------------------


def _ensure_source(pin: RepoPin, cache: Path) -> Path:
    """The local repository holding *pin*'s commit: cloned into *cache* once, and fetched
    only when the commit is missing, so concurrent exports never write to it."""
    source = Path(pin.local_path) if pin.local_path else cache / pin.name
    if not pin.local_path and not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", pin.url, str(source)],
            check=True,
        )
    present = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", f"{pin.commit}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        subprocess.run(
            ["git", "-C", str(source), "fetch", "--quiet", "origin", pin.commit],
            check=False,
            capture_output=True,
        )
    return source


def export_commit(pin: RepoPin, dest: Path, cache: Path, source: Path | None = None) -> Path:
    """Write the tree of *pin* at its commit into *dest* (no .git), from *source* if given."""
    source = source if source is not None else _ensure_source(pin, cache)
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", pin.commit],
        check=True,
        capture_output=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def _checkout_git(checkout: Path, autocrlf_off: bool = True) -> list[str]:
    git = [
        "git", "-C", str(checkout),
        "-c", "user.name=beacon-eval", "-c", "user.email=beacon-eval@example.invalid",
        "-c", "commit.gpgsign=false",
    ]
    return [*git, "-c", "core.autocrlf=false"] if autocrlf_off else git


def _git_out(cmd: list[str]) -> str:
    # UTF-8 explicitly: git prints paths as UTF-8 whatever the console code page.
    return subprocess.run(
        cmd, check=True, capture_output=True, encoding="utf-8", errors="surrogateescape"
    ).stdout.strip()


def _copy_seal(checkout: Path, pack: bool) -> None:
    """The original seal: add every file, commit (``core.autocrlf=false``), optionally pack."""
    git = _checkout_git(checkout)
    subprocess.run([*git, "add", "--all"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "--quiet", "--allow-empty", "-m", "pinned export"],
                   check=True, capture_output=True)
    if pack:
        subprocess.run([*git, "repack", "-a", "-d", "-q"], check=True, capture_output=True)
        subprocess.run([*git, "prune"], check=True, capture_output=True)


def seal_repo(pin: RepoPin, workdir: Path, source: Path | None = None) -> Path:
    """A packed repository holding only the tree the original seal made for *pin*.

    Built once per pin in ``<workdir>/seals/`` from its own export with the original
    ``add --all`` + ``commit`` (same ignore rules, same bytes, ``core.autocrlf=false``),
    packed, working files dropped. Run checkouts borrow its objects, so the agent sees
    exactly the old seal's tree and no other history of the repository. Built in a
    staging folder and renamed into place, so a concurrent reader never sees half of one.
    """
    final = workdir / "seals" / f"{pin.name}-{pin.commit[:12]}"
    if (final / "tree").is_file():
        return final
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=final.name + ".staging-", dir=final.parent))
    try:
        export_commit(pin, staging, workdir / "cache", source)
        subprocess.run([*_checkout_git(staging), "init", "--quiet"], check=True, capture_output=True)
        _copy_seal(staging, pack=True)
        tree = _git_out([*_checkout_git(staging), "rev-parse", "HEAD^{tree}"])
        for entry in staging.iterdir():
            if entry.name != ".git":
                remove_tree(entry) if entry.is_dir() else entry.unlink()
        (staging / "tree").write_text(tree + "\n", encoding="utf-8")
        try:
            os.rename(staging, final)
        except OSError:
            if not (final / "tree").is_file():
                raise
    finally:
        remove_tree(staging)
    return final


def _seal_via_alternates(checkout: Path, seal: Path) -> None:
    """Commit the seal repo's tree in *checkout*, borrowing its objects (none copied).

    Raises CalledProcessError when the files on disk do not match that tree or an object
    is missing; the caller then falls back to the copy seal.
    """
    git = _checkout_git(checkout)
    tree = (seal / "tree").read_text(encoding="utf-8").strip()
    objects = (seal / ".git" / "objects").resolve()
    (checkout / ".git" / "objects" / "info").mkdir(parents=True, exist_ok=True)
    # LF only: git would read a CRLF line as a path ending in "\r".
    (checkout / ".git" / "objects" / "info" / "alternates").write_bytes(
        objects.as_posix().encode("utf-8") + b"\n"
    )
    subprocess.run([*git, "read-tree", tree], check=True, capture_output=True)
    # Non-zero when any indexed file differs from the tree on disk.
    subprocess.run([*git, "update-index", "--refresh", "-q"], check=True, capture_output=True)
    new = _git_out([*git, "commit-tree", tree, "-m", "pinned export"])
    subprocess.run([*git, "update-ref", "HEAD", new], check=True, capture_output=True)
    missing = _git_out([*git, "rev-list", "--objects", "--missing=print", "HEAD"])
    if any(line.startswith("?") for line in missing.splitlines()):
        raise subprocess.CalledProcessError(1, "rev-list", "objects missing from the seal repo")


def seal_checkout(checkout: Path, seal: Path | None = None) -> str:
    """Make *checkout* its own git root with one commit; returns how ("alternates" or "copied").

    OpenCode searches parent directories for project config and instructions up to the
    git root; without this a checkout under the results tree picks up the enclosing
    repositories' ``AGENTS.md`` and ``opencode.json``.

    With *seal* (see :func:`seal_repo`) the commit reuses its tree through git alternates:
    ~20 files instead of one loose object per file, and the same tree, index and config
    the copy seal gives. Otherwise, or if that fails, the files are added and committed
    as before (packed when a seal repo was expected).
    """
    git = _checkout_git(checkout)
    subprocess.run([*git, "init", "--quiet"], check=True, capture_output=True)
    if seal is not None:
        try:
            _seal_via_alternates(checkout, seal)
            return "alternates"
        except (subprocess.CalledProcessError, OSError, UnicodeError):
            _rmtree(checkout / ".git")
            subprocess.run([*git, "init", "--quiet"], check=True, capture_output=True)
    _copy_seal(checkout, pack=seal is not None)
    return "copied"


_rmtree = remove_tree

#: What save_changes writes; cleared first so a rerun never inherits an earlier attempt's.
CHANGE_FILES = ("changes.status", "changes.tar", "changes.deleted.json")


def file_times(root: Path) -> dict[str, int]:
    """``{relative posix path: mtime_ns}`` for every file under *root*, skipping its ``.git``."""
    times: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath) == root and ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            path = Path(dirpath) / name
            times[path.relative_to(root).as_posix()] = path.stat().st_mtime_ns
    return times


def save_changes(checkout: Path, run_dir: Path, baseline: dict[str, int]) -> None:
    """Record what the agent changed in the checkout (usually nothing).

    *baseline* is :func:`file_times` of the fresh export (``git archive`` stamps every file
    with the commit time), so any file created, rewritten or deleted since shows up, git
    ignored or committed alike: scoring reads the files, not git. ``changes.status`` is the
    agent's ``git status --porcelain`` (for reading). When anything changed, ``changes.tar``
    holds the new and changed files as they are on disk and ``changes.deleted.json`` lists
    the removed ones.
    """
    for name in CHANGE_FILES:
        (run_dir / name).unlink(missing_ok=True)
    status = subprocess.run(
        [*_checkout_git(checkout, autocrlf_off=False), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True,
    ).stdout
    (run_dir / "changes.status").write_bytes(status)
    now = file_times(checkout)
    present = sorted(rel for rel, mtime in now.items() if baseline.get(rel) != mtime)
    deleted = sorted(rel for rel in baseline if rel not in now)
    if not present and not deleted:
        return
    with tarfile.open(run_dir / "changes.tar", "w") as tar:
        for rel in present:
            tar.add(checkout / rel, arcname=rel)
    (run_dir / "changes.deleted.json").write_text(json.dumps(deleted), encoding="utf-8")


#: Left in a run folder whose checkout could not be fully deleted; rescore then rebuilds it.
PARTIAL_MARKER = "checkout.partial"


def _retire_checkout(checkout: Path, run_dir: Path, baseline: dict[str, int], keep: bool) -> None:
    """Save the agent's changes, then delete the checkout unless *keep*. Never raises.

    The run's result is already saved and must still reach the matrix (its spend, its
    stop reason), so failures here only warn. If the changes cannot be saved the checkout
    is kept whole; if it cannot be fully deleted (a process the agent started may hold a
    file), a marker tells rescore to rebuild it rather than score what is left.
    """
    try:
        save_changes(checkout, run_dir, baseline)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"warning: could not save changes of {run_dir.name}; keeping its checkout ({exc})", file=sys.stderr)
        return
    if keep:
        return
    try:
        _rmtree(checkout)
    except OSError as exc:
        (run_dir / PARTIAL_MARKER).write_text(str(exc), encoding="utf-8")
        print(f"warning: could not delete the checkout of {run_dir.name} ({exc})", file=sys.stderr)


def _has_changes(run_dir: Path) -> bool:
    return (run_dir / "changes.tar").is_file() or (run_dir / "changes.deleted.json").is_file()


def rebuild_checkout(run_dir: Path, dest: Path, cache: Path) -> Path:
    """Re-create a deleted checkout at *dest* from ``pin.json`` plus the saved changes.

    Uses the clone cache (or the pin's ``local_path``); if either lacks the commit, this
    fetches it, like a run would.
    """
    pin = RepoPin(**{k: v for k, v in json.loads((run_dir / "pin.json").read_text(encoding="utf-8")).items()
                     if k in RepoPin.__dataclass_fields__})
    export_commit(pin, dest, cache)
    if (run_dir / "changes.tar").is_file():
        with tarfile.open(run_dir / "changes.tar") as tar:
            tar.extractall(dest, filter="data")
    deleted = run_dir / "changes.deleted.json"
    for rel in json.loads(deleted.read_text(encoding="utf-8")) if deleted.is_file() else []:
        target = (dest / rel).resolve()
        if target.is_relative_to(dest.resolve()) and target.is_file():
            target.unlink()
    return dest


#: One lock per beacon key, so two threads that would both build or export the same pin
#: serialize and the second reuses the finished result. Across processes the staging
#: rename plays that role (the loser's rename onto the finished directory fails, and it
#: then finds the result there), the way ``seal_repo`` does it.
_pin_locks: dict[str, threading.Lock] = {}
_pin_locks_guard = threading.Lock()


def _pin_lock(key: str) -> threading.Lock:
    with _pin_locks_guard:
        return _pin_locks.setdefault(key, threading.Lock())


def build_beacon(config: RunnerConfig, pin: RepoPin) -> Path:
    """Build the beacon for *pin* in its own export (never the agent's checkout).

    The build lives in ``beacons/<name>-<commit12>/``, so a workdir reused with a
    different commit for the same pin name builds anew instead of reusing the old
    manifest. Only the first writer builds: a per-pin lock within this process, a
    staging directory renamed into place across processes.
    """
    root = config.workdir / "beacons" / f"{pin.name}-{pin.commit[:12]}"
    manifest = root / "beacon.generated.yaml"
    if manifest.is_file():
        return manifest
    with _pin_lock(f"build:{root.name}"):
        if manifest.is_file():
            return manifest
        beacons = config.workdir / "beacons"
        beacons.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=root.name + ".staging-", dir=beacons))
        try:
            export_commit(pin, staging, config.workdir / "cache")
            env = dict(os.environ)
            if config.beacon_src:
                env["PYTHONPATH"] = config.beacon_src
            subprocess.run(
                [config.beacon_python, "-m", "beacon", "build", "--repo", str(staging),
                 "--format", "json"],
                check=True,
                capture_output=True,
                env=env,
            )
            try:
                os.rename(staging, root)
            except OSError:
                if not manifest.is_file():
                    raise
        finally:
            remove_tree(staging)
    return manifest


def build_prompt(task: Task, condition: str, manifest_text: str = "", http_url: str = "") -> str:
    """Task first, then (C) the pasted manifest or (H) the HTTP starting point, then the
    same closing lines."""
    parts = [task.prompt.strip()]
    if condition == "C":
        parts.append(
            "Project knowledge (beacon.generated.yaml) follows.\n```yaml\n"
            + manifest_text.strip()
            + "\n```"
        )
    if condition == "H":
        if not http_url:
            raise ValueError("condition H needs the http_url its server was started on")
        parts.append(
            f"Project knowledge is served over HTTP at {http_url}. Start with GET "
            f"{http_url}/.well-known/archolith-beacon, which lists the routes and a recommended flow."
        )
    parts += [TOOL_NOTE, ANSWER_INSTRUCTIONS]
    return "\n\n".join(parts)


def disabled_tools(condition: str) -> tuple[str, ...]:
    """OpenCode's built-in tools switched off for *condition*.

    D and R run on MCP alone; H also loses every built-in tool but keeps ``webfetch``,
    its only way to reach the Beacon HTTP JSON surface (so ``websearch`` stays off).
    """
    if condition in ("D", "R"):
        return BUILTIN_TOOLS
    if condition == "H":
        return tuple(tool for tool in BUILTIN_TOOLS if tool != "webfetch")
    return ()


def export_beacon_snapshot(config: RunnerConfig, pin: RepoPin) -> Path:
    """The canonical snapshot condition R's MCP server serves, written once per pin.

    Like :func:`build_beacon`, keyed by the pin's commit and written once: a per-pin
    lock serializes this process's first writers (the second reuses the result), and
    the snapshot lands through an atomic rename, so a reader in another process never
    sees half of one.
    """
    root = config.workdir / "beacons" / f"{pin.name}-{pin.commit[:12]}"
    snapshot = root / "beacon.snapshot.json"
    if snapshot.is_file():
        return snapshot
    with _pin_lock(f"export:{root.name}"):
        if snapshot.is_file():
            return snapshot
        manifest = build_beacon(config, pin)
        env = dict(os.environ)
        if config.beacon_src:
            env["PYTHONPATH"] = config.beacon_src
        staging = root / f"beacon.snapshot.{os.getpid()}-{threading.get_ident()}.tmp"
        try:
            subprocess.run(
                [config.beacon_python, "-m", "beacon", "export", str(manifest),
                 "--docs-root", str(root), "--output", str(staging)],
                check=True,
                capture_output=True,
                env=env,
            )
            os.replace(staging, snapshot)  # atomic, so a parallel reader never sees half of one
        finally:
            staging.unlink(missing_ok=True)
    return snapshot


#: The stderr line each managed server prints once it is bound (H, then R).
HTTP_READY_LINE = "Beacon HTTP ready"
MCP_HTTP_READY_LINE = "MCP http listening"

#: The managed server's stderr, kept in the run dir.
SERVER_LOG = "beacon-server.log"


def _http_server_spec(
    config: RunnerConfig, condition: str, manifest: Path | None, snapshot: Path | None
) -> tuple[list[str], str, dict[str, str]] | None:
    """How to start the managed server for *condition* (H or R); None = no server.

    H's command carries ``--port 0`` so the OS assigns the port and the ready line
    reports it (``url=http://127.0.0.1:<port>``); R's command carries a ``{port}``
    placeholder for :func:`beacon_http_process` to fill, because ``serve --transport
    http`` refuses port 0. Both commands are loopback-only.
    """
    if condition == "H" and manifest is not None:
        cmd = [
            config.beacon_python, "-m", "beacon", "serve-http",
            "--manifest", str(manifest), "--docs-root", str(manifest.parent),
            "--host", "127.0.0.1", "--port", "0",
        ]
        ready_line = HTTP_READY_LINE
    elif condition == "R" and snapshot is not None:
        cmd = [
            config.beacon_python, "-m", "beacon", "serve",
            "--snapshot", str(snapshot), "--transport", "http",
            "--host", "127.0.0.1", "--port", "{port}",
        ]
        ready_line = MCP_HTTP_READY_LINE
    else:
        return None
    env = dict(os.environ)
    if config.beacon_src:
        env["PYTHONPATH"] = config.beacon_src
    return cmd, ready_line, env


def _mcp_block(
    config: RunnerConfig, condition: str, manifest: Path | None, base_url: str
) -> dict[str, Any] | None:
    """The ``mcp`` block each condition puts in OpenCode's config; None = none."""
    if condition in ("B", "D") and manifest is not None:
        return beacon_server(config.beacon_python, manifest, config.beacon_src)
    if condition == "R":
        return beacon_remote_server(base_url + "/mcp")
    if condition == "M":
        if config.memory_stdio:
            return memory_stdio_server(
                config.memory_stdio, config.memory_stdio_env, config.memory_stdio_key_var
            )
        return memory_server(str(config.memory_url))
    return None


# ---------------------------------------------------------------------------
# The webfetch audit (condition H)
# ---------------------------------------------------------------------------


def webfetch_hosts(run_dir: Path) -> list[str]:
    """Deduped ``host:port`` of every ``webfetch`` call in the run's ``events.jsonl``.

    OpenCode 1.18.31 cannot restrict ``webfetch`` by URL pattern (its ``permission``
    config takes only a flat allow/ask/deny for this tool), so condition H is audited
    after the run instead: these are the hosts it actually asked to fetch.
    """
    hosts: set[str] = set()
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "tool_use":
            continue
        part = event.get("part")
        state = part.get("state") if isinstance(part, dict) else None
        if not isinstance(part, dict) or part.get("tool") != "webfetch" or not isinstance(state, dict):
            continue
        payload = state.get("input")
        url = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(url, str) or not url:
            continue
        parts = urlsplit(url)
        origin = parts.hostname or ""
        if not origin:
            continue
        hosts.add(f"{origin}:{parts.port}" if parts.port is not None else origin)
    return sorted(hosts)


def off_origin_fetches(hosts: list[str], managed_origin: str) -> list[str]:
    """The fetched hosts that are not the managed server's ``127.0.0.1:<port>`` origin."""
    return [host for host in hosts if host != managed_origin]


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


#: POSIX children get this long to exit on TERM before their whole group gets KILL.
KILL_GRACE_S = 5.0
#: How long _kill_tree waits for a tree to die before giving up on reaping it.
KILL_WAIT_S = 15.0
#: After a stop, how long the parallel matrix drains interrupted runs before leaving.
DRAIN_TIMEOUT_S = 60.0

#: Processes a run started and has not reaped yet, so a stop can end them now
#: instead of waiting for their runs to finish (see _run_parallel).
_live_children: set[subprocess.Popen[str]] = set()
_live_children_lock = threading.Lock()


def _register_child(proc: subprocess.Popen[str]) -> None:
    """Remember a spawned run process for :func:`_kill_live_children`."""
    with _live_children_lock:
        _live_children.add(proc)


def _unregister_child(proc: subprocess.Popen[str]) -> None:
    with _live_children_lock:
        _live_children.discard(proc)


def _kill_live_children() -> None:
    """Kill every process tree a run started and has not reaped yet."""
    with _live_children_lock:
        children = list(_live_children)
    for proc in children:
        _kill_tree(proc)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),  # ULONG_PTR
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _win_creation_time(handle: int) -> int | None:
        """The handle's process creation time (100ns ticks); None if it cannot be read."""
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not ctypes.windll.kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime

    def _win_pid_creation_time(pid: int) -> int | None:
        """The creation time of whatever process now owns *pid*; None if none does."""
        handle = ctypes.windll.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            return _win_creation_time(handle)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    def _attach_job(proc: subprocess.Popen[str]) -> None:
        """Put *proc* and its whole future tree in a kill-on-close job object.

        Closing the job's handle (in ``_kill_tree``) ends every member the kernel still
        lists, including grandchildren whose parent already exited -- which
        ``taskkill /T`` cannot reach, because it needs a live root to walk from. Best
        effort: if a call fails, the tree is left to the taskkill path alone.
        """
        kernel32 = ctypes.windll.kernel32
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        limits = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(limits), ctypes.sizeof(limits)
        ) or not kernel32.AssignProcessToJobObject(job, proc._handle):
            kernel32.CloseHandle(job)
            return
        proc._beacon_eval_job = job  # noqa: SLF001 - carried with the Popen it belongs to


def _child_kwargs() -> dict[str, Any]:
    """Popen keywords for a managed child: its own process group on POSIX.

    The group lets :func:`_kill_tree` signal the whole tree, leaderless if it must.
    Windows needs nothing here; its job object and ``taskkill /T`` cover the tree.
    """
    return {} if sys.platform == "win32" else {"start_new_session": True}


def _track(proc: subprocess.Popen[str]) -> None:
    """Register a spawned run process for stops to kill, plus its Windows job object."""
    if sys.platform == "win32":
        _attach_job(proc)
    _register_child(proc)


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """End the whole tree *proc* started, reap it, and close its pipes.

    POSIX: the child was started in its own process group (``_child_kwargs``), so the
    group gets TERM and, after :data:`KILL_GRACE_S`, KILL -- grandchildren included,
    even after the parent itself already exited (a live member keeps the group). The
    group id stays the parent's pid; a reaped parent's pid is only reused after the
    whole pid space wraps, so the kill window here is safe in practice.

    Windows: ``taskkill /T /F`` while the pid still names the process we started --
    checked against the process's creation time, so a recycled pid is never hit --
    falling back to ``proc.kill()`` when the taskkill fails on a live parent; the
    kill-on-close job (``_attach_job``) then ends whatever taskkill could not reach,
    which is any tree whose root already exited. The pipes close last, so pump threads
    see EOF and end even if a stray grandchild kept a write end open.
    """
    alive = proc.poll() is None
    if sys.platform == "win32":
        if not alive:
            started_at = _win_creation_time(proc._handle)  # noqa: SLF001 - our own child
            alive = started_at is not None and started_at == _win_pid_creation_time(proc.pid)
        if alive:
            taskkilled = subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
            )
            if taskkilled.returncode != 0:
                try:
                    proc.kill()
                except OSError:
                    pass
        job = getattr(proc, "_beacon_eval_job", None)
        if job:
            proc._beacon_eval_job = None  # noqa: SLF001 - ours, closed right here
            ctypes.windll.kernel32.CloseHandle(job)  # kill-on-close: ends the rest of the tree
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
            try:
                proc.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    try:
        proc.wait(timeout=KILL_WAIT_S)
    except (subprocess.TimeoutExpired, OSError):
        pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except (OSError, ValueError):
            pass
    _unregister_child(proc)


def _pump(stream: IO[str], tag: str, sink: queue.Queue[tuple[str, str | None]]) -> None:
    for line in stream:
        sink.put((tag, line))
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
            threading.Thread(target=_pump, args=(stream, tag, lines), daemon=True).start()
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
        # KeyboardInterrupt included) -- ends the tree, so no Ctrl+C can orphan
        # OpenCode or the processes it started.
        _kill_tree(proc)
    return log, reason, proc.returncode


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


#: A rerun moves these into ``attempts/<n>/`` first, so a new attempt never sits beside
#: the old one's logs and the old attempt's cost stays readable for accounting.
ATTEMPT_FILES = ("prompt.txt", "events.jsonl", "stderr.log", SERVER_LOG, "result.json")
ATTEMPTS_DIR = "attempts"


def _archive_previous_attempt(run_dir: Path) -> None:
    """Move a previous attempt's artifacts in *run_dir* to ``attempts/<n>/``.

    ``--resume`` redoes a run whose saved result carries an error or no answer (see
    :func:`_completed_run`), and a plain rerun into an existing run dir overwrites the
    same paths; this keeps the redone attempt's prompt, logs and result -- its spend
    with them -- without letting them stand in for the new attempt's.
    """
    present = [name for name in ATTEMPT_FILES if (run_dir / name).is_file()]
    if not present:
        return
    attempts = run_dir / ATTEMPTS_DIR
    used = (
        {int(path.name) for path in attempts.iterdir() if path.name.isdigit()}
        if attempts.is_dir() else set()
    )
    dest = attempts / str(max(used, default=0) + 1)
    dest.mkdir(parents=True)
    for name in present:
        os.replace(run_dir / name, dest / name)


def run_one(
    config: RunnerConfig, pin: RepoPin, task: Task, condition: str, repeat: int
) -> RunResult:
    """One run. Raises RateLimited, BudgetExhausted or AccountingError after saving it."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "M" and not (config.memory_url and config.memory_key):
        raise ValueError("condition M needs memory_url and memory_key")
    # Absolute paths: PWD and B's --manifest are resolved by processes running in the checkout.
    run_dir = (config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}").resolve()
    cache = config.workdir / "cache"
    # A rerun (--resume) starts from a clean export and inherits nothing of the last attempt.
    _archive_previous_attempt(run_dir)
    _rmtree(run_dir / "checkout")
    for name in (*CHANGE_FILES, PARTIAL_MARKER):
        (run_dir / name).unlink(missing_ok=True)
    checkout = export_commit(pin, run_dir / "checkout", cache)
    baseline = file_times(checkout)
    try:
        repo = seal_repo(pin, config.workdir)
    except (subprocess.CalledProcessError, OSError, UnicodeError) as exc:
        print(f"warning: no seal repo for {pin.name} ({exc}); sealing by copy", file=sys.stderr)
        repo = None
    seal = seal_checkout(checkout, repo)
    # Lets rescore rebuild the checkout once it is deleted.
    (run_dir / "pin.json").write_text(json.dumps({**asdict(pin), "seal": seal}, indent=2), encoding="utf-8")
    manifest = build_beacon(config, pin).resolve() if condition in ("B", "C", "D", "H", "R") else None
    manifest_text = manifest.read_text(encoding="utf-8") if manifest is not None and condition == "C" else ""
    started = time.monotonic()
    keys = load_api_keys(config.env_file) if config.env_file else {}
    secrets: list[str] = [*keys.values(), *([config.memory_key] if config.memory_key else [])]
    log, reason, code, resumes = EventLog(), "", None, 0
    managed_origin, server_error = "", ""
    try:
        # Inside the try: an export or launch failure is a recorded run like
        # ServerNotReady, not a crash (condition H/R startup, see finding 7).
        snapshot = export_beacon_snapshot(config, pin) if condition == "R" else None
        spec = _http_server_spec(config, condition, manifest, snapshot)
        server_cm: AbstractContextManager[int | None] = (
            beacon_http_process(spec[0], run_dir / SERVER_LOG, spec[1], spec[2]) if spec else nullcontext(None)
        )
        with server_cm as port:
            if port is not None:
                managed_origin = f"127.0.0.1:{port}"
            prompt = build_prompt(
                task,
                condition,
                manifest_text,
                http_url=f"http://{managed_origin}" if condition == "H" else "",
            )
            (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            mcp = _mcp_block(config, condition, manifest, f"http://{managed_origin}")
            disabled = disabled_tools(condition)
            # --title skips OpenCode's title-generation request, whose tokens no event reports.
            cmd = [*config.opencode_cmd, "run", "--pure", "--print-logs", "--title", "beacon-eval",
                   "-m", config.model, "--format", "json"]
            with isolated_config_home(
                config.config_source or default_config_source(), config.model, mcp,
                builtin_provider=bool(keys), disabled_tools=disabled,
                deps_template=deps_template_dir(config.opencode_cmd),
            ) as home:
                env = isolated_env(os.environ, home)
                env.update(keys)
                if condition == "M":
                    # Read by OpenCode's {env:...} substitution; never written to the config file.
                    env[MEMORY_KEY_ENV] = str(config.memory_key)
                # Conditions H and R run outside the checkout entirely, in a fresh empty
                # directory no git repository covers, so OpenCode cannot auto-load the
                # checkout's AGENTS.md, CLAUDE.md or project opencode.json (--pure only
                # disables plugins). The checkout is still exported and sealed above for
                # scoring and rescore.
                with empty_cwd() if condition in ("H", "R") else nullcontext(checkout) as agent_dir:
                    # An inherited PWD (Git Bash, MSYS, most shells) may root OpenCode in the
                    # caller's repo instead of the working directory; the fake-provider check exercises this.
                    env["PWD"] = str(agent_dir)
                    reserve_usd = config.run_reserve_usd if config.budget_usd is not None else None
                    log, reason, code = stream_opencode(
                        cmd, prompt, agent_dir, env, run_dir, config.run_reserve_tokens, config.timeout_s,
                        reserve_usd,
                    )
                    # OpenCode's run mode sometimes exits right after a tool-calls step, before the
                    # model's next turn (7/45 Luna runs). Resume the same session so no setup loses it.
                    while (
                        not reason
                        and resumes < MAX_RESUMES
                        and log.last_step_reason == "tool-calls"
                        and log.session_id
                        and extract_answer("\n".join(log.texts)) is None
                    ):
                        resumes += 1
                        log, reason, code = stream_opencode(
                            [*cmd, "--session", log.session_id], RESUME_PROMPT, agent_dir, env, run_dir,
                            config.run_reserve_tokens, config.timeout_s, reserve_usd, log=log,
                        )
    except ServerNotReady as exc:
        server_error = _redact_text(str(exc), secrets)
    except (subprocess.CalledProcessError, OSError) as exc:
        if condition not in ("H", "R"):
            raise
        server_error = _redact_text(_startup_error(exc), secrets)
    fetched = webfetch_hosts(run_dir) if condition == "H" else []
    off_origin = off_origin_fetches(fetched, managed_origin) if condition == "H" else []
    text = "\n".join(log.texts)
    answer = extract_answer(text)
    error = server_error or reason or ("; ".join(log.errors) if log.errors else "")
    if off_origin:
        violation = f"non-loopback fetch: {', '.join(off_origin)}"
        error = f"{error}; {violation}" if error else violation
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
        resumes=resumes,
        tool_calls=log.tool_calls,
        seconds=round(time.monotonic() - started, 1),
        error=error[:500],
        fetched_hosts=fetched,
    )
    result.scores = score(answer, task.gold, checkout)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    _redact(run_dir, secrets)
    _retire_checkout(checkout, run_dir, baseline, config.keep_checkouts)
    if server_error:
        raise ServerNotReady(server_error)
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

    A run whose checkout was deleted is scored against one rebuilt from its ``pin.json``
    and saved changes; runs without changes share one rebuilt export per pin. Rewrites
    each ``result.json`` and ``results.jsonl``; runs whose task is not in *tasks* are left
    untouched and not returned.
    """
    by_id = {(task.repo, task.task_id): task for task in tasks}
    results: list[RunResult] = []
    scratch = workdir / "rescore-tmp"
    _rmtree(scratch)
    shared: dict[str, Path] = {}
    try:
        for path in sorted((workdir / "runs").glob("*/result.json")):
            result = RunResult(**json.loads(path.read_text(encoding="utf-8")))
            task = by_id.get((result.repo, result.task_id))
            if task is None:
                continue
            run_dir = path.parent
            checkout = run_dir / "checkout"
            rebuilt: Path | None = None
            if not checkout.is_dir() or (run_dir / PARTIAL_MARKER).is_file():
                if not (run_dir / "pin.json").is_file():
                    raise FileNotFoundError(f"{run_dir.name}: no checkout and no pin.json to rebuild one")
                if _has_changes(run_dir):
                    rebuilt = rebuild_checkout(run_dir, scratch / run_dir.name, workdir / "cache")
                    checkout = rebuilt
                else:
                    saved_pin = json.loads((run_dir / "pin.json").read_text(encoding="utf-8"))
                    key = f"{saved_pin['name']}@{saved_pin['commit']}"
                    if key not in shared:
                        shared[key] = rebuild_checkout(run_dir, scratch / f"pin-{len(shared)}", workdir / "cache")
                    checkout = shared[key]
            result.scores = score(result.answer, task.gold, checkout)
            if rebuilt is not None:
                _rmtree(rebuilt)
            path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
            results.append(result)
    finally:
        _rmtree(scratch)
    (workdir / "results.jsonl").write_text(
        "".join(json.dumps(asdict(result)) + "\n" for result in results), encoding="utf-8"
    )
    return results


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


def _startup_error(exc: subprocess.CalledProcessError | OSError) -> str:
    """The recorded-failure message for a Beacon export or server-launch failure."""
    if isinstance(exc, subprocess.CalledProcessError):
        tail = str(getattr(exc, "stderr", "") or "").strip()[-SERVER_TAIL_CHARS:]
        return f"the beacon setup failed ({exc}); its stderr ends with: {tail}"
    return f"the beacon server could not be started: {exc}"


def _redact_text(text: str, secrets: Any) -> str:
    """*text* with every secret value of at least 8 characters replaced by ``<redacted>``."""
    for value in secrets:
        if len(value) >= 8:
            text = text.replace(value, "<redacted>")
    return text


def _redact(run_dir: Path, secrets: Any) -> None:
    """Replace any key value that reached a saved file (logs can echo request errors)."""
    values = [value for value in secrets if len(value) >= 8]
    if not values:
        return
    for name in ("events.jsonl", "stderr.log", "result.json", "prompt.txt", SERVER_LOG):
        path = run_dir / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        cleaned = _redact_text(text, values)
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")


def run_matrix(
    config: RunnerConfig,
    pins: dict[str, RepoPin],
    tasks: list[Task],
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    repeats: int = 1,
    resume: bool = False,
    workers: int = 1,
) -> tuple[list[RunResult], str]:
    """Run every task x condition x repeat; returns results and why it stopped ("" = done).

    A run that stops the matrix is still recorded and counted against the budget. With
    *resume*, a run whose saved ``result.json`` has an answer and no error is reused (after a
    killed matrix); a folder without one is run again from scratch. With *workers* > 1, up to
    that many runs execute at once (see :func:`_run_parallel`).
    """
    budget = Budget(
        config.budget_tokens, config.run_reserve_tokens,
        cap_usd=config.budget_usd, reserve_usd=config.run_reserve_usd,
    )
    results: list[RunResult] = []
    if "M" in conditions:
        try:
            check_memory_ready(str(config.memory_url or ""))
        except MemoryNotReady as exc:
            return results, str(exc)
    log = config.workdir / "results.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    logged: set[tuple[str, str, str, int]] = set()
    if resume and log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                logged.add((row["repo"], row["task_id"], row["condition"], int(row["repeat"])))
    if workers > 1:
        return _run_parallel(
            config, pins, tasks, conditions, repeats, resume, workers, budget, log, logged
        )
    try:
        for repeat in range(1, repeats + 1):
            for task in tasks:
                for condition in conditions:
                    if resume:
                        saved = _completed_run(config, task, condition, repeat)
                        if saved is not None:
                            # Reused, not rerun: its spend still counts against the cap.
                            budget.spend(saved.total_tokens, saved.cost_usd)
                            results.append(saved)
                            if (saved.repo, saved.task_id, saved.condition, saved.repeat) not in logged:
                                with log.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(asdict(saved)) + "\n")
                            continue
                    budget.check()
                    check_disk(config.workdir, config.min_free_gb)
                    try:
                        result = run_one(config, pins[task.repo], task, condition, repeat)
                    except (RateLimited, BudgetExhausted, AccountingError, ServerNotReady):
                        _record_stopped(config, task, condition, repeat, budget, results, log)
                        raise
                    budget.spend(result.total_tokens, result.cost_usd)
                    results.append(result)
                    with log.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(result)) + "\n")
    except (BudgetExhausted, RateLimited, AccountingError, LowDisk, ServerNotReady) as exc:
        return results, str(exc)
    return results, ""


def _run_parallel(
    config: RunnerConfig,
    pins: dict[str, RepoPin],
    tasks: list[Task],
    conditions: tuple[str, ...],
    repeats: int,
    resume: bool,
    workers: int,
    budget: Budget,
    log: Path,
    logged: set[tuple[str, str, str, int]],
) -> tuple[list[RunResult], str]:
    """The matrix with up to *workers* runs in flight.

    Admission holds a reserve for every run in flight (``Budget.in_flight``), so the cap is
    never passed by more than what those runs overshoot their own reserve. The first stop
    (rate limit, over-reserve run, missing cost data, budget, or a KeyboardInterrupt)
    admits no new run and ends the runs in flight now: queued runs are cancelled and the
    OpenCode and Beacon processes of running ones are killed, so the stop is prompt and
    nothing is waited on to finish naturally. Interrupted runs are still recorded, with
    the error their killed run saved. Shared state that runs would otherwise race to
    create -- the clone cache and the beacons -- is prepared once, up front. Results come
    back in matrix order, whatever order the runs finished in.
    """
    jobs = [
        (repeat, task, condition)
        for repeat in range(1, repeats + 1)
        for task in tasks
        for condition in conditions
    ]
    order = {(t.repo, t.task_id, c, r): index for index, (r, t, c) in enumerate(jobs)}
    cache = config.workdir / "cache"
    for pin in {pins[t.repo].name: pins[t.repo] for t in tasks}.values():
        source = _ensure_source(pin, cache)
        if source is not None:
            try:
                seal_repo(pin, config.workdir, source)
            except (subprocess.CalledProcessError, OSError, UnicodeError):
                pass  # each run warns and seals by copy
        if set(conditions) & {"B", "C", "D", "H", "R"}:
            build_beacon(config, pin)
        if "R" in conditions:
            export_beacon_snapshot(config, pin)

    lock = threading.Lock()
    results: list[RunResult] = []
    stop = ""

    def record(result: RunResult) -> None:
        with lock:
            budget.spend(result.total_tokens, result.cost_usd)
            results.append(result)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(result)) + "\n")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="beacon-eval") as pool:
        pending: dict[Future[RunResult], tuple[int, Task, str]] = {}
        queue_ = iter(jobs)
        unexpected: BaseException | None = None

        def halt() -> None:
            """The matrix stopped: cancel queued runs, kill the running children now."""
            for future in pending:
                future.cancel()
            _kill_live_children()

        def first_stop(message: str) -> None:
            nonlocal stop
            if not stop:
                stop = message
                halt()

        while True:
            try:
                while not stop and len(pending) < workers:
                    job = next(queue_, None)
                    if job is None:
                        break
                    repeat, task, condition = job
                    if resume:
                        saved = _completed_run(config, task, condition, repeat)
                        if saved is not None:
                            key = (saved.repo, saved.task_id, saved.condition, saved.repeat)
                            if key in logged:
                                with lock:
                                    budget.spend(saved.total_tokens, saved.cost_usd)
                                    results.append(saved)
                            else:
                                record(saved)
                            continue
                    with lock:
                        try:
                            budget.check()
                            check_disk(config.workdir, config.min_free_gb)
                        except (BudgetExhausted, LowDisk) as exc:
                            first_stop(str(exc))
                            break
                        budget.in_flight += 1
                    pending[pool.submit(run_one, config, pins[task.repo], task, condition, repeat)] = job
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    repeat, task, condition = pending.pop(future)
                    with lock:
                        budget.in_flight -= 1
                    try:
                        result = future.result()
                    except CancelledError:
                        continue  # a queued run the stop cancelled; nothing ran
                    except (RateLimited, BudgetExhausted, AccountingError, ServerNotReady) as exc:
                        _record_stopped(config, task, condition, repeat, budget, results, log, lock)
                        first_stop(str(exc))
                        continue
                    except BaseException as exc:  # noqa: BLE001 - re-raised once in-flight runs end
                        unexpected = unexpected or exc
                        first_stop(f"{type(exc).__name__}: {exc}")
                        continue
                    record(result)
            except BaseException as exc:  # noqa: BLE001 - any stop ends the matrix promptly
                # A KeyboardInterrupt (Ctrl+C lands here, in the waiting main thread) or
                # anything else escaping the loop: no new runs, queued runs cancelled,
                # the OpenCode and Beacon processes of runs in flight killed, and those
                # runs drained -- their saved results recorded -- instead of being waited
                # on to finish naturally.
                unexpected = unexpected or exc
                first_stop(f"{type(exc).__name__}: {exc}")
                done, _ = wait(pending, timeout=DRAIN_TIMEOUT_S)
                for future in done:
                    pending.pop(future)
                    with lock:
                        budget.in_flight -= 1
                    try:
                        result = future.result()
                    except BaseException:
                        continue  # cancelled with the queue, or failed by the same stop
                    record(result)
                break
        if unexpected is not None:
            raise unexpected
    results.sort(key=lambda r: order.get((r.repo, r.task_id, r.condition, r.repeat), len(order)))
    return results, stop


def _completed_run(config: RunnerConfig, task: Task, condition: str, repeat: int) -> RunResult | None:
    saved = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}" / "result.json"
    if not saved.is_file():
        return None
    result = RunResult(**json.loads(saved.read_text(encoding="utf-8")))
    return result if result.answer is not None and not result.error else None


def _record_stopped(
    config: RunnerConfig, task: Task, condition: str, repeat: int, budget: Budget,
    results: list[RunResult], log: Path, lock: threading.Lock | None = None,
) -> None:
    saved = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}" / "result.json"
    if not saved.is_file():
        return
    data = json.loads(saved.read_text(encoding="utf-8"))
    result = RunResult(**data)
    with lock if lock is not None else nullcontext():
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
    "LowDisk",
    "RateLimited",
    "RunnerConfig",
    "SERVER_LOG",
    "SERVER_READY_TIMEOUT_S",
    "ServerNotReady",
    "TOOL_NOTE",
    "beacon_http_process",
    "build_prompt",
    "disabled_tools",
    "export_beacon_snapshot",
    "off_origin_fetches",
    "parse_events",
    "pick_free_port",
    "resolve_opencode",
    "run_matrix",
    "rebuild_checkout",
    "run_one",
    "save_changes",
    "seal_checkout",
    "seal_repo",
    "stream_opencode",
    "webfetch_hosts",
]
