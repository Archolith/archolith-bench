"""Run the task x condition x repeat matrix through ``opencode run --format json``.

Every run gets a fresh export of the repository at its pinned commit, sealed as its own
git repository (condition A never sees a beacon), a private OpenCode config home, and a
fixed prompt on stdin. Conditions H and R additionally get a managed Beacon server on a
free loopback port (HTTP JSON for H, MCP over Streamable HTTP for R), started before
OpenCode and stopped afterwards whatever the outcome. OpenCode's events are streamed: a
run is killed as soon as its tokens pass the per-run reserve or a rate-limit error
appears on stdout or stderr. The matrix admits a run only while the reserve still fits
under the cap, and stops at the first rate limit (never retrying), an over-reserve run,
a run with no usage data, or a server that never became ready.
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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

    Parallel ``--workers`` each get their own port, so their servers never collide.
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


def _server_error(what: str, ready_line: str, tail: deque[str]) -> str:
    return (
        f"the beacon server {what} without reporting {ready_line!r}; "
        f"its stderr ends with: {''.join(tail).strip()[-SERVER_TAIL_CHARS:]}"
    )


@contextmanager
def beacon_http_process(
    cmd: list[str],
    log_path: Path,
    ready_line: str,
    env: dict[str, str] | None = None,
    timeout_s: float = SERVER_READY_TIMEOUT_S,
) -> Iterator[int]:
    """Start *cmd* on a free loopback port, yield the port, and always stop it.

    A ``{port}`` placeholder in *cmd*'s arguments is replaced with the port. The server
    is ready once stderr prints *ready_line* or the port accepts connections; on early
    exit or after *timeout_s* the run fails with :class:`ServerNotReady`, carrying the
    server's stderr tail. stderr streams into *log_path* (``beacon-server.log`` in the
    run dir). The process tree is killed on every way out of the body -- success,
    error, timeout, budget stop, rate-limit stop -- the way ``stream_opencode`` stops
    OpenCode.
    """
    port = pick_free_port()
    proc = subprocess.Popen(
        [part.replace("{port}", str(port)) for part in cmd],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stderr is not None
    tail: deque[str] = deque(maxlen=20)
    ready = threading.Event()

    def pump() -> None:
        with log_path.open("w", encoding="utf-8") as sink:
            for line in proc.stderr:
                sink.write(line)
                sink.flush()
                tail.append(line)
                if ready_line in line:
                    ready.set()

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    try:
        deadline = time.monotonic() + timeout_s
        while not ready.is_set() and not _port_accepts(port):
            if proc.poll() is not None:
                pump_thread.join(timeout=5)
                raise ServerNotReady(_server_error("exited", ready_line, tail))
            if time.monotonic() >= deadline:
                raise ServerNotReady(_server_error(f"was not ready within {timeout_s:.0f}s", ready_line, tail))
            time.sleep(0.2)
        yield port
    finally:
        _kill_tree(proc)
        pump_thread.join(timeout=5)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


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
    """The canonical snapshot condition R's MCP server serves, written once per pin."""
    root = config.workdir / "beacons" / pin.name
    snapshot = root / "beacon.snapshot.json"
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

    The returned command is loopback-only and carries a ``{port}`` placeholder for
    :func:`beacon_http_process` to fill.
    """
    if condition == "H" and manifest is not None:
        cmd = [
            config.beacon_python, "-m", "beacon", "serve-http",
            "--manifest", str(manifest), "--docs-root", str(manifest.parent),
            "--host", "127.0.0.1", "--port", "{port}",
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
    log: EventLog | None = None,
) -> tuple[EventLog, str, int | None]:
    """Run OpenCode, feeding events as they arrive; returns (log, stop reason, exit code).

    The raw streams are kept in ``events.jsonl`` and ``stderr.log``. Stop reasons:
    "" (finished), "rate_limited", "over_reserve", "timeout". Passing *log* continues it
    (a resumed session): limits apply to the run's running totals and the files are appended.
    """
    mode = "a" if log is not None else "w"
    log = log if log is not None else EventLog()
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
    if condition == "M" and not (config.memory_url and config.memory_key):
        raise ValueError("condition M needs memory_url and memory_key")
    # Absolute paths: PWD and B's --manifest are resolved by processes running in the checkout.
    run_dir = (config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}").resolve()
    cache = config.workdir / "cache"
    # A rerun (--resume) starts from a clean export and inherits nothing of the last attempt.
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
    snapshot = export_beacon_snapshot(config, pin) if condition == "R" else None
    manifest_text = manifest.read_text(encoding="utf-8") if manifest is not None and condition == "C" else ""
    started = time.monotonic()
    keys = load_api_keys(config.env_file) if config.env_file else {}
    spec = _http_server_spec(config, condition, manifest, snapshot)
    log, reason, code, resumes = EventLog(), "", None, 0
    server_cm: AbstractContextManager[int | None] = (
        beacon_http_process(spec[0], run_dir / SERVER_LOG, spec[1], spec[2]) if spec else nullcontext(None)
    )
    server_error = ""
    try:
        with server_cm as port:
            base_url = f"http://127.0.0.1:{port}" if port is not None else ""
            prompt = build_prompt(
                task,
                condition,
                manifest_text,
                http_url=base_url if condition == "H" else "",
            )
            (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            mcp = _mcp_block(config, condition, manifest, base_url)
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
                # An inherited PWD (Git Bash, MSYS, most shells) may root OpenCode in the
                # caller's repo instead of the checkout; the fake-provider check exercises this.
                env["PWD"] = str(checkout)
                reserve_usd = config.run_reserve_usd if config.budget_usd is not None else None
                log, reason, code = stream_opencode(
                    cmd, prompt, checkout, env, run_dir, config.run_reserve_tokens, config.timeout_s,
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
                        [*cmd, "--session", log.session_id], RESUME_PROMPT, checkout, env, run_dir,
                        config.run_reserve_tokens, config.timeout_s, reserve_usd, log=log,
                    )
    except ServerNotReady as exc:
        server_error = str(exc)
    text = "\n".join(log.texts)
    answer = extract_answer(text)
    error = server_error or reason or ("; ".join(log.errors) if log.errors else "")
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
    )
    result.scores = score(answer, task.gold, checkout)
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    _redact(run_dir, [*keys.values(), *([config.memory_key] if config.memory_key else [])])
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
    (rate limit, over-reserve run, missing cost data, or the budget) admits no new run; runs
    already in flight finish, are recorded and counted. Shared state that runs would otherwise
    race to create -- the clone cache and the beacons -- is prepared once, up front. Results
    come back in matrix order, whatever order the runs finished in.
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
        while True:
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
                        stop = str(exc)
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
                except (RateLimited, BudgetExhausted, AccountingError, ServerNotReady) as exc:
                    _record_stopped(config, task, condition, repeat, budget, results, log, lock)
                    stop = stop or str(exc)
                    continue
                except BaseException as exc:  # noqa: BLE001 - re-raised once in-flight runs end
                    unexpected = unexpected or exc
                    stop = stop or f"{type(exc).__name__}: {exc}"
                    continue
                record(result)
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
]
