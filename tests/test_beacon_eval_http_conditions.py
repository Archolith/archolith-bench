"""Conditions H (Beacon HTTP JSON) and R (Beacon MCP over Streamable HTTP).

Offline by default: the server lifecycle is exercised against a tiny stand-in
script and the runs against a stub in OpenCode's place. One integration test
starts the real Beacon servers from the configured source (skipped, never
failed, when Beacon is not importable there). No model or paid call is made.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

import pytest

from archolith_bench.beacon_eval import CONDITIONS, DEFAULT_CONDITIONS
from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.isolation import beacon_remote_server
from archolith_bench.beacon_eval.models import Gold, RepoPin, Task
from archolith_bench.beacon_eval.report import render
from archolith_bench.beacon_eval.runner import (
    BUILTIN_TOOLS,
    HTTP_READY_LINE,
    MCP_HTTP_READY_LINE,
    SERVER_LOG,
    RunnerConfig,
    ServerNotReady,
    _http_server_spec,
    _port_accepts,
    beacon_http_process,
    build_prompt,
    disabled_tools,
    export_beacon_snapshot,
    pick_free_port,
    rescore,
    run_matrix,
)

GOLD = Gold()
TASK = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD)

#: Where the real-Beacon integration test finds a Beacon build (#26 phases 1-3):
#: BEACON_EVAL_BEACON_SRC (a Beacon ``src`` dir) and, optionally, BEACON_EVAL_BEACON_PYTHON
#: (a Python with Beacon's dependencies; defaults to this interpreter).
BEACON_SRC = os.environ.get("BEACON_EVAL_BEACON_SRC", "")
BEACON_PYTHON = os.environ.get("BEACON_EVAL_BEACON_PYTHON", sys.executable)

#: A stand-in server for lifecycle tests: binds the given host/port, prints its
#: ready line, then serves (mode "ready") or stalls (modes "decoy"/"silent").
STAND_IN = r"""
import socket
import sys
import time

host, port, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if mode == "decoy":
    print("decoy line, never the ready line", file=sys.stderr, flush=True)
    time.sleep(120)
if mode == "silent":
    time.sleep(120)
server = socket.socket()
server.bind((host, port))
server.listen(4)
print(f"STAND-IN ready on {host}:{port}", file=sys.stderr, flush=True)
while True:
    conn, _ = server.accept()
    conn.close()
"""


def _stand_in(tmp_path: Path) -> Path:
    script = tmp_path / "stand_in_server.py"
    script.write_text(STAND_IN, encoding="utf-8")
    return script


def _refuses(port: int, timeout_s: float = 10.0) -> bool:
    """True once nothing accepts on *port* any more (the server was stopped)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _port_accepts(port):
            return True
        time.sleep(0.2)
    return False


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Agents\nNever commit secrets.\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


def _pin(root: Path) -> RepoPin:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return RepoPin(name="demo", url="", commit=commit, local_path=str(root))


def _user_config(tmp_path: Path) -> Path:
    path = tmp_path / "user-opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "other/x",
        "provider": {"deepseek": {"options": {"baseURL": "u"}}, "other": {"options": {}}},
    }), encoding="utf-8")
    return path


def _beacon_fixture_repo(root: Path) -> Path:
    """A tiny repository whose beacon passes Beacon's publication gates cleanly."""
    (root / "docs").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Demo\n\nThe widget flip guard lives in the module.\n", encoding="utf-8")
    (root / "docs" / "guide.md").write_text("# Guide\n\nFlipping widgets needs two spans.\n", encoding="utf-8")
    (root / "beacon.yaml").write_text(
        'schema: beacon/manifest\nversion: "0.1"\n'
        "project:\n"
        "  name: demo\n"
        '  description: "Demo project exercising the beacon eval fixture."\n'
        '  tagline: "Demo fixture."\n'
        "  status: current\n"
        "build_and_test:\n"
        '  test: "pytest -q"\n'
        "guardrails:\n"
        "  - id: no-secrets\n"
        '    rule: "Never commit secrets."\n'
        "    severity: medium\n"
        "    status: current\n",
        encoding="utf-8",
    )
    for args in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


# ---------------------------------------------------------------------------
# Conditions, tool lists, mcp blocks, prompts
# ---------------------------------------------------------------------------


def test_h_and_r_are_opt_in_conditions() -> None:
    assert {"H", "R"} <= set(CONDITIONS)
    assert "H" not in DEFAULT_CONDITIONS and "R" not in DEFAULT_CONDITIONS


def test_condition_r_mcp_block_is_a_remote_beacon_without_headers() -> None:
    mcp = beacon_remote_server("http://127.0.0.1:8123/mcp")
    assert mcp == {"beacon": {"type": "remote", "url": "http://127.0.0.1:8123/mcp", "enabled": True}}


def test_disabled_tools_leave_webfetch_only_for_h() -> None:
    assert disabled_tools("R") == BUILTIN_TOOLS  # like D
    assert disabled_tools("D") == BUILTIN_TOOLS
    assert disabled_tools("A") == ()
    h = disabled_tools("H")
    assert "webfetch" not in h and "websearch" in h
    # Every file and shell tool is off: the checkout is unreachable.
    assert {"bash", "edit", "glob", "grep", "list", "patch", "read", "write"} <= set(h)
    assert set(h) == set(BUILTIN_TOOLS) - {"webfetch"}


def test_the_http_paragraph_appears_only_for_h() -> None:
    h = build_prompt(TASK, "H", http_url="http://127.0.0.1:9999")
    assert "Project knowledge is served over HTTP at http://127.0.0.1:9999." in h
    assert (
        "Start with GET http://127.0.0.1:9999/.well-known/archolith-beacon, "
        "which lists the routes and a recommended flow." in h
    )
    for condition in ("A", "B", "C", "D", "M", "R"):
        assert "served over HTTP" not in build_prompt(TASK, condition, "manifest")
    assert h.index("served over HTTP") < h.index("Use whatever tools") < h.index('"citations"')


def test_h_without_a_port_is_a_wiring_error() -> None:
    with pytest.raises(ValueError, match="http_url"):
        build_prompt(TASK, "H")


# ---------------------------------------------------------------------------
# Port selection and the managed server lifecycle
# ---------------------------------------------------------------------------


def test_port_selection_skips_bound_ports() -> None:
    held = socket.socket()
    try:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert all(pick_free_port() != taken for _ in range(25))
    finally:
        held.close()


def test_port_selection_is_distinct_under_concurrency() -> None:
    picked: list[int] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def pick() -> None:
        barrier.wait()
        try:
            picked.append(pick_free_port())
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=pick) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(set(picked)) == len(picked) == 8


def test_the_server_is_stopped_when_the_body_raises(tmp_path: Path) -> None:
    script = _stand_in(tmp_path)
    log = tmp_path / SERVER_LOG
    ports: list[int] = []
    with pytest.raises(RuntimeError, match="boom"):
        with beacon_http_process(
            [sys.executable, str(script), "127.0.0.1", "{port}", "ready"], log, "STAND-IN ready"
        ) as port:
            ports.append(port)
            raise RuntimeError("boom")
    assert _refuses(ports[0])  # killed, not left listening
    assert "STAND-IN ready" in log.read_text(encoding="utf-8")


def test_a_ready_timeout_fails_with_the_stderr_tail(tmp_path: Path) -> None:
    script = _stand_in(tmp_path)
    log = tmp_path / SERVER_LOG
    with pytest.raises(ServerNotReady) as excinfo:
        with beacon_http_process(
            [sys.executable, str(script), "127.0.0.1", "{port}", "decoy"], log, "NEVER PRINTED",
            timeout_s=2,
        ):
            pytest.fail("the server never became ready")
    message = str(excinfo.value)
    assert "was not ready within 2s" in message and "NEVER PRINTED" in message
    assert "decoy line" in message  # the stderr tail
    assert "decoy line" in log.read_text(encoding="utf-8")


def test_an_exited_server_fails_with_the_stderr_tail(tmp_path: Path) -> None:
    script = tmp_path / "exits.py"
    script.write_text(
        "import sys\nprint('refused: snapshot blocked', file=sys.stderr, flush=True)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    with pytest.raises(ServerNotReady) as excinfo:
        with beacon_http_process(
            [sys.executable, str(script), "{port}"], tmp_path / SERVER_LOG, "NEVER PRINTED"
        ):
            pytest.fail("the server never became ready")
    assert "exited" in str(excinfo.value) and "snapshot blocked" in str(excinfo.value)


def test_managed_servers_bind_nothing_but_loopback(tmp_path: Path) -> None:
    config = RunnerConfig(workdir=tmp_path, beacon_python="py", opencode_cmd=["x"])
    manifest, snapshot = tmp_path / "m.yaml", tmp_path / "s.json"
    assert _http_server_spec(config, "A", manifest, snapshot) is None
    for condition in ("B", "C", "D", "M"):
        assert _http_server_spec(config, condition, manifest, snapshot) is None
    for condition, subcommand in (("H", "serve-http"), ("R", "serve")):
        cmd, ready_line, _env = _http_server_spec(config, condition, manifest, snapshot)
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
        assert "0.0.0.0" not in cmd and "{port}" in cmd
        assert ("serve-http" if condition == "H" else "--transport") in cmd
        assert ready_line == (HTTP_READY_LINE if condition == "H" else MCP_HTTP_READY_LINE)


# ---------------------------------------------------------------------------
# End to end with a stub instead of OpenCode
# ---------------------------------------------------------------------------

STUB = r"""
import json, os, sys
prompt = sys.stdin.read()
home = os.environ.get("XDG_CONFIG_HOME", "")
config = json.load(open(os.path.join(home, "opencode", "opencode.json"), encoding="utf-8"))
beacon = config.get("mcp", {}).get("beacon", {})
answer = {
    "docs": [], "files": [], "commands": [], "guardrails": [], "verdict": "",
    "plan": [], "citations": [],
    "record": {
        "mcp": sorted(config.get("mcp", {})),
        "tools_off": sorted(k for k, v in config.get("tools", {}).items() if v is False),
        "beacon_type": beacon.get("type", ""),
        "beacon_url": beacon.get("url", ""),
        "http_paragraph": "served over HTTP at" in prompt,
        "config_keys": sorted(config),
    },
}
print(json.dumps({"type": "text", "part": {"type": "text", "text": "```json\n" + json.dumps(answer) + "\n```"}}))
print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 100, "output": 10}, "cost": 0.001}}))
"""

FIXED_PORT = 8777


@pytest.fixture
def stub_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A runner whose OpenCode is a stub and whose Beacon servers never start.

    ``beacon_http_process`` is replaced with a fake that yields a fixed port and
    records being entered and exited, so the wiring is checked without Beacon.
    """
    stub = tmp_path / "stub_opencode.py"
    stub.write_text(STUB, encoding="utf-8")
    started: list[tuple[list[str], str, str]] = []
    stopped: list[list[str]] = []

    @contextmanager
    def fake_server(cmd: list[str], log_path: Path, ready_line: str,
                    env: dict[str, str] | None = None, timeout_s: float = 30.0) -> Iterator[int]:
        started.append((list(cmd), ready_line, log_path.name))
        yield FIXED_PORT
        stopped.append(list(cmd))

    def fake_build(config: RunnerConfig, pin: RepoPin) -> Path:
        manifest = config.workdir / "beacons" / pin.name / "beacon.generated.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("project: x\n", encoding="utf-8")
        return manifest

    def fake_export(config: RunnerConfig, pin: RepoPin) -> Path:
        snapshot = config.workdir / "beacons" / pin.name / "beacon.snapshot.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("{}\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(runner_mod, "beacon_http_process", fake_server)
    monkeypatch.setattr(runner_mod, "build_beacon", fake_build)
    monkeypatch.setattr(runner_mod, "export_beacon_snapshot", fake_export)
    config = RunnerConfig(
        workdir=tmp_path / "work",
        beacon_python="py",
        opencode_cmd=[sys.executable, str(stub)],
        model="deepseek/deepseek-v4-flash",  # the provider _user_config defines
        config_source=_user_config(tmp_path),
        budget_tokens=10_000_000,
        run_reserve_tokens=80_000,
        timeout_s=60,
    )
    return config, _pin(_repo(tmp_path)), started, stopped


def test_h_and_r_get_server_tools_prompt_and_lifecycle(stub_harness, tmp_path: Path) -> None:
    config, pin, started, stopped = stub_harness
    results, stopped_why = run_matrix(config, {"demo": pin}, [TASK], ("A", "D", "H", "R"))
    assert stopped_why == "" and all(r.answer is not None for r in results)
    records = {r.condition: r.answer["record"] for r in results}

    # Tool lists: H keeps only webfetch; R is like D; A keeps everything.
    assert records["A"]["tools_off"] == []
    assert set(records["D"]["tools_off"]) == set(BUILTIN_TOOLS)
    assert set(records["H"]["tools_off"]) == set(BUILTIN_TOOLS) - {"webfetch"}
    assert "websearch" in records["H"]["tools_off"] and "webfetch" not in records["H"]["tools_off"]
    assert set(records["R"]["tools_off"]) == set(BUILTIN_TOOLS)

    # mcp blocks: only D (local stdio) and R (remote) have one, and only R's is remote.
    assert records["A"]["mcp"] == [] and records["H"]["mcp"] == []
    assert records["D"]["mcp"] == ["beacon"] and records["R"]["mcp"] == ["beacon"]
    assert records["R"]["beacon_type"] == "remote"
    assert records["R"]["beacon_url"] == f"http://127.0.0.1:{FIXED_PORT}/mcp"

    # The HTTP paragraph is H's alone, and the saved prompt shows the real port.
    assert [records[c]["http_paragraph"] for c in ("A", "D", "H", "R")] == [False, False, True, False]
    h_prompt = (config.workdir / "runs" / f"demo-{TASK.task_id}-H-1" / "prompt.txt").read_text(encoding="utf-8")
    assert f"Start with GET http://127.0.0.1:{FIXED_PORT}/.well-known/archolith-beacon" in h_prompt
    d_prompt = (config.workdir / "runs" / f"demo-{TASK.task_id}-D-1" / "prompt.txt").read_text(encoding="utf-8")
    assert "served over HTTP" not in d_prompt

    # The managed server runs for exactly H and R, loopback-only, with a run-dir log.
    assert len(started) == len(stopped) == 2
    for cmd, ready_line, log_name in started:
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and "0.0.0.0" not in cmd
        assert log_name == SERVER_LOG
    by_ready = {ready: cmd for cmd, ready, _ in started}
    assert by_ready[HTTP_READY_LINE][by_ready[HTTP_READY_LINE].index("serve-http") + 1] == "--manifest"
    assert by_ready[MCP_HTTP_READY_LINE][by_ready[MCP_HTTP_READY_LINE].index("--transport") + 1] == "http"
    assert stopped == [cmd for cmd, _, _ in started]  # every server was stopped

    # Reports and rescore handle the new conditions, server log or not.
    (config.workdir / "runs" / f"demo-{TASK.task_id}-H-1" / SERVER_LOG).write_text(
        HTTP_READY_LINE + "\n", encoding="utf-8"
    )
    report = render(results, {"Model": "stub"})
    table_rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in report.splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    conditions_in_order = [row[0] for row in table_rows if row[0] in {"A", "D", "H", "R"}]
    assert conditions_in_order == ["A", "D", "H", "R"]  # sorted, H between D and M
    for condition in ("A", "D", "H", "R"):
        assert f"| {condition} | 1 |" in report
    assert len(rescore(config.workdir, [TASK])) == 4


def test_a_server_that_never_becomes_ready_fails_the_run_and_stops(
    stub_harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, pin, _started, _stopped = stub_harness

    @contextmanager
    def refusing(cmd: list[str], log_path: Path, ready_line: str,
                 env: dict[str, str] | None = None, timeout_s: float = 30.0) -> Iterator[int]:
        raise ServerNotReady(
            f"the beacon server exited without reporting {ready_line!r}; "
            "its stderr ends with: snapshot blocked: unresolved publication warnings"
        )
        yield  # pragma: no cover

    monkeypatch.setattr(runner_mod, "beacon_http_process", refusing)
    results, stopped = run_matrix(config, {"demo": pin}, [TASK], ("A", "H"))
    assert "snapshot blocked" in stopped
    assert [r.condition for r in results] == ["A", "H"]
    failed = results[1]
    assert "snapshot blocked" in failed.error and failed.answer is None
    run_dir = config.workdir / "runs" / f"demo-{TASK.task_id}-H-1"
    assert not (run_dir / "prompt.txt").exists()  # nothing ran, so nothing was prompted
    assert not (run_dir / "checkout").exists()  # the checkout was still retired


# ---------------------------------------------------------------------------
# Real Beacon integration (skipped without a usable Beacon source)
# ---------------------------------------------------------------------------


def _beacon_skip_reason() -> str:
    if not BEACON_SRC:
        return "set BEACON_EVAL_BEACON_SRC to a Beacon source tree with #26 phases 1-3"
    try:
        probe = subprocess.run(
            [str(BEACON_PYTHON), "-c", "import beacon"],
            env={**os.environ, "PYTHONPATH": str(BEACON_SRC)},
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Beacon is not importable from the configured source ({BEACON_SRC}): {exc}"
    if probe.returncode != 0:
        return f"Beacon is not importable from the configured source ({BEACON_SRC})"
    return ""


_SKIP_BEACON = _beacon_skip_reason()


@pytest.mark.skipif(_SKIP_BEACON != "", reason=_SKIP_BEACON)
def test_real_beacon_serves_discovery_and_mcp_over_http(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    config = RunnerConfig(
        workdir=tmp_path / "work", beacon_python=str(BEACON_PYTHON), beacon_src=str(BEACON_SRC),
        opencode_cmd=["unused-by-this-test"],
    )
    pin = _pin(_beacon_fixture_repo(tmp_path / "repo"))
    manifest = runner_mod.build_beacon(config, pin).resolve()
    snapshot = export_beacon_snapshot(config, pin)
    h_cmd, h_ready, h_env = _http_server_spec(config, "H", manifest, snapshot)
    with beacon_http_process(h_cmd, tmp_path / SERVER_LOG, h_ready, h_env, timeout_s=120) as port:
        assert _port_accepts(port)
        with urlopen(f"http://127.0.0.1:{port}/.well-known/archolith-beacon", timeout=15) as response:
            discovery = json.loads(response.read().decode("utf-8"))
        assert "recommended_flow" in discovery
    assert HTTP_READY_LINE in (tmp_path / SERVER_LOG).read_text(encoding="utf-8")
    r_cmd, r_ready, r_env = _http_server_spec(config, "R", manifest, snapshot)
    with beacon_http_process(r_cmd, tmp_path / "mcp-server.log", r_ready, r_env, timeout_s=120) as port:
        assert _port_accepts(port)

        async def tools() -> list[str]:
            from fastmcp import Client

            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                return sorted(tool.name for tool in await client.list_tools())

        served = asyncio.run(tools())
    assert "beacon_search" in served and "beacon_read" in served
