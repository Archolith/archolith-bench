"""Fail-closed liveness gate for partial resume (scripts/longmemeval/lib/liveness_gate.py)."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "liveness_gate.py"
    spec = importlib.util.spec_from_file_location("lme_liveness_gate_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_liveness_gate_test"] = module
    spec.loader.exec_module(module)
    return module


gate = _load()
SENT = "END-OF-LIVENESS-OUTPUT"


def test_evaluate_unknown_clients_fails_closed() -> None:
    assert gate.evaluate(None, [], own_pid=1)[0] == 2


def test_evaluate_unknown_serves_fails_closed() -> None:
    assert gate.evaluate([], None, own_pid=1)[0] == 2


def test_evaluate_only_ssh_clients_is_clear_any_case() -> None:
    code, _ = gate.evaluate([(10, "ssh"), (11, "SSH.EXE"), (12, " Ssh.exe ")], [], own_pid=1)
    assert code == 0


def test_evaluate_python_bolt_client_is_offender_and_named() -> None:
    code, msg = gate.evaluate([(4242, "python")], [], own_pid=1)
    assert code == 1
    assert "4242" in msg


def test_evaluate_serve_process_is_offender() -> None:
    code, msg = gate.evaluate([], [(777, "menhir serve --port 8151")], own_pid=1)
    assert code == 1
    assert "777" in msg


def test_evaluate_ignores_own_pid_in_both_lists() -> None:
    assert gate.evaluate([(99, "python")], [(99, "x serve --port 8151")], own_pid=99)[0] == 0
    # but another pid alongside own pid still counts
    assert gate.evaluate([(99, "python"), (5, "python")], [], own_pid=99)[0] == 1


def test_parse_tab_lines_without_sentinel_is_none() -> None:
    assert gate._parse_tab_lines("12\tpython\n13\tssh\n") is None
    assert gate._parse_tab_lines("") is None


def test_parse_tab_lines_with_sentinel_skips_blank_and_non_numeric() -> None:
    out = f"12\tpython\r\n\r\nnotanumber\tfoo\n13\tssh.exe\n{SENT}\n99\tafter\n"
    assert gate._parse_tab_lines(out) == [(12, "python"), (13, "ssh.exe")]


def test_parse_tab_lines_sentinel_only_is_empty_not_none() -> None:
    assert gate._parse_tab_lines(SENT + "\n") == []


def test_bolt_clients_windows_parses_and_none_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.os, "name", "nt")
    monkeypatch.setattr(gate, "_powershell", lambda script: f"31\tpython\n{SENT}\n")
    assert gate.bolt_clients(7690) == [(31, "python")]
    monkeypatch.setattr(gate, "_powershell", lambda script: None)
    assert gate.bolt_clients(7690) is None
    # Truncated output (no sentinel) must not read as clear.
    monkeypatch.setattr(gate, "_powershell", lambda script: "31\tpython\n")
    assert gate.bolt_clients(7690) is None


def test_serve_processes_windows_script_does_not_contain_own_needle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.os, "name", "nt")
    seen: list[str] = []

    def fake_ps(script: str) -> str:
        seen.append(script)
        return f"55\tmenhir serve --port 8151\n{SENT}\n"

    monkeypatch.setattr(gate, "_powershell", fake_ps)
    assert gate.serve_processes(8151) == [(55, "menhir serve --port 8151")]
    assert len(seen) == 1
    assert "serve --port 8151" not in seen[0]
    # ...but the needle is still assembled from its parts at runtime.
    assert "'serve --port ' + '8151'" in seen[0]


def test_serve_processes_windows_script_does_not_contain_driver_needle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.os, "name", "nt")
    seen: list[str] = []
    monkeypatch.setattr(gate, "_powershell", lambda script: seen.append(script) or f"{SENT}\n")
    assert gate.serve_processes(8151) == []
    assert "localhost:8151" not in seen[0] and "ingest.py" not in seen[0]


def test_surviving_ingest_driver_for_this_port_is_offender(monkeypatch: pytest.MonkeyPatch) -> None:
    # A driver from the stopped run would post turns to the NEW server on the same port.
    monkeypatch.setattr(gate.os, "name", "posix")
    ps = (
        "101 python lib/ingest.py --limit 8 --menhir-url http://localhost:8151 --manifest m.json\n"
        "102 python lib/ingest.py --limit 8 --menhir-url http://localhost:8152 --manifest m.json\n"
        "103 python liveness_gate.py --bolt-port 7851 --serve-port 8151\n"
    )
    monkeypatch.setattr(gate, "_run", lambda cmd: ps)
    rows = gate.serve_processes(8151)
    assert [pid for pid, _ in rows] == [101]
    code, message = gate.evaluate([], rows, own_pid=103)
    assert code == 1 and "pid=101" in message


def test_serve_processes_windows_none_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.os, "name", "nt")
    monkeypatch.setattr(gate, "_powershell", lambda script: None)
    assert gate.serve_processes(8151) is None


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    argv = ["--bolt-port", "7690", "--serve-port", "8151"]

    monkeypatch.setattr(gate, "bolt_clients", lambda port: [(1, "ssh")])
    monkeypatch.setattr(gate, "serve_processes", lambda port: [])
    assert gate.main(argv) == 0
    assert "ok" in capsys.readouterr().out

    monkeypatch.setattr(gate, "bolt_clients", lambda port: [(os.getpid() + 100000, "python")])
    assert gate.main(argv) == 1
    assert "REFUSED" in capsys.readouterr().err

    monkeypatch.setattr(gate, "bolt_clients", lambda port: [])
    monkeypatch.setattr(gate, "serve_processes", lambda port: None)
    assert gate.main(argv) == 2
