"""build --check: progress from the manifest, state log and a serving Menhir; acceptance via validate_run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from archolith_bench.build import __main__ as build_cli
from archolith_bench.build import check

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = str(REPO / "builds" / "default.yaml")
CONFIG = {"container": {"neo4j_name": "menhir-lme-neo4j", "menhir_port": 8102}, "dataset": {"limit": 30}}


def _config(name: str) -> dict:
    return {**CONFIG, "container": {**CONFIG["container"], "neo4j_name": name}}


def test_paths_follow_config_sh(tmp_path: Path):
    p = check.check_paths(_config("menhir-lme-neo4j"), {}, str(tmp_path))
    assert p["manifest"] == tmp_path / "manifest.json"
    assert p["state_log"] == tmp_path / "ingest-state-manifest.jsonl"
    assert p["provenance"] == tmp_path / "graph-provenance-menhir-lme-neo4j.json"
    p = check.check_paths(_config("smoke"), {}, str(tmp_path))
    assert p["manifest"] == tmp_path / "manifest-smoke.json"
    assert p["state_log"] == tmp_path / "ingest-state-manifest-smoke.jsonl"
    p = check.check_paths(_config("smoke"), {"LME_RESULTS_DIR": str(tmp_path / "r"),
                                             "LME_MANIFEST_PATH": str(tmp_path / "m.json")}, None)
    assert p["results"] == tmp_path / "r" and p["manifest"] == tmp_path / "m.json"


ROWS = [{"namespace": "lme-a", "turns": 4, "episodes": 4, "ready": 4, "failed_remaining": 0},
        {"namespace": "lme-b", "turns": 6, "episodes": 6, "ready": 5, "failed_remaining": 1}]


def test_snapshot_without_menhir_uses_manifest_counts():
    lines = check.snapshot(ROWS, ["lme-a", "lme-b", "lme-c"], 30, None, "t")
    assert lines[0].startswith("[t] items_done=2/30 started=3 turns_done=10 failed_remaining=1")
    assert "no Menhir serving" in lines[0]
    assert len(lines) == 3


class FakeOps:
    def __init__(self, states: dict[str, dict[str, int]], queue: int = 2) -> None:
        self.states, self.queue, self.calls = states, queue, []

    def __call__(self, op: str, body: dict | None = None):
        self.calls.append((op, body))
        if op == "get_queue_depth":
            return self.queue
        assert op == "fetch_namespace_build_counts" and body["sections"] == ["state"]
        return {"namespace": body["namespace"], "state": self.states[body["namespace"]]}


def test_snapshot_with_menhir_reads_done_and_in_flight_namespaces():
    st = {"pending": 0, "enriching": 0, "ready": 4, "failed": 0, "total": 4}
    ops = FakeOps({"lme-a": st, "lme-b": {**st, "ready": 5, "failed": 1, "total": 6},
                   "lme-c": {"pending": 2, "enriching": 1, "ready": 0, "failed": 0, "total": 3}})
    lines = check.snapshot(ROWS, ["lme-c"], 30, ops, "t")
    assert "live: queue=2 pending=2 enriching=1 ready=9 failed=1" in lines[0]
    assert [ln.split()[0] for ln in lines[1:]] == ["lme-a", "lme-b", "lme-c"]
    assert "in-flight" in lines[3] and "episodes=4/4" in lines[1]
    assert {b["namespace"] for op, b in ops.calls if op == "fetch_namespace_build_counts"} == {"lme-a", "lme-b", "lme-c"}


def test_started_namespaces_reads_the_state_log(tmp_path: Path):
    sys.path.insert(0, str(check.LIB))
    from ingest_state import IngestStateLog

    log_path = tmp_path / "ingest-state-manifest.jsonl"
    log = IngestStateLog(log_path)
    log.record_plan("lme-a", "d1", 3)
    log.record_plan("lme-b", "d2", 5)
    before = log_path.read_bytes()
    assert check.started_namespaces(log_path) == ["lme-a", "lme-b"]
    assert log_path.read_bytes() == before
    assert check.started_namespaces(tmp_path / "missing.jsonl") == []


def test_menhir_call_is_none_unless_a_menhir_answers():
    def down(request):
        raise httpx.ConnectError("refused")

    with httpx.Client(transport=httpx.MockTransport(down)) as client:
        assert check.menhir_call(client, "http://127.0.0.1:8102") is None

    def up(request):
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/api/internal/backend/get_queue_depth"
        return httpx.Response(200, json=7)

    with httpx.Client(transport=httpx.MockTransport(up)) as client:
        call = check.menhir_call(client, "http://127.0.0.1:8102")
        assert call is not None and call("get_queue_depth") == 7


def _paths(tmp_path: Path) -> dict[str, Path]:
    paths = check.check_paths(_config("smoke"), {}, str(tmp_path))
    paths["manifest"].write_text(json.dumps(ROWS), encoding="utf-8")
    return paths


def test_run_check_runs_canonical_acceptance(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    seen = {}
    monkeypatch.setattr(check, "menhir_call", lambda client, url: None)
    monkeypatch.setattr(check.subprocess, "run",
                        lambda cmd: seen.setdefault("cmd", cmd) and type("R", (), {"returncode": 1})())
    out: list[str] = []
    assert check.run_check(paths, limit=30, menhir_url="u", expected_items=500, python="py", out=out.append) == 1
    cmd = seen["cmd"]
    assert cmd[0] == "py" and cmd[1].endswith("validate_run.py")
    assert cmd[2:4] == [str(paths["provenance"]), str(paths["manifest"])]
    assert ["--require-fresh-clean", "--expected-items", "500"] == cmd[6:9]
    assert cmd[-2:] == ["--output", str(paths["report"])]
    assert out[0].startswith("[") and "items_done=2/30" in out[0]


def test_run_check_without_expected_items_runs_no_validator(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(check, "menhir_call", lambda client, url: None)
    monkeypatch.setattr(check.subprocess, "run", lambda cmd: pytest.fail("validator ran"))
    assert check.run_check(paths, limit=30, menhir_url="u", expected_items=None, out=lambda s: None) == 0


def test_watch_appends_summary_lines_to_the_progress_log(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(check, "menhir_call", lambda client, url: None)
    sleeps: list[float] = []
    assert check.run_check(paths, limit=30, menhir_url="u", expected_items=None, watch=5, sleep=sleeps.append,
                           out=lambda s: None, max_rounds=3) == 0
    logged = paths["progress_log"].read_text(encoding="utf-8").splitlines()
    assert len(logged) == 3 and all("items_done=2/30" in ln for ln in logged)
    assert sleeps == [5, 5]


@pytest.mark.parametrize("argv", [
    ["--expected-items", "5"], ["--watch", "5"], ["--menhir-url", "http://127.0.0.1:8102"],
    ["--check", "--retry-failed"], ["--check", "--resume"], ["--check", "--dry-run"],
    ["--check", "--watch", "0"], ["--check", "--watch", "5", "--expected-items", "5"],
    ["--check", "--expected-items", "0"],
])
def test_cli_check_flag_combinations_refused(argv, capsys):
    assert build_cli.main(["--build", DEFAULT_BUILD, *argv], environ={}) == 2


def test_cli_check_refuses_a_non_throwaway_menhir(capsys, monkeypatch):
    monkeypatch.delenv("ARCHOLITH_BENCH_ALLOW_PORTS", raising=False)
    monkeypatch.setattr(check, "run_check", lambda *a, **k: pytest.fail("checked a refused target"))
    argv = ["--build", DEFAULT_BUILD, "--check", "--menhir-url", "http://127.0.0.1:8193"]
    assert build_cli.main(argv, environ={}) == 2


def test_cli_check_never_launches_the_build(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(build_cli.subprocess, "run", lambda *a, **k: pytest.fail("build launched"))
    seen = {}
    monkeypatch.setattr(check, "run_check", lambda paths, **kw: seen.update(paths=paths, **kw) or 0)
    argv = ["--build", DEFAULT_BUILD, "--check", "--results-dir", str(tmp_path), "--expected-items", "30"]
    assert build_cli.main(argv, environ={}) == 0
    assert seen["menhir_url"] == "http://127.0.0.1:8102" and seen["limit"] == 30 and seen["expected_items"] == 30
    assert seen["paths"]["manifest"] == tmp_path / "manifest.json"
