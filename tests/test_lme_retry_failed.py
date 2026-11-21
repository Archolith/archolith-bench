"""build --retry-failed: the op-based reset/drain (lib/retry_failed.py) and the pipeline's retry mode."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from archolith_bench.build import __main__ as build_cli
from archolith_bench.build import pipeline

from .test_build_pipeline import Build, Call, FakeServe, make_frontier, make_settings

LIB = Path(__file__).resolve().parents[1] / "scripts" / "longmemeval" / "lib"
sys.path.insert(0, str(LIB))
_spec = importlib.util.spec_from_file_location("lme_retry_failed_test", LIB / "retry_failed.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


# ---------------------------------------------------------------------------
# lib/retry_failed.py over a fake Menhir
# ---------------------------------------------------------------------------

class FakeMenhir:
    """Episodes per namespace as {uuid: state}; reset moves FAILED to PENDING, each poll drains one step."""

    def __init__(self, episodes: dict[str, dict[str, str]], *, refuse: set[str] = frozenset(),
                 refail: set[str] = frozenset(), queue: int = 0) -> None:
        self.episodes = episodes
        self.refuse, self.refail = set(refuse), set(refail)
        self.queue = queue
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, op: str, body: dict | None = None):
        body = body or {}
        self.calls.append((op, body))
        if op == "list_episode_processing":
            assert body["states"] == ["FAILED"]
            ns = self.episodes.get(body["namespace"], {})
            return [{"uuid": u, "namespace": body["namespace"]} for u, st in ns.items() if st == "FAILED"]
        if op == "force_reset_failed_episode":
            uuid = body["episode_uuid"]
            if uuid in self.refuse:
                return False
            for ns in self.episodes.values():
                if uuid in ns:
                    # A re-failing episode fails again at once, before the namespace is re-listed.
                    ns[uuid] = "FAILED" if uuid in self.refail else "PENDING"
            return True
        if op == "get_queue_depth":
            depth, self.queue = self.queue, max(0, self.queue - 1)
            return depth
        if op == "fetch_namespace_build_counts":
            assert body["sections"] == ["state"]
            ns = self.episodes.get(body["namespace"], {})
            for u, st in ns.items():  # enrichment progresses one poll at a time
                if st == "PENDING":
                    ns[u] = "READY"
            counts = {k: 0 for k in ("pending", "ready", "enriching", "failed")}
            for st in ns.values():
                counts[st.lower()] += 1
            return {"namespace": body["namespace"], "state": counts}
        raise AssertionError(op)

    def ops(self, op: str) -> list[dict]:
        return [b for o, b in self.calls if o == op]


def _retry(menhir: FakeMenhir, namespaces: list[str], timeout: float = 100.0) -> int:
    clock = iter(range(10_000))
    return rf.retry(menhir, namespaces, timeout_s=timeout, poll_s=0, sleep=lambda s: None,
                    clock=lambda: next(clock))


def test_resets_every_failed_episode_once_and_drains_to_zero():
    m = FakeMenhir({"lme-a": {"a1": "FAILED", "a2": "READY"}, "lme-b": {"b1": "FAILED", "b2": "FAILED"}})
    assert _retry(m, ["lme-a", "lme-b"]) == 0
    assert sorted(b["episode_uuid"] for b in m.ops("force_reset_failed_episode")) == ["a1", "b1", "b2"]
    assert all(st == "READY" for ns in m.episodes.values() for st in ns.values())


def test_no_failed_episodes_skips_the_drain():
    m = FakeMenhir({"lme-a": {"a1": "READY"}})
    assert _retry(m, ["lme-a"]) == 0
    assert not m.ops("force_reset_failed_episode") and not m.ops("get_queue_depth")


def test_an_episode_that_fails_again_is_reset_once_and_fails_the_run():
    m = FakeMenhir({"lme-a": {"a1": "FAILED", "a2": "FAILED"}}, refail={"a1"})
    assert _retry(m, ["lme-a"]) == 1
    assert [b["episode_uuid"] for b in m.ops("force_reset_failed_episode")].count("a1") == 1


def test_a_refused_reset_leaves_the_episode_failed_and_fails_the_run():
    m = FakeMenhir({"lme-a": {"a1": "FAILED"}}, refuse={"a1"})
    assert _retry(m, ["lme-a"]) == 1


def test_drain_waits_for_the_queue_and_only_polls_touched_namespaces():
    m = FakeMenhir({"lme-a": {"a1": "FAILED"}, "lme-b": {"b1": "READY"}}, queue=4)
    assert _retry(m, ["lme-a", "lme-b"]) == 0
    assert len(m.ops("get_queue_depth")) >= 4 + 3
    drained = [b["namespace"] for b in m.ops("fetch_namespace_build_counts")][:-2]  # last two: residual check
    assert set(drained) == {"lme-a"}


def test_drain_timeout_fails_the_run():
    m = FakeMenhir({"lme-a": {"a1": "FAILED"}}, queue=10_000)
    assert _retry(m, ["lme-a"], timeout=20) == 1


# ---------------------------------------------------------------------------
# pipeline.retry_failed against fakes (no docker, no network)
# ---------------------------------------------------------------------------

def run_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, running: bool = True,
              refuse_check: bool = False, failing: str | None = None, **over: str) -> Build:
    frontier = make_frontier(tmp_path / "frontier")
    build = Build(make_settings(tmp_path, frontier, **over), frontier)

    def fake_run(cmd, env=None, cwd=None, **kwargs):
        build.calls.append(Call(list(cmd), env, kwargs))
        build.order.append("run|" + " ".join(str(c) for c in cmd))
        rc, out = 0, ""
        if "fingerprint" in cmd:
            out = "digest"
        elif cmd[:2] == ["git", "-C"] and "rev-parse" in cmd:
            out = "abc123"
        elif "check-retry" in cmd:
            build.record = Path(cmd[cmd.index("check-retry") + 2]).read_text(encoding="utf-8")
            rc = 1 if refuse_check else 0
        if failing and any(str(c).endswith(failing) for c in cmd):
            rc = 3
        return type("R", (), {"returncode": rc, "stdout": out})()

    def fake_popen(cmd, cwd=None, env=None, **kwargs):
        serve = FakeServe(list(cmd), cwd, env)
        build.serves.append(serve)
        build.calls.append(Call(list(cmd), env, {"cwd": cwd}))
        build.order.append("popen|" + " ".join(str(c) for c in cmd))
        return serve

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pipeline, "http_ok", lambda url: not url.endswith("/api/health") or bool(build.serves))
    monkeypatch.setattr(pipeline, "container_running", lambda name, env: running)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pipeline, "openai_key", lambda s, env: "sk-test")
    Path(build.settings["LME_RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)
    try:
        build.result = pipeline.retry_failed(build.settings, "1", {"PATH": "x"})
    except BaseException as exc:  # noqa: BLE001 - Die is the outcome under test
        build.error = exc
    return build


def test_retry_order_check_then_phase_then_serve_then_retry_then_phase_end(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch)
    assert b.error is None and b.result == 0
    steps = [
        b.index("run_provenance.py check-retry"),
        b.index("run_provenance.py phase-start", "--phase retry-failed"),
        b.index("retired_flags.py check"),
        b.index("frontier-menhir serve", kind="popen"),
        b.index("lib/retry_failed.py"),
        b.index("run_provenance.py phase-end", "--phase retry-failed"),
    ]
    assert steps == sorted(steps)
    assert not b.has("docker run") and not b.has("docker start") and not b.has("lib/ingest.py")
    assert not b.has("run_provenance.py begin")
    assert b.serves[0].terminated
    retry_cmd = b.call("lib/retry_failed.py").cmd
    assert retry_cmd[-4:] == [b.settings["LME_MANIFEST_PATH"], f"http://localhost:{b.settings['LME_PORT_BUILD']}",
                              "--drain-timeout", "86400"]


def test_retry_serves_with_the_build_env(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch, LME_RETRY_DRAIN_TIMEOUT="60")
    env = b.serves[0].env
    for name, _ in pipeline.MENHIR_ENV:
        assert name in env
    assert env["MENHIR_PERSONAL_MEMORY_CHAT_MODEL"] == "gpt-6-luna"
    assert env["MENHIR_OPENAI_SERVICE_TIER"] == "flex"
    assert b.call("lib/retry_failed.py").cmd[-1] == "60"


def test_retry_record_sent_to_check_retry(tmp_path, monkeypatch):
    import json

    b = run_retry(tmp_path, monkeypatch)
    record = json.loads(b.record)
    assert set(record) == set(pipeline.RETRY_RECORD_KEYS)
    assert record["container"] == "menhir-lme-test" and record["menhir_commit"] == "abc123"
    assert record["surface_digest"] == "digest" and record["anchored_time_resolver_enabled"] == 1
    assert not list(Path(b.settings["LME_RESULTS_DIR"]).glob(".graph-retry-*"))


def test_retry_refused_by_provenance_never_serves(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch, refuse_check=True)
    assert isinstance(b.error, pipeline.Die) and "refused this retry" in b.error.message
    assert not b.serves and not b.has("phase-start")


def test_retry_never_starts_a_stopped_graph(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch, running=False)
    assert isinstance(b.error, pipeline.Die) and "never creates or starts" in b.error.message
    assert not b.serves and not b.has("check-retry")


def test_failed_retry_leaves_the_phase_open_and_stops_serve(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch, failing="retry_failed.py")
    assert b.error is not None
    assert not b.has("phase-end")
    assert b.serves[0].terminated


def test_retry_noncanonical_passes_through(tmp_path, monkeypatch):
    b = run_retry(tmp_path, monkeypatch, LME_NONCANONICAL="1")
    assert "--noncanonical" in b.call("check-retry").cmd
    assert "noncanonical=1" in b.call("phase-start").cmd


def test_main_dispatches_retry_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "read_settings", lambda stream, env: {"LME_RETRY_FAILED": "1"})
    monkeypatch.setattr(pipeline, "retry_failed", lambda s, limit, base: seen.append("retry") or 0)
    monkeypatch.setattr(pipeline, "run", lambda s, limit, base: seen.append("build") or 0)
    assert pipeline.main(["1"]) == 0 and seen == ["retry"]


# ---------------------------------------------------------------------------
# build CLI
# ---------------------------------------------------------------------------

DEFAULT_BUILD = str(Path(__file__).resolve().parents[1] / "builds" / "default.yaml")


@pytest.mark.parametrize("extra", [["--resume"], ["--fresh"], ["--reset-file", "x"], ["--stop-after-items", "2"]])
def test_cli_retry_takes_no_build_run_control(extra, capsys):
    assert build_cli.main(["--build", DEFAULT_BUILD, "--retry-failed", *extra], environ={}) == 2
    assert "--retry-failed" in capsys.readouterr().err


def test_cli_drain_timeout_needs_retry(capsys):
    assert build_cli.main(["--build", DEFAULT_BUILD, "--drain-timeout", "5"], environ={}) == 2


def _launched_env(monkeypatch, tmp_path, argv, environ):
    seen = {}

    def fake_run(cmd, env=None, cwd=None):
        seen.update(env)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(build_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(build_cli.shutil, "which", lambda name: "bash")
    assert build_cli.main(["--build", DEFAULT_BUILD, "--results-dir", str(tmp_path), *argv], environ=environ) == 0
    return seen


def test_cli_retry_flags_reach_the_pipeline(monkeypatch, tmp_path):
    env = _launched_env(monkeypatch, tmp_path, ["--retry-failed", "--drain-timeout", "60"], {})
    assert env["LME_RETRY_FAILED"] == "1" and env["LME_RETRY_DRAIN_TIMEOUT"] == "60"


def test_cli_inherited_retry_export_does_not_turn_a_build_into_a_retry(monkeypatch, tmp_path):
    env = _launched_env(monkeypatch, tmp_path, [], {"LME_RETRY_FAILED": "1", "LME_RETRY_DRAIN_TIMEOUT": "5"})
    assert "LME_RETRY_FAILED" not in env and "LME_RETRY_DRAIN_TIMEOUT" not in env
