"""archolith_bench.build.pipeline: run order, refusals and the pure steps (no docker, no network)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from archolith_bench.build import pipeline

# ---------------------------------------------------------------------------
# A recorded run
# ---------------------------------------------------------------------------

BOLT, HTTP, PORT = "7789", "7776", "8202"


@contextlib.contextmanager
def dies(text: str):
    """A Die whose message holds ``text`` (str(Die) is only the exit status)."""
    with pytest.raises(pipeline.Die) as exc:
        yield
    assert text in exc.value.message, exc.value.message


class FakeServe:
    def __init__(self, cmd: list[str], cwd: str, env: dict[str, str]) -> None:
        self.cmd, self.cwd, self.env = cmd, cwd, env
        self.terminated = False
        self.killed = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


@dataclass
class Call:
    cmd: list[str]
    env: dict[str, str] | None
    kwargs: dict


@dataclass
class Build:
    settings: dict[str, str]
    frontier: Path
    calls: list[Call] = field(default_factory=list)
    serves: list[FakeServe] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    error: BaseException | None = None
    result: int | None = None

    def index(self, *needles: str, kind: str = "run") -> int:
        """Position in ``order`` of the first call whose command holds every needle."""
        for i, entry in enumerate(self.order):
            if entry.startswith(kind + "|") and all(n in entry for n in needles):
                return i
        raise AssertionError(f"no {kind} call with {needles}: {self.order}")

    def has(self, *needles: str, kind: str = "run") -> bool:
        try:
            self.index(*needles, kind=kind)
        except AssertionError:
            return False
        return True

    def call(self, *needles: str) -> Call:
        return self.calls[self.index(*needles)]


def make_settings(tmp_path: Path, frontier: Path, **over: str) -> dict[str, str]:
    results = tmp_path / "results" / "lme-test"
    s = {
        "BENCH_DIR": (tmp_path / "bench").as_posix(), "BENCH_PY": "bench-python",
        "MENHIR_MAIN": (tmp_path / "menhir").as_posix(), "MENHIR_MAIN_PY": "menhir-python",
        "MENHIR_FRONTIER": frontier.as_posix(), "MENHIR_FRONTIER_PY": "frontier-python",
        "MENHIR_FRONTIER_BIN": "frontier-menhir",
        "LONGMEMEVAL_VARIANT": "oracle", "LME_DATASET": "xiaowu0162/longmemeval", "LME_NS_PREFIX": "lme-",
        "LME_NEO4J_NAME": "menhir-lme-test", "LME_NEO4J_VOL": "lme-test-vol", "LME_NEO4J_PW": "pw",
        "LME_NEO4J_IMAGE": "neo4j:5-community", "LME_NEO4J_HEAP": "2g",
        "LME_BOLT": BOLT, "LME_HTTP": HTTP, "LME_PORT_BUILD": PORT,
        "LME_RESULTS_DIR": results.as_posix(), "LME_MANIFEST_PATH": (results / "manifest.json").as_posix(),
        "LME_EXTRACT_PROVIDER": "openai", "LME_EXTRACT_MODEL": "gpt-6-luna", "LME_EMBED_MODEL": "text-embedding-3-small",
        "LME_SEGMENTATION": "adaptive", "LME_REQUIRE_FRESH": "0", "LME_RESUME": "0", "LME_NONCANONICAL": "0",
        "LME_REQUIRE_TURN_EVIDENCE": "1", "LME_STORE_ONLY_CONTEXT": "1", "LME_ASSISTANT_STORE_ONLY": "0",
        "LME_EMBED_BACKFILL": "1", "LME_BACKFILL_DATES": "0",
        "LME_CONSOLIDATION_AUDIT_ENABLED": "1", "LME_RECALL_AUDIT_ENABLED": "1",
        "LME_MENHIR_LOG_LEVEL": "INFO",
        "LME_COUNTER_STATE_ENABLED": "0", "LME_EVENT_HISTORY_ENABLED": "0",
        "LME_SCALAR_STATE_ENABLED": "0", "LME_SCALAR_HISTORY_ENABLED": "0",
        "LME_SCALAR_CONSOLIDATION_K": "3", "LME_SCALAR_CALL_BUDGET": "50", "LME_SCALAR_THRESHOLD": "2/3",
        "LME_REQUIRE_SCALAR_OUTPUT": "1", "LME_INGEST_CONCURRENCY": "30", "LME_INGEST_STOP_AFTER_ITEMS": "0",
        "LME_CANONICAL_SELF_BINDING_MODE": "off", "LME_OPENAI_SERVICE_TIER": "flex",
        "LME_ANCHORED_TIME_RESOLVER": "1", "LME_ANCHORED_TIME_MODEL": "gpt-6-luna",
        "LME_ANCHORED_TIME_ATTACH_MODEL": "gpt-6-luna", "LME_MAX_LLM_CALLS_PER_JOB": "100",
    }
    s.update(over)
    return s


def make_frontier(root: Path, *, supported: bool = True) -> Path:
    if supported:
        routes = root / "src" / "menhir" / "api" / "routes_support.py"
        routes.parent.mkdir(parents=True)
        routes.write_text("store_only = True\n", encoding="utf-8")
        script = root / "scripts" / "backfill_episode_embeddings.py"
        script.parent.mkdir(parents=True)
        script.write_text("# --require-complete\n", encoding="utf-8")
    else:
        root.mkdir(parents=True, exist_ok=True)
    return root


def run_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, supported: bool = True,
              failing: str | None = None, serving_already: bool = False, limit: str = "1",
              **over: str) -> Build:
    """Run ``pipeline.run`` against fakes; ``failing`` is a script name whose run exits 3."""
    frontier = make_frontier(tmp_path / "frontier", supported=supported)
    build = Build(make_settings(tmp_path, frontier, **over), frontier)

    def fake_run(cmd, env=None, cwd=None, **kwargs):
        build.calls.append(Call(list(cmd), env, kwargs))
        build.order.append("run|" + " ".join(str(c) for c in cmd))
        rc, out = 0, ""
        if cmd[:3] == ["docker", "volume", "inspect"] or cmd[:3] == ["docker", "container", "inspect"]:
            rc = 1
        elif "fingerprint" in cmd:
            out = "digest"
        elif cmd[:2] == ["git", "-C"] and "rev-parse" in cmd:
            out = "abc123"
        if failing and any(str(c).endswith(failing) for c in cmd):
            rc = 3
        return type("R", (), {"returncode": rc, "stdout": out})()

    def fake_popen(cmd, cwd=None, env=None, **kwargs):
        serve = FakeServe(list(cmd), cwd, env)
        build.serves.append(serve)
        build.calls.append(Call(list(cmd), env, {"cwd": cwd}))
        build.order.append("popen|" + " ".join(str(c) for c in cmd))
        return serve

    def fake_http(url: str) -> bool:
        if url.endswith("/api/health"):
            return serving_already or bool(build.serves)
        return True

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pipeline, "http_ok", fake_http)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(pipeline, "openai_key", lambda s, env: "sk-test")
    try:
        build.result = pipeline.run(build.settings, limit, {"PATH": "x"})
    except BaseException as exc:  # noqa: BLE001 - Die and SystemExit are the outcomes under test
        build.error = exc
    return build


# ---------------------------------------------------------------------------
# Run order
# ---------------------------------------------------------------------------

def test_run_order_from_docker_to_phase_end(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch)
    assert b.error is None and b.result == 0
    s = b.settings
    steps = [
        b.index("docker volume inspect lme-test-vol"),
        b.index("docker container inspect menhir-lme-test"),
        b.index(f"docker run -d --name menhir-lme-test -p 127.0.0.1:{BOLT}:7687"),
        b.index("run_provenance.py begin"),
        b.index("bench_surface.py attach"),
        b.index("run_provenance.py phase-start"),
        b.index("retired_flags.py check"),
        b.index("frontier-menhir serve", kind="popen"),
        b.index("lib/ingest.py"),
        b.index("backfill_episode_embeddings.py"),
        b.index("lib/promote.py"),
        b.index("run_provenance.py phase-end"),
    ]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)

    run_cmd = b.call("docker run")
    assert f"127.0.0.1:{HTTP}:7474" in run_cmd.cmd
    serve = b.serves[0]
    assert serve.cmd == ["frontier-menhir", "serve", "--port", PORT]
    assert serve.cwd == s["MENHIR_FRONTIER"]
    # Stopped at the end, after the last step.
    assert serve.terminated and not serve.killed

    ingest = b.call("lib/ingest.py").cmd
    assert ["--menhir-url", f"http://localhost:{PORT}"] == ingest[ingest.index("--menhir-url"):][:2]


def test_phase_start_and_end_use_the_graph_provenance_path_and_no_direct_write(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch)
    provenance = (Path(b.settings["LME_RESULTS_DIR"]) / "graph-provenance-menhir-lme-test.json").as_posix()
    for verb in ("begin", "phase-start", "phase-end"):
        assert provenance in b.call(f"run_provenance.py {verb}").cmd
    assert b.index("run_provenance.py begin") < b.index("run_provenance.py phase-start") \
        < b.index("run_provenance.py phase-end")
    # Only run_provenance.py writes it; the pipeline leaves just the (removed) attempt record.
    assert not Path(provenance).exists()
    assert not (Path(b.settings["LME_RESULTS_DIR"]) / ".graph-attempt-menhir-lme-test.json").exists()


def test_retired_flags_check_runs_with_the_full_menhir_env(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch)
    check = b.call("retired_flags.py check")
    assert check.cmd[1] == "-I" and check.cmd[-1] == b.settings["MENHIR_FRONTIER"]
    assert {name for name, _ in pipeline.MENHIR_ENV} <= set(check.env)
    assert b.index("retired_flags.py check") < b.index("frontier-menhir serve", kind="popen")
    # The serve process gets the same environment.
    assert {name for name, _ in pipeline.MENHIR_ENV} <= set(b.serves[0].env)


def test_backfill_runs_the_serving_trees_script_against_this_builds_neo4j(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch)
    frontier = b.settings["MENHIR_FRONTIER"]
    backfill = b.call("backfill_episode_embeddings.py")
    assert backfill.cmd == ["frontier-python", f"{frontier}/scripts/backfill_episode_embeddings.py",
                            "--require-complete"]
    assert backfill.env["PYTHONPATH"] == f"{frontier}/src"
    assert backfill.env["NEO4J_URI"] == f"bolt://localhost:{BOLT}"
    assert b.index("lib/ingest.py") < b.index("backfill_episode_embeddings.py") < b.index("lib/promote.py")


def test_backfill_is_skipped_when_switched_off(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch, LME_EMBED_BACKFILL="0")
    assert b.error is None
    assert not b.has("backfill_episode_embeddings.py")
    assert b.has("lib/promote.py")


def test_a_serving_tree_refusal_never_starts_menhir(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch, supported=False)
    assert isinstance(b.error, pipeline.Die)
    assert "has no store_only ingest" in b.error.message
    assert b.serves == []
    assert not b.has("frontier-menhir serve", kind="popen")
    assert not b.has("retired_flags.py check") and not b.has("lib/ingest.py")


def test_something_already_serving_the_port_refuses_before_serve(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch, serving_already=True)
    assert isinstance(b.error, pipeline.Die) and "already serves" in b.error.message
    assert b.serves == []


def test_a_failed_ingest_stops_before_the_post_steps_and_still_stops_serve(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch, failing="ingest.py")
    assert isinstance(b.error, SystemExit) and b.error.code == 3
    assert not b.has("backfill_episode_embeddings.py")
    assert not b.has("lib/promote.py")
    assert not b.has("run_provenance.py phase-end")
    assert len(b.serves) == 1 and b.serves[0].terminated


def test_a_failed_retired_flag_check_refuses_before_serve(tmp_path, monkeypatch) -> None:
    b = run_build(tmp_path, monkeypatch, failing="retired_flags.py")
    assert isinstance(b.error, pipeline.Die) and "has retired" in b.error.message
    assert b.serves == []


# ---------------------------------------------------------------------------
# scalar_problem
# ---------------------------------------------------------------------------

def _row(**over):
    row = {"scalar_consolidated": True, "turn_evidence": 2, "scalar_llm_calls": 3,
           "typed_assertions": 1, "scalar_views": 1, "scalar_history_views": 1}
    row.update(over)
    return row


@pytest.mark.parametrize("rows, kwargs, message", [
    ([], {}, "empty manifest"),
    ([_row()], {"expected": 2}, "manifest has 1 rows, expected 2"),
    ([_row(scalar_consolidated=False)], {}, "at least one namespace was not consolidated"),
    ([_row(turn_evidence=0)], {}, "no TurnEvidence captured"),
    ([_row(scalar_llm_calls=0)], {}, "scalar perception made no LLM calls"),
    ([_row(typed_assertions=0)], {}, "no TypedAssertion materialized"),
    ([_row(scalar_views=0)], {}, "no scalar_state View materialized"),
    ([_row(scalar_history_views=0)], {}, "no scalar_history View materialized"),
])
def test_scalar_problem_messages(rows, kwargs, message) -> None:
    args = {"expected": 1, "require_output": True, "history": True, **kwargs}
    problem = pipeline.scalar_problem(rows, **args)
    assert problem == f"scalar ingest validation failed: {message}"


def test_scalar_problem_passes_and_output_checks_are_optional() -> None:
    assert pipeline.scalar_problem([_row()], expected=1, require_output=True, history=True) is None
    thin = _row(typed_assertions=0, scalar_views=0, scalar_history_views=0)
    assert pipeline.scalar_problem([thin], expected=1, require_output=False, history=True) is None
    thin = _row(scalar_history_views=0)
    assert pipeline.scalar_problem([thin], expected=1, require_output=True, history=False) is None


# ---------------------------------------------------------------------------
# check_fixture
# ---------------------------------------------------------------------------

def _fixture(tmp_path: Path, items) -> tuple[Path, str]:
    path = tmp_path / "fixture.json"
    raw = json.dumps(items).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_check_fixture_without_a_path_changes_nothing() -> None:
    s = {"LME_FIXTURE_PATH": "", "LIMIT": "1"}
    pipeline.check_fixture(s)
    assert s == {"LME_FIXTURE_PATH": "", "LIMIT": "1"}


def test_check_fixture_refusals(tmp_path) -> None:
    with dies("fixture not found"):
        pipeline.check_fixture({"LME_FIXTURE_PATH": (tmp_path / "none.json").as_posix(), "LIMIT": "1"})
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with dies("invalid fixture"):
        pipeline.check_fixture({"LME_FIXTURE_PATH": bad.as_posix(), "LIMIT": "1"})
    path, sha = _fixture(tmp_path, [{"a": 1}, {"a": 2}])
    with dies("must be a list with at least the requested item count"):
        pipeline.check_fixture({"LME_FIXTURE_PATH": path.as_posix(), "LIMIT": "3"})
    with dies("LME_FIXTURE_SHA256 does not match fixture bytes"):
        pipeline.check_fixture({"LME_FIXTURE_PATH": path.as_posix(), "LIMIT": "2", "LME_FIXTURE_SHA256": "0" * 64})
    with dies("LME_FIXTURE_COUNT does not match fixture items"):
        pipeline.check_fixture({"LME_FIXTURE_PATH": path.as_posix(), "LIMIT": "2", "LME_FIXTURE_SHA256": sha,
                                "LME_FIXTURE_COUNT": "5"})


def test_check_fixture_sets_the_measured_sha_and_count(tmp_path) -> None:
    path, sha = _fixture(tmp_path, [{"a": 1}, {"a": 2}])
    s = {"LME_FIXTURE_PATH": path.as_posix(), "LIMIT": "2"}
    pipeline.check_fixture(s)
    assert (s["LME_FIXTURE_SHA256"], s["LME_FIXTURE_COUNT"]) == (sha, "2")
    pinned = {"LME_FIXTURE_PATH": path.as_posix(), "LIMIT": "1", "LME_FIXTURE_SHA256": sha,
              "LME_FIXTURE_COUNT": "2"}
    pipeline.check_fixture(pinned)
    assert pinned["LME_FIXTURE_COUNT"] == "2"


# ---------------------------------------------------------------------------
# expand and read_settings
# ---------------------------------------------------------------------------

def test_expand_references_defaults_and_nesting() -> None:
    s = {"A": "a", "B": "b", "EMPTY": ""}
    assert pipeline.expand("x-${A}-y", s) == "x-a-y"
    assert pipeline.expand("${UNSET:-d}", s) == "d"
    assert pipeline.expand("${EMPTY:-d}", s) == "d"
    assert pipeline.expand("${A:-d}", s) == "a"
    assert pipeline.expand("${UNSET:-${B}}", s) == "b"
    assert pipeline.expand("${UNSET:-}", s) == ""
    assert pipeline.expand("${EMPTY}", s) == ""
    assert pipeline.expand("http://localhost:${A}", s) == "http://localhost:a"


def test_expand_an_unbound_name_is_a_refusal() -> None:
    with pytest.raises(pipeline.Die) as exc:
        pipeline.expand("${MISSING}", {})
    assert exc.value.message == "MISSING: unbound variable" and exc.value.code == 1


def test_read_settings_pairs_overlay_the_environment() -> None:
    data = b"A\0one\0B\0\0"
    s = pipeline.read_settings(io.BytesIO(data), {"A": "env", "C": "kept"})
    assert s == {"A": "one", "B": "", "C": "kept"}


@pytest.mark.skipif(os.name != "nt", reason="MSYS drive paths are converted on Windows only")
def test_read_settings_converts_msys_drive_paths_on_windows() -> None:
    data = b"P\0/c/Users/me/x\0Q\0/d\0R\0relative/c/x\0"
    s = pipeline.read_settings(io.BytesIO(data), {})
    assert s == {"P": "C:/Users/me/x", "Q": "D:/", "R": "relative/c/x"}


@pytest.mark.skipif(os.name == "nt", reason="values are left alone off Windows")
def test_read_settings_leaves_paths_alone_off_windows() -> None:
    assert pipeline.read_settings(io.BytesIO(b"P\0/c/Users/me\0"), {})["P"] == "/c/Users/me"


@pytest.mark.parametrize("data", [b"", b"A\0one", b"A\0", b"A\0one\0B\0"])
def test_read_settings_refuses_empty_or_malformed_input(data: bytes) -> None:
    with dies("no settings on stdin"):
        pipeline.read_settings(io.BytesIO(data), {})
