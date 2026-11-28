"""Offline tests for the sprint v2 harness (rubric `sprint-v2`): verdicts, chains, cost guard."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sqlite3
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from archolith_bench.beacon_eval.runner import Budget
from archolith_bench.core.opencode.process import EventLog
from archolith_bench.engineering_evidence.campaign import (
    Campaign,
    ProgressFeed,
    RunDirInUse,
    _outcome_from_row,
    chain_namespace,
    prepare_run_dir,
)
from archolith_bench.engineering_evidence.live_executor import LiveSprintConfig, OpenCodeSprintExecutor
from archolith_bench.engineering_evidence.server_usage import (
    NON_EPISODE_KEY,
    ServerRates,
    attribute_server_usage,
    backend_post,
    read_usage_rows,
    wait_for_enrichment_drain,
)
from archolith_bench.engineering_evidence.sprint import (
    NOTES_FILE,
    V2_ARMS,
    DryRunExecutor,
    SessionExecution,
    SprintFixture,
    build_session_prompt,
    compose_full_prompt,
    diff_snapshots,
    is_notes_path,
    score_execution,
    snapshot_tree,
)
from archolith_bench.engineering_evidence.sprint_rubric import (
    ChainView,
    accuracy_verdict,
    build_verdict_report,
    efficiency_verdict,
    ranges_overlap,
)
from archolith_bench.staged.rundir import RunConfigMismatch

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# -- verdict functions ---------------------------------------------------------------------


def test_accuracy_verdicts() -> None:
    assert accuracy_verdict([0.80, 0.82], [0.70, 0.75]) == "WIN"
    assert accuracy_verdict([0.80, 0.70], [0.70, 0.75]) == "DIRECTIONAL"  # mean +0.025 passes, ranges overlap
    assert accuracy_verdict([0.80, 0.72], [0.70, 0.74]) == "DIRECTIONAL"  # mean +0.04, ranges overlap
    assert accuracy_verdict([0.60, 0.62], [0.70, 0.75]) == "LOSS"
    assert accuracy_verdict([0.71, 0.72], [0.70, 0.73]) == "NO_DIFFERENCE"
    assert accuracy_verdict([], [0.7]) == "INSUFFICIENT_DATA"


def test_efficiency_verdicts() -> None:
    assert efficiency_verdict([80, 85], [100, 110]) == "WIN"
    assert efficiency_verdict([80, 120], [100, 110]) == "NO_DIFFERENCE"  # mean -4.8%
    assert efficiency_verdict([50, 105], [100, 110]) == "DIRECTIONAL"  # mean -27.5%, ranges overlap
    assert efficiency_verdict([95, 100], [100, 105]) == "NO_DIFFERENCE"
    assert efficiency_verdict([115, 120], [100, 105]) == "COST"
    assert efficiency_verdict([1], [0]) == "INSUFFICIENT_DATA"


def test_ranges_overlap() -> None:
    assert ranges_overlap([1, 3], [2, 4])
    assert ranges_overlap([1, 2], [2, 3])  # touching counts as overlap
    assert not ranges_overlap([1, 2], [3, 4])
    assert not ranges_overlap([5, 6], [1, 4])


def test_combined_verdict_has_no_or_rule() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    campaign_rows = _run_campaign_dry(fixture, repeats=2)
    verdict = campaign_rows["analysis"]["verdicts"]["primary_vs_notes_file"]["combined_verdict"]
    assert verdict.startswith("ACCURACY_") and "__EFFICIENCY_" in verdict


# -- helpers for campaigns -----------------------------------------------------------------


def _run_campaign_dry(fixture, *, repeats: int, executor=None, budget=None, tmp: Path | None = None, arms=V2_ARMS,
                      parallel: int = 2, rates=None) -> dict[str, Any]:
    import tempfile

    root = tmp or Path(tempfile.mkdtemp())
    rd = prepare_run_dir(root, "run", {"x": 1}, resume=False)
    camp = Campaign(
        fixture, executor=executor or DryRunExecutor(), run_dir=rd, run_id="t", arms=arms, repeats=repeats,
        parallel=parallel, budget=budget or Budget(None, None), rates=rates, dry_run=True,
        progress=ProgressFeed("x", "x", 1, ceiling_usd=None, script=Path("/nonexistent/progress.py")),
    )
    chains, stop = camp.run()
    return camp.build_report(chains, stop, provenance={}, structure_index=None)


class RecordingExecutor(DryRunExecutor):
    def __init__(self, cost: float = 0.0) -> None:
        self.cost = cost
        self.calls: list[tuple[str, int | None, int, str]] = []
        self._lock = threading.Lock()

    def run_session(self, **kw):
        res = super().run_session(**kw)
        res.cost_usd = self.cost
        with self._lock:
            self.calls.append((kw["arm"], kw.get("repeat"), kw["session_index"], kw["namespace"]))
        return res


# -- repeats: separate checkouts and namespaces --------------------------------------------


def test_repeats_get_separate_namespaces_and_chains() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    rec = RecordingExecutor()
    report = _run_campaign_dry(fixture, repeats=2, executor=rec)
    by_chain: dict[tuple[str, int | None], set[str]] = {}
    for arm, repeat, _idx, ns in rec.calls:
        by_chain.setdefault((arm, repeat), set()).add(ns)
    assert set(by_chain) == {(a, r) for a in V2_ARMS for r in (1, 2)}
    assert all(len(v) == 1 for v in by_chain.values())  # one namespace per chain
    all_ns = [next(iter(v)) for v in by_chain.values()]
    assert len(set(all_ns)) == len(all_ns)
    assert chain_namespace(fixture, "menhir_longitudinal", 1, "t") != chain_namespace(fixture, "menhir_longitudinal", 2, "t")
    assert len(report["chains"]) == 6


def test_namespace_does_not_resemble_the_checkout_dir() -> None:
    # Run v2-20261127: Menhir-arm agents spliced the look-alike namespace into checkout paths.
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ns = chain_namespace(fixture, "menhir_longitudinal", 2, "v2-20261127")
    assert ns.startswith("sv2-") and len(ns) == 14
    assert fixture.name not in ns and "menhir" not in ns and "-r2" not in ns


def _executor(tmp_path: Path, monkeypatch, *, stream_fn, post_fn=None, **cfg):
    monkeypatch.setattr(
        "archolith_bench.core.opencode.isolation.default_data_source", lambda: tmp_path / "empty_opencode_data"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    src_cfg = tmp_path / "opencode.json"
    src_cfg.write_text(json.dumps({"provider": {"openai": {"options": {}}}}), encoding="utf-8")
    exports: list[Path] = []

    def fake_export(pin, dest: Path, cache: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        f = dest / "llm" / "default_plugins" / "openai_models.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# GPT-5.5 — routes through the Responses API by default; pass\n", encoding="utf-8")
        exports.append(dest)
        return dest

    ex = OpenCodeSprintExecutor(
        LiveSprintConfig(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache", source_config=src_cfg, **cfg),
        stream_fn=stream_fn,
        export_fn=fake_export,
        resolve_cmd_fn=lambda: ["opencode"],
        check_memory_fn=lambda url: None,
        notify_file_dirty_fn=lambda *a, **k: {},
        post_fn=post_fn or (lambda *a, **k: []),
        sleep_fn=lambda s: None,
    )
    return ex, exports


def _stream(text: str = "ok", write: dict[str, str] | None = None):
    def fake(cmd, prompt, cwd, env, run_dir: Path, reserve, timeout_s):
        for rel, body in (write or {}).items():
            (Path(cwd) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(cwd) / rel).write_text(body, encoding="utf-8")
        log = EventLog()
        log.texts.append(text)
        log.input_tokens, log.output_tokens, log.reasoning_tokens = 100, 60, 20
        return log, "", 0

    return fake


def test_executor_gives_each_repeat_its_own_checkout(tmp_path: Path, monkeypatch) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, exports = _executor(tmp_path, monkeypatch, stream_fn=_stream())
    for repeat in (1, 2):
        ex.run_session(fixture=fixture, session=fixture.sessions[0], arm="notes_file", session_index=0,
                       namespace=f"ns{repeat}", repeat=repeat)
    ex.run_session(fixture=fixture, session=fixture.sessions[1], arm="notes_file", session_index=1,
                   namespace="ns1", repeat=1)
    assert len(exports) == 2 and len(set(exports)) == 2  # repeat 1 reused, repeat 2 separate
    # Short, non-repeating names (agents mistyped the long ones); checkouts.json maps them back.
    assert all(e.name.startswith("c") and len(e.name) == 7 and fixture.name not in str(e) for e in exports)
    mapping = json.loads((exports[0].parent / "checkouts.json").read_text(encoding="utf-8"))
    assert mapping == {"notes_file/r1": str(exports[0]), "notes_file/r2": str(exports[1])}


def test_executor_counts_refused_reads_from_stderr(tmp_path: Path, monkeypatch) -> None:
    def stream(cmd, prompt, cwd, env, run_dir, reserve, timeout_s):
        (Path(run_dir) / "stderr.log").write_text(
            "! permission requested: external_directory (C:\\x\\*); auto-rejecting\n" * 2, encoding="utf-8"
        )
        return _stream()(cmd, prompt, cwd, env, run_dir, reserve, timeout_s)

    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, _ = _executor(tmp_path, monkeypatch, stream_fn=stream)
    res = ex.run_session(fixture=fixture, session=fixture.sessions[0], arm="no_menhir", session_index=0,
                         namespace="n", repeat=1)
    assert res.refused_reads == 2


def test_preflight_cap_covers_estimate_margin_and_in_flight_reserve() -> None:
    spec = importlib.util.spec_from_file_location("sprint_cli", ROOT / "scripts" / "run_longitudinal_sprint_bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # Run v2r2-20261127: 45 sessions, 3 in flight, $0.05 reserve; a $0.37 cap could not finish.
    need = cli.preflight_cap_usd(sessions=45, est_usd_per_session=0.0075, parallel=3, reserve_usd=0.05)
    assert need == 0.57 and need > 0.37
    assert cli.preflight_cap_usd(sessions=45, est_usd_per_session=None, parallel=3, reserve_usd=0.05) is None


def test_executor_roots_opencode_in_the_checkout_not_an_inherited_pwd(tmp_path: Path, monkeypatch) -> None:
    # Git Bash exports PWD; OpenCode roots its project there, so a stale PWD sent the 2026-11-27
    # smoke agent into the bench repo instead of the llm checkout.
    monkeypatch.setenv("PWD", str(tmp_path / "caller-repo"))
    seen: dict[str, str] = {}

    def stream(cmd, prompt, cwd, env, run_dir, reserve, timeout_s):
        seen["cwd"], seen["pwd"] = str(cwd), env.get("PWD", "")
        return _stream()(cmd, prompt, cwd, env, run_dir, reserve, timeout_s)

    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, exports = _executor(tmp_path, monkeypatch, stream_fn=stream)
    ex.run_session(fixture=fixture, session=fixture.sessions[0], arm="no_menhir", session_index=0,
                   namespace="n", repeat=1)
    assert seen["pwd"] == seen["cwd"] == str(exports[0])


def _scan_post(responses: list, fingerprints: list):
    calls: list[tuple[str, dict]] = []

    def post(url, op, body, **kw):
        calls.append((op, body))
        if op == "scan_and_write_project":
            return responses.pop(0)
        if op == "get_scan_fingerprint":
            return fingerprints.pop(0) if fingerprints else "fp"
        return []

    return post, calls


def test_index_pristine_answers_identity_decision_and_waits_for_the_write(tmp_path: Path, monkeypatch) -> None:
    needs = {"status": "needs_decision", "reason": "directory_not_bound_no_candidate", "candidates": []}
    ok = {"skipped": False, "background": True, "meta": {"files": 88, "symbols": 1622, "imports": 128}}
    post, calls = _scan_post([needs, ok], [None, None, "fp"])
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, _ = _executor(tmp_path, monkeypatch, stream_fn=_stream(), post_fn=post, menhir_mcp_url="http://x/mcp-http")
    out = ex.index_pristine(fixture, "t")
    scans = [b for op, b in calls if op == "scan_and_write_project"]
    assert [b["identity_action"] for b in scans] == [None, "new"]
    assert sum(op == "get_scan_fingerprint" for op, _ in calls) == 3
    assert out["counts"] == {"files": 88, "imports": 128, "symbols": 1622} and out["identity_action"] == "new"


def test_index_pristine_fails_closed_when_nothing_is_scanned(tmp_path: Path, monkeypatch) -> None:
    import pytest

    needs = {"status": "needs_decision", "reason": "x", "candidates": [{"project_id": "p"}]}
    post, _ = _scan_post([needs], [])
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, _ = _executor(tmp_path, monkeypatch, stream_fn=_stream(), post_fn=post, menhir_mcp_url="http://x/mcp-http")
    with pytest.raises(RuntimeError, match="not written"):
        ex.index_pristine(fixture, "t")


def test_executor_splits_reasoning_from_output(tmp_path: Path, monkeypatch) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ex, _ = _executor(tmp_path, monkeypatch, stream_fn=_stream())
    res = ex.run_session(fixture=fixture, session=fixture.sessions[0], arm="no_menhir", session_index=0,
                         namespace="n", repeat=1)
    assert (res.output_tokens, res.reasoning_tokens) == (40, 20)
    assert res.uncached_tokens == 100 + 40 + 20


# -- notes_file arm ------------------------------------------------------------------------


def test_notes_arm_prompt_has_exception_and_other_arms_do_not() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    s = fixture.sessions[1]
    notes = compose_full_prompt(s, "notes_file", "ns")
    assert NOTES_FILE in notes and "only memory you keep between sessions" in notes
    assert s.notes_instruction.strip() in notes
    assert NOTES_FILE not in compose_full_prompt(s, "no_menhir", "ns")
    assert NOTES_FILE not in build_session_prompt(s, "menhir_longitudinal")


def test_notes_write_is_allowed_but_any_other_write_is_a_violation(tmp_path: Path, monkeypatch) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    ok, _ = _executor(tmp_path / "a", monkeypatch, stream_fn=_stream(write={NOTES_FILE: "notes"}))
    res = ok.run_session(fixture=fixture, session=fixture.sessions[0], arm="notes_file", session_index=0,
                         namespace="n", repeat=1)
    assert res.notes_violations == [] and "notes_file_violation" not in res.error

    bad, _ = _executor(
        tmp_path / "b", monkeypatch,
        stream_fn=_stream(write={NOTES_FILE: "n", "stray.txt": "x", "llm/default_plugins/openai_models.py": "changed"}),
    )
    res = bad.run_session(fixture=fixture, session=fixture.sessions[0], arm="notes_file", session_index=0,
                          namespace="n", repeat=1)
    assert sorted(res.notes_violations) == ["added:stray.txt", "modified:llm/default_plugins/openai_models.py"]
    assert "notes_file_violation=" in res.error


def test_snapshot_diff_ignores_notes_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1")
    before = snapshot_tree(tmp_path)
    (tmp_path / NOTES_FILE).write_text("n")
    (tmp_path / "b.txt").write_text("2")
    (tmp_path / "a.txt").write_text("3")
    assert diff_snapshots(before, snapshot_tree(tmp_path)) == ["added:b.txt", "modified:a.txt"]


def test_notes_file_is_stripped_from_scoring(tmp_path: Path) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    session = fixture.sessions[0]
    base = DryRunExecutor().run_session(fixture=fixture, session=session, arm="no_menhir", session_index=0, namespace="n")
    clean, clean_overall = score_execution(replace(base, repo_root=tmp_path), session, "notes_file")
    payload = json.loads(base.response_text.split("```json\n", 1)[1].rsplit("\n```", 1)[0])
    payload["docs"] = [*payload["docs"], NOTES_FILE]
    payload["files"] = [*payload["files"], f"./{NOTES_FILE}"]
    payload["citations"] = [{"path": NOTES_FILE, "line_start": 1, "line_end": 2}]
    text = f"```json\n{json.dumps(payload)}\n```"
    scored, overall = score_execution(SessionExecution(response_text=text, repo_root=tmp_path), session, "notes_file")
    assert (scored, overall) == (clean, clean_overall)
    # Other arms are not stripped: the extra file costs precision there.
    other, _ = score_execution(SessionExecution(response_text=text, repo_root=tmp_path), session, "no_menhir")
    assert other["file_precision"] < clean["file_precision"]
    assert is_notes_path("`./SPRINT_NOTES.md`") and not is_notes_path("docs/NOTES.md")


# -- token math ----------------------------------------------------------------------------


def test_uncached_token_math() -> None:
    r = SessionExecution(response_text="", input_tokens=100, output_tokens=50, reasoning_tokens=20,
                         cache_read_tokens=1000, cache_write_tokens=10)
    assert r.uncached_tokens == 180
    assert r.total_tokens == 1180


def test_server_cost_rates() -> None:
    rates = ServerRates.parse("1.0, 0.1, 4.0")
    assert rates.cost_usd(input_tokens=1_000_000, cached_input_tokens=500_000, output_tokens=250_000) == pytest.approx(
        0.5 * 1.0 + 0.5 * 0.1 + 0.25 * 4.0
    )
    with pytest.raises(ValueError):
        ServerRates.parse("1,2")


# -- cost guard ----------------------------------------------------------------------------


def test_cost_guard_stops_scheduling_and_marks_not_run(tmp_path: Path) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    assert len(fixture.sessions) == 5
    rec = RecordingExecutor(cost=0.10)
    budget = Budget(None, None, cap_usd=0.26, reserve_usd=0.05)
    report = _run_campaign_dry(fixture, repeats=1, executor=rec, budget=budget, tmp=tmp_path, arms=("no_menhir",),
                               parallel=1)
    sessions = report["chains"][0]["sessions"]
    assert [s["not_run"] for s in sessions] == [False, False, False, True, True]
    assert all(s["overall_score"] == 0.0 and s["scores"] == {} for s in sessions if s["not_run"])
    assert len(rec.calls) == 3  # sessions 4 and 5 never started
    assert report["campaign"]["stopped_early"] is True and "cap" in report["campaign"]["stop_reason"]
    assert report["campaign"]["sessions_not_run"] == 2
    row = report["analysis"]["per_chain"]["no_menhir"][0]
    assert row["complete"] is False and row["sessions_not_run"] == 2
    # not_run is not averaged in as a 0: the accuracy is over the three sessions that ran.
    assert row["accuracy"] == pytest.approx(1.0)


def test_no_session_starts_when_spend_is_at_the_cap(tmp_path: Path) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    rec = RecordingExecutor(cost=0.25)
    budget = Budget(None, None, cap_usd=0.25, reserve_usd=0.0)
    _run_campaign_dry(fixture, repeats=1, executor=rec, budget=budget, tmp=tmp_path, arms=("no_menhir",), parallel=1)
    assert len(rec.calls) == 1


def test_incomplete_chains_are_left_out_of_verdicts(tmp_path: Path) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    rec = RecordingExecutor(cost=0.10)
    budget = Budget(None, None, cap_usd=0.26, reserve_usd=0.05)
    report = _run_campaign_dry(fixture, repeats=1, executor=rec, budget=budget, tmp=tmp_path, parallel=1)
    v = report["analysis"]["verdicts"]["primary_vs_notes_file"]
    assert v["chains_used"] == {"menhir": 0, "baseline": 0}
    assert v["chains_excluded_incomplete"] == {"menhir": 1, "baseline": 1}
    assert v["accuracy"]["verdict"] == "INSUFFICIENT_DATA"


# -- output dir refusal and resume ---------------------------------------------------------


def test_refuses_existing_output_dir(tmp_path: Path) -> None:
    prepare_run_dir(tmp_path, "r1", {"a": 1}, resume=False)
    with pytest.raises(RunDirInUse):
        prepare_run_dir(tmp_path, "r1", {"a": 1}, resume=False)
    prepare_run_dir(tmp_path, "r1", {"a": 1}, resume=True)  # same config resumes
    with pytest.raises(RunConfigMismatch):
        prepare_run_dir(tmp_path, "r1", {"a": 2}, resume=True)


def test_cli_dry_run_v2_writes_report_and_refuses_rerun(tmp_path: Path, capsys) -> None:
    script = _load_script("run_longitudinal_sprint_bench")
    argv = ["--dry-run", "--repeats", "2", "--parallel", "2", "--out-root", str(tmp_path), "--run-id", "t1"]
    assert script.main(argv) == 0
    run_dir = next(tmp_path.iterdir())
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["rubric_version"] == "sprint-v2" and report["provenance"]["fixture_sha256"]
    assert {"report.json", "prompts.json", "sessions.jsonl"} <= {p.name for p in run_dir.iterdir()}
    before = (run_dir / "report.json").read_bytes()
    assert script.main(argv) == 2
    assert (run_dir / "report.json").read_bytes() == before


# -- enrichment wait -----------------------------------------------------------------------


def test_enrichment_wait_drains_then_stops() -> None:
    pending = iter([3, 2, 0])
    calls: list[dict] = []
    sleeps: list[float] = []
    clock = {"t": 0.0}

    def post(url, op, body, **kw):
        calls.append({"op": op, **body})
        return [{}] * next(pending)

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    out = wait_for_enrichment_drain(post, "http://x/mcp-http", "ns", timeout_s=180, poll_s=3,
                                    sleep_fn=sleep, clock_fn=lambda: clock["t"])
    assert out["drained"] is True and out["polls"] == 3 and out["waited_s"] == 6
    assert calls[0] == {"op": "list_episode_processing", "states": ["PENDING", "ENRICHING"], "limit": 200,
                        "namespace": "ns"}


def test_enrichment_wait_times_out_at_180s() -> None:
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    out = wait_for_enrichment_drain(lambda *a, **k: [{}], "http://x", "ns", timeout_s=180, poll_s=3,
                                    sleep_fn=sleep, clock_fn=lambda: clock["t"])
    assert out["drained"] is False and out["pending"] == 1 and out["waited_s"] == 180


def test_enrichment_wait_against_a_fake_http_endpoint() -> None:
    state = {"left": 2, "seen": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["seen"].append((self.path, body, self.headers.get("Authorization")))
            rows = [{"uuid": "e"}] * state["left"]
            state["left"] = max(state["left"] - 1, 0)
            data = json.dumps(rows).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):  # noqa: D401
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/mcp-http"
        out = wait_for_enrichment_drain(
            lambda u, op, body, **kw: backend_post(u, op, body, api_key="k"), url, "ns",
            timeout_s=5, poll_s=0.01,
        )
    finally:
        server.shutdown()
    assert out["drained"] is True
    assert state["seen"][0][0] == "/api/internal/backend/list_episode_processing"
    assert state["seen"][0][2] == "Bearer k"


def test_executor_waits_for_enrichment_before_later_menhir_sessions(tmp_path: Path, monkeypatch) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    polls = iter([2, 0])
    ex, _ = _executor(tmp_path, monkeypatch, stream_fn=_stream(), menhir_mcp_url="http://127.0.0.1:1/mcp-http",
                      post_fn=lambda u, op, body, **kw: [{}] * next(polls))
    first = ex.run_session(fixture=fixture, session=fixture.sessions[0], arm="menhir_longitudinal",
                           session_index=0, namespace="n", repeat=1)
    assert first.enrichment_wait_s == 0.0  # no wait before S1
    second = ex.run_session(fixture=fixture, session=fixture.sessions[1], arm="menhir_longitudinal",
                            session_index=1, namespace="n", repeat=1)
    assert second.enrichment_drained is True


# -- server usage attribution --------------------------------------------------------------


def _usage_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE llm_usage_events (call_id TEXT, recorded_at TEXT, episode_uuid TEXT, kind TEXT, "
        "status TEXT, input_tokens INT, output_tokens INT, total_tokens INT, cached_input_tokens INT, "
        "reasoning_output_tokens INT, cache_write_input_tokens INT)"
    )
    rows = [
        ("1", "2026-01-01T00:00:00", "ep-a", "chat", "ok", 1000, 100, 1100, 400, 10, 0),
        ("2", "2026-01-01T00:00:01", "ep-b", "chat", "ok", 2000, 200, 2200, 0, 0, 0),
        ("3", "2026-01-01T00:00:02", NON_EPISODE_KEY, "chat", "ok", 400, 40, 440, 0, 0, 0),
        ("4", "2026-01-01T00:00:03", "ep-a", "embedding", "ok", 500, 0, 500, 0, 0, 0),
        ("5", "2026-01-02T00:00:00", "ep-a", "chat", "failed", 10, 0, 10, 0, 0, 0),
    ]
    conn.executemany("INSERT INTO llm_usage_events VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_server_usage_attribution_from_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _usage_db(db)
    rows = read_usage_rows(db, since_iso="2026-01-01T00:00:00")
    assert len(rows) == 5 and len(read_usage_rows(db, since_iso="2026-01-02")) == 1
    rates = ServerRates(1.0, 0.5, 2.0)
    out = attribute_server_usage(rows, {"nsA": ["ep-a"], "nsB": ["ep-b"]}, rates=rates, namespaces=["nsA", "nsB"])
    a, b = out["by_namespace"]["nsA"], out["by_namespace"]["nsB"]
    assert out["attribution"] == "namespace"
    assert a["embedding_calls"] == 1 and a["calls"] == 2  # chat ok + chat failed; embedding counted separately
    # nsA direct: (1000-400)+100 = 700 uncached, +10 failed row; plus half of the 440 unattributed pool.
    assert a["attributed_uncached_tokens"] == 700 + 10
    assert a["unattributed_share_uncached_tokens"] == 440 // 2
    assert b["attributed_uncached_tokens"] == 2200
    assert out["unattributed"]["input_tokens"] == 400
    assert a["cost_usd"] > 0 and b["cost_usd"] > 0


def test_server_usage_falls_back_to_even_split(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    _usage_db(db)
    out = attribute_server_usage(read_usage_rows(db), None, rates=None, namespaces=["nsA", "nsB"])
    assert out["attribution"] == "even_split"
    assert out["by_namespace"]["nsA"]["uncached_tokens"] == out["by_namespace"]["nsB"]["uncached_tokens"]


def test_server_usage_counts_in_the_primary_efficiency_metric() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    rec = RecordingExecutor()
    report = _run_campaign_dry(fixture, repeats=1, executor=rec, arms=("menhir_longitudinal",))
    base = report["analysis"]["per_chain"]["menhir_longitudinal"][0]
    outcomes = tuple(_outcome_from_row(s) for s in report["chains"][0]["sessions"])
    row = build_verdict_report(
        [ChainView("menhir_longitudinal", 1, outcomes, {"uncached_tokens": 1000, "total_tokens": 1500})],
        expected_sessions=5,
    )["per_chain"]["menhir_longitudinal"][0]
    assert row["primary_efficiency_tokens"] == base["client_uncached_tokens"] + 1000


# -- rescore never overwrites --------------------------------------------------------------


def test_rescore_rescores_from_saved_events_and_never_overwrites(tmp_path: Path, monkeypatch) -> None:
    rescore = _load_script("rescore_sprint")
    fixture = replace(SprintFixture.from_file(FIXTURE_PATH), sessions=SprintFixture.from_file(FIXTURE_PATH).sessions[:2])
    run_dir = tmp_path / "run"
    rd = prepare_run_dir(tmp_path, "run", {"x": 1}, resume=False)
    camp = Campaign(fixture, executor=DryRunExecutor(), run_dir=rd, run_id="t", arms=("notes_file",), repeats=1,
                    parallel=1, budget=Budget(None, None), dry_run=True,
                    progress=ProgressFeed("x", "x", 1, ceiling_usd=None, script=Path("/nonexistent/p.py")))
    chains, stop = camp.run()
    (run_dir / "report.json").write_text(json.dumps(camp.build_report(chains, stop, provenance={}, structure_index=None)))
    for idx, session in enumerate(fixture.sessions):
        res = DryRunExecutor().run_session(fixture=fixture, session=session, arm="notes_file", session_index=idx,
                                           namespace="n", repeat=1)
        sdir = run_dir / "runs" / "notes_file" / "r1" / f"{idx:02d}-{session.session_id}"
        sdir.mkdir(parents=True)
        with gzip.open(sdir / "events.jsonl.gz", "wt", encoding="utf-8") as gz:
            gz.write(json.dumps({"type": "text", "part": {"text": res.response_text}}) + "\n")

    def fake_export(pin, dest: Path, cache: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr(rescore, "export_commit", fake_export)
    before = (run_dir / "sessions.jsonl").read_bytes()
    out_dir = tmp_path / "rescored"
    report = rescore.rescore_run(run_dir, fixture, tmp_path / "cache", out_dir)
    assert (out_dir / "rescore_report.json").is_file() and (out_dir / "sessions.rescored.jsonl").is_file()
    assert report["analysis"]["per_chain"]["notes_file"][0]["accuracy"] == pytest.approx(1.0)
    assert (run_dir / "sessions.jsonl").read_bytes() == before  # the source is untouched

    marker = out_dir / "rescore_report.json"
    content = marker.read_bytes()
    with pytest.raises(rescore.RescoreTargetExists):
        rescore.rescore_run(run_dir, fixture, tmp_path / "cache", out_dir)
    assert rescore.main(["--run-dir", str(run_dir), "--fixture", str(FIXTURE_PATH), "--out-dir", str(out_dir)]) == 2
    assert marker.read_bytes() == content


# -- cache shaping ---------------------------------------------------------------------------


def _source_config(tmp_path: Path) -> Path:
    src = tmp_path / "opencode.json"
    src.write_text(json.dumps({"$schema": "x", "provider": {"openai": {"options": {"timeout": 5}}}}),
                   encoding="utf-8")
    return src


def test_minimal_config_shared_cache_key_replaces_per_session_key(tmp_path, monkeypatch) -> None:
    from archolith_bench.core.opencode.isolation import SERVICE_TIER_ENV, minimal_config

    monkeypatch.setenv(SERVICE_TIER_ENV, "flex")
    src = _source_config(tmp_path)
    cfg = minimal_config(src, "openai/gpt-6-luna", None, False, (), prompt_cache_key="sv2-llm-notes_file")
    provider = cfg["provider"]["openai"]
    assert provider["options"] == {"timeout": 5, "setCacheKey": False}
    assert provider["models"]["gpt-6-luna"]["options"] == {
        "serviceTier": "flex", "promptCacheKey": "sv2-llm-notes_file"}
    # Default: no key and OpenCode's per-session key left on.
    plain = minimal_config(src, "openai/gpt-6-luna", None, False, ())
    assert "setCacheKey" not in plain["provider"]["openai"]["options"]
    assert "promptCacheKey" not in plain["provider"]["openai"]["models"]["gpt-6-luna"]["options"]


@pytest.mark.parametrize("arm", V2_ARMS)
def test_cache_shaping_keeps_every_block_and_shares_the_prefix(arm) -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    plain = [compose_full_prompt(s, arm, "ns") for s in fixture.sessions]
    shaped = [compose_full_prompt(s, arm, "ns", cache_shaping=True) for s in fixture.sessions]
    for a, b in zip(plain, shaped):
        assert sorted(a.split("\n\n")) == sorted(b.split("\n\n"))
    # Everything before the session's own prompt is the same for every session of the arm.
    prefixes = {p[: p.index(s.prompt.strip())] for p, s in zip(shaped, fixture.sessions)}
    assert len(prefixes) == 1 and len(next(iter(prefixes))) > 200
    # Default order is the one runs v2 and v2r2 sent.
    assert all(p.startswith(s.prompt.strip()) for p, s in zip(plain, fixture.sessions))
