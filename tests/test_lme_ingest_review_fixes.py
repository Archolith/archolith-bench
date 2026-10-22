"""Offline regression tests for the LongMemEval ingest review fixes (F1-F14, T1, T2).

No network, docker or Menhir: every collaborator of ``ingest`` is replaced, and elapsed time is a
fake clock installed over ``ingest.time`` so the suite stays fast.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_review_fixes_test", "scripts/longmemeval/lib/ingest.py")

FAILED_PROVIDER = {
    "processing_state": "FAILED",
    "processing_error": "429 insufficient_quota: you exceeded your current quota",
}
FAILED_CONTENT = {"processing_state": "FAILED", "processing_error": "extraction produced no edges"}
READY = {"processing_state": "READY"}


class _Clock:
    """Fake ``time`` module: only sleeping (or an explicit advance) moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _Client:
    def __init__(self) -> None:
        self.episodes: list[dict] = []

    def record_turn_evidence(self, namespace, text, **kwargs):
        return {"turn_id": "evidence-1"}

    def ingest(self, namespace, role, content, **kwargs):
        self.episodes.append({"namespace": namespace, "content": content, **kwargs})
        return {"episode_id": f"episode-{len(self.episodes)}"}


def _turn(content: str, **kwargs):
    return ingest.IngestTurn(
        role=kwargs.pop("role", "assistant"),
        content=content,
        occurred_at=None,
        session_id="s",
        **kwargs,
    )


def _run(qid: str, n_turns: int = 1):
    plan = [_turn(f"{qid}{i}", turn_key=f"{qid}|s|t{i}") for i in range(n_turns)]
    return ingest.NamespaceRun(
        item={}, question_id=qid, namespace=f"ns-{qid}", plan=plan, digest=f"digest-{qid}"
    )


def _records(state) -> list[dict]:
    return [json.loads(line) for line in state.path.read_text(encoding="utf-8").splitlines()]


def _zero_counts(namespaces):
    return {
        ns: {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "llm_tasks": 0,
             "processing_attempts": 0, "total": 1}
        for ns in namespaces
    }


def _drive(
    monkeypatch,
    tmp_path,
    runs,
    *,
    processing=lambda uuid: READY,
    health=lambda: True,
    counts=_zero_counts,
    sweep=None,
    finalize=None,
    state=None,
    clock=None,
    **pool_kwargs,
):
    """Run ``_ingest_pool`` against fakes. Returns a namespace of everything observable."""
    clock = clock or _Clock()
    monkeypatch.setattr(ingest, "time", clock)
    client = _Client()
    ops: list[tuple[str, dict]] = []
    finalized: list[str] = []

    def backend(admin, url, op, body=None):
        ops.append((op, body))
        return True

    monkeypatch.setattr(ingest, "_backend", backend)
    monkeypatch.setattr(ingest, "_episode_processing", lambda admin, url, uuid: processing(uuid))
    monkeypatch.setattr(ingest, "_check_menhir_health", lambda admin, url: health())
    monkeypatch.setattr(ingest, "_ns_state_counts_many", lambda namespaces: counts(namespaces))
    monkeypatch.setattr(
        ingest, "_retry_failed_evidence_projections",
        sweep or (lambda namespaces, admin, url, **kw: {ns: 0 for ns in namespaces}),
    )
    state = state or ingest.IngestStateLog(tmp_path / "state.jsonl")

    def default_finalize(run, snapshot):
        finalized.append(run.namespace)
        if finalize is not None:
            finalize(run, snapshot)

    kwargs = dict(pool_size=len(runs), drain_timeout=1800.0, poll_s=1.0)
    kwargs.update(pool_kwargs)
    obs = SimpleNamespace(
        state=state, client=client, ops=ops, finalized=finalized, clock=clock, completed=None,
    )
    obs.completed = ingest._ingest_pool(
        runs, client, object(), "http://menhir", state, finalize=default_finalize, **kwargs
    )
    return obs


def _run_expecting(exc, monkeypatch, tmp_path, runs, **kw):
    """Same as ``_drive`` but the pool is expected to raise; returns (exc_info, observables)."""
    holder: dict = {}
    original_state = kw.get("state") or ingest.IngestStateLog(tmp_path / "state.jsonl")
    kw["state"] = original_state
    with pytest.raises(exc) as info:
        _drive(monkeypatch, tmp_path, runs, **kw)
    holder["state"] = original_state
    return info, SimpleNamespace(**holder)


# --------------------------------------------------------------------------------------------
# 1. Provider-caused FAILED episode (F1)
# --------------------------------------------------------------------------------------------


def test_provider_failed_episode_stops_run_and_spends_no_retry_budget(monkeypatch, tmp_path):
    """Health reports the standing quota failure -> pool raises, no "retry" record is written.

    Breaks if the pool treats a provider failure as a content failure (record_retry before the
    health check) or swallows ProviderStoppedError.
    """
    calls = {"n": 0}

    def health():
        calls["n"] += 1
        if calls["n"] > 1:  # first cycle submits; the requeue attempt then sees the standing stop
            raise ingest.ProviderStoppedError("provider_quota_exhausted: yes")
        return True

    info, seen = _run_expecting(
        ingest.ProviderStoppedError, monkeypatch, tmp_path, [_run("a")],
        processing=lambda uuid: FAILED_PROVIDER, health=health,
    )
    types = [record["type"] for record in _records(seen.state)]
    assert "retry" not in types
    assert "requeued" not in types
    assert seen.state.progress("ns-a").retried_episodes == set()


def test_provider_failed_episode_is_requeued_in_place_when_health_clears(monkeypatch, tmp_path):
    """Reset + enqueue, a provider "requeued" record, no retry record, SAME episode stays active.

    Breaks if the pool advances to the next segment or records an episode retry.
    """
    calls: dict[str, int] = defaultdict(int)
    polled: list[tuple[str, int]] = []
    holder: dict = {}

    def processing(uuid):
        calls[uuid] += 1
        polled.append((uuid, len(holder["client"].episodes)))
        return FAILED_PROVIDER if (uuid == "episode-1" and calls[uuid] == 1) else READY

    # The client is created inside _drive; capture it through the first ingest call.
    original_ingest = _Client.ingest

    def spying_ingest(self, *a, **k):
        holder["client"] = self
        return original_ingest(self, *a, **k)

    monkeypatch.setattr(_Client, "ingest", spying_ingest)
    obs = _drive(monkeypatch, tmp_path, [_run("a", n_turns=2)], processing=processing)

    # The first re-poll after the failure is of the SAME episode, with nothing newer submitted.
    assert polled[0] == ("episode-1", 1)
    assert polled[1] == ("episode-1", 1)
    assert [op for op, body in obs.ops if body == {"episode_uuid": "episode-1"}] == [
        "force_reset_failed_episode", "enqueue_pending_episode",
    ]
    progress = obs.state.progress("ns-a")
    assert progress.requeued == {"provider": 1}
    assert progress.retried_episodes == set()
    assert "retry" not in [record["type"] for record in _records(obs.state)]
    # After the row turns READY the namespace continues to its second segment and finishes.
    assert [e["content"] for e in obs.client.episodes] == ["a0", "a1"]
    assert obs.finalized == ["ns-a"]


def test_provider_failure_beyond_max_requeues_stops_the_run(monkeypatch, tmp_path):
    """The same episode failing on the provider more than max_provider_requeues times raises.

    Breaks if the per-episode requeue cap is removed (the run would loop forever).
    """
    info, seen = _run_expecting(
        ingest.ProviderStoppedError, monkeypatch, tmp_path, [_run("a")],
        processing=lambda uuid: FAILED_PROVIDER, max_provider_requeues=3,
    )
    assert "failed on the provider" in str(info.value)
    assert seen.state.progress("ns-a").requeued == {"provider": 3}


def test_budget_retried_episode_failing_on_provider_is_still_requeued(monkeypatch, tmp_path):
    """An episode already in retried_episodes that FAILS on the provider is not terminal.

    Breaks if the provider check sits after the ``already_retried`` terminal branch.
    """
    state = ingest.IngestStateLog(tmp_path / "state.jsonl")
    state.record_retry("ns-a", "episode-1", "episode")
    calls: dict[str, int] = defaultdict(int)

    def processing(uuid):
        calls[uuid] += 1
        return FAILED_PROVIDER if calls[uuid] == 1 else READY

    obs = _drive(monkeypatch, tmp_path, [_run("a")], processing=processing, state=state)

    assert obs.state.progress("ns-a").requeued == {"provider": 1}
    assert ("force_reset_failed_episode", {"episode_uuid": "episode-1"}) in obs.ops
    assert calls["episode-1"] == 2  # polled again after the requeue, not abandoned
    assert obs.finalized == ["ns-a"]


# --------------------------------------------------------------------------------------------
# 2. Projection sweep (F1/F6/F7)
# --------------------------------------------------------------------------------------------


def _sweep_env(monkeypatch, failed, healthy=True):
    events: list[tuple] = []
    monkeypatch.setattr(ingest, "_failed_evidence_projections", lambda ns: failed)
    monkeypatch.setattr(ingest, "_check_menhir_health", lambda admin, url: healthy)

    def backend(admin, url, op, body=None):
        events.append(("backend", op, body["episode_uuid"]))
        return True

    monkeypatch.setattr(ingest, "_backend", backend)
    return events


def test_projection_sweep_provider_failure_uses_provider_path(monkeypatch):
    events = _sweep_env(monkeypatch, [("u1", True)])
    retries: list = []
    result = ingest._retry_failed_evidence_projections(
        ["ns"], object(), "http://m",
        on_retry=lambda ns, uuid: retries.append((ns, uuid)),
        on_requeued=lambda ns, uuid, kind: events.append(("requeued", ns, uuid, kind)),
    )
    assert retries == []
    assert ("requeued", "ns", "u1", "provider") in events
    assert result == {"ns": 1}


def test_projection_sweep_provider_failure_with_unhealthy_menhir_raises_and_records_nothing(
    monkeypatch,
):
    events = _sweep_env(monkeypatch, [("u1", True)], healthy=False)
    with pytest.raises(ingest.GraphUnreadable):
        ingest._retry_failed_evidence_projections(
            ["ns"], object(), "http://m",
            on_retry=lambda ns, uuid: events.append(("retry", ns, uuid)),
            on_requeued=lambda ns, uuid, kind: events.append(("requeued", ns, uuid, kind)),
        )
    assert events == []


def test_projection_sweep_content_failure_records_retry_before_reset_then_requeued(monkeypatch):
    events = _sweep_env(monkeypatch, [("u2", False)])
    ingest._retry_failed_evidence_projections(
        ["ns"], object(), "http://m",
        on_retry=lambda ns, uuid: events.append(("retry", ns, uuid)),
        on_requeued=lambda ns, uuid, kind: events.append(("requeued", ns, uuid, kind)),
    )
    assert events == [
        ("retry", "ns", "u2"),
        ("backend", "force_reset_failed_episode", "u2"),
        ("backend", "enqueue_pending_episode", "u2"),
        ("requeued", "ns", "u2", "projection"),
    ]


def test_pool_survives_projection_enqueue_race_and_finalizes(monkeypatch, tmp_path):
    """o500 attempts 1-2 crashed here: the settle sweep reset a FAILED projection, Menhir
    refused the enqueue because the row was already queued or claimed, and the client raised.
    Now the row is read back, counted as requeued, and the namespace re-settles and finalizes."""
    real_sweep = ingest._retry_failed_evidence_projections
    failed_sets = iter([[("proj-1", False)]])
    sweep_ops: list[str] = []

    def racing_backend(admin, url, op, body=None):
        sweep_ops.append(op)
        return op != "enqueue_pending_episode"

    def sweep(namespaces, admin, url, **kw):
        monkeypatch.setattr(ingest, "_backend", racing_backend)
        monkeypatch.setattr(
            ingest, "_failed_evidence_projections", lambda ns: next(failed_sets, []),
        )
        return real_sweep(namespaces, admin, url, **kw)

    processing = {"proj-1": {"processing_state": "ENRICHING"}}
    state = ingest.IngestStateLog(tmp_path / "state.jsonl")

    obs = _drive(
        monkeypatch, tmp_path, [_run("a")],
        processing=lambda uuid: processing.get(uuid, READY),
        sweep=sweep, state=state,
    )

    assert obs.finalized == ["ns-a"]
    # _drive stubs _episode_processing (the read-back), so the backend sees reset + enqueue only.
    assert sweep_ops == ["force_reset_failed_episode", "enqueue_pending_episode"]
    assert list(state.progress("ns-a").retried_projections) == ["proj-1"]


# --------------------------------------------------------------------------------------------
# 3. _failed_evidence_projections (F7)
# --------------------------------------------------------------------------------------------


def test_failed_projections_empty_count_result_is_unreadable_not_none(monkeypatch):
    """_cypher returns [] on failure; that must not read as "no failed projections"."""
    monkeypatch.setattr(ingest, "_cypher", lambda query: [])
    with pytest.raises(ingest.GraphUnreadable):
        ingest._failed_evidence_projections("ns")


def test_failed_projections_short_list_is_unreadable(monkeypatch):
    answers = [[["2"]], [["u1", "true"]]]
    monkeypatch.setattr(ingest, "_cypher", lambda query: answers.pop(0))
    with pytest.raises(ingest.GraphUnreadable):
        ingest._failed_evidence_projections("ns")


def test_failed_projections_zero_count_returns_empty_without_second_query(monkeypatch):
    queries: list[str] = []

    def fake(query):
        queries.append(query)
        return [["0"]]

    monkeypatch.setattr(ingest, "_cypher", fake)
    assert ingest._failed_evidence_projections("ns") == []
    assert len(queries) == 1


def test_failed_projections_parse_provider_flag_from_rendered_rows(monkeypatch):
    answers = [[["2"]], [["u1", "true"], ["u2", "false"]]]
    monkeypatch.setattr(ingest, "_cypher", lambda query: answers.pop(0))
    assert ingest._failed_evidence_projections("ns") == [("u1", True), ("u2", False)]


# --------------------------------------------------------------------------------------------
# 4. Unreadable projection sweep inside the pool (F7)
# --------------------------------------------------------------------------------------------


def test_unreadable_projection_sweep_is_not_finalized_until_a_sweep_succeeds(
    monkeypatch, tmp_path
):
    """Breaks if GraphUnreadable from the sweep is treated as "no failed projections"."""
    sweeps = {"n": 0}
    snapshots: list[list[str]] = []
    finalized: list[str] = []

    def sweep(namespaces, admin, url, **kw):
        sweeps["n"] += 1
        snapshots.append(list(finalized))
        if sweeps["n"] == 1:
            raise ingest.GraphUnreadable("neo4j down")
        return {ns: 0 for ns in namespaces}

    _drive(
        monkeypatch, tmp_path, [_run("a")], sweep=sweep,
        finalize=lambda run, snap: finalized.append(run.namespace),
    )
    # Two sweeps happened, and at neither was the namespace already finalized.
    assert snapshots == [[], []]
    assert finalized == ["ns-a"]


# --------------------------------------------------------------------------------------------
# 5. Honest counts (F6)
# --------------------------------------------------------------------------------------------


class _MainAdapter:
    items: list[dict] = []

    def load_items(self, **kwargs):
        return self.items

    def sessions(self, item):
        return item["sessions"]

    def question(self, item):
        return item["question"]


def _install_main_fakes(monkeypatch, items, *, backend, processing, drain_many, close_locks):
    clock = _Clock()
    _MainAdapter.items = items
    monkeypatch.setattr(ingest, "time", clock)
    monkeypatch.setattr(ingest, "LongMemEvalMemoryAdapter", _MainAdapter)
    monkeypatch.setattr(ingest, "HttpMenhirClient", lambda url: object())
    monkeypatch.setattr(ingest.httpx, "Client", lambda **kw: object())
    episodes = {"n": 0}

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        episodes["n"] += 1
        return f"episode-{episodes['n']}"

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(ingest, "_backend", backend)
    monkeypatch.setattr(ingest, "_episode_processing", lambda admin, url, uuid: processing(uuid))
    monkeypatch.setattr(ingest, "_drain_many", drain_many)
    monkeypatch.setattr(ingest, "_check_menhir_health", lambda admin, url: True)
    monkeypatch.setattr(
        ingest, "_retry_failed_evidence_projections",
        lambda namespaces, admin, url, **kw: {ns: 0 for ns in namespaces},
    )
    monkeypatch.setattr(
        ingest, "_integrity_counts",
        lambda namespace, submitted: {"cross_namespace_links": 0, "user_turns_admitted": 1},
    )
    monkeypatch.setattr(
        ingest, "_ns_state_counts_many",
        lambda namespaces: {
            ns: {"pending": 0, "ready": 0, "enriching": 0, "failed": 1, "llm_tasks": 0,
                 "processing_attempts": 1, "total": 0}
            for ns in namespaces
        },
    )
    original_lock = ingest._hold_run_lock

    def tracked_lock(path):
        handle = original_lock(path)
        close_locks.append(handle)
        return handle

    monkeypatch.setattr(ingest, "_hold_run_lock", tracked_lock)


def _ok_drain(namespaces, admin, url, **kwargs):
    return {ns: {"failed": 0, "timed_out": False} for ns in namespaces}


def _item(qid: str) -> dict:
    return {
        "question_id": qid, "question": f"{qid}?", "answer": "x", "question_type": "t",
        "sessions": [[{"role": "user", "content": f"{qid}1"}]],
    }


def test_failed_reset_does_not_count_as_a_requeue_in_the_manifest(monkeypatch, tmp_path):
    """A content failure whose reset returns False spent a retry but requeued nothing.

    Breaks if the manifest's failed_requeued is derived from retries spent (or from a record
    written before the reset result is known) rather than from successful requeues.
    """
    locks: list = []

    def backend(admin, url, op, body=None):
        return False if op == "force_reset_failed_episode" else True

    _install_main_fakes(
        monkeypatch, [_item("a")], backend=backend, processing=lambda uuid: FAILED_CONTENT,
        drain_many=_ok_drain, close_locks=locks,
    )
    manifest_path = tmp_path / "manifest.json"
    try:
        assert ingest.main(["--manifest", str(manifest_path), "--segmentation", "none"]) == 0
    finally:
        for handle in locks:
            handle.close()

    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in rows] == ["a"]
    assert rows[0]["failed_requeued"] == 0
    progress = ingest.IngestStateLog(ingest.IngestStateLog.path_for(manifest_path)).progress("lme-a")
    assert progress.retried_episodes == {"episode-1"}  # the retry WAS spent
    assert progress.requeued == {}  # but nothing was requeued


# --------------------------------------------------------------------------------------------
# 6. Watchdog evaluated before finalize (F2)
# --------------------------------------------------------------------------------------------


def test_slow_finalize_of_one_namespace_does_not_trip_the_watchdog_for_another(
    monkeypatch, tmp_path
):
    """A (1 turn) settles a cycle before B (2 turns); A's finalize burns 500s > drain_timeout.

    B, still settling, must not be declared stalled by time spent finalizing A.
    Breaks if the watchdog runs after the finalize loop.
    """
    clock = _Clock()
    obs = _drive(
        monkeypatch, tmp_path, [_run("a", 1), _run("b", 2)],
        clock=clock, drain_timeout=100.0, settle_gap_s=0.0,
        finalize=lambda run, snap: clock.sleep(500.0) if run.namespace == "ns-a" else None,
    )
    assert sorted(obs.finalized) == ["ns-a", "ns-b"]
    assert obs.finalized[0] == "ns-a"  # A really did finalize first, then B was still pending


# --------------------------------------------------------------------------------------------
# 7. Outage enforced mid-cycle (F3)
# --------------------------------------------------------------------------------------------


def test_outage_timeout_is_enforced_during_the_read_loop(monkeypatch, tmp_path):
    clock = _Clock()
    reads = {"n": 0}

    def hanging(uuid):
        reads["n"] += 1
        clock.sleep(200.0)  # a hanging request
        return None

    runs = [_run(f"q{i}") for i in range(10)]
    with pytest.raises(ingest.ServerUnavailableError):
        _drive(
            monkeypatch, tmp_path, runs, clock=clock, processing=hanging, health=lambda: False,
            outage_timeout=300.0,
        )
    # Outage clock starts at the cycle's failed health check (t0); reads end at t0+200 and t0+400.
    assert 2 <= reads["n"] <= 3, reads["n"]


# --------------------------------------------------------------------------------------------
# 8. Settle gap (F4)
# --------------------------------------------------------------------------------------------


def test_finalize_waits_settle_gap_after_first_zero_reading(monkeypatch, tmp_path):
    clock = _Clock()
    zero_reads: list[float] = []
    finalized_at: list[float] = []

    def counts(namespaces):
        zero_reads.append(clock.now)
        return _zero_counts(namespaces)

    _drive(
        monkeypatch, tmp_path, [_run("a")], clock=clock, counts=counts, poll_s=0.5,
        settle_gap_s=2.0, finalize=lambda run, snap: finalized_at.append(clock.now),
    )
    assert len(finalized_at) == 1
    assert finalized_at[0] - zero_reads[0] >= 2.0
    assert len(zero_reads) >= 3  # not two back-to-back polls


# --------------------------------------------------------------------------------------------
# 9. Final drain covers every manifested namespace (F5)
# --------------------------------------------------------------------------------------------


def _main_with_full_manifest(monkeypatch, tmp_path, drain_many):
    locks: list = []
    drains: list[list[str]] = []

    def recording_drain(namespaces, admin, url, **kwargs):
        drains.append(list(namespaces))
        return drain_many(namespaces)

    _install_main_fakes(
        monkeypatch, [_item("a")], backend=lambda *a, **k: True, processing=lambda uuid: READY,
        drain_many=recording_drain, close_locks=locks,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"question_id": "a", "namespace": "lme-a"}]), encoding="utf-8"
    )
    return manifest_path, drains, locks


def test_final_drain_runs_over_prior_manifest_rows_when_nothing_is_left_to_ingest(
    monkeypatch, tmp_path
):
    manifest_path, drains, locks = _main_with_full_manifest(
        monkeypatch, tmp_path,
        lambda namespaces: {ns: {"timed_out": False} for ns in namespaces},
    )
    try:
        assert ingest.main(["--manifest", str(manifest_path), "--segmentation", "none"]) == 0
    finally:
        for handle in locks:
            handle.close()
    assert drains == [["lme-a"]]


def test_final_drain_timeout_over_prior_manifest_rows_raises(monkeypatch, tmp_path):
    manifest_path, drains, locks = _main_with_full_manifest(
        monkeypatch, tmp_path,
        lambda namespaces: {ns: {"timed_out": True} for ns in namespaces},
    )
    try:
        with pytest.raises(RuntimeError, match="final drain timed out.*lme-a"):
            ingest.main(["--manifest", str(manifest_path), "--segmentation", "none"])
    finally:
        for handle in locks:
            handle.close()
    assert drains == [["lme-a"]]


# --------------------------------------------------------------------------------------------
# 10. Run lock (F10)
# --------------------------------------------------------------------------------------------


def test_run_lock_refuses_a_second_holder_until_the_first_releases(tmp_path):
    path = tmp_path / "run.lock"
    first = ingest._hold_run_lock(path)
    try:
        with pytest.raises(RuntimeError, match="another ingest run holds"):
            ingest._hold_run_lock(path)
    finally:
        first.close()
    second = ingest._hold_run_lock(path)  # released by the close above
    second.close()


# --------------------------------------------------------------------------------------------
# 11. Duplicate ingest keys (F14)
# --------------------------------------------------------------------------------------------


def test_turn_plan_rejects_two_extractable_turns_with_the_same_key(monkeypatch):
    dup = [
        _turn("first", turn_key="ns|s|t0", segment_index=0),
        _turn("second", turn_key="ns|s|t0", segment_index=0),
    ]
    monkeypatch.setattr(ingest, "_iter_item_turns", lambda *a, **k: iter(dup))
    with pytest.raises(RuntimeError, match="repeats ingest keys"):
        ingest._turn_plan(object(), {}, "ns", "none")


def test_turn_plan_allows_duplicate_keys_on_evidence_only_turns(monkeypatch):
    dup = [
        _turn("first", turn_key="ns|s|t0", segment_index=0, extract=False),
        _turn("second", turn_key="ns|s|t0", segment_index=0, extract=False),
    ]
    monkeypatch.setattr(ingest, "_iter_item_turns", lambda *a, **k: iter(dup))
    assert len(ingest._turn_plan(object(), {}, "ns", "none")) == 2


# --------------------------------------------------------------------------------------------
# 12. Segmentation mode refusal (F8)
# --------------------------------------------------------------------------------------------


def _startup(monkeypatch, tmp_path, recorded_mode, run_mode, listed=()):
    monkeypatch.setattr(ingest, "_await_stale_episode_settlement", lambda *a, **k: None)
    resets: list[str] = []
    monkeypatch.setattr(ingest, "_reset_namespace", lambda admin, url, ns: resets.append(ns))
    state = ingest.IngestStateLog(tmp_path / "state.jsonl")
    run = _run("a")
    state.record_plan(run.namespace, run.digest, 1, recorded_mode)
    args = SimpleNamespace(
        reset_namespace=list(listed), reset_file=None, segmentation=run_mode,
        reset_settle_timeout=1.0, menhir_url="http://menhir",
    )
    ingest._startup_checks([run], state, args, object())
    return resets, state


def test_startup_refuses_a_resume_under_a_different_segmentation(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="segmentation"):
        _startup(monkeypatch, tmp_path, "claims", "adaptive")


def test_startup_allows_different_segmentation_for_a_namespace_listed_for_reset(
    monkeypatch, tmp_path
):
    resets, state = _startup(monkeypatch, tmp_path, "claims", "adaptive", listed=["ns-a"])
    assert resets == ["ns-a"]


def test_startup_allows_same_segmentation_and_unrecorded_segmentation(monkeypatch, tmp_path):
    _startup(monkeypatch, tmp_path, "adaptive", "adaptive")
    (tmp_path / "legacy").mkdir()
    _startup(monkeypatch, tmp_path / "legacy", None, "adaptive")  # old plan record, no mode


# --------------------------------------------------------------------------------------------
# 13. T1: idempotency key plumbing
# --------------------------------------------------------------------------------------------


def test_submit_next_passes_the_segment_ingest_key_to_ingest_turn(monkeypatch, tmp_path):
    captured: list[dict] = []

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        captured.append(kwargs)
        return f"episode-{len(captured)}"

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    run = ingest.NamespaceRun(
        item={}, question_id="a", namespace="ns-a", digest="d",
        plan=[
            _turn("x", turn_key="ns-a|s|t0", segment_index=0),
            _turn("y", turn_key="ns-a|s|t0", segment_index=1),
        ],
    )
    _drive(monkeypatch, tmp_path, [run])
    assert [kw["ingest_key"] for kw in captured] == ["ns-a|s|t0#s0", "ns-a|s|t0#s1"]


def test_ingest_turn_forwards_the_key_as_idempotency_key():
    sent: list[dict] = []

    class Client:
        def ingest(self, ns, role, content, **kwargs):
            sent.append(kwargs)
            return {"episode_id": "e1"}

    assert ingest._ingest_turn(Client(), "ns", "assistant", "hi", ingest_key="k#s0") == "e1"
    assert sent[0]["idempotency_key"] == "k#s0"


def test_ingest_turn_without_a_key_sends_no_idempotency_kwarg():
    sent: list[dict] = []

    class Client:
        def ingest(self, ns, role, content, **kwargs):
            sent.append(kwargs)
            return {"episode_id": "e1"}

    ingest._ingest_turn(Client(), "ns", "assistant", "hi", ingest_key=None)
    assert "idempotency_key" not in sent[0]


# --------------------------------------------------------------------------------------------
# 14. T2: segment_index across the segments of one source turn
# --------------------------------------------------------------------------------------------


class _Adapter:
    def sessions(self, item):
        return item["sessions"]


def test_segment_index_counts_segments_of_one_source_turn_and_resets_per_turn():
    long_turn = (
        "I moved to Portland in March of this year. My rent is now $2,400 a month. "
        "I switched to the night shift at the hospital downtown."
    )
    assert len(long_turn) > 100
    item = {"sessions": [[
        {"role": "user", "content": long_turn},
        {"role": "user", "content": "Short one."},
    ]]}
    turns = list(ingest._iter_item_turns(_Adapter(), item, "ns", "sentence"))

    first = [t for t in turns if t.turn_key.endswith("|t0")]
    second = [t for t in turns if t.turn_key.endswith("|t1")]
    assert [t.segment_index for t in first] == [0, 1, 2]
    assert [t.segment_index for t in second] == [0]
    assert len({t.turn_key for t in first}) == 1
    assert len({ingest._ingest_key(t) for t in turns}) == len(turns)


# --------------------------------------------------------------------------------------------
# 15. neo4j_http (F11)
# --------------------------------------------------------------------------------------------


class _FakeHttp:
    def __init__(self, rows):
        self.rows = rows

    def post(self, url, **kwargs):
        rows = self.rows
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"errors": [], "results": [{"data": [{"row": r} for r in rows]}]},
        )


def test_namespace_state_counts_raises_when_a_requested_namespace_has_no_row():
    reader = ingest.Neo4jHttpReader("http://neo4j", "u", "p", client=_FakeHttp([["ns-a", 0, 1, 0, 0, 0, 0, 1]]))
    with pytest.raises(ingest.Neo4jUnavailable, match="no row"):
        reader.namespace_state_counts(["ns-a", "ns-b"])


def test_namespace_state_counts_returns_every_requested_namespace_when_complete():
    rows = [["ns-a", 0, 1, 0, 0, 0, 0, 1], ["ns-b", 0, 0, 0, 0, 0, 0, 0]]
    reader = ingest.Neo4jHttpReader("http://neo4j", "u", "p", client=_FakeHttp(rows))
    counts = reader.namespace_state_counts(["ns-a", "ns-b"])
    assert counts["ns-a"]["ready"] == 1 and counts["ns-b"]["total"] == 0
