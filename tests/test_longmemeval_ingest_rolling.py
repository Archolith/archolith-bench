"""Rolling namespace window for LongMemEval ingest (--rolling-window / LME_ROLLING_WINDOW=1)."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_real_sleep = time.sleep


def _load_ingest():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "ingest.py"
    spec = importlib.util.spec_from_file_location("lme_ingest_rolling_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_ingest_rolling_test"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


def _item(qid: str, *contents: str) -> dict:
    return {
        "question_id": qid,
        "question": f"Question {qid}?",
        "answer": qid.upper(),
        "question_type": "temporal-reasoning",
        "sessions": [[{"role": "user", "content": c} for c in contents]],
    }


def _settled(failed: int = 0) -> dict:
    return {
        "pending": 0, "ready": 1, "enriching": 0, "failed": failed, "llm_tasks": 3,
        "processing_attempts": 1, "total": 1, "timed_out": False,
    }


class Harness:
    """Fake Menhir: episodes READY on first poll unless a test overrides ``processing``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, items: list[dict]) -> None:
        self.events: list[str] = []
        self.lock = threading.Lock()
        self.episode_ns: dict[str, str] = {}
        self.drain_calls: list[tuple[list[str], bool]] = []
        self.processing = lambda uuid: {"processing_state": "READY"}
        self.drain = lambda namespace: _settled()
        self.polls = 0

        harness = self

        class FakeAdapter:
            def load_items(self, **kwargs):
                return items

            def sessions(self, item):
                return item["sessions"]

            def question(self, item):
                return item["question"]

        def fake_ingest_turn(client, namespace, role, content, **kwargs):
            with harness.lock:
                uuid = f"ep-{len(harness.episode_ns) + 1}"
                harness.episode_ns[uuid] = namespace
                harness.events.append(f"submit:{content}")
            return uuid

        def fake_processing(admin, url, uuid):
            harness.polls += 1
            if harness.polls > 5000:
                raise AssertionError("scheduler made no progress")
            return harness.processing(uuid)

        def fake_drain_many(namespaces, admin, url, **kwargs):
            with harness.lock:
                harness.drain_calls.append((list(namespaces), kwargs.get("global_queue_gate", True)))
            return {namespace: harness.drain(namespace) for namespace in namespaces}

        def fake_settle(namespaces, **kwargs):
            harness.events.append("settle:" + ",".join(namespaces))

        def fake_reset(admin, url, namespace):
            harness.events.append(f"reset:{namespace}")

        monkeypatch.setattr(ingest, "LongMemEvalMemoryAdapter", FakeAdapter)
        monkeypatch.setattr(ingest, "HttpMenhirClient", lambda url: object())
        monkeypatch.setattr(ingest.httpx, "Client", lambda **kwargs: object())
        monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
        monkeypatch.setattr(ingest, "_episode_processing", fake_processing)
        monkeypatch.setattr(ingest, "_drain_many", fake_drain_many)
        monkeypatch.setattr(ingest, "_await_stale_episode_settlement", fake_settle)
        monkeypatch.setattr(ingest, "_reset_namespace", fake_reset)
        monkeypatch.setattr(
            ingest, "_integrity_counts",
            lambda namespace, submitted: {"cross_namespace_links": 0},
        )
        monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    def run(self, tmp_path: Path, *extra: str) -> list[dict]:
        manifest = tmp_path / "manifest.json"
        result = ingest.main([
            "--limit", "50", "--rolling-window", "--segmentation", "none",
            "--manifest", str(manifest), *extra,
        ])
        assert result == 0
        return json.loads(manifest.read_text(encoding="utf-8"))


def test_parser_reads_rolling_window_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LME_ROLLING_WINDOW", raising=False)
    assert ingest._parse_args([]).rolling_window is False
    monkeypatch.setenv("LME_ROLLING_WINDOW", "1")
    assert ingest._parse_args([]).rolling_window is True


def test_namespace_drain_without_global_gate_ignores_queue_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def busy_queue(admin, url):
        raise AssertionError("global queue must not gate a per-namespace drain")

    monkeypatch.setattr(ingest, "_queue_depth", busy_queue)
    monkeypatch.setattr(ingest, "_ns_state_counts", lambda ns: {**_settled(), "timed_out": None})
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    drained = ingest._drain_many(["lme-a"], object(), "http://x", global_queue_gate=False)

    assert drained["lme-a"]["timed_out"] is False


def test_namespace_drain_without_global_gate_still_waits_on_own_pending_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([
        {**_settled(), "pending": 2},  # evidence projections still queued
        {**_settled(), "pending": -1},  # unreadable is never settled
        _settled(),
        _settled(),
    ])
    monkeypatch.setattr(ingest, "_ns_state_counts", lambda ns: next(states))
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    drained = ingest._drain_many(["lme-a"], object(), "http://x", global_queue_gate=False)

    assert drained["lme-a"]["timed_out"] is False
    with pytest.raises(StopIteration):
        next(states)


def test_next_item_starts_while_a_long_item_is_still_enriching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1", "a2"), _item("b", "b1"), _item("c", "c1")]
    h = Harness(monkeypatch, items)

    def processing(uuid):
        # a's first episode stays ENRICHING until c has been submitted: a fixed window would
        # never start c before a drains, so this only terminates under rolling scheduling.
        if h.episode_ns[uuid] == "lme-a" and "submit:c1" not in h.events:
            return {"processing_state": "ENRICHING"}
        return {"processing_state": "READY"}

    h.processing = processing
    manifest = h.run(tmp_path, "--namespace-window", "2")

    assert [row["question_id"] for row in manifest] == ["b", "a", "c"] or \
        [row["question_id"] for row in manifest] == ["b", "c", "a"]
    assert h.events.index("submit:c1") < h.events.index("submit:a2")
    # Each item settles on stale rows, then resets, before its first submission.
    for ns, first in (("lme-a", "a1"), ("lme-b", "b1"), ("lme-c", "c1")):
        assert h.events.index(f"settle:{ns}") < h.events.index(f"reset:{ns}") < \
            h.events.index(f"submit:{first}")
    # Drains are per item and never wait on the global queue other items keep busy.
    assert sorted(ns for call in h.drain_calls for ns in call[0]) == ["lme-a", "lme-b", "lme-c"]
    assert all(len(namespaces) == 1 and gate is False for namespaces, gate in h.drain_calls)
    assert all(row["rolling_window"] is True and row["namespace_window"] == 2 for row in manifest)


def test_at_most_window_items_are_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item(q, f"{q}1", f"{q}2") for q in "abcdef"]
    h = Harness(monkeypatch, items)
    in_flight: set[str] = set()
    peak = 0
    real_settle = ingest._await_stale_episode_settlement

    def settle(namespaces, **kwargs):
        nonlocal peak
        in_flight.update(namespaces)
        peak = max(peak, len(in_flight))
        real_settle(namespaces, **kwargs)

    def write_manifest(path, manifest):
        in_flight.discard(manifest[-1]["namespace"])
        path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(ingest, "_await_stale_episode_settlement", settle)
    monkeypatch.setattr(ingest, "_write_manifest", write_manifest)
    manifest = h.run(tmp_path, "--namespace-window", "3")

    assert sorted(row["question_id"] for row in manifest) == list("abcdef")
    assert peak == 3


def test_failed_episode_is_retried_once_before_namespace_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1", "a2")]
    h = Harness(monkeypatch, items)
    backend_calls: list[str] = []
    seen: dict[str, int] = {}

    def processing(uuid):
        seen[uuid] = seen.get(uuid, 0) + 1
        if uuid == "ep-1" and seen[uuid] == 1:
            return {"processing_state": "FAILED"}
        return {"processing_state": "READY"}

    def fake_backend(admin, url, operation, body):
        backend_calls.append(f"{operation}:{body['episode_uuid']}")
        return True

    h.processing = processing
    monkeypatch.setattr(ingest, "_backend", fake_backend)
    manifest = h.run(tmp_path, "--namespace-window", "4")

    assert backend_calls == ["force_reset_failed_episode:ep-1", "enqueue_pending_episode:ep-1"]
    # a2 is submitted only after the retried a1 reached READY.
    assert seen["ep-1"] == 2
    assert [e for e in h.events if e.startswith("submit:")] == ["submit:a1", "submit:a2"]
    assert manifest[0]["failed_requeued"] == 1


def test_residual_failure_refuses_consolidation_and_stops_new_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1"), _item("b", "b1")]
    h = Harness(monkeypatch, items)
    consolidated: list[str] = []
    h.drain = lambda namespace: _settled(failed=1)
    monkeypatch.setattr(
        ingest, "_retry_failed_evidence_projections_after_drain",
        lambda namespaces, admin, url, drained, requeued, **kwargs: drained,
    )
    monkeypatch.setattr(
        ingest, "_consolidate_lanes",
        lambda admin, url, namespace, **kwargs: consolidated.append(namespace) or {},
    )

    with pytest.raises(RuntimeError, match="residual FAILED"):
        h.run(tmp_path, "--namespace-window", "1", "--consolidate-counter")

    assert consolidated == []
    assert "settle:lme-b" not in h.events
    assert not (tmp_path / "manifest.json").exists()


def test_drain_timeout_refuses_to_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = Harness(monkeypatch, [_item("a", "a1")])
    h.drain = lambda namespace: {**_settled(), "timed_out": True}

    with pytest.raises(RuntimeError, match="rolling-window drain timed out"):
        h.run(tmp_path, "--namespace-window", "2")

    assert not (tmp_path / "manifest.json").exists()


def test_failure_keeps_paid_work_of_items_already_finishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1"), _item("b", "b1"), _item("c", "c1")]
    h = Harness(monkeypatch, items)

    def drain(namespace):
        if namespace == "lme-b":
            raise RuntimeError("boom")
        _real_sleep(0.3)  # a is still finishing when b's failure surfaces
        return _settled()

    h.drain = drain
    with pytest.raises(RuntimeError, match="boom"):
        h.run(tmp_path, "--namespace-window", "2")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [row["question_id"] for row in manifest] == ["a"]
    assert "settle:lme-c" not in h.events


def test_resume_skips_manifested_items_in_rolling_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1"), _item("b", "b1")]
    h = Harness(monkeypatch, items)
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"question_id": "a", "namespace": "lme-a"}]), encoding="utf-8",
    )

    manifest = h.run(tmp_path, "--namespace-window", "2")

    assert [row["question_id"] for row in manifest] == ["a", "b"]
    assert "reset:lme-a" not in h.events
