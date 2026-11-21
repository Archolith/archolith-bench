"""The LongMemEval ingest build-count readers go through Menhir backend ops."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


ingest = _load_script("lme_ingest_build_ops_test", "scripts/longmemeval/lib/ingest.py")

STATE_KEYS = ("pending", "ready", "enriching", "failed", "llm_tasks", "processing_attempts", "total")
SCALAR_KEYS = (
    "turn_evidence", "typed_assertions", "scalar_views", "scalar_history_views",
    "user_founded_scalar_views",
)
INTEGRITY_KEYS = (
    "user_turns_admitted", "admitted_turns_unprojected", "assertions_unfounded",
    "episodes_valid_at_unsubmitted", "ready_episodes_time_unpaired", "cross_namespace_links",
)


class _Recorder:
    def __init__(self, response=None, error: Exception | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.response = response
        self.error = error

    def __call__(self, admin, url, op, body=None):
        self.calls.append((op, body))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "_BUILD_BACKEND", (object(), "http://menhir"))


def _patch(monkeypatch: pytest.MonkeyPatch, **kwargs) -> _Recorder:
    recorder = _Recorder(**kwargs)
    monkeypatch.setattr(ingest, "_backend", recorder)
    return recorder


def test_ns_state_counts_sends_state_section_and_maps_response(monkeypatch, bound) -> None:
    state = dict(zip(STATE_KEYS, range(1, 8)))
    rec = _patch(monkeypatch, response={"namespace": "ns", "state": state})
    assert ingest._ns_state_counts("ns") == state
    assert rec.calls == [
        ("fetch_namespace_build_counts", {"namespace": "ns", "sections": ["state"]})
    ]


def test_ns_state_counts_many_dedupes_and_keeps_order(monkeypatch, bound) -> None:
    state = dict(zip(STATE_KEYS, range(7)))
    rec = _patch(monkeypatch, response={"state": state})
    result = ingest._ns_state_counts_many(["b", "a", "b"])
    assert list(result) == ["b", "a"]
    assert [body["namespace"] for _, body in rec.calls] == ["b", "a"]


def test_scalar_counts_sends_scalar_section_and_maps_response(monkeypatch, bound) -> None:
    scalar = dict(zip(SCALAR_KEYS, range(10, 15)))
    rec = _patch(monkeypatch, response={"scalar": scalar})
    assert ingest._scalar_counts("ns") == scalar
    assert rec.calls == [
        ("fetch_namespace_build_counts", {"namespace": "ns", "sections": ["scalar"]})
    ]


def test_integrity_counts_sends_submitted_millis_and_maps_response(monkeypatch, bound) -> None:
    integrity = dict(zip(INTEGRITY_KEYS, range(6)))
    rec = _patch(monkeypatch, response={"integrity": integrity})
    assert ingest._integrity_counts("ns", (1, 2)) == integrity
    assert rec.calls == [(
        "fetch_namespace_build_counts",
        {"namespace": "ns", "sections": ["integrity"], "submitted_epoch_millis": [1, 2]},
    )]


def test_integrity_counts_maps_none_to_minus_one(monkeypatch, bound) -> None:
    integrity = dict(zip(INTEGRITY_KEYS, range(6)))
    integrity["assertions_unfounded"] = None
    _patch(monkeypatch, response={"integrity": integrity})
    result = ingest._integrity_counts("ns", [])
    assert result["assertions_unfounded"] == -1
    assert result["user_turns_admitted"] == 0


@pytest.mark.parametrize(
    ("reader", "keys"),
    [
        (lambda: ingest._ns_state_counts("ns"), STATE_KEYS),
        (lambda: ingest._scalar_counts("ns"), SCALAR_KEYS),
        (lambda: ingest._integrity_counts("ns", [1]), INTEGRITY_KEYS),
    ],
)
def test_readers_return_all_minus_one_when_backend_raises(monkeypatch, bound, reader, keys) -> None:
    _patch(monkeypatch, error=RuntimeError("boom"))
    assert reader() == {key: -1 for key in keys}


@pytest.mark.parametrize(
    ("reader", "keys"),
    [
        (lambda: ingest._ns_state_counts("ns"), STATE_KEYS),
        (lambda: ingest._scalar_counts("ns"), SCALAR_KEYS),
        (lambda: ingest._integrity_counts("ns", [1]), INTEGRITY_KEYS),
    ],
)
def test_readers_return_all_minus_one_when_unbound(monkeypatch, reader, keys) -> None:
    monkeypatch.setattr(ingest, "_BUILD_BACKEND", None)
    assert reader() == {key: -1 for key in keys}


def test_failed_projections_raise_graph_unreadable_when_backend_raises(monkeypatch, bound) -> None:
    _patch(monkeypatch, error=RuntimeError("boom"))
    with pytest.raises(ingest.GraphUnreadable):
        ingest._failed_evidence_projections("ns")


def test_failed_projections_classify_drop_blank_and_sort(monkeypatch, bound) -> None:
    marker = ingest._PROVIDER_FAILURE_MARKERS[0]
    rec = _patch(monkeypatch, response=[
        {"uuid": "u2", "processing_error": f"openai: {marker.upper()}"},
        {"uuid": "  ", "processing_error": None},
        {"uuid": "", "processing_error": "x"},
        {"uuid": "u1", "processing_error": "bad content"},
    ])
    assert ingest._failed_evidence_projections("ns") == [("u1", False), ("u2", True)]
    assert rec.calls == [("list_failed_evidence_projections", {"namespace": "ns"})]


def test_reset_namespace_sends_one_forced_reset_op(monkeypatch) -> None:
    rec = _patch(monkeypatch, response={"nodes_deleted": 0, "turn_evidence_deleted": 0})

    class NoDirectPosts:
        def post(self, *a, **k):
            raise AssertionError("reset must not POST /api/phase3/reset itself")

    ingest._reset_namespace(NoDirectPosts(), "http://menhir", "ns")
    assert rec.calls == [("reset_namespace", {"namespace": "ns", "force": True})]


def test_retry_failed_projections_reads_each_namespace_once(monkeypatch, bound) -> None:
    operations: list[tuple[str, dict]] = []

    def fake_backend(admin, url, op, body=None):
        operations.append((op, body))
        if op == "list_failed_evidence_projections":
            if body["namespace"] == "ns-a":
                return [
                    {"uuid": "projection-b", "processing_error": "x"},
                    {"uuid": "projection-a", "processing_error": "x"},
                ]
            return []
        return True

    monkeypatch.setattr(ingest, "_backend", fake_backend)
    requeued = ingest._retry_failed_evidence_projections(
        ["ns-a", "ns-b", "ns-a"], object(), "http://menhir"
    )
    assert requeued == {"ns-a": 2, "ns-b": 0}
    assert operations == [
        ("list_failed_evidence_projections", {"namespace": "ns-a"}),
        ("force_reset_failed_episode", {"episode_uuid": "projection-a"}),
        ("enqueue_pending_episode", {"episode_uuid": "projection-a"}),
        ("force_reset_failed_episode", {"episode_uuid": "projection-b"}),
        ("enqueue_pending_episode", {"episode_uuid": "projection-b"}),
        ("list_failed_evidence_projections", {"namespace": "ns-b"}),
    ]


def test_submitted_epoch_millis_dedupes_and_skips_unparseable() -> None:
    item = {"haystack_dates": ["2023/07/14 (Fri) 08:30", "not a date", "2023/07/14 (Fri) 08:30"]}
    assert ingest._submitted_epoch_millis(item) == [1689323400000]
