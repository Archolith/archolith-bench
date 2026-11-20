"""MenhirSystem against a recording fake backend (no Menhir import, no network)."""

from __future__ import annotations

import asyncio

import pytest

from archolith_bench.staged.menhir_system import MenhirSystem


class FakeBackend:
    def __init__(self, states: dict[str, list[str]] | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.states = states or {}

    def _rec(self, name, a, kw):
        self.calls.append((name, a, kw))

    async def recall(self, *a, **kw):
        self._rec("recall", a, kw)
        return {"results": []}

    async def build_context(self, *a, **kw):
        self._rec("build_context", a, kw)
        return {"context": "ctx"}

    async def delete_namespace(self, *a, **kw):
        self._rec("delete_namespace", a, kw)
        return {"ok": True}

    async def record_turn_evidence(self, *a, **kw):
        self._rec("record_turn_evidence", a, kw)
        return {"turn_id": "turn-1"}

    async def queue_episode(self, *a, **kw):
        self._rec("queue_episode", a, kw)
        return {"episode_id": "ep-1"}

    async def fetch_episode_processing(self, eid):
        seq = self.states.get(eid, ["READY"])
        state = seq.pop(0) if len(seq) > 1 else seq[0]
        return {"processing_state": state}


def run(coro):
    return asyncio.run(coro)


def test_search_kwargs() -> None:
    b = FakeBackend()
    run(MenhirSystem(b).search("ns", "q", limit=7))
    assert b.calls == [("recall", ("q",), dict(
        limit=7, limit_is_explicit=True, include_session=True, include_invalidated=True,
        namespace="ns", counted_verdict_limit=None, trace=False))]


def test_search_counted_authority_caps_limit() -> None:
    b = FakeBackend()
    run(MenhirSystem(b).search("ns", "q", limit=7, counted_authority_only=True, counted_verdict_limit=99, trace=True))
    kw = b.calls[0][2]
    assert kw["counted_verdict_limit"] == 3 and kw["trace"] is True


def test_context_and_reset() -> None:
    b = FakeBackend()
    s = MenhirSystem(b)
    run(s.context("ns", "q", max_tokens=100, preset="knowledge"))
    run(s.reset("ns"))
    assert b.calls[0] == ("build_context", ("q",), dict(max_tokens=100, preset="knowledge", namespace="ns"))
    assert b.calls[1] == ("delete_namespace", ("ns",), dict(force=True))


def test_add_user_turn_records_evidence_first() -> None:
    b = FakeBackend()
    out = run(MenhirSystem(b).add(
        "ns", text="hi", role="user", session_id="s1", occurred_at="2023-01-01",
        idempotency_key="k", turn_key="tk"))
    assert out == {"episode_id": "ep-1"}
    (n1, a1, k1), (n2, a2, k2) = b.calls
    assert n1 == "record_turn_evidence" and a1 == ("hi",)
    assert k1 == dict(role="user", declarant="user", namespace="ns", session_id="s1",
                      occurred_at="2023-01-01", source_kind="archolith-bench",
                      source_client="archolith_bench", turn_key="tk")
    assert n2 == "queue_episode" and a2 == ("hi",)
    assert k2 == dict(user_id="archolith-bench", session_id="s1", source="user", namespace="ns",
                      occurred_at="2023-01-01", turn_evidence_uuid="turn-1", role="user",
                      idempotency_key="k")


def test_add_assistant_turn_no_evidence() -> None:
    b = FakeBackend()
    run(MenhirSystem(b).add("ns", text="yo", role="assistant", session_id="s1",
                            idempotency_key="k", turn_key="tk"))
    assert [c[0] for c in b.calls] == ["queue_episode"]
    kw = b.calls[0][2]
    assert kw["turn_evidence_uuid"] is None and kw["source"] == "remote-api" and kw["role"] == "assistant"


def test_wait_indexed_ready_and_failed() -> None:
    b = FakeBackend({"a": ["PENDING", "ready"], "b": ["Failed"]})
    t = [0.0]

    async def sleep(s):
        t[0] += s

    out = run(MenhirSystem(b).wait_indexed(["a", "b"], clock=lambda: t[0], sleep=sleep))
    assert out == {"a": "READY", "b": "FAILED"}


def test_wait_indexed_timeout_lists_pending() -> None:
    b = FakeBackend({"a": ["ENRICHING"], "b": ["READY"]})
    t = [0.0]

    async def sleep(s):
        t[0] += s

    with pytest.raises(TimeoutError, match="a") as ei:
        run(MenhirSystem(b).wait_indexed(["a", "b"], timeout_s=10, poll_s=2, clock=lambda: t[0], sleep=sleep))
    assert "b" not in str(ei.value).split(":", 1)[1]
