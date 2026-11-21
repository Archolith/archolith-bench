"""MenhirSystem ingest requests are digest-identical to HttpMenhirClient's REST requests (no network)."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from archolith_bench.harness.menhir_client import HttpMenhirClient
from archolith_bench.staged.menhir_system import MenhirSystem

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_menhir_system_test", "scripts/longmemeval/lib/ingest.py")
memory_writer = sys.modules["memory_writer"]


class FakeBackend:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.error = error

    async def record_turn_evidence(self, *a, **kw):
        self.calls.append(("record_turn_evidence", a, kw))
        return {"turn_id": "turn-1"}

    async def queue_episode(self, *a, **kw):
        self.calls.append(("queue_episode", a, kw))
        if self.error is not None:
            raise self.error
        return {"episode_id": "ep-1"}

    async def reset_namespace(self, namespace, *, force=False):
        self.calls.append(("reset_namespace", (namespace,), {"force": force}))
        return {"ok": True}


def run(coro):
    return asyncio.run(coro)


def _rest_payload(role: str, content: str, **kw) -> dict:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"episode_id": "e1"})

    client = HttpMenhirClient("http://menhir.test")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    client.ingest("ns", role, content, wait=False, **kw)
    return captured[0]


def _digest_tuple_rest(payload: dict) -> tuple:
    return (
        payload["episode"],
        payload.get("source", "remote-api"),
        payload.get("role"),
        payload.get("turn_evidence_uuid"),
        payload.get("occurred_at"),
        bool(payload.get("store_only", False)),
    )


def _digest_tuple_backend(call: tuple) -> tuple:
    _, args, kw = call
    return (
        args[0],
        kw["source"],
        kw.get("role"),
        kw.get("turn_evidence_uuid"),
        kw.get("occurred_at"),
        bool(kw.get("store_only", False)),
    )


CASES = {
    "user_with_evidence": ("user", "hi there", dict(source="user", turn_evidence_uuid="t1", occurred_at="2023-01-01T00:00:00+00:00", idempotency_key="k")),
    "user_without_evidence": ("user", "hi there", dict(source="user", occurred_at="2023-01-01T00:00:00+00:00")),
    "store_only_user": ("user", "what should I cook?", dict(source="user", turn_evidence_uuid="t1", store_only=True, idempotency_key="k")),
    "store_only_assistant": ("assistant", "sure", dict(turn_evidence_uuid="t2", store_only=True, memory_role="assistant", idempotency_key="k")),
    "raw_assistant": ("assistant", "sure", dict(turn_evidence_uuid="t2", memory_role="assistant", idempotency_key="k")),
    "memory_writer": ("assistant", "user likes tea", dict(source=memory_writer.WRITER_SOURCE, turn_evidence_uuid="t3", memory_role=memory_writer.WRITER_ROLE, idempotency_key="k")),
    "source_omitted": ("assistant", "ok", dict()),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_ingest_matches_rest_request(case: str) -> None:
    role, content, kw = CASES[case]
    backend = FakeBackend()
    out = run(MenhirSystem(backend, user_id="remote-api").ingest("ns", role, content, **kw))
    assert out == {"episode_id": "ep-1"}
    assert [c[0] for c in backend.calls] == ["queue_episode"]
    call = backend.calls[0]
    assert _digest_tuple_backend(call) == _digest_tuple_rest(_rest_payload(role, content, **kw))
    sent = call[2]
    assert sent["user_id"] == "remote-api" and sent["namespace"] == "ns"
    assert sent["session_id"]
    assert sent["source"] != "mcp"
    assert ("idempotency_key" in sent) == ("idempotency_key" in kw)
    assert ("store_only" in sent) == bool(kw.get("store_only"))


def test_exact_backend_call_shapes() -> None:
    backend = FakeBackend()
    s = MenhirSystem(backend, user_id="remote-api")
    run(s.ingest("ns", "user", "hi", source="user", turn_evidence_uuid="t1", session_id="s1",
                 occurred_at="2023-01-01", idempotency_key="k"))
    run(s.ingest("ns", "user", "hi", session_id="s1"))
    run(s.ingest("ns", "user", "q?", source="user", turn_evidence_uuid="t1", session_id="s1", store_only=True))
    assert backend.calls[0] == ("queue_episode", ("hi",), dict(
        user_id="remote-api", session_id="s1", source="user", namespace="ns", occurred_at="2023-01-01",
        turn_evidence_uuid="t1", idempotency_key="k"))
    assert backend.calls[1] == ("queue_episode", ("user: hi",), dict(
        user_id="remote-api", session_id="s1", source="remote-api", namespace="ns", occurred_at=None,
        turn_evidence_uuid=None))
    assert backend.calls[2][2]["store_only"] is True and "role" not in backend.calls[2][2]


def test_empty_content_returns_none_without_a_call() -> None:
    backend = FakeBackend()
    assert run(MenhirSystem(backend).ingest("ns", "user", "")) is None
    assert backend.calls == []


def test_record_turn_evidence_uses_bench_values() -> None:
    backend = FakeBackend()
    out = run(MenhirSystem(backend).record_turn_evidence(
        "ns", "hello", role="assistant", declarant="assistant", session_id="s1",
        occurred_at="2023-01-01", turn_key="tk"))
    assert out == {"turn_id": "turn-1"}
    assert backend.calls == [("record_turn_evidence", ("hello",), dict(
        role="assistant", declarant="assistant", namespace="ns", session_id="s1",
        occurred_at="2023-01-01", source_kind="archolith-bench", source_client="archolith_bench",
        turn_key="tk"))]


def _facade(error: Exception | None) -> "ingest.SyncMenhir":
    return ingest.SyncMenhir(MenhirSystem(FakeBackend(error), user_id="remote-api"), "http://localhost:8102")


def test_facade_passes_transport_errors_through() -> None:
    facade = _facade(httpx.ConnectError("boom"))
    try:
        with pytest.raises(httpx.HTTPError):
            facade.ingest("ns", "assistant", "hi")
    finally:
        facade.close()


def test_facade_translates_conflict_to_409() -> None:
    facade = _facade(ValueError("idempotency_key was already used with a different request"))
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            facade.ingest("ns", "assistant", "hi", idempotency_key="k")
        assert caught.value.response.status_code == 409
    finally:
        facade.close()


def test_facade_translates_other_refusals() -> None:
    facade = _facade(ValueError("bad request"))
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            facade.ingest("ns", "assistant", "hi")
        assert caught.value.response.status_code == 400
    finally:
        facade.close()
    facade = _facade(PermissionError("nope"))
    try:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            facade.ingest("ns", "assistant", "hi")
        assert caught.value.response.status_code == 403
    finally:
        facade.close()


def test_ingest_turn_raises_idempotency_conflict() -> None:
    facade = _facade(ValueError("idempotency_key was already used with a different request"))
    try:
        with pytest.raises(ingest.IdempotencyConflictError):
            ingest._ingest_turn(facade, "ns", "assistant", "hi", ingest_key="k", tries=2)
    finally:
        facade.close()


def test_facade_ingest_returns_episode() -> None:
    facade = _facade(None)
    try:
        assert ingest._ingest_turn(facade, "ns", "assistant", "hi", ingest_key="k") == "ep-1"
    finally:
        facade.close()


def test_backend_dispatches_to_facade_call(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Bound:
        def call(self, op, body):
            calls.append((op, body))
            return {"ok": op}

    monkeypatch.setattr(ingest, "_FACADE", Bound())
    assert ingest._backend(object(), "http://localhost:8102", "reset_namespace", {"namespace": "n", "force": True}) == {
        "ok": "reset_namespace"}
    assert calls == [("reset_namespace", {"namespace": "n", "force": True})]


def test_facade_call_invokes_backend_op() -> None:
    backend = FakeBackend()
    facade = ingest.SyncMenhir(MenhirSystem(backend), "http://localhost:8102")
    try:
        assert facade.call("reset_namespace", {"namespace": "n", "force": True}) == {"ok": True}
    finally:
        facade.close()
    assert backend.calls == [("reset_namespace", ("n",), {"force": True})]


def test_parser_keeps_old_args_and_menhir_root_falls_back(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MENHIR_FRONTIER", raising=False)
    monkeypatch.delenv("MENHIR_MAIN", raising=False)
    args = ingest._parse_args([
        "--limit", "3", "--menhir-url", "http://localhost:8102", "--manifest", str(tmp_path / "m.json"),
        "--segmentation", "none"])
    assert args.menhir_url == "http://localhost:8102" and args.limit == 3 and args.menhir_root is None
    monkeypatch.setenv("MENHIR_MAIN", "/main")
    assert ingest._parse_args([]).menhir_root == "/main"
    monkeypatch.setenv("MENHIR_FRONTIER", "/frontier")
    assert ingest._parse_args([]).menhir_root == "/frontier"
    assert ingest._parse_args(["--menhir-root", "/explicit"]).menhir_root == "/explicit"


def test_idempotency_conflict_text_matches_menhir() -> None:
    """The facade maps Menhir's conflict to 409 by its message (the backend route has no 409)."""
    import os

    root = os.getenv("MENHIR_ROOT") or os.getenv("MENHIR_FRONTIER") or os.getenv("MENHIR_MAIN")
    source = Path(root or "../menhir") / "src" / "menhir" / "services" / "ingest_intake.py"
    if not source.is_file():
        pytest.skip("Menhir checkout not available (set MENHIR_ROOT)")
    assert ingest._IDEMPOTENCY_CONFLICT_DETAIL in source.read_text(encoding="utf-8")
