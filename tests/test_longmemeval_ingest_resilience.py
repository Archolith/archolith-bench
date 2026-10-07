"""One transient failure must not throw away a build: consolidation retry and per-item tolerance."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.test_longmemeval_ingest_rolling import Harness, _item, _settled, ingest

OK = {
    "namespace": "lme-test", "scalar_enabled": True,
    "scalar_namespaces_processed": 1, "scalar_llm_calls": 3, "llm_calls": 5,
    "counter_enabled": False, "namespaces_processed": 0, "dirty_after": False,
    "event_history_enabled": True, "event_namespaces_failed": 0,
    "event_namespaces_processed": 1, "event_dirty_after": False, "event_llm_calls": 2,
}
# What Phase 3 returned for lme-gpt4_d9af6064: scalar finished, event lane hit a provider 500.
EVENT_FAILED = {**OK, "event_namespaces_failed": 1, "event_dirty_after": True,
                "event_namespaces_processed": 0, "event_llm_calls": 1, "llm_calls": 4}
# The re-run: scalar cursor already at the end, so no scalar work; the event lane finishes.
RERUN_OK = {**OK, "scalar_namespaces_processed": 0, "scalar_llm_calls": 0,
            "llm_calls": 2, "event_llm_calls": 2}


def _consolidate(responses, *, attempts=3, counter=False):
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        response = responses[min(len(sent), len(responses)) - 1]
        return response if isinstance(response, httpx.Response) else httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        try:
            return ingest._consolidate_lanes(
                client, "http://localhost:8124", "lme-test", k=3, call_budget=50,
                scalar_state=True, counter_state=counter, event_history=True,
                attempts=attempts, retry_wait_s=(0.0,),
            ), sent
        except Exception as exc:
            exc.sent = sent
            raise


def test_failed_event_lane_is_retried_and_finished_lanes_are_not_required_again() -> None:
    result, sent = _consolidate([EVENT_FAILED, RERUN_OK])
    assert len(sent) == 2
    assert result["consolidation_attempts"] == 2
    assert result["event_dirty_after"] is False and result["event_namespaces_failed"] == 0
    assert result["scalar_namespaces_processed"] == 1
    assert result["scalar_llm_calls"] == 3 and result["llm_calls"] == 6 and result["event_llm_calls"] == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_status_is_retried(status) -> None:
    result, sent = _consolidate([httpx.Response(status), OK])
    assert len(sent) == 2 and result["event_dirty_after"] is False


def test_transport_error_is_retried() -> None:
    calls = []

    def respond(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json=OK)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = ingest._consolidate_lanes(
            client, "http://x", "lme-test", k=3, call_budget=50, scalar_state=True,
            event_history=True, attempts=2, retry_wait_s=(0.0,),
        )
    assert len(calls) == 2 and result == {**OK, "consolidation_attempts": 2}


def test_single_attempt_returns_the_result_unchanged() -> None:
    result, sent = _consolidate([OK])
    assert result == OK and len(sent) == 1


def test_persistent_failure_raises_after_the_attempt_limit() -> None:
    with pytest.raises(RuntimeError, match="event namespace failed") as info:
        _consolidate([EVENT_FAILED], attempts=3)
    assert len(info.value.sent) == 3


def test_persistent_http_error_raises_after_the_attempt_limit() -> None:
    with pytest.raises(httpx.HTTPStatusError) as info:
        _consolidate([httpx.Response(500)], attempts=2)
    assert len(info.value.sent) == 2


@pytest.mark.parametrize("bad", [
    {**OK, "event_history_enabled": False},
    {**OK, "scalar_enabled": False},
    {**OK, "namespace": "lme-other"},
])
def test_configuration_errors_are_never_retried(bad) -> None:
    with pytest.raises(RuntimeError) as info:
        _consolidate([bad, OK])
    assert len(info.value.sent) == 1


def test_client_error_is_never_retried() -> None:
    with pytest.raises(httpx.HTTPStatusError) as info:
        _consolidate([httpx.Response(400), OK])
    assert len(info.value.sent) == 1


def test_parser_defaults_retry_consolidation_and_keep_fail_fast(monkeypatch) -> None:
    monkeypatch.delenv("LME_CONSOLIDATION_ATTEMPTS", raising=False)
    monkeypatch.delenv("LME_MAX_ITEM_FAILURES", raising=False)
    args = ingest._parse_args(["--limit", "1"])
    assert args.consolidation_attempts == 3 and args.max_item_failures == 0
    monkeypatch.setenv("LME_CONSOLIDATION_ATTEMPTS", "5")
    monkeypatch.setenv("LME_MAX_ITEM_FAILURES", "2")
    args = ingest._parse_args(["--limit", "1"])
    assert args.consolidation_attempts == 5 and args.max_item_failures == 2


def test_tolerated_item_failure_lets_the_rest_of_the_build_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1"), _item("b", "b1"), _item("c", "c1"), _item("d", "d1")]
    h = Harness(monkeypatch, items)

    def drain(namespace):
        if namespace == "lme-b":
            raise RuntimeError("boom")
        return _settled()

    h.drain = drain
    with pytest.raises(RuntimeError, match=r"1 item\(s\) left unmanifested.*lme-b: boom"):
        h.run(tmp_path, "--namespace-window", "2", "--max-item-failures", "1")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(row["question_id"] for row in manifest) == ["a", "c", "d"]


def test_failures_beyond_the_tolerance_stop_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item(q, f"{q}1") for q in "abcdef"]
    h = Harness(monkeypatch, items)

    def drain(namespace):
        if namespace in {"lme-a", "lme-b"}:
            raise RuntimeError(f"boom {namespace}")
        return _settled()

    h.drain = drain
    with pytest.raises(RuntimeError, match="boom lme-"):
        h.run(tmp_path, "--namespace-window", "1", "--max-item-failures", "1")

    assert "settle:lme-c" not in h.events
    assert not (tmp_path / "manifest.json").exists()


def test_resume_retries_only_the_unmanifested_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [_item("a", "a1"), _item("b", "b1"), _item("c", "c1")]
    h = Harness(monkeypatch, items)
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"question_id": "a", "namespace": "lme-a"}, {"question_id": "c", "namespace": "lme-c"},
    ]), encoding="utf-8")

    manifest = h.run(tmp_path, "--namespace-window", "2", "--max-item-failures", "1")

    assert sorted(row["question_id"] for row in manifest) == ["a", "b", "c"]
    assert [e for e in h.events if e.startswith("reset:")] == ["reset:lme-b"]
