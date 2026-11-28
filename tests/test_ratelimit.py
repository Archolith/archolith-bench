"""The one 429/error classifier and the callers wired to it (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from archolith_bench.core.opencode.process import parse_events
from archolith_bench.core import api
from archolith_bench.core.api import call_with_flex_retry, is_flex_capacity_error, send_chat
from archolith_bench.core.openai_flex import run_flex
from archolith_bench.core.ratelimit import Kind, classify, mentions_capacity

OPENAI = "https://api.openai.com/v1"
MSGS = [{"role": "user", "content": "q"}]


@pytest.mark.parametrize(("status", "text", "kind"), [
    (429, "flex_unavailable", Kind.CAPACITY),
    (429, "resource_unavailable", Kind.CAPACITY),
    (429, "Resource unavailable", Kind.CAPACITY),
    (429, "Flex does not have sufficient resources", Kind.CAPACITY),
    (500, "resource_unavailable", Kind.OTHER),
    (429, "insufficient_quota", Kind.QUOTA),
    (429, "rate_limit_exceeded", Kind.RATE_LIMIT),
    (402, "", Kind.BILLING),
    (429, "credit_balance_too_low", Kind.BILLING),
    (401, "", Kind.AUTH),
    (403, "", Kind.AUTH),
    (200, "", Kind.OTHER),
])
def test_classify_table(status: int, text: str, kind: Kind) -> None:
    assert classify(status, text) is kind


def test_mentions_capacity_is_case_insensitive() -> None:
    assert mentions_capacity("429 Resource Unavailable")
    assert not mentions_capacity("429 rate_limit_exceeded")


class _SdkError(Exception):
    def __init__(self, status_code: int, body: dict | str) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.body = body


def _flex_sdk() -> _SdkError:
    return _SdkError(429, {"code": "flex_unavailable", "message": "Flex capacity"})


def _http_status_error(status: int, text: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, text=text, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_is_flex_capacity_error_accepts_flex_unavailable() -> None:
    assert is_flex_capacity_error(_flex_sdk())
    assert is_flex_capacity_error(_http_status_error(429, '{"error":{"code":"flex_unavailable"}}'))
    assert not is_flex_capacity_error(_http_status_error(500, '{"error":{"code":"flex_unavailable"}}'))


def test_call_with_flex_retry_retries_flex_unavailable() -> None:
    calls, waits = [], []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise _flex_sdk()
        return "ok"

    assert call_with_flex_retry(fn, sleep=waits.append) == "ok"
    assert waits == [15]


def _ok() -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    })


def _client(responses: list, seen: list | None = None) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_send_chat_does_not_retry_a_billing_429(monkeypatch) -> None:
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)
    billing = httpx.Response(429, json={"error": {"code": "credit_balance_too_low"}})
    seen: list[httpx.Request] = []
    text, _l, _u = send_chat(_client([billing, _ok()], seen), OPENAI, "k", MSGS, "gpt-6-luna")
    assert text.startswith("[ERROR 429]") and len(seen) == 1


def test_send_chat_retries_flex_unavailable_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)
    cap = httpx.Response(429, json={"error": {"code": "flex_unavailable"}})
    seen: list[httpx.Request] = []
    text, _l, _u = send_chat(_client([cap, _ok()], seen), OPENAI, "k", MSGS, "gpt-6-luna")
    assert text == "yes" and len(seen) == 2


def test_run_flex_retries_flex_unavailable(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, content=json.dumps({"error": {"code": "flex_unavailable"}}).encode())
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    _, summary = run_flex(
        [{"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "hello"}]}}],
        model="gpt-6-luna",
        checkpoint=tmp_path / "cp.jsonl",
        api_key="sk-secret",
        max_usd=10.0,
        rates=(0.40, 1.60),
        client=client,
        sleep=lambda _s: None,
        log=lambda _m: None,
    )
    assert attempts["n"] == 3
    assert summary["completed"] == 1


def test_event_log_treats_capacity_as_not_rate_limited() -> None:
    assert not parse_events('{"error": "429 flex_unavailable"}')["rate_limited"]
    assert not parse_events('{"error": "429 Resource unavailable"}')["rate_limited"]
    assert parse_events('{"error": "429 rate_limit_exceeded"}')["rate_limited"]
