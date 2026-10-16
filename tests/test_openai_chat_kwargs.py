"""SDK-side helpers that make analysis tools send Luna-valid requests on Flex."""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

from archolith_bench.core import api
from archolith_bench.core.api import (
    FLEX_TIMEOUT_S,
    REASONING_MAX_COMPLETION_TOKENS,
    acall_with_flex_retry,
    call_with_flex_retry,
    is_flex_capacity_error,
    openai_chat_kwargs,
)

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "longmemeval" / "analysis"


class _SdkError(Exception):
    def __init__(self, status_code: int, body: dict | str) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code
        self.body = body


def _capacity_sdk() -> _SdkError:
    return _SdkError(429, {"code": "resource_unavailable", "message": "Flex capacity"})


def _http_status_error(status: int, text: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, text=text, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_reasoning_model_gets_flex_and_no_temperature(monkeypatch):
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    kwargs = openai_chat_kwargs("gpt-6-luna", max_tokens=900, temperature=0.0)
    assert kwargs == {
        "max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS,
        "service_tier": "flex",
        "timeout": float(FLEX_TIMEOUT_S),
    }


def test_reasoning_model_keeps_a_larger_budget(monkeypatch):
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    kwargs = openai_chat_kwargs("gpt-6-luna", max_tokens=50000)
    assert kwargs["max_completion_tokens"] == 50000


def test_reasoning_model_without_budget_sends_no_ceiling(monkeypatch):
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    kwargs = openai_chat_kwargs("gpt-6-luna", temperature=0)
    assert "max_completion_tokens" not in kwargs
    assert "temperature" not in kwargs


def test_non_reasoning_model_keeps_temperature_and_max_tokens(monkeypatch):
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "flex")
    kwargs = openai_chat_kwargs("gpt-4o", max_tokens=900, temperature=0.0)
    assert kwargs == {"max_tokens": 900, "temperature": 0.0}


def test_empty_tier_env_sends_no_tier(monkeypatch):
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "")
    kwargs = openai_chat_kwargs("gpt-6-luna", max_tokens=10)
    assert "service_tier" not in kwargs
    assert "timeout" not in kwargs


def test_non_openai_base_gets_no_tier(monkeypatch):
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    kwargs = openai_chat_kwargs("gpt-6-luna", max_tokens=10, base_url="http://localhost:8080/v1")
    assert "service_tier" not in kwargs


def test_capacity_error_detection():
    assert is_flex_capacity_error(_capacity_sdk())
    assert not is_flex_capacity_error(_SdkError(429, {"code": "rate_limit_exceeded"}))
    assert not is_flex_capacity_error(_SdkError(429, {"code": "insufficient_quota"}))
    assert not is_flex_capacity_error(_SdkError(500, {"code": "resource_unavailable"}))
    assert is_flex_capacity_error(_http_status_error(429, '{"error":{"code":"resource_unavailable"}}'))
    assert not is_flex_capacity_error(_http_status_error(429, '{"error":{"code":"rate_limit_exceeded"}}'))


def test_call_with_flex_retry_waits_out_capacity():
    calls, waits = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _capacity_sdk()
        return "ok"

    assert call_with_flex_retry(fn, sleep=waits.append) == "ok"
    assert waits == [15, 30]


def test_call_with_flex_retry_raises_rate_limit_immediately():
    waits = []

    def fn():
        raise _SdkError(429, {"code": "rate_limit_exceeded"})

    with pytest.raises(_SdkError):
        call_with_flex_retry(fn, sleep=waits.append)
    assert waits == []


def test_call_with_flex_retry_gives_up_after_budget():
    waits = []

    def fn():
        raise _capacity_sdk()

    with pytest.raises(_SdkError):
        call_with_flex_retry(fn, retries=2, sleep=waits.append)
    assert len(waits) == 2


def test_async_retry_waits_out_capacity():
    calls, waits = [], []

    async def fake_sleep(s):
        waits.append(s)

    async def fn():
        calls.append(1)
        if len(calls) < 2:
            raise _capacity_sdk()
        return "ok"

    assert asyncio.run(acall_with_flex_retry(fn, sleep=fake_sleep)) == "ok"
    assert waits == [15]


def _load_ab(monkeypatch, tmp_path, module_name):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setenv("AB_OUT", str(tmp_path))
    monkeypatch.setenv("BRIEF_OUT_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ANALYSIS_DIR))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    # answer_ab imports the dataset loader at module load; the judge call never touches it.
    monkeypatch.setitem(sys.modules, "entropy", type(sys)("entropy"))
    mod = importlib.import_module(module_name)
    monkeypatch.delitem(sys.modules, module_name)
    return mod


@pytest.mark.parametrize("module_name", ["answer_ab", "brief_ab"])
def test_ab_judge_sends_luna_valid_flex_body(monkeypatch, tmp_path, module_name):
    mod = _load_ab(monkeypatch, tmp_path, module_name)
    assert mod.JUDGE_MODEL == "gpt-6-luna"

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append((json.loads(request.content), request.extensions.get("timeout")))
        if len(sent) == 1:
            return httpx.Response(429, json={"error": {"code": "resource_unavailable"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "yes"}}]})

    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert mod._chat(c, mod.JUDGE_MODEL, "sys", "user") == "yes"
    assert len(sent) == 2
    body, timeout = sent[-1]
    assert body["model"] == "gpt-6-luna"
    assert body["service_tier"] == "flex"
    assert "temperature" not in body and "max_tokens" not in body
    assert "timeout" not in body
    assert timeout["read"] == float(FLEX_TIMEOUT_S)


@pytest.mark.parametrize("module_name", ["answer_ab", "brief_ab"])
def test_ab_stops_on_rate_limit_and_rejects_empty(monkeypatch, tmp_path, module_name):
    mod = _load_ab(monkeypatch, tmp_path, module_name)

    def limited(request):
        return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})

    with httpx.Client(transport=httpx.MockTransport(limited)) as c:
        with pytest.raises(SystemExit):
            mod._chat(c, "gpt-6-luna", "sys", "user")

    def empty(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    with httpx.Client(transport=httpx.MockTransport(empty)) as c:
        with pytest.raises(RuntimeError):
            mod._chat(c, "gpt-6-luna", "sys", "user")
