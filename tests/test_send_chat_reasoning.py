"""send_chat on reasoning models (gpt-6-luna) and Flex, offline via httpx.MockTransport.

The failures guarded: a request Luna rejects (max_tokens / temperature), a Flex capacity 429
that ends the run or becomes the "answer", and an error or empty reply scored as an answer or
parsed by the judge as "no".
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from archolith_bench.core import api
from archolith_bench.core.api import ChatCallFailed, build_chat_body, send_chat

OPENAI = "https://api.openai.com/v1"
MSGS = [{"role": "user", "content": "q"}]
LME_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "longmemeval_sample.json"


def _ok(text: str = "yes", finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    })


def _client(responses: list, seen: list | None = None) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_reasoning_body_has_no_max_tokens_or_temperature(monkeypatch) -> None:
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    body, timeout = build_chat_body("gpt-6-luna", MSGS, OPENAI)
    assert "max_tokens" not in body and "temperature" not in body
    assert body["max_completion_tokens"] == api.REASONING_MAX_COMPLETION_TOKENS
    assert "service_tier" not in body
    assert timeout == 300.0


def test_non_reasoning_body_unchanged(monkeypatch) -> None:
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "flex")
    body, timeout = build_chat_body("gpt-4o", MSGS, OPENAI)
    assert body["max_tokens"] == 2048 and body["temperature"] == 0.3
    assert "service_tier" not in body and "max_completion_tokens" not in body
    assert timeout == 300.0


def test_flex_tier_only_on_openai_endpoint(monkeypatch) -> None:
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "flex")
    body, timeout = build_chat_body("gpt-6-luna", MSGS, OPENAI)
    assert body["service_tier"] == "flex" and timeout == float(api.FLEX_TIMEOUT_S)
    proxied, proxied_timeout = build_chat_body("gpt-6-luna", MSGS, "http://localhost:9800/v1")
    assert "service_tier" not in proxied and proxied_timeout == 300.0


def test_request_sent_carries_flex_and_long_timeout(monkeypatch) -> None:
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "flex")
    seen: list[httpx.Request] = []
    send_chat(_client([_ok()], seen), OPENAI, "k", MSGS, "gpt-6-luna")
    sent = json.loads(seen[0].content)
    assert sent["service_tier"] == "flex" and "temperature" not in sent
    assert seen[0].extensions["timeout"]["read"] == float(api.FLEX_TIMEOUT_S)


# ---------------------------------------------------------------------------
# 429 handling
# ---------------------------------------------------------------------------

def _capacity_429() -> httpx.Response:
    return httpx.Response(429, json={"error": {"code": "resource_unavailable", "message": "x"}})


def test_flex_capacity_429_is_retried_past_the_rate_limit_budget() -> None:
    responses = [_capacity_429()] * 7 + [_ok("Denver")]
    text, _l, _u = send_chat(_client(responses), OPENAI, "k", MSGS, "gpt-6-luna",
                             max_retries=0, strict=True)
    assert text == "Denver"


def test_flex_capacity_exhausted_raises_when_strict() -> None:
    responses = [_capacity_429()] * (api.FLEX_CAPACITY_RETRIES + 1)
    with pytest.raises(ChatCallFailed, match="429"):
        send_chat(_client(responses), OPENAI, "k", MSGS, "gpt-6-luna", strict=True)


def test_insufficient_quota_is_not_retried() -> None:
    quota = httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
    seen: list[httpx.Request] = []
    text, _l, _u = send_chat(_client([quota, _ok()], seen), OPENAI, "k", MSGS, "gpt-6-luna")
    assert text.startswith("[ERROR 429]") and len(seen) == 1


# ---------------------------------------------------------------------------
# strict: failures raise instead of becoming text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom"),
    httpx.TimeoutException("slow"),
    httpx.ConnectError("down"),
    _ok("", finish="length"),
    _ok("   "),
])
def test_strict_raises_on_any_unusable_reply(response) -> None:
    with pytest.raises(ChatCallFailed):
        send_chat(_client([response]), OPENAI, "k", MSGS, "gpt-6-luna", strict=True)


def test_default_still_returns_error_strings() -> None:
    text, _l, _u = send_chat(_client([httpx.Response(500, text="boom")]), OPENAI, "k", MSGS, "gpt-4o")
    assert text.startswith("[ERROR 500]")


# ---------------------------------------------------------------------------
# Callers that score use strict
# ---------------------------------------------------------------------------

def test_judge_failure_propagates_instead_of_grading_no() -> None:
    from archolith_bench.harness.scoring import LLMJudgeScorer

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        assert kwargs.get("strict") is True
        raise ChatCallFailed("judge down")

    judge = LLMJudgeScorer(base_url=OPENAI, api_key="k", model="gpt-6-luna", send_fn=send_fn)
    with pytest.raises(ChatCallFailed):
        judge({"question": "q", "answer": "a"}, "a")


def test_failed_answer_is_not_checkpointed(tmp_path: Path) -> None:
    from archolith_bench.harness import MemoryCheckpoint, StubMenhirClient, run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    calls = {"n": 0}

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        assert kwargs.get("strict") is True
        calls["n"] += 1
        if calls["n"] == 2:
            raise ChatCallFailed("flex capacity exhausted")
        return "I don't know", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    ckpt = tmp_path / "ckpt.jsonl"
    with pytest.raises(ChatCallFailed):
        run_memory_ab(
            LongMemEvalMemoryAdapter(),
            arms=("no_memory",),
            fixture_path=LME_FIXTURE,
            client=StubMenhirClient(),
            send_fn=send_fn,
            checkpoint=MemoryCheckpoint(ckpt),
            score_fn=lambda item, text: False,
        )
    # Only the item answered before the failure is on record; the failed one is retried on resume.
    assert MemoryCheckpoint(ckpt).done_count() == 1
