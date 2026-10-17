"""The drill-down tool loop over /v1/responses, offline via httpx.MockTransport.

gpt-6-luna rejects function tools on /v1/chat/completions unless reasoning is off, so OpenAI
reasoning models use /v1/responses. Guards: the chat-shaped tool sent unflattened, reasoning
items dropped between rounds (the API requires them back), the tool loop never forced to
answer, a quota 429 retried, an incomplete reply graded, and the wrong loop picked for a model.
"""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

from archolith_bench.core import api
from archolith_bench.core.api import ChatCallFailed, send_responses_with_tools, send_with_tools
from archolith_bench.harness.drill_down import DRILL_DOWN_TOOL

OPENAI = "https://api.openai.com/v1"
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
REASONING = {"type": "reasoning", "id": "rs_1", "summary": []}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)


def _call(arguments: str = '{"view_uuid": "v-1"}', call_id: str = "call_1") -> httpx.Response:
    return httpx.Response(200, json={
        "status": "completed",
        "output": [REASONING, {"type": "function_call", "id": "fc_1", "call_id": call_id,
                               "name": "get_scalar_view_items", "arguments": arguments}],
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    })


def _answer(text: str = "Three") -> httpx.Response:
    return httpx.Response(200, json={
        "status": "completed",
        "output": [REASONING, {"type": "message", "role": "assistant",
                               "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 20, "output_tokens": 2, "total_tokens": 22},
    })


def _client(responses: list, seen: list) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_runs_the_call_passes_reasoning_back_and_answers(monkeypatch) -> None:
    monkeypatch.setenv(api.SERVICE_TIER_ENV, "flex")
    seen: list = []
    ran: list = []

    def run_tool(name, arguments):  # noqa: ANN001
        ran.append((name, arguments))
        return '{"total": 3}'

    text, _latency, usage, calls = send_responses_with_tools(
        _client([_call(), _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
        tools=[DRILL_DOWN_TOOL], run_tool=run_tool,
    )
    assert text == "Three"
    assert ran == [("get_scalar_view_items", {"view_uuid": "v-1"})]
    path, first = seen[0]
    assert path == "/v1/responses"
    assert first["tools"] == [{"type": "function", "strict": False, **DRILL_DOWN_TOOL["function"]}]
    assert first["input"] == MSGS and "tool_choice" not in first
    assert first["service_tier"] == "flex"
    assert first["max_output_tokens"] == api.REASONING_MAX_COMPLETION_TOKENS
    assert "reasoning_effort" not in first and "reasoning" not in first
    followup = seen[1][1]["input"]
    assert followup[:2] == MSGS
    assert followup[2] == REASONING and followup[3]["type"] == "function_call"
    assert followup[4] == {"type": "function_call_output", "call_id": "call_1", "output": '{"total": 3}'}
    assert usage == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36, "tool_rounds": 2}
    assert calls == [{"name": "get_scalar_view_items", "arguments": {"view_uuid": "v-1"},
                      "result": '{"total": 3}'}]


def test_last_round_forbids_tools() -> None:
    seen: list = []
    send_responses_with_tools(
        _client([_call(), _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
        tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}", max_rounds=2,
    )
    assert seen[1][1]["tool_choice"] == "none"


def test_never_answering_fails_strictly() -> None:
    with pytest.raises(ChatCallFailed, match="still calling tools"):
        send_responses_with_tools(
            _client([_call(), _call()], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}", max_rounds=2,
        )


def test_incomplete_reply_is_not_graded() -> None:
    incomplete = httpx.Response(200, json={
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [REASONING], "usage": {"input_tokens": 5, "output_tokens": 9, "total_tokens": 14},
    })
    with pytest.raises(ChatCallFailed, match="incomplete"):
        send_responses_with_tools(
            _client([incomplete], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
        )


def test_incomplete_reply_with_partial_text_is_not_graded() -> None:
    truncated = httpx.Response(200, json={
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [REASONING, {"type": "message", "role": "assistant",
                               "content": [{"type": "output_text", "text": "The answer is par"}]}],
        "usage": {},
    })
    with pytest.raises(ChatCallFailed, match="not completed.*max_output_tokens"):
        send_responses_with_tools(
            _client([truncated], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
        )


def test_truncated_function_call_is_not_run() -> None:
    truncated = httpx.Response(200, json={
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1",
                    "name": "get_scalar_view_items", "arguments": '{"view_uuid": "v-'}],
        "usage": {},
    })
    ran: list = []
    with pytest.raises(ChatCallFailed, match="not completed"):
        send_responses_with_tools(
            _client([truncated, _answer()], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda n, a: ran.append(a) or "{}",
        )
    assert ran == []


def test_failed_response_reports_the_api_error() -> None:
    failed = httpx.Response(200, json={
        "status": "failed", "error": {"code": "server_error", "message": "boom"},
        "output": [], "usage": {},
    })
    with pytest.raises(ChatCallFailed, match="server_error"):
        send_responses_with_tools(
            _client([failed], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
        )


def test_server_error_status_fails_strictly() -> None:
    with pytest.raises(ChatCallFailed, match=r"\[ERROR 500\]"):
        send_responses_with_tools(
            _client([httpx.Response(500, text="oops")], []), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}", max_retries=0,
        )


def test_parallel_calls_all_run_and_return_in_order(monkeypatch) -> None:
    monkeypatch.delenv(api.SERVICE_TIER_ENV, raising=False)
    two = httpx.Response(200, json={
        "status": "completed",
        "output": [REASONING,
                   {"type": "function_call", "id": "fc_1", "call_id": "call_a",
                    "name": "get_scalar_view_items", "arguments": '{"view_uuid": "v-1"}'},
                   {"type": "function_call", "id": "fc_2", "call_id": "call_b",
                    "name": "get_scalar_view_items", "arguments": '{"view_uuid": "v-2"}'}],
        "usage": {},
    })
    seen: list = []
    text, *_ = send_responses_with_tools(
        _client([two, _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
        tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, a: json.dumps({"for": a["view_uuid"]}),
    )
    assert text == "Three"
    assert "service_tier" not in seen[0][1]
    outputs = [item for item in seen[1][1]["input"] if item.get("type") == "function_call_output"]
    assert [(o["call_id"], json.loads(o["output"])["for"]) for o in outputs] == [
        ("call_a", "v-1"), ("call_b", "v-2")]


def test_bad_arguments_go_back_without_running_the_tool() -> None:
    seen: list = []
    ran: list = []
    send_responses_with_tools(
        _client([_call("not json"), _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
        tools=[DRILL_DOWN_TOOL], run_tool=lambda n, a: ran.append(a) or "{}",
    )
    assert ran == []
    assert "error" in json.loads(seen[1][1]["input"][-1]["output"])


def test_insufficient_quota_is_not_retried() -> None:
    seen: list = []
    quota = httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
    with pytest.raises(ChatCallFailed):
        send_responses_with_tools(
            _client([quota, _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
        )
    assert len(seen) == 1


@pytest.mark.parametrize("model, base_url, path", [
    ("gpt-6-luna", OPENAI, "/v1/responses"),
    ("gpt-4o", OPENAI, "/v1/chat/completions"),
    ("gpt-6-luna", "http://localhost:9800/v1", "/v1/chat/completions"),
])
def test_send_with_tools_picks_the_loop(model, base_url, path) -> None:
    seen: list = []
    chat_answer = httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {},
    })
    response = _answer("ok") if path == "/v1/responses" else chat_answer
    text, *_ = send_with_tools(
        _client([response], seen), base_url, "k", MSGS, model,
        tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
    )
    assert text == "ok" and seen[0][0] == path


def test_memory_harness_uses_the_switch_by_default() -> None:
    from archolith_bench.harness.memory_ab import _run_memory_arm, run_memory_ab

    for fn in (run_memory_ab, _run_memory_arm):
        assert inspect.signature(fn).parameters["tool_send_fn"].default is send_with_tools
