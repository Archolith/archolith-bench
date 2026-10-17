"""--drill-down: Menhir's collapsed-View hint, the get_scalar_view_items tool, and its prompt note.

Offline (httpx.MockTransport and fakes). Guards: the hint leaking into runs without the tool,
the model reading another namespace or an unshown View, a tool loop that never answers, a quota
429 retried, and items without a collapsed View answered differently than before.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from archolith_bench.core import api
from archolith_bench.core.api import ChatCallFailed, send_chat_with_tools
from archolith_bench.harness.drill_down import (
    DRILL_DOWN_INSTRUCTION,
    DRILL_DOWN_TOOL,
    drill_down_view_ids,
    make_drill_down_runner,
)
from archolith_bench.harness.menhir_client import HttpMenhirClient, _format_authority_record

OPENAI = "https://api.openai.com/v1"
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
LME_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "longmemeval_sample.json"
HINT = "- built from 3 items; drill down: get_scalar_view_items(view_uuid=v-1)"


def _collapsed(**overrides) -> dict:
    record = {
        "kind": "current", "status": "advisory", "attribute": "movie_watch_count", "scope": "",
        "value": 3, "valid_at": "2023-05-21T13:46:00Z", "view_uuid": "v-1",
        "contributors": [], "contributors_total": 3, "contributors_truncated": True,
    }
    record.update(overrides)
    return record


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda _s: None)


# --- rendering -------------------------------------------------------------------------------

def test_hint_rendered_only_with_drill_down_on_a_collapsed_view() -> None:
    assert HINT in _format_authority_record(_collapsed(), drill_down=True)
    assert "drill down" not in _format_authority_record(_collapsed())


@pytest.mark.parametrize("overrides", [
    {"contributors_truncated": False},
    {"contributors_total": 0},
    {"view_uuid": None},
    {"contributors": [{"stated_span": "saw Toy Story 4", "operation": "occurrence"}]},
])
def test_hint_follows_menhirs_condition(overrides) -> None:
    assert "drill down" not in _format_authority_record(_collapsed(**overrides), drill_down=True)


def _http_client(handler, *, drill_down: bool) -> HttpMenhirClient:
    client = HttpMenhirClient("http://throwaway-menhir.local", api_key="tok", drill_down=drill_down)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_recall_shows_the_hint_and_the_view_id_is_extractable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "authority_layer": [_collapsed()]})

    snippets = _http_client(handler, drill_down=True).recall("lme-x", "how many?")
    assert drill_down_view_ids("\n".join(snippets)) == ["v-1"]
    plain = _http_client(handler, drill_down=False).recall("lme-x", "how many?")
    assert drill_down_view_ids("\n".join(plain)) == []


def test_get_scalar_view_items_hits_the_rest_route() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"view_uuid": "v-1", "total": 3, "items": []})

    page = _http_client(handler, drill_down=True).get_scalar_view_items("lme-x", "v-1", offset=20, limit=5)
    assert page["total"] == 3
    request = seen[0]
    assert request.method == "GET"
    assert request.url.path == "/api/scalar-views/v-1/items"
    assert dict(request.url.params) == {"namespace": "lme-x", "offset": "20", "limit": "5"}
    assert request.headers["Authorization"] == "Bearer tok"


# --- tool runner -----------------------------------------------------------------------------

class _ItemsClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def get_scalar_view_items(self, namespace, view_uuid, *, offset=0, limit=20):  # noqa: ANN001
        self.calls.append((namespace, view_uuid, offset, limit))
        return {
            "view_uuid": view_uuid, "kind": "contributors", "total": 3, "offset": offset,
            "next_offset": None,
            "items": [{"assertion_id": "a1", "episode_uuid": "e1", "relation": "CONTRIBUTED_TO",
                       "operation": "occurrence", "value": 1, "quote": "saw Toy Story 4",
                       "valid_at": "2023-05-21T13:46:00Z", "evidence_tier": None}],
        }


def test_runner_serves_only_shown_views_in_the_items_namespace() -> None:
    client = _ItemsClient()
    run_tool = make_drill_down_runner(client, "lme-x", ["v-1"])

    refused = json.loads(run_tool("get_scalar_view_items", {"view_uuid": "v-other"}))
    assert "error" in refused and client.calls == []
    assert "error" in json.loads(run_tool("delete_everything", {"view_uuid": "v-1"}))

    page = json.loads(run_tool("get_scalar_view_items", {"view_uuid": "v-1", "limit": 500, "offset": -3}))
    assert client.calls == [("lme-x", "v-1", 0, 50)]
    assert page["total"] == 3
    assert page["items"] == [{"relation": "CONTRIBUTED_TO", "operation": "occurrence", "value": 1,
                              "quote": "saw Toy Story 4", "valid_at": "2023-05-21T13:46:00Z"}]


# --- tool loop -------------------------------------------------------------------------------

def _tool_call(arguments: str, call_id: str = "call_1") -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "get_scalar_view_items", "arguments": arguments},
        }]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    })


def _answer(text: str = "Three") -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    })


def _chat(responses: list, seen: list) -> httpx.Client:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_tool_loop_runs_the_call_and_returns_the_answer() -> None:
    seen: list[dict] = []
    ran: list[tuple] = []

    def run_tool(name, arguments):  # noqa: ANN001
        ran.append((name, arguments))
        return '{"total": 3}'

    text, _latency, usage, calls = send_chat_with_tools(
        _chat([_tool_call('{"view_uuid": "v-1"}'), _answer()], seen), OPENAI, "k", MSGS,
        "gpt-6-luna", tools=[DRILL_DOWN_TOOL], run_tool=run_tool,
    )
    assert text == "Three"
    assert ran == [("get_scalar_view_items", {"view_uuid": "v-1"})]
    assert all(body["tools"] == [DRILL_DOWN_TOOL] for body in seen)
    assert "tool_choice" not in seen[0]
    followup = seen[1]["messages"]
    assert followup[-2]["role"] == "assistant" and followup[-2]["tool_calls"][0]["id"] == "call_1"
    assert followup[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"total": 3}'}
    assert usage["prompt_tokens"] == 30 and usage["completion_tokens"] == 6
    assert usage["tool_rounds"] == 2
    assert calls == [{"name": "get_scalar_view_items", "arguments": {"view_uuid": "v-1"},
                      "result": '{"total": 3}'}]
    assert MSGS == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


def test_last_round_forbids_tools_so_the_model_must_answer() -> None:
    seen: list[dict] = []
    send_chat_with_tools(
        _chat([_tool_call('{"view_uuid": "v-1"}'), _answer()], seen), OPENAI, "k", MSGS,
        "gpt-6-luna", tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}", max_rounds=2,
    )
    assert seen[1]["tool_choice"] == "none"


def test_tool_loop_that_never_answers_fails_strictly() -> None:
    seen: list[dict] = []
    with pytest.raises(ChatCallFailed, match="still calling tools"):
        send_chat_with_tools(
            _chat([_tool_call("{}"), _tool_call("{}")], seen), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}", max_rounds=2,
        )


def test_bad_arguments_go_back_to_the_model_without_running_the_tool() -> None:
    seen: list[dict] = []
    ran: list = []
    send_chat_with_tools(
        _chat([_tool_call("not json"), _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
        tools=[DRILL_DOWN_TOOL], run_tool=lambda n, a: ran.append(a) or "{}",
    )
    assert ran == []
    assert "error" in json.loads(seen[1]["messages"][-1]["content"])


def test_insufficient_quota_is_not_retried() -> None:
    seen: list[dict] = []
    quota = httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
    with pytest.raises(ChatCallFailed):
        send_chat_with_tools(
            _chat([quota, _answer()], seen), OPENAI, "k", MSGS, "gpt-6-luna",
            tools=[DRILL_DOWN_TOOL], run_tool=lambda _n, _a: "{}",
        )
    assert len(seen) == 1


# --- harness wiring --------------------------------------------------------------------------

class _RecallClient(_ItemsClient):
    """lme-sample-001 shows a collapsed View; lme-sample-002 does not."""

    def recall(self, group_id, query, limit=10):  # noqa: ANN001
        if group_id == "lme-lme-sample-001":
            return [f"[STRUCTURED MEMORY VERDICT]\nstatus: current | advisory\n{HINT}"]
        return ["[RELATED semantic MEMORY | non-authoritative] I now live in Denver"]


def _run(**kwargs):  # noqa: ANN003
    from archolith_bench.harness import run_memory_ab
    from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter

    return run_memory_ab(
        LongMemEvalMemoryAdapter(), arms=("menhir_recall",), fixture_path=LME_FIXTURE, limit=2,
        **kwargs,
    )


def test_tool_and_prompt_note_only_for_items_with_a_collapsed_view() -> None:
    plain_calls: list[list[dict]] = []
    tool_calls: list[dict] = []
    client = _RecallClient()

    def send_fn(_c, _b, _k, messages, _m, **_kw):  # noqa: ANN001, ANN003
        plain_calls.append(messages)
        return "Denver", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    def tool_send_fn(_c, _b, _k, messages, _m, *, tools, run_tool, strict):  # noqa: ANN001
        tool_calls.append({"messages": messages, "tools": tools, "strict": strict})
        result = run_tool("get_scalar_view_items", {"view_uuid": "v-1"})
        return "Biscuit", 1.0, {"prompt_tokens": 2, "completion_tokens": 1}, [{"result": result}]

    ab = _run(client=client, send_fn=send_fn, tool_send_fn=tool_send_fn, recall_only=True,
              drill_down=True)

    assert len(tool_calls) == 1 and len(plain_calls) == 1
    assert tool_calls[0]["tools"] == [DRILL_DOWN_TOOL] and tool_calls[0]["strict"] is True
    assert DRILL_DOWN_INSTRUCTION in tool_calls[0]["messages"][0]["content"]
    assert DRILL_DOWN_INSTRUCTION not in plain_calls[0][0]["content"]
    assert client.calls == [("lme-lme-sample-001", "v-1", 0, 20)]
    first = ab.arms["menhir_recall"].results[0]
    assert first.raw_usage["drill_down_view_ids"] == ["v-1"]
    assert len(first.raw_usage["drill_down_calls"]) == 1


def test_without_drill_down_the_tool_is_never_offered() -> None:
    tool_calls: list = []

    def send_fn(_c, _b, _k, messages, _m, **_kw):  # noqa: ANN001, ANN003
        assert DRILL_DOWN_INSTRUCTION not in messages[0]["content"]
        return "x", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    _run(client=_RecallClient(), send_fn=send_fn, tool_send_fn=lambda *a, **k: tool_calls.append(1),
         recall_only=True)
    assert tool_calls == []


def test_drill_down_requires_recall_only() -> None:
    with pytest.raises(ValueError, match="recall_only"):
        _run(client=_RecallClient(), drill_down=True, reset_confirmed=True)
