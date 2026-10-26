"""--writer-items: agent-writer items render after the memories, outside the limit.

They come from a second, source-memory-only recall request, so the first request and the context
it renders (memories, counted verdicts, the source block) stay exactly as without the flag.
"""

from __future__ import annotations

from typing import Any

from archolith_bench.harness.checkpoint import checkpoint_prompt_tag
from archolith_bench.harness.memory_ab import _agentic_recall
from archolith_bench.harness.menhir_client import HttpMenhirClient, writer_item_snippets


class _Resp:
    def __init__(self, data: Any) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._data


class _Http:
    """First request gets ``main``; a request with source_memory_limit gets ``lane``."""

    def __init__(self, main: Any, lane: Any) -> None:
        self.main, self.lane = main, lane
        self.payloads: list[dict[str, Any]] = []

    def post(self, _url: str, **kw: Any) -> _Resp:
        payload = kw["json"]
        self.payloads.append(payload)
        return _Resp(self.lane if "source_memory_limit" in payload else self.main)


MEMORIES = [{"uuid": f"m{i}", "name": f"M{i}", "content": f"Memory {i}.", "memory_type": "semantic"}
            for i in range(3)]
DEFAULT_SOURCES = [
    {"uuid": "u1", "content": "user said hi", "source": "user", "cosine": 0.9,
     "reference_time": "2023-05-19"},
    {"uuid": "w2", "content": "assistant recommended lager", "source": "agent-writer",
     "cosine": 0.8, "reference_time": "2023-05-21"},
]
LANE = [
    {"uuid": "u1", "content": "user said hi", "source": "user", "cosine": 0.99,
     "reference_time": "2023-05-19"},
    {"uuid": "w1", "content": "assistant listed A, B, C", "source": "agent-writer",
     "cosine": 0.60, "reference_time": "2023-05-20"},
    {"uuid": "w2", "content": "assistant recommended lager", "source": "agent-writer",
     "cosine": 0.80, "reference_time": "2023-05-21"},
    {"uuid": "w3", "content": "assistant gave a recipe", "source": "agent-writer",
     "cosine": 0.70, "reference_time": "2023-05-22"},
    {"uuid": "w4", "content": "   ", "source": "agent-writer", "cosine": 0.95},
]
MAIN = {"results": MEMORIES, "source_memories": DEFAULT_SOURCES}


def _client(*, writer: int, sources: bool = True, main: Any = MAIN, lane: Any = None):
    client = HttpMenhirClient("http://localhost:9999", source_memories=sources, writer_items=writer)
    http = _Http(main, {"results": [], "source_memories": LANE} if lane is None else lane)
    client._client = http  # type: ignore[assignment]
    return client, http


def _notes(out: list[str]) -> list[str]:
    return [s for s in out if s.startswith("[agent-recorded note")]


def test_off_sends_one_request_and_renders_no_notes() -> None:
    client, http = _client(writer=0)
    out = client.recall("ns", "q", limit=2)
    assert len(http.payloads) == 1 and "source_memory_limit" not in http.payloads[0]
    assert _notes(out) == []


def test_first_request_and_its_context_are_unchanged_by_the_flag() -> None:
    off_client, off_http = _client(writer=0)
    on_client, on_http = _client(writer=2)
    off = off_client.recall("ns", "q", limit=2)
    on = on_client.recall("ns", "q", limit=2)
    assert on_http.payloads[0] == off_http.payloads[0]
    assert list(on)[: len(off)] == list(off)
    assert on.sources == off.sources


def test_second_request_is_source_memory_only_with_the_lane_maximum() -> None:
    client, http = _client(writer=2)
    client.recall("ns", "q", limit=2)
    assert http.payloads[1] == {
        "query": "q", "limit": 1, "namespace": "ns", "include_session": True,
        "source_memory_limit": 50,
    }


def test_top_writer_items_by_cosine_skip_user_blank_and_shown_items() -> None:
    # w2 is already in the source block, so the top two after it are w3 and w1.
    client, _http = _client(writer=2)
    out = client.recall("ns", "q", limit=2)
    assert _notes(out) == [
        "[agent-recorded note | 2023-05-22] assistant gave a recipe",
        "[agent-recorded note | 2023-05-20] assistant listed A, B, C",
    ]
    assert out.writer == tuple(_notes(out))
    assert set(_notes(out)) <= set(out.extras)
    assert len([s for s in out if s not in out.extras]) == 2  # memories keep the full limit


def test_without_source_block_nothing_is_excluded() -> None:
    client, _http = _client(writer=1, sources=False)
    out = client.recall("ns", "q", limit=2)
    assert _notes(out) == ["[agent-recorded note | 2023-05-21] assistant recommended lager"]


def test_blocking_event_advisory_sends_no_second_request() -> None:
    blocked = {
        "results": MEMORIES,
        "event_authority_layer": {"status": "advisory", "gate": "anchor", "answer": "x",
                                  "query": "q"},
    }
    client, http = _client(writer=3, main=blocked)
    out = client.recall("ns", "q", limit=2)
    assert len(http.payloads) == 1
    assert _notes(out) == []


def test_lane_without_source_memories_renders_nothing_extra() -> None:
    client, _http = _client(writer=3, lane={"results": []})
    out = client.recall("ns", "q", limit=5)
    assert _notes(out) == []


def test_writer_item_snippets_handles_bad_input() -> None:
    assert writer_item_snippets(None, 3) == []
    assert writer_item_snippets(LANE, 0) == []
    assert writer_item_snippets([None, "x", {"source": "agent-writer"}], 3) == []


class _PerQueryHttp:
    def post(self, _url: str, **kw: Any) -> _Resp:
        q = kw["json"]["query"]
        if "source_memory_limit" in kw["json"]:
            return _Resp({"source_memories": [
                {"uuid": f"w-{q}-{i}", "content": f"note {q}{i}", "source": "agent-writer",
                 "cosine": 0.5 + i / 10, "reference_time": "t"} for i in range(3)]})
        return _Resp({"results": [{"uuid": f"{q}-{i}", "name": f"{q}{i}", "content": "x.",
                                   "memory_type": "semantic"} for i in range(2)]})


def test_agentic_recall_caps_writer_items_across_subqueries() -> None:
    client = HttpMenhirClient("http://localhost:9999", writer_items=2)
    client._client = _PerQueryHttp()  # type: ignore[assignment]
    context = _agentic_recall(client, "ns", ["a", "b", "c"], recall_limit=4)
    assert context.count("[agent-recorded note") == 2
    # Writer notes do not take memory slots.
    assert sum(line.endswith(": x.") for line in context.splitlines()) == 4


def test_checkpoint_tag_marks_writer_items() -> None:
    class _Adapter:
        prompt_fingerprint = "fp"

    assert checkpoint_prompt_tag(_Adapter()) == "fp"
    assert checkpoint_prompt_tag(_Adapter(), source_memories=True, writer_items=3) == "fpsmwi3"
