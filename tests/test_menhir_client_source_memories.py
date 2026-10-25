"""source_memories: Menhir's raw-episode block renders after the memories, outside ``limit``.

Off (the default), the response's ``source_memories`` are ignored and the context is unchanged. On,
they render as one block in Menhir's build_context format, after the memories and any counted
verdicts. A blocking event advisory still suppresses everything but the advisory.
"""

from __future__ import annotations

from archolith_bench.harness.checkpoint import checkpoint_prompt_tag
from archolith_bench.harness.memory_ab import VERB_FACET_MAX, _agentic_recall
from archolith_bench.harness.menhir_client import HttpMenhirClient, format_source_memories


class _Resp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _Http:
    def __init__(self, data: dict) -> None:
        self.data = data

    def post(self, url, **_kw):  # noqa: ANN001, ANN003, ANN201
        return _Resp(self.data)


def _memories(n: int) -> list[dict]:
    return [{"uuid": f"m{i}", "name": f"M{i}", "content": f"Memory {i}.", "memory_type": "semantic"}
            for i in range(n)]


SOURCES = [
    {"uuid": "s1", "content": "I paid $25 for a tune-up.", "reference_time": "2023-03-02T10:00:00Z",
     "cosine": 0.71, "pool_id": "p1", "pool_anchor": "a1"},
    {"uuid": "s2", "content": "Got new bike lights for $40.", "reference_time": None, "cosine": 0.66},
]


def _client(data: dict, *, on: bool, counted: bool = False) -> HttpMenhirClient:
    client = HttpMenhirClient(
        "http://localhost:9999", source_memories=on, counted_authority_only=counted
    )
    client._client = _Http(data)
    return client


def test_format_matches_build_context_without_pool_tag() -> None:
    assert format_source_memories(SOURCES) == (
        "Source memories (raw episodes, oldest first):\n"
        "  - 2023-03-02T10:00:00Z | I paid $25 for a tune-up.\n"
        "  - unknown time | Got new bike lights for $40."
    )


def test_format_skips_malformed_and_empty_entries() -> None:
    assert format_source_memories(None) == ""
    assert format_source_memories([]) == ""
    assert format_source_memories(["text", {"content": "  "}, {"reference_time": "t"}]) == ""


def test_off_ignores_source_memories() -> None:
    data = {"results": _memories(2), "source_memories": SOURCES}
    recalled = _client(data, on=False).recall("ns", "q", limit=5)
    assert recalled == ["M0: Memory 0.", "M1: Memory 1."]
    assert not getattr(recalled, "sources", ())


def test_on_renders_one_block_last_outside_limit() -> None:
    data = {"results": _memories(5), "source_memories": SOURCES}
    recalled = _client(data, on=True).recall("ns", "q", limit=3)
    block = format_source_memories(SOURCES)
    assert list(recalled) == ["M0: Memory 0.", "M1: Memory 1.", "M2: Memory 2.", block]
    assert recalled.sources == (block,)
    assert block in recalled.extras


def test_on_without_sources_is_unchanged() -> None:
    data = {"results": _memories(2)}
    assert list(_client(data, on=True).recall("ns", "q", limit=5)) == list(
        _client(data, on=False).recall("ns", "q", limit=5)
    )


def test_sources_follow_counted_verdicts() -> None:
    verdict = {"kind": "current", "status": "advisory", "basis": "counted", "attribute": "trips",
               "value": 3, "view_uuid": "sv-1", "has_foundation": False, "contributors": [],
               "contributors_total": 0, "contributors_truncated": False}
    data = {"results": _memories(1), "authority_layer": [verdict], "source_memories": SOURCES}
    recalled = _client(data, on=True, counted=True).recall("ns", "q", limit=5)
    assert recalled[0] == "M0: Memory 0."
    assert recalled.counted and recalled[1] == recalled.counted[0]
    assert recalled[-1] == format_source_memories(SOURCES)


def test_blocking_event_advisory_suppresses_sources() -> None:
    data = {
        "results": _memories(2),
        "event_authority_layer": [
            {"status": "advisory", "gate": "anchor",
             "reason": "no evidence-anchored object could be selected"},
        ],
        "source_memories": SOURCES,
    }
    recalled = _client(data, on=True).recall("ns", "q", limit=5)
    assert len(recalled) == 1
    assert "Source memories" not in recalled[0]


class _PerQueryHttp:
    """Each subquery gets its own memories and its own source block."""

    def post(self, url, **kw):  # noqa: ANN001, ANN003, ANN201
        q = kw["json"]["query"]
        return _Resp({
            "results": [{"uuid": f"{q}-{i}", "name": f"{q}{i}", "content": "x.",
                         "memory_type": "semantic"} for i in range(2)],
            "source_memories": [{"uuid": f"s-{q}", "content": f"raw {q}", "reference_time": "t",
                                 "cosine": 0.7}],
        })


def test_agentic_recall_keeps_one_source_block_outside_cap() -> None:
    client = HttpMenhirClient("http://localhost:9999", source_memories=True)
    client._client = _PerQueryHttp()
    queries = ["a", "b", "c", "d"][: VERB_FACET_MAX + 1]
    context = _agentic_recall(client, "ns", queries, recall_limit=4)
    assert context.count("Source memories (raw episodes, oldest first):") == 1
    assert "raw a" in context
    # The block does not take a memory slot: the cap of 4 is still filled by memories.
    assert sum(line.endswith(": x.") for line in context.splitlines()) == 4


def test_checkpoint_tag_marks_source_memories() -> None:
    class _Adapter:
        prompt_fingerprint = "fp"

    assert checkpoint_prompt_tag(_Adapter(), source_memories=True) == "fpsm"
    assert checkpoint_prompt_tag(_Adapter()) == "fp"
