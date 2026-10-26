"""--recall-trace: opt-in, content-free per-recall trace kept on the client and in the checkpoint."""

from __future__ import annotations

import json

import pytest

from archolith_bench.harness.base import TaskResult
from archolith_bench.harness.checkpoint import (
    MemoryCheckpoint,
    checkpoint_path_for,
    checkpoint_prompt_tag,
)
from archolith_bench.harness.memory_ab import SINGLE_RECALL, _run_memory_arm
from archolith_bench.harness.menhir_client import HttpMenhirClient
from archolith_bench.core.metrics import PRICING_DEFAULTS

RESPONSE = {
    "results": [
        {"uuid": "m1", "name": "M1", "content": "SECRET memory text one.", "memory_type": "semantic",
         "final_score": 0.9, "retrieval_score": 0.5, "scope": "SESSION"},
        {"uuid": "m2", "name": "M2", "content": "SECRET memory text two.", "memory_type": "episodic",
         "final_score": 0.4},
    ],
    "candidates_evaluated": 37,
    "source_memories": [
        {"uuid": "s1", "content": "SECRET raw episode.", "reference_time": "2023-03-02T10:00:00Z",
         "cosine": 0.71, "source": "episode", "pool_id": "p1", "pool_anchor": "a1"},
    ],
    "trace": {
        "phases": {"vector": 12, "bm25": 9},
        "candidates": [
            {"uuid": "m1", "source": "graph", "similarity": 0.5, "survived_floor": True,
             "final_score": 0.9, "rank": 1, "cosine_rank": 3, "content_rank": 2,
             "content_cosine": 0.6, "name": "drop me", "content": "SECRET candidate text"},
            {"uuid": "m9", "source": "episode", "similarity": 0.1, "survived_floor": False},
        ],
    },
}


class _Resp:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _Http:
    def __init__(self, *responses) -> None:  # noqa: ANN002
        self.responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, url, **kw):  # noqa: ANN001, ANN003, ANN201
        self.payloads.append(kw["json"])
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(nxt)


def _client(http: _Http, *, trace: bool) -> HttpMenhirClient:
    client = HttpMenhirClient("http://localhost:9999", recall_trace=trace)
    client._client = http
    return client


def test_flag_off_sends_no_trace_and_keeps_none() -> None:
    http = _Http(RESPONSE)
    client = _client(http, trace=False)
    client.recall("ns", "q", limit=5)
    assert "trace" not in http.payloads[0]
    assert client.last_recall_trace is None


def test_flag_on_sends_trace_and_stores_compact_content_free_dict() -> None:
    http = _Http(RESPONSE)
    client = _client(http, trace=True)
    client.recall("ns", "q", limit=5)
    assert http.payloads[0]["trace"] is True
    t = client.last_recall_trace
    assert t["candidates_evaluated"] == 37
    assert t["limit"] == 5
    assert t["phases"] == {"vector": 12, "bm25": 9}
    assert [r["uuid"] for r in t["results"]] == ["m1", "m2"]
    assert [r["rank"] for r in t["results"]] == [1, 2]
    assert t["results"][0]["final_score"] == 0.9
    assert t["candidates"][0] == {
        "uuid": "m1", "source": "graph", "similarity": 0.5, "survived_floor": True,
        "final_score": 0.9, "rank": 1, "cosine_rank": 3, "content_rank": 2, "content_cosine": 0.6,
    }
    assert t["candidates"][1] == {"uuid": "m9", "source": "episode", "similarity": 0.1,
                                  "survived_floor": False}
    assert t["source_memories"] == [
        {"uuid": "s1", "reference_time": "2023-03-02T10:00:00Z", "cosine": 0.71,
         "source": "episode", "pool_id": "p1"}
    ]
    assert "SECRET" not in json.dumps(t)


def test_trace_tolerates_response_without_trace_block() -> None:
    client = _client(_Http({"results": []}), trace=True)
    client.recall("ns", "q")
    t = client.last_recall_trace
    assert t["candidates"] == [] and t["source_memories"] == [] and t["phases"] is None


def test_stale_trace_is_cleared_on_next_recall() -> None:
    http = _Http(RESPONSE, RuntimeError("boom"), {"results": []})
    client = _client(http, trace=True)
    client.recall("ns", "q1")
    assert client.last_recall_trace is not None
    with pytest.raises(RuntimeError):
        client.recall("ns", "q2")
    assert client.last_recall_trace is None
    off = _client(_Http(RESPONSE), trace=False)
    off.last_recall_trace = {"stale": True}
    off.recall("ns", "q3")
    assert off.last_recall_trace is None


class _Adapter:
    benchmark_id = "longmemeval-menhir"

    def question(self, item):  # noqa: ANN001, ANN201
        return "what?"

    def build_messages(self, ctx, question):  # noqa: ANN001, ANN201
        return [{"role": "user", "content": ctx + question}]

    def score(self, item, text):  # noqa: ANN001, ANN201
        return True


def _run(client: HttpMenhirClient) -> list[TaskResult]:
    arm = _run_memory_arm(
        _Adapter(), SINGLE_RECALL, [{"question_id": "q1", "answer": "a"}, {"question_id": "q2"}],
        client=client, chat_client=None,
        send_fn=lambda *a, **k: ("ans", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}),
        chat_base_url="", api_key="", model="m", recall_limit=5,
        pricing=PRICING_DEFAULTS["deepseek-v4-flash"], reset_memory=False,
        reset_confirmed=False, dry_run_reset=False, recall_only=True,
    )
    return arm.results


def test_task_result_carries_trace_only_when_on() -> None:
    off = _run(_client(_Http(RESPONSE, RESPONSE), trace=False))
    assert [r.recall_trace for r in off] == [{}, {}]
    on = _run(_client(_Http(RESPONSE, {"results": []}), trace=True))
    assert on[0].recall_trace["candidates_evaluated"] == 37
    assert on[1].recall_trace["candidates_evaluated"] is None  # second question's own trace


def test_checkpoint_round_trip_and_old_records_load(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "ck.jsonl"
    ck = MemoryCheckpoint(path)
    tr = TaskResult("t1", "ans", 1, 2, 3.0, True, recall_trace={"limit": 5, "results": []})
    ck.record("arm", "t1", tr)
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["result"]["recall_trace"] == {"limit": 5, "results": []}
    old = {"arm": "arm", "task_id": "t2", "ts": 0, "result": {
        "task_id": "t2", "response_text": "x", "input_tokens": 1, "output_tokens": 1,
        "latency_ms": 1.0, "correct": True}}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(old) + "\n")
    loaded = MemoryCheckpoint(path)
    assert loaded.get("arm", "t2") is not None
    assert loaded.get("arm", "t1") is not None


def test_checkpoint_fingerprint_is_not_affected_by_the_flag(tmp_path) -> None:  # noqa: ANN001
    # The flag is never passed to checkpoint_prompt_tag / checkpoint_path_for, so the name is the same.
    import inspect

    assert "recall_trace" not in inspect.signature(checkpoint_prompt_tag).parameters
    adapter = _Adapter()
    tag = checkpoint_prompt_tag(adapter, source_memories=True)
    assert checkpoint_path_for(tmp_path, "b", "m", variant="s", prompt_tag=tag) == checkpoint_path_for(
        tmp_path, "b", "m", variant="s", prompt_tag=checkpoint_prompt_tag(adapter, source_memories=True)
    )
