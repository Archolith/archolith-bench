"""Menhir verb-facet facts stay outside the recall limit in every memory mode.

The plain arm joins the client's snippets as-is. The value-augmented arms and agentic recall
re-cap the context to recall_limit, and the Recall Lab rescore scripts carry their own copy
of the render loop; each must keep the facet facts without counting them.
"""

from __future__ import annotations

import pytest

from archolith_bench.harness import memory_ab
from archolith_bench.harness.memory_ab import _agentic_recall, _merge_within_limit
from archolith_bench.harness.menhir_client import (
    VERB_FACET_MAX,
    HttpMenhirClient,
    RecallSnippets,
    place_verb_facets,
)


def _facet(uuid: str, fact: str) -> dict:
    return {"uuid": uuid, "name": fact, "content": fact, "memory_type": "SEMANTIC",
            "retrieval_score_kind": "verb_facet"}


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


def test_client_caps_facets_and_reports_them_as_extras() -> None:
    client = HttpMenhirClient("http://localhost:9999")
    client._client = _Http({"results": [
        {"name": "A", "content": "Ranked one."},
        *[_facet(f"f{i}", f"Facet fact {i}.") for i in range(5)],
        {"name": "B", "content": "Ranked two."},
    ]})
    recalled = client.recall("ns", "q", limit=1)
    assert isinstance(recalled, list)
    assert recalled.extras == tuple(f"Facet fact {i}." for i in range(VERB_FACET_MAX))
    assert list(recalled) == [*recalled.extras, "A: Ranked one."]


def test_no_facets_means_no_extras_and_the_plain_cut() -> None:
    out = place_verb_facets(["v", "a", "b"], 1, [], 2)
    assert out == ["v", "a"] and out.extras == ()


def test_merge_keeps_extras_in_place_and_does_not_count_them() -> None:
    snippets = ["val", "verdict", "f1", "r1", "r2", "r3", "r1"]
    assert _merge_within_limit(snippets, 3, ["f1"]) == ["val", "verdict", "f1", "r1"]
    # Without extras it is the old dedup-and-cut, including the old floor of one snippet.
    assert _merge_within_limit(snippets, 3) == ["val", "verdict", "f1"]
    assert _merge_within_limit(snippets, 0) == ["val"]


class _Graph:
    def recall(self, question, limit, **_kw):  # noqa: ANN001, ANN003, ANN201
        return ["VALUE one", "VALUE two"][:limit]

    advisory_recall = recall
    derived_recall = recall

    def stale_value_strings(self, question):  # noqa: ANN001, ANN201
        return {"stale"}


class _Adapter:
    def sessions(self, item):  # noqa: ANN001, ANN201
        return []


@pytest.mark.parametrize("fn, kwargs", [
    (memory_ab._value_augmented_recall, {}),
    (memory_ab._value_augmented_recall_v2, {"emit_history": False}),
    (memory_ab._value_augmented_recall_v3, {"suppress": True}),
    (memory_ab._value_augmented_recall_v4, {}),
    (memory_ab._value_augmented_recall_v5, {}),
])
def test_value_arms_keep_facets_outside_the_limit(monkeypatch, fn, kwargs) -> None:
    monkeypatch.setattr(memory_ab.ValueGraph, "from_item", classmethod(lambda cls, *a, **k: _Graph()))
    monkeypatch.setattr(memory_ab.SupersededValueGraph, "from_item",
                        classmethod(lambda cls, *a, **k: _Graph()))
    recalled = RecallSnippets(["verdict", "FACET fact", "r1", "r2", "r3"], ["FACET fact"])
    out = fn(_Adapter(), {}, recalled, "q", 4, "t1", **kwargs).split("\n")
    # recall_limit 4 // 3 gives one value snippet; the facet is extra.
    assert out == ["VALUE one", "verdict", "FACET fact", "r1", "r2"]


class _AgenticClient:
    def __init__(self, per_query: dict[str, RecallSnippets]) -> None:
        self.per_query = per_query
        self.asked: list[str] = []

    def recall(self, group_id, query, limit=10):  # noqa: ANN001, ANN201
        self.asked.append(query)
        return self.per_query[query]


def test_agentic_keeps_facets_outside_the_cap_and_caps_them_in_total() -> None:
    client = _AgenticClient({
        "a": RecallSnippets(["F1", "F2", "a1", "a2"], ["F1", "F2"]),
        "b": RecallSnippets(["F3", "F4", "b1", "a1"], ["F3", "F4"]),
        "c": RecallSnippets(["c1"]),
    })
    out = _agentic_recall(client, "ns", ["a", "b", "c"], 3).split("\n")
    assert out == ["F1", "F2", "a1", "a2", "F3", "b1"]
    assert client.asked == ["a", "b"]


def test_agentic_without_facets_is_unchanged() -> None:
    client = _AgenticClient({"a": ["a1", "a2"], "b": ["a2", "b1", "b2"]})
    assert _agentic_recall(client, "ns", ["a", "b"], 3) == "a1\na2\nb1"


@pytest.mark.parametrize("module, make", [
    ("scripts.longmemeval.analysis.run_arm_rescore",
     lambda m: m.RecallLabArmClient("http://x", arm_id="d", tuning={})),
    ("scripts.longmemeval.analysis.run_arm_e_rescore",
     lambda m: m.RecallLabArmEClient("http://x")),
])
def test_rescore_clients_render_facets_like_the_client(module, make) -> None:
    import importlib

    client = make(importlib.import_module(module))
    client._client = _Http({"arms": [{"ok": True, "results": [
        {"name": "A", "content": "Ranked one."},
        *[_facet(f"f{i}", f"Facet fact {i}.") for i in range(4)],
        {"name": "B", "content": "Ranked two."},
    ]}]})
    recalled = client.recall("ns", "q", limit=1)
    assert list(recalled) == ["Facet fact 0.", "Facet fact 1.", "Facet fact 2.", "A: Ranked one."]
    assert len(recalled.extras) == VERB_FACET_MAX
