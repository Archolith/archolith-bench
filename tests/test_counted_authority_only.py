"""counted_authority_only: counted advisory verdicts stop relabeling memories and render after them.

Off, every verdict renders first and marks the memories non-authoritative (unchanged). On, only
basis="counted" advisory verdicts move: after the memories, outside ``limit``, capped. Menhir's
``lower_bound`` (its counting wording flag) renders "at least N you mentioned" in either mode.
"""

from __future__ import annotations

from archolith_bench.harness.checkpoint import checkpoint_prompt_tag
from archolith_bench.harness.memory_ab import _agentic_recall, _merge_within_limit
from archolith_bench.harness.menhir_client import (
    COUNTED_AUTHORITY_MAX,
    HttpMenhirClient,
    _format_authority_record,
    is_counted_advisory,
)


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


def _verdict(attribute: str, value, *, basis=None, status="advisory", view="sv-1", **extra) -> dict:
    record = {"kind": "current", "status": status, "attribute": attribute, "value": value,
              "view_uuid": view, "has_foundation": status == "leads", "contributors": [],
              "contributors_total": 0, "contributors_truncated": False}
    if basis is not None:
        record["basis"] = basis
    record.update(extra)
    return record


def _memories(n: int) -> list[dict]:
    return [{"uuid": f"m{i}", "name": f"M{i}", "content": f"Memory {i}.", "memory_type": "semantic"}
            for i in range(n)]


def _client(data: dict, *, on: bool) -> HttpMenhirClient:
    client = HttpMenhirClient("http://localhost:9999", counted_authority_only=on)
    client._client = _Http(data)
    return client


def test_is_counted_advisory_needs_counted_basis_and_advisory_status() -> None:
    assert is_counted_advisory(_verdict("kit_count", 5, basis="counted"))
    assert not is_counted_advisory(_verdict("kit_count", 5, basis="counted", status="leads"))
    assert not is_counted_advisory(_verdict("age", 30))


def test_off_counted_verdict_leads_and_relabels_memories() -> None:
    # Fails if the option acts while off: the verdict must stay first and relabel memories.
    data = {"results": _memories(2), "authority_layer": [_verdict("kit_count", 5, basis="counted")]}
    out = _client(data, on=False).recall("ns", "q", limit=2)
    assert out[0].startswith("[STRUCTURED MEMORY VERDICT]")
    assert out[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert out.counted == () and out.extras == ()


def test_on_counted_verdict_renders_after_memories_without_relabel() -> None:
    # Fails if a counted verdict still relabels memories, renders first, or counts toward limit.
    data = {"results": _memories(3), "authority_layer": [_verdict("kit_count", 5, basis="counted")]}
    out = _client(data, on=True).recall("ns", "q", limit=3)
    assert list(out[:3]) == ["M0: Memory 0.", "M1: Memory 1.", "M2: Memory 2."]
    assert out[3].startswith("[STRUCTURED MEMORY VERDICT]") and "kit count = 5" in out[3]
    assert out.counted == (out[3],) and out[3] in out.extras


def test_on_other_verdicts_unchanged() -> None:
    # Fails if the option touches stated advisory or leading verdicts.
    stated = _verdict("age", 30, view="sv-2")
    leading = _verdict("kit_count", 5, basis="counted", status="leads", view="sv-3")
    data = {"results": _memories(1), "authority_layer": [stated, leading]}
    assert list(_client(data, on=True).recall("ns", "q", limit=3)) == list(
        _client(data, on=False).recall("ns", "q", limit=3))


def test_on_mixed_stated_verdict_still_relabels_and_counted_goes_last() -> None:
    data = {"results": _memories(1), "authority_layer": [
        _verdict("kit_count", 5, basis="counted", view="sv-1"), _verdict("age", 30, view="sv-2")]}
    out = _client(data, on=True).recall("ns", "q", limit=2)
    assert "age = 30" in out[0]
    assert out[1].startswith("[RELATED semantic MEMORY | non-authoritative]")
    assert "kit count = 5" in out[2] and out.counted == (out[2],)


def test_on_counted_verdicts_capped_and_their_view_not_ranked_twice() -> None:
    verdicts = [_verdict(f"c{i}_count", i, basis="counted", view=f"sv-{i}")
                for i in range(COUNTED_AUTHORITY_MAX + 2)]
    ranked_view = {"uuid": "sv-0", "name": "c0", "content": "c0 View.", "memory_type": "scalar_state"}
    data = {"results": [ranked_view, *_memories(2)], "authority_layer": verdicts}
    out = _client(data, on=True).recall("ns", "q", limit=2)
    assert list(out[:2]) == ["M0: Memory 0.", "M1: Memory 1."]
    assert len(out.counted) == COUNTED_AUTHORITY_MAX and len(out) == 2 + COUNTED_AUTHORITY_MAX


def test_counted_verdicts_stay_outside_every_recap() -> None:
    data = {"results": _memories(3), "authority_layer": [_verdict("kit_count", 5, basis="counted")]}
    out = _client(data, on=True).recall("ns", "q", limit=3)
    assert _merge_within_limit(list(out), 3, out.extras) == list(out)

    def _facet(i: int) -> dict:
        return {"uuid": f"f{i}", "name": f"F{i}", "content": f"Facet {i}.",
                "memory_type": "semantic", "retrieval_score_kind": "verb_facet"}

    class _Twice:
        def recall(self, group_id, query, limit=10):  # noqa: ANN001, ANN201
            data_q = {"results": [*_memories(1), _facet(0), _facet(1)] if query == "a" else [
                {"uuid": "x", "name": "X", "content": "Other.", "memory_type": "semantic"}],
                "authority_layer": [_verdict(f"{query}_count", 1, basis="counted", view=query)]}
            return _client(data_q, on=True).recall(group_id, query, limit=limit)

    # 2 facets + 2 counted verdicts: each has its own cap of 3, so none is dropped.
    merged = _agentic_recall(_Twice(), "ns", ["a", "b"], 2).split("\n")
    assert "M0: Memory 0." in merged and "X: Other." in merged
    assert sum(line.startswith("[STRUCTURED MEMORY VERDICT]") for line in merged) == 2
    assert sum(line.startswith("[RELATED semantic MEMORY") is False and "Facet" in line
               for line in merged) == 2


def test_lower_bound_wording_in_either_mode() -> None:
    record = _verdict("kit_count", 5, basis="counted", lower_bound=True)
    assert "current fact: kit count = at least 5 you mentioned" in _format_authority_record(record)
    assert "current fact: kit count = 5" in _format_authority_record(
        _verdict("kit_count", 5, basis="counted"))


def test_checkpoint_tag_marks_counted_authority_only() -> None:
    assert checkpoint_prompt_tag(object(), counted_authority_only=True) == "ca"
    assert checkpoint_prompt_tag(object(), drill_down=True, counted_authority_only=True) == "ddca"
    assert checkpoint_prompt_tag(object()) is None
