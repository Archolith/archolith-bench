"""Menhir 4R match wording: a lower-bound counted verdict adds "plus M more that may be repeats",
per-action sums and item names. Without lower_bound, or without the fields, output is unchanged."""

from __future__ import annotations

from archolith_bench.harness.menhir_client import _format_authority_record


def _verdict(**extra) -> dict:
    record = {"kind": "current", "status": "advisory", "attribute": "model_kit_count", "value": 3,
              "view_uuid": "sv-1", "has_foundation": False, "basis": "counted",
              "contributors": [{"stated_span": "I bought a Zaku", "operation": "occurrence",
                                "valid_at": "2026-05-01"}],
              "contributors_total": 1, "contributors_truncated": False}
    record.update(extra)
    return record


_MATCH = {"count_uncertain": 2.0, "count_actions": '{"bought": 2, "finished": 1.0}',
          "count_items": '["Zaku", "Gundam"]'}


def test_lower_bound_renders_uncertain_actions_and_items_in_order() -> None:
    lines = _format_authority_record(_verdict(lower_bound=True, **_MATCH)).splitlines()
    fact = lines.index(
        "current fact: model kit count = at least 3 you mentioned, plus 2 more that may be repeats")
    assert lines[fact + 1] == "per action: bought 2, finished 1"
    assert lines[fact + 2] == "items: Zaku, Gundam"
    assert lines.index("provenance:") > fact + 2


def test_no_uncertain_keeps_plain_lower_bound_heading() -> None:
    for uncertain in (0, 0.0, None):
        text = _format_authority_record(
            _verdict(lower_bound=True, **{**_MATCH, "count_uncertain": uncertain}))
        assert "current fact: model kit count = at least 3 you mentioned\n" in text
        assert "may be repeats" not in text
        assert "per action: bought 2, finished 1" in text


def test_without_lower_bound_match_fields_are_ignored() -> None:
    assert (_format_authority_record(_verdict(**_MATCH))
            == _format_authority_record(_verdict()))
    assert (_format_authority_record(_verdict(lower_bound=False, **_MATCH))
            == _format_authority_record(_verdict(lower_bound=False)))


def test_lower_bound_without_match_fields_is_unchanged() -> None:
    text = _format_authority_record(_verdict(lower_bound=True))
    assert "current fact: model kit count = at least 3 you mentioned\n" in text
    assert "per action" not in text and "items:" not in text


def test_malformed_match_json_is_skipped() -> None:
    text = _format_authority_record(_verdict(
        lower_bound=True, count_actions="not json", count_items='{"a": 1}', count_uncertain=True))
    assert "per action" not in text and "items:" not in text
    assert "may be repeats" not in text
