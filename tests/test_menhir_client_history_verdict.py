"""History verdicts carry value=None by design; they render as a history line, not "= None"."""

from __future__ import annotations

from archolith_bench.harness.menhir_client import _format_authority_record


def _contrib(span: str, day: str) -> dict:
    return {"stated_span": span, "operation": "occurrence", "valid_at": day}


def _history(**extra) -> dict:
    record = {"kind": "history", "status": "advisory", "value": None,
              "subject": "user", "attribute": "camping_trip_count", "scope": "all_time",
              "contributors": [_contrib("I went camping", "2026-05-01"),
                               _contrib("another camping trip", "2026-06-01")],
              "contributors_total": 2}
    record.update(extra)
    return record


def test_history_verdict_renders_entry_count_not_none() -> None:
    text = _format_authority_record(_history())
    assert "history: user — camping trip count (all time) (2 entries)" in text
    assert "= None" not in text
    assert "current fact:" not in text
    assert "status: history | advisory" in text


def test_history_without_total_counts_contributors() -> None:
    record = _history(contributors=[_contrib("I went camping", "2026-05-01")])
    del record["contributors_total"]
    assert "(1 entries)" in _format_authority_record(record)


def test_current_counted_advisory_unchanged() -> None:
    record = {"kind": "current", "status": "advisory", "attribute": "kit_count", "value": 5,
              "lower_bound": True, "has_foundation": False, "contributors": [],
              "contributors_total": 0}
    text = _format_authority_record(record)
    assert "current fact: kit count = at least 5 you mentioned" in text.splitlines()
    assert "history:" not in text
