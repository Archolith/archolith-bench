"""Occurrence contributors render event vs mentioned time when Menhir supplies it."""

from __future__ import annotations

from archolith_bench.harness.menhir_client import _format_authority_record


def _record(contributor: dict) -> dict:
    return {
        "kind": "current",
        "status": "leads",
        "subject": "user",
        "attribute": "viewings",
        "value": "1",
        "contributors": [contributor],
    }


def _provenance_line(contributor: dict) -> str:
    lines = _format_authority_record(_record(contributor)).splitlines()
    return lines[lines.index("provenance:") + 1]


def test_occurrence_with_event_date_and_sentence():
    line = _provenance_line(
        {
            "operation": "occurrence",
            "stated_span": "viewed a 1-bedroom condo",
            "valid_at": "2023-03-08T01:55:00Z",
            "event_at": "2023-02-10T00:00:00+00:00",
            "event_basis": "explicit",
            "mentioned_at": "2023-03-08T01:55:00Z",
            "source_sentence": "I viewed a 1-bedroom condo on February 10th, but the noise...",
        }
    )
    assert line == (
        '- occurrence | "viewed a 1-bedroom condo" | event 2023-02-10 | mentioned 2023-03-08 '
        '| sentence: "I viewed a 1-bedroom condo on February 10th, but the noise..."'
    )


def test_occurrence_without_event_date_omits_event_part():
    line = _provenance_line(
        {
            "operation": "occurrence",
            "stated_span": "viewed a condo",
            "valid_at": "2023-03-08T01:55:00Z",
            "event_at": None,
            "event_basis": None,
            "mentioned_at": "2023-03-08T01:55:00Z",
            "source_sentence": "I viewed a condo a few weeks ago.",
        }
    )
    assert line == (
        '- occurrence | "viewed a condo" | mentioned 2023-03-08 '
        '| sentence: "I viewed a condo a few weeks ago."'
    )
    assert "event " not in line


def test_occurrence_without_new_keys_keeps_old_line():
    line = _provenance_line(
        {"operation": "occurrence", "stated_span": "q", "valid_at": "2023-03-08T01:55:00Z"}
    )
    assert line == '- occurrence | "q" | 2023-03-08T01:55:00Z'


def test_non_occurrence_with_new_keys_keeps_old_line():
    line = _provenance_line(
        {
            "operation": "absolute",
            "stated_span": "q",
            "valid_at": "2023-03-08T01:55:00Z",
            "event_at": "2023-02-10T00:00:00+00:00",
            "event_basis": "explicit",
            "mentioned_at": "2023-03-08T01:55:00Z",
            "source_sentence": "s",
        }
    )
    assert line == '- absolute | "q" | 2023-03-08T01:55:00Z'
