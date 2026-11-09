"""Menhir #519: a newer hedged value listed on an advisory verdict reaches the answering prompt."""

from archolith_bench.harness.menhir_client import _format_authority_record

_BASE = {
    "kind": "current",
    "status": "advisory",
    "has_foundation": True,
    "subject": "user",
    "attribute": "credit_score",
    "scope": "",
    "value": 1250,
    "valid_at": "2023-05-01T00:00:00Z",
}


def test_newer_approximate_is_rendered_under_the_verdict():
    record = dict(_BASE, newer_approximate=[{
        "assertion_id": "a1", "value": 1300, "quote": "I think I'm close to 1300",
        "valid_at": "2023-06-02T00:00:00Z",
    }])
    text = _format_authority_record(record)
    assert 'newer, approximate: about 1300 ("I think I\'m close to 1300", 2023-06-02T00:00:00Z)' in text
    assert "[STRUCTURED MEMORY VERDICT]" in text


def test_output_unchanged_without_newer_approximate():
    assert _format_authority_record(dict(_BASE, newer_approximate=[])) == _format_authority_record(_BASE)


def test_malformed_entries_are_skipped():
    record = dict(_BASE, newer_approximate=["x", {"quote": "no value"}])
    assert "newer, approximate" not in _format_authority_record(record)
