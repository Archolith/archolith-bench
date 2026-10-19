"""Scalar View scoring reads the typed value (ss_value), not the numeric view_value mirror.

Menhir sets view_value to 0.0 for text kinds, so comparing it scored every correct clock_time and
weekday View as a miss, and let a stray status View slip past the advisory control.
"""

from __future__ import annotations

import pytest

from archolith_bench.harness.menhir_scalar_state import (
    _validate_case,
    default_scalar_state_cases,
)
from archolith_bench.harness.scalar_bolt import _Q_VIEWS

pytestmark = pytest.mark.unit

CASES = {c.case_id: c for c in default_scalar_state_cases()}


def _view(kind: str, ss_value: str, view_value: float = 0.0) -> dict:
    return {"ss_kind": kind, "ss_value": ss_value, "ss_display": ss_value, "value": view_value,
            "ss_attribute": kind}


@pytest.mark.parametrize("case_id,kind,ss_value", [
    ("ss-clock-wake", "clock_time", "07:30"),
    ("ss-weekday-dayoff", "weekday", "wednesday"),
])
def test_correct_text_kind_view_matches(case_id, kind, ss_value):
    result = _validate_case(CASES[case_id], [], [_view(kind, ss_value)], [])
    assert result.matched
    assert result.actual["view"]["value"] == ss_value


def test_wrong_text_kind_view_does_not_match():
    result = _validate_case(CASES["ss-clock-wake"], [], [_view("clock_time", "06:00")], [])
    assert not result.matched


def test_stray_status_view_fails_the_advisory_control():
    result = _validate_case(CASES["ss-control-car-advisory"], [], [_view("status", "red")], [])
    assert not result.matched and result.actual["stray_view"]


def test_numeric_view_still_matches_through_ss_value():
    result = _validate_case(CASES["ss-count-coins"], [], [_view("count", "37", 37.0)], [])
    assert result.matched


def test_numeric_view_without_ss_value_falls_back_to_view_value():
    row = {"ss_kind": "count", "value": 37.0}
    assert _validate_case(CASES["ss-count-coins"], [], [row], []).matched


def test_bolt_reader_returns_the_typed_value():
    assert "v.ss_value" in _Q_VIEWS and "v.ss_display" in _Q_VIEWS
