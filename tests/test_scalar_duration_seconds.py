"""The commute fixture expects the duration Menhir actually stores: seconds, not minutes.

Menhir canonicalizes every admitted duration to seconds, so "My commute takes 45 minutes" lands as
ss_value=2700, ss_unit=seconds, ss_display=45:00. The fixture expected 45.0, so the harness and the
coverage script both scored a correct View as a miss at assertion_emitted (live run, #170 item 4).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from archolith_bench.harness.menhir_scalar_state import _validate_case, default_scalar_state_cases

CASES = {c.case_id: c for c in default_scalar_state_cases()}
COMMUTE = CASES["ss-duration-commute"]

# The View and assertion exactly as the 2026-10-19 live run wrote them.
LIVE_VIEW = {"ss_kind": "duration", "ss_value": "2700", "ss_display": "45:00", "value": 2700.0,
             "ss_unit": "seconds", "ss_attribute": "commute_duration"}
LIVE_ASSERTION = {"value_kind": "duration", "value": "2700", "attribute": "commute_duration",
                  "binding_pending": False, "subject_display": "user"}


def _coverage_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "scalar_state_coverage.py"
    spec = importlib.util.spec_from_file_location("scalar_state_coverage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Session:
    """Answers the coverage script's two reads: assertions, then Views."""

    def __init__(self, assertions, views):
        self._assertions, self._views = assertions, views

    def run(self, query, **_params):
        return self._assertions if "TypedAssertion" in query else self._views


def test_harness_matches_the_live_seconds_view():
    result = _validate_case(COMMUTE, [LIVE_ASSERTION], [LIVE_VIEW], [])
    assert result.matched


def test_harness_rejects_a_minutes_valued_view():
    minutes = {**LIVE_VIEW, "ss_value": "45", "ss_display": "45", "value": 45.0}
    assert not _validate_case(COMMUTE, [], [minutes], []).matched


def test_coverage_reaches_every_stage_for_the_live_rows():
    cov = _coverage_module()
    rows = cov.coverage_for_namespace(_Session([LIVE_ASSERTION], [LIVE_VIEW]), "ns", "default")
    commute = next(r for r in rows if r["case_id"] == "ss-duration-commute")
    assert all(commute[stage] for stage in cov.STAGES)
    assert commute["drops_at"] is None
