"""Focused tests for the paired clean/noisy scalar identity acceptance panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from archolith_bench.scalar_identity_noisy_panel import (
    analyze_panel,
    load_panel_menhir_api,
    render_markdown,
)


def _menhir_root() -> Path:
    candidates = []
    if os.environ.get("MENHIR_ROOT"):
        candidates.append(Path(os.environ["MENHIR_ROOT"]))
    candidates.extend((Path.cwd() / "menhir", Path.cwd().parent / "menhir"))
    for candidate in candidates:
        if (candidate / "src" / "menhir" / "services" / "research_scalar_adapter.py").is_file():
            return candidate.resolve()
    pytest.skip("set MENHIR_ROOT or provide a local ../menhir sibling")


@pytest.fixture(scope="module")
def api():
    return load_panel_menhir_api(_menhir_root())


def test_noisy_panel_reports_clean_noisy_slices_and_pairs(api):
    fixture = Path("fixtures/scalar_identity_noisy_v1.json").resolve()
    report = analyze_panel(fixture, menhir_root=_menhir_root(), generated_at="2026-08-06T00:00:00+00:00")
    aggregate = report["aggregate"]
    assert aggregate["cases_total"] == 30
    assert aggregate["slices"]["clean"]["total"] == 15
    assert aggregate["slices"]["noisy"]["total"] == 15
    assert aggregate["slices"]["clean"]["correct"] == 15
    assert aggregate["slices"]["clean"]["composed"] == 15
    assert aggregate["slices"]["noisy"]["composed"] == 2
    # 12 -> 10 as of menhir 4ecf5e8a ("canonicalize typed scalar identity",
    # 2026-09-06), which made counts source-authoritative and integer-only with
    # fail-closed behavior on ambiguous source constraints. Two noisy count cases
    # (current-books-3-noisy, -3b-noisy) now reject at parse with
    # `count_value_unresolved` instead of being admitted. This is a deliberate
    # accuracy-for-safety trade in menhir, not a bench regression: the clean slice
    # is unaffected at 15/15.
    assert aggregate["slices"]["noisy"]["correct"] == 10
    assert aggregate["false_current_state_errors"] == 0
    assert aggregate["coverage"]["perturbations_total"] == 15
    assert aggregate["coverage"]["paired_perturbations"] == 15
    assert aggregate["coverage"]["clean_cases"] == 15
    assert aggregate["coverage"]["noisy_cases"] == 15
    assert aggregate["paired_invariance"] == {
        "pairs_total": 15,
        "expected_invariant_pairs": 5,
        "actual_invariant_pairs": 2,
        "invariance_correct": 2,
    }
    misspelling = next(row for row in report["cases"] if row["case_id"] == "current-books-6-noisy")
    assert misspelling["false_current_state_error"] is False
    assert misspelling["actual"]["composition_status"] == "abstained"
    assert misspelling["actual"]["target"] is None
    assert misspelling["mismatch_dimensions"] == []


def test_noisy_report_is_source_free_and_not_evaluable(api):
    fixture = Path("fixtures/scalar_identity_noisy_v1.json").resolve()
    report = analyze_panel(fixture, menhir_root=_menhir_root())
    assert report["promotion_status"] == "not_evaluable"
    encoded = json.dumps(report)
    assert "I have 12 books" not in encoded
    markdown = render_markdown(report)
    assert "scalar-identity-noisy-v1" not in markdown
    assert "not_evaluable" in markdown
