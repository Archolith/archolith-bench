"""Focused tests for the opt-in isolated scalar adapter comparison lane."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import archolith_bench.scalar_identity_isolated_comparison as comparison
from archolith_bench.compositional_scalar_panel import PanelError
from archolith_bench.scalar_identity_isolated_comparison import (
    analyze_comparison,
    load_panel_menhir_api,
    render_markdown,
)


def test_isolated_loader_fails_loudly_when_api_is_unavailable(monkeypatch):
    def missing(_name):
        raise ImportError("missing isolated adapter")

    monkeypatch.setattr(comparison.importlib, "import_module", missing)
    with pytest.raises(PanelError, match="isolated research adapter unavailable"):
        comparison._load_isolated_adapter(Path("missing-menhir-root"))


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


def test_isolated_comparison_reports_separate_paths(api):
    fixture = Path("fixtures/scalar_identity_noisy_v1.json").resolve()
    report = analyze_comparison(
        fixture,
        menhir_root=_menhir_root(),
        generated_at="2026-08-06T00:00:00+00:00",
    )
    aggregate = report["aggregate"]
    assert report["promotion_status"] == "not_evaluable"
    assert report["provenance"]["baseline_adapter_version"] == "research-adapter-v1"
    assert report["provenance"]["isolated_adapter_version"] == "research-adapter-isolated-v2"
    assert aggregate["cases_total"] == 30
    assert aggregate["baseline"]["slices"]["clean"]["correct"] == 15
    # Both paths lost the same two noisy count cases (current-books-3-noisy,
    # -3b-noisy) to menhir 4ecf5e8a's fail-closed `count_value_unresolved`
    # rejection: baseline 12 -> 10, isolated 15 -> 13. The isolated adapter's
    # advantage is unchanged (composition_gains still 3 on the noisy slice), so
    # the comparison this test exists to make is unaffected.
    assert aggregate["baseline"]["slices"]["noisy"]["correct"] == 10
    assert aggregate["isolated"]["slices"]["clean"]["correct"] == 15
    assert aggregate["isolated"]["slices"]["noisy"]["correct"] == 13
    assert aggregate["composition_gains"] == {"clean": 0, "noisy": 3, "total": 3}
    assert aggregate["identity_mismatches"] == {
        "total": 3,
        "case_ids": [
            "current-cats-5-noisy",
            "current-records-7-noisy",
            "current-plants-9-noisy",
        ],
    }
    assert aggregate["false_current_state_errors"] == {"baseline": 0, "isolated": 0}


def test_isolated_comparison_report_is_source_free(api):
    fixture = Path("fixtures/scalar_identity_noisy_v1.json").resolve()
    report = analyze_comparison(fixture, menhir_root=_menhir_root())
    encoded = json.dumps(report)
    assert "I have 12 books" not in encoded
    markdown = render_markdown(report)
    assert "scalar-identity-noisy-v1" not in markdown
    assert "not_evaluable" in markdown
