"""Focused tests for the benchmark-agnostic research scalar identity acceptance panel."""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from archolith_bench.scalar_identity_acceptance_panel import (
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
        # The panel loads the deterministic extractor, removed from Menhir main on 2026-10-27.
        if (candidate / "src" / "menhir" / "services" / "deterministic_scalar_extractor.py").is_file():
            return candidate.resolve()
    pytest.skip("set MENHIR_ROOT to a Menhir checkout of archive/bucket1-features-2026-10-27")


@pytest.fixture(scope="module")
def api():
    return load_panel_menhir_api(_menhir_root())


def test_registered_acceptance_panel_scores_adapter_and_composer(api):
    fixture = Path("fixtures/scalar_identity_acceptance_v1.json").resolve()
    report = analyze_panel(
        fixture,
        menhir_root=_menhir_root(),
        generated_at="2026-08-06T00:00:00+00:00",
        api=api,
    )

    aggregate = report["aggregate"]
    assert aggregate["total"] == 15
    assert aggregate["parse_status_correct"] == 15
    assert aggregate["correct"] == 15
    assert aggregate["all_correct"] is True
    assert {row["case_id"] for row in report["cases"] if not row["correct"]} == set()
    assert {
        "current-total", "subset", "delta", "event-amount", "historical", "correction",
        "contraction", "modality", "weak-target", "ambiguous-identity",
    } <= set(aggregate["by_role"])
    assert report["provenance"]["llm_used"] is False
    assert report["provenance"]["research_adapter_version"] == "research-adapter-v1"


def test_acceptance_report_is_source_free(api):
    fixture = Path("fixtures/scalar_identity_acceptance_v1.json").resolve()
    report = analyze_panel(fixture, menhir_root=_menhir_root(), api=api)
    encoded = json.dumps(report)
    assert "I have 12 books" not in encoded
    assert "scalar-identity-acceptance-v1" not in render_markdown(report)
