"""Tests for the deterministic S4 + S5 Engineering Evidence Graph benchmark."""

from __future__ import annotations

from archolith_bench.engineering_evidence.runner import (
    CONDITIONS,
    PILLARS,
    run_engineering_evidence_benchmark,
)


def test_engineering_evidence_benchmark_all_pillars_pass() -> None:
    report = run_engineering_evidence_benchmark()

    assert report["suite"] == "engineering_evidence_s4_s5"
    assert report["zero_llm_calls"] is True
    assert tuple(report["conditions"]) == CONDITIONS
    assert [p["pillar"] for p in report["pillars"]] == list(PILLARS)

    for pillar in report["pillars"]:
        menhir_score = pillar["conditions"]["menhir_s4_s5"]["composite"]
        naive_score = pillar["conditions"]["naive_baseline"]["composite"]
        assert menhir_score == 1.0, f"{pillar['pillar']} failed: {pillar}"
        assert menhir_score > naive_score, f"{pillar['pillar']} did not beat baseline"

    assert report["summary"]["menhir_s4_s5"] == 1.0
    assert report["delta"] > 0.7


def test_engineering_evidence_benchmark_is_deterministic() -> None:
    r1 = run_engineering_evidence_benchmark()
    r2 = run_engineering_evidence_benchmark()
    assert r1 == r2


def test_live_repo_scan_pillar_joins_code_memories_and_todos() -> None:
    report = run_engineering_evidence_benchmark()
    p5 = next(p for p in report["pillars"] if p["pillar"] == "live_repo_scan_to_impact")
    details = p5["details"]

    assert details["files_indexed"] == details["files_eligible"]
    assert details["imports_extracted"] >= 3
    assert details["call_edges_extracted"] >= 2
    assert details["test_files"] == ["tests/test_ledger.py"]
    assert details["related_memories"] == ["mem-ledger-inv-1"]
    assert details["open_todos"] == ["todo-ledger-1"]
    assert any(
        c["caller"] == "process_payment" and c["callee"] == "record_entry"
        for c in details["function_callers"]
    )
