"""Tests for the 3-arm x 4-session Longitudinal Frozen-Repo Coding-Sprint harness."""

from __future__ import annotations

from pathlib import Path

from archolith_bench.engineering_evidence.sprint import (
    SPRINT_ARMS,
    SprintFixture,
    build_session_prompt,
    run_longitudinal_sprint,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sprint_simonw_llm_764dc386.json"
)


def test_simonw_llm_sprint_fixture_loads_with_drift_patch() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    assert fixture.repo.name == "llm"
    assert fixture.repo.commit == "764dc386c58b625f3ad9d203e699715ad208455f"
    assert len(fixture.sessions) == 4
    assert fixture.sessions[2].phase == "drift_check"
    assert fixture.sessions[2].drift_patch is not None
    assert fixture.sessions[2].drift_patch.path == "llm/default_plugins/openai_models.py"


def test_build_session_prompt_scopes_tool_guidance_by_arm() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    s1 = fixture.sessions[0]
    p_base = build_session_prompt(s1, "no_menhir")
    p_struct = build_session_prompt(s1, "menhir_structure")
    p_long = build_session_prompt(s1, "menhir_longitudinal")

    assert "blast_radius" not in p_base
    assert "blast_radius" in p_struct
    assert "add_memory" not in p_struct
    assert "add_memory" in p_long
    assert "recall_memories" in p_long


def test_dry_run_longitudinal_sprint_scores_and_tracks_downstream_tokens() -> None:
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    report = run_longitudinal_sprint(fixture, dry_run=True)

    assert report["dry_run"] is True
    assert set(report["arms"].keys()) == set(SPRINT_ARMS)
    for arm in SPRINT_ARMS:
        arm_rep = report["arms"][arm]
        assert len(arm_rep["sessions"]) == 4
        assert arm_rep["mean_accuracy"] == 1.0

    # Verify longitudinal arm records memories/todos and tracks token & redundant-read deltas
    long_rep = report["arms"]["menhir_longitudinal"]
    assert long_rep["total_memories_saved"] == 3
    assert long_rep["total_todos_saved"] == 1
    cmp_long = report["comparisons_vs_no_menhir"]["menhir_longitudinal"]
    assert cmp_long["downstream_tokens_delta_pct"] < 0
    assert cmp_long["redundant_reads_delta"] < 0
