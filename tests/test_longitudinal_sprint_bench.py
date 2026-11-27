"""Tests for the 3-arm x 4-session Longitudinal Frozen-Repo Coding-Sprint harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archolith_bench.beacon_eval.runner import EventLog, RateLimited
from archolith_bench.engineering_evidence.live_executor import (
    LiveSprintConfig,
    OpenCodeSprintExecutor,
    apply_drift_patch,
    parse_sprint_tool_telemetry,
)
from archolith_bench.engineering_evidence.sprint import (
    SPRINT_ARMS,
    DriftPatch,
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


def test_apply_drift_patch_modifies_file_and_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "llm" / "default_plugins" / "openai_models.py"
    target.parent.mkdir(parents=True)
    target.write_text("# GPT-5.5 — routes through Responses\n", encoding="utf-8")

    patch = DriftPatch(
        path="llm/default_plugins/openai_models.py",
        find="routes through Responses",
        replace="uses LegacyChat (DEPRECATED fallback)",
    )
    apply_drift_patch(tmp_path, patch)
    assert "DEPRECATED fallback" in target.read_text(encoding="utf-8")

    # Applying again must fail closed because the anchor text no longer matches
    with pytest.raises(ValueError, match="anchor text not found"):
        apply_drift_patch(tmp_path, patch)

    # Path traversal outside repo root must fail closed
    with pytest.raises(ValueError, match="escapes repo root"):
        apply_drift_patch(tmp_path, DriftPatch(path="../escape.py", find="a", replace="b"))


def test_parse_sprint_tool_telemetry_extracts_reads_and_menhir_tools(tmp_path: Path) -> None:
    events = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "read",
                        "state": {"input": {"filePath": str(tmp_path / "docs" / "changelog.md")}},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {
                        "tool": "menhir_blast_radius",
                        "state": {"input": {"file_paths": ["llm/default_plugins/openai_models.py"]}},
                    },
                }
            ),
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {"tool": "menhir_add_memory", "state": {"input": {"text": "invariant"}}},
                }
            ),
            json.dumps(
                {
                    "type": "tool_use",
                    "part": {"tool": "menhir_add_todo", "state": {"input": {"content": "add test"}}},
                }
            ),
        ]
    )
    tel = parse_sprint_tool_telemetry(events, repo_root=tmp_path)
    assert tel["files_read"] == ["docs/changelog.md"]
    assert tel["memories_saved"] == 1
    assert tel["todos_saved"] == 1
    assert tel["menhir_tools"] == {"blast_radius": 1, "add_memory": 1, "add_todo": 1}


def test_live_executor_applies_drift_patch_and_stops_fail_closed_on_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "archolith_bench.beacon_eval.isolation.default_data_source",
        lambda: tmp_path / "empty_opencode_data",
    )
    src_cfg = tmp_path / "opencode.json"
    src_cfg.write_text(json.dumps({"provider": {"openai": {"options": {}}}}), encoding="utf-8")

    def fake_export(pin, dest: Path, cache: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        f = dest / "llm" / "default_plugins" / "openai_models.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# GPT-5.5 — routes through the Responses API by default; pass\n", encoding="utf-8")
        return dest

    def fake_stream_429(cmd, prompt, cwd, env, run_dir: Path, reserve, timeout_s):
        log = EventLog()
        log.rate_limited = True
        return log, "rate_limited", 1

    fixture = SprintFixture.from_file(FIXTURE_PATH)
    executor = OpenCodeSprintExecutor(
        LiveSprintConfig(
            work_dir=tmp_path / "work",
            cache_dir=tmp_path / "cache",
            source_config=src_cfg,
            menhir_mcp_url="http://127.0.0.1:9999/mcp",
        ),
        stream_fn=fake_stream_429,
        export_fn=fake_export,
        resolve_cmd_fn=lambda: ["opencode"],
        check_memory_fn=lambda url: None,
    )

    with pytest.raises(RateLimited):
        executor.run_session(
            fixture=fixture,
            session=fixture.sessions[2],  # Session 3 has drift_patch
            arm="menhir_longitudinal",
            session_index=2,
            namespace="sprint-test",
        )
    # Confirm drift patch was applied to the isolated checkout before the session ran
    patched = (
        tmp_path
        / "work"
        / "checkouts"
        / f"{fixture.name}-menhir_longitudinal"
        / "llm"
        / "default_plugins"
        / "openai_models.py"
    )
    assert "DEPRECATED fallback: legacy_chat" in patched.read_text(encoding="utf-8")
