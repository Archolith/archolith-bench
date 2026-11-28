"""sprint-v2.1 prompt set: the memory arm is prompted like the notes arm, with tools that exist."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from archolith_bench.engineering_evidence.sprint import (
    PROMPT_SET,
    SprintFixture,
    compose_full_prompt,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = [ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json", ROOT / "fixtures" / "sprint_smolagents_227ef5e4.json"]

#: Menhir MENHIR_LOOPBACK_PROFILE=lean tool set (menhir src/menhir/mcp/local_profile.py LEAN_TOOLS).
LEAN_TOOLS = {"recall_memories", "recall_context_memories", "read_flagged_memories", "add_memory",
              "query_structure", "get_provenance", "list_todos", "get_todo", "add_todo", "close_todo"}
#: Words in backticks that are not tools: query_structure query types and recall fields.
NOT_TOOLS = {"blast_radius", "affected_tests", "context", "stale_anchor"}
#: The notes arm is told to write its notes in these words; the memory arm must then be told to save.
NOTES_WRITE = re.compile(r"\brecord\b|add what you learn|update the note", re.IGNORECASE)


def _identifiers(text: str) -> set[str]:
    """Backticked snake_case words (paths, file names and other spans are skipped)."""
    return {w for w in re.findall(r"`([a-z][a-z0-9_]*)`", text) if "_" in w}


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
def test_menhir_prompts_name_only_lean_tools_and_query_types(fixture_path: Path) -> None:
    fixture = SprintFixture.from_file(fixture_path)
    for session in fixture.sessions:
        prompt = compose_full_prompt(session, "menhir_longitudinal", "ns", project=fixture.repo.name)
        # The task text names the repo's own identifiers (e.g. `chat_completions`); check the rest.
        menhir_text = prompt.replace(session.prompt.strip(), "")
        unknown = _identifiers(menhir_text) - LEAN_TOOLS - NOT_TOOLS
        assert not unknown, f"{session.session_id} names tools the lean profile lacks: {sorted(unknown)}"
        # sprint-v2 presented blast_radius / affected_tests as tools; they are query types.
        assert "`query_structure` with query_type `blast_radius`" in prompt


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
def test_memory_arm_is_told_to_save_wherever_the_notes_arm_is(fixture_path: Path) -> None:
    fixture = SprintFixture.from_file(fixture_path)
    for session in fixture.sessions:
        if NOTES_WRITE.search(session.notes_instruction):
            assert "add_memory" in session.memory_instruction, session.session_id
    note = compose_full_prompt(fixture.sessions[0], "menhir_longitudinal", "ns")
    assert "before finishing store what a later session would need with `add_memory`" in note


@pytest.mark.parametrize("arm,expected", [
    ("menhir_longitudinal", True), ("menhir_structure", True), ("notes_file", False), ("no_menhir", False),
])
def test_menhir_arms_name_the_index_project(arm: str, expected: bool) -> None:
    fixture = SprintFixture.from_file(FIXTURES[0])
    line = "The Menhir structural index for this repository is project `llm`."
    for shaped in (False, True):
        prompt = compose_full_prompt(fixture.sessions[0], arm, "ns", shaped, project="llm")
        assert (line in prompt) is expected
    assert "structural index" not in compose_full_prompt(fixture.sessions[0], arm, "ns")  # no project, no line


def test_dry_run_records_the_prompt_set(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("sprint_cli", ROOT / "scripts" / "run_longitudinal_sprint_bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--fixture", str(FIXTURES[0]), "--repeats", "1", "--max-sessions", "1",
                     "--run-id", "ps", "--out-root", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("*-ps/report.json")).read_text(encoding="utf-8"))
    assert PROMPT_SET == "sprint-v2.1" and report["provenance"]["prompt_set"] == PROMPT_SET
