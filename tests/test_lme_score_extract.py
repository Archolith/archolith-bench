"""Tests for per-arm score extraction from the harness checkpoint.

This turns a rendered markdown table into the number the ledger cites, so the cases here
are the ways it could publish a wrong one: picking an arm for the reader, double-counting a
resumed task, reporting 0.0 for a run that never scored, or silently averaging arms that
measured different item counts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score_extract = _load_script(
    "lme_score_extract_test", "scripts/longmemeval/lib/score_extract.py"
)


def _checkpoint(run: Path, rows: list[dict], *, at_root: bool = False) -> Path:
    target = run if at_root else run / "harness_recall"
    target.mkdir(parents=True, exist_ok=True)
    path = target / ".checkpoint_longmemeval-menhir_oracle_gpt-4o.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _row(arm: str, task: str, correct: bool, **extra) -> dict:
    return {"arm": arm, "task_id": task, "result": {"task_id": task, "correct": correct, **extra}}


# ---------------------------------------------------------------------------
# Arm handling
# ---------------------------------------------------------------------------

def test_every_arm_is_reported(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _checkpoint(run, [
        _row("no_memory", "t1", False), _row("no_memory", "t2", False),
        _row("menhir_recall", "t1", True), _row("menhir_recall", "t2", False),
        _row("menhir_value_recall", "t1", True), _row("menhir_value_recall", "t2", True),
    ])
    payload = score_extract.extract(run)
    assert set(payload["arms"]) == {"no_memory", "menhir_recall", "menhir_value_recall"}
    assert payload["arms"]["menhir_recall"]["score"] == pytest.approx(0.5)
    assert payload["arms"]["menhir_value_recall"]["score"] == pytest.approx(1.0)
    assert payload["arms"]["no_memory"]["score"] == pytest.approx(0.0)


def test_primary_arm_is_never_chosen_automatically(tmp_path: Path) -> None:
    """value-arm-verify-20260717 records its value arm (0.679), not menhir_recall (0.333).

    Picking by name order would have published the wrong number for that run, so the choice
    belongs to the ledger row.
    """
    run = tmp_path / "run-a"
    _checkpoint(run, [
        _row("menhir_recall", "t1", False),
        _row("menhir_value_recall", "t1", True),
    ])
    assert score_extract.extract(run)["primary_arm"] is None


def test_arm_item_counts_are_reported_separately(tmp_path: Path) -> None:
    """An arm scored on 41 items is not comparable to one scored on 78."""
    run = tmp_path / "run-a"
    _checkpoint(run, [
        _row("no_memory", "t1", False), _row("no_memory", "t2", True),
        _row("menhir_recall", "t1", True),
    ])
    payload = score_extract.extract(run)
    assert payload["arms"]["no_memory"]["n"] == 2
    assert payload["arms"]["menhir_recall"]["n"] == 1


# ---------------------------------------------------------------------------
# Resume safety
# ---------------------------------------------------------------------------

def test_a_rescored_task_is_not_double_counted(tmp_path: Path) -> None:
    """--resume appends, so a re-scored task appears twice; the later verdict wins.

    Double-counting would move the score: a task scored wrong then right would otherwise
    count as 1 of 2 instead of 1 of 1.
    """
    run = tmp_path / "run-a"
    _checkpoint(run, [
        _row("menhir_recall", "t1", False),
        _row("menhir_recall", "t1", True),
    ])
    payload = score_extract.extract(run)
    assert payload["arms"]["menhir_recall"]["n"] == 1
    assert payload["arms"]["menhir_recall"]["score"] == pytest.approx(1.0)
    assert payload["duplicate_entries_collapsed"] == 1


def test_no_duplicates_reports_zero_collapsed(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _checkpoint(run, [_row("menhir_recall", "t1", True)])
    assert score_extract.extract(run)["duplicate_entries_collapsed"] == 0


# ---------------------------------------------------------------------------
# Absence is not zero
# ---------------------------------------------------------------------------

def test_a_run_with_no_checkpoint_raises(tmp_path: Path) -> None:
    """An aborted launch must not be publishable as a 0.0 result."""
    run = tmp_path / "run-a"
    run.mkdir()
    with pytest.raises(score_extract.ScoreError, match="no .* to read"):
        score_extract.extract(run)


def test_a_checkpoint_with_no_verdicts_raises(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _checkpoint(run, [{"arm": "menhir_recall", "task_id": "t1", "result": {"task_id": "t1"}}])
    with pytest.raises(score_extract.ScoreError, match="no scored results"):
        score_extract.extract(run)


def test_malformed_lines_are_counted_not_fatal(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    target = run / "harness_recall"
    target.mkdir(parents=True)
    (target / ".checkpoint_x.jsonl").write_text(
        json.dumps(_row("menhir_recall", "t1", True)) + "\n{broken\n", encoding="utf-8"
    )
    payload = score_extract.extract(run)
    assert payload["malformed_lines"] == 1
    assert payload["arms"]["menhir_recall"]["n"] == 1


def test_a_row_without_a_task_id_is_malformed_not_counted(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _checkpoint(run, [{"arm": "menhir_recall", "result": {"correct": True}}])
    with pytest.raises(score_extract.ScoreError, match="no scored results"):
        score_extract.extract(run)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_checkpoint_at_run_root_is_found(tmp_path: Path) -> None:
    """Rescore directories keep the checkpoint at the run root, not under harness_recall/."""
    run = tmp_path / "run-a"
    run.mkdir()
    _checkpoint(run, [_row("menhir_recall", "t1", True)], at_root=True)
    assert score_extract.extract(run)["arms"]["menhir_recall"]["n"] == 1


def test_harness_subdir_wins_over_root(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    run.mkdir()
    _checkpoint(run, [_row("root_arm", "t1", True)], at_root=True)
    _checkpoint(run, [_row("subdir_arm", "t1", True)])
    assert set(score_extract.extract(run)["arms"]) == {"subdir_arm"}


def test_tokens_are_summed_per_arm(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    _checkpoint(run, [
        _row("menhir_recall", "t1", True, input_tokens=100, output_tokens=10),
        _row("menhir_recall", "t2", True, input_tokens=200, output_tokens=20),
    ])
    arm = score_extract.extract(run)["arms"]["menhir_recall"]
    assert arm["input_tokens"] == 300
    assert arm["output_tokens"] == 30


# ---------------------------------------------------------------------------
# Reproduces the hand-typed history
# ---------------------------------------------------------------------------

CANONICAL = ROOT / "results" / "lme-ku-buildout" / "scalar-canonical-ku78-v1-20260806"


@pytest.mark.skipif(not score_extract.find_checkpoints(CANONICAL),
                    reason="canonical run checkpoints not on disk")
def test_reproduces_the_canonical_runs_recorded_score() -> None:
    """LEDGER.md records 0.872 for this run, with the note "68/78 recall vs 6/78 (0.077)".

    If extraction ever drifts from the number the ledger cites, this is where it shows up.
    """
    payload = score_extract.extract(CANONICAL)
    recall = payload["arms"]["menhir_recall"]
    assert (recall["correct"], recall["n"]) == (68, 78)
    assert recall["score"] == pytest.approx(0.872, abs=0.0005)
    baseline = payload["arms"]["no_memory"]
    assert (baseline["correct"], baseline["n"]) == (6, 78)
    assert baseline["score"] == pytest.approx(0.077, abs=0.0005)
