"""Tests for the machine-readable buildout scoreboard.

The ledger is the record other people cite, so the cases here are the ways it could assert
something untrue: a score present on a run that never produced one, a status outside the
vocabulary, two rows both claiming to be current canonical evidence, a row claiming a
results directory that is not there, a run on disk that never reached the scoreboard, and a
render that silently eats the hand-written analysis around the table.
"""

from __future__ import annotations

import csv
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


ledger = _load_script("lme_ledger_test", "scripts/longmemeval/lib/ledger.py")


def _row(**overrides) -> dict[str, str]:
    base = {
        "run_id": "run-a",
        "date": "2026-08-09",
        "items_scored": "78",
        "items_total": "78",
        "segmentation": "adaptive",
        "score": "0.900",
        "score_raw": "0.900",
        "status": "scored",
        "extract_model": "gpt-4o-mini",
        "canonical": "",
        "has_results_dir": "false",
        "notes": "",
    }
    base.update({key: str(value) for key, value in overrides.items()})
    return base


def _write(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    csv_path = tmp_path / "ledger.csv"
    ledger.write_rows(rows, csv_path)
    return csv_path


def _validate(tmp_path: Path, rows: list[dict[str, str]], results: Path | None = None):
    csv_path = _write(tmp_path, rows)
    results_dir = results if results is not None else tmp_path / "empty_results"
    results_dir.mkdir(exist_ok=True)
    return ledger.validate(csv_path, results_dir)


def _messages(findings, level: str) -> list[str]:
    return [f["message"] for f in findings if f["level"] == level]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_csv_round_trips_a_value_containing_a_comma(tmp_path: Path) -> None:
    """Real run labels contain commas; naive splitting would corrupt the row."""
    rows = [_row(run_id="recall-menhir (prod graph, all subsets)")]
    csv_path = _write(tmp_path, rows)
    read_back = ledger.read_rows(csv_path)
    assert read_back[0]["run_id"] == "recall-menhir (prod graph, all subsets)"


def test_read_rows_rejects_a_csv_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "score"])
        writer.writeheader()
        writer.writerow({"run_id": "x", "score": "0.5"})
    with pytest.raises(ledger.LedgerError, match="missing columns"):
        ledger.read_rows(csv_path)


def test_read_rows_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ledger.LedgerError, match="not found"):
        ledger.read_rows(tmp_path / "absent.csv")


# ---------------------------------------------------------------------------
# A score must never appear without a status that licenses it
# ---------------------------------------------------------------------------

def test_scored_status_without_a_score_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(score="")])
    assert any("no score is recorded" in m for m in _messages(findings, "FAIL"))


def test_score_on_a_killed_run_fails(tmp_path: Path) -> None:
    """A killed run has no comparable number; carrying one would invent a result."""
    findings = _validate(tmp_path, [_row(status="killed", score="0.5")])
    assert any("implies status 'scored'" in m for m in _messages(findings, "FAIL"))


def test_multi_arm_row_must_not_carry_a_single_score(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(status="multi_arm", score="0.667")])
    assert any("implies status 'scored'" in m for m in _messages(findings, "FAIL"))


def test_multi_arm_row_with_only_raw_is_clean(tmp_path: Path) -> None:
    findings = _validate(
        tmp_path,
        [_row(status="multi_arm", score="", score_raw="v2c 0.667 / v2h 0.679")],
    )
    assert _messages(findings, "FAIL") == []


def test_unparseable_score_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(score="about 0.9")])
    assert any("is not a number" in m for m in _messages(findings, "FAIL"))


@pytest.mark.parametrize("bad", ["1.5", "-0.2", "87.2"])
def test_score_outside_the_unit_interval_fails(tmp_path: Path, bad: str) -> None:
    """87.2 is the percentage-vs-fraction slip; it must not pass as a score."""
    findings = _validate(tmp_path, [_row(score=bad)])
    assert any("outside [0, 1]" in m for m in _messages(findings, "FAIL"))


def test_status_outside_the_vocabulary_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(status="probably fine")])
    assert any("closed vocabulary" in m for m in _messages(findings, "FAIL"))


# ---------------------------------------------------------------------------
# Item counts
# ---------------------------------------------------------------------------

def test_items_scored_above_items_total_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(items_scored="80", items_total="78")])
    assert any("exceeds items_total" in m for m in _messages(findings, "FAIL"))


def test_partially_scored_run_warns_that_the_score_is_not_comparable(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(items_scored="53", items_total="78")])
    assert any("not comparable to a full run" in m for m in _messages(findings, "WARN"))


# ---------------------------------------------------------------------------
# Canonical evidence
# ---------------------------------------------------------------------------

def test_two_current_canonical_rows_fail(tmp_path: Path) -> None:
    """Only one run can be the current canonical evidence; two is an unresolved claim."""
    rows = [
        _row(run_id="run-a", canonical="current"),
        _row(run_id="run-b", canonical="current"),
    ]
    findings = _validate(tmp_path, rows)
    assert any("more than one row claims" in m for m in _messages(findings, "FAIL"))


def test_one_current_canonical_row_is_clean(tmp_path: Path) -> None:
    rows = [_row(run_id="run-a", canonical="current"), _row(run_id="run-b")]
    assert _messages(_validate(tmp_path, rows), "FAIL") == []


def test_canonical_claim_on_an_unscored_run_fails(tmp_path: Path) -> None:
    findings = _validate(
        tmp_path, [_row(canonical="current", status="aborted", score="")]
    )
    assert any("claims current canonical" in m for m in _messages(findings, "FAIL"))


def test_no_canonical_row_warns(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row()])
    assert any("no row claims current canonical" in m for m in _messages(findings, "WARN"))


def test_invalid_canonical_value_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(canonical="maybe")])
    assert any("canonical" in m and "must be one of" in m for m in _messages(findings, "FAIL"))


def test_duplicate_run_id_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(run_id="dup"), _row(run_id="dup")])
    assert any("duplicate run_id" in m for m in _messages(findings, "FAIL"))


def test_blank_run_id_fails(tmp_path: Path) -> None:
    findings = _validate(tmp_path, [_row(run_id="")])
    assert any("no run_id" in m for m in _messages(findings, "FAIL"))


# ---------------------------------------------------------------------------
# Cross-check against what is on disk
# ---------------------------------------------------------------------------

def test_score_only_checkout_does_not_claim_full_run_evidence(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "another-run"
    run.mkdir(parents=True)
    (run / "score.json").write_text("{}", encoding="utf-8")
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert not _messages(findings, "FAIL")
    assert any("no full run directories" in m for m in _messages(findings, "WARN"))


def test_claiming_a_results_directory_that_is_absent_fails(tmp_path: Path) -> None:
    # The tree must be materialized for this to be a contradiction rather than "cannot
    # check" -- an empty results/ is a fresh checkout, where the evidence was never
    # committed. See test_a_checkout_without_the_evidence_tree_does_not_fail.
    results = tmp_path / "results"
    (results / "another-run").mkdir(parents=True)
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("claims a results directory" in m for m in _messages(findings, "FAIL"))


def test_denying_a_results_directory_that_exists_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    (results / "run-a").mkdir(parents=True)
    findings = _validate(tmp_path, [_row(has_results_dir="false")], results=results)
    assert any("claims no results directory" in m for m in _messages(findings, "FAIL"))


def test_provenance_recording_a_different_run_id_fails(tmp_path: Path) -> None:
    """A pasted-in provenance file from another run makes the row unattributable."""
    results = tmp_path / "results"
    run = results / "run-a"
    run.mkdir(parents=True)
    (run / "run_provenance.json").write_text(
        json.dumps({"run_id": "some-other-run"}), encoding="utf-8"
    )
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("provenance records run_id" in m for m in _messages(findings, "FAIL"))


def test_unreadable_provenance_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "run-a"
    run.mkdir(parents=True)
    (run / "run_provenance.json").write_text("{not json", encoding="utf-8")
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("unreadable" in m for m in _messages(findings, "FAIL"))


def test_dirty_run_warns(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "run-a"
    run.mkdir(parents=True)
    (run / "run_provenance.json").write_text(
        json.dumps({"run_id": "run-a", "latest_attempt": {"menhir_dirty": True}}),
        encoding="utf-8",
    )
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("DIRTY" in m for m in _messages(findings, "WARN"))


def test_missing_surface_fingerprint_warns(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "run-a"
    run.mkdir(parents=True)
    (run / "run_provenance.json").write_text(
        json.dumps({"run_id": "run-a", "latest_attempt": {}}), encoding="utf-8"
    )
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("no surface fingerprint" in m for m in _messages(findings, "WARN"))


def test_a_run_on_disk_with_no_ledger_row_is_reported(tmp_path: Path) -> None:
    """Rows -> disk is the easy direction. A run that executed and never reached the
    scoreboard is the one that silently drops out of comparisons."""
    results = tmp_path / "results"
    orphan = results / "ran-but-unlisted"
    orphan.mkdir(parents=True)
    (orphan / "run_provenance.json").write_text(
        json.dumps({"run_id": "ran-but-unlisted"}), encoding="utf-8"
    )
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert any("no ledger row" in m for m in _messages(findings, "WARN"))


def test_an_orphan_with_a_readable_score_says_so(tmp_path: Path) -> None:
    results = tmp_path / "results"
    orphan = results / "scored-but-unlisted"
    orphan.mkdir(parents=True)
    (orphan / "run_provenance.json").write_text(json.dumps({}), encoding="utf-8")
    (orphan / "results.json").write_text(json.dumps({"score": 0.77}), encoding="utf-8")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    # A run with no score.json still falls back to the legacy results.json score.
    assert any(
        "a scored result exists" in m and "0.77" in m
        for m in _messages(findings, "WARN")
    )


def test_a_directory_without_provenance_is_not_treated_as_a_run(tmp_path: Path) -> None:
    """Survey and analysis output directories live alongside runs and are not runs."""
    results = tmp_path / "results"
    (results / "reflection-rescue-survey").mkdir(parents=True)
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert not any("no ledger row" in m for m in _messages(findings, "WARN"))


# ---------------------------------------------------------------------------
# extract_score
# ---------------------------------------------------------------------------

def test_extract_score_ignores_manifest_only_directories(tmp_path: Path) -> None:
    """manifest.json proves ingest ran, not that a score exists. Treating it as score
    evidence over-flags build-only and diagnostic runs."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps([{"question_id": "q"}]), encoding="utf-8")
    assert ledger.extract_score(run) is None


def test_extract_score_reads_a_nested_score(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(
        json.dumps({"summary": {"accuracy": 0.42}}), encoding="utf-8"
    )
    assert ledger.extract_score(run) == pytest.approx(0.42)


def test_extract_score_does_not_mistake_a_boolean_for_a_score(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(
        json.dumps({"score_verified": True, "score": 0.3}), encoding="utf-8"
    )
    assert ledger.extract_score(run) == pytest.approx(0.3)


def test_extract_score_returns_none_on_malformed_json(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text("{broken", encoding="utf-8")
    assert ledger.extract_score(run) is None


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _markdown(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "LEDGER.md"
    path.write_text(body, encoding="utf-8")
    return path


def test_render_replaces_only_the_marked_block(tmp_path: Path) -> None:
    """The prose around the table is hand-written analysis; the generator must not touch it."""
    body = (
        "# Ledger\n\n## Scoreboard\n\n"
        f"{ledger.BEGIN_MARKER}\n\nOLD TABLE\n\n{ledger.END_MARKER}\n\n"
        "## Regression Analysis\n\nIrreplaceable prose.\n"
    )
    markdown = _markdown(tmp_path, body)
    csv_path = _write(tmp_path, [_row()])
    ledger.render(csv_path, markdown)
    updated = markdown.read_text(encoding="utf-8")
    assert "OLD TABLE" not in updated
    assert "Irreplaceable prose." in updated
    assert "## Regression Analysis" in updated
    assert "run-a" in updated


def test_render_refuses_when_markers_are_absent(tmp_path: Path) -> None:
    """Guessing where the table is would overwrite hand-maintained analysis."""
    markdown = _markdown(tmp_path, "# Ledger\n\n| Run ID |\n|---|\n| hand-typed |\n")
    csv_path = _write(tmp_path, [_row()])
    with pytest.raises(ledger.LedgerError, match="markers not found"):
        ledger.render(csv_path, markdown)
    assert "hand-typed" in markdown.read_text(encoding="utf-8")


def test_render_is_idempotent(tmp_path: Path) -> None:
    body = f"# L\n\n{ledger.BEGIN_MARKER}\n\nx\n\n{ledger.END_MARKER}\n\nprose\n"
    markdown = _markdown(tmp_path, body)
    csv_path = _write(tmp_path, [_row()])
    ledger.render(csv_path, markdown)
    once = markdown.read_text(encoding="utf-8")
    ledger.render(csv_path, markdown)
    assert markdown.read_text(encoding="utf-8") == once


def test_render_shows_partial_runs_as_a_fraction(tmp_path: Path) -> None:
    table = ledger.render_table([_row(items_scored="53", items_total="78")])
    assert "| 53/78 |" in table


def test_render_shows_a_complete_run_as_a_single_count(tmp_path: Path) -> None:
    table = ledger.render_table([_row(items_scored="78", items_total="78")])
    assert "| 78 |" in table


def test_render_falls_back_to_score_raw_when_there_is_no_single_score(tmp_path: Path) -> None:
    table = ledger.render_table(
        [_row(status="multi_arm", score="", score_raw="v2c 0.667 / v2h 0.679")]
    )
    assert "v2c 0.667 / v2h 0.679" in table


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def test_join_computes_a_delta(tmp_path: Path) -> None:
    rows = [_row(run_id="a", score="0.885"), _row(run_id="b", score="0.910")]
    csv_path = _write(tmp_path, rows)
    output = ledger.join("a", "b", csv_path, tmp_path / "nothing")
    assert "+0.025" in output


def test_join_refuses_a_delta_against_an_unscored_run(tmp_path: Path) -> None:
    rows = [_row(run_id="a", score="0.885"), _row(run_id="b", status="killed", score="")]
    csv_path = _write(tmp_path, rows)
    output = ledger.join("a", "b", csv_path, tmp_path / "nothing")
    assert "not computed" in output


def test_join_warns_across_different_item_counts(tmp_path: Path) -> None:
    """0.467 on 15 items and 0.910 on 78 items is not a +0.443 improvement."""
    rows = [
        _row(run_id="a", items_scored="15", items_total="15", score="0.467"),
        _row(run_id="b", items_scored="78", items_total="78", score="0.910"),
    ]
    csv_path = _write(tmp_path, rows)
    output = ledger.join("a", "b", csv_path, tmp_path / "nothing")
    assert "not directly comparable" in output


def test_join_says_it_cannot_attribute_without_provenance(tmp_path: Path) -> None:
    rows = [_row(run_id="a", score="0.1"), _row(run_id="b", score="0.2")]
    csv_path = _write(tmp_path, rows)
    output = ledger.join("a", "b", csv_path, tmp_path / "nothing")
    assert "cannot attribute the delta" in output


def test_join_rejects_an_unknown_run(tmp_path: Path) -> None:
    csv_path = _write(tmp_path, [_row(run_id="a")])
    with pytest.raises(ledger.LedgerError, match="no ledger row"):
        ledger.join("a", "ghost", csv_path, tmp_path / "nothing")


# ---------------------------------------------------------------------------
# The checked-in ledger must stay valid
# ---------------------------------------------------------------------------

def test_real_ledger_has_no_failures() -> None:
    findings = ledger.validate()
    failures = [f for f in findings if f["level"] == "FAIL"]
    assert not failures, "ledger.csv contradicts itself or the filesystem: " + "; ".join(
        f"{f['run_id']}: {f['message']}" for f in failures
    )


def test_real_ledger_markdown_is_in_sync_with_the_csv() -> None:
    """A hand-edit to the generated table would be silently lost on the next render."""
    rows = ledger.read_rows()
    table = ledger.render_table(rows)
    text = ledger.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert table in text, "LEDGER.md is out of sync; run: ledger.py render"


# ---------------------------------------------------------------------------
# Score cross-check against the run's own per-arm evidence
#
# This is what makes the ledger's number checkable rather than asserted. Before it, a score
# was read off a rendered markdown table and retyped, and nothing could catch a typo, a
# stale copy, or a number quoted from the wrong arm.
# ---------------------------------------------------------------------------

def _run_with_score_json(results: Path, run_id: str, arms: dict) -> Path:
    run = results / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_provenance.json").write_text(
        json.dumps({"run_id": run_id, "latest_attempt": {}}), encoding="utf-8"
    )
    (run / "score.json").write_text(
        json.dumps({"run_id": run_id, "primary_arm": None, "arms": arms}), encoding="utf-8"
    )
    return run


def test_score_disagreeing_with_the_evidence_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {"menhir_recall": {"n": 78, "score": 0.871795}})
    findings = _validate(
        tmp_path,
        [_row(score="0.900", primary_arm="menhir_recall", has_results_dir="true")],
        results=results,
    )
    assert any("score.json measured" in m for m in _messages(findings, "FAIL"))


def test_score_matching_the_evidence_is_clean(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {"menhir_recall": {"n": 78, "score": 0.871795}})
    findings = _validate(
        tmp_path,
        [_row(score="0.872", primary_arm="menhir_recall", has_results_dir="true")],
        results=results,
    )
    assert _messages(findings, "FAIL") == []


def test_quoting_the_wrong_arm_fails(tmp_path: Path) -> None:
    """value-arm-verify-20260717 records 0.679 (its value arm); its recall arm was 0.333."""
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {
        "menhir_recall": {"n": 78, "score": 0.333},
        "menhir_value_recall": {"n": 78, "score": 0.679},
    })
    findings = _validate(
        tmp_path,
        [_row(score="0.679", primary_arm="menhir_recall", has_results_dir="true")],
        results=results,
    )
    assert any("score.json measured 0.333" in m for m in _messages(findings, "FAIL"))


def test_declaring_an_arm_that_does_not_exist_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {"menhir_recall": {"n": 78, "score": 0.9}})
    findings = _validate(
        tmp_path,
        [_row(score="0.900", primary_arm="menhir_ghost", has_results_dir="true")],
        results=results,
    )
    assert any("is not in score.json" in m for m in _messages(findings, "FAIL"))


def test_score_without_a_declared_arm_warns(tmp_path: Path) -> None:
    """With several arms present, an undeclared row's number cannot be checked at all."""
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {
        "menhir_recall": {"n": 78, "score": 0.9},
        "no_memory": {"n": 78, "score": 0.1},
    })
    findings = _validate(
        tmp_path,
        [_row(score="0.900", primary_arm="", has_results_dir="true")],
        results=results,
    )
    assert any("no primary_arm" in m for m in _messages(findings, "WARN"))


def test_arm_item_count_disagreeing_with_the_row_warns(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _run_with_score_json(results, "run-a", {"menhir_recall": {"n": 41, "score": 0.9}})
    findings = _validate(
        tmp_path,
        [_row(score="0.900", primary_arm="menhir_recall", items_scored="78",
              items_total="78", has_results_dir="true")],
        results=results,
    )
    assert any("scored 41 items" in m for m in _messages(findings, "WARN"))


def test_unreadable_score_json_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = _run_with_score_json(results, "run-a", {"menhir_recall": {"n": 78, "score": 0.9}})
    (run / "score.json").write_text("{broken", encoding="utf-8")
    findings = _validate(
        tmp_path, [_row(has_results_dir="true")], results=results
    )
    assert any("score.json is unreadable" in m for m in _messages(findings, "FAIL"))


def test_scored_row_without_score_json_warns(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = results / "run-a"
    run.mkdir(parents=True)
    (run / "run_provenance.json").write_text(
        json.dumps({"run_id": "run-a", "latest_attempt": {}}), encoding="utf-8"
    )
    findings = _validate(tmp_path, [_row(has_results_dir="true")], results=results)
    assert any("no score.json" in m for m in _messages(findings, "WARN"))


def test_describe_score_lists_every_arm_without_choosing(tmp_path: Path) -> None:
    """A summary that picked one arm would make the same mistake primary_arm exists to avoid."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "score.json").write_text(
        json.dumps({"arms": {
            "menhir_recall": {"n": 78, "score": 0.923},
            "no_memory": {"n": 78, "score": 0.064},
        }}),
        encoding="utf-8",
    )
    described = ledger.describe_score(run)
    assert "menhir_recall=0.923" in described
    assert "no_memory=0.064" in described


def test_describe_score_returns_none_with_no_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    assert ledger.describe_score(run) is None


def test_a_checkout_without_the_evidence_tree_does_not_fail(tmp_path: Path) -> None:
    """`results/` is gitignored: CI has ledger.csv and none of the run directories.

    Without this distinction every row recording has_results_dir=true fails in CI for the
    one reason that is not a defect -- the evidence was never committed. An absent tree is
    "cannot check", not "the claim is false".
    """
    results = tmp_path / "results"
    results.mkdir()
    findings = _validate(
        tmp_path,
        [_row(run_id="run-a", has_results_dir="true"),
         _row(run_id="run-b", has_results_dir="true", canonical="current")],
        results=results,
    )
    assert _messages(findings, "FAIL") == []
    assert any("no full run directories under" in m for m in _messages(findings, "WARN"))


def test_schema_checks_still_run_without_the_evidence_tree(tmp_path: Path) -> None:
    """Skipping the on-disk checks must not turn validate into a no-op."""
    results = tmp_path / "results"
    results.mkdir()
    findings = _validate(
        tmp_path,
        [_row(run_id="run-a", status="killed", score="0.5", has_results_dir="true")],
        results=results,
    )
    assert any("implies status 'scored'" in m for m in _messages(findings, "FAIL"))


def test_a_present_evidence_tree_still_catches_a_missing_run_dir(tmp_path: Path) -> None:
    """The gate must not suppress the real contradiction it was guarding."""
    results = tmp_path / "results"
    (results / "some-other-run").mkdir(parents=True)  # tree IS materialized
    findings = _validate(
        tmp_path, [_row(run_id="run-a", has_results_dir="true")], results=results
    )
    assert any("claims a results directory" in m for m in _messages(findings, "FAIL"))


# ---------------------------------------------------------------------------
# Deliberate exclusions
#
# Some run directories hold a real score but are not buildout results -- recall panels,
# rescores, diagnostics. Excluding them records that decision once so the orphan finding
# stops reappearing. The risk is that exclusion becomes a way to hide a real result, so
# every guard below exists to keep it honest.
# ---------------------------------------------------------------------------

def _excluded_file(results: Path, body: str) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "ledger-excluded.txt").write_text(body, encoding="utf-8")


def _scored_orphan(results: Path, run_id: str, score: float = 0.9) -> None:
    run = results / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "run_provenance.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (run / "score.json").write_text(
        json.dumps({"arms": {"menhir_recall": {"n": 5, "score": score}}}), encoding="utf-8"
    )


def test_an_excluded_run_is_not_reported_as_an_orphan(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _scored_orphan(results, "panel-run")
    _excluded_file(results, "panel-run  packet-shape panel, n=5\n")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert not any("no ledger row" in m for m in _messages(findings, "WARN"))


def test_excluded_runs_are_still_counted_so_they_stay_visible(tmp_path: Path) -> None:
    """Suppression must not be silence: a reader still learns the decision was made."""
    results = tmp_path / "results"
    _scored_orphan(results, "panel-run")
    _excluded_file(results, "panel-run  packet-shape panel, n=5\n")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert any("deliberately excluded" in m for m in _messages(findings, "WARN"))


def test_an_exclusion_without_a_reason_is_refused(tmp_path: Path) -> None:
    """"Excluded" with no stated ground is how a real result gets quietly hidden."""
    results = tmp_path / "results"
    _scored_orphan(results, "panel-run")
    _excluded_file(results, "panel-run\n")
    with pytest.raises(ledger.LedgerError, match="excluded with no reason"):
        _validate(tmp_path, [_row(run_id="run-a")], results=results)


def test_a_run_both_excluded_and_recorded_fails(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _scored_orphan(results, "run-a")
    _excluded_file(results, "run-a  claimed to be a panel\n")
    findings = _validate(
        tmp_path, [_row(run_id="run-a", has_results_dir="true")], results=results
    )
    assert any("also has a ledger row" in m for m in _messages(findings, "FAIL"))


def test_a_stale_exclusion_warns(tmp_path: Path) -> None:
    """An exclusion naming a directory that is gone is bookkeeping doing no work."""
    results = tmp_path / "results"
    _scored_orphan(results, "real-run")
    _excluded_file(results, "real-run  a panel\nvanished-run  a panel that no longer exists\n")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert any("the exclusion is stale" in m for m in _messages(findings, "WARN"))


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _scored_orphan(results, "panel-run")
    _excluded_file(results, "# a heading\n\n   \npanel-run  a panel\n# trailing note\n")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert _messages(findings, "FAIL") == []


def test_no_exclusions_file_is_fine(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _scored_orphan(results, "panel-run")
    findings = _validate(tmp_path, [_row(run_id="run-a")], results=results)
    assert any("no ledger row" in m for m in _messages(findings, "WARN"))


def test_real_exclusions_file_parses_and_every_entry_has_a_reason() -> None:
    """Guards the checked-in file itself, which is where a reasonless entry would land."""
    excluded = ledger.read_exclusions()
    assert excluded, "expected the checked-in ledger-excluded.txt to list runs"
    assert all(reason.strip() for reason in excluded.values())
