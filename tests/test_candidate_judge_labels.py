"""Judge labels and the retest reading. Expected values are worked out by hand."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "longmemeval" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


judge = _load("candidate_judge_labels")
evals = sys.modules["recall_signal_eval"]


def meta(question, uuid, question_type="multi-session", kind="entity", query_kind="real"):
    return {"id": question, "question_id": question, "uuid": uuid, "name": uuid, "kind": kind,
            "query_kind": query_kind, "question_type": question_type, "label_strict": True,
            "label_lenient": True, "label_route": "receipt", "is_hub": True}


def test_verdict_needs_both_booleans_and_summary_implies_record():
    assert judge.parse_verdict('{"in_summary": true, "in_record": false}') == {
        "in_summary": True, "in_record": True}
    assert judge.parse_verdict('{"in_summary": false, "in_record": true}') == {
        "in_summary": False, "in_record": True}
    for bad in (None, "", "not json", '{"in_record": true}', '{"in_summary": 1, "in_record": 1}',
                "[1]"):
        assert judge.parse_verdict(bad) is None


def test_only_own_namespace_entities_of_real_non_ssa_questions_are_judged():
    rows = [
        meta("q1", "own"),
        meta("q1", "other"),
        meta("q1", "view", kind="view"),
        meta("q1@ns", "own", query_kind="swapped"),
        meta("q2", "ssa", question_type="single-session-assistant"),
    ]
    groups = {"own": "lme-q1", "other": "lme-q9", "view": "lme-q1", "ssa": "lme-q2"}
    assert judge.own_pairs(rows, groups) == {("q1", "own")}


def test_labels_keep_unknown_apart_from_negative():
    rows = [meta("q1", "yes"), meta("q1", "unjudged"), meta("q1", "other"), meta("q1", "unread")]
    groups = {"yes": "lme-q1", "unjudged": "lme-q1", "other": "lme-q9"}
    verdicts = {("q1", "yes"): {"in_summary": False, "in_record": True}}

    record = {m["uuid"]: m for m in judge.judged_labels(rows, groups, verdicts, "in_record")}
    summary = {m["uuid"]: m for m in judge.judged_labels(rows, groups, verdicts, "in_summary")}

    assert record["yes"]["label_strict"] is True and summary["yes"]["label_strict"] is False
    assert record["unjudged"]["label_strict"] is None
    assert record["other"]["label_strict"] is False
    assert record["other"]["label_route"] == "other_namespace"
    assert record["unread"]["label_strict"] is None
    # The heuristic hub flag must not drop judged rows from the primary cell.
    assert all(m["is_hub"] is False for m in record.values())


def test_request_never_shows_scores_or_heuristic_labels_and_truncates_evidence():
    item = {"question": "Q?", "answer": "A", "haystack_sessions": [
        [{"content": "x" * 5000, "has_answer": True}, {"content": "skipped"}]]}
    record = {"name": "N", "summary": None, "facts": ["F1"]}
    text = json.dumps(judge.request_for("c1", item, record))
    assert "skipped" not in text and "x" * (judge.EVIDENCE_CHARS + 1) not in text
    assert "F1" in text and "Memory summary: (none)" in text
    for leaked in ("similarity", "rank", "label_strict", "cosine"):
        assert leaked not in text


def test_repeat_agreement_on_first_run_positives():
    first = {"a": {"in_record": True}, "b": {"in_record": True}, "c": {"in_record": False},
             "d": {"in_record": False}}
    second = {"a": {"in_record": True}, "b": {"in_record": False}, "c": {"in_record": False},
              "d": {"in_record": False}}
    result = judge.repeat_agreement(first, second, "in_record")
    assert result == {"pairs": 4, "agreement": 0.75, "positives": 2, "positive_agreement": 0.5}


def test_reading_applies_validity_checks_in_order():
    assert judge.reading("supported", 100, 0.02, True) == "supported"
    assert judge.reading("killed", 99, 0.0, True) == "underpowered"
    assert judge.reading("supported", 100, 0.03, True) == "inconclusive"
    assert judge.reading("killed", 100, 0.0, False) == "inconclusive"


def test_overall_needs_every_graph_to_support_and_any_to_kill():
    assert judge.overall(["supported", "supported"]) == "supported"
    assert judge.overall(["supported", "inconclusive"]) == "inconclusive"
    assert judge.overall(["supported", "underpowered"]) == "inconclusive"
    assert judge.overall(["supported", "killed"]) == "killed"


def test_score_graph_reads_a_perfect_signal_as_supported():
    table = []
    for q in range(40):
        for i, positive in enumerate((True, False, False)):
            # RRF ranks the positive last, summary cosine first.
            table.append({
                "id": f"q{q}", "uuid": f"u{i}", "type": "MS", "is_hub": False,
                "label_strict": positive, "label_lenient": positive,
                "signals": {evals.RRF: float(i), evals.COSINE: float(i),
                            "summary_cosine": float(-i), "max_name_summary": float(-i),
                            "final_score": float(i)},
            })
    result = judge.score_graph(table, n_boot=50)
    assert result["questions"] == 40 and result["positives"] == 40
    assert result["signals"]["summary_cosine"]["status"] == "supported"
    assert result["signals"]["summary_cosine"]["minus_rrf"]["within_auc"]["estimate"] == 1.0
    # Identical to RRF: no CI above 0, so the frozen rule kills it.
    assert result["signals"][evals.COSINE]["status"] == "killed"
