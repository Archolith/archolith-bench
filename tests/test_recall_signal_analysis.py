"""Labeller and analysis for the recall real-similarity test, on a hand-built 3-question fixture.

Every expected number here is worked out by hand from the fixture, not read back from the code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "scripts" / "longmemeval" / "analysis"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


labels_mod = _load("candidate_labels")
evals = _load("recall_signal_eval")
summary_mod = _load("recall_summary_cosine")


def _item(qid, qtype, turns):
    return {
        "question_id": qid,
        "question_type": qtype,
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [turns],
    }


ITEMS = [
    _item("q1", "knowledge-update", [
        {"role": "user", "content": "I moved to Lisbon in the spring of last year", "has_answer": True},
        {"role": "assistant", "content": "Nice."},
        {"role": "user", "content": "My dog is called Pip and he is three"},
    ]),
    _item("q2", "temporal-reasoning", [
        {"role": "user", "content": "The conference was on the third of March", "has_answer": True},
    ]),
    _item("q3_abs", "multi-session", [{"role": "user", "content": "Nothing useful in this turn"}]),
]
K1 = labels_mod.turn_key("lme-q1", "s1", 0)
K2 = labels_mod.turn_key("lme-q1", "s1", 2)
K3 = labels_mod.turn_key("lme-q2", "s1", 0)


def _facts():
    facts = labels_mod.GraphFacts()
    facts.turn_keys = {"t1": K1, "t2": K2, "t3": K3}
    for uuid, ns, kind in (
        ("gold-ent", "lme-q1", "entity"),
        ("other-ent", "lme-q1", "entity"),
        ("hub-ent", "lme-q1", "entity"),
        ("orphan-ent", "lme-q1", "entity"),
        ("prefix-ent", "lme-q1", "entity"),
        ("q2-ent", "lme-q2", "entity"),
        ("gold-view", "lme-q1", "view"),
        ("assert-view", "lme-q1", "view"),
    ):
        facts.entities[uuid] = {"namespace": ns, "name": uuid, "kind": kind}
    facts.episodes = {f"g{i}": {"namespace": "lme-q1", "content": "x"} for i in range(1, 7)}
    facts.episodes["g-prefix"] = {
        "namespace": "lme-q1", "content": "said: I moved to Lisbon in the spring of last year!",
    }
    facts.episodes["g-q2"] = {"namespace": "lme-q2", "content": "x"}
    facts.episodes["queue-1"] = {"namespace": "lme-q1", "content": "x"}
    facts.receipt_turns["g1"].add("t1")
    facts.receipt_turns["g2"].add("t2")
    facts.receipt_turns["g-q2"].add("t3")
    facts.self_turns["queue-1"].add("t1")
    facts.mentions["gold-ent"].add("g1")
    facts.mentions["other-ent"].add("g2")
    # Hub: 3 of the 7 mentioning episodes in lme-q1, and it reaches gold through g1.
    facts.mentions["hub-ent"] |= {"g1", "g3", "g4"}
    facts.mentions["orphan-ent"].add("g5")
    facts.mentions["prefix-ent"].add("g-prefix")
    facts.mentions["q2-ent"].add("g-q2")
    # Filler so the namespace has 7 mentioning episodes and only hub-ent exceeds 20%.
    facts.entities["filler"] = {"namespace": "lme-q1", "name": "filler", "kind": "entity"}
    facts.mentions["filler"].add("g6")
    facts.turn_mentions["gold-view"].add("t1")
    facts.assertion_turns["assert-view"].add("t2")
    facts.observations["obs-gold"] = {"t1"}
    facts.observations["obs-other"] = {"t2"}
    return facts


GOLD = labels_mod.load_gold(ITEMS)


def _label(uuid, question="q1", facts=None):
    facts = facts or _facts()
    return labels_mod.label(facts, labels_mod.hub_entities(facts), uuid, GOLD.get(question))


def test_turn_key_matches_the_stored_hash_shape():
    import hashlib

    expected = hashlib.sha256(b"lme-q1\x00lme-q1|lme-q1-s1|t0").hexdigest()
    assert labels_mod.turn_key("lme-q1", "s1", 0) == expected


def test_gold_holds_only_has_answer_turns():
    assert GOLD["q1"].keys == {K1}
    assert GOLD["q3_abs"].keys == frozenset()


def test_entity_reaching_a_gold_turn_is_positive():
    assert _label("gold-ent") == {
        "kind": "entity", "label_strict": True, "label_lenient": True,
        "label_route": "receipt", "is_hub": False,
    }


def test_resolved_entity_on_another_turn_is_a_real_negative():
    result = _label("other-ent")
    assert (result["label_lenient"], result["label_route"]) == (False, "receipt")


def test_hub_is_lenient_positive_but_not_strict():
    result = _label("hub-ent")
    assert result["is_hub"] is True
    assert (result["label_lenient"], result["label_strict"]) == (True, False)


def test_unresolved_candidate_is_unknown_not_negative():
    result = _label("orphan-ent")
    assert (result["label_lenient"], result["label_strict"]) == (None, None)
    assert result["label_route"] == "none"


def test_prefix_fallback_is_marked_and_needs_the_questions_namespace():
    assert _label("prefix-ent")["label_route"] == "prefix"
    assert _label("prefix-ent")["label_lenient"] is True
    # The same text cannot make it positive for a question from another namespace.
    assert _label("prefix-ent", "q2")["label_lenient"] is None


def test_swapped_null_cannot_match_gold_through_the_turn_hash():
    # q2's gold asked against a q1 entity: the link resolves, but the keys embed the namespace.
    result = _label("gold-ent", "q2")
    assert (result["label_lenient"], result["label_route"]) == (False, "receipt")


def test_views_label_from_turn_mentions_and_assertions():
    assert _label("gold-view")["label_route"] == "turn_mention"
    assert _label("gold-view")["label_lenient"] is True
    assert _label("gold-view")["is_hub"] is False
    negative = _label("assert-view")
    assert (negative["kind"], negative["label_route"], negative["label_lenient"]) == (
        "view", "assertion", False,
    )


def test_source_memory_labels_from_its_own_turn():
    result = _label("queue-1")
    assert (result["kind"], result["label_route"], result["label_lenient"]) == (
        "source_memory", "queue_self", True,
    )


def test_observation_labels_from_the_turn_that_founds_it():
    assert _label("obs-gold") == {
        "kind": "observation", "label_strict": True, "label_lenient": True,
        "label_route": "founds", "is_hub": False,
    }
    assert _label("obs-other")["label_lenient"] is False


def test_unknown_uuid_stays_unknown():
    result = _label("nowhere")
    assert (result["kind"], result["label_lenient"]) == ("unknown", None)


def test_labelling_does_not_grow_the_fact_tables():
    facts = _facts()
    before = len(facts.mentions), len(facts.receipt_turns), len(facts.self_turns)
    _label("orphan-ent", facts=facts)
    _label("nowhere", facts=facts)
    assert (len(facts.mentions), len(facts.receipt_turns), len(facts.self_turns)) == before


def test_query_kinds():
    assert labels_mod.query_kind("q1", GOLD["q1"]) == "real"
    assert labels_mod.query_kind("q1@lme-q2", GOLD["q1"]) == "swapped"
    assert labels_mod.query_kind("q3_abs", GOLD["q3_abs"]) == "abstention"
    assert labels_mod.question_of("q1@lme-q2") == "q1"


def test_load_batch_keeps_a_retried_pair_and_counts_unrecovered_ones(tmp_path):
    path = tmp_path / "batch.jsonl"
    rows = [
        {"id": "a", "arm": "base", "error": "boom"},
        {"id": "a", "arm": "base", "results": []},
        {"id": "b", "arm": "base", "name_cosine_warning": "name_cosine_unavailable:X"},
        {"id": "c", "arm": "base", "results": []},
        {"id": "c", "arm": "base", "error": "late failure"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    clean, failed = labels_mod.load_batch(path)
    assert sorted(r["id"] for r in clean) == ["a", "c"]
    assert failed == 1


def test_label_rows_cover_results_and_source_memories_once_across_arms():
    rows = [
        {"id": "q1", "arm": "base", "candidates": [{"uuid": "gold-ent"}],
         "results": [{"uuid": "gold-view"}], "source_memories": [{"uuid": "queue-1"}]},
        {"id": "q1", "arm": "f30", "candidates": [{"uuid": "gold-ent"}]},
    ]
    labelled = labels_mod.label_rows(rows, _facts(), GOLD)
    assert sorted(r["uuid"] for r in labelled) == ["gold-ent", "gold-view", "queue-1"]
    assert {r["query_kind"] for r in labelled} == {"real"}


@pytest.mark.parametrize("uri", ["bolt://10.0.0.5:18342", "bolt://127.0.0.1:7687",
                                 "bolt://127.0.0.1:7688", "bolt://127.0.0.1"])
def test_labeller_refuses_anything_but_a_throwaway_graph(uri):
    with pytest.raises(SystemExit):
        labels_mod.check_target(uri)


# --- Metrics ----------------------------------------------------------------------------------


def test_auc_counts_ties_as_half():
    # Positives 3 and 1; negatives 2 and 1. Pairs: 3>2, 3>1, 1<2, 1=1 -> (1 + 1 + 0 + .5) / 4.
    assert evals.auc_ap([3, 1, 2, 1], [1, 1, 0, 0])[0] == pytest.approx(0.625)


def test_average_precision_by_hand():
    # Ranked + - + -: precision 1 at the first hit and 2/3 at the second.
    assert evals.auc_ap([4, 3, 2, 1], [1, 0, 1, 0])[1] == pytest.approx((1 + 2 / 3) / 2)


def test_tied_block_reads_precision_at_its_end_regardless_of_order():
    first = evals.auc_ap([1, 1, 1, 1], [1, 1, 0, 0])
    second = evals.auc_ap([1, 1, 1, 1], [0, 0, 1, 1])
    assert first == pytest.approx(second)
    assert first == pytest.approx((0.5, 0.5))


def test_missing_signal_ranks_last():
    assert evals.auc_ap([float("-inf"), 0.1], [1, 0])[0] == 0.0


def test_one_class_gives_nan_not_a_number():
    assert np.isnan(evals.auc_ap([1, 2], [1, 1])[0])


def test_weights_match_literally_repeating_a_group():
    scores = np.array([3.0, 1.0, 2.0, 0.0])
    labels = np.array([True, False, True, False])
    groups = np.array([0, 0, 1, 1])
    weighted = evals._weighted_metrics(scores, labels, groups, np.array([[2, 1]]))
    repeated = evals.auc_ap(
        [3, 1, 3, 1, 2, 0], [True, False, True, False, True, False],
    )
    assert weighted[0][0] == pytest.approx(repeated[0])
    assert weighted[1][0] == pytest.approx(repeated[1])


def test_bootstrap_weights_are_seeded_and_sum_to_the_sample():
    first = evals.bootstrap_weights(7, 50, 1)
    assert np.array_equal(first, evals.bootstrap_weights(7, 50, 1))
    assert set(first.sum(axis=1)) == {7}


# --- Three-question fixture for H1-H3 ----------------------------------------------------------


def _cand(uuid, rrf, cosine, survived=True, source="vector", bm25=None, rank=1):
    return {"uuid": uuid, "similarity": rrf, "name_cosine": cosine, "survived_floor": survived,
            "source": source, "bm25_rank": bm25, "cosine_rank": rank,
            "final_score": rrf if survived else None}


def _meta(row_id, uuid, lenient, kind="real", qtype="knowledge-update", hub=False):
    strict = None if lenient is None else lenient and not hub
    return {
        "id": row_id, "uuid": uuid, "question_id": row_id.split("@")[0], "query_kind": kind,
        "question_type": qtype, "kind": "entity", "name": uuid, "is_hub": hub,
        "label_strict": strict, "label_lenient": lenient, "label_route": "receipt",
    }


def _result(uuid, rank, relevance="high", tokens=10):
    return {"uuid": uuid, "rank": rank, "relevance": relevance, "tokens": tokens}


def _fixture():
    """q1 and q2 are real, n1 a swapped null. RRF ranks q1 and q2 wrongly; cosine ranks them right."""
    rows = [
        {"id": "q1", "arm": "base",
         "candidates": [_cand("a", 0.2, 0.8), _cand("b", 1.0, 0.1), _cand("c", 0.1, 0.05, False)],
         "results": [_result("b", 1), _result("a", 2, "low")]},
        {"id": "q2", "arm": "base",
         "candidates": [_cand("d", 0.5, 0.7), _cand("e", 0.9, 0.2)],
         "results": [_result("e", 1), _result("d", 2, "medium")]},
        {"id": "q1@lme-q2", "arm": "base",
         "candidates": [_cand("d", 1.0, 0.1)],
         "results": [_result("d", 1)]},
        {"id": "q1", "arm": "f30",
         "candidates": [_cand("a", 0.2, 0.8), _cand("b", 1.0, 0.1, False)],
         "results": [_result("a", 1, "low")]},
        {"id": "q2", "arm": "f30",
         "candidates": [_cand("d", 0.5, 0.7), _cand("e", 0.9, 0.2, False)],
         "results": [_result("d", 1, "medium")]},
        {"id": "q1@lme-q2", "arm": "f30", "candidates": [_cand("d", 1.0, 0.1, False)],
         "results": []},
        {"id": "q1", "arm": "f90",
         "candidates": [_cand("a", 0.2, 0.8, False), _cand("b", 1.0, 0.1, False)],
         "results": []},
        {"id": "q2", "arm": "f90",
         "candidates": [_cand("d", 0.5, 0.7), _cand("e", 0.9, 0.2, False)],
         "results": [_result("d", 1, "medium")]},
        {"id": "q1@lme-q2", "arm": "f90", "candidates": [_cand("d", 1.0, 0.1, False)],
         "results": []},
    ]
    metas = [
        _meta("q1", "a", True), _meta("q1", "b", False), _meta("q1", "c", False),
        _meta("q2", "d", True, qtype="temporal-reasoning"),
        _meta("q2", "e", False, qtype="temporal-reasoning"),
        _meta("q1@lme-q2", "d", False, kind="swapped"),
    ]
    labels = {(m["id"], m["uuid"]): m for m in metas}
    queries = {"q1": "where did I move", "q2": "when was it", "q1@lme-q2": "where did I move"}
    return rows, labels, queries


FIXTURE_GOLD = {
    "q1": labels_mod.Gold("q1", "lme-q1", "knowledge-update", frozenset({"k1"}), ()),
    "q2": labels_mod.Gold("q2", "lme-q2", "temporal-reasoning", frozenset({"k2"}), ()),
}


def _table():
    rows, labels, queries = _fixture()
    return evals.candidate_table(rows, labels, queries, {})


def test_table_keeps_floor_dropped_candidates_and_skips_null_queries():
    table = _table()
    assert sorted(r["uuid"] for r in table) == ["a", "b", "c", "d", "e"]
    dropped = next(r for r in table if r["uuid"] == "c")
    assert dropped["signals"]["final_score"] == float("-inf")


def test_table_leaves_out_views_and_unknowns():
    rows, labels, queries = _fixture()
    labels[("q1", "b")]["kind"] = "view"
    assert "b" not in {r["uuid"] for r in evals.candidate_table(rows, labels, queries, {})}


def test_h2_cell_by_hand():
    cell = evals.score_cell(_table(), "strict_no_hubs", 200, 1)
    cosine, rrf = cell["signals"][evals.COSINE], cell["signals"][evals.RRF]
    # q1: a over b and c, so cosine is 1.0. RRF: a beats c only, 0.5. q2: cosine 1.0, RRF 0.
    assert cosine["within_auc"]["estimate"] == pytest.approx(1.0)
    assert rrf["within_auc"]["estimate"] == pytest.approx(0.25)
    # Pooled positives a, d; negatives b, c, e. Cosine: both beat all three -> 1.0.
    assert cosine["pooled_auc"]["estimate"] == pytest.approx(1.0)
    # RRF: a=.2 beats c only; d=.5 beats c only -> 2/6.
    assert rrf["pooled_auc"]["estimate"] == pytest.approx(2 / 6)
    assert cell["cosine_minus"][evals.RRF]["within_auc"]["estimate"] == pytest.approx(0.75)
    assert cell["positive_share"] == pytest.approx(2 / 5)


def test_per_query_normalizations_cannot_change_within_question_rank():
    cell = evals.score_cell(_table(), "strict_no_hubs", 50, 1)
    within = {n: cell["signals"][n]["within_auc"]["estimate"]
              for n in (evals.RRF, "rrf_minmax", "rrf_percentile")}
    assert within["rrf_minmax"] == pytest.approx(within[evals.RRF])
    assert within["rrf_percentile"] == pytest.approx(within[evals.RRF])


def test_cells_handle_hubs_and_unknown_labels_as_planned():
    rows, labels, queries = _fixture()
    labels[("q1", "b")].update(is_hub=True, label_lenient=True, label_strict=False)
    labels[("q1", "c")].update(label_lenient=None, label_strict=None)
    table = evals.candidate_table(rows, labels, queries, {})
    assert "b" not in {r["uuid"] for r in evals.cell_rows(table, "strict_no_hubs")[0]}
    with_hubs, key = evals.cell_rows(table, "strict_with_hubs")
    assert next(r for r in with_hubs if r["uuid"] == "b")[key] is False
    assert "c" not in {r["uuid"] for r in with_hubs}


def test_ssa_stays_out_of_every_cell_but_appears_per_type():
    rows, labels, queries = _fixture()
    for key in (("q2", "d"), ("q2", "e")):
        labels[key]["question_type"] = "single-session-assistant"
    table = evals.candidate_table(rows, labels, queries, {})
    assert {r["id"] for r in evals.cell_rows(table, "lenient_with_hubs")[0]} == {"q1"}
    assert "SSA" in evals.per_type(table)


def _gain(estimate, low, high):
    return {"estimate": estimate, "low": low, "high": high}


@pytest.mark.parametrize("pooled, within, expected", [
    (_gain(0.06, 0.01, 0.1), _gain(0.05, 0.01, 0.1), "supported"),
    (_gain(0.04, 0.01, 0.1), _gain(0.06, 0.01, 0.1), "inconclusive"),
    (_gain(0.08, 0.02, 0.1), _gain(0.01, -0.02, 0.04), "inconclusive"),
    (_gain(0.02, -0.01, 0.05), _gain(0.01, -0.02, 0.04), "killed"),
    (_gain(0.08, 0.02, 0.1), _gain(-0.05, -0.08, -0.01), "killed"),
])
def test_h2_verdict_rules(pooled, within, expected):
    primary = {"cosine_minus": {evals.RRF: {"pooled_auc": pooled, "within_auc": within}}}
    assert evals.h2_verdict(primary)["status"] == expected


def test_h1_by_hand():
    rows, _, _ = _fixture()
    kinds, _ = evals.question_meta(rows, FIXTURE_GOLD)
    report = evals.score_h1(rows, kinds, 100, 1)
    assert report["real"]["high_share"] == 1.0
    assert report["swapped"]["high_share"] == 1.0
    assert report["verdict"]["status"] == "supported"


@pytest.mark.parametrize("hits, expected", [(1, "supported"), (0.05, "killed"), (0.3, "inconclusive")])
def test_h1_verdict_thresholds(hits, expected):
    total = 100
    rows, kinds = [], {}
    for i in range(total):
        rows.append({"id": f"r{i}", "arm": "base", "results": [{"relevance": "high"}]})
        kinds[f"r{i}"] = "real"
        high = i < hits * total
        rows.append({"id": f"n{i}", "arm": "base",
                     "results": [{"relevance": "high" if high else "low"}]})
        kinds[f"n{i}"] = "swapped"
    assert evals.score_h1(rows, kinds, 50, 1)["verdict"]["status"] == expected


def test_empty_recalls_stay_in_the_h1_and_coverage_denominators():
    # An empty recall has no label rows, so kinds must come from the batch itself.
    rows, _, _ = _fixture()
    rows = rows + [
        {"id": "q2@lme-q1", "arm": "base", "candidates": [], "results": []},
        {"id": "q9", "arm": "base", "candidates": [], "results": []},
    ]
    gold = FIXTURE_GOLD | {
        "q9": labels_mod.Gold("q9", "lme-q9", "multi-session", frozenset({"k9"}), ()),
    }
    kinds, types = evals.question_meta(rows, gold)
    report = evals.score_h1(rows, kinds, 50, 1)
    assert report["swapped"] == {"queries": 2, "high_share": 0.5}
    assert report["real"]["queries"] == 3
    ceiling = evals.coverage_ceiling(_table(), kinds, types)
    assert ceiling["questions"] == 3
    assert ceiling["label_lenient"] == pytest.approx(2 / 3)


def _h3():
    rows, labels, _ = _fixture()
    kinds, types = evals.question_meta(rows, FIXTURE_GOLD)
    tuning = {"f30": {"name_cosine_floor": 0.3}, "f90": {"name_cosine_floor": 0.9}}
    return evals.score_h3(rows, labels, kinds, types, tuning)


def test_h3_floor_keeps_answers_and_cuts_noise_by_hand():
    cut = next(r for r in _h3()["arms"]["f30"]["by_k"] if r["k"] == 10)
    assert (cut["lost_questions"], cut["gold_questions"]) == (0, 2)
    # Noise b, e and the null's d go away: 3 -> 0. Tokens 50 -> 20.
    assert cut["all"]["noise_cut"] == pytest.approx(1.0)
    assert cut["all"]["token_cut"] == pytest.approx(0.6)
    assert cut["null"]["noise_before"] == 1
    assert cut["flips"] == []


def test_h3_loss_and_flip_name_the_question_and_whether_the_floor_fired():
    cut = next(r for r in _h3()["arms"]["f90"]["by_k"] if r["k"] == 10)
    assert (cut["lost_questions"], cut["loss_rate"]) == (1, 0.5)
    flip = cut["flips"][0]
    assert flip == {
        "id": "q1", "type": "KU", "direction": "lost_last", "left_top_k": ["a"],
        "entered_top_k": [], "floor_fired": True, "floor_dropped_answers": ["a"],
    }


def test_h3_flip_without_the_floor_firing_is_reported_as_such():
    # The answer leaves top-k because of reordering alone, so the floor gets no credit or blame.
    baseline = {"q": {"id": "q", "candidates": [_cand("a", 1, 0.9), _cand("b", 1, 0.9)],
                      "results": [_result("a", 1), _result("b", 2)]}}
    arm = {"q": {"id": "q", "candidates": [_cand("a", 1, 0.9), _cand("b", 1, 0.9)],
                 "results": [_result("b", 1), _result("a", 2)]}}
    labels = {("q", "a"): _meta("q", "a", True), ("q", "b"): _meta("q", "b", False)}
    cut = evals.score_arm(baseline, arm, labels, {"q": "real"}, {"q": "knowledge-update"}, 1)
    assert cut["lost_questions"] == 1
    assert cut["flips"][0]["floor_fired"] is False


def test_h3_unknown_results_are_not_noise():
    labels = {("q", "a"): _meta("q", "a", None)}
    assert evals._noise([_result("a", 1)], labels, "q") == 0


def test_h3_verdict_needs_the_loss_bound_and_a_gain():
    assert _h3()["verdict"] == {
        "status": "pending_accuracy", "candidates": ["f30"], "within_loss_bound": ["f30"],
    }


def test_h3_verdict_killed_when_nothing_stays_in_bound():
    arms = {"x": {"by_k": [{"k": 10, "loss_rate": 0.02,
                            "all": {"noise_cut": 0.5, "token_cut": 0.5}}]}}
    assert evals.h3_verdict(arms)["status"] == "killed"


def test_h3_verdict_small_gain_in_bound_is_inconclusive():
    arms = {"x": {"by_k": [{"k": 10, "loss_rate": 0.0,
                            "all": {"noise_cut": 0.1, "token_cut": 0.01}}]}}
    assert evals.h3_verdict(arms)["status"] == "inconclusive"


# --- H4 ---------------------------------------------------------------------------------------


def test_isotonic_pools_violators_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = evals.fit_isotonic(x, y)
    # 1.0 and 0.0 at x=2,3 pool to 0.5.
    assert evals.predict_isotonic(x, model).tolist() == [0.0, 0.5, 0.5, 1.0]


def test_isotonic_clamps_outside_the_fitted_range():
    model = evals.fit_isotonic(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    assert evals.predict_isotonic(np.array([0.0, 9.0]), model).tolist() == [0.0, 1.0]


def test_ece_by_hand():
    p = np.array([0.15, 0.15, 0.95, 0.95])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    # Bin 1: |.15 - .5| over 2 items; bin 9: |.95 - 1| over 2 items.
    assert evals.expected_calibration_error(p, y) == pytest.approx((0.35 * 2 + 0.05 * 2) / 4)


def test_logistic_recovers_a_separating_direction():
    x = np.c_[np.ones(6), np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])]
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    p = evals.predict_logistic(x, evals.fit_logistic(x, y))
    assert p[:3].max() < 0.5 < p[3:].min()


def _h4_table(signal):
    """Fit and test types where ``signal(i)`` is the cosine of item i; label alternates."""
    rng = np.random.default_rng(3)
    table = []
    for qtype, count in (("KU", 300), ("TR", 300), ("SSA", 20)):
        for i in range(count):
            positive = bool(i % 2)
            cosine = signal(positive, rng)
            table.append({
                "id": f"{qtype}{i // 10}", "uuid": f"{qtype}{i}", "type": qtype, "source": "vector",
                "is_hub": False, "label_strict": positive, "label_lenient": positive,
                "lane_agreement": False,
                "signals": {evals.COSINE: cosine, "summary_cosine": float("-inf"),
                            evals.RRF: float(rng.random()), "rrf_percentile": float(rng.random())},
            })
    return table


def test_h4_is_pending_r1_when_cosine_carries_the_label_and_rrf_does_not():
    report = evals.score_h4(_h4_table(lambda pos, rng: (0.7 if pos else 0.2) + rng.normal(0, 0.1)))
    assert report["verdict"]["status"] == "pending_r1"
    assert report["verdict"]["beats_calibrated_rrf"] is True
    assert report["uses_summary_cosine"] is False
    assert len(report["folds"]) == evals.FOLDS
    assert report["splits"]["ssa"]["candidates"] == 20


def test_h4_is_killed_when_features_add_nothing_over_rrf():
    report = evals.score_h4(_h4_table(lambda pos, rng: float(rng.random())))
    assert report["verdict"]["status"] == "killed"


def test_h4_split_never_trains_on_held_out_types():
    assert not evals.FIT_TYPES & evals.TEST_TYPES
    assert "SSA" not in evals.FIT_TYPES | evals.TEST_TYPES


def test_h4_reports_no_data_without_both_splits():
    assert evals.score_h4([])["verdict"]["status"] == "no_data"


# --- Whole report and summary cosines ----------------------------------------------------------


def test_evaluate_marks_h3_and_h4_exploratory_when_h2_dies(monkeypatch):
    rows, labels, queries = _fixture()
    monkeypatch.setattr(evals, "h2_verdict", lambda primary: {"status": "killed"})
    report = evals.evaluate(rows, labels, FIXTURE_GOLD, queries, {}, {}, n_boot=20)
    assert report["h3"]["exploratory_only"] is True
    assert report["h4"]["exploratory_only"] is True


def test_report_is_strict_json_and_renders():
    rows, labels, queries = _fixture()
    report = evals._finite(evals.evaluate(rows, labels, FIXTURE_GOLD, queries, {}, {}, n_boot=20))
    json.dumps(report, allow_nan=False)
    text = evals.render_markdown(report)
    assert "| H2 cosine beats RRF |" in text
    assert "not `build_context` counts" in text


def test_reports_are_reproducible():
    rows, labels, queries = _fixture()
    first = evals.evaluate(rows, labels, FIXTURE_GOLD, queries, {}, {}, n_boot=30)
    second = evals.evaluate(rows, labels, FIXTURE_GOLD, queries, {}, {}, n_boot=30)
    assert json.dumps(evals._finite(first)) == json.dumps(evals._finite(second))


def test_summary_cosine_feeds_the_table_and_max():
    rows, labels, queries = _fixture()
    table = evals.candidate_table(rows, labels, queries, {("q1", "b"): 0.6})
    row = next(r for r in table if r["uuid"] == "b")
    assert row["signals"]["summary_cosine"] == 0.6
    assert row["signals"]["max_name_summary"] == 0.6
    other = next(r for r in table if r["uuid"] == "a")
    assert other["signals"]["summary_cosine"] == float("-inf")
    assert other["signals"]["max_name_summary"] == 0.8


def test_summaries_embed_each_text_once_and_rows_reuse_pinned_vectors():
    calls = []

    def embed(texts):
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    vectors = summary_mod.embed_summaries({"a": "same", "b": "same"}, embed)
    assert calls == [["same"]]
    rows = [
        {"id": "q", "arm": "base", "candidates": [{"uuid": "a"}, {"uuid": "z"}]},
        {"id": "q", "arm": "f30", "candidates": [{"uuid": "a"}]},
    ]
    out = summary_mod.summary_rows(rows, {"q": "text"}, {json.dumps("text"): [1.0, 0.0]}, vectors)
    assert out == [
        {"id": "q", "uuid": "a", "summary_cosine": 1.0},
        {"id": "q", "uuid": "z", "summary_cosine": None},
    ]
