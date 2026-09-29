"""Tests for the supersession-aware v2 typed-value graph and its two arms."""

from __future__ import annotations

from archolith_bench.harness.memory_ab import (
    VALUE_RECALL_V2_CURRENT,
    VALUE_RECALL_V2_HISTORY,
    VALUE_RECALL_V3_AUTHORITATIVE,
    VALUE_RECALL_V3_COARSE,
    VALUE_RECALL_V4_ADVISORY,
    VALUE_RECALL_V5_DERIVED,
    _snippet_has_stale,
    run_memory_ab,
)
from archolith_bench.harness.value_nodes_v2 import SupersededValueGraph


def _sessions(*turns: str) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": turn} for turn in turns]]


def _current_values(graph: SupersededValueGraph) -> list[str]:
    """Normalized values whose edge is CURRENT after v2 supersession."""
    graph.recall("value", limit=50)  # sets edge.current per v2 selection
    return sorted(
        str(graph.values[e.target_node_id].normalized) for e in graph.edges if e.current
    )


# --- recency: latest-learned wins within a cluster ---------------------------------

def test_latest_learned_value_supersedes_earlier_in_same_cluster() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I have a total of 37 coins in my collection.",
            "I now have 38 coins in my collection.",
        ),
    )
    assert _current_values(graph) == ["38"]


def test_clock_time_supersession_keeps_latest() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I have been waking up at 8:30 am on Saturdays.",
            "I now wake up at 7:30 am on Saturdays.",
        ),
    )
    assert _current_values(graph) == ["07:30"]


# --- attribute (state family) isolation: no false supersession ---------------------

def test_owned_and_sold_counts_do_not_supersede() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 38 coins.", "I sold 5 coins."),
    )
    # Distinct attribute families -> two clusters, both current.
    assert _current_values(graph) == ["38", "5"]


def test_watched_and_watchlist_do_not_supersede() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I watched 5 MCU films.",
            "I have 8 MCU films on my watch list.",
        ),
    )
    assert _current_values(graph) == ["5", "8"]


# --- scope isolation: differentiating modifier survives ----------------------------

def test_scope_keeps_korean_and_italian_distinct() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I have tried four Korean restaurants.",
            "I have tried three Italian restaurants.",
        ),
    )
    assert _current_values(graph) == ["3", "4"]


# --- entity alias merge: paraphrases collapse into one cluster ---------------------

def test_alias_and_container_words_merge_into_one_cluster() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I have 30 coins.",
            "I now own 35 coins.",
            "My collection now has 40 coins.",
        ),
    )
    assert _current_values(graph) == ["40"]


# --- explicit correction precedes latest-learned order -----------------------------

def test_correction_marker_overrides_mention_order() -> None:
    # The corrected value (20) is learned FIRST; the rejected value (25) is learned
    # LATER but carries a past marker, so latest-learned must NOT pick it.
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I now have 20 titles on my list.",
            "I used to have 25 titles on my list.",
        ),
    )
    assert _current_values(graph) == ["20"]


def test_not_correction_marks_rejected_value_historical() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("My list now has 20 titles, not 25 titles."),
    )
    assert _current_values(graph) == ["20"]


# --- global session/turn sequence (not per-session reset) --------------------------

def test_ordering_spans_sessions() -> None:
    sessions = [
        [{"role": "user", "content": "I have 37 coins."}],
        [{"role": "user", "content": "I have 38 coins."}],
    ]
    graph = SupersededValueGraph.from_sessions("t", sessions)
    assert _current_values(graph) == ["38"]


def test_from_item_dates_drive_recency_when_shuffled() -> None:
    sessions = [
        [{"role": "user", "content": "I have 25 coins."}],
        [{"role": "user", "content": "I have 20 coins."}],
    ]
    item = {"haystack_dates": ["2025/02/01 (Sat) 12:00", "2025/01/01 (Wed) 12:00"]}
    graph = SupersededValueGraph.from_item("t", item, sessions)
    # Feb session (25) is later than Jan session (20) -> 25 current.
    assert _current_values(graph) == ["25"]


# --- emission modes ----------------------------------------------------------------

def test_current_only_hides_superseded_snippet() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I now have 38 coins."),
    )
    snippets = graph.recall("How many coins do I have now?", limit=4, emit_history=False)
    joined = "\n".join(snippets)
    assert "38 coins" in joined
    assert "37 coins" not in joined
    assert "[typed-value count current]" in joined
    assert "[typed-value count was]" not in joined


def test_history_mode_shows_prior_value_labeled_was() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I now have 38 coins."),
    )
    snippets = graph.recall("How many coins do I have now?", limit=4, emit_history=True)
    joined = "\n".join(snippets)
    assert "[typed-value count current] " in joined
    assert "[typed-value count was] " in joined
    assert "38 coins" in joined and "37 coins" in joined


# --- offline arm wiring ------------------------------------------------------------

class _ValueAdapter:
    benchmark_id = "value-v2-test"
    name = "Value v2 test"

    def load_items(self, subset, limit, fixture_path):  # noqa: ANN001
        return [
            {
                "question_id": "v2-1",
                "answer": "38",
                "question_type": "knowledge-update",
                "question": "How many coins do I have now?",
                "sessions": _sessions("I have 37 coins.", "I now have 38 coins."),
            }
        ]

    def sessions(self, item: dict) -> list[list[dict]]:
        return item["sessions"]

    def question(self, item: dict) -> str:
        return item["question"]

    def build_messages(self, memory_context: str, question: str) -> list[dict]:
        return [{"role": "user", "content": f"Memory:\n{memory_context}\nQuestion: {question}"}]

    def score(self, item: dict, response_text: str) -> bool:
        return item["answer"] in response_text


class _StaleBlindClient:
    """Ordinary recall that surfaces the STALE value (38 missing)."""

    def new_group(self) -> str:
        return "group"

    def record_turn_evidence(self, namespace: str, text: str, **kwargs) -> dict:
        return {"turn_id": "te-1"}

    def ingest(self, group_id: str, role: str, content: str, **kwargs) -> None:
        return None

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        return ["The user mentioned having 37 coins at one point."]

    def reset(self, group_id: str) -> None:
        return None


def _send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
    context = messages[-1]["content"]
    answer = "38" if "38 coins" in context else "I don't know"
    return answer, 1.0, {"prompt_tokens": 10, "completion_tokens": 2}


def test_v2_current_arm_recovers_superseded_value() -> None:
    result = run_memory_ab(
        _ValueAdapter(),
        arms=(VALUE_RECALL_V2_CURRENT,),
        client=_StaleBlindClient(),
        send_fn=_send_fn,
        recall_limit=4,
    )
    arm = result.arms[VALUE_RECALL_V2_CURRENT]
    assert arm.score == 1.0
    assert "[typed-value count current]" in arm.results[0].recalled
    assert "[typed-value count was]" not in arm.results[0].recalled


def test_v2_history_arm_includes_prior_value() -> None:
    result = run_memory_ab(
        _ValueAdapter(),
        arms=(VALUE_RECALL_V2_HISTORY,),
        client=_StaleBlindClient(),
        send_fn=_send_fn,
        recall_limit=6,
    )
    arm = result.arms[VALUE_RECALL_V2_HISTORY]
    assert arm.score == 1.0
    assert "[typed-value count was]" in arm.results[0].recalled


# --- v3 coarse grouping + authoritative composition --------------------------------

def test_coarse_grouping_merges_across_noisy_context() -> None:
    # Long multi-clause turns that lexical grouping fragments should still merge under
    # coarse grouping (attribute+kind+unit+entity only).
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I've been waking up around 8:30 am on Saturdays, which gives me time for coffee.",
            "Considering my jog, I now like to wake up at 7:30 am on Saturdays before breakfast.",
        ),
        grouping="coarse",
    )
    assert _current_values(graph) == ["07:30"]


def test_stale_value_strings_targets_question_kind() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I now have 38 coins."),
        grouping="coarse",
    )
    stale = graph.stale_value_strings("How many coins do I have now?")
    assert "37 coins" in stale
    assert "38 coins" not in stale


def test_snippet_has_stale_is_word_bounded() -> None:
    assert _snippet_has_stale("User mentioned 37 coins earlier.", {"37 coins"})
    # substring-of-a-number must not spuriously match
    assert not _snippet_has_stale("A 1937 nickel is valuable.", {"37"})


def test_v3_authoritative_suppresses_stale_untyped_snippet() -> None:
    class _StaleUntypedAdapter(_ValueAdapter):
        def load_items(self, subset, limit, fixture_path):  # noqa: ANN001
            return [
                {
                    "question_id": "v3-1",
                    "answer": "38",
                    "question_type": "knowledge-update",
                    "question": "How many coins do I have now?",
                    "sessions": _sessions("I have 37 coins.", "I now have 38 coins."),
                }
            ]

    class _StaleUntypedClient(_StaleBlindClient):
        def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
            return ["The user has 37 coins in their collection."]  # stale untyped mention

    result = run_memory_ab(
        _StaleUntypedAdapter(),
        arms=(VALUE_RECALL_V3_COARSE, VALUE_RECALL_V3_AUTHORITATIVE),
        client=_StaleUntypedClient(),
        send_fn=_send_fn,
        recall_limit=6,
    )
    coarse_ctx = result.arms[VALUE_RECALL_V3_COARSE].results[0].recalled
    auth_ctx = result.arms[VALUE_RECALL_V3_AUTHORITATIVE].results[0].recalled
    # coarse keeps the stale untyped snippet; authoritative suppresses it.
    assert "37 coins in their collection" in coarse_ctx
    assert "37 coins in their collection" not in auth_ctx
    assert "38 coins" in auth_ctx


# --- v4 advisory composition (annotate, don't delete) ------------------------------

def test_advisory_marker_cluster_marks_current_and_drops_rejected() -> None:
    # Explicit correction ("used to" on 37, "now" on 38): the author rejected 37, so it is
    # the one deletion that cannot backfire. 38 is annotated current.
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I used to have 37 coins.", "I now have 38 coins."),
        grouping="coarse",
    )
    joined = "\n".join(graph.advisory_recall("How many coins do I have now?", limit=4))
    assert "[typed-value count current]" in joined
    assert "38 coins" in joined
    assert "37 coins" not in joined  # author-rejected -> dropped


def test_advisory_keeps_all_candidates_when_no_correction_marker() -> None:
    # 71315a70 shape: two durations for the same attribute, NO correction marker. Advisory
    # must keep BOTH (never delete the correct one) and label them neutral candidates -
    # latest-mention is not asserted as latest-truth without a marker.
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "I've spent 5-6 hours on my abstract ocean sculpture.",
            "I've spent 10-12 hours on my abstract ocean sculpture.",
        ),
        grouping="coarse",
    )
    joined = "\n".join(graph.advisory_recall("How many hours on my sculpture?", limit=4))
    assert "5-6 hours" in joined
    assert "10-12 hours" in joined  # correct answer never dropped
    assert "[typed-value duration candidate]" in joined
    assert "current" not in joined  # no unearned recency claim


def test_advisory_keeps_distinct_referents_as_candidates() -> None:
    # dfde3500 shape: two weekdays for two different people that coarse grouping merges.
    # Advisory keeps both as candidates so the answer model can use the named referent.
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions(
            "My language exchange class with Juan is on Wednesday evening.",
            "I'm actually meeting Maria on Thursday.",
        ),
        grouping="coarse",
    )
    joined = "\n".join(graph.advisory_recall("What day do I meet Juan?", limit=4))
    assert "Wednesday" in joined  # correct referent's value retained
    assert "Thursday" in joined
    assert joined.count("[typed-value weekday candidate]") == 2


def test_advisory_single_value_cluster_has_no_role_label() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t", _sessions("I have 42 coins."), grouping="coarse"
    )
    joined = "\n".join(graph.advisory_recall("How many coins?", limit=4))
    assert "[typed-value count]" in joined
    assert "candidate" not in joined and "current" not in joined


def test_advisory_arm_does_not_suppress_untyped_backfill() -> None:
    # Unlike v3_authoritative, advisory never removes untyped recall snippets - it only
    # annotates the typed sidecar. The stale untyped mention survives; the model decides.
    class _StaleUntypedClient(_StaleBlindClient):
        def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
            return ["The user has 37 coins in their collection."]

    result = run_memory_ab(
        _ValueAdapter(),
        arms=(VALUE_RECALL_V4_ADVISORY,),
        client=_StaleUntypedClient(),
        send_fn=_send_fn,
        recall_limit=6,
    )
    ctx = result.arms[VALUE_RECALL_V4_ADVISORY].results[0].recalled
    assert "37 coins in their collection" in ctx  # untyped backfill left intact
    assert "38 coins" in ctx  # typed current candidate present


# --- v5 derivation ("assumptions") layer -------------------------------------------

def _derived(graph: SupersededValueGraph, query: str) -> str:
    return "\n".join(graph.derived_recall(query, limit=6))


def test_derived_increment_by_one_from_added_a_new() -> None:
    # 69fee5aa shape: a stated anchor (37) + "added a new coin" -> 38, never stated literally.
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins in my collection.", "I just added a new coin to my collection."),
        grouping="coarse",
    )
    out = _derived(graph, "How many coins do I have now?")
    assert "[typed-value count derived]" in out
    assert "~38" in out
    assert "37 coins" in out  # stated anchor retained (advisory base, nothing deleted)


def test_derived_increment_by_n_more() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I own 10 books.", "I bought 3 more books at the shop."),
        grouping="coarse",
    )
    out = _derived(graph, "How many books do I have?")
    assert "~13" in out  # single +3 event; the "3 books" in the delta is not a second anchor


def test_derived_decrement_from_sold() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 38 coins.", "I sold 5 coins last week."),
        grouping="coarse",
    )
    out = _derived(graph, "How many coins do I have?")
    assert "~33" in out


def test_derived_requires_an_anchor() -> None:
    # Delta with no stated absolute -> nothing to fold onto.
    graph = SupersededValueGraph.from_sessions(
        "t", _sessions("I added a new coin to my collection."), grouping="coarse"
    )
    assert "derived" not in _derived(graph, "How many coins do I have?")


def test_derived_skips_ambiguous_two_anchors() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I have 50 coins in total now.", "I added a new coin."),
        grouping="coarse",
    )
    # Two distinct stated anchors (37, 50) in one cluster -> ambiguous base -> no derived hint.
    assert "derived" not in _derived(graph, "How many coins do I have?")


def test_derived_skips_when_result_is_already_stated() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I added a new coin.", "Now I have 38 coins."),
        grouping="coarse",
    )
    # 38 is stated, so the fold is redundant; v4 advisory already surfaces 38.
    assert "derived" not in _derived(graph, "How many coins do I have?")


def test_derived_only_when_noun_is_asked_about() -> None:
    graph = SupersededValueGraph.from_sessions(
        "t",
        _sessions("I have 37 coins.", "I added a new coin."),
        grouping="coarse",
    )
    # Question about an unrelated noun -> no derived coin hint.
    assert "derived" not in _derived(graph, "How many books do I own?")


def test_v5_arm_emits_derived_hint_and_keeps_backfill() -> None:
    class _DeltaAdapter(_ValueAdapter):
        def load_items(self, subset, limit, fixture_path):  # noqa: ANN001
            return [
                {
                    "question_id": "v5-1",
                    "answer": "38",
                    "question_type": "knowledge-update",
                    "question": "How many coins do I have now?",
                    "sessions": _sessions(
                        "I have 37 coins in my collection.",
                        "I just added a new coin to my collection.",
                    ),
                }
            ]

    class _CoinClient(_StaleBlindClient):
        def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
            return ["The user collects coins as a hobby."]  # untyped, no number

    def _derive_send(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
        context = messages[-1]["content"]
        return ("38" if "~38" in context else "unknown"), 1.0, {"prompt_tokens": 10, "completion_tokens": 1}

    result = run_memory_ab(
        _DeltaAdapter(),
        arms=(VALUE_RECALL_V5_DERIVED,),
        client=_CoinClient(),
        send_fn=_derive_send,
        recall_limit=6,
    )
    arm = result.arms[VALUE_RECALL_V5_DERIVED]
    ctx = arm.results[0].recalled
    assert "[typed-value count derived]" in ctx and "~38" in ctx
    assert "collects coins as a hobby" in ctx  # untyped backfill intact
    assert arm.score == 1.0
