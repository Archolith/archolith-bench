"""Tests for adaptive claim segmentation's durable-content routing.

The failure these guard against is silent: a turn misrouted to CONTEXT_ONLY is
never extracted, so its knowledge update is missing from the graph and only shows
up later as an unexplained recall miss.
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


segmenter = _load_script(
    "lme_claim_segmenter_test", "scripts/longmemeval/lib/claim_segmenter.py"
)
MODE = segmenter.SegmentationMode


@pytest.mark.parametrize(
    "text",
    [
        # A question is an ordinary carrier for a knowledge update. Every sentence
        # here ends in "?", but the move is still a durable fact.
        "Since I moved to Portland, which neighborhoods should I check out?",
        "I switched to the night shift last month, any tips for sleeping?",
        # Mixed statement + question, durable half first.
        "I quit my job at Acme. What should I put on my resume?",
    ],
)
def test_questions_carrying_durable_facts_are_not_interrogative(text: str) -> None:
    assert segmenter.is_purely_interrogative(text) is False
    assert segmenter.segmentation_mode("user", text) == MODE.EXTRACT_WHOLE


@pytest.mark.parametrize(
    "text",
    [
        # Possessive assertions: no first-person verb, no capitalized subject, but
        # this is exactly the shape a knowledge-update item takes.
        "yes, my rent is $2,400 now",
        "my sister's address is 44 Oak Lane",
        "my commute is 45 minutes each way",
    ],
)
def test_possessive_and_quantity_updates_stay_extractable(text: str) -> None:
    assert segmenter.has_durable_signals(text) is True
    assert segmenter.segmentation_mode("user", text) == MODE.EXTRACT_WHOLE


@pytest.mark.parametrize(
    "text",
    [
        # A wh-question about a past change asserts the change happened -- the fact lives in
        # the presupposition, not the surface form.
        "Why did I move to Seattle?",
        "When did I switch to the new plan?",
        # Frequencies are durable personal facts, and the count is spelled as a word so no
        # digit pattern sees it.
        "I go three times a week.",
        "three times a week.",
        # Possessed subject rather than "I": no first-person verb pattern catches this.
        "my mom uses the same app.",
    ],
)
def test_reviewed_detector_misses_are_extractable(text: str) -> None:
    """Exact turns a review found routed to CONTEXT_ONLY despite carrying durable facts."""
    assert segmenter.has_durable_signals(text) is True
    assert segmenter.segmentation_mode("user", text) == MODE.EXTRACT_WHOLE


@pytest.mark.parametrize(
    "text",
    [
        # Yes/no questions presuppose nothing -- they may be genuinely asking -- so the
        # presuppositional rule stays wh-only and these remain conservative.
        "Did I mention that already?",
        "Do you have any tips?",
        "Have you tried it?",
    ],
)
def test_yes_no_questions_stay_conservative(text: str) -> None:
    assert segmenter.segmentation_mode("user", text) == MODE.CONTEXT_ONLY


def test_first_person_durable_signal_is_clause_aware() -> None:
    """A durable verb nested inside a question object is not itself an assertion."""
    fact_free_question = (
        "What's the best way to organize my shoe rack to make it easy to find the pair I need? "
        "Should I sort them by type, color, or occasion?"
    )
    declarative_then_question = (
        "I need to get some fresh herbs like cilantro and parsley for my fajitas, "
        "do you have any tips on how to keep them fresh for a longer period?"
    )

    # Both strings contain the same lexical signal; only one uses it in a declarative clause.
    assert segmenter._FIRST_PERSON_DURABLE.search(fact_free_question)
    assert segmenter._FIRST_PERSON_DURABLE.search(declarative_then_question)
    assert segmenter._has_first_person_durable_assertion(fact_free_question) is False
    assert segmenter._has_first_person_durable_assertion(declarative_then_question) is True
    assert segmenter.segmentation_mode("user", fact_free_question) == MODE.CONTEXT_ONLY
    assert segmenter.segmentation_mode("user", declarative_then_question) == MODE.EXTRACT_WHOLE


@pytest.mark.parametrize(
    "text",
    [
        "Thanks again for your help!",
        "That sounds great!",
        "Sure, sounds good.",
        "I think that sounds great. What kind of elements were you thinking of?",
    ],
)
def test_phatic_and_reactive_turns_remain_context_only(text: str) -> None:
    """Regression guard: broadening durable detection must not start extracting filler.

    These produce only ``{"name":"user"}`` with zero edges, which fails the episode
    and -- with the strict FAILED gate -- stops the whole build.
    """
    assert segmenter.segmentation_mode("user", text) == MODE.CONTEXT_ONLY


@pytest.mark.parametrize(
    "text",
    [
        # Representative shapes from a fixture-wide audit: user turns that carry the answer but
        # match no durable pattern. Under an allowlist all of these vanished silently.
        "I've completed 30 videos of the course so far.",
        "I used my Canon EOS 80D on five trips.",
        "I caught 7 largemouth bass at the lake.",
        # Mixed question + unrecognised declarative: the statement half is the knowledge update.
        "I picked up a new hobby recently. Any suggestions?",
        "The trail was muddy. What boots do you recommend?",
        # Plain declaratives no pattern anticipates. Unknown means extract, not discard.
        "The plumber replaced the whole fixture.",
        "We ended up going with the blue one.",
    ],
)
def test_unrecognized_statements_fail_open_to_extraction(text: str) -> None:
    """CONTEXT_ONLY is a narrow denylist, not the complement of an allowlist.

    A fact dropped here is dropped silently and looks identical to a model with nothing to say.
    A fact wrongly extracted, if it collapses, trips the strict FAILED gate -- loud and fixable.
    """
    assert segmenter.is_context_only_user_turn(text) is False
    assert segmenter.segmentation_mode("user", text) != MODE.CONTEXT_ONLY


def test_one_unrecognized_declarative_carries_the_whole_turn() -> None:
    """Every declarative must be provably phatic; one that is not sends the turn to extraction."""
    mixed = "Thanks, that's really helpful! I switched my plan to the annual one. Anything else?"

    assert segmenter.is_phatic_sentence("Thanks, that's really helpful!") is True
    assert segmenter.is_phatic_sentence("I switched my plan to the annual one.") is False
    assert segmenter.is_context_only_user_turn(mixed) is False


@pytest.mark.parametrize(
    "text",
    [
        "I'm planning a trip to Europe, can you recommend a hotel?",
        "I've been experimenting with noodles, what should I try next?",
        "I'm feeling overwhelmed with work and was wondering if you could help?",
        "I've been using my Starbucks Rewards app, do you have any tips?",
        "I hope to exercise tomorrow, what routine would you suggest?",
        "I know Alice from work, should I invite her?",
    ],
)
def test_personal_declarative_clause_inside_question_is_extracted(text: str) -> None:
    assert segmenter.has_first_person_declarative(text) is True
    assert segmenter.segmentation_mode("user", text) != MODE.CONTEXT_ONLY


@pytest.mark.parametrize(
    "sentence, phatic",
    [
        ("Thanks again for your help!", True),
        ("That sounds great!", True),
        ("I think that sounds great.", True),
        ("Got it, thanks.", True),
        # A number is a value; a value is not small talk.
        ("Sounds good, 30 of them.", False),
        # A named thing away from the sentence opener is content.
        ("Sounds good, I'll check Amazon.", False),
        # Too long to prove anything about.
        ("I think the whole approach to the migration was reasonable given the constraints.", False),
    ],
)
def test_phatic_sentence_boundaries(sentence: str, phatic: bool) -> None:
    assert segmenter.is_phatic_sentence(sentence) is phatic


def test_thanks_again_is_not_read_as_a_correction_marker() -> None:
    """"again" corrects a fact in "I moved again" and is pure pleasantry in "thanks again"."""
    assert segmenter._has_correction_signal("Thanks again for your help!") is False
    assert segmenter._has_correction_signal("I moved again last spring") is True


def test_assistant_turns_are_unaffected_by_user_durable_routing() -> None:
    assert segmenter.segmentation_mode("assistant", "That's a great question!") == MODE.CONTEXT_ONLY
    assert (
        segmenter.segmentation_mode("assistant", "I've scheduled your appointment.")
        == MODE.EXTRACT_WHOLE
    )


# ---------------------------------------------------------------------------
# Assistant durable-content gate: precision
#
# The gate previously ended in a bare alternation of past participles, which matched 58 of
# 915 assistant turns on the KU fixture -- every one a false positive, each admitting ~2.3KB
# of advice prose into the graph as a whole episode. The cases below are the measured
# false-positive forms and the transactional forms that must survive narrowing.
# ---------------------------------------------------------------------------

ASSISTANT_DURABLE = [
    "I've scheduled your appointment.",
    "I've booked your reservation for Friday.",
    "I have configured your settings.",
    "Your order has been confirmed.",
    "The subscription was cancelled.",
    "Your account is suspended.",
    "I'll remember that you prefer aisle seats.",
]

# Verbatim from fixtures/longmemeval/knowledge_update_subset.json assistant turns.
ASSISTANT_ADVICE_PROSE = [
    # Bare participles with no subject binding: 51 of the 58 measured matches.
    "Once you've completed that, focus on Task 1 (Finish project report).",
    "Any deadlines or time-sensitive tasks that need to be completed this week?",
    "Given that you've already done 2,000 miles on your current wheels, it's likely time.",
    "If they're still in excellent shape, they'll be more likely to be accepted by the thrift store.",
    "I don't recall you getting pre-approved for a mortgage or any other personal detail.",
    # Assistant generating advice content, not recording a user fact.
    "Based on your input, I've created a suggested schedule for you to tackle your projects.",
    "I've created a 20-mile route that meets your requirements.",
    "With that information, I've created a daily routine that incorporates your episodes.",
    # Instructional prose about filling in fields, not a record in a terminal state.
    "Optimize your profile: Ensure your profile is complete, including a profile picture, bio.",
    "To access Insights, make sure your account is set to a Business or Creator profile.",
]


@pytest.mark.parametrize("text", ASSISTANT_DURABLE)
def test_transactional_assistant_turns_are_extracted(text: str) -> None:
    assert segmenter.is_memory_bearing_assistant_output(text) is True
    assert segmenter.segmentation_mode("assistant", text) == MODE.EXTRACT_WHOLE


@pytest.mark.parametrize("text", ASSISTANT_ADVICE_PROSE)
def test_assistant_advice_prose_stays_context_only(text: str) -> None:
    """Each of these matched the old gate and entered the graph as a whole episode."""
    assert segmenter.is_memory_bearing_assistant_output(text) is False
    assert segmenter.segmentation_mode("assistant", text) == MODE.CONTEXT_ONLY


def test_possessive_is_required_after_a_completed_action() -> None:
    """"I've created a schedule" is content generation; "I've created your schedule" is a
    fact about the user's records. The possessive is what separates them."""
    assert segmenter.is_memory_bearing_assistant_output("I've created a schedule.") is False
    assert segmenter.is_memory_bearing_assistant_output("I've created your schedule.") is True


def test_bare_complete_is_not_a_terminal_state() -> None:
    """"is complete" describes fullness and appears in advice; "is completed" is an action
    that finished. Accepting the adjective reintroduced a measured false positive."""
    assert segmenter.is_memory_bearing_assistant_output("Your profile is complete.") is False
    assert segmenter.is_memory_bearing_assistant_output("Your order is completed.") is True


def test_no_assistant_turn_in_the_ku_fixture_matches_the_gate() -> None:
    """Whole-corpus precision check: 58 matches before this narrowing, 0 after.

    If a future edit widens the gate, this is where it shows up -- as a count, against real
    advice prose, rather than as a judgement call about one example.
    """
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "longmemeval" / "knowledge_update_subset.json"
    )
    items = json.loads(fixture.read_text(encoding="utf-8"))
    matched = [
        turn["content"]
        for item in items
        for session in item["haystack_sessions"]
        for turn in session
        if turn.get("role") == "assistant"
        and segmenter.is_memory_bearing_assistant_output(turn.get("content") or "")
    ]
    assert matched == [], (
        f"{len(matched)} assistant turn(s) now match the durable gate; "
        f"first: {matched[0][:160]!r}"
    )
