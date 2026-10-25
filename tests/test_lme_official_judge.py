"""LongMemEval items are graded with LongMemEval's official judge prompts; `_abs` ids are abstentions."""

from __future__ import annotations

import pytest

from archolith_bench.harness.checkpoint import checkpoint_prompt_tag
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter
from archolith_bench.harness.scoring import (
    _JUDGE_SYSTEM,
    _UNOFFICIAL_JUDGE_SYSTEM,
    LLMJudgeScorer,
    UnofficialJudgeScorer,
    _is_abstention,
    _judge_messages,
)

EXPLANATION = "The information provided is not enough. You mentioned Hawaii but not Seattle."


def _item(qid: str, qtype: str, answer: str = "3") -> dict:
    return {"question_id": qid, "question_type": qtype, "question": "How many?", "answer": answer}


@pytest.mark.parametrize("qtype", ["single-session-user", "single-session-assistant", "multi-session",
                                   "temporal-reasoning", "knowledge-update", "single-session-preference"])
def test_each_lme_type_gets_one_official_user_prompt(qtype) -> None:
    messages = _judge_messages(_item("q1", qtype), "There are 3.")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert content.startswith("I will give you a question, a ")
    assert content.endswith("Model Response: There are 3.\n\nIs the model response correct? Answer yes or no only.")
    assert _JUDGE_SYSTEM not in content


def test_type_specific_rules() -> None:
    def prompt(qtype: str) -> str:
        return _judge_messages(_item("q1", qtype), "r")[0]["content"]

    assert "do not penalize off-by-one errors" in prompt("temporal-reasoning")
    assert "off-by-one" not in prompt("multi-session")
    assert "some previous information along with an updated answer" in prompt("knowledge-update")
    assert "contains all the intermediate steps" in prompt("multi-session")
    pref = prompt("single-session-preference")
    assert "Rubric: 3" in pref and "does not need to reflect all the points in the rubric" in pref


@pytest.mark.parametrize("qtype", ["multi-session", "temporal-reasoning", "knowledge-update",
                                   "single-session-user"])
def test_abs_question_id_is_an_abstention_graded_against_its_explanation(qtype) -> None:
    item = _item("edced276_abs", qtype, EXPLANATION)
    assert _is_abstention(item)
    content = _judge_messages(item, "I don't know.")[0]["content"]
    assert content.startswith("I will give you an unanswerable question, an explanation")
    assert f"Explanation: {EXPLANATION}" in content
    assert "correctly identify the question as unanswerable" in content


def test_non_lme_items_keep_the_generic_prompts() -> None:
    msgs = _judge_messages({"question": "q", "answer": "a"}, "r")
    assert msgs[0] == {"role": "system", "content": _JUDGE_SYSTEM}
    assert not _is_abstention({"question_id": "q1", "question_type": "multi-session", "answer": "3"})


def _record(reply: str, seen: list):
    def send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
        seen.append(messages)
        return reply, 1.0, {}

    return send_fn


def test_scorer_sends_the_official_prompt_and_parses_the_verdict() -> None:
    seen: list = []
    judge = LLMJudgeScorer(base_url="http://x", api_key="k", model="m", send_fn=_record("Yes", seen))
    item = _item("ba358f49_abs", "multi-session", EXPLANATION)
    assert judge(item, "I don't know.") is True
    assert seen[0] == _judge_messages(item, "I don't know.")


def test_unofficial_judge_grades_abs_items_with_the_official_abstention_prompt() -> None:
    seen: list = []
    judge = UnofficialJudgeScorer(base_url="http://x", api_key="k", model="m", send_fn=_record("no", seen))
    item = _item("q9_abs", "temporal-reasoning", EXPLANATION)
    judge(item, "It was March.")
    assert seen[0] == _judge_messages(item, "It was March.")
    assert all(m.get("content") != _UNOFFICIAL_JUDGE_SYSTEM for m in seen[0])


def test_containment_scorer_treats_abs_ids_as_abstentions() -> None:
    adapter = LongMemEvalMemoryAdapter()
    item = _item("edced276_abs", "multi-session", EXPLANATION)
    assert adapter.score(item, "I don't know how many days.") is True
    assert adapter.score(item, "You spent 10 days.") is False
    # The same text under a normal id is graded against the gold, as before.
    assert adapter.score(_item("q1", "multi-session", "10 days"), "You spent 10 days.") is True


def test_official_judge_runs_get_their_own_checkpoint_tag() -> None:
    plain = checkpoint_prompt_tag(LongMemEvalMemoryAdapter, source_memories=True)
    judged = checkpoint_prompt_tag(LongMemEvalMemoryAdapter, source_memories=True, official_judge=True)
    assert judged == f"{plain}oj"
    assert checkpoint_prompt_tag(object(), official_judge=True) == "oj"
