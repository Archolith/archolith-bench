"""Offline tests for the LLM-judge scorer (no network: a stub judge send_fn)."""

from __future__ import annotations

from archolith_bench.harness.scoring import LLMJudgeScorer, _is_abstention, _parse_yes


def test_parse_yes_variants():
    assert _parse_yes("yes")
    assert _parse_yes("Yes, correct.")
    assert _parse_yes('"yes"')
    assert not _parse_yes("no")
    assert not _parse_yes("No, the answer is wrong")
    assert not _parse_yes("")


def test_is_abstention_detection():
    assert _is_abstention({"question_type": "abstention", "answer": "whatever"})
    assert _is_abstention({"question_type": "single-session-user", "answer": ""})
    assert _is_abstention({"question_type": "x", "answer": "no answer"})
    assert not _is_abstention({"question_type": "single-session-user", "answer": "Denver"})


def _stub_send_fn_yes_if_gold():
    """Judge says 'yes' iff the gold answer text appears in the model response."""
    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        prompt = messages[-1]["content"]
        # gold is in the judge prompt as "Gold answer: <gold>"
        # (LongMemEval items get the official prompt: "Correct Answer:" / "Model Response:")
        gold = ""
        for line in prompt.splitlines():
            if line.startswith(("Gold answer:", "Correct Answer:")):
                gold = line.split(":", 1)[1].strip()
        # the response is in "Model response: <resp>"
        resp = ""
        for line in prompt.splitlines():
            if line.startswith(("Model response:", "Model Response:")):
                resp = line.split(":", 1)[1].strip()
        verdict = "yes" if gold and gold.lower() in resp.lower() else "no"
        return verdict, 1.0, {"prompt_tokens": 10, "completion_tokens": 1}
    return send_fn


def test_llm_judge_scores_correct_and_incorrect():
    judge = LLMJudgeScorer(base_url="http://x/v1", api_key="k", model="gpt-4o-mini",
                           send_fn=_stub_send_fn_yes_if_gold())
    item = {"question": "What city?", "answer": "Denver", "question_type": "single-session-user"}
    assert judge(item, "I think it was Denver in March") is True
    assert judge(item, "It was Boston") is False
    assert judge.last_usage == {"prompt_tokens": 10, "completion_tokens": 1}
    judge.close()


def test_llm_judge_abstention_uses_abstention_branch():
    """Abstention items must not require a gold match; judge sees the abstention prompt."""
    seen = {}

    def send_fn(client, base_url, api_key, messages, model, **kwargs):
        seen["system"] = messages[0]["content"]
        # say yes if the response declines
        resp_line = next((ln for ln in messages[-1]["content"].splitlines() if ln.startswith("Model response:")), "")
        verdict = "yes" if "don't know" in resp_line.lower() else "no"
        return verdict, 1.0, {"prompt_tokens": 5, "completion_tokens": 1}

    judge = LLMJudgeScorer(base_url="http://x/v1", api_key="k", model="gpt-4o-mini", send_fn=send_fn)
    item = {"question": "What is X?", "answer": "", "question_type": "abstention"}
    assert judge(item, "I don't know that.") is True
    assert judge(item, "It is 42.") is False
    assert "UNANSWERABLE" in seen["system"]
    judge.close()
