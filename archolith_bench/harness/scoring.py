"""Pluggable answer scorers for the memory harness.

The default `containment` scorer (longmemeval._score_answer) is fast and offline but
is NOT how the field reports LongMemEval: the official protocol and the published
Mem0/Zep numbers use an LLM judge (a GPT-4-class model decides correctness). To make
our accuracy comparable to those, this provides an LLM-judge scorer with the same
shape as `adapter.score(item, response_text) -> bool`.

For a publishable, apples-to-apples number use the same judge model the field uses
(gpt-4o); gpt-6-luna is the default for internal runs (gpt-4o-mini is deprecated).
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ..core.api import send_chat

# score_fn(item, response_text) -> bool
ScoreFn = Callable[[dict, str], bool]

_JUDGE_SYSTEM = (
    "You are a strict grader for a question-answering benchmark. You are given a "
    "question, the reference (gold) answer, and a model's response. Decide whether the "
    "response is correct: it is correct if it conveys the gold answer (exact wording is "
    "not required), and incorrect if it is wrong, contradicts the gold answer, or fails "
    "to answer. Reply with a single word: 'yes' if correct, 'no' if incorrect."
)

_JUDGE_SYSTEM_ABSTENTION = (
    "You are a strict grader for a question-answering benchmark. This question is "
    "UNANSWERABLE from the available information, so the correct behavior is for the "
    "model to decline or say it does not know. You are given the question and the "
    "model's response. Reply with a single word: 'yes' if the response correctly "
    "declines / says it cannot answer or lacks the information, 'no' if it instead "
    "makes up or asserts an answer."
)


# LongMemEval's official judge prompts, verbatim from get_anscheck_prompt() in
# xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py (commit d6dc8b50a2d9). Each is sent as a
# single user message; placeholders are question, answer (rubric / explanation), response.
_LME_BASE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response is equivalent "
    "to the correct answer or contains all the intermediate steps to get the correct answer, you should "
    "also answer yes. If the response only contains a subset of the information required by the "
    "answer, answer no. "
)
_LME_QA_TAIL = (
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_JUDGE_TEMPLATES = {
    "single-session-user": _LME_BASE + _LME_QA_TAIL,
    "single-session-assistant": _LME_BASE + _LME_QA_TAIL,
    "multi-session": _LME_BASE + _LME_QA_TAIL,
    "temporal-reasoning": _LME_BASE + (
        "In addition, do not penalize off-by-one errors for the number of days. If the question asks "
        "for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
        "predicting 19 days when the answer is 18), the model's response is still correct. "
    ) + _LME_QA_TAIL,
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes "
        "if the response contains the correct answer. Otherwise, answer no. If the response contains "
        "some previous information along with an updated answer, the response should be considered as "
        "correct as long as the updated answer is the required answer."
    ) + _LME_QA_TAIL,
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a "
        "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
        "The model does not need to reflect all the points in the rubric. The response is correct as "
        "long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\n"
        "Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
    ),
}
_LME_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say "
    "that the information is incomplete, or some other information is given but the asked information "
    "is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly "
    "identify the question as unanswerable? Answer yes or no only."
)


def is_lme_abstention(item: dict) -> bool:
    """LongMemEval marks unanswerable questions by an ``_abs`` question id, not by question_type."""
    return "_abs" in str(item.get("question_id") or "")


def _is_abstention(item: dict) -> bool:
    gold = str(item.get("answer") or "").strip().lower()
    return (is_lme_abstention(item) or item.get("question_type") == "abstention"
            or gold in {"", "no answer", "unknown"})


def _lme_judge_prompt(item: dict, response_text: str) -> str | None:
    """The official LongMemEval judge prompt for a LongMemEval item, else None."""
    question, answer = str(item.get("question") or ""), str(item.get("answer") or "")
    if is_lme_abstention(item):
        return _LME_ABSTENTION_TEMPLATE.format(question, answer, response_text)
    template = _LME_JUDGE_TEMPLATES.get(str(item.get("question_type") or ""))
    return template.format(question, answer, response_text) if template else None


def _generic_user_message(item: dict, response_text: str) -> dict:
    question, gold = str(item.get("question") or ""), str(item.get("answer") or "")
    user = (f"Question: {question}\n\nGold answer: {gold}\n\nModel response: {response_text}\n\n"
            "Is the model response correct? Answer yes or no.")
    return {"role": "user", "content": user}


def _judge_messages(item: dict, response_text: str) -> list[dict]:
    lme = _lme_judge_prompt(item, response_text)
    if lme is not None:
        return [{"role": "user", "content": lme}]
    question = str(item.get("question") or "")
    if _is_abstention(item):
        user = f"Question: {question}\n\nModel response: {response_text}\n\nIs the response a correct abstention? Answer yes or no."
        return [{"role": "system", "content": _JUDGE_SYSTEM_ABSTENTION},
                {"role": "user", "content": user}]
    return [{"role": "system", "content": _JUDGE_SYSTEM}, _generic_user_message(item, response_text)]


def _parse_yes(text: str) -> bool:
    t = (text or "").strip().lower().lstrip("\"'`*.- ")
    return t.startswith("yes") or t.startswith("y ") or t == "y"


class LLMJudgeScorer:
    """LLM-judge scorer: grades each answer with a judge model (LongMemEval protocol)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        send_fn=send_chat,  # noqa: ANN001
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.send_fn = send_fn
        self._client = client or httpx.Client(timeout=60)
        self._own_client = client is None
        self.last_usage: dict = {}

    def __call__(self, item: dict, response_text: str) -> bool:
        messages = _judge_messages(item, response_text)
        # strict: a judge failure must not parse as "no" (a silent wrong grade).
        text, _latency, usage = self.send_fn(
            self._client, self.base_url, self.api_key, messages, self.model, strict=True
        )
        self.last_usage = dict(usage or {})
        return _parse_yes(text)

    def close(self) -> None:
        if self._own_client:
            self._client.close()


_UNOFFICIAL_JUDGE_SYSTEM = (
    "You are grading a question-answering response in substance. It is correct if its "
    "committed answer matches the gold answer. A lower bound phrased as 'at least N' "
    "counts as matching when N equals the gold number. An amount phrased as approximate "
    "or as a lower bound ('about', 'roughly', 'over' N) counts as matching when N equals "
    "the gold amount; for a count of items, 'more than N' or 'over N' does not match a "
    "gold of N. When the gold names specific items, the items the response counts must "
    "be the same items (paraphrase is fine); a matching number made up of different "
    "items is incorrect, while a matching number with no item list still matches. Extra "
    "caveats, hedges, or mentions of excluded items do not make it wrong when the stated "
    "answer matches the gold. It is incorrect if it commits to a different number or "
    "answer, contradicts "
    "the gold, or fails to answer. Reply with a single word: 'yes' if correct, 'no' if "
    "incorrect."
)


def _unofficial_judge_messages(item: dict, response_text: str) -> list[dict]:
    if _is_abstention(item):
        return _judge_messages(item, response_text)
    return [{"role": "system", "content": _UNOFFICIAL_JUDGE_SYSTEM}, _generic_user_message(item, response_text)]


class UnofficialJudgeScorer:
    """Substance-grading judge run only on official-wrong items; reported separately."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        send_fn=send_chat,  # noqa: ANN001
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.send_fn = send_fn
        self._client = client or httpx.Client(timeout=60)
        self._own_client = client is None
        self.last_usage: dict = {}

    def __call__(self, item: dict, response_text: str) -> bool:
        messages = _unofficial_judge_messages(item, response_text)
        text, _latency, usage = self.send_fn(
            self._client, self.base_url, self.api_key, messages, self.model, strict=True
        )
        self.last_usage = dict(usage or {})
        return _parse_yes(text)

    def close(self) -> None:
        if self._own_client:
            self._client.close()
