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


def _is_abstention(item: dict) -> bool:
    gold = str(item.get("answer") or "").strip().lower()
    return item.get("question_type") == "abstention" or gold in {"", "no answer", "unknown"}


def _judge_messages(item: dict, response_text: str) -> list[dict]:
    question = str(item.get("question") or "")
    if _is_abstention(item):
        user = f"Question: {question}\n\nModel response: {response_text}\n\nIs the response a correct abstention? Answer yes or no."
        return [{"role": "system", "content": _JUDGE_SYSTEM_ABSTENTION},
                {"role": "user", "content": user}]
    gold = str(item.get("answer") or "")
    user = (f"Question: {question}\n\nGold answer: {gold}\n\nModel response: {response_text}\n\n"
            "Is the model response correct? Answer yes or no.")
    return [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}]


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
    "counts as matching when N equals the gold number. Extra caveats, hedges, or "
    "mentions of excluded items do not make it wrong when the stated answer matches the "
    "gold. It is incorrect if it commits to a different number or answer, contradicts "
    "the gold, or fails to answer. Reply with a single word: 'yes' if correct, 'no' if "
    "incorrect."
)


def _unofficial_judge_messages(item: dict, response_text: str) -> list[dict]:
    if _is_abstention(item):
        return _judge_messages(item, response_text)
    msgs = _judge_messages(item, response_text)
    return [{"role": "system", "content": _UNOFFICIAL_JUDGE_SYSTEM}, msgs[1]]


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
