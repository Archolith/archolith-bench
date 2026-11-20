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

from .. import prompts as _prompts
from ..core.api import send_chat

# score_fn(item, response_text) -> bool
ScoreFn = Callable[[dict, str], bool]

# Prompt text lives in archolith_bench/prompts/judge/.
_JUDGE_SYSTEM = _prompts.load(_prompts.JUDGE_GENERIC)
_JUDGE_SYSTEM_ABSTENTION = _prompts.load(_prompts.JUDGE_GENERIC_ABSTENTION)

# LongMemEval's official judge prompts, verbatim from get_anscheck_prompt() in
# xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py (commit d6dc8b50a2d9). Each is sent as a
# single user message; placeholders are question, answer (rubric / explanation), response.
_LME_JUDGE_TEMPLATES = {t: _prompts.load(_prompts.lme_judge(t)) for t in _prompts.LME_QUESTION_TYPES}
_LME_ABSTENTION_TEMPLATE = _prompts.load(_prompts.JUDGE_LME_ABSTENTION)


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


_UNOFFICIAL_JUDGE_SYSTEM = _prompts.load(_prompts.JUDGE_UNOFFICIAL)


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
