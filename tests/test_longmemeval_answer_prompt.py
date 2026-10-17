"""LongMemEval memory answer prompt: counting rule, scoped veto, and the question's date."""

from __future__ import annotations

from archolith_bench.harness.longmemeval import _MEMORY_SYSTEM_PROMPT, LongMemEvalMemoryAdapter
from archolith_bench.harness.memory_ab import run_memory_ab

DATE = "2023/02/15 (Wed) 19:47"


def test_counting_rule_rates_fit_and_counts_high_and_medium() -> None:
    assert 'For "how many" or list questions: list every candidate item' in _MEMORY_SYSTEM_PROMPT
    assert "high (clearly fits)" in _MEMORY_SYSTEM_PROMPT
    assert "medium (fits under a reasonable everyday reading" in _MEMORY_SYSTEM_PROMPT
    assert "low (only a stretch)" in _MEMORY_SYSTEM_PROMPT
    assert "Count high and medium. Give one final number" in _MEMORY_SYSTEM_PROMPT


def test_veto_policy_is_scoped_to_verdict_records() -> None:
    assert ("Safe veto policy (structured verdict and authoritative records only; never use it to "
            "drop an ordinary memory that answers the question)") in _MEMORY_SYSTEM_PROMPT
    assert "ignore a memory item only when" not in _MEMORY_SYSTEM_PROMPT


def test_prompt_names_no_benchmark_specific_terms() -> None:
    # The rule must stay generic: no place, category or item taken from a benchmark question.
    lowered = _MEMORY_SYSTEM_PROMPT.lower()
    for term in ("store", "shop", "dry clean", "cloth", "camping", "project", "kit", "movie"):
        assert term not in lowered, term


def test_user_message_leads_with_the_question_date_when_given() -> None:
    messages = LongMemEvalMemoryAdapter().build_messages("fact A", "how many?", question_date=DATE)
    assert messages[1]["content"] == f"Current date: {DATE}\n\nRetrieved memory:\nfact A\n\nQuestion: how many?"


def test_user_message_is_unchanged_without_a_date() -> None:
    messages = LongMemEvalMemoryAdapter().build_messages("fact A", "how many?")
    assert messages[1]["content"] == "Retrieved memory:\nfact A\n\nQuestion: how many?"


def test_question_date_reads_the_item() -> None:
    adapter = LongMemEvalMemoryAdapter()
    assert adapter.question_date({"question_date": f" {DATE} "}) == DATE
    assert adapter.question_date({}) is None
    assert adapter.question_date({"question_date": ""}) is None


class _Client:
    def new_group(self) -> str:
        return "ns-1"

    def record_turn_evidence(self, namespace: str, text: str, **kwargs) -> dict:
        return {"turn_id": "turn-1"}

    def ingest(self, group_id: str, role: str, content: str, **kwargs) -> None:
        pass

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        return ["fact A"]

    def reset(self, group_id: str) -> None:
        pass


class _OneItemAdapter(LongMemEvalMemoryAdapter):
    def load_items(self, subset=None, limit=None, fixture_path=None) -> list[dict]:  # noqa: ANN001
        return [{"question_id": "q1", "question": "how many?", "answer": "1", "question_date": DATE,
                 "haystack_sessions": [[{"role": "user", "content": "I have one"}]]}]


def test_run_loop_passes_the_item_date_into_the_answer_prompt() -> None:
    sent: list[list[dict]] = []

    def send_fn(chat_client, base_url, api_key, messages, model, **kw):  # noqa: ANN001
        sent.append(messages)
        return "1", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    run_memory_ab(_OneItemAdapter(), arms=("menhir_recall",), client=_Client(), send_fn=send_fn,
                  reset_memory=False)
    assert len(sent) == 1
    assert sent[0][1]["content"].startswith(f"Current date: {DATE}\n\nRetrieved memory:\n")
