"""LongMemEval memory answer prompt: counting rule, scoped veto, and the question's date."""

from __future__ import annotations

from archolith_bench.harness.checkpoint import checkpoint_path_for
from archolith_bench.harness.longmemeval import _MEMORY_SYSTEM_PROMPT, LongMemEvalMemoryAdapter
from archolith_bench.harness.memory_ab import run_memory_ab

DATE = "2023/02/15 (Wed) 19:47"


def test_counting_rule_rates_fit_and_counts_high_and_medium() -> None:
    assert 'For questions that count distinct things ("how many" items' in _MEMORY_SYSTEM_PROMPT
    assert "list every candidate item" in _MEMORY_SYSTEM_PROMPT
    assert "high (clearly fits)" in _MEMORY_SYSTEM_PROMPT
    assert "medium (fits under a reasonable everyday reading" in _MEMORY_SYSTEM_PROMPT
    assert "low (only a stretch)" in _MEMORY_SYSTEM_PROMPT
    assert "Count high and medium, each distinct item once. Give one final number" in _MEMORY_SYSTEM_PROMPT


def test_current_quantity_questions_answer_the_value_not_a_count() -> None:
    assert ('When "how many" instead asks for a single current quantity (an amount the user has or has '
            "reached now), do not count mentions: answer with the latest supported value"
            ) in _MEMORY_SYSTEM_PROMPT


def test_candidate_answers_get_the_best_supported_one_not_a_bare_dont_know() -> None:
    assert "Answer whenever the retrieved memory addresses the question" in _MEMORY_SYSTEM_PROMPT
    assert ('do not stop at "I don\'t know": give the best-supported candidate as your answer, then briefly '
            "name the other candidate") in _MEMORY_SYSTEM_PROMPT
    assert ("Say you don't know only when no retrieved memory addresses what the question asks about"
            ) in _MEMORY_SYSTEM_PROMPT
    # Undated candidates: a value stated as a change outranks one stated without such wording.
    assert 'prefer a value the user states as a change or update ("now"' in _MEMORY_SYSTEM_PROMPT
    assert "say you don't know; do not arbitrarily choose one" not in _MEMORY_SYSTEM_PROMPT


def test_computed_numbers_are_stated_plainly_not_as_a_lower_bound() -> None:
    assert ("state the number the retrieved figures give as your answer, not a lower bound"
            ) in _MEMORY_SYSTEM_PROMPT
    assert 'do not write "at least N", "over N", or "more than N"' in _MEMORY_SYSTEM_PROMPT
    assert "briefly note which figures are approximate or what is unknown" in _MEMORY_SYSTEM_PROMPT


def test_harness_recall_limit_defaults_to_20(monkeypatch) -> None:
    import inspect

    from archolith_bench import cli
    from archolith_bench.ci.stratified import run_stratified_slice

    seen = {}
    monkeypatch.setattr(cli, "_run_harness", lambda args: seen.update(vars(args)))
    cli.main(["harness", "longmemeval-menhir"])
    assert seen["recall_limit"] == 20
    assert inspect.signature(run_stratified_slice).parameters["recall_limit"].default == 20
    assert inspect.signature(run_memory_ab).parameters["recall_limit"].default == 20


def test_harness_source_memories_default_on_with_opt_out(monkeypatch) -> None:
    from archolith_bench import cli

    seen = {}
    monkeypatch.setattr(cli, "_run_harness", lambda args: seen.update(vars(args)))
    cli.main(["harness", "longmemeval-menhir"])
    assert seen["source_memories"] is True
    cli.main(["harness", "longmemeval-menhir", "--no-source-memories"])
    assert seen["source_memories"] is False


def test_checkpoint_is_keyed_by_the_answer_prompt(tmp_path) -> None:
    fp = LongMemEvalMemoryAdapter.prompt_fingerprint
    assert len(fp) == 10
    path = checkpoint_path_for(tmp_path, "longmemeval-menhir", "m", variant="oracle", prompt_tag=fp)
    assert path.name == f".checkpoint_longmemeval-menhir_oracle_m_p{fp}.jsonl"
    # Callers without a tag keep the old name.
    assert checkpoint_path_for(tmp_path, "b", "m", variant="s").name == ".checkpoint_b_s_m.jsonl"


def test_drill_down_runs_get_their_own_checkpoint(tmp_path) -> None:
    from archolith_bench.harness.checkpoint import checkpoint_prompt_tag

    adapter = LongMemEvalMemoryAdapter
    plain = checkpoint_prompt_tag(adapter, drill_down=False)
    drill = checkpoint_prompt_tag(adapter, drill_down=True)
    assert plain == adapter.prompt_fingerprint and drill == f"{plain}dd"
    paths = {checkpoint_path_for(tmp_path, "b", "m", variant="s", prompt_tag=tag) for tag in (plain, drill)}
    assert len(paths) == 2
    assert checkpoint_prompt_tag(object()) is None
    assert checkpoint_prompt_tag(object(), drill_down=True) == "dd"


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
