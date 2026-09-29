"""Tests for the bench-only typed-value graph and recall arm."""

from __future__ import annotations

from archolith_bench.harness.memory_ab import VALUE_RECALL, run_memory_ab
from archolith_bench.harness.value_nodes import ValueKind, ValueGraph


def _sessions(*turns: str) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": turn} for turn in turns]]


def test_extracts_and_normalizes_core_value_kinds() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            "I earned $420 at the market and usually go to the gym at 6:00 pm.",
            "I have used my Fitbit for 9 months and attend yoga three times a week.",
            "My to-watch list now has 25 titles, and the class is on Wednesday.",
        ),
    )

    observed = {(value.kind, str(value.normalized)) for value in graph.values.values()}
    assert (ValueKind.MONEY, "420") in observed
    assert (ValueKind.CLOCK_TIME, "18:00") in observed
    assert (ValueKind.DURATION, "9") in observed
    assert (ValueKind.FREQUENCY, "3/week") in observed
    assert (ValueKind.COUNT, "25") in observed
    assert (ValueKind.WEEKDAY, "wednesday") in observed


def test_equal_values_in_unrelated_assertions_are_distinct_nodes() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            "I have used my Fitbit for 9 months.",
            "My parents have stayed with me for 9 months.",
            "I have had my cat Luna for 9 months.",
        ),
    )

    nine_month_nodes = [
        value
        for value in graph.values.values()
        if value.kind == ValueKind.DURATION and value.normalized == 9
    ]
    assert len(nine_month_nodes) == 3
    assert len({value.node_id for value in nine_month_nodes}) == 3


def test_recall_uses_question_subject_and_value_kind() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            "I have tried four Korean restaurants in my city.",
            "My to-watch list now has 25 titles.",
            "I have watched 15 Crash Course videos recently.",
        ),
    )

    recalled = graph.recall("How many titles are currently on my to-watch list?", limit=2)

    assert recalled
    assert "25 titles" in recalled[0]
    assert "four Korean restaurants" not in recalled[0]


def test_word_counts_and_ranges_are_typed() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            "I have tried four Korean restaurants.",
            "I have tried four different ones.",
            "I spent 10-12 hours on the sculpture.",
            "The hotel award covered two free nights.",
            "I am now on page 220.",
        ),
    )

    values = [(value.kind, value.normalized, value.unit) for value in graph.values.values()]
    assert (ValueKind.COUNT, 4, "restaurants") in values
    assert (ValueKind.COUNT, 4, "ones") in values
    assert (ValueKind.DURATION, [10, 12], "hours") in values
    assert (ValueKind.COUNT, 2, "nights") in values
    assert (ValueKind.COUNT, 220, "page") in values


def test_explicit_status_and_boolean_values_are_typed_and_recalled() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            'I put down "The Nightingale" temporarily.',
            'I recently finished "The Nightingale" by Kristin Hannah.',
            "I actually have a spare screwdriver.",
            "I do not have a spare battery.",
        ),
    )

    observed = {(value.kind, value.normalized) for value in graph.values.values()}
    assert (ValueKind.STATUS, "paused") in observed
    assert (ValueKind.STATUS, "finished") in observed
    assert (ValueKind.BOOLEAN, True) in observed
    assert (ValueKind.BOOLEAN, False) in observed
    assert "recently finished" in graph.recall('Did I finish "The Nightingale"?', limit=1)[0]
    assert "have a spare screwdriver" in graph.recall("Do I have a spare screwdriver?", limit=1)[0]


def test_boolean_values_require_explicit_polarity() -> None:
    graph = ValueGraph.from_sessions(
        "test",
        _sessions(
            "I misplaced the small screwdriver and need to pick one up.",
            "Do I have a spare screwdriver for opening up my laptop?",
        ),
    )

    assert all(value.kind != ValueKind.BOOLEAN for value in graph.values.values())


def test_item_dates_determine_recency_when_sessions_are_shuffled() -> None:
    sessions = [
        [{"role": "user", "content": "My to-watch list now has 25 titles."}],
        [{"role": "user", "content": "My to-watch list has 20 titles."}],
    ]
    item = {
        "haystack_dates": [
            "2025/02/01 (Sat) 12:00",
            "2025/01/01 (Wed) 12:00",
        ]
    }

    graph = ValueGraph.from_item("test", item, sessions)
    recalled = graph.recall("How many titles are currently on my to-watch list?", limit=2)

    assert "25 titles" in recalled[0]


class _ValueAdapter:
    benchmark_id = "value-test"
    name = "Value test"

    def load_items(self, subset, limit, fixture_path):  # noqa: ANN001
        return [
            {
                "question_id": "value-1",
                "answer": "25",
                "question_type": "knowledge-update",
                "question": "How many titles are currently on my to-watch list?",
                "sessions": _sessions("My to-watch list now has 25 titles."),
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


class _ValueBlindClient:
    def new_group(self) -> str:
        return "group"

    def record_turn_evidence(self, namespace: str, text: str, **kwargs) -> dict:
        return {"turn_id": "te-1"}

    def ingest(self, group_id: str, role: str, content: str, **kwargs) -> None:
        return None

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        return ["The user maintains a to-watch list, but its size was not extracted."]

    def reset(self, group_id: str) -> None:
        return None


def test_value_recall_arm_recovers_value_missing_from_base_recall() -> None:
    def send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
        context = messages[-1]["content"]
        answer = "25" if "25 titles" in context else "I don't know"
        return answer, 1.0, {"prompt_tokens": 10, "completion_tokens": 2}

    result = run_memory_ab(
        _ValueAdapter(),
        arms=("menhir_recall", VALUE_RECALL),
        client=_ValueBlindClient(),
        send_fn=send_fn,
        recall_limit=4,
    )

    assert result.arms["menhir_recall"].score == 0.0
    assert result.arms[VALUE_RECALL].score == 1.0
    assert "[typed-value count current]" in result.arms[VALUE_RECALL].results[0].recalled

