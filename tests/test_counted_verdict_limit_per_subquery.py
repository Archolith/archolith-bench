"""Per-call counted_verdict_limit: the client shows every counted verdict Menhir returns.

Menhir drops the occurrence evidence of each counted slot whose verdict it returns, so a verdict
returned but not shown loses both. ``_agentic_recall`` therefore tells each subquery how many
counted verdicts can still be shown.
"""

from __future__ import annotations

import pytest

from archolith_bench.harness.memory_ab import _agentic_recall
from archolith_bench.harness.menhir_client import (
    COUNTED_AUTHORITY_MAX,
    HttpMenhirClient,
    RecallSnippets,
)
from tests.test_counted_authority_only import _CapturingHttp, _memories, _verdict


def _recall_with(limit, *, on: bool, n_verdicts: int = 3):  # noqa: ANN001, ANN202
    client = HttpMenhirClient("http://localhost:9999", counted_authority_only=on)
    verdicts = [_verdict(f"c{i}_count", i, basis="counted", view=f"sv-{i}") for i in range(n_verdicts)]
    http = _CapturingHttp({"results": _memories(1), "authority_layer": verdicts})
    client._client = http
    out = client.recall("ns", "q", limit=3, counted_verdict_limit=limit)
    return http.posted, out


@pytest.mark.parametrize(
    ("given", "sent"),
    [(None, COUNTED_AUTHORITY_MAX), (1, 1), (0, 0), (5, COUNTED_AUTHORITY_MAX), (-2, 0)],
)
def test_effective_limit_is_sent_and_clamped(given, sent) -> None:  # noqa: ANN001
    payload, out = _recall_with(given, on=True)
    assert payload["counted_verdict_limit"] == sent
    assert len(out.counted) == sent


def test_limit_one_renders_at_most_one_verdict_even_if_response_has_three() -> None:
    _, out = _recall_with(1, on=True)
    assert len(out.counted) == 1 and "c0 count" in out.counted[0]


@pytest.mark.parametrize("given", [None, 0, 1, 5])
def test_off_ignores_argument_and_sends_nothing(given) -> None:  # noqa: ANN001
    payload, out = _recall_with(given, on=False)
    assert "counted_verdict_limit" not in payload
    assert out.counted == ()


class _MenhirLike:
    """Emulates Menhir: at most ``counted_verdict_limit`` verdicts per call, in relevance order."""

    counted_authority_only = True

    def __init__(self, plan: dict[str, list[str]]) -> None:
        self.plan = plan
        self.limits: list = []
        self.returned: list[str] = []

    def recall(self, group_id, query, limit=10, *, counted_verdict_limit=None):  # noqa: ANN001, ANN201
        self.limits.append(counted_verdict_limit)
        cap = COUNTED_AUTHORITY_MAX if counted_verdict_limit is None else counted_verdict_limit
        verdicts = [f"[STRUCTURED MEMORY VERDICT] {name}" for name in self.plan[query][:cap]]
        self.returned.extend(verdicts)
        plain = [f"Plain {query}."]
        return RecallSnippets([*plain, *verdicts], verdicts, verdicts)


def _lines(text: str) -> list[str]:
    return text.split("\n")


def test_agentic_recall_shows_every_verdict_menhir_returns() -> None:
    client = _MenhirLike({"a": ["A"], "b": ["B"], "c": ["C", "D", "E"]})
    merged = _lines(_agentic_recall(client, "ns", ["a", "b", "c"], 6))
    assert client.limits == [3, 2, 1]
    assert set(client.returned) <= set(merged)
    assert sum(line.startswith("[STRUCTURED MEMORY VERDICT]") for line in merged) == 3


def test_duplicate_verdict_is_not_double_counted() -> None:
    client = _MenhirLike({"a": ["A"], "b": ["A", "B"], "c": ["C", "D", "E"]})
    merged = _lines(_agentic_recall(client, "ns", ["a", "b", "c"], 6))
    # Subquery 2 re-returns A: only B is new, so subquery 3 may still show one verdict.
    assert client.limits == [3, 2, 1]
    assert set(client.returned) <= set(merged)
    verdicts = [line for line in merged if line.startswith("[STRUCTURED MEMORY VERDICT]")]
    assert len(verdicts) == len(set(verdicts)) == 3


def test_limit_floors_at_zero_once_cap_is_reached() -> None:
    client = _MenhirLike({"a": ["A", "B", "C"], "b": ["D"]})
    merged = _lines(_agentic_recall(client, "ns", ["a", "b"], 6))
    assert client.limits == [3, 0]
    assert set(client.returned) <= set(merged)
