"""memory_ab records :TurnEvidence for every user turn before ingest (default on).

Menhir's scalar lane reads user input ONLY from :TurnEvidence (the legacy
"user:"-prefix fallback is removed), so the ingest loop must capture evidence
for user-role turns and cite its UUID; assistant turns stay ungrounded.
"""

from __future__ import annotations

from archolith_bench.harness.memory_ab import run_memory_ab


class FakeClient:
    """Duck-typed MenhirClient that records evidence/ingest calls."""

    def __init__(self) -> None:
        self.evidence: list[tuple[str, str, dict]] = []
        self.ingests: list[tuple[str, str, str, dict]] = []

    def new_group(self) -> str:
        return "ns-1"

    def record_turn_evidence(self, namespace: str, text: str, **kwargs) -> dict:
        self.evidence.append((namespace, text, kwargs))
        return {"turn_id": f"turn-{len(self.evidence)}", "created": True}

    def ingest(self, group_id: str, role: str, content: str, **kwargs) -> None:
        self.ingests.append((group_id, role, content, kwargs))

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        return []

    def reset(self, group_id: str) -> None:
        pass


class TinyAdapter:
    """Minimal MemoryQAAdapter: one item, one user turn + one assistant turn."""

    benchmark_id = "tiny"
    name = "tiny"
    sessions_data = [
        [
            {"role": "user", "content": "I have 20 coins"},
            {"role": "assistant", "content": "noted"},
        ]
    ]

    def load_items(self, subset, limit, fixture_path) -> list[dict]:  # noqa: ANN001
        return [{"question_id": "q1"}]

    def sessions(self, item: dict) -> list[list[dict]]:
        return self.sessions_data

    def question(self, item: dict) -> str:
        return "how many coins?"

    def build_messages(self, memory_context: str, question: str) -> list[dict]:
        return [{"role": "user", "content": memory_context + question}]

    def score(self, item: dict, response_text: str) -> bool:
        return True


def _run(client) -> None:
    run_memory_ab(
        TinyAdapter(),
        arms=("menhir_recall",),
        client=client,
        send_fn=lambda chat_client, base_url, api_key, messages, model, **kw: (
            "ok", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}
        ),
        reset_memory=False,
    )


def test_turns_with_a_shared_prefix_get_distinct_evidence_keys() -> None:
    # Keys must identify the turn, not its opening words, or two turns collapse into one record.
    class SharedPrefixAdapter(TinyAdapter):
        sessions_data = [
            [
                {"role": "user", "content": "Can you help me with my 20 coins"},
                {"role": "user", "content": "Can you help me with my 30 stamps"},
            ]
        ]

    client = FakeClient()
    run_memory_ab(
        SharedPrefixAdapter(),
        arms=("menhir_recall",),
        client=client,
        send_fn=lambda chat_client, base_url, api_key, messages, model, **kw: (
            "ok", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}
        ),
        reset_memory=False,
    )

    keys = [kwargs["turn_key"] for _, _, kwargs in client.evidence]
    assert len(keys) == 2 and len(set(keys)) == 2


def test_user_turns_record_evidence_then_ingest_with_uuid() -> None:
    client = FakeClient()
    _run(client)

    assert len(client.evidence) == 1
    namespace, text, kwargs = client.evidence[0]
    assert namespace == "ns-1"
    assert text == "I have 20 coins"
    assert kwargs.get("turn_key")

    assert len(client.ingests) == 2
    user_group, user_role, user_content, user_kwargs = client.ingests[0]
    assert (user_group, user_role, user_content) == ("ns-1", "user", "I have 20 coins")
    assert user_kwargs.get("turn_evidence_uuid") == "turn-1"

    _, assistant_role, _, assistant_kwargs = client.ingests[1]
    assert assistant_role == "assistant"
    assert assistant_kwargs.get("turn_evidence_uuid") is None


def test_option_off_records_no_evidence() -> None:
    client = FakeClient()
    run_memory_ab(
        TinyAdapter(),
        arms=("menhir_recall",),
        client=client,
        send_fn=lambda chat_client, base_url, api_key, messages, model, **kw: (
            "ok", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}
        ),
        reset_memory=False,
        record_turn_evidence=False,
    )

    assert client.evidence == []
    assert len(client.ingests) == 2
    assert all(kwargs.get("turn_evidence_uuid") is None for _, _, _, kwargs in client.ingests)


def test_empty_user_content_records_no_evidence() -> None:
    client = FakeClient()
    client_adapter = TinyAdapter()
    client_adapter.sessions_data = [[{"role": "user", "content": ""}]]
    _run_with = run_memory_ab(
        client_adapter,
        arms=("menhir_recall",),
        client=client,
        send_fn=lambda chat_client, base_url, api_key, messages, model, **kw: (
            "ok", 1.0, {"prompt_tokens": 1, "completion_tokens": 1}
        ),
        reset_memory=False,
    )
    assert _run_with is not None
    assert client.evidence == []
    # Empty content is still passed to ingest (the client no-ops on it), but
    # ungrounded and without a prior evidence call.
    assert len(client.ingests) == 1
    assert client.ingests[0][3].get("turn_evidence_uuid") is None
