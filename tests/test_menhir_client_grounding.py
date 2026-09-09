"""A grounded user-tier claim must be byte-identical to the :TurnEvidence it cites.

Menhir's admission gate tests exact equality, so the ``"{role}: "`` decoration the bench applied to
every episode made each claim differ from its own evidence by six characters. Every user-tier claim
was denied and silently downgraded to ``agent_inference`` -- zero ADMITTED_ON edges across every run
to date. These pin both halves: grounded text is raw, ungrounded text keeps the speaker label.
"""

from typing import Any

import pytest

from archolith_bench.harness import HttpMenhirClient

TURN_TEXT = "I've been dedicating about an hour each day to coding exercises."


class _CapturingResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"episode_id": "ep-1"}


class _CapturingHttp:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def post(self, _url: str, **kwargs: Any) -> _CapturingResponse:
        self.payload = kwargs.get("json")
        return _CapturingResponse()


def _ingest(**kwargs: Any) -> dict[str, Any]:
    client = HttpMenhirClient("http://throwaway-menhir.local")
    http = _CapturingHttp()
    client._client = http  # type: ignore[assignment]
    client.ingest("ns", "user", TURN_TEXT, **kwargs)
    assert http.payload is not None
    return http.payload


def test_grounded_claim_matches_turn_evidence_text_exactly() -> None:
    """The whole bug: with a cited turn, the episode text carried a 'user: ' prefix the
    :TurnEvidence did not, so exact-equality grounding could never succeed."""
    payload = _ingest(source="user", turn_evidence_uuid="turn-1")

    assert payload["episode"] == TURN_TEXT
    assert not payload["episode"].startswith("user: ")
    assert payload["turn_evidence_uuid"] == "turn-1"


def test_ungrounded_turn_keeps_the_speaker_label() -> None:
    """Scoped fix: episodes that cite no evidence are unchanged, so extraction input on the
    existing benchmark arms stays comparable."""
    payload = _ingest(source="remote-api")

    assert payload["episode"] == f"user: {TURN_TEXT}"


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
def test_prefix_is_dropped_for_any_role_when_grounding(role: str) -> None:
    client = HttpMenhirClient("http://throwaway-menhir.local")
    http = _CapturingHttp()
    client._client = http  # type: ignore[assignment]
    client.ingest("ns", role, TURN_TEXT, turn_evidence_uuid="turn-1")

    assert http.payload is not None
    assert http.payload["episode"] == TURN_TEXT
