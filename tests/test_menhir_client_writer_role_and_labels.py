"""memory_role is forwarded as the payload role; recall renders provenance.said_by labels."""

from __future__ import annotations

from typing import Any

import pytest

from archolith_bench.harness import HttpMenhirClient

TURN_TEXT = "I recommended the blue route."

AUTHORITY = {
    "kind": "current",
    "status": "leads",
    "subject": "user",
    "attribute": "colour",
    "value": "blue",
}


class _Response:
    def __init__(self, body: Any) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._body


class _Http:
    def __init__(self, body: Any = None) -> None:
        self.payload: dict[str, Any] | None = None
        self.body = body if body is not None else {"episode_id": "ep-1"}

    def post(self, _url: str, **kwargs: Any) -> _Response:
        self.payload = kwargs.get("json")
        return _Response(self.body)


def _client(body: Any = None) -> tuple[HttpMenhirClient, _Http]:
    client = HttpMenhirClient("http://throwaway-menhir.local")
    http = _Http(body)
    client._client = http  # type: ignore[assignment]
    return client, http


def test_memory_role_sets_payload_role_and_keeps_grounded_text_unprefixed() -> None:
    client, http = _client()
    client.ingest("ns", "assistant", TURN_TEXT, turn_evidence_uuid="turn-1", memory_role="agent")

    assert http.payload is not None
    assert http.payload["role"] == "agent"
    assert http.payload["episode"] == TURN_TEXT


def test_memory_role_none_sends_no_role_key() -> None:
    client, http = _client()
    client.ingest("ns", "assistant", TURN_TEXT)

    assert http.payload is not None
    assert "role" not in http.payload
    assert http.payload["episode"] == f"assistant: {TURN_TEXT}"


def _recall(item: dict[str, Any], *, authority: bool) -> list[str]:
    body: dict[str, Any] = {"results": [item]}
    if authority:
        body["authority_layer"] = [AUTHORITY]
    client, _ = _client(body)
    return client.recall("ns", "q")


def _item_snippet(item: dict[str, Any], *, authority: bool) -> str:
    snippets = _recall(item, authority=authority)
    return snippets[-1]


@pytest.mark.parametrize(
    ("said_by", "label"),
    [("agent", "agent-recorded"), ("mixed", "user + agent"), ("unknown", "provenance unknown")],
)
@pytest.mark.parametrize("authority", [False, True])
def test_said_by_maps_to_label_with_name(said_by: str, label: str, authority: bool) -> None:
    item = {"name": "Route", "content": "Take the blue route.", "provenance": {"said_by": said_by}}
    snippet = _item_snippet(item, authority=authority)

    assert f"Route ({label}): Take the blue route." in snippet
    if authority:
        assert snippet.startswith("[RELATED memory MEMORY | non-authoritative] ")


@pytest.mark.parametrize(
    ("said_by", "label"),
    [("agent", "agent-recorded"), ("mixed", "user + agent"), ("unknown", "provenance unknown")],
)
@pytest.mark.parametrize("authority", [False, True])
def test_said_by_without_name_prefixes_label(said_by: str, label: str, authority: bool) -> None:
    item = {"content": "Take the blue route.", "provenance": {"said_by": said_by}}
    snippet = _item_snippet(item, authority=authority)

    assert f"({label}) Take the blue route." in snippet


@pytest.mark.parametrize(
    "provenance",
    [None, {}, {"said_by": "user"}, {"said_by": 5}, {"said_by": None}, "agent", ["agent"], 7],
)
@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("authority", [False, True])
def test_missing_or_unrecognised_provenance_is_byte_identical(
    provenance: Any, named: bool, authority: bool
) -> None:
    base: dict[str, Any] = {"content": "Take the blue route."}
    if named:
        base["name"] = "Route"
    with_key = {**base, "provenance": provenance}

    assert _recall(with_key, authority=authority) == _recall(base, authority=authority)


def test_string_items_are_untouched() -> None:
    client, _ = _client({"results": ["plain memory"]})
    assert client.recall("ns", "q") == ["plain memory"]
