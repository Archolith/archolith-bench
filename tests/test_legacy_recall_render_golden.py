"""The legacy context renderer is byte-identical to HttpMenhirClient.recall before the extraction.

`fixtures/legacy_recall_render_golden.json` was recorded from origin/master 55d0c23 by running the whole
test suite with HttpMenhirClient.recall wrapped (every canned response the renderer tests send, with
the client's settings and its exact output), plus real event-canary and F5 recall responses under
every drill_down / counted_authority_only / source_memories combination.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archolith_bench.harness.menhir_client import (
    HttpMenhirClient,
    effective_counted_verdict_limit,
    render_recall_snippets,
)

_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_recall_render_golden.json").read_text(encoding="utf-8")
)
_CASES = _GOLDEN["cases"]
_RESPONSES = _GOLDEN["responses"]


def _flags(case: dict) -> dict:
    return {
        "drill_down": case["drill_down"],
        "counted_authority_only": case["counted_authority_only"],
        "source_memories": case["source_memories"],
    }


def _assert_same(out, case: dict) -> None:
    assert list(out) == case["out"]
    assert "\n".join(out) == "\n".join(case["out"])
    assert type(out).__name__ == case["out_type"]
    assert list(getattr(out, "extras", ())) == case["extras"]
    assert list(getattr(out, "counted", ())) == case["counted"]
    assert list(getattr(out, "sources", ())) == case["sources"]


def test_corpus_covers_every_renderer_branch() -> None:
    def has(pred):
        return any(pred(_RESPONSES[c["response"]], c) for c in _CASES)

    def d(r):
        return r if isinstance(r, dict) else {}

    assert len(_CASES) >= 200
    assert has(lambda r, c: c["out_type"] == "list" and d(r).get("event_authority_layer"))  # event-blocked
    assert has(lambda r, c: d(r).get("authority_layer"))
    assert has(lambda r, c: d(r).get("query_hint"))
    assert has(lambda r, c: isinstance(r, list))
    assert has(lambda r, c: "memories" in d(r)) and has(lambda r, c: "snippets" in d(r))
    assert has(lambda r, c: c["extras"]) and has(lambda r, c: c["counted"]) and has(lambda r, c: c["sources"])
    assert has(lambda r, c: c["drill_down"]) and has(lambda r, c: c["counted_verdict_limit"] is not None)


@pytest.mark.parametrize("case", _CASES, ids=[f"{c['response']}-{i}" for i, c in enumerate(_CASES)])
def test_pure_renderer_matches_recorded_output(case: dict) -> None:
    out = render_recall_snippets(
        json.loads(json.dumps(_RESPONSES[case["response"]])),
        case["limit"],
        counted_verdict_limit=case["counted_verdict_limit"],
        **_flags(case),
    )
    _assert_same(out, case)


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return json.loads(json.dumps(self._data))


class _Http:
    def __init__(self, data):
        self.data = data
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None):  # noqa: A002
        self.payloads.append(json)
        return _Resp(self.data)


@pytest.mark.parametrize("case", _CASES[::7], ids=lambda c: c["response"])
def test_client_still_renders_and_requests_as_before(case: dict) -> None:
    client = HttpMenhirClient("http://menhir.test", **_flags(case))
    client._client = _Http(_RESPONSES[case["response"]])
    out = client.recall("ns", "q", limit=case["limit"], counted_verdict_limit=case["counted_verdict_limit"])
    _assert_same(out, case)
    (payload,) = client._client.payloads
    if case["counted_authority_only"]:
        assert payload["counted_verdict_limit"] == effective_counted_verdict_limit(case["counted_verdict_limit"])
    else:
        assert "counted_verdict_limit" not in payload
