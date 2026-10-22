"""Tests for the read-only Neo4j HTTP client. All traffic goes through httpx.MockTransport."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nh = _load_script("lme_neo4j_http_test", "scripts/longmemeval/lib/neo4j_http.py")

PASSWORD = "s3cr3t-Zx9-distinctive"
KEYS = ("pending", "ready", "enriching", "failed", "llm_tasks", "processing_attempts", "total")


def _reader(handler, base="http://host:7474", password=PASSWORD):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return nh.Neo4jHttpReader(base, "neo4j", password, client=client)


def _ok(rows):
    return httpx.Response(
        200, json={"results": [{"columns": [], "data": [{"row": r} for r in rows]}], "errors": []}
    )


def test_request_shape_and_auth():
    seen = []

    def handler(request):
        seen.append(request)
        return _ok([[1]])

    reader = _reader(handler, base="http://host:7474/")
    assert reader.rows("RETURN $x", {"x": 1}) == [[1]]
    (req,) = seen
    assert req.method == "POST"
    assert str(req.url) == "http://host:7474/db/neo4j/tx/commit"
    expected = base64.b64encode(f"neo4j:{PASSWORD}".encode()).decode()
    assert req.headers["authorization"] == f"Basic {expected}"
    assert json.loads(req.content) == {
        "statements": [{"statement": "RETURN $x", "parameters": {"x": 1}}]
    }


def test_rows_default_parameters_empty_dict():
    seen = []
    reader = _reader(lambda r: (seen.append(json.loads(r.content)), _ok([]))[1])
    assert reader.rows("RETURN 1") == []
    assert seen[0]["statements"][0]["parameters"] == {}


def _raise_transport(request):
    raise httpx.ConnectError("boom", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        _raise_transport,
        lambda r: httpx.Response(500, json={"results": [], "errors": []}),
        lambda r: httpx.Response(200, content=b"<html>not json"),
        lambda r: httpx.Response(200, json=["x"]),
        lambda r: httpx.Response(
            200, json={"results": [], "errors": [{"code": "Neo.ClientError", "message": "bad"}]}
        ),
        lambda r: httpx.Response(
            200,
            json={"results": [{"data": [{"row": [1]}]}], "errors": [{"code": "X", "message": "m"}]},
        ),
        lambda r: httpx.Response(200, json={"errors": []}),
        lambda r: httpx.Response(200, json={"results": [], "errors": []}),
        lambda r: httpx.Response(200, json={"results": [{"data": "nope"}], "errors": []}),
        lambda r: httpx.Response(200, json={"results": [{}], "errors": []}),
        lambda r: httpx.Response(200, json={"results": [{"data": [{"row": "x"}]}], "errors": []}),
        lambda r: httpx.Response(200, json={"results": [{"data": [{"meta": []}]}], "errors": []}),
        lambda r: httpx.Response(200, json={"results": [{"data": [5]}], "errors": []}),
    ],
)
def test_rows_failures_raise_never_empty(handler):
    reader = _reader(handler)
    with pytest.raises(nh.Neo4jUnavailable):
        reader.rows("RETURN 1")


def test_namespace_counts_exact_shape():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _ok([["a", 1, 2, 3, 4, 50, 60, 10], ["b", 0, 0, 0, 0, 0, 0, 0]])

    out = _reader(handler).namespace_state_counts(["a", "b"])
    assert out == {
        "a": dict(zip(KEYS, [1, 2, 3, 4, 50, 60, 10])),
        "b": {k: 0 for k in KEYS},
    }
    stmt = seen[0]["statements"][0]
    assert "$namespaces" in stmt["statement"]
    assert stmt["parameters"] == {"namespaces": ["a", "b"]}


def test_namespace_counts_missing_row_is_unreadable_not_zero():
    # The query returns a row per requested namespace; a missing one is a bad answer.
    reader = _reader(lambda r: _ok([["a", 1, 2, 3, 4, 50, 60, 10]]))
    with pytest.raises(nh.Neo4jUnavailable):
        reader.namespace_state_counts(["a", "b"])


def test_namespace_counts_none_values_become_zero():
    out = _reader(lambda r: _ok([["a", None, 1, None, 0, None, None, 0]])).namespace_state_counts(["a"])
    assert out["a"]["pending"] == 0 and out["a"]["ready"] == 1


def test_namespace_counts_empty_input_makes_no_request():
    calls = []
    reader = _reader(lambda r: (calls.append(r), _ok([]))[1])
    assert reader.namespace_state_counts([]) == {}
    assert calls == []


def test_namespace_counts_unrequested_namespace_raises():
    reader = _reader(lambda r: _ok([["zzz", 0, 0, 0, 0, 0, 0, 0]]))
    with pytest.raises(nh.Neo4jUnavailable):
        reader.namespace_state_counts(["a"])


def test_namespace_counts_non_integer_raises():
    reader = _reader(lambda r: _ok([["a", "many", 0, 0, 0, 0, 0, 0]]))
    with pytest.raises(nh.Neo4jUnavailable):
        reader.namespace_state_counts(["a"])


def test_namespace_counts_wrong_width_row_raises():
    reader = _reader(lambda r: _ok([["a", 1, 2]]))
    with pytest.raises(nh.Neo4jUnavailable):
        reader.namespace_state_counts(["a"])


def test_namespace_counts_propagates_unavailable():
    with pytest.raises(nh.Neo4jUnavailable):
        _reader(lambda r: httpx.Response(503)).namespace_state_counts(["a"])


@pytest.mark.parametrize(
    "handler",
    [
        _raise_transport,
        lambda r: httpx.Response(401, text="unauthorized"),
        lambda r: httpx.Response(200, content=b"garbage"),
        lambda r: httpx.Response(200, json={"results": [], "errors": [{"message": "x"}]}),
    ],
)
def test_password_never_in_exception_message(handler):
    reader = _reader(handler)
    with pytest.raises(nh.Neo4jUnavailable) as exc:
        reader.rows("RETURN 1")
    assert PASSWORD not in str(exc.value)
    assert PASSWORD not in repr(exc.value)
    base64_pw = base64.b64encode(f"neo4j:{PASSWORD}".encode()).decode()
    assert base64_pw not in str(exc.value)


def test_from_env_unset_or_blank_returns_none(monkeypatch):
    monkeypatch.delenv("LME_NEO4J_HTTP_URL", raising=False)
    assert nh.Neo4jHttpReader.from_env() is None
    monkeypatch.setenv("LME_NEO4J_HTTP_URL", "   ")
    assert nh.Neo4jHttpReader.from_env() is None


def test_from_env_set_builds_reader_with_url_and_password(monkeypatch):
    monkeypatch.setenv("LME_NEO4J_HTTP_URL", "http://envhost:9999/")
    monkeypatch.setenv("LME_NEO4J_PW", "env-pw-123")
    reader = nh.Neo4jHttpReader.from_env()
    assert reader is not None
    seen = []

    def handler(request):
        seen.append(request)
        return _ok([[1]])

    reader._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert reader.rows("RETURN 1") == [[1]]
    (req,) = seen
    assert str(req.url) == "http://envhost:9999/db/neo4j/tx/commit"
    expected = base64.b64encode(b"neo4j:env-pw-123").decode()
    assert req.headers["authorization"] == f"Basic {expected}"
