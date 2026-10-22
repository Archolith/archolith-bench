"""Live parity tests: the HTTP read path in ingest.py vs the legacy docker-exec cypher-shell path.

Requires a real Neo4j. Skipped unless both are set:
  LME_TEST_NEO4J_HTTP_URL   e.g. http://localhost:17998
  LME_TEST_NEO4J_CONTAINER  docker container name (reachable via the docker CLI's current context)
Password comes from LME_TEST_NEO4J_PW (default "testpassword").
On Windows with an ssh:// docker context, put Windows OpenSSH first on PATH: Git Bash's bundled
ssh intermittently fails to resolve the host, which makes the docker-exec baseline return -1.

All data is created under uuid-suffixed namespaces and deleted by those namespaces only.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path

import pytest

HTTP_URL = os.getenv("LME_TEST_NEO4J_HTTP_URL", "").strip()
CONTAINER = os.getenv("LME_TEST_NEO4J_CONTAINER", "").strip()
PW = os.getenv("LME_TEST_NEO4J_PW", "testpassword")

if not (HTTP_URL and CONTAINER):
    pytest.skip(
        "set LME_TEST_NEO4J_HTTP_URL and LME_TEST_NEO4J_CONTAINER to run live neo4j parity tests",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_http_live_test", "scripts/longmemeval/lib/ingest.py")
from neo4j_http import Neo4jHttpReader  # noqa: E402  (ingest.py puts its lib dir on sys.path)

EPOCH_MS = 1704067200000  # 2024-01-01T00:00:00Z
KEYS = ("pending", "ready", "enriching", "failed", "llm_tasks", "processing_attempts", "total")


def _reader() -> Neo4jHttpReader:
    return Neo4jHttpReader(HTTP_URL, "neo4j", PW, timeout_s=30.0)


@pytest.fixture(scope="module")
def data():
    tag = uuid.uuid4().hex[:10]
    a, b, c = (f"lmehttp-{tag}-{s}" for s in "abc")
    proj_uuid = f"{tag}-proj-failed"
    plain_failed_uuid = f"{tag}-plain-failed"
    graphiti_uuid = f"{tag}-graphiti"
    reader = _reader()
    try:
        # ns a: PENDING(tasks 2, attempts 1), READY(tasks 5, attempts "3" string),
        # ENRICHING, FAILED projection (attempts 2), FAILED plain.
        reader.rows(
            "CREATE (:Episodic {namespace:$a, uuid:$u1, processing_state:'PENDING',"
            " processing_llm_tasks_total:2, processing_attempts:1})"
            " CREATE (:Episodic {namespace:$a, uuid:$u2, processing_state:'READY',"
            " processing_llm_tasks_total:5, processing_attempts:'3',"
            " resolved_episode_uuid:$g, reference_time:datetime('2024-01-01T00:00:00Z')})"
            " CREATE (:Episodic {namespace:$a, uuid:$u3, processing_state:'ENRICHING'})"
            " CREATE (:Episodic {namespace:$a, uuid:$pu, processing_state:'FAILED',"
            " is_evidence_projection:true, processing_attempts:2})"
            " CREATE (:Episodic {namespace:$a, uuid:$fu, processing_state:'FAILED'})"
            # Graphiti-style: group_id only, NULL processing_state, dated.
            " CREATE (:Episodic {group_id:$a, uuid:$g, valid_at:datetime('2024-01-01T00:00:00Z')})"
            " RETURN 1",
            {"a": a, "u1": f"{tag}-u1", "u2": f"{tag}-u2", "u3": f"{tag}-u3",
             "pu": proj_uuid, "fu": plain_failed_uuid, "g": graphiti_uuid},
        )
        # ns a: evidence + assertion + scalar view.
        reader.rows(
            "CREATE (t:TurnEvidence {namespace:$a, declarant:'user'})"
            " CREATE (x:TypedAssertion {namespace:$a})"
            " CREATE (v:Entity {group_id:$a, view_kind:'scalar_state'})"
            " CREATE (t)-[:FOUNDS]->(x) CREATE (v)-[:CURRENT_ANCHOR]->(x) RETURN 1",
            {"a": a},
        )
        # ns b: two READY, one NULL-state node carrying namespace.
        reader.rows(
            "CREATE (:Episodic {namespace:$b, processing_state:'READY', processing_llm_tasks_total:1})"
            " CREATE (:Episodic {namespace:$b, processing_state:'READY', processing_llm_tasks_total:4})"
            " CREATE (:Episodic {namespace:$b}) RETURN 1",
            {"b": b},
        )
        yield {"a": a, "b": b, "c": c, "proj_uuid": proj_uuid}
    finally:
        reader.rows(
            "MATCH (n) WHERE n.namespace IN $ns OR n.group_id IN $ns DETACH DELETE n",
            {"ns": [a, b, c]},
        )


@pytest.fixture
def http_path(monkeypatch):
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", _reader())


@pytest.fixture
def docker_path(monkeypatch):
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", None)
    monkeypatch.setattr(ingest, "NEO4J_CONTAINER", CONTAINER)
    monkeypatch.setattr(ingest, "NEO4J_PW", PW)


def _both(monkeypatch, fn):
    """Run fn() via HTTP, then via docker; return (http_result, docker_result)."""
    monkeypatch.setattr(ingest, "NEO4J_CONTAINER", CONTAINER)
    monkeypatch.setattr(ingest, "NEO4J_PW", PW)
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", _reader())
    via_http = fn()
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", None)
    via_docker = fn()
    return via_http, via_docker


def test_ns_state_counts_identical_and_correct(data, monkeypatch):
    expected = {
        "a": dict(zip(KEYS, (1, 1, 1, 2, 7, 6, 5))),
        "b": dict(zip(KEYS, (0, 2, 0, 0, 5, 0, 3))),
        "c": dict(zip(KEYS, (0, 0, 0, 0, 0, 0, 0))),
    }
    for label, want in expected.items():
        via_http, via_docker = _both(monkeypatch, lambda: ingest._ns_state_counts(data[label]))
        assert via_http == via_docker == want, label


def test_ns_state_counts_many_http_equals_per_ns_docker(data, monkeypatch):
    names = [data["a"], data["b"], data["c"]]
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", _reader())
    batched = ingest._NEO4J_HTTP.namespace_state_counts(names)
    assert ingest._ns_state_counts_many(names) == batched
    monkeypatch.setattr(ingest, "_NEO4J_HTTP", None)
    monkeypatch.setattr(ingest, "NEO4J_CONTAINER", CONTAINER)
    monkeypatch.setattr(ingest, "NEO4J_PW", PW)
    legacy = {ns: ingest._ns_state_counts(ns) for ns in names}
    assert batched == legacy
    assert all(v >= 0 for counts in legacy.values() for v in counts.values())


def test_scalar_counts_identical(data, monkeypatch):
    via_http, via_docker = _both(monkeypatch, lambda: ingest._scalar_counts(data["a"]))
    assert via_http == via_docker
    assert via_docker == {
        "turn_evidence": 1,
        "typed_assertions": 1,
        "scalar_views": 1,
        "scalar_history_views": 0,
        "user_founded_scalar_views": 1,
    }
    empty_http, empty_docker = _both(monkeypatch, lambda: ingest._scalar_counts(data["c"]))
    assert empty_http == empty_docker
    assert all(v == 0 for v in empty_docker.values())


@pytest.mark.parametrize("which,millis", [("a", [EPOCH_MS]), ("a", []), ("b", [EPOCH_MS]),
                                          ("c", [EPOCH_MS])])
def test_integrity_counts_identical(data, monkeypatch, which, millis):
    via_http, via_docker = _both(
        monkeypatch, lambda: ingest._integrity_counts(data[which], millis)
    )
    assert via_http == via_docker


def test_integrity_counts_a_values(data, monkeypatch):
    # Hand-checked: admitted turns 0, assertion founded, graphiti episode dated in the submitted
    # set, READY node paired to it with equal reference_time, no cross-namespace links.
    via_http, via_docker = _both(
        monkeypatch, lambda: ingest._integrity_counts(data["a"], [EPOCH_MS])
    )
    assert via_http == via_docker
    assert via_docker["assertions_unfounded"] == 0
    assert via_docker["episodes_valid_at_unsubmitted"] == 0
    assert via_docker["ready_episodes_time_unpaired"] == 0
    assert via_docker["cross_namespace_links"] == 0


def test_failed_evidence_projection_uuids_identical(data, monkeypatch):
    via_http, via_docker = _both(
        monkeypatch, lambda: ingest._failed_evidence_projection_uuids(data["a"])
    )
    assert via_http == via_docker == [data["proj_uuid"]]
    none_http, none_docker = _both(
        monkeypatch, lambda: ingest._failed_evidence_projection_uuids(data["b"])
    )
    assert none_http == none_docker == []


def test_cypher_string_null_integer_rows_identical(data, monkeypatch):
    query = "RETURN 'abc' AS s, null AS n, 42 AS i;"
    via_http, via_docker = _both(monkeypatch, lambda: ingest._cypher(query))
    assert via_http == via_docker == [["abc", "NULL", "42"]]


def test_closed_http_port_falls_back_to_docker(data, monkeypatch):
    monkeypatch.setattr(ingest, "NEO4J_CONTAINER", CONTAINER)
    monkeypatch.setattr(ingest, "NEO4J_PW", PW)
    monkeypatch.setattr(
        ingest, "_NEO4J_HTTP", Neo4jHttpReader("http://127.0.0.1:1", "neo4j", PW, timeout_s=2.0)
    )
    assert ingest._cypher("RETURN 'abc' AS s, null AS n, 42 AS i;") == [["abc", "NULL", "42"]]
    counts = ingest._ns_state_counts(data["a"])
    assert counts["total"] == 5 and counts["failed"] == 2
    many = ingest._ns_state_counts_many([data["a"], data["c"]])
    assert many[data["a"]] == counts
    assert many[data["c"]]["total"] == 0


def test_both_paths_broken_returns_empty_and_minus_one(data, monkeypatch):
    monkeypatch.setattr(ingest, "NEO4J_CONTAINER", f"no-such-container-{uuid.uuid4().hex[:8]}")
    monkeypatch.setattr(ingest, "NEO4J_PW", PW)
    monkeypatch.setattr(
        ingest, "_NEO4J_HTTP", Neo4jHttpReader("http://127.0.0.1:1", "neo4j", PW, timeout_s=2.0)
    )
    assert ingest._cypher("RETURN 1;") == []
    assert ingest._ns_state_counts(data["a"]) == {k: -1 for k in KEYS}
    assert ingest._ns_state_counts_many([data["a"]]) == {data["a"]: {k: -1 for k in KEYS}}
