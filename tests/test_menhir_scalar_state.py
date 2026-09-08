"""Offline tests for the Menhir ScalarStateView (Piece C) e2e benchmark driver.

Network-free: a deterministic in-memory stub models the happy Piece C consumer (both the ingest
client and the bolt reader), so the driver, invariant validators, per-case soft matches, metrics, and
report writers are covered without a live menhir, a real LLM, or Neo4j. Invariant-failure detection is
exercised with small fault-injecting fake bolt readers.
"""

from __future__ import annotations

import json

import pytest

from archolith_bench.harness import (
    MenhirScalarStateAdapter,
    StubScalarStateClient,
    default_scalar_state_cases,
    get_adapter,
    is_scalar_state,
    run_scalar_state,
    scalar_state_result_to_dict,
    write_scalar_state_evidence,
)
from archolith_bench.harness.menhir_scalar_state import (
    VALUE_KINDS,
    fixture_hash,
    _value_matches,
)
from archolith_bench.harness.scalar_bolt import ProdBoltRefused, assert_not_prod


# ---------------------------------------------------------------------------- fixtures


def test_positive_denominator_is_seven_kinds():
    # Contract: 7 eligible POSITIVE "view" kinds. money + status are exercised as NEGATIVE CONTROLS
    # (a one-off payment event; a possessed object that must not bind to self), not Views.
    cases = default_scalar_state_cases()
    view_kinds = {c.expect_kind for c in cases if c.outcome == "view"}
    assert view_kinds == set(VALUE_KINDS) - {"money", "status"}
    assert sum(1 for c in cases if c.outcome == "view") == 7
    # money + status still exercised, but as negative controls
    control_kinds = {c.expect_kind for c in cases if c.outcome in ("advisory", "nothing")}
    assert {"money", "status"} <= control_kinds
    assert any(c.outcome == "advisory" for c in cases)
    assert any(c.outcome == "nothing" for c in cases)


def test_negative_controls_are_explicit():
    cases = {c.case_id: c for c in default_scalar_state_cases()}
    assert cases["ss-control-car-advisory"].outcome == "advisory"
    assert cases["ss-control-money-event"].outcome == "nothing"


def test_fixture_hash_is_stable_and_sensitive():
    a = default_scalar_state_cases()
    assert fixture_hash(a) == fixture_hash(default_scalar_state_cases())
    b = a[:-1]
    assert fixture_hash(a) != fixture_hash(b)


def test_adapter_registered_and_typed():
    adapter = get_adapter("menhir-scalar-state")
    assert isinstance(adapter, MenhirScalarStateAdapter)
    assert is_scalar_state(adapter)
    assert not is_scalar_state(object())


# ---------------------------------------------------------------------------- value matching


@pytest.mark.parametrize(
    "expected,actual,ok",
    [
        (37.0, 37.0, True),
        (37.0, "37", True),
        (250.0, 249.0, False),
        ("Wednesday", "wednesday", True),
        ("07:30", "07:30", True),
        (True, "true", True),
        (True, "false", False),
        (True, True, True),
        (None, None, True),
    ],
)
def test_value_matches(expected, actual, ok):
    assert _value_matches(expected, actual) is ok


# ---------------------------------------------------------------------------- happy path (stub)


def test_stub_happy_path_passes_all_invariants():
    stub = StubScalarStateClient()
    result = run_scalar_state(
        stub, stub, reset_confirmed=True, max_wait_s=0.0, poll_interval_s=0.0,
        menhir_url="offline-stub", neo4j_uri="offline-stub", model="offline-stub",
    )
    assert result.verdict is True
    inv = result.invariants
    assert inv["at_least_one_view"] is True
    assert inv["duplicate_current_keys"] == 0
    assert inv["duplicate_slots"] == 0
    assert inv["non_agent_tiers"] == []
    assert inv["wrong_namespace_views"] == 0
    assert inv["default_silo_leak"] == 0
    # 7 eligible positive view slots recognized by the stub; both negative controls clean
    assert inv["views_current"] == 7
    assert result.metrics["view_slots_committed"] == 7
    assert result.metrics["expected_view_slots"] == 7
    assert result.metrics["controls_clean"] is True     # money-event + non-scalar produced nothing
    assert result.metrics["advisories_clean"] is True    # car-advisory + no-entity did not bind to a View
    assert result.metrics["advisories_pending"] == 1


def test_stub_requires_reset_confirmation():
    stub = StubScalarStateClient()
    with pytest.raises(ValueError):
        run_scalar_state(stub, stub, reset_confirmed=False)


def test_stub_teardown_leaves_no_residue():
    stub = StubScalarStateClient()
    result = run_scalar_state(
        stub, stub, reset_confirmed=True, cleanup=True, max_wait_s=0.0, poll_interval_s=0.0,
    )
    # after cleanup the namespace is purged, so a re-read finds nothing
    assert stub.read_scalar_state_views(result.namespace) == []


# ---------------------------------------------------------------------------- invariant failures


class _FaultyBolt:
    """A bolt reader that returns injected rows so invariant-failure detection can be exercised."""

    def __init__(self, assertions, views, advisories=None, default_views=None) -> None:
        self._assertions = assertions
        self._views = views
        self._advisories = advisories or []
        self._default_views = default_views or []

    def read_typed_assertions(self, namespace):
        return self._assertions

    def read_scalar_state_views(self, namespace):
        return self._default_views if namespace == "default" else self._views

    def read_pending_advisories(self, namespace):
        return self._advisories


def _view(ns, subject, attr, kind, value, key):
    return {
        "subject_uuid": subject, "ss_attribute": attr, "ss_kind": kind, "ss_unit": None,
        "value": value, "group_id": ns, "view_key": key,
    }


def _assertion(subject, attr, kind, value, tier="agent"):
    return {
        "subject_uuid": subject, "subject_display": subject, "attribute": attr,
        "value_kind": kind, "value": value, "evidence_tier": tier,
        "binding_pending": False, "source_key": f"k:{attr}",
    }


def _run_with_bolt(bolt):
    stub = StubScalarStateClient()  # ingest role only; verification comes from the injected bolt
    return run_scalar_state(
        stub, bolt, reset_confirmed=True, max_wait_s=0.0, poll_interval_s=0.0,
        namespace="ns-fault",
    )


def test_no_views_fails():
    result = _run_with_bolt(_FaultyBolt(assertions=[], views=[]))
    assert result.verdict is False
    assert result.invariants["at_least_one_view"] is False


def test_duplicate_current_keys_fails():
    dup = [
        _view("ns-fault", "s1", "coin_count", "count", 37.0, "K"),
        _view("ns-fault", "s2", "gym_frequency", "frequency", 3.0, "K"),  # same view_key
    ]
    result = _run_with_bolt(_FaultyBolt(assertions=[], views=dup))
    assert result.verdict is False
    assert result.invariants["duplicate_current_keys"] == 1


def test_non_agent_tier_fails():
    views = [_view("ns-fault", "s1", "coin_count", "count", 37.0, "K1")]
    bad = [_assertion("s1", "coin_count", "count", 37.0, tier="user")]
    result = _run_with_bolt(_FaultyBolt(assertions=bad, views=views))
    assert result.verdict is False
    assert "user" in result.invariants["non_agent_tiers"]


def test_default_silo_leak_fails():
    views = [_view("ns-fault", "s1", "coin_count", "count", 37.0, "K1")]
    leaked = [_view("default", "s9", "coin_count", "count", 37.0, "KD")]
    result = _run_with_bolt(_FaultyBolt(assertions=[], views=views, default_views=leaked))
    assert result.verdict is False
    assert result.invariants["default_silo_leak"] == 1


def test_wrong_namespace_view_fails():
    views = [_view("some-other-ns", "s1", "coin_count", "count", 37.0, "K1")]
    result = _run_with_bolt(_FaultyBolt(assertions=[], views=views))
    assert result.verdict is False
    assert result.invariants["wrong_namespace_views"] == 1


# ---------------------------------------------------------------------------- reporting


def test_write_markdown_evidence(tmp_path):
    stub = StubScalarStateClient()
    result = run_scalar_state(
        stub, stub, reset_confirmed=True, max_wait_s=0.0, poll_interval_s=0.0,
    )
    out = tmp_path / "evidence.md"
    write_scalar_state_evidence(result, out, output_format="markdown")
    text = out.read_text(encoding="utf-8")
    assert "Menhir ScalarStateView E2E Benchmark" in text
    assert "**PASS**" in text
    assert "Hard invariants" in text
    assert "archolith-bench harness menhir-scalar-state" in text


def test_write_json_evidence(tmp_path):
    stub = StubScalarStateClient()
    result = run_scalar_state(
        stub, stub, reset_confirmed=True, max_wait_s=0.0, poll_interval_s=0.0,
    )
    out = tmp_path / "evidence.json"
    write_scalar_state_evidence(result, out, output_format="json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] == "pass"
    assert data["menhir_url_safety_class"] == "throwaway"
    assert data["invariants"]["views_current"] == 7
    assert data == scalar_state_result_to_dict(result)


# ---------------------------------------------------------------------------- bolt safety


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://192.168.86.33:7687",
        "bolt://localhost:7687",
        "bolt://127.0.0.1:7687",
        "neo4j://myhost:7687",
    ],
)
def test_assert_not_prod_refuses_prod(uri):
    with pytest.raises(ProdBoltRefused):
        assert_not_prod(uri)


@pytest.mark.parametrize(
    "uri",
    ["bolt://localhost:7688", "bolt://127.0.0.1:7688", "bolt://localhost:7690"],
)
def test_assert_not_prod_allows_loopback_throwaway(uri):
    assert_not_prod(uri)  # does not raise


@pytest.mark.parametrize(
    "uri",
    [
        "bolt://192.168.86.33:7688",   # remote LAN host, any port
        "neo4j://throwaway:7690",      # named remote host
        "neo4j+s://memory.ctharvey.me",  # the real production endpoint
        "bolt://memory.ctharvey.me:7688",
    ],
)
def test_assert_not_prod_refuses_non_loopback(uri):
    """Policy change: the guard allow-lists loopback instead of deny-listing prod.

    The deny-list failed open. When production moved to a host containing none
    of its markers, the real instance was permitted -- `neo4j+s://memory.ctharvey.me`
    passed. Any non-loopback host is now refused by default, whatever the port.
    """
    with pytest.raises(ProdBoltRefused):
        assert_not_prod(uri)


def test_non_loopback_can_be_opted_in_deliberately(monkeypatch):
    monkeypatch.setenv("ARCHOLITH_BENCH_ALLOW_HOSTS", "192.168.86.40")
    assert_not_prod("bolt://192.168.86.40:7688")  # does not raise


def test_opt_in_cannot_defeat_the_prod_name_check(monkeypatch):
    """Allow-listing a host must not override a production-looking name."""
    monkeypatch.setenv("ARCHOLITH_BENCH_ALLOW_HOSTS", "prod-menhir.internal")
    with pytest.raises(ProdBoltRefused):
        assert_not_prod("bolt://prod-menhir.internal:7688")
