"""Menhir ScalarStateView (Piece C) end-to-end / acceptance benchmark.

This is the successor to the CLOSED lexical typed-value sidecar line (`scripts/longmemeval/analysis/
TYPED-VALUE-ARM.md`), whose terminal finding was: *"Further progress requires integration with Menhir's
entity and View layers, not more sidecar tuning."* Piece C IS that integration -- real entity-resolved
typed scalars in Menhir's View layer -- so this harness exercises the REAL path end to end:

    /api/memory ingest (real :Episodic episodes)
        -> background scheduler tick (consolidate_personal_memory, enable_scalar_state=True)
            -> TypedScalarPerceptionService: k-sample LLM perceive -> gate -> bind
                -> durable :TypedAssertion event log
                    -> deterministic fold -> materialized `scalar_state` Views
                        -> [verify over BOLT -- shadow Views are NOT recall-visible]

Unlike `menhir-phase3` (which triggers the numeric-counter path over HTTP via POST /api/phase3/run),
the typed-scalar path runs ONLY inside scheduled consolidation gated by
`MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED=1`. So this harness drives a throwaway menhir launched with
the flag on + the background scheduler ON (benchmark mode OFF) + a short consolidation interval, ingests
known typed-scalar episodes, waits for the scheduler to materialize the Views, and asserts INVARIANTS
(not exact stochastic values) over bolt: one current View per slot, correct namespace, tier=agent, no
duplicate current keys, no default-silo leakage.

Result is a purpose-built `ScalarStateResult` (assertions/Views/advisories/invariants). Driven by
`run_scalar_state`; reported by `write_scalar_state_evidence`. See
`benchmarks/RUNBOOK-scalar-state-e2e.md` for the launch profile.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

# The nine typed-scalar ValueKinds Menhir perception proposes (typed_scalar_perception.py schema).
VALUE_KINDS = (
    "boolean", "status", "count", "duration", "frequency",
    "money", "measurement", "clock_time", "weekday",
)

_FLOAT_TOL = 1e-6


# ---------------------------------------------------------------------------- fixtures


@dataclass(frozen=True)
class ScalarStateCase:
    """One known typed-scalar episode plus the View/assertion it should drive.

    `prompt` is ingested verbatim as a `user:` episode. `expect_kind`/`expect_value` are the typed
    scalar a correct perception should produce (soft-matched -- perception is stochastic). `outcome`
    selects the invariant class:
        "view"     -> a resolvable entity: expect a current `scalar_state` View for this slot
        "advisory" -> no uniquely-resolvable entity: expect a pending `unbound:` advisory OR an abstain
        "nothing"  -> a one-off happening / non-scalar: expect NO assertion and NO View
    `subject_needles` help tie a stochastic result row back to this fixture for reporting.
    """

    case_id: str
    prompt: str
    outcome: str  # view | advisory | nothing
    expect_kind: str | None = None
    expect_value: float | str | None = None
    subject_needles: tuple[str, ...] = ()


def default_scalar_state_cases() -> list[ScalarStateCase]:
    """Known-input fixtures: 7 eligible POSITIVE "view" cases + 4 negative/edge controls.

    Positive denominator is 7, not 9: two prompts are explicit NEGATIVE CONTROLS, not Views —
    "My car is red" (`advisory`: a possessed object must NOT bind to the canonical self entity or
    materialize a self View) and "I paid $250 ..." (`nothing`: a one-off past event is not a standing
    property, so no assertion and no View). Report positives as x/7 and controls as y/2 (plus the two
    long-standing controls: the no-entity advisory and the non-scalar happening). Every "view" case
    names a single unambiguous subject and a distinct value so a stochastic row maps back to its fixture.
    The nine ValueKinds are all still exercised (money via the money-event control, status via the
    car-advisory control).
    """
    return [
        ScalarStateCase(
            case_id="ss-count-coins", outcome="view", expect_kind="count", expect_value=37.0,
            prompt="I own 37 rare coins.", subject_needles=("coin",),
        ),
        ScalarStateCase(
            case_id="ss-measurement-height", outcome="view", expect_kind="measurement", expect_value=180.0,
            prompt="My height is 180 centimeters.", subject_needles=("height",),
        ),
        ScalarStateCase(
            case_id="ss-duration-commute", outcome="view", expect_kind="duration", expect_value=45.0,
            prompt="My commute takes 45 minutes.", subject_needles=("commute",),
        ),
        ScalarStateCase(
            case_id="ss-frequency-gym", outcome="view", expect_kind="frequency", expect_value=3.0,
            prompt="I go to the gym 3 times a week.", subject_needles=("gym",),
        ),
        ScalarStateCase(
            case_id="ss-clock-wake", outcome="view", expect_kind="clock_time", expect_value="07:30",
            prompt="I wake up at 7:30 every morning.", subject_needles=("wake",),
        ),
        ScalarStateCase(
            case_id="ss-weekday-dayoff", outcome="view", expect_kind="weekday", expect_value="Wednesday",
            prompt="My day off is Wednesday.", subject_needles=("day off", "dayoff"),
        ),
        ScalarStateCase(
            case_id="ss-boolean-book", outcome="view", expect_kind="boolean", expect_value=True,
            prompt="I have finished reading Dune.", subject_needles=("dune", "book", "read"),
        ),
        # NEGATIVE CONTROL (possessed object, NOT the self speaker): "my car" is not a first-person
        # self reference and has no uniquely-resolvable :Entity, so a correct perceiver leaves it an
        # unbound advisory. The ONLY failure is binding it to self / materializing a status='red' View
        # (that would mean a possessed object wrongly resolved to the canonical self entity). Expect
        # kind/value are kept so the validator can DETECT such a stray self View, not to demand one.
        ScalarStateCase(
            case_id="ss-control-car-advisory", outcome="advisory", expect_kind="status", expect_value="red",
            prompt="My car is red.", subject_needles=("car",),
        ),
        # NEGATIVE CONTROL (one-off past EVENT, not a standing property): "I paid $250" is a completed
        # transaction, not a current state, so a correct perceiver emits NO standing money assertion and
        # NO current View. kind/value kept so a stray money=250 assertion/View is caught as over-perception.
        ScalarStateCase(
            case_id="ss-control-money-event", outcome="nothing", expect_kind="money", expect_value=250.0,
            prompt="I paid $250 for my new headphones.", subject_needles=("headphone", "paid"),
        ),
        # No uniquely-resolvable subject -> pending `unbound:` advisory OR a safe abstain (both ok).
        ScalarStateCase(
            case_id="ss-advisory-noentity", outcome="advisory", expect_kind="count", expect_value=12.0,
            prompt="There are 12 of them left.", subject_needles=("them",),
        ),
        # A one-off happening, not a current property -> perception must emit NOTHING.
        ScalarStateCase(
            case_id="ss-control-nonscalar", outcome="nothing",
            prompt="I went for a walk this morning.", subject_needles=("walk",),
        ),
    ]


def third_party_scalar_state_cases() -> list[ScalarStateCase]:
    """Named third-party subject fixtures (Lever 3): constant identity 'Alice', multiple ValueKinds.

    Isolates the binder from the first-person self-entity gap: every subject is a nameable KG entity
    ('Alice'), so IF Graphiti extracts + links an 'Alice' :Entity, `_bind_subject` should bind and a
    View should materialize. If nothing binds, the blocker is upstream entity extraction/linkage, not
    the self-entity gap. Kept tiny + constant-identity on purpose (multiple kinds, one entity).
    """
    return [
        ScalarStateCase(
            case_id="tp-alice-coins", outcome="view", expect_kind="count", expect_value=37.0,
            prompt="Alice owns 37 coins.", subject_needles=("alice",),
        ),
        ScalarStateCase(
            case_id="tp-alice-books", outcome="view", expect_kind="count", expect_value=12.0,
            prompt="Alice has read 12 books.", subject_needles=("alice",),
        ),
        ScalarStateCase(
            case_id="tp-alice-wake", outcome="view", expect_kind="clock_time", expect_value="07:30",
            prompt="Alice wakes up at 7:30 AM.", subject_needles=("alice",),
        ),
    ]


SCALAR_FIXTURE_SETS = {
    "default": default_scalar_state_cases,
    "third-party": third_party_scalar_state_cases,
}


def fixture_hash(cases: list[ScalarStateCase]) -> str:
    """Stable digest of the fixture set (prompts + expectations) for reproducibility."""
    blob = json.dumps([asdict(c) for c in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------- result types


@dataclass
class ScalarStateCaseResult:
    """Per-fixture soft-match outcome (reporting; not a hard verdict gate on stochastic values)."""

    case_id: str
    outcome: str
    prompt: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    matched: bool
    notes: str = ""


@dataclass
class ScalarStateResult:
    """Full ScalarStateView e2e result -- invariant-oriented, not score/token oriented."""

    benchmark_id: str
    name: str
    namespace: str
    menhir_url: str
    neo4j_uri: str
    cases: list[ScalarStateCaseResult]
    invariants: dict[str, Any]
    metrics: dict[str, Any]
    verdict: bool
    warnings: list[str] = field(default_factory=list)
    run_meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------- client protocols


class _IngestClient(Protocol):
    """The subset of the menhir HTTP client the driver needs (ingest + turn evidence + teardown)."""

    def record_turn_evidence(self, namespace: str, text: str, **kwargs: Any) -> dict[str, Any]: ...
    def ingest(self, group_id: str, role: str, content: str, **kwargs: Any) -> None: ...
    def reset(self, group_id: str) -> None: ...


class _BoltReader(Protocol):
    """The read-only bolt surface the driver verifies against (scalar_bolt.ScalarBoltReader)."""

    def read_typed_assertions(self, namespace: str) -> list[dict[str, Any]]: ...
    def read_scalar_state_views(self, namespace: str) -> list[dict[str, Any]]: ...
    def read_pending_advisories(self, namespace: str) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------- match helpers


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_matches(expected: float | str | bool | None, actual: Any) -> bool:
    """Soft equality across the typed representations (numeric tolerance; case-insensitive text)."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        # booleans may surface as bool, or as "true"/"false"/"yes" text, or 1.0/0.0
        s = str(actual).strip().lower()
        truthy = s in {"true", "yes", "1", "1.0"} or actual is True
        falsy = s in {"false", "no", "0", "0.0"} or actual is False
        return truthy if expected else falsy
    ef = _to_float(expected)
    if ef is not None:
        af = _to_float(actual)
        return af is not None and abs(af - ef) <= _FLOAT_TOL
    # text (status/weekday/clock_time): case-insensitive substring either direction
    es = str(expected).strip().lower()
    as_ = str(actual).strip().lower()
    return bool(es) and (es in as_ or as_ in es)


def _find_assertion(rows: list[dict[str, Any]], case: ScalarStateCase) -> dict[str, Any] | None:
    """First assertion row whose kind+value match the fixture (value is the disambiguator)."""
    for row in rows:
        if case.expect_kind and str(row.get("value_kind") or "") != case.expect_kind:
            continue
        if _value_matches(case.expect_value, row.get("value")):
            return row
    return None


def _view_value(row: dict[str, Any]) -> Any:
    """A View's typed value: ss_value, else ss_display, else `value`. `value` is view_value, a numeric
    mirror that is 0.0 for text kinds (clock_time, weekday, status), so it cannot be compared there."""
    for key in ("ss_value", "ss_display"):
        if row.get(key) not in (None, ""):
            return row[key]
    return row.get("value")


def _find_view(rows: list[dict[str, Any]], case: ScalarStateCase) -> dict[str, Any] | None:
    """First current View whose kind+value match the fixture."""
    for row in rows:
        if case.expect_kind and str(row.get("ss_kind") or "") != case.expect_kind:
            continue
        if _value_matches(case.expect_value, _view_value(row)):
            return row
    return None


# ---------------------------------------------------------------------------- per-case validators


def _validate_case(
    case: ScalarStateCase,
    assertions: list[dict[str, Any]],
    views: list[dict[str, Any]],
    advisories: list[dict[str, Any]],
) -> ScalarStateCaseResult:
    """Soft-match one fixture to the observed graph state (reporting-level, not a hard gate).

    Stochastic perception fails closed, so a single miss on a "view" case is a soft NON-match
    (reported), never a harness failure by itself -- the hard verdict is the structural invariants.
    """
    expected = {"outcome": case.outcome, "kind": case.expect_kind, "value": case.expect_value}
    if case.outcome == "view":
        view = _find_view(views, case)
        assertion = _find_assertion(assertions, case)
        matched = view is not None
        actual = {
            "view": {"kind": view.get("ss_kind"), "value": _view_value(view)} if view else None,
            "assertion_tier": assertion.get("evidence_tier") if assertion else None,
        }
        notes = "" if matched else "no current scalar_state View matched this slot (stochastic abstain?)"
        return ScalarStateCaseResult(case.case_id, case.outcome, case.prompt, expected, actual, matched, notes)

    if case.outcome == "advisory":
        adv = _find_assertion(advisories, case)
        # advisory is satisfied by a pending unbound advisory OR by a safe abstain (no view, no assertion)
        stray_view = _find_view(views, case)
        matched = stray_view is None  # the only real failure is binding it to a WRONG entity/View
        actual = {"pending_advisory": bool(adv), "stray_view": bool(stray_view)}
        notes = "" if matched else "no-entity statement bound to a concrete View (should stay advisory/abstain)"
        return ScalarStateCaseResult(case.case_id, case.outcome, case.prompt, expected, actual, matched, notes)

    # outcome == "nothing": a non-scalar / one-off happening must produce no assertion and no View
    stray_assertion = _find_assertion(assertions, case) if case.expect_kind else None
    stray_view = _find_view(views, case) if case.expect_kind else None
    # with no expect_kind we cannot key on kind/value; fall back to needle scan over subject_display
    if case.expect_kind is None:
        needles = [n.lower() for n in case.subject_needles]
        stray_assertion = next(
            (a for a in assertions if any(n in str(a.get("subject_display", "")).lower() for n in needles)),
            None,
        )
    matched = stray_assertion is None and stray_view is None
    actual = {"stray_assertion": bool(stray_assertion), "stray_view": bool(stray_view)}
    notes = "" if matched else "non-scalar control produced a typed assertion/View (over-perception)"
    return ScalarStateCaseResult(case.case_id, case.outcome, case.prompt, expected, actual, matched, notes)


# ---------------------------------------------------------------------------- invariants


def _compute_invariants(
    namespace: str,
    assertions: list[dict[str, Any]],
    views: list[dict[str, Any]],
    default_views: list[dict[str, Any]],
) -> dict[str, Any]:
    """The HARD structural invariants that gate the verdict (independent of stochastic values).

    These mirror the Piece C guarantees the handoff calls out: one current View per slot, strict
    namespace fidelity (no default-silo leakage), perception tier always `agent`, and no duplicate
    current View keys.
    """
    view_keys = [str(v.get("view_key") or "") for v in views if v.get("view_key")]
    slot_keys = [(str(v.get("subject_uuid") or ""), str(v.get("ss_attribute") or "")) for v in views]

    duplicate_current_keys = len(view_keys) - len(set(view_keys)) if view_keys else 0
    duplicate_slots = len(slot_keys) - len(set(slot_keys))

    non_agent_tiers = sorted(
        {str(a.get("evidence_tier")) for a in assertions if str(a.get("evidence_tier")) != "agent"}
    )
    wrong_namespace_views = sum(
        1 for v in views if str(v.get("group_id") or "") != namespace
    )
    default_silo_leak = len(default_views)  # any current scalar_state View in `default` from this run

    return {
        "views_current": len(views),
        "assertions_current": len(assertions),
        "duplicate_current_keys": duplicate_current_keys,
        "duplicate_slots": duplicate_slots,
        "non_agent_tiers": non_agent_tiers,
        "wrong_namespace_views": wrong_namespace_views,
        "default_silo_leak": default_silo_leak,
        "at_least_one_view": len(views) > 0,
    }


def _verdict_from_invariants(inv: dict[str, Any]) -> bool:
    """PASS iff every hard invariant holds. Per-case value matches are reported, not gated."""
    return bool(
        inv["at_least_one_view"]
        and inv["duplicate_current_keys"] == 0
        and inv["duplicate_slots"] == 0
        and not inv["non_agent_tiers"]
        and inv["wrong_namespace_views"] == 0
        and inv["default_silo_leak"] == 0
    )


# ---------------------------------------------------------------------------- driver


def run_scalar_state(
    client: _IngestClient,
    bolt: _BoltReader,
    *,
    cases: list[ScalarStateCase] | None = None,
    namespace: str | None = None,
    reset_confirmed: bool = False,
    cleanup: bool = True,
    poll_interval_s: float = 3.0,
    max_wait_s: float = 180.0,
    menhir_url: str = "",
    neo4j_uri: str = "",
    model: str = "",
) -> ScalarStateResult:
    """Run the ScalarStateView e2e scenario against a throwaway menhir with the scalar scheduler ON.

    Ingests known typed-scalar `user:` episodes, waits for the BACKGROUND scheduler to materialize
    `scalar_state` Views (perception is async + stochastic), then verifies invariants over bolt.
    Requires `reset_confirmed=True` (it writes to + tears down a real throwaway namespace).

    Preconditions (see the runbook): the menhir at `menhir_url` runs with
    `MENHIR_BENCHMARK_MODE=0`, `MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED=1`,
    `MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED=1`, a short consolidation interval, and a real LLM;
    `bolt` targets the SAME throwaway Neo4j.
    """
    if not reset_confirmed:
        raise ValueError(
            "run_scalar_state mutates and tears down a real (throwaway) Menhir namespace; "
            "pass --confirm-menhir-reset to proceed."
        )
    cases = cases or default_scalar_state_cases()
    namespace = namespace or f"scalar-e2e-{uuid.uuid4().hex[:12]}"
    warnings: list[str] = []

    # 0. isolate: best-effort teardown for a clean slate ----------------------------------------
    client.reset(namespace)

    # 1. ingest known typed-scalar episodes (real :Episodic, entity-resolved) -------------------
    # Ground each source="user" claim against a recorded :TurnEvidence uuid so admission + entity
    # resolution behave like production. Backdate occurred_at monotonically so the scalar cursor
    # order is deterministic across the batch.
    base_time = datetime.now(timezone.utc) - timedelta(minutes=len(cases) + 1)
    ingested = 0
    for i, case in enumerate(cases):
        occurred_at = (base_time + timedelta(minutes=i)).isoformat()
        turn_uuid: str | None = None
        try:
            ev = client.record_turn_evidence(
                namespace, case.prompt, turn_key=f"{namespace}:{case.case_id}"
            )
            turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
        except Exception as exc:  # grounding is best-effort; ingest still creates the episode
            warnings.append(f"{case.case_id}: turn-evidence grounding failed: {exc}")
        client.ingest(
            namespace, "user", case.prompt,
            source="user", occurred_at=occurred_at, turn_evidence_uuid=turn_uuid, wait=True,
        )
        ingested += 1

    # 2. wait for the background scheduler to materialize Views ----------------------------------
    # The scalar pass runs asynchronously on the consolidation interval. Poll bolt until Views for
    # this namespace appear or the bound elapses. "view" fixtures are the ones that should commit.
    expected_views = sum(1 for c in cases if c.outcome == "view")
    deadline = time.monotonic() + max_wait_s
    views: list[dict[str, Any]] = []
    first_view_at: float | None = None
    start = time.monotonic()
    while time.monotonic() < deadline:
        views = bolt.read_scalar_state_views(namespace)
        if views and first_view_at is None:
            first_view_at = time.monotonic() - start
        # Stop once we have at least a majority of the expected view-slots (perception is lossy),
        # or as soon as any View exists if we only expected a couple.
        if len(views) >= max(1, (expected_views + 1) // 2):
            break
        time.sleep(poll_interval_s)
    waited_s = round(time.monotonic() - start, 1)

    # 3. read the full graph state over bolt ----------------------------------------------------
    # Authoritative final read of Views (the poll loop may have exited before its last read, and
    # a zero wait -- offline stub -- skips the loop body entirely).
    views = bolt.read_scalar_state_views(namespace)
    assertions = bolt.read_typed_assertions(namespace)
    if not views and expected_views:
        warnings.append(
            f"no scalar_state Views materialized within {max_wait_s}s -- is the scheduler on "
            "(MENHIR_BENCHMARK_MODE=0, CONSOLIDATION_ENABLED=1, SCALAR_STATE_ENABLED=1) and the LLM up?"
        )
    advisories = bolt.read_pending_advisories(namespace)
    try:
        default_views = bolt.read_scalar_state_views("default")
    except Exception as exc:
        default_views = []
        warnings.append(f"default-silo leak probe failed: {exc}")

    # 4. per-case soft match + hard invariants --------------------------------------------------
    case_results = [_validate_case(c, assertions, views, advisories) for c in cases]
    invariants = _compute_invariants(namespace, assertions, views, default_views)
    verdict = _verdict_from_invariants(invariants)

    committed_views = sum(1 for cr in case_results if cr.outcome == "view" and cr.matched)
    metrics = {
        "episodes_ingested": ingested,
        "expected_view_slots": expected_views,
        "view_slots_committed": committed_views,
        "assertions_current": len(assertions),
        "advisories_pending": len(advisories),
        "waited_s": waited_s,
        "first_view_after_s": round(first_view_at, 1) if first_view_at is not None else None,
        "controls_clean": all(cr.matched for cr in case_results if cr.outcome == "nothing"),
        "advisories_clean": all(cr.matched for cr in case_results if cr.outcome == "advisory"),
    }

    run_meta = {
        "namespace": namespace,
        "menhir_url": menhir_url,
        "neo4j_uri": neo4j_uri,
        "model": model,
        "fixture_hash": fixture_hash(cases),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if cleanup:
        client.reset(namespace)

    return ScalarStateResult(
        benchmark_id=MenhirScalarStateAdapter.benchmark_id,
        name=MenhirScalarStateAdapter.name,
        namespace=namespace,
        menhir_url=menhir_url,
        neo4j_uri=neo4j_uri,
        cases=case_results,
        invariants=invariants,
        metrics=metrics,
        verdict=verdict,
        warnings=warnings,
        run_meta=run_meta,
    )


# ---------------------------------------------------------------------------- adapter


class MenhirScalarStateAdapter:
    """Registry entry so the harness CLI can find the ScalarStateView e2e benchmark by id."""

    benchmark_id = "menhir-scalar-state"
    name = "Menhir ScalarStateView E2E"
    is_scalar_state_adapter = True

    def cases(self) -> list[ScalarStateCase]:
        """Return the default fixture set."""
        return default_scalar_state_cases()


def is_scalar_state(adapter: object) -> bool:
    """True if the adapter drives the ScalarStateView e2e benchmark (run via run_scalar_state)."""
    return bool(getattr(adapter, "is_scalar_state_adapter", False))


# ---------------------------------------------------------------------------- reporting


def scalar_state_result_to_dict(result: ScalarStateResult) -> dict[str, Any]:
    """Serialize a ScalarStateResult to a stable JSON-compatible dict."""
    return {
        "benchmark_id": result.benchmark_id,
        "run_id": result.namespace,
        "timestamp": result.run_meta.get("timestamp"),
        "menhir_url_safety_class": "throwaway",
        "menhir_url": result.menhir_url,
        "neo4j_uri": result.neo4j_uri,
        "namespace": result.namespace,
        "model": result.run_meta.get("model"),
        "fixture_hash": result.run_meta.get("fixture_hash"),
        "verdict": "pass" if result.verdict else "fail",
        "invariants": result.invariants,
        "metrics": result.metrics,
        "cases": [asdict(c) for c in result.cases],
        "warnings": result.warnings,
        "run_meta": result.run_meta,
    }


def write_scalar_state_evidence(
    result: ScalarStateResult, out_path: str | Path, output_format: str = "markdown",
) -> Path:
    """Write purpose-built ScalarStateView e2e evidence (markdown or JSON)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scalar_state_result_to_dict(result), f, indent=2, ensure_ascii=False)
        return out_path

    inv = result.invariants
    m = result.metrics
    lines: list[str] = [
        "# Menhir ScalarStateView E2E Benchmark\n\n",
        "## Run metadata\n\n",
        f"- Benchmark: `{result.benchmark_id}`\n",
        f"- Menhir URL: `{result.menhir_url}` (throwaway)\n",
        f"- Neo4j (bolt): `{result.neo4j_uri}` (throwaway)\n",
        f"- Namespace: `{result.namespace}`\n",
        f"- Model/provider: `{result.run_meta.get('model') or '(server default)'}`\n",
        f"- Fixture hash: `{result.run_meta.get('fixture_hash')}`\n",
        f"- Timestamp: `{result.run_meta.get('timestamp')}`\n\n",
        "## Verdict\n\n",
        f"**{'PASS' if result.verdict else 'FAIL'}**\n\n",
        "## Hard invariants (verdict gate)\n\n",
        "| Invariant | Value |\n|-----------|-------|\n",
        f"| At least one current View | {inv['at_least_one_view']} |\n",
        f"| Current Views | {inv['views_current']} |\n",
        f"| Current assertions | {inv['assertions_current']} |\n",
        f"| Duplicate current View keys | {inv['duplicate_current_keys']} |\n",
        f"| Duplicate (subject, slot) pairs | {inv['duplicate_slots']} |\n",
        f"| Non-`agent` evidence tiers | {inv['non_agent_tiers'] or 'none'} |\n",
        f"| Wrong-namespace Views | {inv['wrong_namespace_views']} |\n",
        f"| Default-silo leak | {inv['default_silo_leak']} |\n\n",
        "## Metrics\n\n",
        "| Metric | Value |\n|--------|-------|\n",
        f"| Episodes ingested | {m['episodes_ingested']} |\n",
        f"| Expected view slots | {m['expected_view_slots']} |\n",
        f"| View slots committed | {m['view_slots_committed']} |\n",
        f"| Advisories pending | {m['advisories_pending']} |\n",
        f"| Advisory cases clean | {m['advisories_clean']} |\n",
        f"| Non-scalar controls clean | {m['controls_clean']} |\n",
        f"| Waited (s) | {m['waited_s']} |\n",
        f"| First View after (s) | {m['first_view_after_s']} |\n\n",
        "## Per-case detail\n\n",
    ]
    for cr in result.cases:
        status = "MATCH" if cr.matched else ("MISS" if cr.outcome == "view" else "FAIL")
        lines.append(f"### {cr.case_id} ({cr.outcome}) - {status}\n\n")
        lines.append(f"- Prompt: {cr.prompt!r}\n")
        lines.append(f"- Expected: {cr.expected}\n")
        lines.append(f"- Actual: {cr.actual}\n")
        if cr.notes:
            lines.append(f"- Notes: {cr.notes}\n")
        lines.append("\n")

    if result.warnings:
        lines.append("## Warnings\n\n")
        for w in result.warnings:
            lines.append(f"- {w}\n")
        lines.append("\n")

    lines.append("## Reproduction command\n\n")
    lines.append("```bash\n")
    lines.append(
        "archolith-bench harness menhir-scalar-state \\\n"
        f"  --menhir-url {result.menhir_url or 'http://127.0.0.1:<throwaway-port>'} \\\n"
        f"  --neo4j-uri {result.neo4j_uri or 'bolt://localhost:7688'} \\\n"
        "  --confirm-menhir-reset \\\n"
        "  --format markdown --out results/menhir_scalar_state_e2e.md\n"
    )
    lines.append("```\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path
