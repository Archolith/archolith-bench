"""Phase D — counterfactual recall evaluation for ScalarStateView (OFFLINE, no menhir recall change).

Compares, per current-state question, a BASELINE recall (menhir's live graph/episode recall over HTTP)
against a VIEW-AWARE answer composed HERE in the harness from the current `scalar_state` Views. Nothing
in production recall changes and no user-facing suppression happens: the View-aware answer is a pure
counterfactual, so we can measure whether trusting a current View WOULD improve recall before ever
wiring authority into the live path.

Seven dimensions are scored SEPARATELY per question (never collapsed into one rate):
  1. current_state_detected  -- the question asks for a CURRENT standing value (not history);
  2. subject_resolved        -- the View/assertion subject is the entity the question is about;
  3. slot_overlap            -- the question's attribute maps EXACTLY to a current View slot;
  4. view_status             -- the selected View's value vs ground-truth current: correct|wrong|absent;
  5. stale_in_baseline       -- baseline recall surfaced the STALE predecessor value;
  6. answer_improved         -- View-aware gives the current answer where baseline is stale/ambiguous;
  7. wrongful_suppression    -- View-aware would suppress a fact that is ACTUALLY still valid (must be 0).

PRIMARY Phase D metrics are computed ONLY over rows where view_status == "correct" (so a MISSING View is
never confused with WRONG USE of an available one). End-to-end coverage across ALL questions is reported
separately.

Methodology (frozen, same as the coverage matrix): every live measurement uses a FRESH ISOLATED
menhir+neo4j stack. The gate stays k=3 / threshold=1.0 (step 5 frozen); a 2/3 shadow A/B comes later.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------- fixtures


@dataclass(frozen=True)
class PhaseDCase:
    """One counterfactual recall question over a scalar slot.

    A `kind`:
      * "current"    -- a current-state question with a materialized View AND an older STALE predecessor
                        fact in the graph. The current View SHOULD win; baseline may surface the stale value.
      * "historical" -- a PAST question ("...last year?") whose correct answer IS the stale value. A View
                        must NOT be applied/suppress it -- this is the wrongful-suppression tripwire.
      * "noview"     -- a current-state question with NO eligible View (advisory/absent). View-aware must
                        fall back to baseline, never fabricate authority.
    """

    case_id: str
    kind: str  # current | historical | noview
    slot_attribute: str  # the View ss_attribute this question targets (e.g. "owned")
    expect_kind: str
    question: str
    current_answer: str | float  # ground-truth CURRENT value
    stale_answer: str | float | None = None  # ground-truth predecessor value (None if no stale fact)
    current_prompt: str | None = None  # episode that establishes the current value
    stale_prompt: str | None = None  # older episode that establishes the stale value
    subject_needles: tuple[str, ...] = ()


def phase_d_cases() -> list[PhaseDCase]:
    """Current-state questions with stale predecessors, plus historical + no-View negative controls.

    Values are distinct across cases so a recalled snippet maps unambiguously back to its case. The
    current prompts reuse the ScalarStateView fixture shapes (they materialize Views); the stale prompts
    seed an OLDER overlapping graph fact so baseline recall can surface a superseded value.
    """
    return [
        # Stale predecessors use natural "used to / a while ago" phrasing -- how a real user states a
        # superseded fact. This is deliberately the REALISTIC, harder case: a correct pipeline must
        # recognize the current statement as the standing value and NOT let the past one win. Phase D
        # reports whatever the live pipeline actually does; we do NOT reword the input to make it pass.
        PhaseDCase(
            case_id="pd-coins-current", kind="current", slot_attribute="owned", expect_kind="count",
            question="How many rare coins do I own now?", current_answer=37.0, stale_answer=20.0,
            stale_prompt="A while ago I owned 20 rare coins.",
            current_prompt="I own 37 rare coins.", subject_needles=("coin",),
        ),
        PhaseDCase(
            case_id="pd-wake-current", kind="current", slot_attribute="wake_time", expect_kind="clock_time",
            question="What time do I wake up these days?", current_answer="07:30", stale_answer="09:00",
            stale_prompt="I used to wake up at 9:00 every morning.",
            current_prompt="I wake up at 7:30 every morning.", subject_needles=("wake",),
        ),
        PhaseDCase(
            case_id="pd-dayoff-current", kind="current", slot_attribute="day_off", expect_kind="weekday",
            question="Which weekday is my day off currently?", current_answer="Wednesday", stale_answer="Monday",
            stale_prompt="My day off used to be Monday.",
            current_prompt="My day off is Wednesday.", subject_needles=("day off", "dayoff"),
        ),
        # NEGATIVE CONTROL (historical): the correct answer is the STALE value; a View must NOT suppress it.
        PhaseDCase(
            case_id="pd-coins-historical", kind="historical", slot_attribute="owned", expect_kind="count",
            question="How many rare coins did I own a while ago?", current_answer=37.0, stale_answer=20.0,
            subject_needles=("coin",),  # reuses the coins episodes seeded by pd-coins-current
        ),
        # NEGATIVE CONTROL (no View): a current-state question whose slot has no eligible View -> fall back.
        PhaseDCase(
            case_id="pd-noview-current", kind="noview", slot_attribute="mood", expect_kind="status",
            question="What is my current mood?", current_answer="happy", stale_answer=None,
            subject_needles=("mood",),  # never ingested -> no View, no assertion
        ),
        # EXPIRED (used-to with NO replacement): "I used to live in Dallas" with no current residence.
        # Correct outcome is EXPIRED-with-current-unknown: NO current residence View. A lingering stale
        # 'Dallas' View is the defect (the expire did not end the value). current_answer is a sentinel:
        # there is no current value, so the View must be ABSENT, not any concrete value.
        PhaseDCase(
            case_id="pd-residence-expired", kind="expired", slot_attribute="residence", expect_kind="status",
            question="Where do I live now?", current_answer="(none)", stale_answer="Dallas",
            stale_prompt="I used to live in Dallas.", current_prompt=None, subject_needles=("dallas", "live"),
        ),
    ]


# ---------------------------------------------------------------------------- current-state detection


# Cues that a question asks for a CURRENT standing value. Kept bench-local (Phase D is an offline eval);
# menhir's own query_intent lens is consulted separately as a cross-reference, not as the gate.
_CURRENT_CUES = ("now", "currently", "current", "these days", "right now", "at the moment", "today", "this week")
# Cues that a question is explicitly about the PAST -> NOT a current-state query (a View must not apply).
_HISTORICAL_CUES = ("used to", "did i", "a while ago", "last year", "last month", "back then",
                    "previously", "before", "originally", "in the past", "once")


def is_current_state_question(question: str) -> bool:
    """True if the question asks for a CURRENT standing value (not a historical one). A historical cue
    always wins (it is the wrongful-suppression tripwire), so 'how many did I own a while ago?' is
    NEVER current-state even though it is about a standing quantity."""
    q = question.strip().lower()
    if any(c in q for c in _HISTORICAL_CUES):
        return False
    return any(c in q for c in _CURRENT_CUES)


# ---------------------------------------------------------------------------- value matching


_FLOAT_TOL = 1e-6


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def value_matches(expected: Any, actual: Any) -> bool:
    """Soft equality across typed representations (numeric tolerance; case-insensitive text)."""
    if expected is None:
        return actual is None
    ef = _to_float(expected)
    if ef is not None:
        af = _to_float(actual)
        return af is not None and abs(af - ef) <= _FLOAT_TOL
    es, as_ = str(expected).strip().lower(), str(actual).strip().lower()
    return bool(es) and (es in as_ or as_ in es)


def _value_in_text(value: Any, snippets: list[str]) -> bool:
    """True if the value appears in any recall snippet. Numbers match as a whole token (so 20 does not
    match inside 2023); text matches case-insensitive substring."""
    fv = _to_float(value)
    blob = " \n ".join(snippets).lower()
    if fv is not None:
        token = str(int(fv)) if fv == int(fv) else str(fv)
        return re.search(rf"(?<!\d){re.escape(token)}(?!\d)", blob) is not None
    return str(value).strip().lower() in blob


# ---------------------------------------------------------------------------- the 7-dimension scorer


@dataclass
class PhaseDQuestionResult:
    """Per-question counterfactual outcome: the seven dimensions + the composed answers."""

    case_id: str
    kind: str
    question: str
    current_state_detected: bool
    subject_resolved: bool
    slot_overlap: bool
    view_status: str  # correct | wrong | absent
    stale_in_baseline: bool
    answer_improved: bool
    wrongful_suppression: bool
    baseline_answer_correct: bool
    view_aware_answer: Any
    baseline_snippets: list[str] = field(default_factory=list)
    notes: str = ""


def _find_view(views: list[dict[str, Any]], case: PhaseDCase) -> dict[str, Any] | None:
    """Current View whose attribute matches the case's slot (attribute is the durable slot identity)."""
    for v in views:
        if str(v.get("ss_attribute") or "") == case.slot_attribute:
            return v
    return None


def score_question(
    case: PhaseDCase,
    baseline_snippets: list[str],
    views: list[dict[str, Any]],
    assertions: list[dict[str, Any]],
) -> PhaseDQuestionResult:
    """Score one question on all seven dimensions and compose the counterfactual View-aware answer.

    `views` rows carry `ss_attribute`, `ss_kind`, `ss_value`/`ss_display`, `subject_uuid`. `assertions`
    are current bound TypedAssertions (subject_uuid/attribute/value). Pure and deterministic.
    """
    detected = is_current_state_question(case.question)

    view = _find_view(views, case)
    slot_overlap = view is not None
    # View authoritative value: ss_value/ss_display carry the real typed value (view_value is a numeric
    # mirror that is 0.0 for string kinds), so prefer ss_value.
    view_value = None
    if view is not None:
        view_value = view.get("ss_value")
        if view_value in (None, ""):
            view_value = view.get("ss_display")

    if view is None:
        view_status = "absent"
    elif value_matches(case.current_answer, view_value):
        view_status = "correct"
    else:
        view_status = "wrong"

    # subject_resolved: an eligible View exists (its subject is bound) OR a bound assertion for this slot.
    subject_resolved = slot_overlap or any(
        str(a.get("attribute") or "") == case.slot_attribute and not a.get("binding_pending")
        for a in assertions
    )

    stale_in_baseline = (
        case.stale_answer is not None and _value_in_text(case.stale_answer, baseline_snippets)
    )
    current_in_baseline = _value_in_text(case.current_answer, baseline_snippets)
    # baseline is "correct" only if it surfaces the current value UNAMBIGUOUSLY (not also the stale one).
    baseline_answer_correct = current_in_baseline and not (
        case.kind == "current" and stale_in_baseline
    )

    # Counterfactual composition: apply the View as authority ONLY for a detected current-state question
    # with an eligible View; otherwise fall back to baseline (never fabricate authority).
    apply_view = detected and slot_overlap and view_status != "absent"
    view_aware_answer = view_value if apply_view else (baseline_snippets[0] if baseline_snippets else None)

    # answer_improved: a current-state question where the View gives the correct current answer AND the
    # baseline did not already give it unambiguously.
    answer_improved = (
        case.kind == "current" and view_status == "correct" and apply_view and not baseline_answer_correct
    )

    # wrongful_suppression: we would apply View authority in a case where the View's current value is NOT
    # the correct answer -- i.e. a HISTORICAL question (correct answer is the stale value) that we wrongly
    # treat as current-state, or a case where applying the View contradicts a still-valid answer.
    if case.kind == "historical":
        # correct answer is the stale value; applying the current View would suppress it.
        wrongful_suppression = apply_view and not value_matches(case.stale_answer, view_value)
    elif case.kind in ("noview", "expired"):
        # no eligible current View should exist (noview: never ingested; expired: "used to" ended it).
        # If a View is (wrongly) present and applied with a non-current value, that is wrongful -- for an
        # expired slot a lingering stale View means the expire did not end the value.
        wrongful_suppression = apply_view and not value_matches(case.current_answer, view_value)
    else:  # current
        # applying the current View is correct; wrongful only if the View is WRONG yet applied as truth.
        wrongful_suppression = apply_view and view_status == "wrong"

    notes = ""
    if case.kind == "historical" and detected:
        notes = "historical question misclassified as current-state"
    elif case.kind == "noview" and slot_overlap:
        notes = "unexpected View for a no-View control"

    return PhaseDQuestionResult(
        case_id=case.case_id, kind=case.kind, question=case.question,
        current_state_detected=detected, subject_resolved=subject_resolved, slot_overlap=slot_overlap,
        view_status=view_status, stale_in_baseline=stale_in_baseline, answer_improved=answer_improved,
        wrongful_suppression=wrongful_suppression, baseline_answer_correct=baseline_answer_correct,
        view_aware_answer=view_aware_answer, baseline_snippets=list(baseline_snippets), notes=notes,
    )


# ---------------------------------------------------------------------------- aggregate metrics


def aggregate_phase_d(results: list[PhaseDQuestionResult]) -> dict[str, Any]:
    """Primary metrics over view_status==correct rows + end-to-end coverage over ALL questions."""
    current = [r for r in results if r.kind == "current"]
    correct_view = [r for r in current if r.view_status == "correct"]

    # PRIMARY: computed only where an available View is CORRECT (missing View != wrong use).
    primary = {
        "correct_view_rows": len(correct_view),
        "answer_improved": sum(r.answer_improved for r in correct_view),
        "answer_improved_rate": (
            round(sum(r.answer_improved for r in correct_view) / len(correct_view), 3)
            if correct_view else None
        ),
        "wrongful_suppression": sum(r.wrongful_suppression for r in correct_view),
    }

    # END-TO-END coverage across ALL questions (incl. view_status != correct and the controls).
    dims = ("current_state_detected", "subject_resolved", "slot_overlap",
            "stale_in_baseline", "answer_improved")
    coverage = {d: sum(getattr(r, d) for r in results) for d in dims}
    coverage["view_correct"] = sum(r.view_status == "correct" for r in results)
    coverage["total_questions"] = len(results)

    # Controls: historical + no-View + expired must never be wrongfully suppressed (for an expired slot
    # that means no lingering stale View).
    controls = [r for r in results if r.kind in ("historical", "noview", "expired")]
    control_violations = [r.case_id for r in controls if r.wrongful_suppression]

    return {
        "primary_over_correct_views": primary,
        "coverage_all_questions": coverage,
        "controls_total": len(controls),
        "control_violations": control_violations,
        "controls_clean": not control_violations,
    }


# ---------------------------------------------------------------------------- client protocols


class _RecallClient(Protocol):
    def record_turn_evidence(self, namespace: str, text: str, **kwargs: Any) -> dict[str, Any]: ...
    def ingest(self, group_id: str, role: str, content: str, **kwargs: Any) -> None: ...
    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]: ...
    def reset(self, group_id: str) -> None: ...


class _ViewReader(Protocol):
    def read_scalar_views(self, namespace: str) -> list[dict[str, Any]]: ...
    def read_assertions(self, namespace: str) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------- ss_value-aware bolt reader


_Q_PD_VIEWS = (
    "MATCH (v:Entity {view_kind: 'scalar_state', group_id: $ns}) "
    "WHERE coalesce(v.view_current, true) "
    "RETURN v.view_subject_uuid AS subject_uuid, v.ss_attribute AS ss_attribute, "
    "v.ss_kind AS ss_kind, v.ss_value AS ss_value, v.ss_display AS ss_display, "
    "v.view_value AS view_value"
)
_Q_PD_ASSERTIONS = (
    "MATCH (a:TypedAssertion) WHERE a.namespace = $ns AND NOT coalesce(a.superseded, false) "
    "RETURN a.subject_uuid AS subject_uuid, a.subject_display AS subject_display, "
    "a.attribute AS attribute, a.value_kind AS value_kind, a.value_json AS value, "
    "coalesce(a.binding_pending, false) AS binding_pending"
)


@dataclass
class PhaseDBoltReader:
    """Read-only bolt reader that returns the AUTHORITATIVE ss_value/ss_display (the shared
    ScalarBoltReader returns only the numeric view_value mirror, which is 0.0 for string kinds)."""

    uri: str
    user: str = "neo4j"
    password: str = "scalarthrowaway"
    database: str = "neo4j"
    _driver: Any = None

    def __post_init__(self) -> None:
        from archolith_bench.harness.scalar_bolt import assert_not_prod
        assert_not_prod(self.uri)

    def _connect(self) -> None:
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def _read(self, query: str, ns: str) -> list[dict[str, Any]]:
        self._connect()
        with self._driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(query, ns=ns)]

    def read_scalar_views(self, namespace: str) -> list[dict[str, Any]]:
        return self._read(_Q_PD_VIEWS, namespace)

    def read_assertions(self, namespace: str) -> list[dict[str, Any]]:
        return self._read(_Q_PD_ASSERTIONS, namespace)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


# ---------------------------------------------------------------------------- driver


@dataclass
class PhaseDResult:
    """Full Phase D counterfactual result."""

    namespace: str
    menhir_url: str
    neo4j_uri: str
    questions: list[PhaseDQuestionResult]
    metrics: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _seed_prompts(cases: list[PhaseDCase]) -> list[str]:
    """Ordered episode bodies: STALE predecessors first, then CURRENT, so the current value supersedes."""
    stale = [c.stale_prompt for c in cases if c.stale_prompt]
    current = [c.current_prompt for c in cases if c.current_prompt]
    return stale + current


def run_phase_d(
    client: _RecallClient,
    bolt: _ViewReader,
    *,
    cases: list[PhaseDCase] | None = None,
    namespace: str,
    recall_limit: int = 10,
    ingest: bool = True,
) -> PhaseDResult:
    """Seed stale+current episodes, then score each question: BASELINE recall (live HTTP) vs a
    counterfactual VIEW-AWARE answer composed from current Views. Offline-friendly (inject fakes).

    `ingest=False` skips seeding (for a namespace already populated, e.g. an offline stub that recognizes
    prompts on read)."""
    cases = cases or phase_d_cases()
    warnings: list[str] = []

    if ingest:
        for body in _seed_prompts(cases):
            try:
                ev = client.record_turn_evidence(namespace, body, turn_key=f"{namespace}:{body[:24]}")
                turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
            except Exception as exc:  # grounding best-effort
                turn_uuid = None
                warnings.append(f"turn-evidence grounding failed: {exc}")
            client.ingest(namespace, "user", body, source="user", turn_evidence_uuid=turn_uuid, wait=True)

    views = bolt.read_scalar_views(namespace)
    assertions = bolt.read_assertions(namespace)

    results: list[PhaseDQuestionResult] = []
    for c in cases:
        try:
            snippets = client.recall(namespace, c.question, recall_limit)
        except Exception as exc:
            snippets = []
            warnings.append(f"{c.case_id}: recall failed: {exc}")
        results.append(score_question(c, snippets, views, assertions))

    metrics = aggregate_phase_d(results)
    return PhaseDResult(
        namespace=namespace, menhir_url=getattr(client, "base_url", ""),
        neo4j_uri=getattr(bolt, "uri", ""), questions=results, metrics=metrics, warnings=warnings,
    )


def phase_d_result_to_dict(result: PhaseDResult) -> dict[str, Any]:
    """JSON-serializable view of a PhaseDResult."""
    return {
        "namespace": result.namespace,
        "menhir_url": result.menhir_url,
        "neo4j_uri": result.neo4j_uri,
        "metrics": result.metrics,
        "questions": [asdict(q) for q in result.questions],
        "warnings": result.warnings,
    }


# ---------------------------------------------------------------------------- offline stub


class StubPhaseDClient:
    """Network-free stand-in implementing BOTH roles (recall client + View reader) for CI smoke.

    Models the CORRECT consumer: on read it exposes the current Views (current value per slot) and a
    baseline recall that returns BOTH the stale and current sentences for a slot (so baseline is
    ambiguous -- the realistic pre-View state the View-aware path is meant to disambiguate). Historical
    and no-View controls behave correctly (no current View wrongly applied)."""

    # slot_attribute -> (ss_kind, current ss_value)
    _VIEWS: dict[str, tuple[str, Any]] = {
        "owned": ("count", "37"),
        "wake_time": ("clock_time", "07:30"),
        "day_off": ("weekday", "Wednesday"),
    }
    # slot_attribute -> baseline snippets a live recall would surface (stale + current, ambiguous)
    _BASELINE: dict[str, list[str]] = {
        "owned": ["A while ago I owned 20 rare coins.", "I own 37 rare coins."],
        "wake_time": ["I used to wake up at 9:00 every morning.", "I wake up at 7:30 every morning."],
        "day_off": ["My day off used to be Monday.", "My day off is Wednesday."],
        "mood": [],  # no-View control: nothing ingested
        # expired control: baseline still surfaces the past residence, but the correct consumer holds
        # NO current residence View (it was expired), so view-aware must NOT answer "Dallas".
        "residence": ["I used to live in Dallas."],
    }

    def __init__(self) -> None:
        self._episodes: dict[str, list[str]] = {}

    # ---- recall-client role ----
    def record_turn_evidence(self, namespace: str, text: str, **_: Any) -> dict[str, Any]:
        return {"turn_id": "stub", "created": True}

    def ingest(self, group_id: str, role: str, content: str, **_: Any) -> None:
        if content:
            self._episodes.setdefault(group_id, []).append(content)

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        q = query.lower()
        for attr, snippets in self._BASELINE.items():
            key = attr.replace("_", " ")
            if (key in q or (attr == "owned" and "coin" in q) or (attr == "day_off" and "day off" in q)
                    or (attr == "residence" and "live" in q)):
                return list(snippets)[:limit]
        return []

    def reset(self, group_id: str) -> None:
        self._episodes.pop(group_id, None)

    def close(self) -> None:
        pass

    # ---- view-reader role ----
    def read_scalar_views(self, namespace: str) -> list[dict[str, Any]]:
        return [
            {"subject_uuid": "stub-self", "ss_attribute": attr, "ss_kind": kind,
             "ss_value": val, "ss_display": val, "view_value": 0.0}
            for attr, (kind, val) in self._VIEWS.items()
        ]

    def read_assertions(self, namespace: str) -> list[dict[str, Any]]:
        return [
            {"subject_uuid": "stub-self", "subject_display": "user", "attribute": attr,
             "value_kind": kind, "value": val, "binding_pending": False}
            for attr, (kind, val) in self._VIEWS.items()
        ]
