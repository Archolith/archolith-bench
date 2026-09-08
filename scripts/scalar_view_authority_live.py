#!/usr/bin/env python3
"""Step 7c -- LIVE production-recall A/B for current-state View authority suppression.

Unlike Phase D (which composes the View-aware answer OFFLINE in the harness), this drives the
ACTUAL menhir recall endpoint, so it exercises the production suppression hook gated by
MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED. It seeds a current->current REPLACEMENT in
TWO SEPARATE consolidation passes (stale value first, wait for its View, then the current value):
a dateless observation's valid_at is the perception batch's `learned_fallback` time, so seeding
both in one pass gives them the SAME valid_at and the fold abstains (AMBIGUOUS_ANCHOR) -- retiring
the slot's View. Two passes give the current value a strictly-later valid_at, so the fold picks it
(LWW) while the stale absolute survives as the older, different-valued predecessor the
provenance-linked gate can suppress. It then waits for a current View AND that older predecessor to
materialize, and runs live recall for:

  * a CURRENT-STATE query  ("...do I own right now?")   -- the stale value MUST disappear when the
                                                           authority flag is ON (suppressed).
  * a HISTORICAL query     ("...did I used to own?")     -- the stale value MUST remain (advisory);
                                                           this proves the stale episode is still
                                                           recallable, so its absence above is
                                                           suppression, not unrecallability.

Emits JSON so a caller can diff a flag-ON run against a flag-OFF run (the OFF run must surface the
stale value in BOTH queries). Exit 2 if the graph never materialized (cannot conclude).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from archolith_bench.harness.memory_ab import assert_not_production
from archolith_bench.harness.menhir_client import HttpMenhirClient
from archolith_bench.harness.scalar_bolt import assert_not_prod


@dataclass(frozen=True)
class Slot:
    attribute: str        # the ss_attribute the View materializes
    stale_value: str      # older value (should be suppressed on current-state)
    current_value: str    # newer value (the current View)
    stale_prompt: str
    current_prompt: str
    current_query: str
    historical_query: str
    needle_stale: str     # token to look for in recall snippets (the stale value)
    needle_current: str


SLOTS = [
    Slot(
        attribute="owned", stale_value="20", current_value="37",
        stale_prompt="I own 20 rare coins.", current_prompt="I own 37 rare coins.",
        current_query="How many rare coins do I own right now?",
        historical_query="How many rare coins did I used to own?",
        needle_stale="20", needle_current="37",
    ),
    Slot(
        attribute="wake_time", stale_value="9:00", current_value="7:30",
        stale_prompt="I wake up at 9:00 every morning.",
        current_prompt="I wake up at 7:30 every morning.",
        current_query="What time do I wake up currently?",
        historical_query="What time did I used to wake up?",
        needle_stale="9:00", needle_current="7:30",
    ),
]


# --------------------------------------------------------------------------- bolt probes
_Q_VIEWS = (
    "MATCH (v:Entity {view_kind: 'scalar_state', group_id: $ns}) "
    "WHERE coalesce(v.view_current, true) "
    "RETURN v.ss_attribute AS attribute, v.ss_value AS value, v.ss_display AS display"
)
# The gate's real precondition (cross-episode replacement makes NO SUPERSEDES edge): a slot with a
# current View AND >=2 DISTINCT grounded absolute values -> an older non-selected value survives as a
# suppressible fact for the same slot.
_Q_SUPERSEDED = (
    "MATCH (v:Entity {view_kind: 'scalar_state', group_id: $ns}) "
    "WHERE coalesce(v.view_current, true) "
    "MATCH (a:TypedAssertion) "
    "WHERE a.namespace = $ns AND NOT coalesce(a.binding_pending, false) "
    "AND a.operation = 'absolute' AND a.attribute = v.ss_attribute AND a.value_kind = v.ss_kind "
    "WITH v.ss_attribute AS attribute, collect(DISTINCT a.value_json) AS vals "
    "WHERE size(vals) >= 2 "
    "RETURN attribute, vals AS values"
)


@dataclass
class Bolt:
    uri: str
    user: str
    password: str
    _driver: Any = None

    def _read(self, q: str, ns: str) -> list[dict[str, Any]]:
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        with self._driver.session(database="neo4j") as s:
            return [dict(r) for r in s.run(q, ns=ns)]

    def views(self, ns: str) -> list[dict[str, Any]]:
        return self._read(_Q_VIEWS, ns)

    def superseded(self, ns: str) -> list[dict[str, Any]]:
        return self._read(_Q_SUPERSEDED, ns)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()


def _present(needle: str, snippets: list[str]) -> bool:
    """Whole-token match so a numeric needle like 20 does not match inside a date (2026)."""
    import re
    blob = " \n ".join(snippets).lower()
    n = needle.lower()
    if re.fullmatch(r"[\d.:]+", n):   # numeric / clock value -> require non-digit boundaries
        return re.search(rf"(?<!\d){re.escape(n)}(?!\d)", blob) is not None
    return n in blob


@dataclass
class SlotOutcome:
    attribute: str
    view_ok: bool
    predecessor_ok: bool
    current_query_stale_present: bool
    current_query_current_present: bool
    historical_query_stale_present: bool
    # the pass condition for a flag-ON run: stale gone on current-state, retained on historical
    suppressed_on_current: bool = False
    retained_on_historical: bool = False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--menhir-url", required=True)
    ap.add_argument("--neo4j-uri", required=True)
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="scalarthrowaway")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--label", default="run", help="tag for this run (e.g. flag-on / flag-off)")
    ap.add_argument("--max-wait-s", type=float, default=180.0)
    ap.add_argument("--settle-s", type=float, default=10.0)
    ap.add_argument("--no-ingest", action="store_true", help="graph already seeded; only recall")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Both targets are written to (reset + seed), so both are guarded at
    # the point the arguments resolve, before any client is constructed.
    assert_not_production(args.menhir_url)
    assert_not_prod(args.neo4j_uri)

    ns = args.namespace or f"va-{int(time.time())}"
    client = HttpMenhirClient(args.menhir_url)
    bolt = Bolt(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    print(f"== Step 7c live authority A/B [{args.label}] ==\n  menhir={args.menhir_url} "
          f"bolt={args.neo4j_uri} ns={ns}\n")

    def _ingest(body: str) -> None:
        ev = client.record_turn_evidence(ns, body, turn_key=f"{ns}:{body[:24]}")
        turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
        client.ingest(ns, "user", body, source="user", turn_evidence_uuid=turn_uuid, wait=True)

    def _wait_for_view_attrs(want_attrs: set[str], budget_s: float) -> set[str]:
        end = time.monotonic() + budget_s
        got: set[str] = set()
        while time.monotonic() < end:
            got = {str(v["attribute"]) for v in bolt.views(ns)}
            if want_attrs <= got:
                return got
            time.sleep(3.0)
        return got

    want = {s.attribute for s in SLOTS}
    if not args.no_ingest:
        client.reset(ns)
        # TWO-PASS SEED (see module docstring). A dateless observation's valid_at is the perception
        # batch's `learned_fallback` time; seeding both absolutes in one pass ties their valid_at and
        # the fold abstains (AMBIGUOUS_ANCHOR), retiring the slot's View. So: (pass 1) seed every
        # STALE value and WAIT for its View to materialize -- proving pass 1 consolidated on its own
        # `learned_at` (T1); then (pass 2) seed every CURRENT value in a fresh pass (`learned_at` T2 >
        # T1). The fold now picks the current value (latest valid_at) and the stale absolute survives
        # as the older, different-valued predecessor the gate can suppress.
        for slot in SLOTS:
            _ingest(slot.stale_prompt)
        stale_views = _wait_for_view_attrs(want, args.max_wait_s / 2)
        print(f"  pass1 (stale) seeded {len(SLOTS)}; stale views materialized={sorted(stale_views)}")
        for slot in SLOTS:
            _ingest(slot.current_prompt)
        print(f"  pass2 (current) seeded {len(SLOTS)} replacements")

    # wait for BOTH a current View and a superseded predecessor per slot, then settle.
    start = time.monotonic()
    deadline = start + args.max_wait_s
    last_sig = None
    stable_since = None
    views: list = []
    supers: list = []
    while time.monotonic() < deadline:
        views = bolt.views(ns)
        supers = bolt.superseded(ns)
        view_attrs = {str(v["attribute"]) for v in views}
        sup_attrs = {str(s["attribute"]) for s in supers}
        sig = (len(views), len(supers))
        now_m = time.monotonic()
        if sig != last_sig:
            last_sig = sig
            stable_since = now_m
        ready = want <= view_attrs and want <= sup_attrs
        settled = stable_since is not None and (now_m - stable_since) >= args.settle_s
        if ready and settled:
            break
        time.sleep(3.0)
    waited = round(time.monotonic() - start, 1)
    view_attrs = {str(v["attribute"]) for v in views}
    sup_attrs = {str(s["attribute"]) for s in supers}
    print(f"  after {waited}s: views={sorted(view_attrs)} superseded_predecessors={sorted(sup_attrs)}\n")

    outcomes: list[SlotOutcome] = []
    for slot in SLOTS:
        cur_snips = client.recall(ns, slot.current_query, 10)
        hist_snips = client.recall(ns, slot.historical_query, 10)
        oc = SlotOutcome(
            attribute=slot.attribute,
            view_ok=slot.attribute in view_attrs,
            predecessor_ok=slot.attribute in sup_attrs,
            current_query_stale_present=_present(slot.needle_stale, cur_snips),
            current_query_current_present=_present(slot.needle_current, cur_snips),
            historical_query_stale_present=_present(slot.needle_stale, hist_snips),
        )
        oc.suppressed_on_current = oc.predecessor_ok and not oc.current_query_stale_present
        oc.retained_on_historical = oc.historical_query_stale_present
        outcomes.append(oc)

    print(f"{'slot':12} {'view':5} {'pred':5} {'cur:stale':10} {'cur:now':8} {'hist:stale':10}")
    print("-" * 60)
    for oc in outcomes:
        y = lambda b: "yes" if b else "-"  # noqa: E731
        print(f"{oc.attribute:12} {y(oc.view_ok):5} {y(oc.predecessor_ok):5} "
              f"{y(oc.current_query_stale_present):10} {y(oc.current_query_current_present):8} "
              f"{y(oc.historical_query_stale_present):10}")

    # A slot only exercises the gate once its precondition materialized (a current View AND an older
    # distinct-valued sibling). Consolidation is stochastic, so score per slot over those that did.
    testable = [oc for oc in outcomes if oc.view_ok and oc.predecessor_ok]
    materialized = len(testable) >= 1
    result = {
        "label": args.label,
        "namespace": ns,
        "materialized": materialized,
        "testable_slots": [oc.attribute for oc in testable],
        "slots": [oc.__dict__ for oc in outcomes],
        # flag-ON pass: at least one slot exercised the gate, and EVERY testable slot suppressed the
        # stale value on the current-state query while retaining it on the historical query.
        "flag_on_pass": materialized and all(
            oc.suppressed_on_current and oc.retained_on_historical for oc in testable),
    }
    print(f"\n  testable_slots={result['testable_slots']}  materialized={materialized}  "
          f"flag_on_pass={result['flag_on_pass']}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.out}")
    bolt.close()
    sys.exit(0 if materialized else 2)


if __name__ == "__main__":
    main()
