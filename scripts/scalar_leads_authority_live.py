#!/usr/bin/env python3
"""LIVE measurement of the ADDITIVE authority LEADS path (menhir plan
menhir-observation-nodes-and-view-authority-recall, G14 + Phase 4b).

The Step-7c A/B (`scalar_view_authority_live.py`) measures the old SUPPRESSION model on `:Episodic`
fixtures, where the foundation gate always yields ADVISORY (agent-tier extraction, no declarant
foundation) -- so the LEADS path is never exercised and the wrongful-authority rate only reflects the
advisory arm. This fixture seeds via `:TurnEvidence` with `declarant='user'` so a scalar View can trace
to a real user-declared foundation (the G14 bridge), then MEASURES whether:

  * the View LEADS (structured 7.J verdict or legacy `is_scalar_authority=true` marker) exactly when
    it HAS a user foundation, and
  * nothing leads WITHOUT a foundation (wrongful-authority = 0).

It cross-checks the recall marker against the GRAPH truth over bolt (per current View: does its
`CURRENT_ANCHOR` absolute trace to a `declarant='user'` `:TurnEvidence` via `FOUNDS`?), and prints
diagnostics (TurnEvidence/FOUNDS/anchor counts) so an all-advisory result is explained, not mysterious.

Emits JSON. Exit 2 if no View materialized (cannot conclude).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from archolith_bench.harness.memory_ab import assert_not_production
from archolith_bench.harness.menhir_client import HttpMenhirClient
from archolith_bench.harness.scalar_bolt import assert_not_prod


@dataclass(frozen=True)
class Slot:
    attribute: str
    value: str
    prompt: str
    query: str
    needle: str


SLOTS = [
    Slot("owned", "37", "I own 37 rare coins.", "How many rare coins do I own right now?", "37"),
    Slot("wake_time", "7:30", "I wake up at 7:30 every morning.",
         "What time do I wake up currently?", "7:30"),
]

_Q_VIEWS = (
    "MATCH (v:Entity {view_kind:'scalar_state', group_id:$ns}) WHERE coalesce(v.view_current, true) "
    "RETURN v.ss_attribute AS attribute, v.ss_value AS value"
)
# Per current View: does its head (CURRENT_ANCHOR absolute) trace to a declarant='user' :TurnEvidence
# via FOUNDS? This is exactly what ViewRepository.scalar_view_has_user_foundation checks -- the graph
# truth the recall marker MUST agree with.
_Q_FOUNDATION = (
    "MATCH (v:Entity {view_kind:'scalar_state', group_id:$ns}) WHERE coalesce(v.view_current, true) "
    "OPTIONAL MATCH (v)-[:CURRENT_ANCHOR]->(a:TypedAssertion) "
    "OPTIONAL MATCH (a)<-[:FOUNDS]-(te:TurnEvidence {declarant:'user'}) "
    "RETURN v.ss_attribute AS attribute, count(DISTINCT a) AS anchors, count(DISTINCT te) AS founds"
)
_Q_DIAG_TE = "MATCH (t:TurnEvidence {declarant:'user'}) WHERE t.namespace=$ns RETURN count(t) AS n"
_Q_DIAG_ASSERT = (
    "MATCH (a:TypedAssertion {namespace:$ns}) WHERE NOT coalesce(a.binding_pending, false) "
    "RETURN count(a) AS n"
)
_Q_DIAG_FOUNDS = (
    "MATCH (:TurnEvidence)-[f:FOUNDS]->(a:TypedAssertion {namespace:$ns}) RETURN count(f) AS n"
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

    def views(self, ns): return self._read(_Q_VIEWS, ns)
    def foundations(self, ns): return self._read(_Q_FOUNDATION, ns)
    def scalar1(self, q, ns): return int((self._read(q, ns) or [{"n": 0}])[0]["n"])

    def close(self):
        if self._driver is not None:
            self._driver.close()


def _numeric_value_matches(needle: str, blob: str) -> bool:
    """Match numeric values while accepting canonical clock-time zero padding."""
    if re.fullmatch(r"\d{1,2}:\d{2}", needle):
        hour, minute = needle.split(":", 1)
        target = f"{int(hour)}:{minute}"
        return any(
            f"{int(found_hour)}:{found_minute}" == target
            for found_hour, found_minute in re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", blob)
        )
    return re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", blob) is not None


def _leads_current(needle: str, results: list[dict[str, Any]]) -> bool:
    """A result carrying the current value is marked the authority (is_scalar_authority=true)."""
    for r in results:
        blob = f"{r.get('name','')} {r.get('content','')}".lower()
        n = needle.lower()
        hit = _numeric_value_matches(n, blob) if re.fullmatch(r"[\d.:]+", n) else n in blob
        if hit and bool(r.get("is_scalar_authority")):
            return True
    return False


def _structured_leads_current(
    attribute: str, needle: str, authority_layer: list[dict[str, Any]],
) -> bool:
    """Recognize the Phase 4c/7.J authority verdict without conflating it with ranked results."""
    for verdict in authority_layer:
        if str(verdict.get("kind") or "") != "current":
            continue
        if str(verdict.get("status") or "") != "leads":
            continue
        if str(verdict.get("attribute") or "") != attribute:
            continue
        value = str(verdict.get("value") or "")
        if _numeric_value_matches(needle.lower(), value.lower()):
            return True
    return False


def _measurement_exit_code(
    materialized: bool,
    leads_proven: bool,
    wrongful_authority: int,
) -> int:
    if not materialized:
        return 2
    return 0 if leads_proven and wrongful_authority == 0 else 1


@dataclass
class SlotOutcome:
    attribute: str
    view_ok: bool = False
    has_current_anchor: bool = False
    has_user_foundation: bool = False       # GRAPH truth (bolt)
    leads_in_recall: bool = False           # either supported public authority representation
    flat_authority_marker: bool = False
    structured_authority_verdict: bool = False
    wrongful: bool = False                  # leads WITHOUT a graph foundation (must be False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--menhir-url", required=True)
    ap.add_argument("--neo4j-uri", required=True)
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="scalarthrowaway")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--label", default="leads")
    ap.add_argument("--max-wait-s", type=float, default=180.0)
    ap.add_argument("--settle-s", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Both targets are written to (reset + seed), so both are guarded at
    # the point the arguments resolve, before any client is constructed.
    assert_not_production(args.menhir_url)
    assert_not_prod(args.neo4j_uri)

    ns = args.namespace or f"lead-{int(time.time())}"
    client = HttpMenhirClient(args.menhir_url)
    bolt = Bolt(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    print(f"== authority LEADS live [{args.label}] ==\n  menhir={args.menhir_url} bolt={args.neo4j_uri} "
          f"ns={ns}\n")

    def _ingest(body: str) -> None:
        # declarant defaults to 'user' -> the G14 discovery reads it, grounds the assertion to the
        # :TurnEvidence anchor, and draws FOUNDS (the declarant foundation the gate needs).
        ev = client.record_turn_evidence(ns, body, turn_key=f"{ns}:{body[:24]}")
        turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
        client.ingest(ns, "user", body, source="user", turn_evidence_uuid=turn_uuid, wait=True)

    client.reset(ns)
    for slot in SLOTS:
        _ingest(slot.prompt)

    want = {s.attribute for s in SLOTS}
    start = time.monotonic()
    deadline = start + args.max_wait_s
    last_sig = None
    stable_since = None
    views: list = []
    while time.monotonic() < deadline:
        views = bolt.views(ns)
        got = {str(v["attribute"]) for v in views}
        sig = len(views)
        now_m = time.monotonic()
        if sig != last_sig:
            last_sig = sig
            stable_since = now_m
        if want <= got and stable_since is not None and (now_m - stable_since) >= args.settle_s:
            break
        time.sleep(3.0)
    waited = round(time.monotonic() - start, 1)

    # graph diagnostics -- explain the result rather than leaving it mysterious.
    te_user = bolt.scalar1(_Q_DIAG_TE, ns)
    n_assert = bolt.scalar1(_Q_DIAG_ASSERT, ns)
    founds_edges = bolt.scalar1(_Q_DIAG_FOUNDS, ns)
    found_by_attr = {str(r["attribute"]): r for r in bolt.foundations(ns)}
    view_attrs = {str(v["attribute"]) for v in views}
    print(f"  after {waited}s: views={sorted(view_attrs)}")
    print(f"  DIAG: turn_evidence(user)={te_user}  bound_assertions={n_assert}  FOUNDS_edges={founds_edges}\n")

    outcomes: list[SlotOutcome] = []
    for slot in SLOTS:
        f = found_by_attr.get(slot.attribute, {})
        oc = SlotOutcome(attribute=slot.attribute)
        oc.view_ok = slot.attribute in view_attrs
        oc.has_current_anchor = int(f.get("anchors", 0)) > 0
        oc.has_user_foundation = int(f.get("founds", 0)) > 0
        data = client.recall_raw(ns, slot.query, 10)
        results = data.get("results", []) if isinstance(data, dict) else []
        authority_layer = data.get("authority_layer", []) if isinstance(data, dict) else []
        oc.flat_authority_marker = _leads_current(slot.needle, results)
        oc.structured_authority_verdict = _structured_leads_current(
            slot.attribute, slot.needle, authority_layer)
        oc.leads_in_recall = oc.flat_authority_marker or oc.structured_authority_verdict
        oc.wrongful = oc.leads_in_recall and not oc.has_user_foundation
        outcomes.append(oc)

    print(f"{'slot':12} {'view':5} {'anchor':7} {'user_found':11} {'leads':6} {'wrongful':8}")
    print("-" * 60)
    y = lambda b: "yes" if b else "-"  # noqa: E731
    for oc in outcomes:
        print(f"{oc.attribute:12} {y(oc.view_ok):5} {y(oc.has_current_anchor):7} "
              f"{y(oc.has_user_foundation):11} {y(oc.leads_in_recall):6} {y(oc.wrongful):8}")

    testable = [oc for oc in outcomes if oc.view_ok]
    materialized = want <= view_attrs
    founded = [oc for oc in testable if oc.has_user_foundation]
    # LEADS proven: at least one founded slot, and EVERY founded slot LEADS in recall; and NO slot leads
    # without a foundation (wrongful-authority = 0).
    leads_proven = bool(founded) and all(oc.leads_in_recall for oc in founded)
    wrongful_authority = sum(1 for oc in testable if oc.wrongful)
    result = {
        "label": args.label,
        "namespace": ns,
        "materialized": materialized,
        "turn_evidence_user": te_user,
        "founds_edges": founds_edges,
        "founded_slots": [oc.attribute for oc in founded],
        "leads_proven": leads_proven,
        "wrongful_authority": wrongful_authority,
        "slots": [oc.__dict__ for oc in outcomes],
    }
    print(f"\n  founded_slots={result['founded_slots']}  leads_proven={leads_proven}  "
          f"wrongful_authority={wrongful_authority}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  wrote {args.out}")
    bolt.close()
    sys.exit(_measurement_exit_code(materialized, leads_proven, wrongful_authority))


if __name__ == "__main__":
    main()
