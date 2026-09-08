"""Bounded root-cause pass: capture entity->episode MENTIONS provenance AFTER EACH ingest.

Isolates the first incorrect provenance boundary behind the scalar-state binding failure. Ingests the
three Alice statements ONE AT A TIME (wait=true, so enrichment/entity-linking completes per call) and,
after EACH call, snapshots over bolt:
  * the Episodic UUID created for the submitted content (distinct identity? -> bucket 1)
  * the NON-View entities linked to THAT episode (the exact binder candidate set:
    (:Episodic)-[]-(:Entity) minus view nodes)
  * ALL episodes currently linked to `Alice` (does the link land on the source episode, or move?)
  * the full plain-KG entity census so far

Classification (per the handoff contract):
  - one reused Episodic identity across calls        -> Menhir/harness episode-keying bug
  - distinct episodes, links appear only after ep3   -> batching/delayed/final-call attribution
  - links correct then move/disappear later          -> Graphiti consolidation/update behavior
  - ep1/ep2 entities attributed to ep3 as context    -> Menhir<->Graphiti context/provenance contract

Read-only except for the three ingests it performs into a fresh throwaway namespace. Also verifies the
clock_time scalar View's ss_value/ss_display (view_value=0.0 is an expected numeric mirror, not a bug).

Usage: diagnose_mentions_provenance.py --menhir-url URL --neo4j-uri BOLT [--neo4j-password PW]
"""

from __future__ import annotations

import argparse
import time
import uuid

from neo4j import GraphDatabase

from archolith_bench.harness import HttpMenhirClient
from archolith_bench.harness.memory_ab import assert_not_production
from archolith_bench.harness.scalar_bolt import assert_not_prod

STATEMENTS = [
    "Alice owns 37 coins.",
    "Alice has read 12 books.",
    "Alice wakes up at 7:30 AM.",
]

_NONVIEW = (
    "NOT coalesce(n.is_view,false) AND NOT coalesce(n.is_quantstate,false) AND n.view_kind IS NULL"
)


def snapshot(session, ns: str, submitted: str) -> None:
    """Print the per-call provenance bundle after one ingest."""
    eps = session.run(
        "MATCH (e:Episodic {group_id:$ns}) "
        "RETURN e.uuid AS uuid, e.content AS content, toString(e.created_at) AS created "
        "ORDER BY e.created_at",
        ns=ns,
    ).data()
    print(f"    Episodic nodes now: {len(eps)}")
    for e in eps:
        marker = "  <== just submitted" if e["content"].endswith(submitted) else ""
        print(f"      {e['uuid']}  {e['content']!r}{marker}")

    # entities linked to the just-submitted episode (undirected, mirrors fetch_linked_entities)
    this = session.run(
        "MATCH (e:Episodic {group_id:$ns}) WHERE e.content ENDS WITH $s "
        f"OPTIONAL MATCH (e)-[r]-(n:Entity) WHERE {_NONVIEW} "
        "RETURN e.uuid AS euuid, collect(DISTINCT {name:n.name, rel:type(r)}) AS linked",
        ns=ns, s=submitted,
    ).data()
    for row in this:
        names = [x for x in row["linked"] if x["name"]]
        print(f"    linked non-View entities on THIS episode ({row['euuid']}): {names or '(none)'}")

    # every episode currently linked to Alice
    alice = session.run(
        "MATCH (e:Episodic {group_id:$ns})-[r]-(n:Entity) WHERE toLower(coalesce(n.name,''))='alice' "
        f"AND {_NONVIEW} "
        "RETURN DISTINCT substring(e.content,0,45) AS ep, type(r) AS rel ORDER BY ep",
        ns=ns,
    ).data()
    print(f"    episodes linked to 'Alice': {[(a['ep'], a['rel']) for a in alice] or '(none)'}")

    kg = session.run(
        f"MATCH (n:Entity {{group_id:$ns}}) WHERE {_NONVIEW} RETURN n.name AS name ORDER BY n.name",
        ns=ns,
    ).data()
    print(f"    plain-KG entity census: {[k['name'] for k in kg] or '(none)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menhir-url", required=True)
    ap.add_argument("--neo4j-uri", required=True)
    ap.add_argument("--neo4j-password", default="scalarthrowaway")
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--clock-wait-s", type=float, default=90.0)
    args = ap.parse_args()

    assert_not_prod(args.neo4j_uri)
    assert_not_production(args.menhir_url)
    ns = f"mentions-diag-{uuid.uuid4().hex[:10]}"
    print(f"== MENTIONS provenance diagnostic ==\n  menhir={args.menhir_url}  bolt={args.neo4j_uri}  ns={ns}\n")

    client = HttpMenhirClient(args.menhir_url)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        client.reset(ns)
        for i, stmt in enumerate(STATEMENTS, 1):
            ev = client.record_turn_evidence(ns, stmt, turn_key=f"{ns}:ep{i}")
            turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
            # per-call args at the /api/memory boundary (menhir generates the Episodic UUID +
            # passes name/body/group_id/reference_time/source to Graphiti; NO previous_episodes).
            print(f"--- after ingest call {i}: {stmt!r} ---")
            print(f"    submitted: role=user source=user group_id={ns} grounded_turn={turn_uuid}")
            client.ingest(ns, "user", stmt, source="user", turn_evidence_uuid=turn_uuid, wait=True)
            with driver.session(database="neo4j") as s:
                snapshot(s, ns, stmt)
            print()

        # clock_time verification: view_value=0.0 is the expected numeric mirror; the real value
        # lives in ss_value/ss_display. Wait briefly for the scalar scheduler to materialize it.
        print("--- clock_time View verification (ss_value/ss_display vs view_value) ---")
        deadline = time.monotonic() + args.clock_wait_s
        rows: list = []
        while time.monotonic() < deadline:
            with driver.session(database="neo4j") as s:
                rows = s.run(
                    "MATCH (v:Entity {group_id:$ns, view_kind:'scalar_state'}) "
                    "WHERE coalesce(v.view_current,true) AND v.ss_kind='clock_time' "
                    "RETURN v.ss_kind AS ss_kind, v.ss_value AS ss_value, v.ss_display AS ss_display, "
                    "       v.view_value AS view_value",
                    ns=ns,
                ).data()
            if rows:
                break
            time.sleep(5)
        if rows:
            for r in rows:
                print(f"    ss_kind={r['ss_kind']!r} ss_value={r['ss_value']!r} "
                      f"ss_display={r['ss_display']!r} view_value={r['view_value']!r}")
        else:
            print(f"    (no clock_time scalar_state View within {args.clock_wait_s}s)")
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
