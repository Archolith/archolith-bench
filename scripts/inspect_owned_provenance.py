#!/usr/bin/env python3
"""Ad-hoc: trace why the `owned` slot returns row_count=0 in Step-7 View suppression.

Compares, for the stale value memory recall surfaces, the MENTIONS-episode identity vs the
typed-assertion episode_uuid, and the assertion subject_uuid vs the current View view_subject_uuid.
"""
from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402

URI = sys.argv[1] if len(sys.argv) > 1 else "bolt://localhost:7690"
NS = sys.argv[2]
PW = sys.argv[3] if len(sys.argv) > 3 else "scalarthrowaway"

assert_not_prod(URI)
drv = GraphDatabase.driver(URI, auth=("neo4j", PW))


def q(cypher: str, **kw):
    with drv.session(database="neo4j") as s:
        return [dict(r) for r in s.run(cypher, ns=NS, **kw)]


print(f"== ns={NS} ==\n")

print("--- scalar_state Views (current + retired) ---")
for r in q(
    "MATCH (v:Entity {view_kind:'scalar_state', group_id:$ns}) "
    "RETURN v.ss_attribute AS attr, v.ss_value AS val, v.view_current AS current, "
    "v.view_subject_uuid AS subj, v.uuid AS uuid ORDER BY attr, current DESC"
):
    print(f"  {r}")

print("\n--- TypedAssertions (owned slot) ---")
for r in q(
    "MATCH (a:TypedAssertion {namespace:$ns, attribute:'owned'}) "
    "RETURN a.value_json AS val, a.subject_uuid AS subj, a.episode_uuid AS epi, "
    "a.superseded AS superseded, a.binding_pending AS pending, toString(a.valid_at) AS valid_at "
    "ORDER BY a.valid_at"
):
    print(f"  {r}")

print("\n--- Value :Entity memories mentioning 'coin' (recall candidates) ---")
for r in q(
    "MATCH (n:Entity {group_id:$ns}) WHERE toLower(n.name) CONTAINS 'coin' "
    "OPTIONAL MATCH (epi:Episodic)-[:MENTIONS]->(n) "
    "RETURN n.uuid AS uuid, n.name AS name, collect(DISTINCT epi.uuid) AS mention_episodes"
):
    print(f"  {r}")

print("\n--- Episodics in ns (uuid -> body) ---")
for r in q(
    "MATCH (e:Episodic {group_id:$ns}) RETURN e.uuid AS uuid, "
    "coalesce(e.content, e.name, e.episode_body, '') AS body ORDER BY e.uuid"
):
    b = str(r["body"])[:60]
    print(f"  {r['uuid']}  {b!r}")

print("\n--- BRIDGE probe: stale '20' entity -> MENTIONS episode -> owned assertion ---")
for r in q(
    "MATCH (n:Entity {group_id:$ns}) WHERE toLower(n.name) CONTAINS 'coin' "
    "MATCH (epi:Episodic)-[:MENTIONS]->(n) "
    "OPTIONAL MATCH (c:TypedAssertion {namespace:$ns, attribute:'owned', operation:'absolute'}) "
    "WHERE c.episode_uuid = epi.uuid "
    "RETURN n.name AS cand, epi.uuid AS mention_epi, c.value_json AS assertion_val, "
    "c.subject_uuid AS assertion_subj, c.episode_uuid AS assertion_epi"
):
    print(f"  {r}")

drv.close()
