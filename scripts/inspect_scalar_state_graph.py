"""Diagnostic: dump the scalar-state graph over bolt to explain unbound perception.

Read-only. Auto-discovers the matching namespace(s) and prints, for the binding question:
  * user :Episodic count + bodies (is the producer shape right? content STARTS WITH 'user:')
  * resolved :Entity nodes in the namespace (did Graphiti resolve entities to bind to?)
  * :TypedAssertion rows in full (what perception extracted; binding_pending; subject)
  * :TypedAssertionHead + :ScalarConsolidationWatermark state
  * current scalar_state Views

Usage: python scripts/inspect_scalar_state_graph.py [bolt_uri] [password]
                                                   [--ns-prefix PREFIX] [--all] [--quiet-episodes]
Defaults: bolt://localhost:7691  scalarthrowaway  --ns-prefix scalar-e2e

--ns-prefix retargets the discovery scan, so the same instrument works on any corpus that names
its namespaces by prefix (e.g. --ns-prefix lme- for the LongMemEval builds). --all inspects every
matching namespace instead of only the largest one -- required when the question is "which of my N
namespaces failed to bind", not "why did this one namespace fail". --quiet-episodes drops the
per-episode body dump, which is unreadable past a couple hundred episodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("uri", nargs="?", default="bolt://localhost:7691")
    p.add_argument("password", nargs="?", default="scalarthrowaway")
    p.add_argument("--ns-prefix", default="scalar-e2e")
    p.add_argument("--all", action="store_true", help="inspect every matching namespace")
    p.add_argument("--quiet-episodes", action="store_true", help="skip the per-episode body dump")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    URI, PW = args.uri, args.password
    assert_not_prod(URI)
    driver = GraphDatabase.driver(URI, auth=("neo4j", PW))
    with driver.session(database="neo4j") as s:
        # discover the namespace(s)
        ns_rows = s.run(
            "MATCH (e:Episodic) WHERE e.group_id STARTS WITH $prefix "
            "RETURN e.group_id AS ns, count(*) AS episodes ORDER BY episodes DESC",
            prefix=args.ns_prefix,
        ).data()
        print(f"== namespaces ({URI}, prefix={args.ns_prefix!r}) ==")
        for r in ns_rows:
            print(f"  {r['ns']}: {r['episodes']} episodes")
        if not ns_rows:
            print("  (none found -- did the --keep run finish and preserve the namespace?)")
            # still show any global scalar nodes
            for label in ("TypedAssertion", "TypedAssertionHead", "ScalarConsolidationWatermark"):
                n = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
                print(f"  global {label}: {n}")
            return
        targets = [r["ns"] for r in ns_rows] if args.all else [ns_rows[0]["ns"]]
        for ns in targets:
            _inspect_namespace(s, ns, quiet_episodes=args.quiet_episodes)
    driver.close()


def _inspect_namespace(s, ns: str, *, quiet_episodes: bool = False) -> None:
        print(f"\n== inspecting namespace: {ns} ==")

        # 1. FOUR-CHECKPOINT bundle per episode:
        #    (a) Graphiti processed/completed (control), (b) linked NON-View KG entities (the exact
        #    binder candidate set), so "no entity" can be told apart from "never processed".
        eps = s.run(
            "MATCH (e:Episodic {group_id:$ns}) "
            "OPTIONAL MATCH (e)-[]-(n:Entity) "
            "  WHERE NOT coalesce(n.is_view,false) AND NOT coalesce(n.is_quantstate,false) "
            "        AND n.view_kind IS NULL "
            "WITH e, collect(DISTINCT {uuid:n.uuid, name:n.name}) AS linked "
            "RETURN e.uuid AS uuid, e.content AS content, "
            "       (e.content STARTS WITH 'user:') AS is_user, "
            "       toString(e.processing_completed_at) AS completed_at, "
            "       e.processing_steps_completed AS steps_done, e.processing_steps_total AS steps_total, "
            "       [x IN linked WHERE x.uuid IS NOT NULL] AS linked "
            "ORDER BY e.created_at",
            ns=ns,
        ).data()
        unlinked = sum(1 for e in eps if not e["linked"])
        print(f"\n-- Episodic + Graphiti processing + linked KG entities ({len(eps)}; "
              f"{unlinked} with NO bindable entity) --")
        if quiet_episodes:
            print("  (per-episode dump suppressed by --quiet-episodes)")
        else:
            for e in eps:
                done = "COMPLETED" if e["completed_at"] else "NOT-COMPLETED"
                linked = e["linked"]
                names = [x["name"] for x in linked]
                print(f"  [{done} {e['steps_done']}/{e['steps_total']}] {e['content']!r}")
                print(f"       linked non-View entities: {names or '(none)'}")

        # 2. all entities (binding-target census; flag the plain KG ones)
        ents = s.run(
            "MATCH (n:Entity {group_id:$ns}) "
            "RETURN n.name AS name, n.view_kind AS view_kind, "
            "       (n.view_kind IS NULL AND NOT coalesce(n.is_view,false) "
            "        AND NOT coalesce(n.is_quantstate,false)) AS is_plain_kg "
            "ORDER BY is_plain_kg DESC, n.name",
            ns=ns,
        ).data()
        plain = [n for n in ents if n["is_plain_kg"]]
        print(f"\n-- Entity nodes ({len(ents)}; plain KG binding targets: {len(plain)}) --")
        for n in ents:
            tag = " <-- PLAIN KG (bindable)" if n["is_plain_kg"] else f" view_kind={n['view_kind']}"
            print(f"  {n['name']!r}{tag}")

        # 3. TypedAssertion rows in full
        tas = s.run(
            "MATCH (a:TypedAssertion) WHERE a.namespace=$ns OR a.group_id=$ns "
            "RETURN a.subject_uuid AS subject_uuid, a.subject_display AS subject_display, "
            "       a.attribute AS attribute, a.value_kind AS value_kind, a.value AS value, "
            "       a.evidence_tier AS tier, coalesce(a.binding_pending,false) AS pending, "
            "       coalesce(a.superseded,false) AS superseded, a.source_key AS source_key "
            "ORDER BY a.attribute",
            ns=ns,
        ).data()
        print(f"\n-- TypedAssertion ({len(tas)}) --")
        for a in tas:
            print(f"  attr={a['attribute']!r} kind={a['value_kind']} value={a['value']!r} "
                  f"pending={a['pending']} superseded={a['superseded']} tier={a['tier']}")
            print(f"       subject_uuid={a['subject_uuid']!r} subject_display={a['subject_display']!r} "
                  f"source_key={a['source_key']!r}")

        # 4. heads + watermark
        heads = s.run(
            "MATCH (h:TypedAssertionHead) WHERE h.namespace=$ns OR h.group_id=$ns RETURN count(h) AS c",
            ns=ns,
        ).single()["c"]
        wm = s.run(
            "MATCH (w:ScalarConsolidationWatermark {group_id:$ns}) "
            "RETURN w.perceiver_version AS pv, toString(w.cursor_at) AS cursor_at, "
            "       w.cursor_uuid AS cursor_uuid",
            ns=ns,
        ).data()
        print(f"\n-- TypedAssertionHead: {heads} --")
        print(f"-- ScalarConsolidationWatermark: {wm} --")

        # 5. current scalar_state Views
        views = s.run(
            "MATCH (v:Entity {view_kind:'scalar_state'}) "
            "WHERE (v.group_id=$ns) AND coalesce(v.view_current,true) "
            "RETURN v.ss_attribute AS attr, v.ss_kind AS kind, v.view_value AS value, "
            "       v.view_subject_uuid AS subject",
            ns=ns,
        ).data()
        print(f"\n-- current scalar_state Views ({len(views)}) --")
        for v in views:
            print(f"  attr={v['attr']!r} kind={v['kind']} value={v['value']!r} subject={v['subject']!r}")


if __name__ == "__main__":
    main()
