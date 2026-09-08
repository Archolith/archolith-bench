"""Arm C — the honest end-to-end write: gated perception -> fold -> counter Views IN THE GRAPH.

The capstone measurement's write phase. Arm A wrote ORACLE Views (gold answers, perfect perception);
the tuning sweeps scored what the gate WOULD commit with no graph. This runs the real thing:
`menhir.services.perception.perceive_and_fold` (k-sample gpt-4o-mini extractor, conjunctive veto-gate
at the precision-conservative threshold, holistic cross-check ON) over each `lme-<qid>` namespace's
USER episodes from the LIVE benchmark graph, committing counter Views via `ViewRepository` — real
MENTIONS provenance to real graph episodes, real embedded surfaces. Abstains are no-ops by design.

Run AFTER a baseline `entropy.sh both` and BEFORE the post-write one; the DELIVERED diff on the
counting slice is the honest collapse number, and the diff on the HELD-OUT slice is the Goodhart
guard (a true-but-irrelevant committed View must not push gold out of reach).

Cleanup (all writes carry source='perception-capstone'):
  MATCH (n:Entity {source:'perception-capstone'}) DETACH DELETE n

Env: LME_BOLT (bolt://localhost:7689), LME_NEO4J_PW, PC_MODEL=gpt-4o-mini, PC_TEMP=0.7, PC_K=5,
     PC_THRESHOLD=1.0, PC_COUNT_LIMIT=14, PC_HELDOUT_LIMIT=12, PC_OUT (~/perception-write.json).
     OPENAI_API_KEY else archolith-bench/.env. STOP on 429 per protocol.
"""

from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "menhir-frontier", "src")))
sys.path.insert(0, _HERE)

from menhir.infrastructure.view_repository import ViewRepository  # noqa: E402
from menhir.services.perception import Episode, perceive_and_fold  # noqa: E402

import entropy  # noqa: E402  (dataset loader — same slice as the sweeps)
import perception_tune as pt  # noqa: E402  (key loading, LLM/embed seams, slice filter, 429 stop)

_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402

BOLT = os.getenv("LME_BOLT", "bolt://localhost:7689")
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
assert_not_prod(BOLT)
PW = os.getenv("LME_NEO4J_PW", "lmedata123")
MODEL = os.getenv("PC_MODEL", "gpt-4o-mini")
TEMP = float(os.getenv("PC_TEMP", "0.7"))
K = int(os.getenv("PC_K", "5"))
THRESHOLD = float(os.getenv("PC_THRESHOLD", "1.0"))
COUNT_LIMIT = int(os.getenv("PC_COUNT_LIMIT", "14"))
HELDOUT_LIMIT = int(os.getenv("PC_HELDOUT_LIMIT", "12"))
OUT = os.getenv("PC_OUT", os.path.expanduser("~/perception-write.json"))
SOURCE = os.getenv("PC_SOURCE", "perception-capstone")
SCOPE = os.getenv("PC_SCOPE", "counting+heldout")  # or "all" (the full stratified sample)


class _Neo4jShim:
    """The minimal `.execute(query, params) -> rows` contract ViewRepository needs."""

    def __init__(self, driver):
        self._driver = driver

    def execute(self, query: str, params: dict | None = None):
        with self._driver.session() as s:
            return s.run(query, **(params or {})).data()


def _graph_user_episodes(driver, ns: str) -> list[Episode]:
    """USER-turn episodes from the live graph (real uuids -> real MENTIONS provenance), content
    date-grounded from valid_at like the sweep's dataset episodes."""
    with driver.session() as s:
        rows = s.run(
            "MATCH (e:Episodic {group_id:$ns}) WHERE e.content STARTS WITH 'user:' "
            "RETURN e.uuid AS uuid, toString(e.valid_at) AS v, e.content AS c ORDER BY e.valid_at",
            ns=ns).data()
    out = []
    for r in rows:
        body = r["c"][len("user:"):].strip()[:2000]
        if len(body) < 8:
            continue
        out.append(Episode(uuid=r["uuid"], content=f"[{(r['v'] or '')[:10]}] {body}"))
    return out


def main():
    from neo4j import GraphDatabase
    from openai import OpenAI

    client = OpenAI(api_key=pt._load_key())
    llm = pt.make_llm(client)
    embed = pt.make_embed(client)
    driver = GraphDatabase.driver(BOLT, auth=("neo4j", PW))
    views = ViewRepository(_Neo4jShim(driver))

    items = entropy._items()
    if SCOPE == "all":
        # the full stratified eval sample — every namespace, tagged by whether it's a count question
        groups = [("all", items)]
    else:
        counting = [it for it in items if pt._is_count_answer(it["answer"])][:COUNT_LIMIT]
        heldout = [it for it in items if not pt._is_count_answer(it["answer"])][:HELDOUT_LIMIT]
        groups = [("counting", counting), ("heldout", heldout)]
    n_total = sum(len(g) for _, g in groups)
    print(f"perception consolidation: scope={SCOPE} ({n_total} namespaces); "
          f"k={K}, threshold={THRESHOLD}, cross-check ON, model={MODEL}, source={SOURCE}")

    results, t0 = [], time.time()
    try:
        for tag, group in groups:
            for it in group:
                qid = str(it["question_id"]); ns = f"lme-{qid}"
                eps = _graph_user_episodes(driver, ns)
                res = perceive_and_fold(
                    episodes=eps, llm_complete=llm, graph_adapter=views,
                    k=K, threshold=THRESHOLD, namespace=ns, source=SOURCE,
                    embed=embed, enable_cross_check=True,
                )
                committed = [{"view_key": r.get("view_key"), "value": r.get("value"),
                              "uuid": r.get("uuid"), "reducer": r.get("reducer"),
                              "agreement": r.get("agreement"), "triangulated": r.get("triangulated")}
                             for r in res.committed]
                results.append({"qid": qid, "tag": tag, "answer": str(it.get("answer")),
                                "n_eps": len(eps), "committed": committed,
                                "abstained": [{"measure": d.measure, "reason": d.reason}
                                              for d in res.abstained]})
                mark = f"WROTE {len(committed)}" if committed else "abstained-all"
                print(f"  [{tag:8s}] {qid:14s} gold={str(it.get('answer'))[:10]:10s} {mark} "
                      f"{[ (c['value']) for c in committed]}")
    finally:
        driver.close()

    json.dump(results, open(OUT, "w"), indent=2)
    n_views = sum(len(r["committed"]) for r in results)
    print(f"\nArm C done in {time.time()-t0:.0f}s: {n_views} Views written "
          f"(source={SOURCE}) across {sum(1 for r in results if r['committed'])} namespaces -> {OUT}")
    print(f"cleanup: MATCH (n:Entity {{source:'{SOURCE}'}}) DETACH DELETE n")


if __name__ == "__main__":
    main()
