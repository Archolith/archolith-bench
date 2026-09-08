"""Backfill valid_at (world-time) on LME episodes + fact edges from the real session dates.

WHY: menhir currently does NOT backdate a fresh benchmark ingest — an episode ingested with
occurred_at=<session date> still comes out valid_at=now() (verified live 2026-07-02). So a
freshly built LME graph has fake temporal grounding: every episode/edge stamped with the build
date, which breaks temporal-reasoning and the BriefBuilder Timeline. Until the ingest path is
fixed in menhir, this repairs an already-built graph WITHOUT a ~1-day re-ingest.

The correct date is fully recoverable: episode.session_id -> dataset haystack date;
edge.episodes[0] -> that source episode. Only valid_at (world-time) is touched; created_at/
expired_at (belief-time = ingestion) are correct as-is. Edges are backfilled ONLY where they
carry the ingestion default (edge valid_at day == source episode's current valid_at day),
preserving genuine LLM-extracted dates.

Writes a revert snapshot (uuid -> old valid_at) before mutating. DRY_RUN=1 to preview.

Env: LME_BOLT (bolt uri), LME_NEO4J_PW, LME_NS_PREFIX, LME_REVERT_SNAPSHOT, DRY_RUN.
"""
import glob, json, os, sys
from datetime import datetime, timezone
from neo4j import GraphDatabase

_BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402

BOLT = os.getenv("LME_BOLT", "bolt://localhost:7689")
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
assert_not_prod(BOLT)
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
PW = os.getenv("LME_NEO4J_PW", "lmedata123")
PREFIX = os.getenv("LME_NS_PREFIX", "lme-")
DRY = os.getenv("DRY_RUN", "0") == "1"
SNAP = os.getenv("LME_REVERT_SNAPSHOT",
                 os.path.expanduser("~/lme-date-backfill-revert.json"))


def parse_lme_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def day(v):
    return (str(v) or "")[:10]


def main():
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not cached:
        raise SystemExit("LongMemEval oracle dataset not found in HF cache")
    items = json.load(open(cached[0], encoding="utf-8"))
    sid2date = {}
    for it in items:
        sids, dates = it.get("haystack_session_ids") or [], it.get("haystack_dates") or []
        for i, sid in enumerate(sids):
            if i < len(dates):
                d = parse_lme_date(dates[i])
                if d:
                    sid2date[sid] = d

    drv = GraphDatabase.driver(BOLT, auth=("neo4j", PW))
    revert = {"episodes": {}, "edges": {}}
    ep_updates, edge_updates = [], []
    with drv.session() as s:
        eps = s.run(f"MATCH (e:Episodic) WHERE e.group_id STARTS WITH '{PREFIX}' "
                    "RETURN e.uuid AS uuid, e.session_id AS sid, toString(e.valid_at) AS valid_at").data()
        ep_old_day = {}
        unmappable = 0
        for r in eps:
            correct = sid2date.get(r["sid"])
            ep_old_day[r["uuid"]] = day(r["valid_at"])
            if not correct:
                # This repair keys off e.session_id, which ONLY menhir-written :Episodic nodes
                # carry -- graphiti writes its own EpisodicNodes with no session_id, so they were
                # skipped in silence. On the 2026-07-22 scalar-ku corpus that was 1760 of 2862
                # episodes (62%), and the build reported success anyway. Never fail quiet again.
                unmappable += 1
                continue
            if day(r["valid_at"]) != day(correct):
                revert["episodes"][r["uuid"]] = r["valid_at"]
                ep_updates.append({"uuid": r["uuid"], "valid_at": correct})

        ep_sid = {r["uuid"]: r["sid"] for r in eps}
        edges = s.run(f"MATCH ()-[r:RELATES_TO]->() WHERE r.group_id STARTS WITH '{PREFIX}' "
                      "RETURN r.uuid AS uuid, toString(r.valid_at) AS valid_at, r.episodes AS eps").data()
        skipped_no_ep = skipped_extracted = 0
        for r in edges:
            eps_list = r["eps"] or []
            src = eps_list[0] if eps_list else None
            if not src or src not in ep_sid:
                skipped_no_ep += 1
                continue
            correct = sid2date.get(ep_sid[src])
            if not correct:
                continue
            if day(r["valid_at"]) == ep_old_day.get(src) and day(r["valid_at"]) != day(correct):
                revert["edges"][r["uuid"]] = r["valid_at"]
                edge_updates.append({"uuid": r["uuid"], "valid_at": correct})
            else:
                skipped_extracted += 1

        print(f"episodes: {len(eps)} total, {len(ep_updates)} to backfill, "
              f"{unmappable} UNMAPPABLE (no session_id -> date)")
        if unmappable:
            share = 100.0 * unmappable / max(1, len(eps))
            print(f"  !! WARNING: {share:.0f}% of episodes cannot be date-repaired by this script.",
                  "\n  !! These are graphiti-written EpisodicNodes (no session_id). If menhir's",
                  "\n  !! ingest backdating is working, their valid_at is already correct and this",
                  "\n  !! is benign. If it is NOT, this build has fake temporal grounding and the",
                  "\n  !! script cannot tell you which -- verify valid_at against the fixture.")
        print(f"edges: {len(edges)} total, {len(edge_updates)} to backfill, "
              f"{skipped_extracted} kept (extracted/non-default), {skipped_no_ep} no source-episode")
        if DRY:
            print("DRY_RUN — no writes.")
            print("  sample episode:", ep_updates[:2])
            print("  sample edge:", edge_updates[:2])
            drv.close()
            return

        json.dump(revert, open(SNAP, "w"), indent=2)
        print(f"revert snapshot -> {SNAP}")
        for i in range(0, len(ep_updates), 1000):
            s.run("UNWIND $rows AS row MATCH (e:Episodic {uuid: row.uuid}) "
                  "SET e.valid_at = datetime(row.valid_at)", rows=ep_updates[i:i + 1000])
        for i in range(0, len(edge_updates), 1000):
            s.run("UNWIND $rows AS row MATCH ()-[r:RELATES_TO {uuid: row.uuid}]->() "
                  "SET r.valid_at = datetime(row.valid_at)", rows=edge_updates[i:i + 1000])
        print("backfill applied.")
    drv.close()


main()
