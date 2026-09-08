"""D0 — Retrieval Entropy instrument. Deterministic, GPT-free.

MATH / FRAMING (be honest about this — it is NOT Shannon entropy):
This measures how far the memory is from a QUERY-SUFFICIENT STATE — the smallest bundle of
evidence that still lets you answer the question. Inspired by (not claiming the formal property
of) a *minimal sufficient statistic*: the greatest data reduction that preserves all information
needed for the inference. FLOOR = the size/spread of that minimal set (intrinsic complexity of
the answer's evidence); DELIVERED = grow the set by retrieval rank until sufficient, i.e. a
*greedy set-cover* walk. Lower = closer to query-sufficient. See
`menhir-frontier/.agent/plans/aggregation-as-consolidation.md` ("query-sufficient state").

The campaign's binding constraint was the absence of an objective function that wasn't
downstream of an LLM. This measures the ORGANIZATION of memory itself: the minimal-sufficient
evidence footprint for each question, in two columns.

  FLOOR     — the dispersion of the answer's own evidence in the graph, independent of any
              retriever. This is what consolidation compresses (8 scattered facts -> 1 state
              fact). Needs Neo4j + dataset only.
  DELIVERED — how far the current retriever makes you walk before that evidence is reached.
              Needs MENHIR_URL. The gap DELIVERED - FLOOR is retriever inefficiency; FLOOR
              itself is the consolidation target.

Sufficiency (deterministic, dataset-grounded, NOT an LLM "can you answer?"): a set is sufficient
when it touches an entity MENTIONED by a `has_answer` episode. The dataset marks which turns carry
the answer, so gold support is exact — no keyword-matching on numeric answers.

Each column reports a VECTOR, not one number:
  memories · tokens · episodes · sessions · entities · timespan_days
so a pass can say "dropped session-dispersion 3->1" not just "fewer memories".

Modes:  MODE=floor  (Neo4j + dataset)   |   MODE=delivered (+ MENHIR_URL)   |   MODE=both
Env: LME_BOLT, LME_NEO4J_PW, LME_NS_PREFIX, LME_ENTROPY_PER_TYPE, LME_ENTROPY_K, MENHIR_URL,
     LME_ENTROPY_OUT (rows json), LME_ENTROPY_TYPES (comma; default all 6).
"""
import os, re, json, glob, statistics, collections, sys
from neo4j import GraphDatabase

_BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402

try:
    import tiktoken; _ENC = tiktoken.get_encoding("cl100k_base")
    def toks(s): return len(_ENC.encode(s or ""))
except Exception:
    def toks(s): return (len(s or "") + 3) // 4

MODE = os.getenv("MODE", "floor")
BOLT = os.getenv("LME_BOLT", "bolt://localhost:7689")
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
assert_not_prod(BOLT)
PW = os.getenv("LME_NEO4J_PW", "lmedata123")
PREFIX = os.getenv("LME_NS_PREFIX", "lme-")
PER = int(os.getenv("LME_ENTROPY_PER_TYPE", "15"))
K = int(os.getenv("LME_ENTROPY_K", "20"))
MENHIR_URL = os.getenv("MENHIR_URL", "").rstrip("/")
OUT = os.getenv("LME_ENTROPY_OUT", os.path.expanduser("~/lme-entropy-rows.json"))
TYPES = os.getenv("LME_ENTROPY_TYPES",
                  "temporal-reasoning,multi-session,knowledge-update,"
                  "single-session-user,single-session-assistant,single-session-preference").split(",")

_PUNCT = str.maketrans({c: " " for c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})
def _norm(t): return re.sub(r"\s+", " ", (t or "").lower().translate(_PUNCT)).strip()


def _items():
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not cached:
        raise SystemExit("LongMemEval oracle dataset not found in HF cache")
    allit = json.load(open(cached[0], encoding="utf-8"))
    by = collections.defaultdict(list)
    for it in allit:
        by[it.get("question_type")].append(it)
    return [it for t in TYPES for it in by.get(t, [])[:PER]]


def _evidence_prefixes(item):
    """has_answer turns + adjacent assistant reply -> normalized 60-char prefixes."""
    out = []
    for sess in (item.get("haystack_sessions") or []):
        if not isinstance(sess, list):
            continue
        for i, turn in enumerate(sess):
            if isinstance(turn, dict) and turn.get("has_answer"):
                out.append(_norm(turn.get("content", "")))
                if i + 1 < len(sess) and isinstance(sess[i + 1], dict) and sess[i + 1].get("role") == "assistant":
                    out.append(_norm(sess[i + 1].get("content", "")))
    return [c[:60] for c in out if len(c) >= 12]


def _daydiff(dates):
    ds = sorted(d[:10] for d in dates if d)
    if len(ds) < 2:
        return 0
    from datetime import date
    a = date.fromisoformat(ds[0]); b = date.fromisoformat(ds[-1])
    return (b - a).days


def gold_support(driver, ns, prefixes):
    """Deterministic gold support: episodes whose content matches a has_answer prefix, the
    entities they MENTION (gold_entities = sufficiency target), and the episode footprint (floor)."""
    # Match in Python (both sides _norm'd) — Cypher can't replicate _norm's punctuation table,
    # so a Cypher CONTAINS against a punctuation-stripped prefix silently misses on apostrophes.
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (e:Episodic {group_id:$ns})
            OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity)
            RETURN e.uuid AS euuid, e.session_id AS sid, e.content AS content,
                   toString(e.valid_at) AS valid_at, collect(DISTINCT n.uuid) AS ents
            """, ns=ns).data()
    gold_entities, episodes = set(), []
    for r in rows:
        ec = _norm(r["content"])
        if not any(p in ec for p in prefixes):
            continue
        for u in r["ents"]:
            if u:
                gold_entities.add(u)
        episodes.append(r)
    floor = {
        "episodes": len(episodes),
        "sessions": len({r["sid"] for r in episodes if r["sid"]}),
        "tokens": sum(toks(r["content"]) for r in episodes),
        "timespan_days": _daydiff([r["valid_at"] for r in episodes]),
        "entities": len(gold_entities),
    }
    return gold_entities, floor


def entity_provenance(driver, ns, uuids):
    """uuid -> {content_tokens, session_ids, episode_uuids, valid_at} for delivered footprint."""
    if not uuids:
        return {}
    with driver.session() as s:
        rows = s.run(
            """
            MATCH (n:Entity {group_id:$ns}) WHERE n.uuid IN $uuids
            OPTIONAL MATCH (e:Episodic {group_id:$ns})-[:MENTIONS]->(n)
            RETURN n.uuid AS uuid, coalesce(n.summary,n.name,'') AS text,
                   collect(DISTINCT e.uuid) AS eps, collect(DISTINCT e.session_id) AS sids,
                   collect(DISTINCT toString(e.valid_at)) AS vats
            """, ns=ns, uuids=list(uuids)).data()
    return {r["uuid"]: r for r in rows}


def delivered_recall(q, ns):
    import httpx
    r = httpx.post(f"{MENHIR_URL}/api/recall",
                   json={"query": q, "limit": K, "namespace": ns, "include_session": True},
                   timeout=90)
    r.raise_for_status()
    return [str(it.get("uuid") or "") for it in (r.json().get("results") or []) if isinstance(it, dict)]


def _footprint(prov, uuids):
    eps, sids, vats = set(), set(), []
    tk = 0
    for u in uuids:
        p = prov.get(u)
        if not p:
            continue
        tk += toks(p["text"])
        eps |= {e for e in p["eps"] if e}
        sids |= {s for s in p["sids"] if s}
        vats += [v for v in p["vats"] if v]
    return {"memories": len(uuids), "tokens": tk, "episodes": len(eps),
            "sessions": len(sids), "entities": len(uuids), "timespan_days": _daydiff(vats)}


def main():
    items = _items()
    driver = GraphDatabase.driver(BOLT, auth=("neo4j", PW))
    rows = []
    try:
        for it in items:
            qid = str(it["question_id"]); ns = f"{PREFIX}{qid}"; q = it["question"]
            prefixes = _evidence_prefixes(it)
            gold_entities, floor = gold_support(driver, ns, prefixes)
            row = {"qid": qid, "qtype": it["question_type"], "answer": str(it.get("answer", "")),
                   "gold_resolved": bool(gold_entities), "floor": floor}
            if MODE in ("delivered", "both") and MENHIR_URL and gold_entities:
                ranked = delivered_recall(q, ns)
                hit = next((i for i, u in enumerate(ranked, 1) if u in gold_entities), None)
                if hit is None:
                    row["delivered"] = None; row["censored"] = True
                else:
                    prov = entity_provenance(driver, ns, ranked[:hit])
                    fp = _footprint(prov, ranked[:hit]); fp["rank"] = hit
                    row["delivered"] = fp; row["censored"] = False
            rows.append(row)
    finally:
        driver.close()
    json.dump(rows, open(OUT, "w"), indent=2)

    # ---- report ----
    def med(vals): return round(statistics.median(vals), 1) if vals else 0
    resolved = [r for r in rows if r["gold_resolved"]]
    print(f"\n===== RETRIEVAL ENTROPY (mode={MODE}, N={len(rows)}, {len(resolved)} gold-resolved, top-{K}) =====")
    print("\n--- FLOOR: dispersion of the answer's evidence (consolidation target) ---")
    hdr = ("type", "n", "episodes", "sessions", "tokens", "span_d", "entities")
    print("{:26s}{:>4s}{:>10s}{:>9s}{:>8s}{:>8s}{:>9s}".format(*hdr))
    for t in TYPES:
        sub = [r["floor"] for r in resolved if r["qtype"] == t]
        if not sub:
            continue
        print("{:26s}{:>4d}{:>10.1f}{:>9.1f}{:>8.0f}{:>8.0f}{:>9.1f}".format(
            t[:26], len(sub), med([x["episodes"] for x in sub]), med([x["sessions"] for x in sub]),
            med([x["tokens"] for x in sub]), med([x["timespan_days"] for x in sub]),
            med([x["entities"] for x in sub])))

    if MODE in ("delivered", "both") and MENHIR_URL:
        print("\n--- DELIVERED: retriever walk to first gold hit (memories · rank · tokens · %reached) ---")
        print("{:26s}{:>4s}{:>10s}{:>7s}{:>9s}{:>10s}".format("type", "n", "memories", "rank", "tokens", "reached%"))
        for t in TYPES:
            sub = [r for r in resolved if r["qtype"] == t]
            got = [r["delivered"] for r in sub if r.get("delivered")]
            if not sub:
                continue
            reached = 100 * len(got) / len(sub)
            print("{:26s}{:>4d}{:>10.1f}{:>7.1f}{:>9.0f}{:>9.0f}%".format(
                t[:26], len(sub), med([x["memories"] for x in got]), med([x["rank"] for x in got]),
                med([x["tokens"] for x in got]), reached))
    print(f"\n(rows -> {OUT}; re-run before/after a consolidation pass and diff the FLOOR vector)")


if __name__ == "__main__":
    main()
