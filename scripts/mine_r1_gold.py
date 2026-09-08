"""Mine an R1 gold answer set from the DUMMY graph (a clone of the live graph).

Target: $R1_DUMMY_BOLT (default bolt://localhost:7690). Must be the SAME graph
`run_r1_dummy.py` scores against -- gold is mined here and mapped back by node
uuid. The clone cannot live on 7687: the guard cannot distinguish it there from
a real graph, so that port is reserved.

Why: the only R1 fixture (`fixtures/r1_demo.json`) saturates at recall=1.0, so the
ladder can never graduate and `hybrid_alpha` stays unset. The dummy is a full clone of
the live menhir graph (23k+ real Entity nodes with `structure_project`/`symbol_kind`/
`symbol_signature`), so we can derive GOLD LABELS mechanically for the three families
the R1 win gate actually needs, with real recall headroom:

  symbol_name_query   DESCRIBE a distinctive class (LLM paraphrase of its body, no
                      identifier overlap) so the source-aware floor must rescue the gold;
                      gold = that one node (globally unique name) -> drives `symbol_recall`
                      with real headroom. Raw-identifier fallback when no LLM client / the
                      body is too thin (saturates -> exempt under the recalibrated gate);
                      the `vehicle` field records which was used. (Needs a client, i.e.
                      run with --paraphrase; the de-CamelCased spacing vehicle is removed.)
  exact_error_string  query a distinctive underscore identifier (function/method name)
                      VERBATIM; gold = that node -> drives `exact_string_recall` (saturates
                      -- a floor/no-regression guard, exempt from the gate's must-beat).
  wrong_repo_same_topic  a symbol name that exists in EXACTLY two projects; query = a
                      paraphrase of project A's node body (identifier removed, so the scope
                      warden -- not lexical match -- must pick A), gold = the project-A node,
                      the project-B node is a real wrong-scope distractor -> drives
                      `wrong_scope_injection_rate`. Raw-identifier fallback as above.

  paraphrased_debug_question  (opt-in, --paraphrase N, needs OPENAI_API_KEY) an LLM rewrites a
                      node's own summary into a natural question that SHARES NO identifier
                      words with the node -> lexically distant, semantically dead-on. gold =
                      that uuid; target_symbol = its name (so a successful semantic-gap rescue
                      moves `symbol_recall`, where the dummy has real headroom). This is the
                      family that most directly exercises R1's source-aware floor; the
                      symbol/scope families now share this same paraphrase vehicle (with a
                      raw-identifier fallback), so they get semantic-gap headroom too, while
                      exact stays verbatim as a saturating floor guard. See
                      `.agent/benchmark-notes/r1-dummy-gold-run.md`.

NOT mined (honest scope): `stale_semantic_neighbor` / `historical_only_vs_current_truth`
(the clone's `conflict_status` has no `superseded`/historical marker, only
`false_positive`/`resolved`) and `buried_relevant_memory` (needs an LLM judge). Those stay
with the `_eval_frontier.py` judge path or a hand-authored fixture (drop hand-authored
queries into the same JSON — the runner scores them identically). So every mined memory
carries `stale=false`; `stale_hit_rate` is therefore 0 for all conditions and the win gate's
"no stale regression" check is trivially satisfied — documented, not hidden.

Memory ids ARE the node uuids, so the read-mode runner (`run_r1_dummy.py`) maps recall
results straight back to gold with no seed-episode grounding step.

READ-ONLY graph access. Opens one driver against the dummy and only runs MATCH ... RETURN.
Never writes, never touches prod (prod is a different bolt host in menhir/.env). With
--paraphrase it additionally makes N one-shot LLM calls (gpt-4.1-nano) to author questions.

Run:  python scripts/mine_r1_gold.py [--out fixtures/r1_dummy_gold.json]
                                     [--symbols 40] [--exact 30] [--scope 25] [--paraphrase 40]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Shares R1_DUMMY_BOLT with run_r1_dummy.py by design: gold mined here is
# scored against that same graph by node uuid, so the pair must never diverge.
# See the note there on why the clone cannot live on 7687.
DUMMY_URI = os.environ.get("R1_DUMMY_BOLT", "bolt://localhost:7690")
DUMMY_AUTH = ("neo4j", os.environ.get("R1_DUMMY_PW", "menhirdummy123"))
REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ENV = REPO_ROOT / ".env"

# Names too generic to be a fair retrieval target (ambiguous / ubiquitous).
_STOP_NAMES = {
    "main", "run", "setup", "teardown", "handler", "wrapper", "init", "close",
    "start", "stop", "execute", "process", "build", "create", "update", "delete",
}


def _is_identifierish(name: str) -> bool:
    """A single code-token name (no spaces / punctuation beyond _.), not a sentence."""
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


_MIN_PARAPHRASE_BODY = 50  # a body shorter than this is too thin to paraphrase faithfully


def _describe_query(name: str, body: str, client) -> tuple[str, str]:
    """Query text + vehicle for a symbol / scope family.

    Primary vehicle (client present, body >= _MIN_PARAPHRASE_BODY): an LLM paraphrase
    of the node body sharing no identifier words -- lexically distant, so the
    source-aware floor (symbol_name_query) or the scope warden (wrong_repo_same_topic)
    must do the work and the gold keeps real recall headroom. Fallback (no client /
    thin body / leaked paraphrase): the raw identifier -- the baseline saturates on it,
    which the recalibrated win gate now exempts (a floor guard, not a headroom metric).
    The de-CamelCased spacing vehicle is REMOVED: r1-dummy-gold-run.md showed it strips
    the lexical signal AND leaves the single gold node unretrievable in a 23.8k graph.
    Returns ``(query_text, vehicle)`` -- vehicle is ``paraphrase`` or ``identifier``.
    """
    if client is not None and body and len(body) >= _MIN_PARAPHRASE_BODY:
        q = _paraphrase(client, name, body)
        if q:
            return q, "paraphrase"
    return name, "identifier"


def _load_openai_key() -> str:
    """OPENAI_API_KEY from the environment, falling back to Bench's .env."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key and BENCH_ENV.exists():
        from dotenv import dotenv_values

        key = dotenv_values(str(BENCH_ENV)).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("--paraphrase needs OPENAI_API_KEY (env or archolith-bench/.env)")
    return key


_PARAPHRASE_SYS = (
    "You write ONE short natural-language question a developer would type to find the code "
    "or note described below. HARD RULE: do not reuse any identifier, class, function, file, "
    "or module name from the input — describe what it DOES in plain words. Output only the "
    "question, no quotes, no preamble."
)


def _paraphrase(client, name: str, body: str) -> str | None:
    """LLM-rewrite a node's summary into a lexically-distant, semantically-faithful question."""
    resp = client.chat.completions.create(
        model="gpt-4.1-nano", temperature=0,
        messages=[
            {"role": "system", "content": _PARAPHRASE_SYS},
            {"role": "user", "content": f"Name: {name}\nDescription: {body[:500]}"},
        ],
    )
    q = (resp.choices[0].message.content or "").strip().strip('"')
    # Reject if the model leaked the identifier (lexical distance is the whole point).
    if not q or name.lower() in q.lower():
        return None
    return q


def mine(session, n_symbols: int, n_exact: int, n_scope: int, n_paraphrase: int = 0, client=None) -> dict:
    memories: dict[str, dict] = {}
    queries: list[dict] = []

    def add_memory(uuid, name, project, text, *, symbols=None, exact_strings=None):
        mem = memories.setdefault(
            uuid,
            {
                "id": uuid, "text": text or name or "", "repo": None,
                "project": project, "symbols": [], "exact_strings": [], "stale": False,
            },
        )
        for s in symbols or []:
            if s not in mem["symbols"]:
                mem["symbols"].append(s)
        for s in exact_strings or []:
            if s not in mem["exact_strings"]:
                mem["exact_strings"].append(s)

    # --- symbol_name_query: globally-unique CLASS names ---
    rows = session.run(
        """
        MATCH (n:Entity)
        WHERE n.symbol_kind = 'class' AND n.structure_project IS NOT NULL
              AND n.name IS NOT NULL AND size(n.name) >= 8
        WITH n.name AS nm, collect(n) AS ns
        WHERE size(ns) = 1
        WITH ns[0] AS n
        RETURN n.uuid AS uuid, n.name AS name, n.structure_project AS project,
               coalesce(n.symbol_signature, n.summary, n.content, '') AS body
        ORDER BY n.uuid
        """
    )
    taken = 0
    for r in rows:
        name = r["name"]
        if taken >= n_symbols or not _is_identifierish(name) or name.lower() in _STOP_NAMES:
            continue
        uuid = r["uuid"]
        text = f"{name}\n{r['body']}"
        add_memory(uuid, name, r["project"], text, symbols=[name])
        q_text, vehicle = _describe_query(name, r["body"], client)
        queries.append({
            "id": f"sym_{taken:03d}", "text": q_text,
            "family": "symbol_name_query", "support_ids": [uuid],
            "intent": "current", "project": r["project"], "target_symbol": name,
            "vehicle": vehicle,
        })
        taken += 1

    # --- exact_error_string: globally-unique underscore FUNCTION/METHOD identifiers ---
    rows = session.run(
        """
        MATCH (n:Entity)
        WHERE n.symbol_kind IN ['function', 'method'] AND n.structure_project IS NOT NULL
              AND n.name IS NOT NULL AND n.name CONTAINS '_' AND size(n.name) >= 10
        WITH n.name AS nm, collect(n) AS ns
        WHERE size(ns) = 1
        WITH ns[0] AS n
        RETURN n.uuid AS uuid, n.name AS name, n.structure_project AS project,
               coalesce(n.symbol_signature, n.summary, n.content, '') AS body
        ORDER BY n.uuid
        """
    )
    taken = 0
    for r in rows:
        name = r["name"]
        if taken >= n_exact or not _is_identifierish(name) or name.lower() in _STOP_NAMES:
            continue
        uuid = r["uuid"]
        text = f"{name}\n{r['body']}"
        # The literal identifier must appear verbatim in the gold text (it does: it's the name).
        add_memory(uuid, name, r["project"], text, exact_strings=[name])
        queries.append({
            "id": f"exact_{taken:03d}", "text": name,
            "family": "exact_error_string", "support_ids": [uuid],
            "intent": "current", "project": r["project"], "target_exact_string": name,
        })
        taken += 1

    # --- wrong_repo_same_topic: a symbol name in EXACTLY two projects ---
    rows = session.run(
        """
        MATCH (n:Entity)
        WHERE n.symbol_kind IS NOT NULL AND n.structure_project IS NOT NULL
              AND n.name IS NOT NULL AND size(n.name) >= 8
        WITH n.name AS nm, collect(DISTINCT n.structure_project) AS projs, collect(n) AS ns
        WHERE size(projs) = 2 AND size(ns) = 2
        RETURN nm AS name, projs AS projects,
               [x IN ns | {uuid: x.uuid, project: x.structure_project,
                           body: coalesce(x.symbol_signature, x.summary, x.content, '')}] AS nodes
        ORDER BY nm
        """
    )
    taken = 0
    for r in rows:
        name = r["name"]
        if taken >= n_scope or not _is_identifierish(name) or name.lower() in _STOP_NAMES:
            continue
        nodes = r["nodes"]
        if len(nodes) != 2:
            continue
        a, b = nodes[0], nodes[1]
        # Query scoped to project A; A's node is gold, B's node is the wrong-scope distractor.
        add_memory(a["uuid"], name, a["project"], f"{name}\n{a['body']}", symbols=[name])
        add_memory(b["uuid"], name, b["project"], f"{name}\n{b['body']}", symbols=[name])
        q_text, vehicle = _describe_query(name, a["body"], client)
        queries.append({
            "id": f"scope_{taken:03d}", "text": q_text,
            "family": "wrong_repo_same_topic", "support_ids": [a["uuid"]],
            "intent": "current", "project": a["project"], "vehicle": vehicle,
            "note": f"wrong-scope distractor in {b['project']}",
        })
        taken += 1

    # --- paraphrased_debug_question: LLM rewrites a node's summary into a distant question ---
    if n_paraphrase > 0 and client is not None:
        rows = session.run(
            """
            MATCH (n:Entity)
            WHERE n.symbol_kind IN ['function', 'method', 'class']
                  AND n.structure_project IS NOT NULL AND n.name IS NOT NULL
                  AND size(n.name) >= 8 AND n.summary IS NOT NULL AND size(n.summary) >= 50
                  AND NOT n.summary STARTS WITH 'def ' AND NOT n.summary STARTS WITH 'class '
                  AND NOT n.summary STARTS WITH 'async def '
            WITH n.name AS nm, collect(n) AS ns
            WHERE size(ns) = 1
            WITH ns[0] AS n
            RETURN n.uuid AS uuid, n.name AS name, n.structure_project AS project, n.summary AS body
            ORDER BY n.uuid
            """
        )
        taken = 0
        for r in rows:
            if taken >= n_paraphrase:
                break
            name = r["name"]
            if not _is_identifierish(name) or name.lower() in _STOP_NAMES:
                continue
            q_text = _paraphrase(client, name, r["body"])
            if not q_text:
                continue
            uuid = r["uuid"]
            add_memory(uuid, name, r["project"], f"{name}\n{r['body']}", symbols=[name])
            queries.append({
                "id": f"para_{taken:03d}", "text": q_text,
                "family": "paraphrased_debug_question", "support_ids": [uuid],
                "intent": "current", "project": r["project"], "target_symbol": name,
                "note": f"paraphrase of {name}'s summary; no identifier overlap",
            })
            taken += 1

    return {
        "name": "r1_dummy_gold",
        "description": (
            "Auto-mined R1 gold answer set from the dummy (clone of the live graph). "
            "Families: symbol_name_query, exact_error_string, wrong_repo_same_topic, "
            "paraphrased_debug_question (LLM, semantic-gap). stale/historical/buried NOT "
            "covered (see mine_r1_gold.py docstring)."
        ),
        "memories": list(memories.values()),
        "queries": queries,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mine an R1 gold answer set from the dummy graph.")
    p.add_argument("--out", default="fixtures/r1_dummy_gold.json")
    p.add_argument("--symbols", type=int, default=40)
    p.add_argument("--exact", type=int, default=30)
    p.add_argument("--scope", type=int, default=25)
    p.add_argument("--paraphrase", type=int, default=0,
                   help="N LLM-authored semantic-gap queries (needs OPENAI_API_KEY)")
    args = p.parse_args(argv)

    client = None
    if args.paraphrase > 0:
        from openai import OpenAI

        client = OpenAI(api_key=_load_openai_key())

    from neo4j import GraphDatabase

    from archolith_bench.harness.scalar_bolt import assert_not_prod

    assert_not_prod(DUMMY_URI)
    driver = GraphDatabase.driver(DUMMY_URI, auth=DUMMY_AUTH)
    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()  # fail fast if dummy is down
            fixture = mine(session, args.symbols, args.exact, args.scope, args.paraphrase, client)
    finally:
        driver.close()

    fam_counts: dict[str, int] = {}
    for q in fixture["queries"]:
        fam_counts[q["family"]] = fam_counts.get(q["family"], 0) + 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    print(f"mined {len(fixture['queries'])} queries over {len(fixture['memories'])} memories")
    for fam, c in sorted(fam_counts.items()):
        print(f"  {fam:24s} {c}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
