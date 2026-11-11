#!/usr/bin/env python
"""Label recall candidates as answer-bearing from LME ``has_answer`` turns. No LLM, read-only.

Plan: menhir-recall-real-similarity-test-plan-2026-11-05, Labels section. Oracle haystacks equal
the answer sessions, so labels are per turn. Reads the batch JSONL from Menhir's
``scripts/recall_lab_batch.py``, the dataset, and a throwaway copy of the graph:

    python candidate_labels.py --batch out.jsonl --dataset o500.json \
        --neo4j-uri bolt://127.0.0.1:18342 --output labels.jsonl

``--stats-only`` skips the batch and prints whole-graph label coverage instead. Point this at the
unmerged graph even for the merged-namespace distractor run: labels join on uuid, and hubs are
per original namespace.

Batch ids: a real or abstention query uses the question id. A swapped null uses
``<question_id>@<namespace>``, the question asked and the namespace searched. Turn keys embed the
namespace, so a swapped row cannot match gold by accident.

Labels are ``None`` when the candidate cannot be traced at all, so an unknown never counts as a
negative. ``label_route`` names the link that decided the row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HUB_SHARE = 0.20
PREFIX_CHARS = 60
MIN_PREFIX_CHARS = 12
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
SHARED_PORTS = frozenset({7687, 7688, 7689})

# Tried in order; the first route reaching gold names the row.
ENTITY_ROUTES = ("receipt", "queue_self", "retention")
VIEW_ROUTES = ("turn_mention", "assertion", "view_episode")
EPISODE_ROUTES = ("queue_self", "receipt")

_PUNCT = str.maketrans({c: " " for c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})


def turn_key(namespace: str, session_id: str, index: int) -> str:
    """The stored ``TurnEvidence.turn_key`` for one dataset turn."""
    key = f"{namespace}|{namespace}-{session_id}|t{index}"
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8", "replace"))
    digest.update(b"\x00")
    digest.update(key.encode("utf-8", "replace"))
    return digest.hexdigest()


def normalize(text: str | None) -> str:
    """Same normalization as ``entropy.py``."""
    return re.sub(r"\s+", " ", (text or "").lower().translate(_PUNCT)).strip()


@dataclass(frozen=True)
class Gold:
    question_id: str
    namespace: str
    question_type: str
    keys: frozenset[str]
    prefixes: tuple[str, ...]


def load_gold(items: Iterable[dict[str, Any]]) -> dict[str, Gold]:
    """Question id to its ``has_answer`` turns. Prefixes cover those turns only."""
    gold: dict[str, Gold] = {}
    for item in items:
        qid = str(item["question_id"])
        namespace = f"lme-{qid}"
        keys: set[str] = set()
        prefixes: list[str] = []
        for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            for index, turn in enumerate(session):
                if not turn.get("has_answer"):
                    continue
                keys.add(turn_key(namespace, session_id, index))
                text = normalize(turn.get("content"))
                if len(text) >= MIN_PREFIX_CHARS:
                    prefixes.append(text[:PREFIX_CHARS])
        gold[qid] = Gold(qid, namespace, str(item.get("question_type")), frozenset(keys),
                         tuple(prefixes))
    return gold


def question_of(row_id: str) -> str:
    """The question a batch row asked."""
    return row_id.split("@", 1)[0]


def query_kind(row_id: str, gold: Gold | None) -> str:
    if "@" in row_id:
        return "swapped"
    if "_abs" in row_id:
        return "abstention"
    return "real" if gold and gold.keys else "no_gold"


@dataclass
class GraphFacts:
    """Everything the labels need, read once. Plain data so tests can build it by hand."""

    turn_keys: dict[str, str] = field(default_factory=dict)  # TurnEvidence.turn_id -> turn_key
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)  # uuid -> ns, name, kind
    mentions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    retention: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    turn_mentions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    assertion_turns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    view_episodes: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    receipt_turns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    self_turns: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    episodes: dict[str, dict[str, Any]] = field(default_factory=dict)  # uuid -> ns, content
    observations: dict[str, set[str]] = field(default_factory=dict)  # assertion_id -> turn_ids

    def keys(self, turn_ids: Iterable[str]) -> set[str]:
        return {self.turn_keys[t] for t in turn_ids if t in self.turn_keys}

    def episode_turns(self, episode_uuids: Iterable[str], table: dict[str, set[str]]) -> set[str]:
        found: set[str] = set()
        for uuid in episode_uuids:
            found |= self.keys(table.get(uuid, ()))
        return found


def evidence(facts: GraphFacts, uuid: str) -> tuple[str, dict[str, set[str]]]:
    """A node's kind and its turn keys per route. Plain ``dict.get`` keeps the tables unmutated."""
    entity = facts.entities.get(uuid)
    if uuid in facts.observations:
        return "observation", {"founds": facts.keys(facts.observations[uuid])}
    if entity is None:
        if uuid not in facts.episodes:
            return "unknown", {}
        return "source_memory", {
            "queue_self": facts.keys(facts.self_turns.get(uuid, ())),
            "receipt": facts.keys(facts.receipt_turns.get(uuid, ())),
        }
    if entity["kind"] == "view":
        episodes = facts.view_episodes.get(uuid, ())
        return "view", {
            "turn_mention": facts.keys(facts.turn_mentions.get(uuid, ())),
            "assertion": facts.keys(facts.assertion_turns.get(uuid, ())),
            "view_episode": facts.episode_turns(episodes, facts.receipt_turns)
            | facts.episode_turns(episodes, facts.self_turns),
        }
    mentioned = facts.mentions.get(uuid, ())
    retained = facts.retention.get(uuid, ())
    return "entity", {
        "receipt": facts.episode_turns(mentioned, facts.receipt_turns),
        "queue_self": facts.episode_turns(mentioned, facts.self_turns),
        "retention": facts.episode_turns(retained, facts.receipt_turns)
        | facts.episode_turns(retained, facts.self_turns),
    }


def hub_entities(facts: GraphFacts) -> set[str]:
    """Plain entities MENTIONed by more than 20% of their namespace's mentioning episodes."""
    per_namespace: dict[str, set[str]] = defaultdict(set)
    for uuid, entity in facts.entities.items():
        if entity["kind"] == "entity":
            per_namespace[entity["namespace"]] |= facts.mentions.get(uuid, set())
    return {
        uuid
        for uuid, entity in facts.entities.items()
        if entity["kind"] == "entity"
        and facts.mentions.get(uuid)
        and len(facts.mentions[uuid]) / len(per_namespace[entity["namespace"]]) > HUB_SHARE
    }


def _fallback_texts(facts: GraphFacts, uuid: str, kind: str) -> list[str]:
    if kind == "source_memory":
        linked: Iterable[str] = (uuid,)
    else:
        linked = facts.mentions.get(uuid, set()) | facts.view_episodes.get(uuid, set())
    return [normalize(facts.episodes[e].get("content")) for e in linked if e in facts.episodes]


def _namespace(facts: GraphFacts, uuid: str, kind: str) -> str | None:
    table = facts.episodes if kind == "source_memory" else facts.entities
    return (table.get(uuid) or {}).get("namespace")


def label(facts: GraphFacts, hubs: set[str], uuid: str, gold: Gold | None) -> dict[str, Any]:
    """Label one node for one question. ``label_*`` is None when no link or prefix decides it."""
    kind, routes = evidence(facts, uuid)
    is_hub = uuid in hubs
    order = {"entity": ENTITY_ROUTES, "view": VIEW_ROUTES, "observation": ("founds",)}.get(
        kind, EPISODE_ROUTES
    )
    linked = [route for route in order if routes.get(route)]
    gold_keys = gold.keys if gold else frozenset()

    if linked:
        reached = [route for route in linked if routes[route] & gold_keys]
        lenient: bool | None = bool(reached)
        route = (reached or linked)[0]
    elif (
        gold
        and _namespace(facts, uuid, kind) == gold.namespace
        and any(p in text for text in _fallback_texts(facts, uuid, kind) for p in gold.prefixes)
    ):
        lenient, route = True, "prefix"
    else:
        lenient, route = None, "none"

    strict = None if lenient is None else lenient and not is_hub
    return {"kind": kind, "label_strict": strict, "label_lenient": lenient, "label_route": route,
            "is_hub": is_hub}


def is_clean(row: dict[str, Any]) -> bool:
    return not (row.get("error") or row.get("name_cosine_warning"))


def load_batch(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Clean batch rows, last one per (id, arm), and how many failed pairs never recovered.

    The batch appends, so a retried pair has an earlier failed row.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            pair = (row["id"], row["arm"])
            if is_clean(row) or pair not in latest or not is_clean(latest[pair]):
                latest[pair] = row
    clean = [row for row in latest.values() if is_clean(row)]
    return clean, len(latest) - len(clean)


def row_uuids(row: dict[str, Any]) -> set[str]:
    return {
        item["uuid"]
        for section in ("candidates", "results", "source_memories")
        for item in row.get(section) or ()
    }


def label_rows(
    rows: Iterable[dict[str, Any]], facts: GraphFacts, gold: dict[str, Gold]
) -> list[dict[str, Any]]:
    """One label per (id, uuid). Labels do not depend on the arm."""
    hubs = hub_entities(facts)
    uuids_by_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        uuids_by_id[row["id"]] |= row_uuids(row)

    labelled = []
    for row_id, uuids in sorted(uuids_by_id.items()):
        question_id = question_of(row_id)
        question = gold.get(question_id)
        for uuid in sorted(uuids):
            labelled.append({
                "id": row_id,
                "question_id": question_id,
                "query_kind": query_kind(row_id, question),
                "question_type": question.question_type if question else None,
                "uuid": uuid,
                "name": (facts.entities.get(uuid) or {}).get("name"),
                **label(facts, hubs, uuid, question),
            })
    return labelled


def coverage_stats(facts: GraphFacts, gold: dict[str, Gold]) -> dict[str, Any]:
    """Whole-graph label coverage, comparable with the plan's Phase 0 note. Uses no cosine."""
    hubs = hub_entities(facts)
    by_namespace: dict[str, list[str]] = defaultdict(list)
    entities = [u for u, e in facts.entities.items() if e["kind"] == "entity"]
    for uuid in entities:
        by_namespace[facts.entities[uuid]["namespace"]].append(uuid)

    routes: Counter[str] = Counter()
    lenient_questions = strict_questions = 0
    for question in gold.values():
        labels = [label(facts, hubs, u, question) for u in by_namespace.get(question.namespace, ())]
        routes.update(item["label_route"] for item in labels)
        if question.keys:
            lenient_questions += any(item["label_lenient"] for item in labels)
            strict_questions += any(item["label_strict"] for item in labels)
    return {
        "plain_entities": len(entities),
        "entities_with_turn": sum(any(evidence(facts, u)[1].values()) for u in entities),
        "hubs": len(hubs),
        "questions_with_gold": sum(bool(q.keys) for q in gold.values()),
        "questions_with_positive_lenient": lenient_questions,
        "questions_with_positive_strict": strict_questions,
        "routes_over_questions": dict(routes),
    }


def check_target(uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.hostname not in LOOPBACK_HOSTS or (parsed.port or 7687) in SHARED_PORTS:
        raise SystemExit(f"refusing graph {parsed.hostname}:{parsed.port}: use a throwaway copy")


def read_facts(driver: Any) -> GraphFacts:
    """One read-only pass over the LME groups."""
    facts = GraphFacts()

    def rows(cypher: str) -> list[dict[str, Any]]:
        with driver.session(default_access_mode="READ") as session:
            return [dict(record) for record in session.run(cypher)]

    for r in rows("MATCH (t:TurnEvidence) WHERE t.namespace STARTS WITH 'lme-' "
                  "RETURN t.turn_id AS id, t.turn_key AS key"):
        facts.turn_keys[r["id"]] = r["key"]
    for r in rows("MATCH (n:Entity) WHERE n.group_id STARTS WITH 'lme-' "
                  "RETURN n.uuid AS uuid, n.group_id AS ns, n.name AS name, "
                  "n.view_kind AS view_kind, n.episode_uuids AS episodes"):
        kind = "view" if r["view_kind"] else "entity"
        facts.entities[r["uuid"]] = {"namespace": r["ns"], "name": r["name"], "kind": kind}
        if kind == "view":
            facts.view_episodes[r["uuid"]] |= set(r["episodes"] or ())
    # Queue Episodics carry namespace, Graphiti ones group_id.
    for r in rows("MATCH (e:Episodic) WITH e, coalesce(e.group_id, e.namespace) AS ns "
                  "WHERE ns STARTS WITH 'lme-' "
                  "RETURN e.uuid AS uuid, ns, e.content AS content"):
        facts.episodes[r["uuid"]] = {"namespace": r["ns"], "content": r["content"]}
    for r in rows("MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) RETURN n.uuid AS n, e.uuid AS e"):
        facts.mentions[r["n"]].add(r["e"])
    for r in rows("MATCH (e:Episodic)-[:RETENTION_SOURCE]->(n:Entity) "
                  "RETURN n.uuid AS n, e.uuid AS e"):
        facts.retention[r["n"]].add(r["e"])
    for r in rows("MATCH (t:TurnEvidence)-[:MENTIONS]->(n:Entity) "
                  "RETURN n.uuid AS n, t.turn_id AS t"):
        facts.turn_mentions[r["n"]].add(r["t"])
    for r in rows("MATCH (t:TurnEvidence)-[:FOUNDS]->(:TypedAssertion)"
                  "<-[:HISTORY_ENTRY|CURRENT_ANCHOR|SUPERSEDED_ANCHOR|CONTRIBUTED_TO]-(n:Entity) "
                  "WHERE n.view_kind IS NOT NULL RETURN n.uuid AS n, t.turn_id AS t"):
        facts.assertion_turns[r["n"]].add(r["t"])
    for r in rows("MATCH (t:TurnEvidence)-[:FOUNDS]->(a:TypedAssertion) "
                  "WHERE a.namespace STARTS WITH 'lme-' "
                  "RETURN a.assertion_id AS a, t.turn_id AS t"):
        facts.observations.setdefault(r["a"], set()).add(r["t"])
    for r in rows("MATCH (e:Episodic)-[:ADMITTED_ON]->(t:TurnEvidence) "
                  "RETURN e.uuid AS e, e.resolved_episode_uuid AS res, t.turn_id AS t"):
        facts.self_turns[r["e"]].add(r["t"])
        if r["res"]:
            facts.receipt_turns[r["res"]].add(r["t"])
    return facts


def load_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("items") or raw["data"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--password-file", type=Path,
                        default=Path.home() / ".lme-bench" / "neo4j-pw")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    if not args.stats_only and not (args.batch and args.output):
        parser.error("--batch and --output are required unless --stats-only is set")
    check_target(args.neo4j_uri)

    from neo4j import GraphDatabase

    gold = load_gold(load_items(args.dataset))
    password = args.password_file.read_text(encoding="utf-8").strip()
    with GraphDatabase.driver(args.neo4j_uri, auth=("neo4j", password)) as driver:
        facts = read_facts(driver)

    if args.stats_only:
        print(json.dumps(coverage_stats(facts, gold), indent=2))
        return 0

    rows, failed = load_batch(args.batch)
    labelled = label_rows(rows, facts, gold)
    with args.output.open("w", encoding="utf-8") as sink:
        for item in labelled:
            sink.write(json.dumps(item) + "\n")
    routes = Counter(item["label_route"] for item in labelled)
    print(f"rows={len(rows)} failed={failed} labels={len(labelled)} routes={dict(routes)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
