#!/usr/bin/env python
"""Judge-based labels for recall candidates, and the pre-registered retest against RRF. Paid.

Plan: menhir-recall-real-similarity-test-plan-2026-11-05, "Judge-label retest". An LLM judges each
own-namespace entity of real questions outside SSA, once per (question, entity). ``label`` writes
labels in the ``candidate_labels.py`` shape, one file per target and graph; ``score`` then runs the
frozen ``recall_signal_eval`` code on them:

    python candidate_judge_labels.py label --run name=batch.jsonl,labels.jsonl [--run ...] \
        --dataset o500.json --neo4j-uri bolt://127.0.0.1:18345 --out-dir judged --pilot 200
    python candidate_judge_labels.py score --run name=batch.jsonl,queries.jsonl,summary.jsonl \
        --out-dir judged --output retest.json

``--pilot N`` judges N pairs and prints only cost, never verdicts. One checkpoint holds every call,
so the cost cap covers the pilot, the run and the repeats together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recall_signal_eval as ev  # noqa: E402
from candidate_labels import check_target, load_batch, load_items, question_of  # noqa: E402

Row = dict[str, Any]
MODEL = "gpt-6-luna"
RATES = (0.05, 0.25)
MAX_USD = 0.50
MAX_FACTS = 8
EVIDENCE_CHARS = 600
REPEATS = 200
PILOT_SEED = ev.SEED
TARGETS = ("in_record", "in_summary")
MIN_POSITIVES = 100
MAX_UNJUDGED = 0.02
MIN_REPEAT_AGREEMENT = 0.8
SIGNALS = (ev.COSINE, "summary_cosine", "max_name_summary", "final_score")
SYSTEM = (
    "You check whether a stored memory carries the answer to a question. Reply with JSON only: "
    '{"in_summary": true|false, "in_record": true|false}. '
    "in_summary: the memory's name or summary states the gold answer, or a specific fact needed "
    "to derive it. in_record: the same test, over the name, summary and facts together. "
    "Being about the same general topic is not enough."
)


def pair_id(question_id: str, uuid: str) -> str:
    return hashlib.sha256(f"{question_id}\0{uuid}".encode()).hexdigest()[:20]


def gold_text(item: Row) -> str:
    """The question's answer-bearing turns, truncated."""
    turns = [
        str(turn.get("content") or "")
        for session in item.get("haystack_sessions") or ()
        for turn in session
        if turn.get("has_answer")
    ]
    return "\n".join(turns)[:EVIDENCE_CHARS] or "(none)"


def request_for(custom_id: str, item: Row, record: Row) -> Row:
    facts = "\n".join(f"- {fact}" for fact in record["facts"]) or "(none)"
    user = (
        f"Question: {item['question']}\nGold answer: {item['answer']}\n"
        f"Source text behind the answer: {gold_text(item)}\n\n"
        f"Memory name: {record['name']}\nMemory summary: {record['summary'] or '(none)'}\n"
        f"Memory facts:\n{facts}"
    )
    return {"custom_id": custom_id, "body": {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 2000,
    }}


def parse_verdict(content: str | None) -> dict[str, bool] | None:
    """Both verdicts, or None if unusable. The record contains the summary, so summary implies it."""
    try:
        data = json.loads(content or "")
        in_summary, in_record = data["in_summary"], data["in_record"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(in_summary, bool) or not isinstance(in_record, bool):
        return None
    return {"in_summary": in_summary, "in_record": in_record or in_summary}


def scored_rows(labels: Iterable[Row]) -> list[Row]:
    """The label rows the retest covers: entities of real questions outside SSA."""
    return [
        m for m in labels
        if m["kind"] == "entity" and m["query_kind"] == "real"
        and ev.TYPE_CODES.get(m["question_type"]) != "SSA"
    ]


def own_pairs(labels: Iterable[Row], groups: dict[str, str]) -> set[tuple[str, str]]:
    """(question, uuid) pairs from the question's own namespace, the only ones worth judging."""
    return {
        (m["question_id"], m["uuid"]) for m in scored_rows(labels)
        if groups.get(m["uuid"]) == f"lme-{m['question_id']}"
    }


def judged_labels(
    labels: Iterable[Row],
    groups: dict[str, str],
    verdicts: dict[tuple[str, str], dict[str, bool]],
    target: str,
) -> list[Row]:
    """Label rows carrying the judge's verdict. Another namespace is a negative by construction."""
    out = []
    for meta in scored_rows(labels):
        question, uuid = meta["question_id"], meta["uuid"]
        if groups.get(uuid) == f"lme-{question}":
            verdict = verdicts.get((question, uuid))
            value = None if verdict is None else verdict[target]
        else:
            value = None if uuid not in groups else False
        route = "judge" if groups.get(uuid) == f"lme-{question}" else "other_namespace"
        out.append({**meta, "label_strict": value, "label_lenient": value,
                    "label_route": route, "is_hub": False})
    return out


def read_records(driver: Any, uuids: list[str]) -> dict[str, Row]:
    """Name, summary, group and up to MAX_FACTS facts per entity, in a stable order."""
    query = (
        "UNWIND $uuids AS id MATCH (n:Entity {uuid: id}) WHERE n.view_kind IS NULL "
        "OPTIONAL MATCH (n)-[r:RELATES_TO]-() WITH n, r.fact AS fact ORDER BY fact "
        "WITH n, [f IN collect(DISTINCT fact) WHERE f IS NOT NULL] AS facts "
        "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, n.group_id AS group_id, "
        "facts[0..$cap] AS facts, size(facts) AS total"
    )
    records: dict[str, Row] = {}
    with driver.session(default_access_mode="READ") as session:
        for start in range(0, len(uuids), 2000):
            part = uuids[start:start + 2000]
            for r in session.run(query, uuids=part, cap=MAX_FACTS):
                records[r["uuid"]] = {
                    "name": r["name"], "summary": r["summary"], "group": r["group_id"],
                    "facts": list(r["facts"]), "total_facts": r["total"],
                }
    return records


def repeat_agreement(
    first: dict[str, dict[str, bool]], second: dict[str, dict[str, bool]], target: str
) -> dict[str, Any]:
    shared = [k for k in first if k in second]
    positives = [k for k in shared if first[k][target]]
    same = [k for k in shared if first[k][target] == second[k][target]]
    return {
        "pairs": len(shared),
        "agreement": len(same) / len(shared) if shared else None,
        "positives": len(positives),
        "positive_agreement": (
            sum(second[k][target] for k in positives) / len(positives) if positives else None
        ),
    }


def reading(status: str, positives: int, unjudged_share: float, repeat_ok: bool) -> str:
    """The pre-registered reading after the validity checks."""
    if positives < MIN_POSITIVES:
        return "underpowered"
    if unjudged_share > MAX_UNJUDGED or not repeat_ok:
        return "inconclusive"
    return status


def overall(readings: list[str]) -> str:
    """Supported only on every graph, killed on any."""
    if "killed" in readings:
        return "killed"
    if readings and all(r == "supported" for r in readings):
        return "supported"
    return "inconclusive"


def score_graph(table: list[Row], n_boot: int = ev.N_BOOT, seed: int = ev.SEED) -> dict[str, Any]:
    """Paired gains over RRF in the primary cell, read by the frozen H2 rule."""
    cell, key = ev.cell_rows(table, ev.PRIMARY_CELL)
    groups, y, ids = ev._arrays(cell, key)
    weights = ev.bootstrap_weights(len(ids), n_boot, seed)

    def metrics(name: str) -> dict[str, Any]:
        scores = np.array([r["signals"][name] for r in cell])
        return ev.signal_metrics(scores, y, groups, weights)

    rrf = metrics(ev.RRF)
    signals = {}
    for name in SIGNALS:
        values = metrics(name)
        versus = {
            m: ev.interval(values[m][0] - rrf[m][0], values[m][1] - rrf[m][1])
            for m in ("pooled_auc", "within_auc")
        }
        signals[name] = {"minus_rrf": versus,
                         "status": ev.h2_verdict({"cosine_minus": {ev.RRF: versus}})["status"]}
    return {"questions": len(ids), "candidates": len(cell), "positives": int(y.sum()),
            "signals": signals}


def _jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Row]) -> None:
    with path.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")


def _parse_runs(values: list[str], parts: int) -> dict[str, list[Path]]:
    runs = {}
    for value in values:
        name, paths = value.split("=", 1)
        runs[name] = [Path(p) for p in paths.split(",")]
        if len(runs[name]) != parts:
            raise SystemExit(f"--run {name}: expected {parts} paths")
    return runs


def label(args: argparse.Namespace) -> int:
    check_target(args.neo4j_uri)
    from archolith_bench.core.openai_flex import run_flex
    from neo4j import GraphDatabase

    sys.path.insert(0, str(Path.home() / ".claude" / "tools"))
    from progress import Progress

    runs = _parse_runs(args.run, 2)
    items = {str(i["question_id"]): i for i in load_items(args.dataset)}
    labels = {name: _jsonl(paths[1]) for name, paths in runs.items()}
    uuids = sorted({m["uuid"] for rows in labels.values() for m in scored_rows(rows)})
    password = args.password_file.read_text(encoding="utf-8").strip()
    with GraphDatabase.driver(args.neo4j_uri, auth=("neo4j", password)) as driver:
        records = read_records(driver, uuids)
    groups = {uuid: record["group"] for uuid, record in records.items()}
    pairs = sorted(set().union(*(own_pairs(rows, groups) for rows in labels.values())))
    missing = len(uuids) - len(records)
    capped = sum(record["total_facts"] > MAX_FACTS for record in records.values())
    print(f"pairs={len(pairs)} entities={len(records)} unread={missing} facts_capped={capped}")

    rng = random.Random(PILOT_SEED)
    requests = {pair_id(*p): request_for(pair_id(*p), items[p[0]], records[p[1]]) for p in pairs}
    repeats = rng.sample(pairs, min(REPEATS, len(pairs)))
    if args.pilot:
        chosen = [requests[pair_id(*p)] for p in rng.sample(pairs, args.pilot)]
    else:
        chosen = list(requests.values())
        for p in repeats:
            base = requests[pair_id(*p)]
            chosen.append({**base, "custom_id": "r-" + base["custom_id"]})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress("menhir-recall-judge-labels", "Recall judge labels", total=len(chosen),
                        unit="judgements", ceiling_usd=MAX_USD)
    try:
        results, summary = run_flex(
            chosen, model=MODEL, checkpoint=args.out_dir / "judge.ckpt.jsonl",
            api_key=os.environ["OPENAI_API_KEY"], max_usd=MAX_USD, rates=RATES, workers=16,
        )
    except Exception as exc:
        progress.finish("failed", note=type(exc).__name__)
        raise
    cost = summary["cost_usd"]
    progress.update(done=len(results), spend_usd=cost)
    if args.pilot:
        estimate = cost / max(1, len(results)) * (len(pairs) + len(repeats))
        print(f"pilot calls={len(results)} cost=${cost:.4f} projected_total=${estimate:.3f}")
        progress.finish(note="pilot finished")
        return 0

    parsed = {cid: parse_verdict(r.content) if not r.error else None for cid, r in results.items()}
    verdicts = {p: parsed[pair_id(*p)] for p in pairs if parsed.get(pair_id(*p))}
    first = {pair_id(*p): parsed[pair_id(*p)] for p in repeats if parsed.get(pair_id(*p))}
    second = {pair_id(*p): parsed["r-" + pair_id(*p)] for p in repeats
              if parsed.get("r-" + pair_id(*p))}
    for name, rows in labels.items():
        for target in TARGETS:
            _write_jsonl(args.out_dir / f"{name}-{target}.jsonl",
                         judged_labels(rows, groups, verdicts, target))
    meta = {
        "pairs": len(pairs), "judged": len(verdicts), "cost_usd": cost, "model": MODEL,
        "facts_capped": capped, "unread_entities": missing,
        "repeat": {t: repeat_agreement(first, second, t) for t in TARGETS},
    }
    (args.out_dir / "judge-meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"judged={len(verdicts)} of {len(pairs)} cost=${cost:.4f}")
    progress.finish(note="judge labels finished")
    return 0


def score(args: argparse.Namespace) -> int:
    meta = json.loads((args.out_dir / "judge-meta.json").read_text(encoding="utf-8"))
    unjudged = 1 - meta["judged"] / meta["pairs"]
    repeat = meta["repeat"]["in_record"]["positive_agreement"]
    repeat_ok = repeat is not None and repeat >= MIN_REPEAT_AGREEMENT
    report: dict[str, Any] = {"meta": meta, "unjudged_share": unjudged, "repeat_ok": repeat_ok,
                              "targets": {}}
    for target in TARGETS:
        graphs = {}
        for name, (batch, queries_path, summary_path) in _parse_runs(args.run, 3).items():
            rows, _ = load_batch(batch)
            table_labels = {(m["id"], m["uuid"]): m
                            for m in _jsonl(args.out_dir / f"{name}-{target}.jsonl")}
            queries = {q["id"]: q["query"] for q in _jsonl(queries_path)}
            summaries = {(s["id"], s["uuid"]): s["summary_cosine"] for s in _jsonl(summary_path)}
            result = score_graph(ev.candidate_table(rows, table_labels, queries, summaries))
            for signal in result["signals"].values():
                signal["reading"] = reading(signal["status"], result["positives"], unjudged,
                                            repeat_ok)
            graphs[name] = result
        report["targets"][target] = {
            "graphs": graphs,
            "overall": {s: overall([g["signals"][s]["reading"] for g in graphs.values()])
                        for s in (ev.COSINE, "summary_cosine")},
        }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({t: v["overall"] for t, v in report["targets"].items()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    labeller = commands.add_parser("label")
    labeller.add_argument("--run", action="append", required=True)
    labeller.add_argument("--dataset", type=Path, required=True)
    labeller.add_argument("--neo4j-uri", required=True)
    labeller.add_argument("--password-file", type=Path,
                          default=Path.home() / ".lme-bench" / "neo4j-pw")
    labeller.add_argument("--out-dir", type=Path, required=True)
    labeller.add_argument("--pilot", type=int, default=0)
    labeller.set_defaults(func=label)
    scorer = commands.add_parser("score")
    scorer.add_argument("--run", action="append", required=True)
    scorer.add_argument("--out-dir", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.set_defaults(func=score)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
