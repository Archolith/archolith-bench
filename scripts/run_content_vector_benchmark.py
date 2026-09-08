"""Run the pre-registered A/B/C content-vector benchmark on the frozen clone."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from dotenv import dotenv_values
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MENHIR = ROOT.parent / "menhir"
CLONE_URI = "bolt://192.168.86.33:7689"
CLONE_PASSWORD = "testpassword"
CLONE_DUMP_SHA256 = "b941c956b9dd1e7d762c8e8b5693ff85fdadee5e5d5bc7c982015ce056aea50f"

sys.path[:0] = [str(ROOT), str(MENHIR / "src")]

from archolith_bench.r1.autogen_eval import (  # noqa: E402
    AutogenQuery,
    EvalSetResult,
    build_eval_set,
)
from archolith_bench.r1.autogen_eval_live import (  # noqa: E402
    Neo4jCorpusReader,
    OpenAIQueryGenerator,
)
from archolith_bench.r1.graph_fingerprint import (  # noqa: E402
    assert_no_writes,
    graph_write_fingerprint,
)
from archolith_bench.r1.metrics import known_item_rank  # noqa: E402
from archolith_bench.r1.runner import ConditionResult, evaluate_win_gate  # noqa: E402


def _load_env() -> None:
    for key, value in dotenv_values(ROOT / ".env").items():
        if value is not None:
            os.environ[key] = value
    os.environ.update({
        "NEO4J_URI": CLONE_URI,
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": CLONE_PASSWORD,
        "NEO4J_DATABASE": "neo4j",
        "OTEL_SDK_DISABLED": "true",
        "LANGFUSE_TRACING_ENABLED": "false",
    })


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=path, text=True).strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _metrics(
    cases: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    latencies: dict[str, float],
    metadata: dict[str, dict[str, str]],
    *,
    limit: int,
) -> tuple[dict[str, float], dict[str, int]]:
    ranks: dict[str, int] = {}
    stale = wrong = returned = 0
    for case in cases:
        ids = rankings[case["id"]]
        ranks[case["id"]] = known_item_rank(ids, case["gold_ids"], limit=limit)
        for uuid in ids[:10]:
            returned += 1
            row = metadata.get(uuid, {})
            stale += int(row.get("freshness") not in ("", "ACTIVE"))
            wrong += int(row.get("namespace", "default") != case["namespace"])
    rank_values = list(ranks.values())
    n = len(cases) or 1
    metrics = {
        "recall_at_1": sum(rank <= 1 for rank in rank_values) / n,
        "recall_at_5": sum(rank <= 5 for rank in rank_values) / n,
        "recall_at_10": sum(rank <= 10 for rank in rank_values) / n,
        "mrr": sum(1.0 / rank for rank in rank_values) / n,
        "stale_hit_rate": stale / returned if returned else 0.0,
        "wrong_scope_injection_rate": wrong / returned if returned else 0.0,
        "latency_ms": mean(latencies[case["id"]] for case in cases) if cases else 0.0,
    }
    return metrics, ranks


def _gate(
    results: dict[str, ConditionResult],
    ranks: dict[str, dict[str, int]],
    baseline: str,
    challenger: str,
    *,
    rank_tolerance: int,
) -> dict[str, Any]:
    """Run one pairwise gate with rank regression measured against this baseline.

    Rank regression cannot be stored as one reusable condition metric: C must be compared
    with B, while B must be compared with A. Build comparison-local results so the gate's
    baseline is always zero regression and the challenger is the observed regression rate.
    """
    keys = ranks[challenger]
    regressions = mean(
        keys[key] > ranks[baseline][key] + rank_tolerance for key in keys
    ) if keys else 0.0
    pair: dict[str, ConditionResult] = {}
    for name in (baseline, challenger):
        metrics = dict(results[name].metrics)
        metrics["rank_regression_rate"] = 0.0 if name == baseline else regressions
        pair[name] = ConditionResult(name, metrics, results[name].per_query)
    return evaluate_win_gate(
        pair,
        baseline_condition=baseline,
        challenger_prefix=challenger,
        primary_improvement_metrics=("recall_at_5", "recall_at_10", "mrr"),
        improvement_mode="any",
        guards_lower_is_better=(
            "stale_hit_rate", "wrong_scope_injection_rate",
            "negative_query_false_positive_rate", "rank_regression_rate",
        ),
        guards_higher_is_better=("exact_string_recall", "symbol_recall"),
    )


def _load_generated(path: Path) -> EvalSetResult:
    """Reuse the exact generated corpus from an earlier artifact."""
    payload = _read_json(path)["generated"]

    def query(row: dict[str, Any]) -> AutogenQuery:
        return AutogenQuery(
            query_text=row["query_text"],
            gold_cluster_ids=tuple(row["gold_cluster_ids"]),
            source_uuid=row["source_uuid"],
            source_text_sha256=row["source_text_sha256"],
            namespace=row["namespace"],
            stratum=row.get("stratum", ""),
        )

    return EvalSetResult(
        queries=[query(row) for row in payload["clean"]],
        leaked=[query(row) for row in payload.get("leaked", [])],
        skipped_empty=int(payload.get("skipped_empty", 0)),
        cluster_sizes=[int(value) for value in payload.get("cluster_sizes", [])],
    )


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from menhir.config import MemorySettings
    from menhir.domain.retrieval_tuning import RetrievalTuningConfig
    from menhir.infrastructure.graphiti_client import GraphitiClient
    from menhir.infrastructure.memory_graph_adapter import MemoryGraphAdapter
    from menhir.infrastructure.neo4j import Neo4jRepository
    from menhir.services.recall_service import RecallService
    from menhir.services.scoring_service import ScoringService

    from archolith_bench.harness.scalar_bolt import assert_not_prod

    assert_not_prod(CLONE_URI)
    driver = GraphDatabase.driver(CLONE_URI, auth=("neo4j", CLONE_PASSWORD))
    corpus = Neo4jCorpusReader(driver)
    generator_model = args.generator_model
    if args.generated_from:
        generated = _load_generated(args.generated_from)
    else:
        generated = build_eval_set(
            corpus,
            OpenAIQueryGenerator(OpenAI(api_key=os.environ["OPENAI_API_KEY"]), generator_model),
            n=args.n,
            seed=args.seed,
            duplicate_threshold=args.duplicate_threshold,
            stratum_of=lambda node: node.namespace,
        )
    anchors_path = ROOT / "corpora" / "menhir_recall_anchors.json"
    negatives_path = ROOT / "corpora" / "menhir_recall_negatives.json"
    guards_path = ROOT / "fixtures" / "r1_dummy_gold.json"
    anchors = _read_json(anchors_path)["cases"]
    negatives = _read_json(negatives_path)["cases"]
    guard_candidates = [
        row for row in _read_json(guards_path)["queries"]
        if row["family"] in ("exact_error_string", "symbol_name_query")
    ]
    auto_cases = [
        {"id": f"auto-{i:04d}", "query": q.query_text,
         "gold_ids": list(q.gold_cluster_ids), "namespace": q.namespace}
        for i, q in enumerate(generated.queries, start=1)
    ]
    all_known = auto_cases + anchors

    with driver.session() as session:
        before = graph_write_fingerprint(session)
        graph_counts = dict(session.run(
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() "
            "RETURN nodes, count(r) AS relationships"
        ).single())
        indexes = session.run(
            "SHOW INDEXES YIELD name,type,state,labelsOrTypes,properties "
            "RETURN name,type,state,labelsOrTypes,properties ORDER BY name"
        ).data()
        embedding_counts = dict(session.run(
            "MATCH (n:Entity) RETURN count(n) AS entities, "
            "count(n.name_embedding) AS name_embeddings, "
            "count(n.content_embedding) AS content_embeddings"
        ).single())
        metadata = {
            row["uuid"]: {"freshness": row["freshness"] or "", "namespace": row["namespace"]}
            for row in session.run(
                "MATCH (n:Entity) RETURN n.uuid AS uuid,n.freshness AS freshness,"
                "coalesce(n.namespace,'default') AS namespace"
            )
        }
        missing_anchors = [
            uuid for case in anchors for uuid in case["gold_ids"] if uuid not in metadata
        ]
        if missing_anchors:
            raise RuntimeError(f"anchor UUIDs missing from clone: {missing_anchors}")

        guard_cases: list[dict[str, Any]] = []
        missing_guard_cases: list[str] = []
        for row in guard_candidates:
            present_gold = [uuid for uuid in row["support_ids"] if uuid in metadata]
            if not present_gold:
                missing_guard_cases.append(row["id"])
                continue
            guard_cases.append({
                "id": f"guard-{row['id']}",
                "query": row["text"],
                "gold_ids": present_gold,
                "namespace": "default",
                "family": row["family"],
            })
        guard_families = {case["family"] for case in guard_cases}
        required_guard_families = {"exact_error_string", "symbol_name_query"}
        if guard_families != required_guard_families:
            raise RuntimeError(
                "exact/symbol guard corpus has no live cases for: "
                f"{sorted(required_guard_families - guard_families)}"
            )

        source_ids = [query.source_uuid for query in generated.queries]
        content_neighborhood_sizes = {
            row["source_uuid"]: row["neighborhood_size"]
            for row in session.run(
                "UNWIND $source_ids AS source_uuid "
                "MATCH (s:Entity {uuid: source_uuid}) "
                "WHERE s.content_embedding IS NOT NULL "
                "MATCH (n:Entity) WHERE n.content_embedding IS NOT NULL "
                "WITH source_uuid, vector.similarity.cosine("
                "s.content_embedding, n.content_embedding) AS cosine "
                "RETURN source_uuid, sum(CASE WHEN cosine >= $threshold "
                "THEN 1 ELSE 0 END) AS neighborhood_size",
                source_ids=source_ids,
                threshold=args.duplicate_threshold,
            )
        }

    settings = MemorySettings.from_env()
    graphiti = GraphitiClient.from_settings(settings)
    repository = Neo4jRepository(
        uri=CLONE_URI, database="neo4j", user="neo4j", password=CLONE_PASSWORD
    )
    service = RecallService(
        graphiti_client=graphiti,
        graph_adapter=MemoryGraphAdapter(neo4j=repository),
        scoring_service=ScoringService(),
    )
    tunings = {
        "production": RetrievalTuningConfig(),
        "A": RetrievalTuningConfig(
            enable_bm25=True, fusion_admission_policy="production_fused"
        ),
        "B": RetrievalTuningConfig(
            enable_bm25=True, enable_content_vector=True,
            fusion_admission_policy="production_fused",
        ),
        "C": RetrievalTuningConfig(
            enable_bm25=True, enable_content_vector=True,
            content_vector_replace_name=True,
            fusion_admission_policy="production_fused",
        ),
    }

    async def recall_case(case: dict[str, Any], tuning: Any) -> tuple[list[str], float]:
        started = perf_counter()
        result = await service.recall(
            case["query"], limit=args.limit, candidate_k=args.candidate_k,
            namespace=case.get("namespace", "default"), tuning=tuning,
            update_access=False, trace=False,
        )
        return [row.uuid for row in result.results], (perf_counter() - started) * 1000

    all_evaluation_cases = all_known + guard_cases
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in tunings}
    latencies: dict[str, dict[str, float]] = {name: {} for name in tunings}
    async def run_condition(name: str) -> None:
        tuning = tunings[name]
        for index, case in enumerate(all_evaluation_cases, start=1):
            ids, latency = await recall_case(case, tuning)
            rankings[name][case["id"]] = ids
            latencies[name][case["id"]] = latency
        print(f"condition {name}: {len(all_evaluation_cases)} known-item queries", flush=True)

    for name in ("production", "A", "B"):
        await run_condition(name)

    parity_diffs = [
        case["id"] for case in all_evaluation_cases
        if rankings["production"][case["id"]] != rankings["A"][case["id"]]
    ]
    if parity_diffs:
        raise RuntimeError(f"VOID: Arm A parity failed for {parity_diffs[:10]}")

    negative_bounds: list[dict[str, Any]] = []
    with driver.session() as session:
        for case in negatives:
            absent = session.run(
                "MATCH (n:Entity) WHERE toLower(coalesce(n.summary,'')+' '+"
                "coalesce(n.content,'')+' '+coalesce(n.name,'')) CONTAINS toLower($keyword) "
                "RETURN count(n) AS count", keyword=case["absent_keyword"]
            ).single()["count"]
            vector = await graphiti.embed_query(case["query"])
            max_cosine = session.run(
                "MATCH (n:Entity) WHERE n.content_embedding IS NOT NULL "
                "RETURN max(vector.similarity.cosine(n.content_embedding,$vector)) AS value",
                vector=vector,
            ).single()["value"]
            negative_bounds.append({**case, "keyword_matches": absent, "max_content_cosine": max_cosine})
            if absent:
                raise RuntimeError(f"negative keyword is present: {case['absent_keyword']}")

    async def run_negatives(name: str) -> None:
        positives = 0
        for case in negatives:
            ids, _latency = await recall_case({**case, "namespace": "default"}, tunings[name])
            positives += int(bool(ids))
        rankings[name]["__negative_positives__"] = [str(positives)]

    for name in ("A", "B"):
        await run_negatives(name)

    condition_results: dict[str, dict[str, ConditionResult]] = {"auto": {}, "anchors": {}}
    rank_sets: dict[str, dict[str, dict[str, int]]] = {"auto": {}, "anchors": {}}

    def record_condition_metrics(label: str, cases: list[dict[str, Any]], name: str) -> None:
        metrics, ranks = _metrics(
            cases, rankings[name], latencies[name], metadata, limit=args.limit
        )
        positives = int(rankings[name]["__negative_positives__"][0])
        metrics["negative_query_false_positive_rate"] = positives / len(negatives)
        for family, metric_name in (
            ("exact_error_string", "exact_string_recall"),
            ("symbol_name_query", "symbol_recall"),
        ):
            family_cases = [case for case in guard_cases if case["family"] == family]
            metrics[metric_name] = mean(
                known_item_rank(
                    rankings[name][case["id"]], case["gold_ids"], limit=args.limit
                ) <= 5
                for case in family_cases
            )
        rank_sets[label][name] = ranks
        condition_results[label][name] = ConditionResult(name, metrics, rankings[name])

    for label, cases in (("auto", auto_cases), ("anchors", anchors)):
        for name in ("A", "B"):
            record_condition_metrics(label, cases, name)

    auto_gate = _gate(
        condition_results["auto"], rank_sets["auto"], "A", "B",
        rank_tolerance=args.rank_tolerance,
    )
    anchor_gate = _gate(
        condition_results["anchors"], rank_sets["anchors"], "A", "B",
        rank_tolerance=args.rank_tolerance,
    )
    graduated = bool(auto_gate["graduates"] and anchor_gate["graduates"])
    auto_c_gate = anchor_c_gate = None
    if graduated:
        await run_condition("C")
        await run_negatives("C")
        for label, cases in (("auto", auto_cases), ("anchors", anchors)):
            record_condition_metrics(label, cases, "C")
        auto_c_gate = _gate(
            condition_results["auto"], rank_sets["auto"], "B", "C",
            rank_tolerance=args.rank_tolerance,
        )
        anchor_c_gate = _gate(
            condition_results["anchors"], rank_sets["anchors"], "B", "C",
            rank_tolerance=args.rank_tolerance,
        )
    prefer_c = bool(
        auto_c_gate and anchor_c_gate
        and auto_c_gate["graduates"] and anchor_c_gate["graduates"]
    )

    with driver.session() as session:
        after = graph_write_fingerprint(session)
    assert_no_writes(before, after, context="content-vector benchmark")
    repository.close()
    await graphiti.close()
    driver.close()

    parity = {"passed": True, "queries": len(all_evaluation_cases)}
    cluster_distribution = dict(sorted(Counter(generated.cluster_sizes).items()))
    freeze_basis = {
        "dump": "2026-07-14 production offline dump",
        "dump_sha256": CLONE_DUMP_SHA256,
        "graph_counts": graph_counts,
        "embedding_counts": embedding_counts,
        "indexes": indexes,
        "write_fingerprint": before.hashes,
    }
    clone_freeze_id = hashlib.sha256(
        json.dumps(freeze_basis, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "decision": "GRADUATE" if graduated else "NO-GRADUATE",
        "recommended_arm": "C" if prefer_c else ("B" if graduated else None),
        "parity": parity,
        "gates": {"auto_B_over_A": auto_gate, "anchors_B_over_A": anchor_gate,
                  "auto_C_over_B": auto_c_gate, "anchors_C_over_B": anchor_c_gate},
        "metrics": {
            label: {name: result.metrics for name, result in rows.items()}
            for label, rows in condition_results.items()
        },
        "ranks": rank_sets,
        "generated": {
            "clean": [asdict(q) for q in generated.queries],
            "leaked": [asdict(q) for q in generated.leaked],
            "skipped_empty": generated.skipped_empty,
            "cluster_sizes": generated.cluster_sizes,
            "content_cosine_neighborhood_sizes": content_neighborhood_sizes,
        },
        "guard_corpus": {
            "used": guard_cases,
            "missing_case_ids": missing_guard_cases,
        },
        "negative_bounds": negative_bounds,
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "clone_uri": CLONE_URI,
            "clone_dump": "2026-07-14 production offline dump",
            "clone_dump_sha256": CLONE_DUMP_SHA256,
            "clone_freeze_id": clone_freeze_id,
            "graph_counts": graph_counts,
            "embedding_counts": embedding_counts,
            "indexes": indexes,
            "write_fingerprint_before": before.hashes,
            "write_fingerprint_after": after.hashes,
            "no_writes": True,
            "edge_count_parity": {
                "passed": True,
                "basis": "offline dump clone",
                "dump_sha256": CLONE_DUMP_SHA256,
            },
            "menhir": _git_state(MENHIR),
            "bench": _git_state(ROOT),
            "embedding": {
                "query": {"provider": settings.graphiti_embed_provider,
                          "model": settings.openai_embed_model, "dimensions": 1536},
                "stored": {"provider": settings.graphiti_embed_provider,
                           "model": settings.openai_embed_model, "dimensions": 1536},
            },
            "generator_model": generator_model,
            "generated_from": str(args.generated_from) if args.generated_from else None,
            "text_builder": "summary->content->name:v1",
            "seed": args.seed,
            "requested_n": args.n,
            "clean_n": len(generated.queries),
            "leaked_n": len(generated.leaked),
            "duplicate_threshold": args.duplicate_threshold,
            "lexical_cluster_size_distribution": cluster_distribution,
            "content_cosine_neighborhood_sizes": content_neighborhood_sizes,
            "candidate_k": args.candidate_k,
            "limit": args.limit,
            "rank_tolerance": args.rank_tolerance,
            "floor": 0.15,
            "parity": parity,
            "anchor_sha256": _sha256(anchors_path),
            "negative_sha256": _sha256(negatives_path),
            "exact_symbol_guard_sha256": _sha256(guards_path),
            "exact_symbol_guard_cases": len(guard_cases),
            "exact_symbol_guard_missing": len(missing_guard_cases),
            "tunings": {name: asdict(tuning) for name, tuning in tunings.items()},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--duplicate-threshold", type=float, default=0.9)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rank-tolerance", type=int, default=0)
    parser.add_argument("--generator-model", default="gpt-4.1-nano")
    parser.add_argument("--generated-from", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    _load_env()
    artifact = asyncio.run(_run(args))
    out = args.out or ROOT / "results" / f"content-vector-{datetime.now():%Y%m%d-%H%M%S}" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"decision={artifact['decision']} artifact={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
