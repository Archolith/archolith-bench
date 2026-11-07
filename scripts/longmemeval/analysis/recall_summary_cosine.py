#!/usr/bin/env python
"""Query-to-summary cosines for recall candidates, the H2 rival to name cosines. Embeddings only.

Plan: menhir-recall-real-similarity-test-plan-2026-11-05. Reads a batch from Menhir's
``scripts/recall_lab_batch.py`` and its pinned ``<output>.vectors.json``, embeds each plain
entity's summary once, and writes JSONL rows of ``{"id", "uuid", "summary_cosine"}``:

    OPENAI_API_KEY=... python recall_summary_cosine.py --batch out.jsonl --queries queries.jsonl \
        --neo4j-uri bolt://127.0.0.1:18342 --output summary.jsonl

A candidate with no summary gets ``null``. Any API error stops the run, so a 429 is never retried
here. Reusing the pinned query vectors keeps these cosines comparable with the batch's.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_labels import check_target, load_batch  # noqa: E402

MODEL = "text-embedding-3-small"
EMBED_BATCH = 256
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

Embed = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float | None:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else None


def embed_openai(texts: list[str]) -> list[list[float]]:
    import httpx

    response = httpx.post(
        EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": MODEL, "input": texts},
        timeout=120,
    )
    response.raise_for_status()
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def embed_summaries(summaries: dict[str, str], embed: Embed) -> dict[str, list[float]]:
    """Embed each distinct summary text once."""
    texts = sorted(set(summaries.values()))
    vectors: dict[str, list[float]] = {}
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start:start + EMBED_BATCH]
        vectors.update(zip(chunk, embed(chunk)))
    return {uuid: vectors[text] for uuid, text in summaries.items()}


def summary_rows(
    rows: Iterable[dict[str, Any]],
    queries: dict[str, str],
    query_vectors: dict[str, list[float]],
    summary_vectors: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """One row per (id, candidate), across arms."""
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        vector = query_vectors[json.dumps(queries[row["id"]], sort_keys=True, default=str)]
        for candidate in row.get("candidates") or ():
            pair = (row["id"], candidate["uuid"])
            if pair in seen:
                continue
            seen.add(pair)
            stored = summary_vectors.get(candidate["uuid"])
            out.append({"id": pair[0], "uuid": pair[1],
                        "summary_cosine": cosine(vector, stored) if stored else None})
    return out


def read_summaries(driver: Any, uuids: list[str]) -> dict[str, str]:
    with driver.session(default_access_mode="READ") as session:
        records = session.run(
            "UNWIND $uuids AS id MATCH (n:Entity {uuid: id}) WHERE n.view_kind IS NULL "
            "RETURN n.uuid AS uuid, n.summary AS summary",
            uuids=uuids,
        )
        return {r["uuid"]: r["summary"] for r in records if (r["summary"] or "").strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, help="default: <batch>.vectors.json")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--password-file", type=Path,
                        default=Path.home() / ".lme-bench" / "neo4j-pw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_target(args.neo4j_uri)

    from neo4j import GraphDatabase

    rows, failed = load_batch(args.batch)
    if failed:
        raise SystemExit(f"{failed} (id, arm) pairs failed; rerun the batch first")
    lines = args.queries.read_text(encoding="utf-8").splitlines()
    queries = {q["id"]: q["query"] for q in map(json.loads, filter(None, lines))}
    vector_path = args.vectors or args.batch.with_suffix(".vectors.json")
    query_vectors = json.loads(vector_path.read_text(encoding="utf-8"))
    uuids = sorted({c["uuid"] for row in rows for c in row.get("candidates") or ()})

    password = args.password_file.read_text(encoding="utf-8").strip()
    with GraphDatabase.driver(args.neo4j_uri, auth=("neo4j", password)) as driver:
        summaries = read_summaries(driver, uuids)

    summary_vectors = embed_summaries(summaries, embed_openai)
    out = summary_rows(rows, queries, query_vectors, summary_vectors)
    with args.output.open("w", encoding="utf-8") as sink:
        for item in out:
            sink.write(json.dumps(item) + "\n")
    print(f"candidates={len(uuids)} summaries={len(summaries)} rows={len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
