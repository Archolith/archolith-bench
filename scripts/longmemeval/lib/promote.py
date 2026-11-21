"""Promote one build's namespaces from SESSION to PERSISTENT through Menhir.

Usage: promote.py <manifest.json> <menhir_url>

Calls Menhir's ``promote_namespace_scope`` backend op once per manifest namespace. It replaces the
prefix-wide cypher-shell write promote_persistent.sh used to run. Exits non-zero if the manifest has
no namespaces or any promotion fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


def manifest_namespaces(manifest: Path) -> list[str]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    names = [str(row.get("namespace") or "").strip() for row in rows]
    if not names or any(not name for name in names):
        raise SystemExit(f"promote: manifest has missing namespaces: {manifest}")
    return list(dict.fromkeys(names))


def promote(namespaces: list[str], menhir_url: str, client: httpx.Client) -> tuple[int, int]:
    url = menhir_url.rstrip("/") + "/api/internal/backend/promote_namespace_scope"
    entities = edges = 0
    for namespace in namespaces:
        resp = client.post(url, json={"namespace": namespace})
        resp.raise_for_status()
        result = resp.json()
        entities += int(result["entities_promoted"])
        edges += int(result["edges_promoted"])
    return entities, edges


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: promote.py <manifest.json> <menhir_url>")
    namespaces = manifest_namespaces(Path(argv[1]))
    with httpx.Client(timeout=600.0) as client:
        entities, edges = promote(namespaces, argv[2], client)
    print(
        f"[lme-promote] {len(namespaces)} namespaces: "
        f"entities_promoted={entities} edges_promoted={edges}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
