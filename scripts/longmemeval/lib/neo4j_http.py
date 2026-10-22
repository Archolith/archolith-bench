"""Read-only Neo4j HTTP transaction client for the bench.

Lets ingest read graph state over the HTTP API instead of spawning ``docker exec
cypher-shell`` per poll. Every failure raises ``Neo4jUnavailable``; an error is never
reported as an empty result, because callers treat "no rows" as a real answer.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

_COUNT_KEYS = (
    "pending", "ready", "enriching", "failed", "llm_tasks", "processing_attempts", "total",
)

_NAMESPACE_COUNTS = """
UNWIND $namespaces AS ns
OPTIONAL MATCH (e:Episodic {namespace: ns})
RETURN ns,
  sum(CASE WHEN e.processing_state='PENDING' THEN 1 ELSE 0 END),
  sum(CASE WHEN e.processing_state='READY' THEN 1 ELSE 0 END),
  sum(CASE WHEN e.processing_state='ENRICHING' THEN 1 ELSE 0 END),
  sum(CASE WHEN e.processing_state='FAILED' THEN 1 ELSE 0 END),
  sum(coalesce(toInteger(e.processing_llm_tasks_total), 0)),
  sum(coalesce(toInteger(e.processing_attempts), 0)),
  count(e)
"""


class Neo4jUnavailable(RuntimeError):
    """Neo4j could not be read over HTTP (transport, status, query error, or bad body)."""


class Neo4jHttpReader:
    def __init__(
        self,
        base_url: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/db/{database}/tx/commit"
        self._auth = httpx.BasicAuth(user, password)
        self._timeout_s = timeout_s
        self._client = client if client is not None else httpx.Client(timeout=timeout_s)

    @classmethod
    def from_env(cls) -> "Neo4jHttpReader | None":
        url = os.getenv("LME_NEO4J_HTTP_URL", "").strip()
        if not url:
            return None
        return cls(url, "neo4j", os.getenv("LME_NEO4J_PW", "lmedata123"))

    def rows(self, statement: str, parameters: dict | None = None) -> list[list[object]]:
        body = {"statements": [{"statement": statement, "parameters": parameters or {}}]}
        try:
            response = self._client.post(
                self._url, json=body, auth=self._auth, timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise Neo4jUnavailable(f"neo4j http request failed: {type(exc).__name__}") from exc
        if not 200 <= response.status_code < 300:
            raise Neo4jUnavailable(f"neo4j http status {response.status_code}")
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise Neo4jUnavailable("neo4j http response was not JSON") from exc
        if not isinstance(payload, dict):
            raise Neo4jUnavailable("neo4j http response was malformed")
        errors = payload.get("errors")
        if errors:
            raise Neo4jUnavailable(f"neo4j query error: {errors!r}")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise Neo4jUnavailable("neo4j http response had no result set")
        data = results[0].get("data")
        if not isinstance(data, list):
            raise Neo4jUnavailable("neo4j http result had no data list")
        out: list[list[object]] = []
        for item in data:
            row = item.get("row") if isinstance(item, dict) else None
            if not isinstance(row, list):
                raise Neo4jUnavailable("neo4j http result row was malformed")
            out.append(row)
        return out

    def namespace_state_counts(self, namespaces: list[str]) -> dict[str, dict[str, int]]:
        """One row per requested namespace (OPTIONAL MATCH grouped by ns, zeros when empty).
        A missing row means a malformed answer, never "no episodes", so it raises."""
        wanted = set(namespaces)
        counts: dict[str, dict[str, int]] = {}
        if not namespaces:
            return counts
        for row in self.rows(_NAMESPACE_COUNTS, {"namespaces": list(namespaces)}):
            if len(row) != len(_COUNT_KEYS) + 1 or row[0] not in wanted:
                raise Neo4jUnavailable("neo4j namespace counts row was malformed")
            try:
                values = [int(value or 0) for value in row[1:]]
            except (TypeError, ValueError) as exc:
                raise Neo4jUnavailable("neo4j namespace counts were not integers") from exc
            counts[row[0]] = dict(zip(_COUNT_KEYS, values))
        missing = [ns for ns in namespaces if ns not in counts]
        if missing:
            raise Neo4jUnavailable(f"neo4j namespace counts had no row for {missing[:5]}")
        return counts
