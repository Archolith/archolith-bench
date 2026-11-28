"""Menhir server-side cost: enrichment drain, usage attribution and pricing.

Attribution: Menhir's `llm_usage_events` rows carry `episode_uuid` (and a bench-wide `run_id`),
never a namespace. A namespace is tied to its rows through its episodes:
`POST /api/internal/backend/list_episode_processing` with a `namespace` returns the episode
uuids (and their processing state), and the telemetry DB is summed over those uuids. Rows whose
episode is not in any chain's namespace (for example the `__non_episode__` sentinel) are
"unattributed" and are split evenly across the Menhir chains. If the episode listing is not
available at all, every row is split evenly (`even_split`).
"""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

NON_EPISODE_KEY = "__non_episode__"
PENDING_STATES = ("PENDING", "ENRICHING")
EPISODE_LIST_LIMIT = 200  # the backend clamps list_episode_processing to 200 rows


@dataclass(frozen=True)
class ServerRates:
    """USD per 1M tokens for the server's LLM calls."""

    input: float
    cached_input: float
    output: float

    @classmethod
    def parse(cls, text: str) -> "ServerRates":
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 3:
            raise ValueError("--server-rates needs input,cached_input,output ($ per 1M tokens)")
        try:
            values = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"--server-rates must be numbers: {text!r}") from exc
        if any(v < 0 for v in values):
            raise ValueError("--server-rates must not be negative")
        return cls(*values)

    def cost_usd(self, *, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
        cached = min(max(cached_input_tokens, 0), max(input_tokens, 0))
        fresh = max(input_tokens, 0) - cached
        return (
            fresh * self.input + cached * self.cached_input + max(output_tokens, 0) * self.output
        ) / 1_000_000.0


def backend_origin(mcp_url: str) -> str:
    parts = urlsplit(mcp_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def backend_post(
    mcp_url: str,
    operation: str,
    body: Mapping[str, Any],
    *,
    api_key: str = "",
    timeout_s: float = 20.0,
) -> Any:
    """`POST <origin>/api/internal/backend/<operation>` with a JSON body; returns parsed JSON."""
    url = f"{backend_origin(mcp_url)}/api/internal/backend/{operation}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(dict(body)).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else None


PostFn = Callable[..., Any]


def pending_enrichment(post_fn: PostFn, mcp_url: str, namespace: str, *, api_key: str = "") -> int:
    """Episodes of *namespace* still PENDING or ENRICHING (capped at 200 by the backend)."""
    rows = post_fn(
        mcp_url,
        "list_episode_processing",
        {"states": list(PENDING_STATES), "limit": EPISODE_LIST_LIMIT, "namespace": namespace},
        api_key=api_key,
    )
    return len(rows or [])


def wait_for_enrichment_drain(
    post_fn: PostFn,
    mcp_url: str,
    namespace: str,
    *,
    api_key: str = "",
    timeout_s: float = 180.0,
    poll_s: float = 3.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Block until *namespace* has no pending enrichment, or *timeout_s* passes.

    Returns ``{"waited_s", "drained", "pending", "polls"}``. A failed poll counts as not
    drained for that poll and is retried until the timeout.
    """
    start = clock_fn()
    polls, pending, last_error = 0, -1, ""
    while True:
        polls += 1
        try:
            pending = pending_enrichment(post_fn, mcp_url, namespace, api_key=api_key)
            last_error = ""
        except Exception as exc:  # noqa: BLE001 - the harness records and retries
            last_error = f"{type(exc).__name__}: {exc}"[:200]
        waited = clock_fn() - start
        if pending == 0 and not last_error:
            return {"waited_s": round(waited, 3), "drained": True, "pending": 0, "polls": polls}
        if waited >= timeout_s:
            out = {"waited_s": round(waited, 3), "drained": False, "pending": pending, "polls": polls}
            if last_error:
                out["error"] = last_error
            return out
        sleep_fn(min(poll_s, max(timeout_s - waited, 0.0)))


def list_namespace_episodes(
    post_fn: PostFn, mcp_url: str, namespace: str, *, api_key: str = ""
) -> dict[str, Any]:
    """Episode uuids of *namespace* (all states) and how many are FAILED."""
    rows = post_fn(
        mcp_url,
        "list_episode_processing",
        {"states": None, "limit": EPISODE_LIST_LIMIT, "namespace": namespace},
        api_key=api_key,
    ) or []
    uuids = [str(r["uuid"]) for r in rows if isinstance(r, dict) and r.get("uuid")]
    failed = sum(1 for r in rows if isinstance(r, dict) and str(r.get("processing_state", "")).upper() == "FAILED")
    return {"uuids": uuids, "failed_episodes": failed, "truncated": len(rows) >= EPISODE_LIST_LIMIT}


def _empty_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "uncached_tokens": 0,
        "embedding_calls": 0,
        "embedding_tokens": 0,
    }


def _add_row(acc: dict[str, Any], row: sqlite3.Row | tuple, columns: Sequence[str]) -> None:
    r = dict(zip(columns, row))
    if str(r.get("kind") or "").lower() == "embedding":
        acc["embedding_calls"] += 1
        acc["embedding_tokens"] += int(r.get("total_tokens") or r.get("input_tokens") or 0)
        return
    inp = int(r.get("input_tokens") or 0)
    cached = min(int(r.get("cached_input_tokens") or 0), inp)
    out = int(r.get("output_tokens") or 0)
    acc["calls"] += 1
    if str(r.get("status") or "") == "failed":
        acc["failed_calls"] += 1
    acc["input_tokens"] += inp
    acc["cached_input_tokens"] += cached
    acc["output_tokens"] += out
    acc["reasoning_output_tokens"] += int(r.get("reasoning_output_tokens") or 0)
    acc["cache_write_tokens"] += int(r.get("cache_write_input_tokens") or 0)
    acc["total_tokens"] += int(r.get("total_tokens") or (inp + out))
    # input_tokens includes cached input; output_tokens is taken to include reasoning.
    acc["uncached_tokens"] += (inp - cached) + out + int(r.get("cache_write_input_tokens") or 0)


def read_usage_rows(db_path: Path, *, since_iso: str | None = None) -> list[dict[str, Any]]:
    """Every `llm_usage_events` row (optionally recorded at/after *since_iso*)."""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        sql = (
            "SELECT call_id, recorded_at, episode_uuid, kind, status, input_tokens, output_tokens, "
            "total_tokens, cached_input_tokens, reasoning_output_tokens, cache_write_input_tokens "
            "FROM llm_usage_events"
        )
        params: list[Any] = []
        if since_iso:
            sql += " WHERE recorded_at >= ?"
            params.append(since_iso)
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def attribute_server_usage(
    rows: Sequence[Mapping[str, Any]],
    episodes_by_namespace: Mapping[str, Sequence[str]] | None,
    *,
    rates: ServerRates | None = None,
    namespaces: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Per-namespace server usage from telemetry *rows*.

    *episodes_by_namespace* maps each Menhir chain's namespace to its episode uuids, or is None
    when the mapping is unavailable (then every row is split evenly across *namespaces*).
    Returns ``{"attribution", "by_namespace": {ns: usage+cost_usd}, "unattributed": usage}``.
    """
    names = list(namespaces) if namespaces is not None else list((episodes_by_namespace or {}).keys())
    columns = (
        "kind", "status", "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
        "reasoning_output_tokens", "cache_write_input_tokens",
    )

    def tup(r: Mapping[str, Any]) -> tuple:
        return tuple(r.get(c) for c in columns)

    owner: dict[str, str] = {}
    for ns, uuids in (episodes_by_namespace or {}).items():
        for u in uuids:
            owner[str(u)] = ns
    direct = {ns: _empty_usage() for ns in names}
    pool = _empty_usage()
    for r in rows:
        ns = owner.get(str(r.get("episode_uuid") or "")) if episodes_by_namespace is not None else None
        if ns is not None and ns in direct:
            _add_row(direct[ns], tup(r), columns)
        else:
            _add_row(pool, tup(r), columns)

    n = max(len(names), 1)
    by_namespace: dict[str, dict[str, Any]] = {}
    for ns in names:
        usage = dict(direct[ns])
        usage["attributed_uncached_tokens"] = usage["uncached_tokens"]
        for key, value in pool.items():
            # Integer shares; the remainder is not distributed (rounds down), so sums can
            # undercount by < len(names) tokens per field.
            share = value // n
            usage[key] = usage[key] + share
        usage["unattributed_share_uncached_tokens"] = pool["uncached_tokens"] // n
        usage["cost_usd"] = (
            rates.cost_usd(
                input_tokens=usage["input_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                output_tokens=usage["output_tokens"],
            )
            if rates is not None
            else 0.0
        )
        by_namespace[ns] = usage
    return {
        "attribution": "even_split" if episodes_by_namespace is None else "namespace",
        "by_namespace": by_namespace,
        "unattributed": pool,
    }
