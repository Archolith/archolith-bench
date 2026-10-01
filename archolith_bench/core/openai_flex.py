"""OpenAI flex-processing client for non-live benchmark LLM calls.

Batch-style benchmark calls (answering, judging) use OpenAI flex processing
(``service_tier: "flex"``, synchronous, ~50% of standard price) instead of the
OpenRouter Batch API. Flex calls complete in minutes at reduced cost and avoid
the multi-hour queue deadlocks seen with the batch API.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

_CAPACITY_MARKERS = ("resource_unavailable", "resource unavailable")
_RETRYABLE_STATUS = {408, 500, 502, 503, 504}
_IMMEDIATE_STOP_STATUS = {429, 402, 401, 403}
_MAX_TRANSIENT_RETRIES = 5
_PROGRESS_EVERY = 250


@dataclass(frozen=True)
class FlexResult:
    """Outcome of one flex chat-completion request."""

    custom_id: str
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str | None = None


class FlexError(RuntimeError):
    """Stop condition: budget cap, non-capacity 429, 402, or auth failure."""


class _RunStopped(Exception):
    pass


def _row_from_result(result: FlexResult) -> dict:
    return {
        "custom_id": result.custom_id,
        "content": result.content,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "error": result.error,
    }


def _result_from_row(row: dict) -> FlexResult:
    return FlexResult(
        custom_id=row["custom_id"],
        content=row.get("content"),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        cost_usd=float(row.get("cost_usd") or 0.0),
        error=row.get("error"),
    )


def _load_checkpoint(path: Path) -> dict[str, FlexResult]:
    done: dict[str, FlexResult] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("custom_id"):
                done[row["custom_id"]] = _result_from_row(row)
    return done


def _classify_capacity(body_text: str) -> bool:
    lowered = body_text.lower()
    return any(marker in lowered for marker in _CAPACITY_MARKERS)


def _cost(prompt_tokens: int, completion_tokens: int, rates: tuple[float, float]) -> float:
    return prompt_tokens * rates[0] / 1e6 + completion_tokens * rates[1] / 1e6


def run_flex(
    requests: list[dict],
    *,
    model: str,
    checkpoint: Path,
    api_key: str,
    max_usd: float,
    rates: tuple[float, float],
    workers: int = 16,
    base_url: str = "https://api.openai.com/v1",
    timeout_s: float = 900.0,
    max_capacity_retries: int = 5,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, FlexResult], dict]:
    """Run chat-completion requests through OpenAI flex processing.

    ``requests`` items are ``{"custom_id": str, "body": {...}}`` where ``body``
    is a chat.completions body WITHOUT ``model``. ``model`` and
    ``service_tier="flex"`` are always set by this client; caller body fields
    are never overridden. Rates are (input $/1M, output $/1M) at flex prices
    and must be passed in explicitly.
    """
    ids = [req["custom_id"] for req in requests]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate custom_id in requests")
    id_set = set(ids)

    done = {k: v for k, v in _load_checkpoint(checkpoint).items() if k in id_set}
    pending = [req for req in requests if req["custom_id"] not in done]
    total = len(requests)
    started = time.monotonic()

    lock = threading.Lock()
    state = {"cost": sum(r.cost_usd for r in done.values()), "finished": len(done)}
    stop_event = threading.Event()
    stop_error: list[FlexError] = []

    def append_result(result: FlexResult) -> None:
        with lock:
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_row_from_result(result)) + "\n")
            state["cost"] += result.cost_usd
            state["finished"] += 1
            finished_now = state["finished"]
            cost_now = state["cost"]
        if finished_now % _PROGRESS_EVERY == 0:
            log(f"done {finished_now}/{total} ${cost_now:.4f}")

    def execute(req: dict) -> FlexResult:
        custom_id = req["custom_id"]
        body = {**req["body"], "model": model, "service_tier": "flex"}
        url = f"{base_url.rstrip('/')}/chat/completions"
        attempt = 0
        while True:
            if stop_event.is_set():
                raise _RunStopped()
            try:
                resp = client.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= _MAX_TRANSIENT_RETRIES:
                    return FlexResult(custom_id, None, 0, 0, 0.0, error=f"transport: {type(exc).__name__}")
                delay = min(60, 5 * 2**attempt)
                attempt += 1
                sleep(delay)
                continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    content = None
                    choices = data.get("choices") or []
                    if choices:
                        content = (choices[0].get("message") or {}).get("content")
                except Exception as exc:
                    return FlexResult(custom_id, None, 0, 0, 0.0, error=f"bad response: {type(exc).__name__}")
                return FlexResult(custom_id, content, pt, ct, _cost(pt, ct, rates))
            body_text = resp.text or ""
            if resp.status_code == 429 and _classify_capacity(body_text):
                if attempt >= max_capacity_retries:
                    raise FlexError(f"capacity retries exhausted for {custom_id}")
                delay = min(120, 5 * 2**attempt)
                attempt += 1
                sleep(delay)
                continue
            if resp.status_code in _IMMEDIATE_STOP_STATUS:
                raise FlexError(f"HTTP {resp.status_code} for {custom_id}: stopping run")
            if resp.status_code in _RETRYABLE_STATUS:
                if attempt >= _MAX_TRANSIENT_RETRIES:
                    return FlexResult(custom_id, None, 0, 0, 0.0, error=f"HTTP {resp.status_code}")
                delay = min(60, 5 * 2**attempt)
                attempt += 1
                sleep(delay)
                continue
            return FlexResult(custom_id, None, 0, 0, 0.0, error=f"HTTP {resp.status_code}")

    def worker(req: dict) -> FlexResult | None:
        try:
            with lock:
                over_budget = state["cost"] >= max_usd
            if over_budget:
                raise FlexError(f"cost cap ${max_usd} reached; stopping")
            result = execute(req)
        except FlexError as exc:
            stop_event.set()
            with lock:
                if not stop_error:
                    stop_error.append(exc)
            raise _RunStopped() from exc
        append_result(result)
        return result

    own_client = client is None
    client = client or httpx.Client()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, req) for req in pending]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except _RunStopped:
                    pass
    finally:
        if own_client:
            client.close()

    # Collect results appended during the run by re-reading the checkpoint,
    # filtered to this run's request ids (the file may hold rows from other runs).
    results = {k: v for k, v in _load_checkpoint(checkpoint).items() if k in id_set}

    if stop_error:
        raise stop_error[0]

    completed = sum(1 for r in results.values() if r.error is None)
    summary = {
        "requests": total,
        "completed": completed,
        "failed": len(results) - completed,
        "resumed": len(done),
        "cost_usd": sum(r.cost_usd for r in results.values()),
        "seconds": time.monotonic() - started,
        "model": model,
    }
    return results, summary
