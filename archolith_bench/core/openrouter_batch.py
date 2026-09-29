"""OpenRouter Batch API client: half-price chat completions for work that need not be live.

Protocol (as used in production by cth.painscan): POST ``/api/beta/batches`` with
``{"endpoint": "/v1/chat/completions", "model": <plain id>, "requests": [{custom_id, body}]}``,
then GET ``/api/beta/batches/{id}`` until the status is terminal; results come back inline under
``results``, each keyed by ``custom_id``.

Safety properties:
* **Never double-submit.** The batch id is written to the checkpoint file *before* anything else
  happens after submission; calling :func:`run_batch` again with the same checkpoint resumes
  polling that batch instead of submitting a new one.
* **Fail closed.** Any terminal status other than ``completed`` raises; so do duplicate or
  unexpected result ids. A request with no usable reply is returned with an ``error`` rather than
  a guessed answer.
* **Cost cap before submission.** A conservative estimate (characters/3 prompt tokens plus
  ``max_tokens`` and a reasoning allowance, at the live catalog price) must fit the cap, or nothing
  is sent. The provider-reported actual cost is returned after collection.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BATCH_URL = "https://openrouter.ai/api/beta/batches"
MODELS_URL = "https://openrouter.ai/api/v1/models"
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})
DEFAULT_MAX_TOKENS = 4_000
DEFAULT_REASONING_ALLOWANCE = 4_000


class BatchError(RuntimeError):
    """The batch cannot produce trustworthy results (terminal failure, bad ids, over cap)."""


@dataclass
class BatchResult:
    custom_id: str
    content: str | None
    usage: dict = field(default_factory=dict)
    error: str = ""


def find_chat_document(result: dict) -> dict | None:
    """The chat-completion document inside one batch result, wherever the provider nests it."""
    if not isinstance(result, dict):
        return None
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    candidates = [
        response.get("body"),
        (nested.get("response") or {}).get("body") if isinstance(nested.get("response"), dict) else None,
        nested.get("body"),
        result.get("body"),
        response,
        nested,
        result,
    ]
    return next((c for c in candidates if isinstance(c, dict) and "choices" in c), None)


def parse_result(result: dict) -> BatchResult:
    custom_id = str(result.get("custom_id") or "")
    document = find_chat_document(result)
    if document is None:
        response = result.get("response") if isinstance(result.get("response"), dict) else {}
        err = result.get("error") or response.get("error")
        return BatchResult(custom_id, None, {}, f"missing_chat_completion: {err}" if err else "missing_chat_completion")
    try:
        content = document["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        return BatchResult(custom_id, None, {}, f"malformed_chat_completion: {exc}")
    return BatchResult(custom_id, str(content or ""), dict(document.get("usage") or {}), "")


def estimate_max_usd(requests: list[dict], prompt_rate: float, completion_rate: float,
                     reasoning_allowance: int = DEFAULT_REASONING_ALLOWANCE) -> float:
    """Conservative upper bound: ~chars/3 prompt tokens, plus max_tokens + reasoning per request."""
    total = 0.0
    for req in requests:
        body = req["body"]
        total += len(json.dumps(body.get("messages", []))) / 3 * prompt_rate
        total += (int(body.get("max_tokens") or DEFAULT_MAX_TOKENS) + reasoning_allowance) * completion_rate
    return total


def catalog_rates(model: str, client: httpx.Client, api_key: str) -> tuple[float, float]:
    """Per-token prompt/completion price for the :batch variant of ``model`` (live catalog)."""
    resp = client.get(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    resp.raise_for_status()
    wanted = model if model.endswith(":batch") else f"{model}:batch"
    for entry in resp.json().get("data", []):
        if entry.get("id") == wanted:
            pricing = entry.get("pricing") or {}
            return float(pricing.get("prompt") or 0), float(pricing.get("completion") or 0)
    raise BatchError(f"batch model not in the live OpenRouter catalog: {wanted}")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def run_batch(
    requests: list[dict],
    *,
    model: str,
    checkpoint: Path,
    api_key: str,
    max_usd: float,
    client: httpx.Client | None = None,
    poll_s: float = 30.0,
    timeout_s: float = 24 * 3600,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rates: tuple[float, float] | None = None,
    log: Callable[[str], None] = print,
    not_found_grace_s: float = 600.0,
) -> tuple[dict[str, BatchResult], dict]:
    """Submit (or resume) one batch and block until its results are collected.

    ``requests`` are ``{"custom_id": str, "body": {chat-completion body without "model"}}``. Returns
    ``(results_by_custom_id, summary)``. Raises :class:`BatchError` on a non-completed terminal
    status, a timeout, duplicate/unexpected result ids, or an estimate over ``max_usd``.
    """
    ids = [str(r["custom_id"]) for r in requests]
    if len(ids) != len(set(ids)):
        raise BatchError("custom_id values must be unique")
    own = client is None
    client = client or httpx.Client(timeout=180)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    base_model = model.removesuffix(":batch")
    try:
        state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
        if state is not None:
            if sorted(state.get("custom_ids") or []) != sorted(ids):
                raise BatchError(f"checkpoint {checkpoint} belongs to a different request set")
            log(f"resuming batch {state['batch_id']} (no resubmission)")
        else:
            prompt_rate, completion_rate = rates or catalog_rates(base_model, client, api_key)
            estimate = estimate_max_usd(requests, prompt_rate, completion_rate)
            if estimate > max_usd:
                raise BatchError(f"conservative estimate ${estimate:.4f} exceeds cap ${max_usd:.4f}")
            payload = {"endpoint": "/v1/chat/completions", "model": base_model,
                       "requests": [{"custom_id": r["custom_id"], "body": {"model": base_model, **r["body"]}}
                                    for r in requests]}
            resp = client.post(BATCH_URL, headers=headers, json=payload, timeout=180)
            if resp.status_code == 429:
                raise BatchError(f"429 on submit: {resp.text[:300]}")
            resp.raise_for_status()
            created = resp.json()
            batch_id = created.get("id")
            if not batch_id:
                raise BatchError(f"submit returned no batch id: {str(created)[:300]}")
            state = {"batch_id": batch_id, "custom_ids": ids, "model": base_model,
                     "estimate_usd": estimate, "submitted_at": time.time(), "status": created.get("status")}
            _write_atomic(checkpoint, state)
            log(f"submitted batch {batch_id}: {len(ids)} requests, estimate <= ${estimate:.4f}")

        deadline = clock() + timeout_s
        visible_by = clock() + not_found_grace_s
        while True:
            polled = client.get(f"{BATCH_URL}/{state['batch_id']}", headers=headers, timeout=120)
            if polled.status_code == 429:
                raise BatchError(f"429 on poll: {polled.text[:300]}")
            if polled.status_code == 404 and clock() < visible_by:
                # A just-created batch can 404 for a short while before it becomes visible.
                sleep(poll_s)
                continue
            polled.raise_for_status()
            data = polled.json()
            status = data.get("status")
            if status in TERMINAL_STATUSES:
                break
            if clock() >= deadline:
                raise BatchError(f"batch {state['batch_id']} not finished after {timeout_s:.0f}s (status {status})")
            sleep(poll_s)
        state.update({"status": status, "finalized_at": data.get("finalized_at"), "usage": data.get("usage")})
        _write_atomic(checkpoint, state)
        if status != "completed":
            raise BatchError(f"batch {state['batch_id']} ended with status {status}")

        results: dict[str, BatchResult] = {}
        for raw in data.get("results") or []:
            parsed = parse_result(raw)
            if parsed.custom_id in results:
                raise BatchError(f"duplicate result for {parsed.custom_id}")
            if parsed.custom_id not in set(ids):
                raise BatchError(f"unexpected result id {parsed.custom_id}")
            results[parsed.custom_id] = parsed
        for missing in set(ids) - set(results):
            results[missing] = BatchResult(missing, None, {}, "missing_result")
        summary = {"batch_id": state["batch_id"], "status": status, "requests": len(ids),
                   "errors": sum(1 for r in results.values() if r.error),
                   "actual_cost_usd": (data.get("usage") or {}).get("cost"),
                   "estimate_usd": state.get("estimate_usd"),
                   "seconds": round(time.time() - float(state.get("submitted_at") or time.time()), 1)}
        return results, summary
    finally:
        if own:
            client.close()
