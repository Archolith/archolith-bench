#!/usr/bin/env python3
"""Write a stable aggregate of Menhir's provider-reported LLM usage telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


def _token_count(usage: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def summarize_harness_usage(
    checkpoint_path: Path,
    *,
    require_judge_usage: bool = False,
) -> dict[str, Any]:
    """Aggregate successful answer and optional judge calls captured by the checkpoint."""

    if not checkpoint_path.exists():
        return {"available": False, "reason": "harness_checkpoint_missing"}

    totals = {
        "calls": 0,
        "missing_usage_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    by_arm: dict[str, dict[str, int | str]] = {}
    by_operation: dict[str, dict[str, int | str]] = {}

    def add_usage(arm: str, operation: str, usage: dict[str, Any]) -> None:
        input_tokens = _token_count(usage, "prompt_tokens", "input_tokens")
        output_tokens = _token_count(usage, "completion_tokens", "output_tokens")
        total_tokens = _token_count(usage, "total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        prompt_details = usage.get("prompt_tokens_details")
        completion_details = usage.get("completion_tokens_details")
        cached_tokens = _token_count(
            prompt_details if isinstance(prompt_details, dict) else {}, "cached_tokens"
        )
        reasoning_tokens = _token_count(
            completion_details if isinstance(completion_details, dict) else {},
            "reasoning_tokens",
        )

        arm_totals = by_arm.setdefault(arm, _new_harness_group("arm", arm))
        operation_totals = by_operation.setdefault(
            operation, _new_harness_group("operation", operation)
        )
        for target in (totals, arm_totals, operation_totals):
            target["calls"] += 1
            if total_tokens is None:
                target["missing_usage_calls"] += 1
            target["input_tokens"] += input_tokens or 0
            target["output_tokens"] += output_tokens or 0
            target["total_tokens"] += total_tokens or 0
            target["cached_input_tokens"] += cached_tokens or 0
            target["reasoning_output_tokens"] += reasoning_tokens or 0

    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm = str(row.get("arm") or "unknown")
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        usage = result.get("raw_usage") if isinstance(result.get("raw_usage"), dict) else {}
        add_usage(arm, "answer", usage)
        scorer_usage = (
            result.get("scorer_raw_usage")
            if isinstance(result.get("scorer_raw_usage"), dict)
            else {}
        )
        if scorer_usage:
            add_usage(arm, "judge", scorer_usage)
        elif require_judge_usage:
            add_usage(arm, "judge", {})

    return {
        "available": True,
        "checkpoint": str(checkpoint_path),
        **totals,
        "by_arm": sorted(by_arm.values(), key=lambda item: str(item["arm"])),
        "by_operation": sorted(
            by_operation.values(), key=lambda item: str(item["operation"])
        ),
    }


def _new_harness_group(label: str, value: str) -> dict[str, int | str]:
    return {
        label: value,
        "calls": 0,
        "missing_usage_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }



# Per-model rates in USD per 1M tokens, for costing Menhir's ingest traffic. Static and dated,
# matching core/metrics.py's convention -- no live pricing lookups.
#
# core.metrics.PRICING_DEFAULTS is not reused here because it is keyed by *provider* and its
# "openai" entry is gpt-4o ($2.50/$10.00), the answer model. Ingest runs on gpt-4o-mini, 16x
# cheaper on input; pricing ingest at the gpt-4o rate would overstate a 78-item buildout by
# roughly an order of magnitude.
#
# Embeddings have no output tokens, so their output rate is 0.0 rather than unknown.
# (input, output, cached_input) in USD per 1M tokens.
#
# `cached_input` is load-bearing, not a refinement: provider usage reports `input_tokens` as the
# FULL prompt with `cached_input_tokens` as a SUBSET of it (OpenAI's prompt_tokens vs
# prompt_tokens_details.cached_tokens; see menhir observability._normalized_usage). Pricing all
# input at the full rate therefore overcharges every cached call -- and for this workload input
# is ~96% of tokens, so a cache-heavy run would be reported far above what it actually cost.
#
# A model with no published cache rate gets its full input rate here. Never invent a discount:
# assuming one understates real spend, which is the direction that matters.
INGEST_RATES_USD_PER_1M: dict[str, tuple[float, float, float]] = {
    # chat -- OpenAI direct
    "gpt-4o": (2.50, 10.00, 1.25),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4.1-mini": (0.40, 1.60, 0.10),
    "gpt-4.1-nano": (0.10, 0.40, 0.025),
    # chat -- OpenRouter slugs. Fetched from openrouter.ai/api/v1/models on 2026-09-08:
    # prompt 2e-7/token = $0.20/M, completion 1.2e-6 = $1.20/M, cache read 2e-8 = $0.02/M.
    # luna and luna-pro are priced identically, so Pro is capability upside at no extra cost.
    "openai/gpt-5.6-luna": (0.20, 1.20, 0.02),
    "openai/gpt-5.6-luna-pro": (0.20, 1.20, 0.02),
    # The :batch variants are exactly 50% off. They are an ASYNCHRONOUS submit-and-poll API
    # (202 Accepted, status "validating"), so they are priced here for comparison but are NOT
    # reachable from Menhir's ingest, whose extraction calls are sequentially dependent.
    "openai/gpt-5.6-luna:batch": (0.10, 0.60, 0.01),
    "openai/gpt-5.6-luna-pro:batch": (0.10, 0.60, 0.01),
    # embeddings -- no output tokens, so the output rate is 0.0 rather than unknown
    "text-embedding-3-small": (0.02, 0.0, 0.02),
    "text-embedding-3-large": (0.13, 0.0, 0.13),
}
RATES_DATED = (
    "OpenAI rates as of 2026-09 (openai.com/api/pricing); OpenRouter slugs fetched from "
    "openrouter.ai/api/v1/models 2026-09-08"
)


def price_usage(by_model: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost each (kind, model) row, and say plainly what could not be priced.

    An unknown model is reported, never priced at zero. Silently costing it at 0.0 would make
    a run with an unrecognised model look cheaper than one without it, which is the opposite of
    what a cost record is for.
    """
    priced: list[dict[str, Any]] = []
    total = 0.0
    unpriced: dict[str, int] = {}
    for row in by_model:
        model = str(row.get("model") or "")
        rates = INGEST_RATES_USD_PER_1M.get(model)
        if rates is None:
            unpriced[model] = unpriced.get(model, 0) + int(row.get("calls") or 0)
            priced.append({**row, "cost_usd": None})
            continue
        input_rate, output_rate, cached_rate = rates
        total_input = int(row.get("input_tokens") or 0)
        cached_input = int(row.get("cached_input_tokens") or 0)
        # cached_input is a subset of total_input; clamp so a provider quirk cannot make the
        # fresh half negative and silently credit the bill.
        cached_input = max(0, min(cached_input, total_input))
        fresh_input = total_input - cached_input
        cost = (
            fresh_input / 1_000_000 * input_rate
            + cached_input / 1_000_000 * cached_rate
            + int(row.get("output_tokens") or 0) / 1_000_000 * output_rate
        )
        total += cost
        priced.append({**row, "cost_usd": round(cost, 6)})
    return {
        "by_model": priced,
        "cost_usd": round(total, 6),
        "rates_dated": RATES_DATED,
        # Present and non-empty means `cost_usd` is a floor, not the total.
        "unpriced_models": dict(sorted(unpriced.items())),
        "unpriced_calls": sum(unpriced.values()),
    }

def summarize_llm_usage(db_path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "provider_reported",
        "run_id": run_id,
    }
    if not db_path.exists():
        return {**payload, "available": False, "reason": "telemetry_db_missing"}

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'llm_usage_events'"
        ).fetchone()
        if table is None:
            return {**payload, "available": False, "reason": "llm_usage_events_missing"}
        where = "WHERE run_id = ?" if run_id is not None else ""
        params: tuple[Any, ...] = (run_id,) if run_id is not None else ()
        totals = conn.execute(
            f"""
            SELECT COUNT(*) AS calls,
                   SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_calls,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_calls,
                   SUM(CASE WHEN status = 'completed' AND total_tokens IS NULL THEN 1 ELSE 0 END)
                       AS missing_usage_calls,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(total_tokens), 0) AS total_tokens,
                   COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                   COALESCE(SUM(reasoning_output_tokens), 0) AS reasoning_output_tokens
            FROM llm_usage_events
            {where}
            """,
            params,
        ).fetchone()
        by_model = conn.execute(
            f"""
            SELECT kind, model, endpoint, COUNT(*) AS calls,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   COALESCE(SUM(total_tokens), 0) AS total_tokens,
                   COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                   COALESCE(SUM(reasoning_output_tokens), 0) AS reasoning_output_tokens
            FROM llm_usage_events
            {where}
            GROUP BY kind, model, endpoint
            ORDER BY total_tokens DESC, calls DESC
            """,
            params,
        ).fetchall()

    rows = [dict(row) for row in by_model]

    # A run_id filter that matches nothing, against a table that HAS rows, is a wiring bug --
    # not a free run. Returning zeros here would report $0.00 for a run that really spent money,
    # the most dangerous possible cost record because it reads as authoritative. Caught when
    # build_graph.sh passed the container name while Menhir stamps MENHIR_BENCH_ACTIVE_RUN_ID.
    if run_id is not None and not rows:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) AS n FROM llm_usage_events").fetchone()["n"]
            known = [
                str(r["run_id"])
                for r in conn.execute(
                    "SELECT DISTINCT run_id FROM llm_usage_events LIMIT 10"
                ).fetchall()
            ]
        if total:
            return {
                **payload,
                "available": False,
                "reason": "run_id_matched_no_rows",
                "rows_in_table": total,
                "run_ids_present": known,
            }

    pricing = price_usage(rows)
    return {
        **payload,
        "available": True,
        **dict(totals),
        **pricing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry_db", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--harness-checkpoint", type=Path)
    parser.add_argument("--require-judge-usage", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    menhir = summarize_llm_usage(args.telemetry_db, run_id=args.run_id)
    if args.harness_checkpoint is None:
        summary = menhir
    else:
        harness = summarize_harness_usage(
            args.harness_checkpoint,
            require_judge_usage=args.require_judge_usage,
        )
        combined = {
            field: int(menhir.get(field, 0)) + int(harness.get(field, 0))
            for field in (
                "calls",
                "missing_usage_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_input_tokens",
                "reasoning_output_tokens",
            )
        }
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "provider_reported",
            "run_id": args.run_id,
            "menhir": menhir,
            "harness": harness,
            "combined": combined,
            "complete": (
                bool(menhir.get("available"))
                and bool(harness.get("available"))
                and combined["missing_usage_calls"] == 0
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
