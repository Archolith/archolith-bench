from __future__ import annotations

import json
import sqlite3

from scripts.longmemeval.lib.summarize_llm_usage import (
    summarize_harness_usage,
    summarize_llm_usage,
)


def test_summarize_llm_usage_reports_provider_counts(tmp_path) -> None:
    db_path = tmp_path / "telemetry.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_usage_events (
                call_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, run_id TEXT,
                episode_uuid TEXT, operation TEXT, kind TEXT NOT NULL, model TEXT,
                endpoint TEXT, status TEXT NOT NULL, duration_ms INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
                cached_input_tokens INTEGER, reasoning_output_tokens INTEGER,
                provider_usage_json TEXT, error TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO llm_usage_events (
                call_id, recorded_at, run_id, kind, model, endpoint, status,
                input_tokens, output_tokens, total_tokens, cached_input_tokens,
                reasoning_output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a", "2026-08-07T00:00:00Z", "run-a", "chat", "gpt", "chat", "completed", 100, 20, 120, 40, 5),
                ("b", "2026-08-07T00:00:01Z", "run-a", "embedding", "embed", "embed", "completed", 10, 0, 10, 0, 0),
                ("c", "2026-08-07T00:00:02Z", "run-b", "chat", "gpt", "chat", "completed", 999, 1, 1000, 0, 0),
            ],
        )

    summary = summarize_llm_usage(db_path, run_id="run-a")

    assert summary["available"] is True
    assert summary["calls"] == 2
    assert summary["input_tokens"] == 110
    assert summary["output_tokens"] == 20
    assert summary["total_tokens"] == 130
    assert summary["cached_input_tokens"] == 40
    assert summary["reasoning_output_tokens"] == 5
    assert len(summary["by_model"]) == 2


def test_summarize_llm_usage_marks_legacy_database_unavailable(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path):
        pass

    summary = summarize_llm_usage(db_path, run_id="legacy")

    assert summary["available"] is False
    assert summary["reason"] == "llm_usage_events_missing"


def test_summarize_harness_usage_groups_exact_provider_counts(tmp_path) -> None:
    checkpoint = tmp_path / ".checkpoint.jsonl"
    rows = [
        {
            "arm": "no_memory",
            "result": {
                "raw_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            },
        },
        {
            "arm": "menhir_recall",
            "result": {
                "raw_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
                "scorer_raw_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 1,
                    "total_tokens": 8,
                },
            },
        },
    ]
    checkpoint.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = summarize_harness_usage(checkpoint)

    assert summary["available"] is True
    assert summary["calls"] == 3
    assert summary["input_tokens"] == 37
    assert summary["output_tokens"] == 6
    assert summary["total_tokens"] == 43
    assert summary["cached_input_tokens"] == 4
    assert summary["reasoning_output_tokens"] == 1
    assert [row["arm"] for row in summary["by_arm"]] == [
        "menhir_recall",
        "no_memory",
    ]
    assert [row["operation"] for row in summary["by_operation"]] == [
        "answer",
        "judge",
    ]


def test_summarize_harness_usage_marks_required_missing_judge_usage(tmp_path) -> None:
    checkpoint = tmp_path / ".checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "arm": "menhir_recall",
                "result": {
                    "raw_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_harness_usage(checkpoint, require_judge_usage=True)

    assert summary["calls"] == 2
    assert summary["missing_usage_calls"] == 1
    judge = next(row for row in summary["by_operation"] if row["operation"] == "judge")
    assert judge["calls"] == 1
    assert judge["missing_usage_calls"] == 1


# ---------------------------------------------------------------------------
# Ingest cost
#
# Recall cost was always provider-reported in results.md; ingest cost was not surfaced at all,
# so a 78-item buildout had an exactly-known $0.40 recall figure and an ingest figure nobody
# could state. These cases cover the ways the new pricing could misreport it.
# ---------------------------------------------------------------------------

def _usage_db(tmp_path, rows):
    db_path = tmp_path / "telemetry.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_usage_events (
                call_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, run_id TEXT,
                episode_uuid TEXT, operation TEXT, kind TEXT NOT NULL, model TEXT,
                endpoint TEXT, status TEXT NOT NULL, duration_ms INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
                cached_input_tokens INTEGER, reasoning_output_tokens INTEGER,
                provider_usage_json TEXT, error TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO llm_usage_events (
                call_id, recorded_at, run_id, kind, model, endpoint, status,
                input_tokens, output_tokens, total_tokens, cached_input_tokens,
                reasoning_output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return db_path


def test_ingest_cost_is_priced_per_model(tmp_path) -> None:
    """gpt-4o-mini at $0.15/$0.60 per 1M: 1M in + 1M out = $0.75."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-4o-mini", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["cost_usd"] == 0.75
    assert summary["unpriced_calls"] == 0


def test_ingest_is_not_priced_at_the_answer_model_rate(tmp_path) -> None:
    """core.metrics PRICING_DEFAULTS["openai"] is gpt-4o ($2.50/$10.00). Pricing ingest there
    would overstate a buildout by roughly an order of magnitude."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-4o-mini", "chat", "completed", 1_000_000, 0, 1_000_000, 0, 0),
    ])
    assert summarize_llm_usage(db, run_id="r")["cost_usd"] == 0.15


def test_embeddings_cost_their_input_only(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "embedding", "text-embedding-3-small", "embed", "completed",
         1_000_000, 0, 1_000_000, 0, 0),
    ])
    assert summarize_llm_usage(db, run_id="r")["cost_usd"] == 0.02


def test_an_unknown_model_is_reported_not_priced_at_zero(tmp_path) -> None:
    """Costing an unrecognised model at 0.0 would make a run carrying one look CHEAPER than
    one without it -- the opposite of what a cost record is for."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-4o-mini", "chat", "completed", 1_000_000, 0, 1_000_000, 0, 0),
        ("b", "t", "r", "chat", "some-new-model", "chat", "completed", 9_000_000, 0, 9_000_000, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["unpriced_models"] == {"some-new-model": 1}
    assert summary["unpriced_calls"] == 1
    # The priced total stands as a floor, and the unknown row carries no fabricated number.
    assert summary["cost_usd"] == 0.15
    unknown = [r for r in summary["by_model"] if r["model"] == "some-new-model"][0]
    assert unknown["cost_usd"] is None


def test_rates_are_dated(tmp_path) -> None:
    """Static rates must say when they were taken, matching core/metrics.py's convention."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-4o-mini", "chat", "completed", 10, 1, 11, 0, 0),
    ])
    assert "2026" in summarize_llm_usage(db, run_id="r")["rates_dated"]
