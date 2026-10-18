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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def test_cached_input_is_priced_at_the_cache_rate(tmp_path) -> None:
    """input_tokens is the FULL prompt and cached_input_tokens a SUBSET of it, so pricing all
    input at the full rate overcharges every cached call. For this workload input is ~96% of
    tokens, so that error would swamp the comparison it exists to support."""
    db = _usage_db(tmp_path, [
        # 1M prompt of which 900k cache-read, on Luna: 100k @ $0.20/M + 900k @ $0.02/M
        ("a", "t", "r", "chat", "openai/gpt-5.6-luna", "chat", "completed",
         1_000_000, 0, 1_000_000, 900_000, 0),
    ])
    cost = summarize_llm_usage(db, run_id="r")["cost_usd"]
    assert cost == round(0.1 * 0.20 + 0.9 * 0.02, 6)


def test_fully_uncached_luna_costs_the_full_input_rate(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "openai/gpt-5.6-luna", "chat", "completed",
         1_000_000, 0, 1_000_000, 0, 0),
    ])
    assert summarize_llm_usage(db, run_id="r")["cost_usd"] == 0.20


def test_cached_exceeding_total_cannot_credit_the_bill(tmp_path) -> None:
    """A provider quirk reporting cached > prompt must not drive the fresh half negative."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "openai/gpt-5.6-luna", "chat", "completed",
         1_000, 0, 1_000, 999_999, 0),
    ])
    cost = summarize_llm_usage(db, run_id="r")["cost_usd"]
    assert cost >= 0
    assert cost == round(1_000 / 1_000_000 * 0.02, 6)


def test_batch_slug_is_half_the_sync_rate(tmp_path) -> None:
    """Priced for comparison only -- :batch is an async submit-and-poll API that Menhir's
    sequentially-dependent ingest cannot use."""
    def rows(model):
        return [("a", "t", "r", "chat", model, "chat", "completed",
                 1_000_000, 1_000_000, 2_000_000, 0, 0)]
    sync = summarize_llm_usage(_usage_db(tmp_path / "s", rows("openai/gpt-5.6-luna")), run_id="r")
    batch = summarize_llm_usage(
        _usage_db(tmp_path / "b", rows("openai/gpt-5.6-luna:batch")), run_id="r"
    )
    assert batch["cost_usd"] == round(sync["cost_usd"] / 2, 6)


def test_gpt6_luna_is_priced_at_the_published_standard_rate(tmp_path) -> None:
    """$0.10 in / $0.01 cached / $0.50 out per 1M (developers.openai.com, 2026-10-15)."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 400_000, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["unpriced_calls"] == 0
    assert summary["cost_usd"] == round(0.6 * 0.10 + 0.4 * 0.01 + 1.0 * 0.50, 6)


def test_flex_figure_halves_luna_and_leaves_other_models_at_their_rate(tmp_path) -> None:
    """The tier is not recorded per call, so Flex is reported beside Standard, never instead."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
        ("b", "t", "r", "embedding", "text-embedding-3-small", "embed", "completed",
         1_000_000, 0, 1_000_000, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["cost_usd"] == round(0.60 + 0.02, 6)
    assert summary["cost_usd_flex"] == round(0.30 + 0.02, 6)
    assert summary["flex_rate_models"] == ["gpt-6-luna"]


def test_unpriced_rows_carry_no_flex_figure(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "some-new-model", "chat", "completed", 10, 1, 11, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["by_model"][0]["cost_usd_flex"] is None
    assert summary["cost_usd_flex"] == 0.0


def test_failures_are_counted_by_error_code_when_recorded(tmp_path) -> None:
    db = _usage_db(tmp_path, [])
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE llm_usage_events ADD COLUMN error_code TEXT")
        conn.executemany(
            "INSERT INTO llm_usage_events (call_id, recorded_at, run_id, kind, model, status,"
            " error_code) VALUES (?, 't', ?, 'chat', 'gpt-6-luna', ?, ?)",
            [
                ("a", "r", "failed", "RateLimitError:429:resource_unavailable"),
                ("b", "r", "failed", "RateLimitError:429:resource_unavailable"),
                ("c", "r", "failed", None),
                ("d", "r", "completed", None),
                ("e", "other-run", "failed", "APITimeoutError"),
            ],
        )
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["error_code_recorded"] is True
    assert summary["failed_by_error_code"] == {
        "RateLimitError:429:resource_unavailable": 2,
        "unrecorded": 1,
    }
    unfiltered = summarize_llm_usage(db)
    assert unfiltered["failed_by_error_code"]["APITimeoutError"] == 1


def test_a_sidecar_without_error_code_says_so(tmp_path) -> None:
    """An empty breakdown from an old sidecar must not read as 'no failures'."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "failed", 0, 0, 0, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["failed_calls"] == 1
    assert summary["error_code_recorded"] is False
    assert summary["failed_by_error_code"] == {}


def _set_usage_json(db, call_id, usage) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE llm_usage_events SET provider_usage_json = ? WHERE call_id = ?",
            (json.dumps(usage), call_id),
        )


def test_cache_writes_are_priced_at_the_write_rate_from_provider_json(tmp_path) -> None:
    """Old-schema DB: cache_write_tokens come from provider_usage_json."""
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 0, 1_000_000, 300_000, 0),
    ])
    _set_usage_json(db, "a", {
        "prompt_tokens": 1_000_000,
        "prompt_tokens_details": {"cached_tokens": 300_000, "cache_write_tokens": 200_000},
    })
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["cache_write_input_tokens"] == 200_000
    assert summary["by_model"][0]["cache_write_input_tokens"] == 200_000
    # fresh 500k @ 0.10 + cached 300k @ 0.01 + write 200k @ 0.125
    assert summary["cost_usd"] == round(0.5 * 0.10 + 0.3 * 0.01 + 0.2 * 0.125, 6)
    assert summary["cost_usd_flex"] == round(0.5 * 0.05 + 0.3 * 0.005 + 0.2 * 0.0625, 6)
    assert summary["tier_source"] == "unrecorded"


def test_cache_write_column_is_used_when_present(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 0, 1_000_000, 0, 0),
    ])
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE llm_usage_events ADD COLUMN cache_write_input_tokens INTEGER")
        conn.execute("UPDATE llm_usage_events SET cache_write_input_tokens = 400000")
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["cache_write_input_tokens"] == 400_000
    assert summary["cost_usd"] == round(0.6 * 0.10 + 0.4 * 0.125, 6)


def test_cache_write_cannot_drive_fresh_negative(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed", 1_000, 0, 1_000, 600, 0),
    ])
    _set_usage_json(db, "a", {"prompt_tokens_details": {"cache_write_tokens": 999_999}})
    cost = summarize_llm_usage(db, run_id="r")["cost_usd"]
    assert cost == round(600 / 1e6 * 0.01 + 400 / 1e6 * 0.125, 6)


def test_other_models_write_at_their_input_rate(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-4o-mini", "chat", "completed",
         1_000_000, 0, 1_000_000, 0, 0),
    ])
    _set_usage_json(db, "a", {"prompt_tokens_details": {"cache_write_tokens": 500_000}})
    assert summarize_llm_usage(db, run_id="r")["cost_usd"] == 0.15


def test_recorded_service_tier_prices_flex_rows_at_flex(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
        ("b", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
        ("c", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
    ])
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE llm_usage_events ADD COLUMN service_tier TEXT")
        conn.execute("UPDATE llm_usage_events SET service_tier = 'flex' WHERE call_id = 'a'")
        conn.execute("UPDATE llm_usage_events SET service_tier = 'default' WHERE call_id = 'b'")
        # call c keeps a NULL tier and must price at Standard
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["tier_source"] == "recorded"
    assert summary["cost_usd"] == round(0.30 + 0.60 + 0.60, 6)
    assert summary["cost_usd_standard"] == round(0.60 * 3, 6)
    assert summary["cost_usd_flex"] == round(0.30 * 3, 6)
    assert {r["service_tier"] for r in summary["by_model"]} == {"flex", "default", None}


def test_unrecorded_tier_keeps_standard_headline(tmp_path) -> None:
    db = _usage_db(tmp_path, [
        ("a", "t", "r", "chat", "gpt-6-luna", "chat", "completed",
         1_000_000, 1_000_000, 2_000_000, 0, 0),
    ])
    summary = summarize_llm_usage(db, run_id="r")
    assert summary["tier_source"] == "unrecorded"
    assert summary["cost_usd"] == summary["cost_usd_standard"] == 0.60
    assert summary["cost_usd_flex"] == 0.30


def test_luna_and_luna_pro_are_priced_identically(tmp_path) -> None:
    def rows(model):
        return [("a", "t", "r", "chat", model, "chat", "completed",
                 500_000, 10_000, 510_000, 0, 0)]
    a = summarize_llm_usage(_usage_db(tmp_path / "a", rows("openai/gpt-5.6-luna")), run_id="r")
    b = summarize_llm_usage(
        _usage_db(tmp_path / "p", rows("openai/gpt-5.6-luna-pro")), run_id="r"
    )
    assert a["cost_usd"] == b["cost_usd"]
