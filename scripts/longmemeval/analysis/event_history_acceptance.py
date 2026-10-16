#!/usr/bin/env python3
"""Durable NONCANONICAL live Bench runner for Menhir event-history acceptance.

Loads a versioned acceptance fixture, materializes ``ExperimentCase`` objects from the frozen
LongMemEval knowledge-update subset, and evaluates each through the real Menhir event-history
perception -> projection -> selection path using ``archolith_bench.event_history_acceptance``.

The source fixture identity is validated against BOTH the raw byte SHA-256 and an EOL-canonical
SHA-256 (every line ending normalized to CRLF), so a checkout whose git EOL policy produced an
LF working copy is still recognized.  Only when neither matches the declared hash does the runner
fail loudly.

This runner is Bench-only and NONCANONICAL.  Query routing, production authority, repository
persistence, and public recall are explicitly unmeasured and disabled.  It makes live direct
OpenAI chat-completion calls only for event perception, records per-call provider usage, and stops
immediately on a 429 rate limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archolith_bench.event_history_acceptance import (  # noqa: E402
    ExperimentCase,
    ExperimentEpisode,
    ProbeEventHistoryApi,
    analyze_case,
    load_menhir_event_history_api,
)
from archolith_bench.deterministic_scalar_shadow import (  # noqa: E402
    _git_metadata,
    resolve_menhir_root,
)
from archolith_bench.core.api import (  # noqa: E402
    call_with_flex_retry,
    is_flex_capacity_error,
    openai_chat_kwargs,
)

FIXTURE_SCHEMA_VERSION = 1
# gpt-4o-mini is deprecated. A reasoning model takes no temperature, so on Luna the vote samples
# come from the model's default sampling, not TEMPERATURE_DEFAULT; the report records the
# temperature as null in that case. Vote thresholds were calibrated on mini at 0.7.
MODEL_DEFAULT = "gpt-6-luna"
TEMPERATURE_DEFAULT = 0.7
SAMPLES_DEFAULT = 3
REQUIRED_VOTES_DEFAULT = 2
MAX_TOKENS_DEFAULT = 512
LLM_ERROR_RETRIES_DEFAULT = 1


class EventHistoryRunnerError(RuntimeError):
    """Raised when a fixture, source identity, or runtime validation fails."""


def _error(context: str, message: str) -> EventHistoryRunnerError:
    return EventHistoryRunnerError(f"{context}: {message}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _eol_canonical_crlf(data: bytes) -> bytes:
    """Normalize every line ending to CRLF so LF vs CRLF checkouts hash identically."""
    lf = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return lf.replace(b"\n", b"\r\n")


def _normalize_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    value = " ".join(text.strip().lower().split())
    for article in ("a ", "an ", "the "):
        if value.startswith(article):
            value = value[len(article):]
            break
    return value.strip()


LME_TIME_FORMAT = "%Y/%m/%d (%a) %H:%M"


def normalize_reference_time(raw: str) -> str:
    """Normalize a dataset ``reference_time`` to canonical UTC ISO-8601.

    This is a generic dataset-adapter shim over the two timestamp layouts a source
    fixture may carry:

    * LongMemEval ``%Y/%m/%d (%a) %H:%M`` (e.g. ``2023/08/30 (Wed) 04:01``) as used
      by knowledge-update ``haystack_dates``.
    * an already-parseable ISO-8601 timestamp, preserved/normalized safely.

    Unrecognized or blank values raise loudly so a missing or malformed time can never
    silently become a ``no_valid_time`` build drop.

    UTC is a dataset convention, not a measured property of the fixture: LongMemEval
    carries no timezone, and ordering — not absolute wall-clock — is the property under
    test. The UTC ``Z`` suffix is attached to give Menhir's event-history selectors a
    comparable, monotonic basis.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise _error("reference_time", "must be a non-empty string")
    value = raw.strip()
    try:
        parsed = datetime.strptime(value, LME_TIME_FORMAT)
    except ValueError:
        parsed = None
    if parsed is not None:
        return parsed.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _error("reference_time", f"unrecognized time format {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_key(_menhir_root: Path) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line[len("OPENAI_API_KEY="):].strip().strip('"').strip("'")
    raise EventHistoryRunnerError("OPENAI_API_KEY not set and not found in archolith-bench/.env")


def load_source_fixture(source_fixture: str | Path) -> list[dict[str, Any]]:
    path = Path(source_fixture).expanduser().resolve()
    if not path.is_file():
        raise _error(f"source fixture {path}", "file does not exist")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise _error(f"source fixture {path}", "must be a JSON array")
    if not data:
        raise _error(f"source fixture {path}", "must not be empty")
    return data


def validate_source_identity(
    source_fixture: str | Path, expected_sha: str
) -> dict[str, Any]:
    path = Path(source_fixture).expanduser().resolve()
    raw = path.read_bytes()
    raw_sha = _sha256_bytes(raw)
    canonical_sha = _sha256_bytes(_eol_canonical_crlf(raw))
    if raw_sha == expected_sha:
        match_mode = "raw"
    elif canonical_sha == expected_sha:
        match_mode = "canonical_crlf"
    else:
        raise _error(
            f"source fixture {path}",
            f"hash mismatch; expected {expected_sha}, raw={raw_sha}, canonical_crlf={canonical_sha}",
        )
    return {
        "expected_sha256": expected_sha,
        "raw_sha256": raw_sha,
        "canonical_crlf_sha256": canonical_sha,
        "match_mode": match_mode,
    }


def load_fixture(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    context = f"fixture {resolved}"
    if not resolved.is_file():
        raise _error(context, "file does not exist")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise _error(context, "must be an object")
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise _error(context, f"schema_version must be {FIXTURE_SCHEMA_VERSION}")
    if payload.get("noncanonical") is not True:
        raise _error(context, "noncanonical must be true")
    if not isinstance(payload.get("source_fixture"), str) or not payload["source_fixture"]:
        raise _error(context, "source_fixture must be a non-empty string")
    if not isinstance(payload.get("source_sha256"), str) or len(payload["source_sha256"]) != 64:
        raise _error(context, "source_sha256 must be a SHA-256 hex string")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise _error(context, "cases must be a non-empty list")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise _error(context, "case_id must be unique")
    return payload


def _materialize_episodes(episode_rows: list[dict[str, Any]], context: str) -> tuple[ExperimentEpisode, ...]:
    episodes: list[ExperimentEpisode] = []
    for index, row in enumerate(episode_rows):
        item_context = f"{context}.episodes[{index}]"
        if not isinstance(row, dict):
            raise _error(item_context, "must be an object")
        uuid = row.get("uuid")
        content = row.get("content")
        if not isinstance(uuid, str) or not uuid:
            raise _error(item_context, "uuid must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise _error(item_context, "content must be a non-empty string")
        reference_time = row.get("reference_time")
        evidence = row.get("turn_evidence_uuid")
        if not isinstance(reference_time, str) or not reference_time:
            raise _error(item_context, "reference_time must be a non-empty string")
        if not isinstance(evidence, str) or not evidence:
            raise _error(item_context, "turn_evidence_uuid must be a non-empty string")
        episodes.append(
            ExperimentEpisode(
                uuid=uuid,
                content=content,
                reference_time=reference_time,
                turn_evidence_uuid=evidence,
            )
        )
    return tuple(episodes)


def build_episodes_from_question(question: dict[str, Any], qid: str) -> tuple[ExperimentEpisode, ...]:
    """Build an ExperimentEpisode from every USER turn in every session."""
    haystack_dates = question.get("haystack_dates") or []
    sessions = question.get("haystack_sessions") or []
    episodes: list[ExperimentEpisode] = []
    for session_index, session in enumerate(sessions):
        reference_time = haystack_dates[session_index] if session_index < len(haystack_dates) else None
        if not isinstance(reference_time, str) or not reference_time:
            raise _error(
                f"question {qid}",
                f"missing haystack_date for session {session_index}",
            )
        reference_time = normalize_reference_time(reference_time)
        for turn_index, turn in enumerate(session):
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            content = turn.get("content")
            if not isinstance(content, str) or not content:
                continue
            uuid = f"{qid}-{session_index}-{turn_index}"
            evidence = f"evidence-{qid}-{session_index}-{turn_index}"
            episodes.append(
                ExperimentEpisode(
                    uuid=uuid,
                    content=content,
                    reference_time=reference_time,
                    turn_evidence_uuid=evidence,
                )
            )
    if not episodes:
        raise _error(f"question {qid}", "no user turns found to materialize episodes")
    return tuple(episodes)


def _find_question(data: list[dict[str, Any]], qid: str) -> dict[str, Any]:
    matches = [q for q in data if q.get("question_id") == qid]
    if len(matches) != 1:
        raise _error(
            f"question {qid}",
            f"expected exactly one source match, found {len(matches)}",
        )
    question = matches[0]
    if question.get("question_type") != "knowledge-update":
        raise _error(f"question {qid}", "question_type must be 'knowledge-update'")
    return question


def _validate_source_answer(question: dict[str, Any], case: dict[str, Any], qid: str) -> None:
    if not case.get("expected_object_key"):
        return
    expected_normalized = _normalize_answer(case["expected_object_key"])
    source_normalized = _normalize_answer(question.get("answer"))
    if expected_normalized != source_normalized:
        raise _error(
            f"question {qid}",
            f"expected source answer mismatch; normalized expected={expected_normalized!r}, "
            f"source={source_normalized!r}",
        )


def materialize_cases(
    fixture: dict[str, Any], source_data: list[dict[str, Any]]
) -> list[tuple[str, ExperimentCase]]:
    cases: list[tuple[str, ExperimentCase]] = []
    for raw in fixture["cases"]:
        case_id = raw.get("case_id")
        qid = raw.get("question_id")
        intent = raw.get("intent")
        expected_status = raw.get("expected_status")
        context = f"fixture case {case_id}"
        if not isinstance(intent, str) or intent not in {"latest", "predecessor"}:
            raise _error(context, "intent must be latest or predecessor")
        if not isinstance(expected_status, str) or expected_status not in {"unique", "none", "ambiguous"}:
            raise _error(context, "expected_status must be unique, none, or ambiguous")
        episodes: tuple[ExperimentEpisode, ...]
        question_text: str | None = None
        if qid:
            if not isinstance(qid, str) or not qid:
                raise _error(context, "question_id must be a non-empty string")
            question = _find_question(source_data, qid)
            _validate_source_answer(question, raw, qid)
            episodes = build_episodes_from_question(question, qid)
            if isinstance(question.get("question"), str) and question["question"].strip():
                question_text = question["question"].strip()
        else:
            episodes = _materialize_episodes(raw.get("episodes") or [], context)
        expected_object_key = raw.get("expected_object_key")
        anchor_object_key = raw.get("anchor_object_key")
        if intent == "predecessor" and not anchor_object_key:
            raise _error(context, "predecessor cases require anchor_object_key")
        if expected_status == "unique" and not expected_object_key:
            raise _error(context, "unique cases require expected_object_key")
        cases.append(
            (
                case_id,
                ExperimentCase(
                    case_id=case_id,
                    namespace="lme-event-history",
                    subject_uuid=f"subject-{case_id}",
                    episodes=episodes,
                    intent=intent,
                    expected_status=expected_status,
                    expected_object_key=expected_object_key,
                    anchor_object_key=anchor_object_key,
                    lane_domain=raw.get("domain"),
                    safety_control=bool(raw.get("safety_control", False)),
                    question=question_text,
                ),
            )
        )
    return cases


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """Serialize the provider's raw usage object for audit, if possible."""
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        try:
            value = dump()
            if isinstance(value, dict):
                return value
        except Exception:  # noqa: BLE001 - fall through to attribute extraction
            pass
    record: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            record[name] = value
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            record["prompt_tokens_details"] = {"cached_tokens": cached}
    return record or None


class UsageRecorder:
    """Records per-call and aggregate OpenAI chat-completion usage telemetry."""

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.request_kwargs = openai_chat_kwargs(
            model, max_tokens=max_tokens, temperature=temperature
        )
        # The temperature actually sent: None when the model does not take one.
        self.temperature = self.request_kwargs.get("temperature")
        self.calls: list[dict[str, Any]] = []
        self._total = Counter()
        self._missing_usage = 0

    def record(self, latency_ms: float, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            self._missing_usage += 1
            record = {
                "usage_missing": True,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cached_tokens": None,
            }
            raw_usage = None
        else:
            prompt = getattr(usage, "prompt_tokens", None) or 0
            completion = getattr(usage, "completion_tokens", None) or 0
            total = getattr(usage, "total_tokens", None) or (prompt + completion)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details is not None else None
            self._total["prompt"] += prompt
            self._total["completion"] += completion
            self._total["total"] += total
            self._total["cached"] += cached or 0
            record = {
                "usage_missing": False,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "cached_tokens": cached,
            }
            raw_usage = _usage_to_dict(usage)
        full = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "latency_ms": round(latency_ms, 3),
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            **record,
            "raw_usage": raw_usage,
        }
        self.calls.append(full)
        return full

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "input_tokens": self._total["prompt"],
            "output_tokens": self._total["completion"],
            "total_tokens": self._total["total"],
            "cached_tokens": self._total["cached"],
            "missing_usage_calls": self._missing_usage,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.request_kwargs.get("max_completion_tokens"),
            "service_tier": self.request_kwargs.get("service_tier"),
            "per_call": list(self.calls),
        }


def make_llm_complete(
    client: Any,
    recorder: UsageRecorder,
) -> Callable[[str, str], str]:
    def llm_complete(system: str, user: str) -> str:
        started = time.monotonic()
        try:
            # Flex capacity 429s are waited out; any other 429 still stops the run.
            response = call_with_flex_retry(
                lambda: client.chat.completions.create(
                    model=recorder.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **recorder.request_kwargs,
                )
            )
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            status_code = getattr(exc, "status_code", None)
            if status_code == 429 and is_flex_capacity_error(exc):
                raise EventHistoryRunnerError(
                    "OpenAI Flex capacity unavailable (429) after retries; stopping"
                ) from exc
            if status_code == 429:
                reset = getattr(getattr(exc, "response", None), "headers", None)
                reset_epoch = reset.get("x-ratelimit-reset-epoch", "unknown") if reset else "unknown"
                raise EventHistoryRunnerError(
                    "OpenAI rate limit (429) reached; stopping immediately "
                    f"(reset_epoch={reset_epoch})"
                ) from exc
            raise
        latency_ms = (time.monotonic() - started) * 1000
        recorder.record(latency_ms, response)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise EventHistoryRunnerError("OpenAI returned empty content")
        return content

    return llm_complete


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, str(path))
    except BaseException:
        if Path(temp_name).exists():
            Path(temp_name).unlink()
        raise


def _bench_metadata() -> dict[str, Any]:
    return _git_metadata(REPO_ROOT, None)


def run_acceptance(
    *,
    fixture_path: str | Path,
    source_fixture: str | Path,
    menhir_root: str | Path,
    output: str | Path,
    model: str = MODEL_DEFAULT,
    temperature: float = TEMPERATURE_DEFAULT,
    samples: int = SAMPLES_DEFAULT,
    required_votes: int = REQUIRED_VOTES_DEFAULT,
    max_tokens: int = MAX_TOKENS_DEFAULT,
    llm_error_retries: int = LLM_ERROR_RETRIES_DEFAULT,
    client: Any = None,
    api: ProbeEventHistoryApi | None = None,
    generated_at: str | None = None,
) -> int:
    fixture = load_fixture(fixture_path)
    source_data = load_source_fixture(source_fixture)
    source_identity = validate_source_identity(source_fixture, fixture["source_sha256"])
    cases = materialize_cases(fixture, source_data)

    if api is None:
        root = resolve_menhir_root(menhir_root)
        api = load_menhir_event_history_api(root)
    else:
        root = Path(menhir_root).expanduser().resolve()
    key = _load_key(root) if client is None else ""
    if client is None:
        import openai

        client = openai.OpenAI(api_key=key)

    if max_tokens < 1:
        raise EventHistoryRunnerError("max_tokens must be positive")
    if llm_error_retries < 0:
        raise EventHistoryRunnerError("llm_error_retries must be non-negative")
    recorder = UsageRecorder(model, temperature, max_tokens)
    llm_complete = make_llm_complete(client, recorder)
    generated = generated_at or datetime.now(timezone.utc).isoformat()

    case_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    safety_violations: list[str] = []
    routing_measured_cases = 0
    for case_id, case in cases:
        report = analyze_case(
            case,
            root,
            llm_complete,
            samples=samples,
            required_votes=required_votes,
            llm_error_retries=llm_error_retries,
            generated_at=generated,
            api=api,
        )
        aggregate = report["aggregate"]
        case_reports.append({"case_id": case_id, **report})
        if report["query_routing_measured"]:
            routing_measured_cases += 1
        if not aggregate["passed"]:
            failures.append(case_id)
        if aggregate["safety_violation"]:
            safety_violations.append(case_id)

    try:
        menhir_metadata = _git_metadata(root, None)
    except Exception:  # noqa: BLE001
        menhir_metadata = {"state": "unavailable"}

    report = {
        "report_schema_version": 2,
        "promotion_status": "not_evaluable",
        "canonical": False,
        "noncanonical": True,
        "production_authority_enabled": False,
        "persistence_used": False,
        "query_routing_measured": routing_measured_cases > 0,
        "routing_coverage": {
            "measured_cases": routing_measured_cases,
            "total_cases": len(cases),
        },
        "llm_used": True,
        "generated_at": generated,
        "provenance": {
            "fixture": {
                "path": str(Path(fixture_path).expanduser().resolve()),
                "schema_version": fixture["schema_version"],
                "panel_id": fixture["panel_id"],
                "raw_sha256": _sha256_bytes(Path(fixture_path).expanduser().resolve().read_bytes()),
            },
            "source_fixture": {
                "path": str(Path(source_fixture).expanduser().resolve()),
                "declared_source_fixture": fixture["source_fixture"],
                "kind": fixture.get("source_fixture_kind"),
                "sha256": source_identity,
            },
            "menhir": menhir_metadata,
            "bench": _bench_metadata(),
            "perceiver_version": fixture.get("perceiver_version"),
            "predicate": fixture.get("predicate"),
            "llm_used": True,
            "reference_time_normalization": {
                "adapter": "generic-dataset-adapter",
                "accepted_formats": [
                    "ISO-8601",
                    "LongMemEval %Y/%m/%d (%a) %H:%M",
                ],
                "emitted_format": "canonical UTC YYYY-MM-DDTHH:MM:SSZ",
                "utc_is_dataset_convention": True,
                "note": (
                    "fixture carries no timezone; ordering is the measured property, "
                    "so UTC Z is attached to give event-history selectors a monotonic basis"
                ),
            },
        },
        "config": {
            "model": model,
            "temperature": recorder.temperature,
            "requested_temperature": temperature,
            "service_tier": recorder.request_kwargs.get("service_tier"),
            "samples": samples,
            "required_votes": required_votes,
            "max_tokens": max_tokens,
            "llm_error_retries": llm_error_retries,
        },
        "cases": case_reports,
        "aggregate": {
            "total_cases": len(cases),
            "passed_cases": len(cases) - len(failures),
            "failed_cases": len(failures),
            "failed_case_ids": failures,
            "safety_violations": safety_violations,
            "llm_error_retries": sum(
                case_report["aggregate"]["llm_error_retries"]
                for case_report in case_reports
            ),
        },
        "usage": recorder.summary(),
    }

    _write_atomic(Path(output).expanduser().resolve(), report)
    if failures or safety_violations:
        return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the NONCANONICAL event-history acceptance probe.")
    parser.add_argument("--fixture", required=True, help="acceptance fixture JSON")
    parser.add_argument("--source-fixture", required=True, help="frozen LongMemEval knowledge-update subset")
    parser.add_argument("--menhir-root", required=True, help="Menhir source checkout")
    parser.add_argument("--output", required=True, help="atomic JSON report path")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="OpenAI chat model")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    parser.add_argument("--samples", type=int, default=SAMPLES_DEFAULT)
    parser.add_argument("--required-votes", type=int, default=REQUIRED_VOTES_DEFAULT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT)
    parser.add_argument(
        "--llm-error-retries", type=int, default=LLM_ERROR_RETRIES_DEFAULT,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run_acceptance(
            fixture_path=args.fixture,
            source_fixture=args.source_fixture,
            menhir_root=args.menhir_root,
            output=args.output,
            model=args.model,
            temperature=args.temperature,
            samples=args.samples,
            required_votes=args.required_votes,
            max_tokens=args.max_tokens,
            llm_error_retries=args.llm_error_retries,
        )
    except EventHistoryRunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
