"""Tests for the NONCANONICAL event-history acceptance CLI runner.

Uses a temp synthetic LongMemEval-style source fixture and a fake OpenAI client so no network,
key, or live Menhir service is required.  Verifies loading, materialization, source-identity hash
validation, usage aggregation, atomic output, and exit behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.longmemeval.analysis.event_history_acceptance import (
    _eol_canonical_crlf,
    _normalize_answer,
    _sha256_bytes,
    build_episodes_from_question,
    load_fixture,
    materialize_cases,
    normalize_reference_time,
    run_acceptance,
    validate_source_identity,
)

from archolith_bench.event_history_acceptance import ProbeEventHistoryApi


def _source_item(qid: str = "41698283", answer: str = "a 70-200mm zoom lens") -> dict:
    return {
        "question_id": qid,
        "question_type": "knowledge-update",
        "question": f"question for {qid}",
        "answer": answer,
        "question_date": "2023/01/01 (Sun) 00:00",
        "haystack_dates": ["2023/03/11 (Sat) 22:01"],
        "haystack_session_ids": [f"answer_{qid}_1"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I recently bought a zoom lens for my camera.", "has_answer": True},
                {"role": "assistant", "content": "Nice choice!", "has_answer": False},
            ]
        ],
        "answer_session_ids": [f"answer_{qid}_1"],
    }


def _write_source(tmp_path: Path) -> Path:
    data = [_source_item("41698283"), _source_item("0977f2af", "Instant Pot")]
    path = tmp_path / "knowledge_update_subset.json"
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


def _acceptance_fixture(tmp_path: Path, source: Path) -> Path:
    fixture = {
        "schema_version": 1,
        "panel_id": "test-event-history-acceptance",
        "noncanonical": True,
        "source_fixture": str(source),
        "source_sha256": _sha256_bytes(source.read_bytes()),
        "source_fixture_kind": "knowledge-update",
        "perceiver_version": "v1",
        "predicate": "acquired",
        "cases": [
            {
                "case_id": "41698283",
                "question_id": "41698283",
                "intent": "latest",
                "expected_status": "unique",
                "expected_object_key": "70-200mm zoom lens",
                "anchor_object_key": None,
                "domain": "camera lens",
                "safety_control": False,
            },
            {
                "case_id": "0977f2af",
                "question_id": "0977f2af",
                "intent": "predecessor",
                "expected_status": "unique",
                "expected_object_key": "instant pot",
                "anchor_object_key": "air fryer",
                "domain": "kitchen appliance",
                "safety_control": False,
            },
            {
                "case_id": "control-intent-only",
                "question_id": None,
                "intent": "latest",
                "expected_status": "none",
                "expected_object_key": None,
                "anchor_object_key": None,
                "domain": None,
                "safety_control": True,
                "episodes": [
                    {
                        "uuid": "ctl-intent-1",
                        "content": "I am planning to buy a new espresso machine next month.",
                        "reference_time": "2026-01-10T00:00:00+00:00",
                        "turn_evidence_uuid": "evidence-ctl-intent-1",
                    }
                ],
            },
        ],
    }
    path = tmp_path / "event_history_acceptance_v1.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    return path


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: object = None


@dataclass
class FakeProposal:
    episode_uuid: str
    object_key: str
    domain: str | None = None


class FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = type("Msg", (), {"content": content})
        self.finish_reason = finish_reason


class FakeCompletion:
    def __init__(self, content: str, usage=None, finish_reason: str = "stop"):
        self.choices = [FakeChoice(content, finish_reason)]
        self.usage = usage


class FakeChatCompletions:
    def __init__(self, responder):
        self.responder = responder

    def create(self, **kwargs):
        return self.responder(**kwargs)


class FakeOpenAI:
    def __init__(self, responder):
        self.chat = type("Chat", (), {"completions": FakeChatCompletions(responder)})()


def _canned_response(**kwargs):
    return FakeCompletion(json.dumps([{"episode": 0, "events": []}]))


def _fake_menhir_api() -> ProbeEventHistoryApi:
    """Functional fake Menhir seam so tests exercise real core logic with no network/service."""

    def extract_events_once(episodes, llm_complete, *, on_drop=None):
        rows = json.loads(llm_complete("", ""))
        proposals = []
        for row in rows:
            for event in row.get("events", []):
                proposals.append(
                    FakeProposal(
                        episode_uuid=episodes[row["episode"]].uuid,
                        domain=event.get("domain") or None,
                        object_key=event.get("object"),
                    )
                )
        return proposals

    def build_event_assertion(
        proposal, *, subject_uuid, namespace, learned_at, episode_reference_time,
        turn_evidence_uuid, perceiver_version,
    ):
        assertion = type(
            "A",
            (),
            {
                "subject_uuid": subject_uuid,
                "namespace": namespace,
                "predicate": "acquired",
                "domain": proposal.domain,
                "object_key": proposal.object_key,
                "stated_span": "acquired item",
                "valid_at": episode_reference_time,
                "time_basis": "asserted",
                "materializable": True,
            },
        )()
        return type("Build", (), {"built": True, "assertion": assertion, "reason": None})()

    class Lane:
        def __init__(self, subject_uuid, predicate, namespace=None, domain=None):
            self.subject_uuid = subject_uuid
            self.predicate = predicate
            self.namespace = namespace
            self.domain = domain

    class Intent:
        LATEST = "latest"
        PREDECESSOR = "predecessor"

    def select_event_assertion(
        events, *, lane, intent, as_of=None, anchor_time=None, anchor_assertion_key=None
    ):
        selected = events[-1] if events else None
        return type(
            "Sel",
            (),
            {
                "status": "unique" if selected is not None else "none",
                "gate": "ok",
                "reason": None,
                "selected": selected,
                "has_unique_selection": selected is not None,
            },
        )()

    class Service:
        def __init__(self, source, sink):
            self.source = source
            self.sink = sink

        def rebuild_lane(self, lane):
            entries = self.source.assertions_for_lane(lane)
            view = self.sink.record_event_timeline(subject_uuid=lane.subject_uuid, namespace=lane.namespace)
            self.sink.draw_event_timeline_entries(view["uuid"], entries)
            return {"complete": True, "view": view["uuid"], "drawn": len(entries)}

    return ProbeEventHistoryApi(
        extract_events_once=extract_events_once,
        build_event_assertion=build_event_assertion,
        select_event_assertion=select_event_assertion,
        event_lane_type=Lane,
        selection_intent_type=Intent,
        event_history_service_type=Service,
    )


def test_source_identity_matches_raw_and_canonical(tmp_path):
    source = _write_source(tmp_path)
    raw_sha = _sha256_bytes(source.read_bytes())
    identity = validate_source_identity(source, raw_sha)
    assert identity["match_mode"] == "raw"
    assert identity["canonical_crlf_sha256"] == _sha256_bytes(_eol_canonical_crlf(source.read_bytes()))
    # canonical_crlf also matches the raw hash of a CRLF-normalized byte string
    assert validate_source_identity(source, identity["canonical_crlf_sha256"])["match_mode"] == "canonical_crlf"


def test_source_identity_hash_mismatch_fails_loud(tmp_path):
    source = _write_source(tmp_path)
    with pytest.raises(Exception, match="hash mismatch"):
        validate_source_identity(source, "0" * 64)


def test_fixture_loading_and_materialization(tmp_path):
    source = _write_source(tmp_path)
    fixture_path = _acceptance_fixture(tmp_path, source)
    fixture = load_fixture(fixture_path)
    assert fixture["noncanonical"] is True
    source_data = json.loads(source.read_text(encoding="utf-8"))
    cases = materialize_cases(fixture, source_data)
    assert [case_id for case_id, _ in cases] == ["41698283", "0977f2af", "control-intent-only"]
    latest_case = cases[0][1]
    assert latest_case.intent == "latest"
    assert latest_case.expected_object_key == "70-200mm zoom lens"
    assert latest_case.lane_domain == "camera lens"
    assert latest_case.question == "question for 41698283"
    assert any(episode.content == "I recently bought a zoom lens for my camera." for episode in latest_case.episodes)
    predecessor_case = cases[1][1]
    assert predecessor_case.anchor_object_key == "air fryer"
    assert predecessor_case.intent == "predecessor"
    assert predecessor_case.question == "question for 0977f2af"
    control_case = cases[2][1]
    assert control_case.question is None
    assert latest_case.episodes[0].uuid == "41698283-0-0"
    assert latest_case.episodes[0].turn_evidence_uuid == "evidence-41698283-0-0"


def test_source_answer_normalization_rejects_mismatch(tmp_path):
    source = _write_source(tmp_path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw[0]["answer"] = "a completely different lens"
    source.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    fixture = load_fixture(_acceptance_fixture(tmp_path, source))
    source_data = json.loads(source.read_text(encoding="utf-8"))
    with pytest.raises(Exception, match="source answer mismatch"):
        materialize_cases(fixture, source_data)


def test_normalize_answer_handles_articles_and_case():
    assert _normalize_answer("a 70-200mm zoom lens") == "70-200mm zoom lens"
    assert _normalize_answer("Instant Pot") == "instant pot"
    assert _normalize_answer("The instant pot") == "instant pot"


def test_build_episodes_uses_user_turns_only():
    question = _source_item("abc")
    episodes = build_episodes_from_question(question, "abc")
    assert len(episodes) == 1
    assert episodes[0].uuid == "abc-0-0"
    assert episodes[0].content == "I recently bought a zoom lens for my camera."
    assert episodes[0].reference_time == "2023-03-11T22:01:00Z"


def test_normalize_reference_time_lme_exact():
    assert normalize_reference_time("2023/08/30 (Wed) 04:01") == "2023-08-30T04:01:00Z"
    assert normalize_reference_time("2023/03/11 (Sat) 22:01") == "2023-03-11T22:01:00Z"


def test_normalize_reference_time_iso():
    assert normalize_reference_time("2026-01-10T00:00:00+00:00") == "2026-01-10T00:00:00Z"
    assert normalize_reference_time("2026-01-10T00:00:00") == "2026-01-10T00:00:00Z"


def test_normalize_reference_time_invalid_fails_loud():
    with pytest.raises(Exception, match="reference_time"):
        normalize_reference_time("")
    with pytest.raises(Exception, match="reference_time"):
        normalize_reference_time("not a time")
    with pytest.raises(Exception, match="reference_time"):
        normalize_reference_time("2023/13/40 (Xxx) 99:99")


def test_materialized_episodes_hold_iso_time():
    question = _source_item("lme-time")
    episodes = build_episodes_from_question(question, "lme-time")
    assert all("T" in episode.reference_time and episode.reference_time.endswith("Z") for episode in episodes)


def test_run_acceptance_writes_report_and_aggregates_usage(tmp_path):
    source = _write_source(tmp_path)
    fixture = _acceptance_fixture(tmp_path, source)
    output = tmp_path / "report.json"

    class Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15
        prompt_tokens_details = type("D", (), {"cached_tokens": 3})()

    requests = []

    def responder(**kwargs):
        requests.append(kwargs)
        return FakeCompletion(json.dumps([{"episode": 0, "events": []}]), Usage())

    exit_code = run_acceptance(
        fixture_path=fixture,
        source_fixture=source,
        menhir_root=tmp_path,
        output=output,
        client=FakeOpenAI(responder),
        api=_fake_menhir_api(),
        generated_at="2026-08-07T00:00:00+00:00",
    )
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["canonical"] is False
    assert report["noncanonical"] is True
    assert report["promotion_status"] == "not_evaluable"
    assert report["production_authority_enabled"] is False
    assert report["persistence_used"] is False
    assert report["llm_used"] is True
    assert report["report_schema_version"] == 2
    # two qid cases carry real source questions, so routing was evaluated for them
    assert report["query_routing_measured"] is True
    assert report["routing_coverage"] == {"measured_cases": 2, "total_cases": 3}
    by_case = {c["case_id"]: c for c in report["cases"]}
    assert by_case["41698283"]["query_routing_measured"] is True
    assert by_case["0977f2af"]["query_routing_measured"] is True
    assert by_case["control-intent-only"]["query_routing_measured"] is False
    assert report["config"]["model"] == "gpt-6-luna"
    assert report["config"]["temperature"] is None  # a reasoning model takes none
    assert report["config"]["requested_temperature"] == 0.7
    assert report["config"]["max_tokens"] == 512
    assert report["config"]["llm_error_retries"] == 1
    num_calls = report["usage"]["calls"]
    assert num_calls == 3 * 3  # 3 cases x samples=3
    assert report["usage"]["total_tokens"] == num_calls * 15
    assert report["usage"]["input_tokens"] == num_calls * 10
    assert report["usage"]["output_tokens"] == num_calls * 5
    assert report["usage"]["cached_tokens"] == num_calls * 3
    assert all(call["raw_usage"]["prompt_tokens"] == 10 for call in report["usage"]["per_call"])
    assert all(call["finish_reason"] == "stop" for call in report["usage"]["per_call"])
    assert all(request["max_completion_tokens"] == 16384 for request in requests)
    assert all("max_tokens" not in request and "temperature" not in request for request in requests)
    assert report["aggregate"]["llm_error_retries"] == 0
    assert report["provenance"]["source_fixture"]["sha256"]["match_mode"] in {"raw", "canonical_crlf"}
    assert report["provenance"]["fixture"]["raw_sha256"]
    assert report["provenance"]["source_fixture"]["declared_source_fixture"] == str(source)
    assert len(report["cases"]) == 3
    assert report["aggregate"]["total_cases"] == 3
    # empty canned events produce no unique selection for the two unique-expected cases
    assert exit_code == 1


def test_run_acceptance_missing_usage_reported(tmp_path):
    source = _write_source(tmp_path)
    fixture = _acceptance_fixture(tmp_path, source)
    output = tmp_path / "report2.json"

    def responder(**kwargs):
        return FakeCompletion(json.dumps([{"episode": 0, "events": []}]), usage=None)

    run_acceptance(
        fixture_path=fixture,
        source_fixture=source,
        menhir_root=tmp_path,
        output=output,
        client=FakeOpenAI(responder),
        api=_fake_menhir_api(),
        generated_at="2026-08-07T00:00:00+00:00",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["usage"]["missing_usage_calls"] > 0
    assert all(call["usage_missing"] is True for call in report["usage"]["per_call"])
