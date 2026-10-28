"""Offline contract tests for the Menhir deterministic scalar shadow instrument.

These tests use the real Menhir source checkout when ``MENHIR_ROOT`` is set (or when a local
``../menhir`` sibling exists).  They create captures in pytest temporary directories and never
call a network, LLM, Neo4j, Docker, or Menhir service.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from archolith_bench.deterministic_scalar_shadow import (
    CaptureError,
    DEFAULT_THRESHOLD,
    analyze_captures,
    load_menhir_api,
    render_markdown,
    _validate_menhir_import_identity,
    write_reports,
)


def _menhir_root() -> Path:
    candidates = []
    if os.environ.get("MENHIR_ROOT"):
        candidates.append(Path(os.environ["MENHIR_ROOT"]))
    candidates.extend((Path.cwd() / "menhir", Path.cwd().parent / "menhir"))
    for candidate in candidates:
        # The deterministic extractor was removed from Menhir main on 2026-10-27.
        if (candidate / "src" / "menhir" / "services" / "deterministic_scalar_extractor.py").is_file():
            return candidate.resolve()
    pytest.skip("set MENHIR_ROOT to a Menhir checkout of archive/bucket1-features-2026-10-27")


@pytest.fixture(scope="module")
def api():
    return load_menhir_api(_menhir_root())


def _proposal_dict(api, episode, *, start=None, end=None, value=None):
    episode_obj = type("Episode", (), {"uuid": episode["uuid"], "content": episode["content"]})()
    extracted = api.extractor_type().extract([episode_obj])
    assert extracted.proposals, f"synthetic episode did not match a deterministic template: {episode!r}"
    proposal = extracted.proposals[0]
    if start is not None or end is not None or value is not None:
        start = proposal.span_start if start is None else start
        end = proposal.span_end if end is None else end
        proposal = dataclasses.replace(
            proposal,
            stated_span=episode["content"][start:end],
            span_start=start,
            span_end=end,
            value=proposal.value if value is None else value,
        )
    return dataclasses.asdict(proposal)


def _capture(tmp_path, api, namespaces, *, k=3, truncated=0, llm_calls=None, name="capture.json"):
    if llm_calls is None:
        llm_calls = len(namespaces) * k
    payload = {
        "settings": {
            "model": "synthetic",
            "k": k,
            "temp": 0.7,
            "max_tokens": 2048,
            "truncated_completions": truncated,
            "llm_calls": llm_calls,
        },
        "namespaces": namespaces,
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _capture_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _namespace(api, uuid, content, samples=None):
    episode = {"uuid": uuid, "content": content}
    proposal = _proposal_dict(api, episode)
    return {"episodes": [episode], "samples": samples or [[proposal], [proposal], [proposal]]}


def test_happy_path_uses_real_menhir_gate_and_extractor(tmp_path, api):
    path = _capture(tmp_path, api, {"z-ns": _namespace(api, "ep-z", "I have 3 coins.")})

    report = analyze_captures([path], menhir_root=_menhir_root(), generated_at="2026-08-05T00:00:00Z")

    namespace = report["namespaces"][0]
    assert namespace["namespace"] == "z-ns"
    assert namespace["episodes"] == {"total": 1, "fully_eligible": 1, "fallback_required": 0}
    assert namespace["deterministic_proposals"] == {"all": 1, "router_eligible": 1}
    assert namespace["committed_llm_claims"] == 1
    assert namespace["agreement"]["exact_one_to_one"]["numerator"] == 1
    assert namespace["agreement"]["aligned_one_to_one"]["numerator"] == 1
    assert namespace["compositional"]["deterministic_composed"] == 1
    assert namespace["compositional"]["llm_composed"] == 1
    assert namespace["comparison_detail"]["canonical_schema_version"] == 2
    assert namespace["comparison_detail"]["compositional_schema_version"] == 1
    assert namespace["comparison_detail"]["composer_version"] == "structural-v4"
    assert namespace["compositional"]["diagnostic_vs_llm"][
        "compositional_exact_agreements"] == 1
    assert report["aggregate"]["compositional"]["deterministic_composed"] == 1
    assert report["aggregate"]["compositional"]["promotion_status"] == "not_evaluable"
    assert namespace["call_savings"]["conservative_future_calls_saved"] == 3
    assert report["measurements"]["token_savings"] is None


def test_aligned_not_exact_is_reported_separately(tmp_path, api):
    episode = {"uuid": "ep-aligned", "content": "I have 3 coins."}
    full = _proposal_dict(api, episode)
    narrowed = _proposal_dict(api, episode, start=2)
    path = _capture(
        tmp_path,
        api,
        {"aligned": {"episodes": [episode], "samples": [[full], [narrowed], [full]]}},
    )

    report = analyze_captures([path], menhir_root=_menhir_root())
    agreement = report["namespaces"][0]["agreement"]
    assert agreement["exact_one_to_one"]["numerator"] == 0
    assert agreement["aligned_one_to_one"]["numerator"] == 1


def test_one_to_one_matching_does_not_count_one_det_claim_twice(tmp_path, api):
    episode = {"uuid": "ep-duplicate", "content": "I have 3 coins."}
    full = _proposal_dict(api, episode)
    narrow = _proposal_dict(api, episode, start=2)
    samples = [[full, narrow], [full, narrow], [full, narrow]]
    path = _capture(tmp_path, api, {"duplicates": {"episodes": [episode], "samples": samples}})

    report = analyze_captures([path], menhir_root=_menhir_root(), threshold=1 / 3)
    namespace = report["namespaces"][0]
    assert namespace["committed_llm_claims"] == 6
    assert namespace["agreement"]["aligned_one_to_one"]["numerator"] == 1
    assert namespace["router_missed_committed_llm_claims"] == 5
    assert sum(
        row["aligned_agreements"] for row in namespace["per_class_agreement"].values()
    ) == 1


def test_router_miss_and_class_attribution_are_explicit(tmp_path, api):
    episode = {"uuid": "ep-miss", "content": "I have 3 coins."}
    det = _proposal_dict(api, episode)
    llm = dict(det, value=4)
    path = _capture(tmp_path, api, {"miss": {"episodes": [episode], "samples": [[llm], [llm], [llm]]}})

    report = analyze_captures([path], menhir_root=_menhir_root())
    namespace = report["namespaces"][0]
    assert namespace["router_missed_committed_llm_claims"] == 1
    assert namespace["per_class_agreement"]["c_count"]["router_missed_committed_llm_claims"] == 1


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda payload: payload["settings"].update({"truncated_completions": 1}), "truncated_completions"),
        (lambda payload: payload["settings"].update({"llm_calls": 2}), "settings.llm_calls"),
        (lambda payload: payload["namespaces"]["ns"]["samples"].pop(), "sample-count mismatch"),
        (
            lambda payload: payload["namespaces"]["ns"]["episodes"].append(
                payload["namespaces"]["ns"]["episodes"][0]
            ),
            "duplicate episode uuid",
        ),
    ],
)
def test_contaminated_or_malformed_captures_fail_closed(tmp_path, api, mutator, message):
    payload = {
        "settings": {
            "model": "synthetic",
            "k": 3,
            "temp": 0.7,
            "max_tokens": 2048,
            "truncated_completions": 0,
            "llm_calls": 3,
        },
        "namespaces": {"ns": _namespace(api, "ep-bad", "I have 3 coins.")},
    }
    mutator(payload)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CaptureError, match=message):
        analyze_captures([path], menhir_root=_menhir_root())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["namespaces"]["ns"]["samples"][0][0].update(
                {"value": [1, 2, 3]}),
            "value fails Menhir kind/operation validation",
        ),
        (
            lambda payload: payload["namespaces"]["ns"]["samples"][0][0].update(
                {"when": "not-a-time"}),
            "canonical normalized ISO timestamp",
        ),
        (
            lambda payload: payload["settings"].update({"temp": -0.1}),
            "settings.temp must be a non-negative finite number",
        ),
    ],
)
def test_capture_uses_menhir_value_time_and_sampling_validation(
    tmp_path, api, mutate, message,
):
    namespace = _namespace(api, "ep-contract", "I have 3 coins.")
    payload = {
        "settings": {
            "model": "synthetic",
            "k": 3,
            "temp": 0.7,
            "max_tokens": 2048,
            "truncated_completions": 0,
            "llm_calls": 3,
        },
        "namespaces": {"ns": namespace},
    }
    mutate(payload)
    path = tmp_path / "invalid-contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CaptureError, match=message):
        analyze_captures([path], menhir_root=_menhir_root())


def test_capture_requires_unique_menhir_grounding(tmp_path, api):
    original = {"uuid": "ep-duplicate-span", "content": "I have 3 coins."}
    proposal = _proposal_dict(api, original)
    duplicate_episode = {
        "uuid": original["uuid"],
        "content": "I have 3 coins. I have 3 coins.",
    }
    namespace = {
        "episodes": [duplicate_episode],
        "samples": [[proposal], [proposal], [proposal]],
    }
    path = _capture(tmp_path, api, {"ns": namespace})

    with pytest.raises(CaptureError, match="unique case-insensitive grounding"):
        analyze_captures([path], menhir_root=_menhir_root())


def test_invalid_span_and_unknown_proposal_episode_fail_closed(tmp_path, api):
    episode = {"uuid": "ep-invalid", "content": "I have 3 coins."}
    proposal = _proposal_dict(api, episode)
    proposal["span_end"] += 1
    bad_span = _capture(
        tmp_path,
        api,
        {"ns": {"episodes": [episode], "samples": [[proposal], [proposal], [proposal]]}},
        name="span.json",
    )
    with pytest.raises(CaptureError, match="stated_span does not equal"):
        analyze_captures([bad_span], menhir_root=_menhir_root())

    unknown = dict(_proposal_dict(api, episode), episode_uuid="not-captured")
    unknown_path = _capture(
        tmp_path,
        api,
        {"unknown": {"episodes": [episode], "samples": [[unknown], [unknown], [unknown]]}},
        name="unknown.json",
    )
    with pytest.raises(CaptureError, match="not present"):
        analyze_captures([unknown_path], menhir_root=_menhir_root())


def test_namespace_batch_savings_are_conservative_with_one_fallback(tmp_path, api):
    eligible = {"uuid": "ep-good", "content": "I have 3 coins."}
    fallback = {"uuid": "ep-fallback", "content": "I paid $250."}
    good_proposal = _proposal_dict(api, eligible)
    path = _capture(
        tmp_path,
        api,
        {
            "all-good": {
                "episodes": [eligible],
                "samples": [[good_proposal], [good_proposal], [good_proposal]],
            },
            "partial": {
                "episodes": [eligible, fallback],
                "samples": [[good_proposal], [good_proposal], [good_proposal]],
            },
        },
        llm_calls=6,
    )
    report = analyze_captures([path], menhir_root=_menhir_root())
    by_name = {row["namespace"]: row for row in report["namespaces"]}
    assert by_name["all-good"]["call_savings"]["conservative_future_calls_saved"] == 3
    assert by_name["partial"]["call_savings"]["conservative_future_calls_saved"] == 0
    assert report["aggregate"]["call_savings"]["conservative_future_calls_saved"] == 3


def test_labels_absent_are_null_not_zero(tmp_path, api):
    path = _capture(tmp_path, api, {"unlabeled": _namespace(api, "ep-unlabeled", "I have 3 coins.")})
    report = analyze_captures([path], menhir_root=_menhir_root())
    labels = report["aggregate"]["labels"]
    assert report["labels"]["status"] == "absent"
    assert labels["false_positive"]["status"] == "not_measured"
    assert labels["false_positive"]["semantics"] == "known_negative_target_hit_rate"
    assert labels["false_positive"]["hit_count"] is None
    assert labels["false_current"]["hit_rate"] is None


def test_labeled_false_positive_and_false_current_are_measured(tmp_path, api):
    namespace = _namespace(api, "ep-labeled", "I have 3 coins.")
    proposal = namespace["samples"][0][0]
    sidecar = {
        "schema_version": 1,
        "labels": [
            {
                "namespace": "labeled",
                "episode_uuid": "ep-labeled",
                "span_start": proposal["span_start"],
                "span_end": proposal["span_end"],
                "label": "false_positive",
            },
            {
                "namespace": "labeled",
                "episode_uuid": "ep-labeled",
                "span_start": 0,
                "span_end": 1,
                "label": "false_current",
            },
        ],
    }
    capture = _capture(tmp_path, api, {"labeled": namespace})
    sidecar["capture_sha256"] = [_capture_sha256(capture)]
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(sidecar), encoding="utf-8")
    report = analyze_captures([capture], menhir_root=_menhir_root(), labels_path=labels)
    measured = report["aggregate"]["labels"]
    assert measured["false_positive"] == {
        "status": "measured",
        "semantics": "known_negative_target_hit_rate",
        "hit_count": 1,
        "labeled_negative_targets": 1,
        "hit_rate": 1.0,
    }
    assert measured["false_current"]["hit_count"] == 0
    assert measured["false_current"]["labeled_negative_targets"] == 1
    assert measured["false_current"]["hit_rate"] == 0.0
    assert report["labels"]["sha256"] == _capture_sha256(labels)
    assert report["labels"]["capture_sha256"] == [_capture_sha256(capture)]
    assert report["provenance"]["label_sidecar"]["sha256"] == _capture_sha256(labels)


def test_sidecar_category_without_labeled_targets_is_not_measured(tmp_path, api):
    namespace = _namespace(api, "ep-one-label", "I have 3 coins.")
    capture = _capture(tmp_path, api, {"one-label": namespace})
    proposal = namespace["samples"][0][0]
    sidecar = {
        "schema_version": 1,
        "capture_sha256": [_capture_sha256(capture)],
        "labels": [
            {
                "namespace": "one-label",
                "episode_uuid": "ep-one-label",
                "span_start": proposal["span_start"],
                "span_end": proposal["span_end"],
                "label": "false_positive",
            }
        ],
    }
    labels = tmp_path / "one-label.json"
    labels.write_text(json.dumps(sidecar), encoding="utf-8")

    report = analyze_captures([capture], menhir_root=_menhir_root(), labels_path=labels)
    false_current = report["aggregate"]["labels"]["false_current"]
    assert false_current == {
        "status": "not_measured",
        "semantics": "known_negative_target_hit_rate",
        "hit_count": None,
        "labeled_negative_targets": None,
        "hit_rate": None,
    }


@pytest.mark.parametrize("kind", ["missing", "empty", "duplicate", "malformed", "extra", "mismatch"])
def test_sidecar_capture_hash_binding_fails_closed(tmp_path, api, kind):
    capture = _capture(tmp_path, api, {"hashes": _namespace(api, "ep-hashes", "I have 3 coins.")})
    capture_hash = _capture_sha256(capture)
    sidecar = {"schema_version": 1, "capture_sha256": [capture_hash], "labels": []}
    expected = "capture_sha256"
    if kind == "missing":
        sidecar.pop("capture_sha256")
        expected = "required"
    elif kind == "empty":
        sidecar["capture_sha256"] = []
        expected = "non-empty"
    elif kind == "duplicate":
        sidecar["capture_sha256"] = [capture_hash, capture_hash]
        expected = "duplicate"
    elif kind == "malformed":
        sidecar["capture_sha256"] = ["not-a-sha256"]
        expected = "canonical lowercase SHA-256"
    elif kind == "extra":
        sidecar["capture_sha256"] = [capture_hash, "a" * 64]
        expected = "exactly match"
    elif kind == "mismatch":
        sidecar["capture_sha256"] = ["b" * 64]
        expected = "exactly match"
    labels = tmp_path / f"{kind}.json"
    labels.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(CaptureError, match=expected):
        analyze_captures([capture], menhir_root=_menhir_root(), labels_path=labels)


def test_multiple_captures_require_shared_sampling_policy(tmp_path, api):
    first = _capture(tmp_path, api, {"first": _namespace(api, "ep-first", "I have 3 coins.")}, name="first.json")
    second = _capture(tmp_path, api, {"second": _namespace(api, "ep-second", "I have 4 coins.")}, name="second.json")
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["settings"]["temp"] = 0.8
    second.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CaptureError, match="capture sampling policy mismatch.*settings.temp"):
        analyze_captures([first, second], menhir_root=_menhir_root())


def test_menhir_import_identity_rejects_module_outside_selected_root(tmp_path):
    outside = tmp_path / "other-menhir" / "src" / "menhir" / "services" / "typed_scalar_rules.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    module = SimpleNamespace(__name__="menhir.services.typed_scalar_rules", __file__=str(outside))

    with pytest.raises(CaptureError, match="Menhir import identity mismatch.*selected root.*imported from"):
        _validate_menhir_import_identity(_menhir_root(), (module,))


def test_output_ordering_and_json_markdown_rendering(tmp_path, api):
    a = _namespace(api, "ep-a", "I have 1 coins.")
    z = _namespace(api, "ep-z", "I have 2 coins.")
    path = _capture(tmp_path, api, {"z": z, "a": a})
    report = analyze_captures([path], menhir_root=_menhir_root(), generated_at="2026-08-05T00:00:00Z")
    assert [row["namespace"] for row in report["namespaces"]] == ["a", "z"]
    markdown = render_markdown(report)
    assert "Exact one-to-one agreement" in markdown
    assert "namespace-batch savings" in markdown
    assert "Known-negative target hit metrics" in markdown
    json_path = tmp_path / "out" / "report.json"
    markdown_path = tmp_path / "out" / "report.md"
    write_reports(report, json_path, markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["report_schema_version"] == 2
    assert markdown_path.read_text(encoding="utf-8").startswith("# Deterministic scalar shadow measurement")
    assert "Compositional aligned / compared pairs" in markdown_path.read_text(encoding="utf-8")


def test_default_gate_settings_are_candidate_policy():
    assert DEFAULT_THRESHOLD == 2 / 3
