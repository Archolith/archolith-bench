"""Anchored-time identity and retry parity for the LongMemEval build.

The failure each test guards: one graph holding two ingest configurations. A resume or a
build --retry-failed pass that enriches with anchored time (or a different model) on a graph built
without it leaves nothing in the scores to show it happened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from archolith_bench.build import pipeline

ROOT = Path(__file__).resolve().parents[1]
LME = ROOT / "scripts" / "longmemeval"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


provenance = _load("lme_run_provenance_retry_test", LME / "lib" / "run_provenance.py")

ANCHORED = {
    "extract_model": "gpt-4o-mini",
    "anchored_time_resolver_enabled": 1,
    "anchored_time_model": "gpt-6-luna",
    "anchored_time_attach_model": "gpt-6-luna",
}
# Recorded later than the original identity: a graph without them is refused.
RECORDED_LATER = dict(ANCHORED)
# No longer settings since the #168 freeze; old provenance files still carry them.
RETIRED_RECORD_KEYS = {
    "anchored_time_attach_enabled": 1,
    "anchored_time_expiry_enabled": 0,
    "assistant_turns_store_only_enabled": 1,
}


def _build(**overrides) -> dict:
    record = {
        "container": "menhir-lme-neo4j",
        "volume": "menhir-lme-data",
        "dataset": "xiaowu0162/longmemeval",
        "variant": "oracle",
        "namespace_prefix": "lme-",
        "fixture_sha256": "deadbeef",
        "fixture_count": 500,
        "scalar_state_enabled": 0,
        "counter_state_enabled": 0,
        "event_history_enabled": 0,
        "canonical_self_binding_mode": "off",
        "menhir_commit": "aaaa111",
        "bench_commit": "bbbb222",
        "surface_digest": "f" * 64,
        "started_at": "2026-10-15T00:00:00Z",
        **RECORDED_LATER,
    }
    record.update(overrides)
    return record


def _retry(**overrides) -> dict:
    keys = (*provenance.RETRY_BEHAVIOR_KEYS, "menhir_commit", "bench_commit", "surface_digest")
    record = {key: _build()[key] for key in keys}
    record.update(overrides)
    return record


def _write_build(path: Path, record: dict) -> None:
    provenance._write_atomic(path, provenance.begin(path, record))


# ---------------------------------------------------------------------------
# Resume identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [
    ("extract_model", "gpt-6-luna"),
    ("anchored_time_resolver_enabled", 0),
    ("anchored_time_model", "gpt-4o-mini"),
    ("anchored_time_attach_model", "gpt-4o-mini"),
])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_changed_anchored_setting(tmp_path: Path, field, value, noncanonical) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    with pytest.raises(provenance.ProvenanceMismatch, match=field):
        provenance.begin(path, _build(**{field: value}), noncanonical=noncanonical)


@pytest.mark.parametrize("field", sorted(RECORDED_LATER))
@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_setting_a_previously_unrecorded_behavior(tmp_path: Path, field, noncanonical) -> None:
    """A graph built before anchored time was recorded cannot be resumed with it set."""
    path = tmp_path / "graph-provenance.json"
    legacy = _build()
    for key in RECORDED_LATER:
        del legacy[key]
    _write_build(path, legacy)
    with pytest.raises(provenance.ProvenanceMismatch, match="was not recorded"):
        provenance.begin(path, _build(), noncanonical=noncanonical)


def test_resume_and_retry_of_a_graph_recording_retired_keys_ignore_them(tmp_path: Path) -> None:
    """A pre-freeze provenance file still names attach, expiry and the policy; they are not identity."""
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build(**RETIRED_RECORD_KEYS))
    document = provenance.begin(path, _build(started_at="2026-10-28T00:00:00Z"))
    assert not set(RETIRED_RECORD_KEYS) & set(document["identity"])
    provenance.assert_retry_matches(provenance._read(path), _retry())


def test_matching_anchored_resume_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    document = provenance.begin(path, _build(started_at="2026-10-16T00:00:00Z"))
    assert document["identity"]["anchored_time_model"] == "gpt-6-luna"
    assert document["attempt_count"] == 2


# ---------------------------------------------------------------------------
# Retry check
# ---------------------------------------------------------------------------

def test_retry_with_build_settings_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    provenance.assert_retry_matches(provenance._read(path), _retry())


@pytest.mark.parametrize("field, value", [
    ("container", "menhir-lme-other"),
    ("scalar_state_enabled", 1),
    ("counter_state_enabled", 1),
    ("event_history_enabled", 1),
    ("canonical_self_binding_mode", "enforce"),
    ("extract_model", "gpt-6-luna"),
    ("anchored_time_resolver_enabled", 0),
    ("anchored_time_model", "gpt-4o-mini"),
    ("anchored_time_attach_model", "gpt-4o-mini"),
])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_retry_refuses_different_ingest_behavior(tmp_path: Path, field, value, noncanonical) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    with pytest.raises(provenance.ProvenanceMismatch, match=field):
        provenance.assert_retry_matches(
            provenance._read(path), _retry(**{field: value}), noncanonical=noncanonical
        )


@pytest.mark.parametrize("field", sorted(RECORDED_LATER))
def test_retry_refuses_a_graph_that_never_recorded_the_behavior(tmp_path: Path, field) -> None:
    path = tmp_path / "graph-provenance.json"
    legacy = _build()
    del legacy[field]
    _write_build(path, legacy)
    with pytest.raises(provenance.ProvenanceMismatch, match=f"{field}: not recorded by the build"):
        provenance.assert_retry_matches(provenance._read(path), _retry(), noncanonical=True)


def test_retry_refuses_when_it_omits_a_behavior(tmp_path: Path) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    incomplete = _retry()
    del incomplete["anchored_time_attach_model"]
    with pytest.raises(provenance.ProvenanceMismatch, match="not set by this retry"):
        provenance.assert_retry_matches(provenance._read(path), incomplete, noncanonical=True)


@pytest.mark.parametrize("field, value", [
    ("menhir_commit", "cccc333"), ("bench_commit", "dddd444"), ("surface_digest", "0" * 64),
])
def test_retry_code_drift_refused_canonical_allowed_noncanonical(tmp_path: Path, field, value) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    document = provenance._read(path)
    with pytest.raises(provenance.ProvenanceMismatch, match="canonical"):
        provenance.assert_retry_matches(document, _retry(**{field: value}))
    provenance.assert_retry_matches(document, _retry(**{field: value}), noncanonical=True)


def test_cli_check_retry_is_read_only_when_canonical(tmp_path: Path) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    before = path.read_bytes()
    record = tmp_path / "retry.json"
    record.write_text(json.dumps(_retry()), encoding="utf-8")
    assert provenance.main(["check-retry", str(path), str(record)]) == 0
    assert path.read_bytes() == before


def test_cli_check_retry_noncanonical_labels_the_graph(tmp_path: Path) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    record = tmp_path / "retry.json"
    record.write_text(json.dumps(_retry(menhir_commit="cccc333")), encoding="utf-8")
    assert provenance.main(["check-retry", str(path), str(record), "--noncanonical"]) == 0
    assert provenance._read(path)["noncanonical"] is True


def test_cli_check_retry_refuses_without_provenance(tmp_path: Path) -> None:
    record = tmp_path / "retry.json"
    record.write_text(json.dumps(_retry()), encoding="utf-8")
    with pytest.raises(provenance.ProvenanceMismatch, match="no graph provenance"):
        provenance.main(["check-retry", str(tmp_path / "missing.json"), str(record)])


# ---------------------------------------------------------------------------
# Script parity (static)
# ---------------------------------------------------------------------------

# Retired by the #168 freeze; Menhir ignores them, so exporting one would record a setting
# that does not apply.
RETIRED_ENV = (
    "MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_ATTRIBUTE",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SCOPE",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SUBJECT",
    "MENHIR_PERSONAL_MEMORY_SUM_GROUNDING",
    "MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED",
    "MENHIR_ANCHORED_TIME_ATTACH",
    "MENHIR_ANCHORED_TIME_EXPIRY",
    "MENHIR_PERSONAL_MEMORY_SCALAR_CANONICAL_SELF",
)


def test_build_pipeline_exports_no_retired_flag() -> None:
    names = {name for name, _ in (*pipeline.MENHIR_ENV, *pipeline._EMBED_ENV,
                                  *pipeline._PROVIDER_ENV["openai"], *pipeline._PROVIDER_ENV["local"])}
    assert not set(RETIRED_ENV) & names


def test_retry_record_keys_are_what_check_retry_compares() -> None:
    assert set(pipeline.RETRY_RECORD_KEYS) == {
        *provenance.RETRY_BEHAVIOR_KEYS, *provenance.COMMIT_KEYS, "surface_digest"}


def test_retry_record_renders_each_key_as_the_build_attempt_record_does() -> None:
    """check-retry compares JSON values, so retry and build must render each key identically."""
    s = {
        "LME_NEO4J_NAME": "g", "LME_NEO4J_VOL": "v", "VOLUME_PRE_EXISTED": "true", "GRAPH_FRESH": "false",
        "LME_REQUIRE_FRESH": "0", "LME_RESUME": "0", "LME_DATASET": "d", "LONGMEMEVAL_VARIANT": "oracle",
        "LIMIT": "1", "INGEST_TARGET": "1", "LME_NS_PREFIX": "lme-", "LME_SEGMENTATION": "adaptive",
        "LME_INGEST_CONCURRENCY": "8", "LME_COUNTER_STATE_ENABLED": "0", "LME_EVENT_HISTORY_ENABLED": "0",
        "LME_SCALAR_STATE_ENABLED": "1", "LME_SCALAR_HISTORY_ENABLED": "0", "LME_SCALAR_CONSOLIDATION_K": "3",
        "LME_SCALAR_THRESHOLD": "2/3", "LME_CANONICAL_SELF_BINDING_MODE": "off", "LME_EXTRACT_MODEL": "m",
        "LME_ANCHORED_TIME_RESOLVER": "1", "LME_ANCHORED_TIME_MODEL": "a", "LME_ANCHORED_TIME_ATTACH_MODEL": "a",
        "LME_OPENAI_SERVICE_TIER": "flex", "LME_MAX_LLM_CALLS_PER_JOB": "100", "LME_REQUIRE_SCALAR_OUTPUT": "1",
        "LME_REQUIRE_TURN_EVIDENCE": "1", "LME_CONSOLIDATION_AUDIT_ENABLED": "1", "LME_RECALL_AUDIT_ENABLED": "1",
        "LME_MENHIR_LOG_LEVEL": "INFO", "MENHIR_COMMIT": "c1", "BENCH_COMMIT": "c2",
        "GRAPH_SURFACE_DIGEST": "dg", "STARTED_AT": "t", "WRITER_MODEL_RECORD": "",
        "WRITER_GRANULARITY_RECORD": "", "WRITER_PROMPT_SHA": "",
    }
    build = pipeline.attempt_record(s)
    retry = pipeline.retry_record(s)
    assert set(retry) == set(pipeline.RETRY_RECORD_KEYS)
    assert retry == {k: build[k] for k in retry}


# ---------------------------------------------------------------------------
# Defaults and early validation (bash)
# ---------------------------------------------------------------------------

def _bash() -> str:
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    return bash


_CONFIG_KEYS = (
    "LME_EXTRACT_PROVIDER=", "LME_OPENAI_SERVICE_TIER=", "LME_ANCHORED_TIME_", "LME_MAX_LLM_CALLS_PER_JOB=",
)


def _config_lines() -> str:
    config = (LME / "config.sh").read_text(encoding="utf-8")
    return "\n".join(line for line in config.splitlines() if line.startswith(_CONFIG_KEYS))


def _clean_env(**values: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LME_", "MENHIR_"))}
    env.update(values)
    return env


def test_config_defaults() -> None:
    command = "set -eu\n" + _config_lines() + '\nprintf "%s\\n" "$LME_ANCHORED_TIME_RESOLVER" ' \
        '"$LME_ANCHORED_TIME_MODEL" ' \
        '"$LME_ANCHORED_TIME_ATTACH_MODEL" "$LME_OPENAI_SERVICE_TIER" "$LME_MAX_LLM_CALLS_PER_JOB"'
    result = subprocess.run([_bash(), "-c", command], env=_clean_env(), capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["1", "gpt-6-luna", "gpt-6-luna", "flex", "100"]


@pytest.mark.parametrize("overrides, expected", [
    ({}, "flex"),
    ({"LME_OPENAI_SERVICE_TIER": "default"}, "default"),
    ({"MENHIR_OPENAI_SERVICE_TIER": "priority"}, "priority"),
])
def test_config_forwards_the_service_tier_to_every_menhir_serve(overrides, expected) -> None:
    """Recall/analysis wrappers start Menhir after sourcing config.sh only, never build_graph.sh."""
    keys = ("LME_OPENAI_SERVICE_TIER=", "export MENHIR_OPENAI_SERVICE_TIER=")
    config = (LME / "config.sh").read_text(encoding="utf-8")
    lines = "\n".join(line for line in config.splitlines() if line.startswith(keys))
    assert "MENHIR_OPENAI_SERVICE_TIER" in lines
    command = "set -eu\n" + lines + '\nbash -c \'printf "%s" "$MENHIR_OPENAI_SERVICE_TIER"\''
    result = subprocess.run([_bash(), "-c", command], env=_clean_env(**overrides), capture_output=True, text=True, check=True)
    assert result.stdout == expected


def test_attach_model_follows_resolver_model_override() -> None:
    command = "set -eu\n" + _config_lines() + '\nprintf "%s\\n" "$LME_ANCHORED_TIME_ATTACH_MODEL"'
    env = _clean_env(LME_ANCHORED_TIME_MODEL="gpt-6-sol")
    result = subprocess.run([_bash(), "-c", command], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "gpt-6-sol"


@pytest.mark.parametrize("overrides, message", [
    ({"LME_ANCHORED_TIME_RESOLVER": "yes"}, "must be 0 or 1"),
    ({"LME_EXTRACT_PROVIDER": "local"}, "needs LME_EXTRACT_PROVIDER=openai"),
    ({}, None),
    ({"LME_ANCHORED_TIME_RESOLVER": "0"}, None),
    ({"LME_ANCHORED_TIME_RESOLVER": "0", "LME_EXTRACT_PROVIDER": "local"}, None),
])
def test_build_validates_anchored_settings_before_any_work(overrides, message) -> None:
    names = ("LME_EXTRACT_PROVIDER", "LME_ANCHORED_TIME_RESOLVER")
    command = "set -eu\n" + _config_lines() + "\nprintf '%s\\n' " + " ".join(f'"${name}"' for name in names)
    result = subprocess.run([_bash(), "-c", command], env=_clean_env(**overrides),
                            capture_output=True, text=True, check=True)
    settings = {
        "LME_CANONICAL_SELF_BINDING_MODE": "off", "LME_COUNTER_STATE_ENABLED": "0",
        "LME_SCALAR_STATE_ENABLED": "0", "LME_EVENT_HISTORY_ENABLED": "0",
        "LME_INGEST_CONCURRENCY": "30", "LME_RESUME": "0", "LME_INGEST_STOP_AFTER_ITEMS": "0",
        **dict(zip(names, result.stdout.split(), strict=True)),
    }
    if message is None:
        pipeline.validate(settings, "5")
        assert settings["LIMIT"] == "5"
    else:
        with pytest.raises(pipeline.Die) as exc:
            pipeline.validate(settings, "5")
        assert message in exc.value.message
