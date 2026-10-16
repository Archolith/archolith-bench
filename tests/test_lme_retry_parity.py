"""Anchored-time identity and retry parity for the LongMemEval build.

The failure each test guards: one graph holding two ingest configurations. A resume or a
retry.sh pass that enriches with anchored time (or a different model) on a graph built
without it leaves nothing in the scores to show it happened.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    "anchored_time_attach_enabled": 1,
    "anchored_time_expiry_enabled": 0,
    "anchored_time_model": "gpt-6-luna",
    "anchored_time_attach_model": "gpt-6-luna",
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
        **ANCHORED,
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
    ("anchored_time_attach_enabled", 0),
    ("anchored_time_expiry_enabled", 1),
    ("anchored_time_model", "gpt-4o-mini"),
    ("anchored_time_attach_model", "gpt-4o-mini"),
])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_changed_anchored_setting(tmp_path: Path, field, value, noncanonical) -> None:
    path = tmp_path / "graph-provenance.json"
    _write_build(path, _build())
    with pytest.raises(provenance.ProvenanceMismatch, match=field):
        provenance.begin(path, _build(**{field: value}), noncanonical=noncanonical)


@pytest.mark.parametrize("field", sorted(ANCHORED))
@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_setting_a_previously_unrecorded_behavior(tmp_path: Path, field, noncanonical) -> None:
    """A graph built before anchored time was recorded cannot be resumed with it set."""
    path = tmp_path / "graph-provenance.json"
    legacy = _build()
    for key in ANCHORED:
        del legacy[key]
    _write_build(path, legacy)
    with pytest.raises(provenance.ProvenanceMismatch, match="was not recorded"):
        provenance.begin(path, _build(), noncanonical=noncanonical)


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
    ("anchored_time_attach_enabled", 0),
    ("anchored_time_expiry_enabled", 1),
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


@pytest.mark.parametrize("field", sorted(ANCHORED))
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
    del incomplete["anchored_time_attach_enabled"]
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

# Every Menhir setting that changes what enrichment writes. retry.sh re-enriches episodes in a
# graph build_graph.sh made, so each must be exported with the same value expression.
INGEST_ENV = (
    "MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED",
    "MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED",
    "MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED",
    "MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED",
    "MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED",
    "MENHIR_PERSONAL_MEMORY_SCALAR_THRESHOLD",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_ATTRIBUTE",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SCOPE",
    "MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SUBJECT",
    "MENHIR_PERSONAL_MEMORY_SCALAR_CANONICAL_SELF",
    "MENHIR_CANONICAL_SELF_BINDING_MODE",
    "MENHIR_PERSONAL_MEMORY_CHAT_MODEL",
    "MENHIR_PERSONAL_MEMORY_SUM_GROUNDING",
    "LME_REQUIRE_TURN_EVIDENCE",
    "MENHIR_MAX_LLM_CALLS_PER_JOB",
    "MENHIR_ANCHORED_TIME_RESOLVER",
    "MENHIR_ANCHORED_TIME_ATTACH",
    "MENHIR_ANCHORED_TIME_EXPIRY",
    "MENHIR_ANCHORED_TIME_MODEL",
    "MENHIR_ANCHORED_TIME_ATTACH_MODEL",
    "MENHIR_OPENAI_SERVICE_TIER",
    "MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED",
    "MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED",
    "MENHIR_LOG_LEVEL",
    "MENHIR_MCP_TELEMETRY_DB",
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBED_MODEL",
)

_ASSIGN = re.compile(r'([A-Z_][A-Z0-9_]*)(?:=("(?:[^"\\]|\\.)*"|\S+))?')


def _exports(script: str) -> dict[str, str | None]:
    """NAME -> value expression for every `export` line (None for a bare `export NAME`)."""
    found: dict[str, str | None] = {}
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped.startswith("export "):
            continue
        body = stripped[len("export "):].split(";")[0]
        for name, value in _ASSIGN.findall(body):
            found[name] = value or None
    return found


def test_retry_exports_the_same_ingest_settings_as_the_build() -> None:
    build = _exports((LME / "build_graph.sh").read_text(encoding="utf-8"))
    retry = _exports((LME / "retry.sh").read_text(encoding="utf-8"))
    for name in INGEST_ENV:
        assert name in build, f"build_graph.sh does not export {name}"
        assert name in retry, f"retry.sh does not export {name}"
        assert retry[name] == build[name], f"{name}: build={build[name]!r} retry={retry[name]!r}"


def test_retry_record_lines_match_the_build_attempt_record() -> None:
    """The check compares JSON values, so each key must be rendered the same way (quoted or not)."""
    build = (LME / "build_graph.sh").read_text(encoding="utf-8")
    retry = (LME / "retry.sh").read_text(encoding="utf-8")
    for key in provenance.RETRY_BEHAVIOR_KEYS:
        pattern = re.compile(rf'^\s*"{key}": .*$', re.MULTILINE)
        build_lines = {m.group(0).strip().rstrip(",") for m in pattern.finditer(build)}
        retry_lines = {m.group(0).strip().rstrip(",") for m in pattern.finditer(retry)}
        assert len(build_lines) == 1, f"{key}: {build_lines}"
        assert retry_lines == build_lines, f"{key}: build={build_lines} retry={retry_lines}"


def test_retry_checks_provenance_before_starting_menhir() -> None:
    retry = (LME / "retry.sh").read_text(encoding="utf-8")
    assert retry.index('"check-retry"') < retry.index("serve --port")
    assert 'phase-start "${GRAPH_PROVENANCE_PATH}"' in retry
    assert "--phase retry-failed" in retry


def test_buildout_ab_has_no_hardcoded_models() -> None:
    script = (LME / "buildout_ab.sh").read_text(encoding="utf-8")
    assert not re.search(r"gpt-[0-9]", script)
    assert "MENHIR_MAX_LLM_CALLS_PER_JOB=20" not in script


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
        '"$LME_ANCHORED_TIME_ATTACH" "$LME_ANCHORED_TIME_EXPIRY" "$LME_ANCHORED_TIME_MODEL" ' \
        '"$LME_ANCHORED_TIME_ATTACH_MODEL" "$LME_OPENAI_SERVICE_TIER" "$LME_MAX_LLM_CALLS_PER_JOB"'
    result = subprocess.run([_bash(), "-c", command], env=_clean_env(), capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["1", "1", "0", "gpt-6-luna", "gpt-6-luna", "flex", "100"]


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
    ({"LME_ANCHORED_TIME_EXPIRY": "2"}, "must be 0 or 1"),
    ({"LME_ANCHORED_TIME_RESOLVER": "0"}, "requires LME_ANCHORED_TIME_RESOLVER=1"),
    ({"LME_EXTRACT_PROVIDER": "local"}, "needs LME_EXTRACT_PROVIDER=openai"),
    ({}, None),
    ({"LME_ANCHORED_TIME_RESOLVER": "0", "LME_ANCHORED_TIME_ATTACH": "0", "LME_EXTRACT_PROVIDER": "local"}, None),
])
def test_build_validates_anchored_settings_before_any_work(overrides, message) -> None:
    script = (LME / "build_graph.sh").read_text(encoding="utf-8")
    start = script.index('for flag in "${LME_ANCHORED_TIME_RESOLVER}"')
    end = script.index('case "${LME_INGEST_CONCURRENCY}" in', start)
    die = "die(){ printf '%s\\n' \"$*\" >&2; exit 1; }\n"
    command = "set -eu\n" + _config_lines() + "\n" + die + script[start:end] + "\necho ok"
    result = subprocess.run([_bash(), "-c", command], env=_clean_env(**overrides), capture_output=True, text=True)
    if message is None:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
    else:
        assert result.returncode != 0
        assert message in result.stderr
