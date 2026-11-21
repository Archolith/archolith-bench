"""Build config schema, YAML loading, resolved-config record and build id.

A build YAML is the user-facing way to configure a LongMemEval graph build. It is resolved into
the existing LME_* variables; settings.py fills in the rest as config.sh would and hands them to
pipeline.py. Secrets never live in the YAML: ``extraction.api_key_env`` names an environment
variable that is read at run time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SECRET_VARS = frozenset({"LME_EXTRACT_API_KEY"})
API_KEY_VAR = "LME_EXTRACT_API_KEY"


@dataclass(frozen=True)
class Field:
    section: str
    key: str
    var: str | None  # None: not an environment variable (build_graph.sh positional arg)
    kind: str  # "str" | "int" | "float" | "bool"
    nullable: bool = False

    @property
    def path(self) -> str:
        return f"{self.section}.{self.key}"


def _f(section: str, key: str, var: str | None, kind: str, nullable: bool = False) -> Field:
    return Field(section, key, var, kind, nullable)


SCHEMA: tuple[Field, ...] = (
    _f("dataset", "dataset", "LME_DATASET", "str"),
    _f("dataset", "fixture_path", "LME_FIXTURE_PATH", "str", True),
    _f("dataset", "fixture_count", "LME_FIXTURE_COUNT", "int", True),
    _f("dataset", "fixture_sha256", "LME_FIXTURE_SHA256", "str", True),
    _f("dataset", "ns_prefix", "LME_NS_PREFIX", "str"),
    _f("dataset", "limit", None, "int"),
    _f("container", "neo4j_name", "LME_NEO4J_NAME", "str"),
    _f("container", "neo4j_vol", "LME_NEO4J_VOL", "str"),
    _f("container", "neo4j_image", "LME_NEO4J_IMAGE", "str"),
    _f("container", "neo4j_heap", "LME_NEO4J_HEAP", "str"),
    _f("container", "bolt", "LME_BOLT", "int"),
    _f("container", "http", "LME_HTTP", "int"),
    _f("container", "menhir_port", "LME_PORT_BUILD", "int"),
    _f("extraction", "provider", "LME_EXTRACT_PROVIDER", "str"),
    _f("extraction", "model", "LME_EXTRACT_MODEL", "str"),
    _f("extraction", "base_url", "LME_EXTRACT_BASE_URL", "str", True),
    _f("extraction", "service_tier", "LME_OPENAI_SERVICE_TIER", "str"),
    _f("extraction", "embed_model", "LME_EMBED_MODEL", "str"),
    _f("extraction", "max_llm_calls_per_job", "LME_MAX_LLM_CALLS_PER_JOB", "int"),
    _f("extraction", "api_key_env", API_KEY_VAR, "str", True),
    _f("ingest", "segmentation", "LME_SEGMENTATION", "str"),
    _f("ingest", "store_only_context", "LME_STORE_ONLY_CONTEXT", "bool"),
    _f("ingest", "assistant_store_only", "LME_ASSISTANT_STORE_ONLY", "bool"),
    _f("ingest", "writer_model", "LME_WRITER_MODEL", "str"),
    _f("ingest", "writer_granularity", "LME_WRITER_GRANULARITY", "str"),
    _f("ingest", "writer_max_usd", "LME_WRITER_MAX_USD", "float", True),
    _f("ingest", "writer_rates", "LME_WRITER_RATES", "str", True),
    _f("ingest", "concurrency", "LME_INGEST_CONCURRENCY", "int"),
    _f("ingest", "require_turn_evidence", "LME_REQUIRE_TURN_EVIDENCE", "bool"),
    _f("consolidation", "scalar_state", "LME_SCALAR_STATE_ENABLED", "bool"),
    _f("consolidation", "scalar_history", "LME_SCALAR_HISTORY_ENABLED", "bool"),
    _f("consolidation", "counter_state", "LME_COUNTER_STATE_ENABLED", "bool"),
    _f("consolidation", "event_history", "LME_EVENT_HISTORY_ENABLED", "bool"),
    _f("consolidation", "consolidation_k", "LME_SCALAR_CONSOLIDATION_K", "int"),
    _f("consolidation", "call_budget", "LME_SCALAR_CALL_BUDGET", "int"),
    _f("consolidation", "scalar_threshold", "LME_SCALAR_THRESHOLD", "str"),
    _f("consolidation", "require_scalar_output", "LME_REQUIRE_SCALAR_OUTPUT", "bool"),
    _f("consolidation", "consolidation_audit", "LME_CONSOLIDATION_AUDIT_ENABLED", "bool"),
    _f("consolidation", "recall_audit", "LME_RECALL_AUDIT_ENABLED", "bool"),
    _f("time", "anchored_time_resolver", "LME_ANCHORED_TIME_RESOLVER", "bool"),
    _f("time", "anchored_time_model", "LME_ANCHORED_TIME_MODEL", "str"),
    _f("time", "anchored_time_attach_model", "LME_ANCHORED_TIME_ATTACH_MODEL", "str", True),
    _f("time", "canonical_self_binding_mode", "LME_CANONICAL_SELF_BINDING_MODE", "str"),
    _f("post", "embed_backfill", "LME_EMBED_BACKFILL", "bool"),
    _f("post", "backfill_dates", "LME_BACKFILL_DATES", "bool"),
    _f("menhir", "log_level", "LME_MENHIR_LOG_LEVEL", "str"),
)

_BY_PATH = {f.path: f for f in SCHEMA}
_BY_VAR = {f.var: f for f in SCHEMA if f.var}
_SECTIONS: dict[str, set[str]] = {}
for _field in SCHEMA:
    _SECTIONS.setdefault(_field.section, set()).add(_field.key)

MAPPED_VARS = frozenset(_BY_VAR)

#: Run-control variables: not part of the build config, never hashed.
RUN_CONTROL_VARS = (
    "LME_RESUME",
    "LME_GRAPH_FRESH",
    "LME_REQUIRE_FRESH",
    "LME_RESET_FILE",
    "LME_RESULTS_DIR",
    "LME_MANIFEST_PATH",
    "LME_REVERT_SNAPSHOT",
    "LME_REVERT_SNAPSHOT_PATH",
    "LME_NONCANONICAL",
    "LME_INGEST_STOP_AFTER_ITEMS",
)


def _check_value(field: Field, value: Any) -> Any:
    if value is None:
        if not field.nullable:
            raise ValueError(f"{field.path}: null is not allowed")
        return None
    if field.kind == "bool":
        ok = isinstance(value, bool)
    elif field.kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif field.kind == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, str) and value != ""
    if not ok:
        raise ValueError(f"{field.path}: expected {field.kind}, got {value!r}")
    return value


def _load_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: build config must be a mapping")
    return data


def _validate_shape(data: dict, label: str) -> None:
    for sec, val in data.items():
        if sec == "extends":
            continue
        if sec not in _SECTIONS:
            raise ValueError(f"unknown key in {label}: {sec}")
        if not isinstance(val, dict):
            raise ValueError(f"{label}: {sec} must be a mapping")
        for key in val:
            if key not in _SECTIONS[sec]:
                raise ValueError(f"unknown key in {label}: {sec}.{key}")


def resolve_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a build YAML (with optional one-level ``extends``) into {section: {key: value}}."""
    path = Path(path)
    data = _load_mapping(path)
    merged: dict[str, dict[str, Any]] = {}
    base_name = data.get("extends")
    if base_name is not None:
        if not isinstance(base_name, str):
            raise ValueError("extends must be a string path")
        base_path = path.parent / base_name
        base = _load_mapping(base_path)
        if "extends" in base:
            raise ValueError(f"{base_path}: extends is supported one level deep only")
        _validate_shape(base, str(base_path.name))
        for sec, val in base.items():
            merged[sec] = dict(val)
    _validate_shape(data, str(path.name))
    for sec, val in data.items():
        if sec == "extends":
            continue
        merged.setdefault(sec, {}).update(val)
    for f in SCHEMA:
        if f.key not in merged.get(f.section, {}):
            raise ValueError(f"missing key: {f.path}")
        merged[f.section][f.key] = _check_value(f, merged[f.section][f.key])
    return {sec: {k: merged[sec][k] for k in sorted(merged[sec])} for sec in sorted(merged)}


def _coerce_env(field: Field, raw: str) -> Any:
    if raw == "" and field.nullable:
        return None
    try:
        if field.kind == "bool":
            if raw not in ("0", "1"):
                raise ValueError("expected 0 or 1")
            return raw == "1"
        if field.kind == "int":
            return int(raw)
        if field.kind == "float":
            return float(raw)
    except ValueError as exc:
        raise ValueError(f"{field.var}: cannot parse {raw!r} for {field.path}: {exc}") from exc
    return raw


def env_to_string(field: Field, value: Any) -> str:
    if field.kind == "bool":
        return "1" if value else "0"
    return str(value)


def apply_env_overrides(
    config: dict[str, dict[str, Any]], environ: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    """Apply deprecated LME_* overrides; returns (config, env_overrides, warnings)."""
    out = {sec: dict(val) for sec, val in config.items()}
    overrides: dict[str, Any] = {}
    warnings: list[str] = []
    for var in sorted(MAPPED_VARS):
        if var not in environ:
            continue
        field = _BY_VAR[var]
        warnings.append(
            f"deprecated env override {var} is set; it overrides the build YAML "
            f"({field.path}). Move it into the build YAML."
        )
        if var in SECRET_VARS:
            overrides[var] = "<redacted>"
            continue
        value = _check_value(field, _coerce_env(field, environ[var]))
        out[field.section][field.key] = value
        overrides[var] = value
    return out, overrides, warnings


def hashable_config(config: dict[str, dict[str, Any]], env_overrides: dict[str, Any]) -> dict:
    """The part of a build that identifies it: no secrets, no api_key_env name."""
    cfg = {sec: dict(val) for sec, val in config.items()}
    cfg["extraction"].pop("api_key_env", None)
    return {"config": cfg, "env_overrides": dict(env_overrides)}


def build_id(stem: str, config: dict[str, dict[str, Any]], env_overrides: dict[str, Any]) -> str:
    blob = json.dumps(hashable_config(config, env_overrides), sort_keys=True, separators=(",", ":"))
    return f"{stem}-{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:10]}"


def resolved_record(
    yaml_path: str | Path,
    config: dict[str, dict[str, Any]],
    env_overrides: dict[str, Any],
) -> dict:
    """The resolved build config as written to build.resolved.yaml (no secrets)."""
    return {
        "build_id": build_id(Path(yaml_path).stem, config, env_overrides),
        "build_yaml": Path(yaml_path).as_posix(),
        "config": config,
        "env_overrides": env_overrides,
    }


def env_assignments(
    config: dict[str, dict[str, Any]],
    environ: dict[str, str],
    env_overrides: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """The LME_* assignments to set, and the mapped variables to remove from the child env.

    The API key is read from the env var named by extraction.api_key_env (or taken from a
    deprecated LME_EXTRACT_API_KEY override); a missing named variable raises KeyError.
    """
    sets: dict[str, str] = {}
    unset: list[str] = []
    for f in SCHEMA:
        if f.var is None or f.var == API_KEY_VAR:
            continue
        value = config[f.section][f.key]
        if value is None:
            unset.append(f.var)
        else:
            sets[f.var] = env_to_string(f, value)
    if API_KEY_VAR in env_overrides:
        sets[API_KEY_VAR] = environ[API_KEY_VAR]
    else:
        key_env = config["extraction"]["api_key_env"]
        if key_env is None:
            unset.append(API_KEY_VAR)
        else:
            if not environ.get(key_env):
                raise KeyError(key_env)
            sets[API_KEY_VAR] = environ[key_env]
    return sets, unset


def child_env(
    environ: dict[str, str] | None,
    sets: dict[str, str],
    unset: list[str],
) -> dict[str, str]:
    """os.environ minus every mapped LME_* variable, plus the resolved assignments."""
    env = dict(os.environ if environ is None else environ)
    for var in MAPPED_VARS:
        env.pop(var, None)
    for var in unset:
        env.pop(var, None)
    env.update(sets)
    return env
