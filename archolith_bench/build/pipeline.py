"""LongMemEval graph build: checks, Neo4j, provenance, Menhir serve, ingest and post-steps.

Moved out of build_graph.sh (P4b step 5). build_graph.sh still sources config.sh, then hands every
LME_*/MENHIR_*/BENCH_* shell variable over on stdin. os.environ stays exactly the set the shell
exported: children (ingest.py takes argparse defaults from it) see what they saw before.

Usage (from build_graph.sh only): <NUL-separated name/value pairs on stdin> pipeline.py <limit>

The settings Menhir, the attempt record and the ingest phase receive are tables of the shell
expressions build_graph.sh used, so tests can compare them with retry.sh line for line.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LME_DIR = Path(__file__).resolve().parents[2] / "scripts" / "longmemeval"
LIB = LME_DIR / "lib"


class Die(SystemExit):
    """A refusal with build_graph.sh's message and exit status 1."""

    def __init__(self, message: str) -> None:
        super().__init__(1)
        self.message = message


def log(message: str) -> None:
    print(f"[lme-build] {message}", file=sys.stderr, flush=True)


# ---- settings -------------------------------------------------------------------------------

_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\{[^{}]*\})*))?\}")
_MSYS_DRIVE = re.compile(r"^/([A-Za-z])(/|$)")


def expand(expr: str, s: dict[str, str]) -> str:
    """``${A}`` and ``${A:-default}`` (default may hold one nested reference), like bash ``set -u``."""

    def sub(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        value = s.get(name)
        if default is not None and not value:
            return expand(default, s)
        if value is None:
            raise Die(f"{name}: unbound variable")
        return value

    return _REF.sub(sub, expr)


def native(value: str) -> str:
    """An MSYS path (/c/Users/...) as Windows sees it. Bash converts exported values on exec;
    values read from stdin skip that, so they are converted here."""
    if os.name != "nt":
        return value
    return _MSYS_DRIVE.sub(lambda m: f"{m.group(1).upper()}:/", value)


def read_settings(stream, environ: dict[str, str]) -> dict[str, str]:
    data = stream.read()
    if isinstance(data, str):
        data = data.encode("utf-8", "surrogateescape")
    parts = data.split(b"\0")
    if not data or len(parts) % 2 != 1 or parts[-1] != b"":
        raise Die("no settings on stdin; run scripts/longmemeval/build_graph.sh, not this module")
    s = dict(environ)
    for i in range(0, len(parts) - 1, 2):
        s[parts[i].decode("utf-8")] = native(parts[i + 1].decode("utf-8", "surrogateescape"))
    return s


# ---- tables (shell expressions from build_graph.sh) ---------------------------------------------

# Everything the ingest Menhir and the later steps inherit, in export order. Values that need code
# (the API key, ENV_FILE, the run id) are put into the settings under their shell names first.
MENHIR_ENV: tuple[tuple[str, str], ...] = (
    ("ENV_FILE", "${EMPTY_ENV}"),
    # LongMemEval variant: default 'oracle' (evidence-only haystacks, ~22 turns/item vs ~494 for 's').
    ("LONGMEMEVAL_VARIANT", "${LONGMEMEVAL_VARIANT:-oracle}"),
    ("LME_NEO4J_CONTAINER", "${LME_NEO4J_NAME}"),
    ("LME_NEO4J_PW", "${LME_NEO4J_PW}"),
    # Own log dir, so the ingest Menhir never shares server.log with another Menhir process.
    ("MENHIR_LOG_DIR", "${LME_RESULTS_DIR}/menhir-logs"),
    ("MENHIR_BENCHMARK_MODE", "1"),
    ("MENHIR_API_HOST", "127.0.0.1"),
    ("MENHIR_API_PORT", "${MENHIR_PORT}"),
    ("MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED", "${LME_SCALAR_STATE_ENABLED}"),
    ("MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED", "${LME_SCALAR_HISTORY_ENABLED}"),
    # Typed-scalar vote threshold (see config.sh for the grid and the 0.67-vs-2/3 trap).
    ("MENHIR_PERSONAL_MEMORY_SCALAR_THRESHOLD", "${LME_SCALAR_THRESHOLD}"),
    ("MENHIR_CANONICAL_SELF_BINDING_MODE", "${LME_CANONICAL_SELF_BINDING_MODE}"),
    ("MENHIR_PERSONAL_MEMORY_CHAT_MODEL", "${LME_EXTRACT_MODEL}"),
    ("LME_REQUIRE_TURN_EVIDENCE", "${LME_REQUIRE_TURN_EVIDENCE}"),
    ("LME_STORE_ONLY_CONTEXT", "${LME_STORE_ONLY_CONTEXT}"),
    ("LME_ASSISTANT_STORE_ONLY", "${LME_ASSISTANT_STORE_ONLY}"),
    ("MENHIR_MAX_LLM_CALLS_PER_JOB", "${LME_MAX_LLM_CALLS_PER_JOB}"),
    # retry.sh exports the same values (tests/test_lme_retry_parity.py).
    ("MENHIR_ANCHORED_TIME_RESOLVER", "${LME_ANCHORED_TIME_RESOLVER}"),
    ("MENHIR_ANCHORED_TIME_MODEL", "${LME_ANCHORED_TIME_MODEL}"),
    ("MENHIR_ANCHORED_TIME_ATTACH_MODEL", "${LME_ANCHORED_TIME_ATTACH_MODEL}"),
    ("MENHIR_OPENAI_SERVICE_TIER", "${LME_OPENAI_SERVICE_TIER}"),
    # ingest.py uses the same value for its namespace window (one active episode per namespace).
    ("MENHIR_INGEST_CONCURRENCY", "${LME_INGEST_CONCURRENCY}"),
    ("MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED", "${LME_CONSOLIDATION_AUDIT_ENABLED}"),
    ("MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED", "${LME_RECALL_AUDIT_ENABLED}"),
    ("MENHIR_LOG_LEVEL", "${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}"),
    # Per-run telemetry sidecar (vote receipts, lifecycle events, scalar audits, LLM usage).
    ("MENHIR_MCP_TELEMETRY_DB", "${LME_RESULTS_DIR}/mcp_telemetry.db"),
    ("MENHIR_BENCH_RESULTS_ROOT", "${BENCH_DIR}/results"),
    ("MENHIR_BENCH_ACTIVE_RUN_ID", "${RESULTS_DIR_NAME}"),
    ("OTEL_SDK_DISABLED", "true"),
    ("LANGFUSE_TRACING_ENABLED", "false"),
    ("LANGFUSE_PUBLIC_KEY", ""),
    ("LANGFUSE_SECRET_KEY", ""),
    ("LANGFUSE_HOST", ""),
    ("MENHIR_OPERATOR_KEY", ""),
    ("MENHIR_AGENT_KEY", ""),
    ("MENHIR_READONLY_KEY", ""),
    ("MENHIR_API_KEY", ""),
    ("NEO4J_URI", "bolt://localhost:${LME_BOLT}"),
    ("NEO4J_USER", "neo4j"),
    ("NEO4J_PASSWORD", "${LME_NEO4J_PW}"),
    ("NEO4J_DATABASE", "neo4j"),
    ("LME_NEO4J_HTTP_URL", "http://localhost:${LME_HTTP}"),
)

_PROVIDER_ENV: dict[str, tuple[tuple[str, str], ...]] = {
    "openai": (
        ("GRAPHITI_LLM_PROVIDER", "openai"), ("MEMORY_GRAPHITI_PROVIDER", "openai"),
        ("LLM_CHAT_PROVIDER", "openai"), ("MEMORY_CHAT_PROVIDER", "openai"),
        ("GRAPHITI_RERANKER_PROVIDER", "openai"), ("MEMORY_GRAPHITI_RERANKER_PROVIDER", "openai"),
    ),
    "local": (
        ("GRAPHITI_LLM_PROVIDER", "local"), ("MEMORY_GRAPHITI_PROVIDER", "local"),
        ("LLM_CHAT_PROVIDER", "local"), ("MEMORY_CHAT_PROVIDER", "local"),
        ("GRAPHITI_RERANKER_PROVIDER", "local"), ("MEMORY_GRAPHITI_RERANKER_PROVIDER", "local"),
        ("LOCAL_LLM_BASE_URL", "${LME_EXTRACT_BASE_URL}"),
        ("LOCAL_LLM_API_KEY", "${LME_EXTRACT_API_KEY}"),
        ("LOCAL_LLM_CHAT_MODEL", "${LME_EXTRACT_MODEL}"),
    ),
}

_EMBED_ENV: tuple[tuple[str, str], ...] = (
    ("GRAPHITI_EMBED_PROVIDER", "openai"), ("MEMORY_GRAPHITI_EMBED_PROVIDER", "openai"),
    ("OPENAI_API_KEY", "${OPENAI_KEY}"),
    ("OPENAI_CHAT_MODEL", "${LME_EXTRACT_MODEL}"),
    ("OPENAI_EMBED_MODEL", "${LME_EMBED_MODEL}"),
)

# The attempt record ``run_provenance.py begin`` checks. "int" and "bool" render unquoted, as the
# heredoc did; retry.sh must render RETRY_BEHAVIOR_KEYS the same way.
ATTEMPT_RECORD: tuple[tuple[str, str, str], ...] = (
    ("container", "${LME_NEO4J_NAME}", "str"),
    ("volume", "${LME_NEO4J_VOL}", "str"),
    ("volume_pre_existed", "${VOLUME_PRE_EXISTED}", "bool"),
    ("graph_fresh", "${GRAPH_FRESH}", "bool"),
    ("require_fresh", "${LME_REQUIRE_FRESH}", "int"),
    ("resumed", "${LME_RESUME}", "int"),
    ("dataset", "${LME_DATASET}", "str"),
    ("variant", "${LONGMEMEVAL_VARIANT}", "str"),
    ("fixture_sha256", "${LME_FIXTURE_SHA256:-}", "str"),
    ("fixture_count", "${LME_FIXTURE_COUNT:-0}", "int"),
    ("fixture", "${LME_FIXTURE_PATH:-}", "str"),
    ("requested_items", "${LIMIT}", "int"),
    ("ingest_target_items", "${INGEST_TARGET}", "int"),
    ("namespace_prefix", "${LME_NS_PREFIX}", "str"),
    ("segmentation", "${LME_SEGMENTATION}", "str"),
    ("ingest_concurrency", "${LME_INGEST_CONCURRENCY}", "int"),
    ("counter_state_enabled", "${LME_COUNTER_STATE_ENABLED}", "int"),
    ("event_history_enabled", "${LME_EVENT_HISTORY_ENABLED}", "int"),
    ("scalar_state_enabled", "${LME_SCALAR_STATE_ENABLED}", "int"),
    ("scalar_history_enabled", "${LME_SCALAR_HISTORY_ENABLED}", "int"),
    ("scalar_consolidation_k", "${LME_SCALAR_CONSOLIDATION_K}", "int"),
    ("scalar_threshold", "${LME_SCALAR_THRESHOLD}", "str"),
    ("canonical_self_binding_mode", "${LME_CANONICAL_SELF_BINDING_MODE}", "str"),
    ("extract_model", "${LME_EXTRACT_MODEL}", "str"),
    ("anchored_time_resolver_enabled", "${LME_ANCHORED_TIME_RESOLVER}", "int"),
    ("anchored_time_model", "${LME_ANCHORED_TIME_MODEL}", "str"),
    ("anchored_time_attach_model", "${LME_ANCHORED_TIME_ATTACH_MODEL}", "str"),
    ("openai_service_tier", "${LME_OPENAI_SERVICE_TIER}", "str"),
    ("max_llm_calls_per_job", "${LME_MAX_LLM_CALLS_PER_JOB}", "int"),
    ("scalar_output_required", "${LME_REQUIRE_SCALAR_OUTPUT}", "int"),
    ("turn_evidence_required", "${LME_REQUIRE_TURN_EVIDENCE}", "int"),
    ("consolidation_audit_enabled", "${LME_CONSOLIDATION_AUDIT_ENABLED}", "int"),
    ("recall_audit_enabled", "${LME_RECALL_AUDIT_ENABLED}", "int"),
    ("menhir_log_level", "${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}", "str"),
    ("menhir_commit", "${MENHIR_COMMIT}", "str"),
    ("bench_commit", "${BENCH_COMMIT}", "str"),
    ("surface_digest", "${GRAPH_SURFACE_DIGEST}", "str"),
    ("started_at", "${STARTED_AT}", "str"),
    ("build_started_at", "${STARTED_AT}", "str"),
    ("writer_model", "${WRITER_MODEL_RECORD}", "str"),
    ("writer_granularity", "${WRITER_GRANULARITY_RECORD}", "str"),
    ("writer_prompt_sha", "${WRITER_PROMPT_SHA}", "str"),
)

# ``phase-start --phase ingest-graph`` settings, after the optional fixture settings.
PHASE_SETTINGS: tuple[tuple[str, str], ...] = (
    ("menhir_commit", "${MENHIR_COMMIT}"),
    ("bench_commit", "${BENCH_COMMIT}"),
    ("menhir_dirty", "${MENHIR_DIRTY}"),
    ("bench_dirty", "${BENCH_DIRTY}"),
    ("menhir_untracked", "${MENHIR_UNTRACKED_COUNT}"),
    ("bench_untracked", "${BENCH_UNTRACKED_COUNT}"),
    ("dataset", "${LME_DATASET}"),
    ("variant", "${LONGMEMEVAL_VARIANT}"),
    ("namespace_prefix", "${LME_NS_PREFIX}"),
    ("graph_fresh", "${GRAPH_FRESH}"),
    ("volume_pre_existed", "${VOLUME_PRE_EXISTED}"),
    ("require_fresh", "${LME_REQUIRE_FRESH}"),
    ("resumed", "${LME_RESUME}"),
    ("segmentation", "${LME_SEGMENTATION}"),
    ("store_only_context", "${LME_STORE_ONLY_CONTEXT}"),
    ("assistant_store_only", "${LME_ASSISTANT_STORE_ONLY}"),
    ("embed_backfill", "${LME_EMBED_BACKFILL}"),
    ("strict_failure_policy", "zero-failed-episodes-per-namespace"),
    ("ingest_target_items", "${INGEST_TARGET}"),
    ("ingest_concurrency", "${LME_INGEST_CONCURRENCY}"),
    ("counter_state_enabled", "${LME_COUNTER_STATE_ENABLED}"),
    ("event_history_enabled", "${LME_EVENT_HISTORY_ENABLED}"),
    ("scalar_state_enabled", "${LME_SCALAR_STATE_ENABLED}"),
    ("scalar_history_enabled", "${LME_SCALAR_HISTORY_ENABLED}"),
    ("scalar_consolidation_k", "${LME_SCALAR_CONSOLIDATION_K}"),
    ("scalar_consolidation_call_budget", "${LME_SCALAR_CALL_BUDGET}"),
    ("scalar_threshold", "${LME_SCALAR_THRESHOLD}"),
    ("consolidation_audit_enabled", "${LME_CONSOLIDATION_AUDIT_ENABLED}"),
    ("recall_audit_enabled", "${LME_RECALL_AUDIT_ENABLED}"),
    ("menhir_log_level", "${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}"),
    ("turn_evidence_required", "${LME_REQUIRE_TURN_EVIDENCE}"),
    ("extract_model", "${LME_EXTRACT_MODEL}"),
    ("anchored_time_resolver_enabled", "${LME_ANCHORED_TIME_RESOLVER}"),
    ("anchored_time_model", "${LME_ANCHORED_TIME_MODEL}"),
    ("anchored_time_attach_model", "${LME_ANCHORED_TIME_ATTACH_MODEL}"),
    ("openai_service_tier", "${LME_OPENAI_SERVICE_TIER}"),
    ("max_llm_calls_per_job", "${LME_MAX_LLM_CALLS_PER_JOB}"),
)


# ---- pure steps -----------------------------------------------------------------------------

def _uint(value: str) -> bool:
    return value.isdigit()


def validate(s: dict[str, str], limit: str) -> None:
    """Refuse bad settings before Docker or any paid call. Sets INGEST_TARGET and LIMIT."""
    if s.get("LME_CANONICAL_SELF_BINDING_MODE") not in ("off", "observe", "enforce"):
        raise Die("LME_CANONICAL_SELF_BINDING_MODE must be off, observe, or enforce")
    for name in ("LME_COUNTER_STATE_ENABLED", "LME_SCALAR_STATE_ENABLED", "LME_EVENT_HISTORY_ENABLED"):
        if s.get(name) not in ("0", "1"):
            raise Die("processing lane settings must be 0 or 1")
    # Menhir removed the consolidation counter lane (2026-10-28); kept as a setting so earlier
    # graphs' provenance (counter_state_enabled=0) resumes.
    if s["LME_COUNTER_STATE_ENABLED"] != "0":
        raise Die("LME_COUNTER_STATE_ENABLED=1: Menhir removed the consolidation counter lane (2026-10-28)")
    if s["LME_EVENT_HISTORY_ENABLED"] != "0":
        raise Die("LME_EVENT_HISTORY_ENABLED=1: Menhir archived the event-history lane (2026-10-28)")
    if s.get("LME_ANCHORED_TIME_RESOLVER") not in ("0", "1"):
        raise Die("LME_ANCHORED_TIME_RESOLVER must be 0 or 1")
    # The resolver uses the extraction client, so a local endpoint would get the OpenAI model name.
    if s.get("LME_EXTRACT_PROVIDER") != "openai" and s["LME_ANCHORED_TIME_RESOLVER"] != "0":
        raise Die("anchored time needs LME_EXTRACT_PROVIDER=openai; set LME_ANCHORED_TIME_RESOLVER=0 for local extraction")
    concurrency = s.get("LME_INGEST_CONCURRENCY", "")
    if not _uint(concurrency) or int(concurrency) == 0:
        raise Die("LME_INGEST_CONCURRENCY must be a positive integer")
    if s.get("LME_RESUME") not in ("0", "1"):
        raise Die("LME_RESUME must be 0 or 1")
    stop_after = s.get("LME_INGEST_STOP_AFTER_ITEMS") or "0"
    if not _uint(stop_after):
        raise Die("LME_INGEST_STOP_AFTER_ITEMS must be a non-negative integer")
    if not _uint(limit):
        raise Die("item limit must be a non-negative integer")
    if int(stop_after) > int(limit):
        raise Die("LME_INGEST_STOP_AFTER_ITEMS cannot exceed the requested item limit")
    s["LME_INGEST_STOP_AFTER_ITEMS"] = stop_after
    s["LIMIT"] = limit
    s["INGEST_TARGET"] = stop_after if int(stop_after) > 0 else limit


def check_fixture(s: dict[str, str]) -> None:
    """A fixture is part of the graph identity: check its bytes and count against any pins."""
    path_text = s.get("LME_FIXTURE_PATH") or ""
    if not path_text:
        return
    path = Path(path_text)
    if not path.is_file():
        raise Die(f"fixture not found: {path_text}")
    raw = path.read_bytes()
    try:
        items = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Die(f"invalid fixture: {path_text} ({exc})") from exc
    if not isinstance(items, list) or len(items) < int(s["LIMIT"]):
        raise Die(f"invalid fixture: {path_text} (must be a list with at least the requested item count)")
    sha, count = hashlib.sha256(raw).hexdigest(), str(len(items))
    if s.get("LME_FIXTURE_SHA256") and s["LME_FIXTURE_SHA256"] != sha:
        raise Die("LME_FIXTURE_SHA256 does not match fixture bytes")
    if s.get("LME_FIXTURE_COUNT") and s["LME_FIXTURE_COUNT"] != count:
        raise Die("LME_FIXTURE_COUNT does not match fixture items")
    s["LME_FIXTURE_SHA256"], s["LME_FIXTURE_COUNT"] = sha, count


def check_graph_gates(s: dict[str, str], *, container_exists: bool, volume_exists: bool) -> None:
    """Resume reuses only the graph an earlier attempt built; a fresh build refuses any leftover."""
    name, vol = s["LME_NEO4J_NAME"], s["LME_NEO4J_VOL"]
    provenance = Path(s["LME_RESULTS_DIR"]) / f"graph-provenance-{name}.json"
    manifest = Path(s["LME_MANIFEST_PATH"])
    if s["LME_RESUME"] == "1":
        if not provenance.is_file():
            raise Die(f"resume refused: provenance file not found: {provenance.as_posix()}")
        if not container_exists:
            raise Die(f"resume refused: container does not exist: {name}")
        if not volume_exists:
            raise Die(f"resume refused: volume does not exist: {vol}")
        if not manifest.parent.is_dir():
            raise Die(f"resume refused: manifest directory not found: {manifest.parent.as_posix()}")
        try:
            identity = json.loads(provenance.read_text(encoding="utf-8")).get("identity") or {}
        except (OSError, ValueError, AttributeError) as exc:
            raise Die(f"resume refused: cannot read identity from {provenance.as_posix()}") from exc
        if identity.get("container", "") != name:
            raise Die(f"resume refused: provenance container '{identity.get('container', '')}' != {name}")
        if identity.get("volume", "") != vol:
            raise Die(f"resume refused: provenance volume '{identity.get('volume', '')}' != {vol}")
    elif s.get("LME_REQUIRE_FRESH") == "1":
        if volume_exists:
            raise Die(f"fresh build refused: volume already exists: {vol}")
        if container_exists:
            raise Die(f"fresh build refused: container already exists: {name}")
        if manifest.exists():
            raise Die(f"fresh build refused: manifest already exists: {manifest.as_posix()}")
        # Not deleted automatically: provenance is an audit record (--clean removes it).
        if provenance.exists():
            raise Die(
                f"fresh build refused: provenance from an earlier attempt already exists: "
                f"{provenance.as_posix()} (remove it, or re-run with --clean)"
            )


def neo4j_run_args(s: dict[str, str]) -> list[str]:
    # Loopback only: the docker host may be remote (reach it through an ssh -L tunnel).
    return [
        "docker", "run", "-d", "--name", s["LME_NEO4J_NAME"],
        "-p", expand("127.0.0.1:${LME_BOLT}:7687", s),
        "-p", expand("127.0.0.1:${LME_HTTP}:7474", s),
        "-e", expand("NEO4J_AUTH=neo4j/${LME_NEO4J_PW}", s),
        "-e", expand("NEO4J_server_memory_heap_max__size=${LME_NEO4J_HEAP}", s),
        "-v", expand("${LME_NEO4J_VOL}:/data", s),
        s["LME_NEO4J_IMAGE"],
    ]


def attempt_record(s: dict[str, str]) -> dict[str, object]:
    record: dict[str, object] = {}
    for key, expr, kind in ATTEMPT_RECORD:
        value = expand(expr, s)
        if kind == "int":
            try:
                record[key] = int(value)
            except ValueError as exc:
                raise Die(f"attempt record: {key}={value!r} is not an integer") from exc
        elif kind == "bool":
            if value not in ("true", "false"):
                raise Die(f"attempt record: {key}={value!r} is not true/false")
            record[key] = value == "true"
        else:
            record[key] = value
    return record


def phase_start_settings(s: dict[str, str]) -> list[str]:
    """Fixture facts only when known (build_graph.sh can run without one), then PHASE_SETTINGS."""
    args: list[str] = []
    for key, var in (("fixture", "LME_FIXTURE_PATH"), ("fixture_sha256", "LME_FIXTURE_SHA256"),
                     ("fixture_count", "LME_FIXTURE_COUNT")):
        if s.get(var):
            args += ["--setting", f"{key}={s[var]}"]
    for key, expr in PHASE_SETTINGS:
        args += ["--setting", f"{key}={expand(expr, s)}"]
    if s["LME_SEGMENTATION"] == "adaptive-writer":
        for key, var in (("writer_model", "WRITER_MODEL_RECORD"), ("writer_granularity", "WRITER_GRANULARITY_RECORD"),
                         ("writer_prompt_sha", "WRITER_PROMPT_SHA")):
            args += ["--setting", f"{key}={s[var]}"]
    return args


def menhir_env(s: dict[str, str], base: dict[str, str]) -> dict[str, str]:
    """``base`` plus every export, applied in order to both the result and ``s``."""
    provider = s.get("LME_EXTRACT_PROVIDER", "")
    if provider not in _PROVIDER_ENV:
        raise Die(f"unsupported LME_EXTRACT_PROVIDER: {provider} (expected openai or local)")
    if provider == "local" and not s.get("LME_EXTRACT_BASE_URL"):
        raise Die("LME_EXTRACT_BASE_URL is required for local extraction")
    env = dict(base)
    for name, expr in (*MENHIR_ENV, *_PROVIDER_ENV[provider], *_EMBED_ENV):
        env[name] = s[name] = expand(expr, s)
    return env


def ingest_args(s: dict[str, str]) -> list[str]:
    args = [
        "--limit", s["LIMIT"],
        "--menhir-url", s["MENHIR_URL"],
        "--manifest", s["LME_MANIFEST_PATH"],
        "--segmentation", s["LME_SEGMENTATION"],
        "--namespace-window", s["LME_INGEST_CONCURRENCY"],
    ]
    if s.get("LME_FIXTURE_PATH"):
        args += ["--fixture", s["LME_FIXTURE_PATH"]]
    # config.sh does not export the writer settings, so they go on the command line.
    if s["LME_SEGMENTATION"] == "adaptive-writer":
        args += ["--writer-granularity", s["LME_WRITER_GRANULARITY"], "--writer-model", s["LME_WRITER_MODEL"]]
        if s.get("LME_WRITER_MAX_USD"):
            args += ["--writer-max-usd", s["LME_WRITER_MAX_USD"]]
        if s.get("LME_WRITER_RATES"):
            args += ["--writer-rates", s["LME_WRITER_RATES"]]
    if s.get("LME_RESET_FILE"):
        args += ["--reset-file", s["LME_RESET_FILE"]]
    if int(s["LME_INGEST_STOP_AFTER_ITEMS"]) > 0:
        args += ["--manifest-item-limit", s["LME_INGEST_STOP_AFTER_ITEMS"]]
    if s["LME_COUNTER_STATE_ENABLED"] == "1":
        args += ["--consolidate-counter"]
    args += ["--consolidation-k", s["LME_SCALAR_CONSOLIDATION_K"],
             "--consolidation-call-budget", s["LME_SCALAR_CALL_BUDGET"]]
    if s["LME_SCALAR_STATE_ENABLED"] == "1":
        args += ["--consolidate-scalar"]
    return args


def serving_tree_problem(s: dict[str, str]) -> str | None:
    """Why the serving tree cannot run these settings, checked before any paid work. An older
    Menhir ignores an unknown field or flag and would extract what the settings store only."""
    frontier = Path(s["MENHIR_FRONTIER"])

    def has(relative: str, needle: str) -> bool:
        try:
            return needle in (frontier / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

    if s["LME_ASSISTANT_STORE_ONLY"] == "1" and s["LME_SEGMENTATION"] != "adaptive":
        return f"LME_ASSISTANT_STORE_ONLY=1 needs LME_SEGMENTATION=adaptive (got {s['LME_SEGMENTATION']})"
    if s["LME_STORE_ONLY_CONTEXT"] == "1" and not has("src/menhir/api/routes_support.py", "store_only"):
        return (f"LME_STORE_ONLY_CONTEXT=1 but {s['MENHIR_FRONTIER']} has no store_only ingest (Menhir #388); "
                "set it to 0 or update the tree")
    # The name stays in feature_flags.py (as RETIRED) after the #168 freeze made the policy always on.
    if s["LME_ASSISTANT_STORE_ONLY"] == "1" and not has(
            "src/menhir/config/feature_flags.py", "MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED"):
        return (f"LME_ASSISTANT_STORE_ONLY=1 but {s['MENHIR_FRONTIER']} has no assistant-turn policy "
                "(MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED); update the tree")
    if s["LME_EMBED_BACKFILL"] == "1" and not has("scripts/backfill_episode_embeddings.py", "--require-complete"):
        return (f"LME_EMBED_BACKFILL=1 but {s['MENHIR_FRONTIER']} has no backfill --require-complete "
                "(Menhir #389); set it to 0 or update the tree")
    return None


def scalar_problem(rows: list[dict], *, expected: int, require_output: bool, history: bool) -> str | None:
    if not rows:
        return "scalar ingest validation failed: empty manifest"
    if len(rows) != expected:
        return f"scalar ingest validation failed: manifest has {len(rows)} rows, expected {expected}"
    if any(not row.get("scalar_consolidated") for row in rows):
        return "scalar ingest validation failed: at least one namespace was not consolidated"
    if sum(int(row.get("turn_evidence", 0)) for row in rows) <= 0:
        return "scalar ingest validation failed: no TurnEvidence captured"
    if sum(int(row.get("scalar_llm_calls", 0)) for row in rows) <= 0:
        return "scalar ingest validation failed: scalar perception made no LLM calls"
    if require_output:
        if sum(int(row.get("typed_assertions", 0)) for row in rows) <= 0:
            return "scalar ingest validation failed: no TypedAssertion materialized"
        if sum(int(row.get("scalar_views", 0)) for row in rows) <= 0:
            return "scalar ingest validation failed: no scalar_state View materialized"
        if history and sum(int(row.get("scalar_history_views", 0)) for row in rows) <= 0:
            return "scalar ingest validation failed: no scalar_history View materialized"
    return None


# ---- side effects ---------------------------------------------------------------------------

def _run(cmd: list[str], *, env: dict[str, str], cwd: str | None = None) -> None:
    """A step whose failure stops the build with its exit status (``set -e``)."""
    rc = subprocess.run(cmd, env=env, cwd=cwd).returncode
    if rc != 0:
        log(f"step failed (exit {rc}): {Path(cmd[1]).name if len(cmd) > 1 else cmd[0]}")
        raise SystemExit(rc)


def _out(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def _docker_ok(args: list[str], env: dict[str, str]) -> bool:
    return subprocess.run(["docker", *args], env=env, capture_output=True).returncode == 0


def container_running(name: str, env: dict[str, str]) -> bool:
    _, out = _out(["docker", "container", "inspect", "--format", "{{.State.Running}}", name], env=env)
    return out == "true"


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310 - loopback health URL
            return 200 <= resp.status < 300
    except (OSError, ValueError):
        return False


def wait_http(url: str, tries: int, *, alive=lambda: True) -> bool:
    for _ in range(tries):
        if http_ok(url):
            return True
        if not alive():
            return False
        time.sleep(2)
    return http_ok(url)


def git_facts(repo: str, env: dict[str, str], pathspecs: list[str]) -> tuple[str, bool, int, list[str]]:
    rc, commit = _out(["git", "-C", repo, "rev-parse", "HEAD"], env=env)
    _, status = _out(["git", "-C", repo, "status", "--porcelain"], env=env)
    _, untracked = _out(["git", "-C", repo, "ls-files", "--others", "--exclude-standard"], env=env)
    _, source = _out(["git", "-C", repo, "ls-files", "--others", "--exclude-standard", "--", *pathspecs], env=env)
    return (commit if rc == 0 and commit else "unknown", bool(status),
            len(untracked.splitlines()), source.split())


def openai_key(s: dict[str, str], env: dict[str, str]) -> str:
    key = s.get("OPENAI_API_KEY") or ""
    if not key:
        code = "import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],''))"
        _, key = _out([s["MENHIR_FRONTIER_PY"], "-c", code, f"{s['BENCH_DIR']}/.env", "OPENAI_API_KEY"], env=env)
    if not key:
        raise Die(f"OPENAI_API_KEY is unset and absent from {s['BENCH_DIR']}/.env")
    return key


def run(s: dict[str, str], limit: str, base: dict[str, str]) -> int:
    py = s["BENCH_PY"]
    validate(s, limit)
    check_fixture(s)
    s["OPENAI_KEY"] = openai_key(s, base)
    s["MENHIR_PORT"] = s["LME_PORT_BUILD"]
    s["MENHIR_URL"] = expand("http://localhost:${MENHIR_PORT}", s)

    # ---- persistent Neo4j (reuse if already up) ----
    # "graph_fresh" must mean the DATA is new: `docker run -v` silently reattaches an existing
    # named volume, so the volume is checked before docker run would create it.
    name = s["LME_NEO4J_NAME"]
    volume_existed = _docker_ok(["volume", "inspect", s["LME_NEO4J_VOL"]], base)
    container_existed = _docker_ok(["container", "inspect", name], base)
    check_graph_gates(s, container_exists=container_existed, volume_exists=volume_existed)
    s["VOLUME_PRE_EXISTED"] = "true" if volume_existed else "false"
    s["GRAPH_FRESH"] = "false"
    if not container_running(name, base):
        if container_existed:
            log(f"starting existing {name}...")
            _run(["docker", "start", name], env=base)
        else:
            log(f"creating persistent Neo4j {name} (bolt {s['LME_BOLT']})...")
            _run(neo4j_run_args(s), env=base)
            if not volume_existed:
                s["GRAPH_FRESH"] = "true"
    log(f"waiting for Neo4j HTTP ({s['LME_HTTP']})...")
    if not wait_http(expand("http://localhost:${LME_HTTP}", s), 60):
        raise Die("Neo4j not ready")

    # ---- provenance: append an attempt, then open the ingest-graph phase ----
    results = Path(s["LME_RESULTS_DIR"])
    results.mkdir(parents=True, exist_ok=True)
    provenance = (results / f"graph-provenance-{name}.json").as_posix()
    tool = (LIB / "run_provenance.py").as_posix()
    if not Path(tool).is_file():
        raise Die(f"provenance helper missing: {tool}")
    surface_tool = (LIB / "bench_surface.py").as_posix()
    # Fingerprinted before the attempt record so `begin` can treat the digest as identity; empty
    # on failure, which run_provenance.py reads as absent.
    rc, digest = _out([py, surface_tool, "fingerprint", "--digest-only"], env=base)
    s["GRAPH_SURFACE_DIGEST"] = digest if rc == 0 else ""
    s["WRITER_PROMPT_SHA"] = s["WRITER_MODEL_RECORD"] = s["WRITER_GRANULARITY_RECORD"] = ""
    if s["LME_SEGMENTATION"] == "adaptive-writer":
        s["WRITER_MODEL_RECORD"] = s["LME_WRITER_MODEL"]
        s["WRITER_GRANULARITY_RECORD"] = s["LME_WRITER_GRANULARITY"]
        rc, sha = _out([py, (LIB / "memory_writer.py").as_posix(), "prompt-sha", s["LME_WRITER_GRANULARITY"]], env=base)
        if rc != 0 or not sha:
            raise Die("could not compute writer prompt sha")
        s["WRITER_PROMPT_SHA"] = sha
    menhir_facts = git_facts(s["MENHIR_MAIN"], base, ["src/**/*.py", "scripts/**/*.py", "src/**/*.sh", "scripts/**/*.sh"])
    bench_facts = git_facts(s["BENCH_DIR"], base, ["scripts/**/*.py", "archolith_bench/**/*.py", "scripts/**/*.sh"])
    s["MENHIR_COMMIT"], s["BENCH_COMMIT"] = menhir_facts[0], bench_facts[0]
    s["MENHIR_DIRTY"] = "true" if menhir_facts[1] else "false"
    s["BENCH_DIRTY"] = "true" if bench_facts[1] else "false"
    s["MENHIR_UNTRACKED_COUNT"], s["BENCH_UNTRACKED_COUNT"] = str(menhir_facts[2]), str(bench_facts[2])
    s["STARTED_AT"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    attempt = results / f".graph-attempt-{name}.json"
    attempt.write_text(json.dumps(attempt_record(s), indent=2) + "\n", encoding="utf-8", newline="\n")
    begin = [py, tool, "begin", provenance, attempt.as_posix()]
    if s.get("LME_NONCANONICAL") == "1":
        begin.append("--noncanonical")
    if subprocess.run(begin, env=base).returncode != 0:
        raise Die("graph provenance refused this attempt (identity, commit, or surface mismatch); see above")
    attempt.unlink(missing_ok=True)
    # A fingerprint is evidence about a run, not a precondition for it: failure only warns.
    if subprocess.run([py, surface_tool, "attach", provenance], env=base).returncode != 0:
        log("WARNING: surface fingerprint failed; this graph will not be attributable by file")
    _run([py, tool, "phase-start", provenance, "--phase", "ingest-graph", "--manifest", s["LME_MANIFEST_PATH"],
          *phase_start_settings(s)], env=base)
    log(f"graph provenance recorded: {provenance} (graph_fresh={s['GRAPH_FRESH']})")

    # ---- untracked-file preflight ----
    # Untracked source can shadow committed modules, so the commit hash would not name the code
    # that ran. Source extensions only: a stray results JSON cannot shadow anything.
    if menhir_facts[3] or bench_facts[3]:
        log("WARNING: untracked source files detected:")
        for path in menhir_facts[3]:
            print(f"  menhir: {path}", file=sys.stderr)
        for path in bench_facts[3]:
            print(f"  bench:  {path}", file=sys.stderr)
        if s.get("LME_NONCANONICAL") != "1":
            raise Die("canonical build refused: untracked source files may shadow committed code. "
                      "Commit them, remove them, or set LME_NONCANONICAL=1.")
        log("continuing (LME_NONCANONICAL=1)")

    # ---- ingest Menhir (stopped on exit; Neo4j persists) ----
    with tempfile.NamedTemporaryFile(prefix="lme-empty-env-", delete=False) as handle:
        s["EMPTY_ENV"] = handle.name
    s["RESULTS_DIR_NAME"] = results.name
    serve: subprocess.Popen | None = None
    try:
        env = menhir_env(s, base)
        Path(env["MENHIR_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
        log(f"telemetry DB: {env['MENHIR_MCP_TELEMETRY_DB']}")
        # A Menhir already on the port (a killed attempt's serve?) would answer the health check
        # while ours fails to bind, and ingest would drive the old writer.
        if http_ok(f"{s['MENHIR_URL']}/api/health"):
            raise Die(f"something already serves {s['MENHIR_URL']}; stop it before building (an earlier attempt's menhir?)")
        problem = serving_tree_problem(s)
        if problem:
            raise Die(problem)
        # Menhir ignores a retired flag, so an inherited export of one would silently run it on.
        if subprocess.run([py, "-I", (LIB / "retired_flags.py").as_posix(), "check", s["MENHIR_FRONTIER"]],
                          env=env).returncode != 0:
            raise Die(f"the environment sets flags {s['MENHIR_FRONTIER']} has retired; see above")
        log(f"starting ingest menhir on {s['MENHIR_URL']}...")
        serve = subprocess.Popen([s["MENHIR_FRONTIER_BIN"], "serve", "--port", s["MENHIR_PORT"]],
                                 cwd=s["MENHIR_FRONTIER"], env=env)
        if not wait_http(f"{s['MENHIR_URL']}/api/health", 90, alive=lambda: serve.poll() is None):
            raise Die("menhir not healthy")
        if serve.poll() is not None:
            raise Die(f"our menhir exited, yet {s['MENHIR_URL']} answers; refusing")
        log(f"menhir healthy. ingesting {s['LIMIT']} items...")
        _run([py, (LIB / "ingest.py").as_posix(), *ingest_args(s)], env={**env, "MENHIR_FRONTIER": s["MENHIR_FRONTIER"]})
        post_ingest(s, env, provenance)
    finally:
        if serve is not None:
            log(f"stopping ingest menhir (Neo4j {name} stays up)...")
            stop(serve)
        Path(s["EMPTY_ENV"]).unlink(missing_ok=True)
    log(f"build complete. Neo4j {name} (bolt {s['LME_BOLT']}) holds the data; manifest at {s['LME_MANIFEST_PATH']}.")
    return 0


def post_ingest(s: dict[str, str], env: dict[str, str], provenance: str) -> None:
    py = s["BENCH_PY"]
    # Benchmark mode runs no scheduler, so nothing repairs a READY episode whose content embedding
    # the worker failed to write. The serving tree's script embeds the gaps and fails if any remain;
    # PYTHONPATH pins that tree's src (a shared venv's editable install may point elsewhere).
    if s["LME_EMBED_BACKFILL"] == "1":
        log("backfilling missing episode content embeddings (--require-complete)...")
        _run([s["MENHIR_FRONTIER_PY"], f"{s['MENHIR_FRONTIER']}/scripts/backfill_episode_embeddings.py",
              "--require-complete"], env={**env, "PYTHONPATH": f"{s['MENHIR_FRONTIER']}/src"})
    else:
        log("embedding backfill SKIPPED (LME_EMBED_BACKFILL=0): unembedded READY episodes are not checked.")

    if s["LME_SCALAR_STATE_ENABLED"] == "1":
        rows = json.loads(Path(s["LME_MANIFEST_PATH"]).read_text(encoding="utf-8"))
        problem = scalar_problem(rows, expected=int(s["INGEST_TARGET"]),
                                 require_output=s["LME_REQUIRE_SCALAR_OUTPUT"] not in ("", "0"),
                                 history=s["LME_SCALAR_HISTORY_ENABLED"] not in ("", "0"))
        if problem:
            raise Die(problem)
        print("scalar ingest validation: PASS", flush=True)

    # Freshly extracted nodes are SESSION-scoped and invisible to build_context / plain recall
    # (promote_persistent.sh has the rationale); promote this manifest's namespaces while the
    # ingest Menhir still serves.
    log("ingest complete. promoting SESSION -> PERSISTENT (regular memories)...")
    _run([py, (LIB / "promote.py").as_posix(), s["LME_MANIFEST_PATH"], s["MENHIR_URL"]], env=env)

    # Off by default: dates come from the ingest since Menhir 27d9bad (see config.sh LME_BACKFILL_DATES).
    if s["LME_BACKFILL_DATES"] == "1":
        log("repairing temporal grounding (backfill-dates)... [LME_BACKFILL_DATES=1]")
        revert = expand("${LME_REVERT_SNAPSHOT:-${LME_REVERT_SNAPSHOT_PATH}}", s)
        _run([s["MENHIR_MAIN_PY"], (LIB / "backfill_dates.py").as_posix()],
             env={**env, "LME_BOLT": s["LME_BOLT"], "LME_NEO4J_PW": s["LME_NEO4J_PW"],
                  "LME_NS_PREFIX": s["LME_NS_PREFIX"], "LME_REVERT_SNAPSHOT": revert})
    else:
        log("backfill-dates SKIPPED (LME_BACKFILL_DATES=0): valid_at comes from the ingest.")
        log("  verify with menhir scripts/_verify_valid_at_repair.py before trusting this graph.")

    # Closed only on the success path: an earlier failure leaves the phase `started`, which the
    # next attempt's `begin` marks `interrupted`.
    _run([py, (LIB / "run_provenance.py").as_posix(), "phase-end", provenance, "--phase", "ingest-graph",
          "--status", "completed", "--setting", f"backfill_dates={s['LME_BACKFILL_DATES']}"], env=env)

    # Ingest cost from Menhir's provider-reported usage telemetry (the expensive half of a run).
    usage_tool, db = LIB / "summarize_llm_usage.py", Path(env["MENHIR_MCP_TELEMETRY_DB"])
    if usage_tool.is_file() and db.is_file():
        out = Path(s["LME_RESULTS_DIR"]) / "run_llm_usage.json"
        rc = subprocess.run([py, usage_tool.as_posix(), db.as_posix(), "--run-id", env["MENHIR_BENCH_ACTIVE_RUN_ID"],
                             "--output", out.as_posix()], env=env, stdout=subprocess.DEVNULL).returncode
        log(f"ingest LLM usage: {out.as_posix()}" if rc == 0
            else "WARNING: ingest LLM usage summary failed; cost for this build is unrecorded")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: <settings on stdin> pipeline.py <limit>", file=sys.stderr)
        return 2
    base = dict(os.environ)
    try:
        settings = read_settings(sys.stdin.buffer, base)
        return run(settings, argv[0], base)
    except Die as exc:
        print(f"[lme-build] ERROR: {exc.message}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
