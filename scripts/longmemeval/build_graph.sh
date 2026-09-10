#!/usr/bin/env bash
# Build a PERSISTENT menhir/Neo4j with LongMemEval haystacks pre-ingested under
# stable namespaces, so recall-only A/B (main vs frontier) can run without re-ingesting.
# Neo4j stays up after; only the temp menhir (used for ingestion) is stopped.
# Usage: build_graph.sh [limit]
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

LIMIT="${1:-30}"
MENHIR_PORT="${LME_PORT_BUILD}"
MENHIR_URL="http://localhost:${MENHIR_PORT}"

log(){ printf '[lme-build] %s\n' "$*" >&2; }
die(){ printf '[lme-build] ERROR: %s\n' "$*" >&2; exit 1; }
case "${LME_INGEST_CONCURRENCY}" in
  ''|*[!0-9]*|0) die "LME_INGEST_CONCURRENCY must be a positive integer" ;;
esac
LME_INGEST_STOP_AFTER_ITEMS="${LME_INGEST_STOP_AFTER_ITEMS:-0}"
case "${LME_INGEST_STOP_AFTER_ITEMS}" in
  *[!0-9]*) die "LME_INGEST_STOP_AFTER_ITEMS must be a non-negative integer" ;;
esac
[ "${LME_INGEST_STOP_AFTER_ITEMS}" -le "${LIMIT}" ] ||
  die "LME_INGEST_STOP_AFTER_ITEMS cannot exceed the requested item limit"
INGEST_TARGET="${LIMIT}"
if [ "${LME_INGEST_STOP_AFTER_ITEMS}" -gt 0 ]; then
  INGEST_TARGET="${LME_INGEST_STOP_AFTER_ITEMS}"
fi

OPENAI_KEY="$("${MENHIR_FRONTIER_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"; [ -n "${OPENAI_KEY}" ] || die "no OPENAI_API_KEY in ${BENCH_DIR}/.env"

# ---- persistent Neo4j (reuse if already up) ----
# Freshness provenance: "graph_fresh" must mean the DATA is new, not just that the
# container object is new -- `docker run -v <name>:/data` silently reattaches an
# existing named volume even for a brand-new container. Check the volume BEFORE the
# docker run call (which would auto-create it and make the check always say "existed").
VOLUME_PRE_EXISTED="false"
docker volume inspect "${LME_NEO4J_VOL}" >/dev/null 2>&1 && VOLUME_PRE_EXISTED="true"

CONTAINER_PRE_EXISTED="false"
lme_container_exists "${LME_NEO4J_NAME}" && CONTAINER_PRE_EXISTED="true"
if [ "${LME_REQUIRE_FRESH}" = "1" ]; then
  [ "${VOLUME_PRE_EXISTED}" = "false" ] || die "fresh build refused: volume already exists: ${LME_NEO4J_VOL}"
  [ "${CONTAINER_PRE_EXISTED}" = "false" ] || die "fresh build refused: container already exists: ${LME_NEO4J_NAME}"
  [ ! -e "${LME_MANIFEST_PATH}" ] || die "fresh build refused: manifest already exists: ${LME_MANIFEST_PATH}"
  # Checked here so the error names the stale file. Without it the run proceeds to `begin`,
  # which refuses on a surface_digest or identity mismatch against an attempt whose graph is
  # already gone -- a confusing failure for a genuinely fresh build. Not deleted automatically:
  # provenance is an audit record, and removing one is the caller's explicit decision (--clean).
  [ ! -e "${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json" ] ||
    die "fresh build refused: provenance from an earlier attempt already exists: ${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json (remove it, or re-run with --clean)"
fi

GRAPH_FRESH="false"
if ! lme_container_running "${LME_NEO4J_NAME}"; then
  if lme_container_exists "${LME_NEO4J_NAME}"; then
    log "starting existing ${LME_NEO4J_NAME}..."; docker start "${LME_NEO4J_NAME}" >/dev/null
  else
    log "creating persistent Neo4j ${LME_NEO4J_NAME} (bolt ${LME_BOLT})..."
    docker run -d --name "${LME_NEO4J_NAME}" -p ${LME_BOLT}:7687 -p ${LME_HTTP}:7474 \
      -e NEO4J_AUTH=neo4j/${LME_NEO4J_PW} -e NEO4J_server_memory_heap_max__size=${LME_NEO4J_HEAP} \
      -v ${LME_NEO4J_VOL}:/data ${LME_NEO4J_IMAGE} >/dev/null
    [ "${VOLUME_PRE_EXISTED}" = "false" ] && GRAPH_FRESH="true"
  fi
fi
log "waiting for Neo4j HTTP (${LME_HTTP})..."
for _ in $(seq 1 60); do curl -sf "http://localhost:${LME_HTTP}" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://localhost:${LME_HTTP}" >/dev/null 2>&1 || die "Neo4j not ready"

# Record freshness provenance so `lme.sh ir-gate` doesn't have to trust a manually-set
# LME_GRAPH_FRESH env var -- it reads this file for the container currently configured.
mkdir -p "${LME_RESULTS_DIR}"
GRAPH_PROVENANCE_PATH="${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json"
GRAPH_PROVENANCE_TOOL="$(dirname "${BASH_SOURCE[0]}")/lib/run_provenance.py"
[ -f "${GRAPH_PROVENANCE_TOOL}" ] || die "provenance helper missing: ${GRAPH_PROVENANCE_TOOL}"
GRAPH_ATTEMPT_RECORD="${LME_RESULTS_DIR}/.graph-attempt-${LME_NEO4J_NAME}.json"

# Each build attempt APPENDS. A resume re-runs this script against a graph an earlier attempt
# already partly ingested; overwriting the record here would restate that graph's freshness,
# dataset and settings as whatever this attempt happens to be configured with -- and
# `lme.sh ir-gate` reads exactly this file to decide whether the data is fresh.
#
# Fingerprint the declared surface BEFORE writing the attempt record, so `begin` can treat the
# digest as immutable identity for a canonical run and refuse a resume that would mix two code
# states. Empty on failure, which `run_provenance.py` reads as "absent" rather than as a digest
# of "" -- fingerprinting stays non-fatal, consistent with the attach call further down.
GRAPH_SURFACE_DIGEST="$("${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/bench_surface.py" fingerprint --digest-only 2>/dev/null || true)"
cat > "${GRAPH_ATTEMPT_RECORD}" <<EOF
{
  "container": "${LME_NEO4J_NAME}",
  "volume": "${LME_NEO4J_VOL}",
  "volume_pre_existed": ${VOLUME_PRE_EXISTED},
  "graph_fresh": ${GRAPH_FRESH},
  "require_fresh": ${LME_REQUIRE_FRESH},
  "dataset": "${LME_DATASET}",
  "variant": "${LONGMEMEVAL_VARIANT}",
  "fixture_sha256": "${LME_FIXTURE_SHA256:-}",
  "fixture_count": ${LME_FIXTURE_COUNT:-0},
  "fixture": "${LME_FIXTURE_PATH:-}",
  "requested_items": ${LIMIT},
  "ingest_target_items": ${INGEST_TARGET},
  "namespace_prefix": "${LME_NS_PREFIX}",
  "segmentation": "${LME_SEGMENTATION}",
  "ingest_concurrency": ${LME_INGEST_CONCURRENCY},
  "scalar_state_enabled": ${LME_SCALAR_STATE_ENABLED},
  "scalar_history_enabled": ${LME_SCALAR_HISTORY_ENABLED},
  "scalar_consolidation_k": ${LME_SCALAR_CONSOLIDATION_K},
  "scalar_threshold": "${LME_SCALAR_THRESHOLD}",
  "scalar_reconcile_attribute": ${LME_SCALAR_RECONCILE_ATTRIBUTE},
  "scalar_reconcile_scope": ${LME_SCALAR_RECONCILE_SCOPE},
  "scalar_reconcile_subject": ${LME_SCALAR_RECONCILE_SUBJECT},
  "scalar_canonical_self": ${LME_SCALAR_CANONICAL_SELF},
  "scalar_output_required": ${LME_REQUIRE_SCALAR_OUTPUT},
  "turn_evidence_required": ${LME_REQUIRE_TURN_EVIDENCE},
  "consolidation_audit_enabled": ${LME_CONSOLIDATION_AUDIT_ENABLED},
  "recall_audit_enabled": ${LME_RECALL_AUDIT_ENABLED},
  "menhir_log_level": "${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}",
  "menhir_commit": "$(git -C "${MENHIR_MAIN}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "bench_commit": "$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "surface_digest": "${GRAPH_SURFACE_DIGEST}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "build_started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
PROVENANCE_BEGIN_ARGS=("begin" "${GRAPH_PROVENANCE_PATH}" "${GRAPH_ATTEMPT_RECORD}")
[ "${LME_NONCANONICAL}" = "1" ] && PROVENANCE_BEGIN_ARGS+=("--noncanonical")
"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" "${PROVENANCE_BEGIN_ARGS[@]}" ||
  die "graph provenance refused this attempt (identity, commit, or surface mismatch); see above"
rm -f "${GRAPH_ATTEMPT_RECORD}"

# Hash the declared benchmark-affecting surface into this graph's provenance. The
# `menhir_untracked`/`bench_untracked` counts recorded below say how many untracked files
# existed; they cannot say what was in them, and a dirty tree leaves the commit SHA
# unchanged while the executed code differs. Every run_ku_*.sh wrapper reaches a graph
# through this script, so attaching here is what makes those runs attributable at all.
#
# Non-fatal, for the same reason as in the buildout wrapper: a fingerprint is evidence about
# a run, not a precondition for it, and refusing to launch would trade a recorded gap for no
# run. `bench_surface.py blame` reports a run with no fingerprint as not-comparable rather
# than guessing.
GRAPH_SURFACE_TOOL="$(dirname "${BASH_SOURCE[0]}")/lib/bench_surface.py"
if [ -f "${GRAPH_SURFACE_TOOL}" ]; then
  "${BENCH_PY}" "${GRAPH_SURFACE_TOOL}" attach "${GRAPH_PROVENANCE_PATH}" ||
    log "WARNING: surface fingerprint failed; this graph will not be attributable by file"
else
  log "WARNING: ${GRAPH_SURFACE_TOOL} missing; no surface fingerprint recorded"
fi

# A phase is only reproducible if it names the code that ran it and the data it ran on.
# build_graph.sh can be invoked directly, so the fixture hash/count are recorded when the
# caller exported them and omitted otherwise rather than guessed.
GRAPH_FIXTURE_SETTINGS=()
if [ -n "${LME_FIXTURE_PATH:-}" ]; then
  GRAPH_FIXTURE_SETTINGS+=(--setting "fixture=${LME_FIXTURE_PATH}")
fi
if [ -n "${LME_FIXTURE_SHA256:-}" ]; then
  GRAPH_FIXTURE_SETTINGS+=(--setting "fixture_sha256=${LME_FIXTURE_SHA256}")
fi
if [ -n "${LME_FIXTURE_COUNT:-}" ]; then
  GRAPH_FIXTURE_SETTINGS+=(--setting "fixture_count=${LME_FIXTURE_COUNT}")
fi

"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" phase-start "${GRAPH_PROVENANCE_PATH}" \
  --phase ingest-graph --manifest "${LME_MANIFEST_PATH}" \
  "${GRAPH_FIXTURE_SETTINGS[@]}" \
  --setting "menhir_commit=$(git -C "${MENHIR_MAIN}" rev-parse HEAD 2>/dev/null || echo unknown)" \
  --setting "bench_commit=$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)" \
  --setting "menhir_dirty=$([ -n "$(git -C "${MENHIR_MAIN}" status --porcelain 2>/dev/null)" ] && echo true || echo false)" \
  --setting "bench_dirty=$([ -n "$(git -C "${BENCH_DIR}" status --porcelain 2>/dev/null)" ] && echo true || echo false)" \
  --setting "menhir_untracked=$(git -C "${MENHIR_MAIN}" ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')" \
  --setting "bench_untracked=$(git -C "${BENCH_DIR}" ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ')" \
  --setting "dataset=${LME_DATASET}" \
  --setting "variant=${LONGMEMEVAL_VARIANT}" \
  --setting "namespace_prefix=${LME_NS_PREFIX}" \
  --setting "graph_fresh=${GRAPH_FRESH}" \
  --setting "volume_pre_existed=${VOLUME_PRE_EXISTED}" \
  --setting "require_fresh=${LME_REQUIRE_FRESH}" \
  --setting "segmentation=${LME_SEGMENTATION}" \
  --setting "strict_failure_policy=zero-failed-episodes-per-namespace" \
  --setting "ingest_target_items=${INGEST_TARGET}" \
  --setting "ingest_concurrency=${LME_INGEST_CONCURRENCY}" \
  --setting "scalar_state_enabled=${LME_SCALAR_STATE_ENABLED}" \
  --setting "scalar_history_enabled=${LME_SCALAR_HISTORY_ENABLED}" \
  --setting "scalar_consolidation_k=${LME_SCALAR_CONSOLIDATION_K}" \
  --setting "scalar_consolidation_call_budget=${LME_SCALAR_CALL_BUDGET}" \
  --setting "scalar_threshold=${LME_SCALAR_THRESHOLD}" \
  --setting "scalar_reconcile_attribute=${LME_SCALAR_RECONCILE_ATTRIBUTE}" \
  --setting "scalar_reconcile_scope=${LME_SCALAR_RECONCILE_SCOPE}" \
  --setting "scalar_reconcile_subject=${LME_SCALAR_RECONCILE_SUBJECT}" \
  --setting "consolidation_audit_enabled=${LME_CONSOLIDATION_AUDIT_ENABLED}" \
  --setting "recall_audit_enabled=${LME_RECALL_AUDIT_ENABLED}" \
  --setting "menhir_log_level=${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}" \
  --setting "turn_evidence_required=${LME_REQUIRE_TURN_EVIDENCE}"
log "graph provenance recorded: ${GRAPH_PROVENANCE_PATH} (graph_fresh=${GRAPH_FRESH})"

# ---- untracked-file preflight ----
# Tracked-only dirty checks prove committed source is clean, but untracked .py files in
# src/ or scripts/ can shadow committed modules and make the executed code differ from what
# the commit hash claims. Warn loudly; refuse in canonical mode.
#
# Scoped to SOURCE extensions. The path globs alone matched every untracked file under those
# directories, including benchmark outputs that land in scripts/longmemeval/results/ -- so a
# stale JSON from a July gate run refused an unrelated canonical build. Only files Python
# imports or bash sources can shadow committed code; a data file cannot, and refusing on one
# trains people to reach for LME_NONCANONICAL=1, which disables the check that matters.
MENHIR_UNTRACKED="$(git -C "${MENHIR_MAIN}" ls-files --others --exclude-standard   -- 'src/**/*.py' 'scripts/**/*.py' 'src/**/*.sh' 'scripts/**/*.sh' 2>/dev/null || true)"
BENCH_UNTRACKED="$(git -C "${BENCH_DIR}" ls-files --others --exclude-standard   -- 'scripts/**/*.py' 'archolith_bench/**/*.py' 'scripts/**/*.sh' 2>/dev/null || true)"
if [ -n "${MENHIR_UNTRACKED}" ] || [ -n "${BENCH_UNTRACKED}" ]; then
  log "WARNING: untracked source files detected:"
  [ -n "${MENHIR_UNTRACKED}" ] && printf '  menhir: %s\n' ${MENHIR_UNTRACKED} >&2
  [ -n "${BENCH_UNTRACKED}" ] && printf '  bench:  %s\n' ${BENCH_UNTRACKED} >&2
  if [ "${LME_NONCANONICAL}" != "1" ]; then
    die "canonical build refused: untracked source files may shadow committed code. Commit them, remove them, or set LME_NONCANONICAL=1."
  fi
  log "continuing (LME_NONCANONICAL=1)"
fi

# ---- temp menhir for ingestion (stopped on exit; Neo4j persists) ----
MENHIR_PID=""
cleanup(){ log "stopping ingest menhir (Neo4j ${LME_NEO4J_NAME} stays up)..."; [ -n "${MENHIR_PID}" ] && kill "${MENHIR_PID}" 2>/dev/null || true; }
trap cleanup EXIT

EMPTY_ENV="$(mktemp)"; export ENV_FILE="${EMPTY_ENV}"
# LongMemEval variant: default to 'oracle' (evidence-only haystacks, ~22 turns/item vs ~494
# for 's') so the full 500-item build is ~1 day instead of ~weeks. Override by exporting
# LONGMEMEVAL_VARIANT=s|m before calling.
export LONGMEMEVAL_VARIANT="${LONGMEMEVAL_VARIANT:-oracle}"
export LME_NEO4J_CONTAINER="${LME_NEO4J_NAME}" LME_NEO4J_PW="${LME_NEO4J_PW}"
# Dedicated log dir so the ingest menhir's RotatingFileHandler never shares server.log with
# another menhir process (the cross-process WinError 32 on rollover is the log-noise source).
export MENHIR_LOG_DIR="${LME_RESULTS_DIR}/menhir-logs"; mkdir -p "${MENHIR_LOG_DIR}"
export MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="${MENHIR_PORT}"
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED=0
export MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED="${LME_SCALAR_STATE_ENABLED}"
export MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED}"
export MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED=0
# Typed-scalar gate relaxations. These are default-OFF inside menhir, so without these exports a
# scalar build silently reproduces the unanimous-threshold baseline (20/100 correct current Views)
# no matter what else changed. See config.sh for the measured grid and the 0.67-vs-2/3 trap.
export MENHIR_PERSONAL_MEMORY_SCALAR_THRESHOLD="${LME_SCALAR_THRESHOLD}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_ATTRIBUTE="${LME_SCALAR_RECONCILE_ATTRIBUTE}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SCOPE="${LME_SCALAR_RECONCILE_SCOPE}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SUBJECT="${LME_SCALAR_RECONCILE_SUBJECT}"
export MENHIR_PERSONAL_MEMORY_SCALAR_CANONICAL_SELF="${LME_SCALAR_CANONICAL_SELF}"
export MENHIR_PERSONAL_MEMORY_CHAT_MODEL="${LME_EXTRACT_MODEL}"
export MENHIR_PERSONAL_MEMORY_SUM_GROUNDING=1
export LME_REQUIRE_TURN_EVIDENCE="${LME_REQUIRE_TURN_EVIDENCE}"
# Raise the per-episode LLM extraction budget (default 10) so long turns finish enrichment
# instead of hitting FAILED; the ingest script does a best-effort FAILED-retry for the rest.
export MENHIR_MAX_LLM_CALLS_PER_JOB="${LME_MAX_LLM_CALLS_PER_JOB:-20}"
# Menhir allows this many distinct namespaces to enrich concurrently. ingest.py uses the same
# value for its namespace window and keeps only one active episode in each namespace.
export MENHIR_INGEST_CONCURRENCY="${LME_INGEST_CONCURRENCY}"
# Per-run telemetry: each build gets its own SQLite sidecar so vote receipts, lifecycle
# events, and scalar audit trails are preserved with the results and attributable to this
# exact run. The dashboard's ScalarTaskReader reads from this path.
# Decision-point evidence. Without these the telemetry DB holds llm_usage_events but NOT the
# consolidation audit, so a scalar pass that writes nothing leaves no record of WHY -- the drop
# reasons and vote distributions live only here. Defaults are set in config.sh; both audits are
# behavior-neutral.
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED="${LME_CONSOLIDATION_AUDIT_ENABLED}"
export MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED="${LME_RECALL_AUDIT_ENABLED}"
export MENHIR_LOG_LEVEL="${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}"
export MENHIR_MCP_TELEMETRY_DB="${LME_RESULTS_DIR}/mcp_telemetry.db"
log "telemetry DB: ${MENHIR_MCP_TELEMETRY_DB}"
# Bench-run explorer: point at the results root and identify this run so
# the Menhir Recall Lab can display artifact manifests and (when running
# against the same graph) live scalar projections.
export MENHIR_BENCH_RESULTS_ROOT="${BENCH_DIR}/results"
# Derive active run ID from the results directory name (basename of
# LME_RESULTS_DIR). For the persistent LME graph this is "lme-ingest";
# for KU buildouts it is the RUN_ID.
export MENHIR_BENCH_ACTIVE_RUN_ID="$(basename "$LME_RESULTS_DIR")"
export OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST=""
export MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY=""
export NEO4J_URI="bolt://localhost:${LME_BOLT}" NEO4J_USER="neo4j" NEO4J_PASSWORD="${LME_NEO4J_PW}" NEO4J_DATABASE="neo4j"
case "${LME_EXTRACT_PROVIDER}" in
  openai)
    export GRAPHITI_LLM_PROVIDER="openai" MEMORY_GRAPHITI_PROVIDER="openai"
    export LLM_CHAT_PROVIDER="openai" MEMORY_CHAT_PROVIDER="openai"
    export GRAPHITI_RERANKER_PROVIDER="openai" MEMORY_GRAPHITI_RERANKER_PROVIDER="openai"
    ;;
  local)
    [ -n "${LME_EXTRACT_BASE_URL}" ] || die "LME_EXTRACT_BASE_URL is required for local extraction"
    export GRAPHITI_LLM_PROVIDER="local" MEMORY_GRAPHITI_PROVIDER="local"
    export LLM_CHAT_PROVIDER="local" MEMORY_CHAT_PROVIDER="local"
    export GRAPHITI_RERANKER_PROVIDER="local" MEMORY_GRAPHITI_RERANKER_PROVIDER="local"
    export LOCAL_LLM_BASE_URL="${LME_EXTRACT_BASE_URL}"
    export LOCAL_LLM_API_KEY="${LME_EXTRACT_API_KEY}"
    export LOCAL_LLM_CHAT_MODEL="${LME_EXTRACT_MODEL}"
    ;;
  *)
    die "unsupported LME_EXTRACT_PROVIDER: ${LME_EXTRACT_PROVIDER} (expected openai or local)"
    ;;
esac
export GRAPHITI_EMBED_PROVIDER="openai" MEMORY_GRAPHITI_EMBED_PROVIDER="openai"
export OPENAI_API_KEY="${OPENAI_KEY}" OPENAI_CHAT_MODEL="${LME_EXTRACT_MODEL}" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}"

log "starting ingest menhir on ${MENHIR_URL}..."
( cd "${MENHIR_FRONTIER}" && "${MENHIR_FRONTIER_BIN}" serve --port "${MENHIR_PORT}" ) & MENHIR_PID=$!
for _ in $(seq 1 90); do curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 || die "menhir not healthy"
log "menhir healthy. ingesting ${LIMIT} items..."

INGEST_ARGS=(
  --limit "${LIMIT}"
  --menhir-url "${MENHIR_URL}"
  --manifest "${LME_MANIFEST_PATH}"
  --segmentation "${LME_SEGMENTATION}"
  --namespace-window "${LME_INGEST_CONCURRENCY}"
)
if [ "${LME_INGEST_STOP_AFTER_ITEMS}" -gt 0 ]; then
  INGEST_ARGS+=(--manifest-item-limit "${LME_INGEST_STOP_AFTER_ITEMS}")
fi
if [ "${LME_SCALAR_STATE_ENABLED}" = "1" ]; then
  INGEST_ARGS+=(
    --consolidate-scalar
    --consolidation-k "${LME_SCALAR_CONSOLIDATION_K}"
    --consolidation-call-budget "${LME_SCALAR_CALL_BUDGET}"
  )
fi
"${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/ingest.py" "${INGEST_ARGS[@]}"

if [ "${LME_SCALAR_STATE_ENABLED}" = "1" ]; then
  "${BENCH_PY}" - "${LME_MANIFEST_PATH}" "${LME_REQUIRE_SCALAR_OUTPUT}" "${INGEST_TARGET}" \
    "${LME_SCALAR_HISTORY_ENABLED}" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not rows:
    raise SystemExit("scalar ingest validation failed: empty manifest")
expected = int(sys.argv[3])
if len(rows) != expected:
    raise SystemExit(
        f"scalar ingest validation failed: manifest has {len(rows)} rows, expected {expected}"
    )
if any(not row.get("scalar_consolidated") for row in rows):
    raise SystemExit("scalar ingest validation failed: at least one namespace was not consolidated")
if sum(int(row.get("turn_evidence", 0)) for row in rows) <= 0:
    raise SystemExit("scalar ingest validation failed: no TurnEvidence captured")
if sum(int(row.get("scalar_llm_calls", 0)) for row in rows) <= 0:
    raise SystemExit("scalar ingest validation failed: scalar perception made no LLM calls")
require_output = int(sys.argv[2])
if require_output:
    if sum(int(row.get("typed_assertions", 0)) for row in rows) <= 0:
        raise SystemExit("scalar ingest validation failed: no TypedAssertion materialized")
    if sum(int(row.get("scalar_views", 0)) for row in rows) <= 0:
        raise SystemExit("scalar ingest validation failed: no scalar_state View materialized")
    history_enabled = int(sys.argv[4])
    if history_enabled and sum(int(row.get("scalar_history_views", 0)) for row in rows) <= 0:
        raise SystemExit("scalar ingest validation failed: no scalar_history View materialized")
print("scalar ingest validation: PASS", flush=True)
PY
fi
log "ingest complete. promoting SESSION -> PERSISTENT (regular memories)..."

# Write the benchmark as regular memories: freshly-extracted nodes are SESSION-scoped and
# would be invisible to build_context / plain recall (recall_service.py:937). See
# promote_persistent.sh for the full rationale.
"$(dirname "${BASH_SOURCE[0]}")/promote_persistent.sh"

# NO LONGER MANDATORY -- and off by default. The old comment here claimed graphiti-core does not
# apply reference_time to valid_at; that is false (graphiti.py:1110 sets valid_at=reference_time).
# The real defect was menhir dropping reference_time in its claim projection (fixed in 27d9bad), and
# running this repair unconditionally HID that bug: it keys off e.session_id, which graphiti-written
# EpisodicNodes lack, so it skipped 62% of the corpus and still printed success. A build's dates
# should now come from the ingest. See config.sh LME_BACKFILL_DATES.
if [ "${LME_BACKFILL_DATES}" = "1" ]; then
  log "repairing temporal grounding (backfill-dates)... [LME_BACKFILL_DATES=1]"
  LME_BOLT="${LME_BOLT}" LME_NEO4J_PW="${LME_NEO4J_PW}" LME_NS_PREFIX="${LME_NS_PREFIX}" \
    LME_REVERT_SNAPSHOT="${LME_REVERT_SNAPSHOT:-${LME_REVERT_SNAPSHOT_PATH}}" \
    "${MENHIR_MAIN_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/backfill_dates.py"
else
  log "backfill-dates SKIPPED (LME_BACKFILL_DATES=0): valid_at comes from the ingest."
  log "  verify with menhir scripts/_verify_valid_at_repair.py before trusting this graph."
fi

# Close the phase only here, on the success path. `set -e` means any earlier failure exits
# without this line, leaving the phase recorded as `started` -- which the next attempt's
# `begin` marks `interrupted`. An interrupted build is exactly what a reader needs to see.
"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" phase-end "${GRAPH_PROVENANCE_PATH}" \
  --phase ingest-graph --status completed \
  --setting "backfill_dates=${LME_BACKFILL_DATES}"

# Ingest cost, from Menhir's provider-reported usage telemetry. Without this the tokens sit in
# mcp_telemetry.db and nothing surfaces them, which is how a 78-item buildout came to have an
# exactly-known recall cost ($0.40) and an ingest cost nobody could state. Ingest is the
# expensive half, so it is the half that most needs recording.
#
# Ingest-only by design: the harness has not run at this point, and `summarize_llm_usage.py`
# treats `--harness-checkpoint` as optional. A wrapper that later scores the graph overwrites
# this file with the combined menhir+harness summary, which is strictly more information.
USAGE_TOOL="$(dirname "${BASH_SOURCE[0]}")/lib/summarize_llm_usage.py"
if [ -f "${USAGE_TOOL}" ] && [ -f "${MENHIR_MCP_TELEMETRY_DB}" ]; then
  if "${BENCH_PY}" "${USAGE_TOOL}" "${MENHIR_MCP_TELEMETRY_DB}"        --run-id "${MENHIR_BENCH_ACTIVE_RUN_ID}"        --output "${LME_RESULTS_DIR}/run_llm_usage.json" >/dev/null; then
    log "ingest LLM usage: ${LME_RESULTS_DIR}/run_llm_usage.json"
  else
    log "WARNING: ingest LLM usage summary failed; cost for this build is unrecorded"
  fi
fi

log "build complete. Neo4j ${LME_NEO4J_NAME} (bolt ${LME_BOLT}) holds the data; manifest at ${LME_MANIFEST_PATH}."
