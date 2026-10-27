#!/usr/bin/env bash
# Re-enrich FAILED Episodic nodes in the existing LME Neo4j DB.
# Starts menhir-frontier on LME_PORT_BUILD, resets all FAILED episodes, drains, stops menhir.
# The Neo4j container must already be running.
# The Neo4j is NOT stopped on exit — it holds the data for the subsequent A/B.
#
# The retry must re-enrich with the SAME ingest behavior the graph was built with, or one graph
# ends up holding two configurations. It exports the same settings as build_graph.sh
# (tests/test_lme_retry_parity.py) and refuses to start unless they match the graph's recorded
# provenance (run_provenance.py check-retry).
#
# Usage:
#   retry.sh [--drain-timeout <seconds>]  # default 86400 (24h)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

MENHIR_PORT="${LME_PORT_BUILD}"
MENHIR_URL="http://localhost:${MENHIR_PORT}"

DRAIN_TIMEOUT=86400
for ((i=1; i<=$#; i++)); do
  case "${!i}" in
    --drain-timeout) n=$((i+1)); DRAIN_TIMEOUT="${!n}"; i=$n ;;
  esac
done

log() { printf '[retry] %s\n' "$*" >&2; }
die() { printf '[retry] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- preflight --------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker not on PATH"
lme_container_running "${LME_NEO4J_NAME}" \
  || die "${LME_NEO4J_NAME} is not running — start it with: docker start ${LME_NEO4J_NAME}"
[ -f "${MENHIR_FRONTIER_PY}" ] || die "menhir-frontier venv python missing: ${MENHIR_FRONTIER_PY}"
[ -f "${MENHIR_FRONTIER_BIN}" ] || [ -f "${MENHIR_FRONTIER_BIN}.exe" ] \
  || die "menhir console script missing — pip install -e ${MENHIR_FRONTIER}"
[ -f "${BENCH_PY}" ] || die "archolith-bench venv python missing: ${BENCH_PY}"
# Providers below are OpenAI only; a graph built against a local endpoint is not retryable here.
[ "${LME_EXTRACT_PROVIDER}" = "openai" ] || die "retry.sh supports LME_EXTRACT_PROVIDER=openai only"
case "${LME_ASSISTANT_TURN_POLICY}" in 0|1) ;; *) die "LME_ASSISTANT_TURN_POLICY must be 0 or 1" ;; esac

OPENAI_KEY="$("${MENHIR_FRONTIER_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"; [ -n "${OPENAI_KEY}" ] || die "OPENAI_API_KEY not found in ${BENCH_DIR}/.env"
log "preflight OK. keys resolved."

# ---- provenance: refuse a retry that would mix configurations ----------------
GRAPH_PROVENANCE_PATH="${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json"
GRAPH_PROVENANCE_TOOL="$(dirname "${BASH_SOURCE[0]}")/lib/run_provenance.py"
RETRY_RECORD="$(mktemp)"
RETRY_SURFACE_DIGEST="$("${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/bench_surface.py" fingerprint --digest-only 2>/dev/null || true)"
cat > "${RETRY_RECORD}" <<EOF
{
  "container": "${LME_NEO4J_NAME}",
  "counter_state_enabled": ${LME_COUNTER_STATE_ENABLED},
  "event_history_enabled": ${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED},
  "scalar_state_enabled": ${LME_SCALAR_STATE_ENABLED},
  "canonical_self_binding_mode": "${LME_CANONICAL_SELF_BINDING_MODE}",
  "extract_model": "${LME_EXTRACT_MODEL}",
  "anchored_time_resolver_enabled": ${LME_ANCHORED_TIME_RESOLVER},
  "anchored_time_attach_enabled": ${LME_ANCHORED_TIME_ATTACH},
  "anchored_time_expiry_enabled": ${LME_ANCHORED_TIME_EXPIRY},
  "anchored_time_model": "${LME_ANCHORED_TIME_MODEL}",
  "anchored_time_attach_model": "${LME_ANCHORED_TIME_ATTACH_MODEL}",
  "assistant_turns_store_only_enabled": ${LME_ASSISTANT_TURN_POLICY},
  "menhir_commit": "$(git -C "${MENHIR_MAIN}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "bench_commit": "$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "surface_digest": "${RETRY_SURFACE_DIGEST}"
}
EOF
RETRY_CHECK_ARGS=("check-retry" "${GRAPH_PROVENANCE_PATH}" "${RETRY_RECORD}")
[ "${LME_NONCANONICAL}" = "1" ] && RETRY_CHECK_ARGS+=("--noncanonical")
"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" "${RETRY_CHECK_ARGS[@]}" ||
  die "graph provenance refused this retry (settings, commit, or surface mismatch); see above"
rm -f "${RETRY_RECORD}"

# ---- menhir-frontier lifecycle (stops on exit; Neo4j stays up) --------------
MENHIR_PID=""
cleanup() {
  log "stopping menhir-frontier (Neo4j ${LME_NEO4J_NAME} stays up)..."
  [ -n "${MENHIR_PID}" ] && kill "${MENHIR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Blank env so menhir's default .env (local-llama / prod Neo4j) is NOT loaded.
EMPTY_ENV="$(mktemp)"; export ENV_FILE="${EMPTY_ENV}"

# Dedicated log dir (avoids WinError 32 RotatingFileHandler collisions).
export MENHIR_LOG_DIR="${LME_RESULTS_DIR}/menhir-logs"; mkdir -p "${MENHIR_LOG_DIR}"

export MENHIR_BENCHMARK_MODE=1
export MENHIR_API_HOST=127.0.0.1
export MENHIR_API_PORT="${MENHIR_PORT}"
# Parallelize enrichment: N episodes extract at once, serialized per namespace (cloud
# provider, so the single-flight local-model lock does not apply). Override by exporting
# MENHIR_INGEST_CONCURRENCY before calling.
export MENHIR_INGEST_CONCURRENCY="${MENHIR_INGEST_CONCURRENCY:-8}"

# ---- ingest behavior: identical to build_graph.sh ----------------------------
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED="${LME_COUNTER_STATE_ENABLED}"
export MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED
export MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED="${LME_SCALAR_STATE_ENABLED}"
export MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED}"
export MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED=0
export MENHIR_PERSONAL_MEMORY_SCALAR_THRESHOLD="${LME_SCALAR_THRESHOLD}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_ATTRIBUTE="${LME_SCALAR_RECONCILE_ATTRIBUTE}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SCOPE="${LME_SCALAR_RECONCILE_SCOPE}"
export MENHIR_PERSONAL_MEMORY_SCALAR_RECONCILE_SUBJECT="${LME_SCALAR_RECONCILE_SUBJECT}"
export MENHIR_PERSONAL_MEMORY_SCALAR_CANONICAL_SELF="${LME_SCALAR_CANONICAL_SELF}"
export MENHIR_CANONICAL_SELF_BINDING_MODE="${LME_CANONICAL_SELF_BINDING_MODE}"
export MENHIR_PERSONAL_MEMORY_CHAT_MODEL="${LME_EXTRACT_MODEL}"
export MENHIR_PERSONAL_MEMORY_SUM_GROUNDING=1
export MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED="${LME_ASSISTANT_TURN_POLICY}"
export LME_REQUIRE_TURN_EVIDENCE="${LME_REQUIRE_TURN_EVIDENCE}"
export MENHIR_MAX_LLM_CALLS_PER_JOB="${LME_MAX_LLM_CALLS_PER_JOB}"
export MENHIR_ANCHORED_TIME_RESOLVER="${LME_ANCHORED_TIME_RESOLVER}"
export MENHIR_ANCHORED_TIME_ATTACH="${LME_ANCHORED_TIME_ATTACH}"
export MENHIR_ANCHORED_TIME_EXPIRY="${LME_ANCHORED_TIME_EXPIRY}"
export MENHIR_ANCHORED_TIME_MODEL="${LME_ANCHORED_TIME_MODEL}"
export MENHIR_ANCHORED_TIME_ATTACH_MODEL="${LME_ANCHORED_TIME_ATTACH_MODEL}"
export MENHIR_OPENAI_SERVICE_TIER="${LME_OPENAI_SERVICE_TIER}"
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED="${LME_CONSOLIDATION_AUDIT_ENABLED}"
export MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED="${LME_RECALL_AUDIT_ENABLED}"
export MENHIR_LOG_LEVEL="${MENHIR_LOG_LEVEL:-${LME_MENHIR_LOG_LEVEL}}"
# Same telemetry DB as the build, so retry calls count toward the graph's recorded cost.
export MENHIR_MCP_TELEMETRY_DB="${LME_RESULTS_DIR}/mcp_telemetry.db"

export OTEL_SDK_DISABLED=true
export LANGFUSE_TRACING_ENABLED=false
export LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST=""
export MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY=""
# Existing persistent Neo4j
export NEO4J_URI="bolt://localhost:${LME_BOLT}"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="${LME_NEO4J_PW}"
export NEO4J_DATABASE="neo4j"
export LME_NEO4J_CONTAINER="${LME_NEO4J_NAME}" LME_NEO4J_PW="${LME_NEO4J_PW}"
# OpenAI extraction + embedding (matches original build config)
export GRAPHITI_LLM_PROVIDER="openai"   MEMORY_GRAPHITI_PROVIDER="openai"
export LLM_CHAT_PROVIDER="openai"        MEMORY_CHAT_PROVIDER="openai"
export GRAPHITI_RERANKER_PROVIDER="openai" MEMORY_GRAPHITI_RERANKER_PROVIDER="openai"
export GRAPHITI_EMBED_PROVIDER="openai"  MEMORY_GRAPHITI_EMBED_PROVIDER="openai"
export OPENAI_API_KEY="${OPENAI_KEY}"
export OPENAI_CHAT_MODEL="${LME_EXTRACT_MODEL}"
export OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}"

"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" phase-start "${GRAPH_PROVENANCE_PATH}" \
  --phase retry-failed \
  --setting "drain_timeout=${DRAIN_TIMEOUT}" \
  --setting "ingest_concurrency=${MENHIR_INGEST_CONCURRENCY}" \
  --setting "max_llm_calls_per_job=${LME_MAX_LLM_CALLS_PER_JOB}" \
  --setting "openai_service_tier=${LME_OPENAI_SERVICE_TIER}" \
  --setting "noncanonical=${LME_NONCANONICAL}"

log "starting menhir-frontier on ${MENHIR_URL} (pointing at ${LME_NEO4J_NAME})..."
( cd "${MENHIR_FRONTIER}" && "${MENHIR_FRONTIER_BIN}" serve --port "${MENHIR_PORT}" ) & MENHIR_PID=$!

log "waiting for menhir-frontier /api/health..."
for _ in $(seq 1 90); do
  curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 || die "menhir-frontier not healthy on ${MENHIR_PORT}"
log "menhir-frontier healthy. starting retry pass..."

# ---- retry ------------------------------------------------------------------
"${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/retry.py" \
  --menhir-url "${MENHIR_URL}" \
  --drain-timeout "${DRAIN_TIMEOUT}"

# Success path only: a failed retry leaves the phase `started`, which the next begin marks interrupted.
"${BENCH_PY}" "${GRAPH_PROVENANCE_TOOL}" phase-end "${GRAPH_PROVENANCE_PATH}" \
  --phase retry-failed --status completed
log "retry pass complete."
