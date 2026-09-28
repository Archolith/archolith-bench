#!/usr/bin/env bash
# End-to-end run of the menhir-scalar-state benchmark against a THROWAWAY menhir with the
# typed-scalar consolidation scheduler ON. Brings up a FRESH ephemeral Neo4j (activation is
# fresh-only), starts menhir serve with the scalar flag + background scheduler, runs the
# harness, and tears everything down. Reuses the LongMemEval config.sh conventions.
#
# Usage: run_scalar_state_e2e.sh [--keep] [-- <extra archolith-bench harness args>]
#   --keep   leave the menhir server + Neo4j container up after the run (debugging)
#
# Differs from build_graph.sh in one load-bearing way: MENHIR_BENCHMARK_MODE=0 so the
# background scheduler runs (benchmark mode disables it); typed-scalar perception ONLY runs
# inside the scheduled consolidation job, never over the HTTP phase3 route.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/longmemeval/config.sh"

# ---- scalar-state throwaway knobs (all overridable) ----
SS_NEO4J_NAME="${SS_NEO4J_NAME:-menhir-scalar-neo4j}"
SS_BOLT="${SS_BOLT:-7691}"                 # host bolt port for the throwaway (maps to container 7687)
SS_HTTP="${SS_HTTP:-7477}"                 # host browser/http port (maps to container 7474)
SS_NEO4J_PW="${SS_NEO4J_PW:-scalarthrowaway}"
SS_NEO4J_IMAGE="${SS_NEO4J_IMAGE:-${LME_NEO4J_IMAGE}}"
SS_PORT="${SS_PORT:-8098}"                 # menhir serve port
SS_INTERVAL_S="${SS_INTERVAL_S:-5}"        # consolidation tick interval (short so a tick lands fast)
SS_PERCEIVER_VERSION="${SS_PERCEIVER_VERSION:-v1}"
SS_CHAT_MODEL="${SS_CHAT_MODEL:-${LME_EXTRACT_MODEL}}"   # gpt-4o-mini by default
SS_MAX_WAIT_S="${SS_MAX_WAIT_S:-120}"
SS_FIXTURE="${SS_FIXTURE:-default}"       # default | third-party (Lever 3 binder-isolation)
SS_OUT="${SS_OUT:-${BENCH_DIR}/results/menhir_scalar_state_e2e.md}"
SS_FORMAT="${SS_FORMAT:-markdown}"

KEEP="false"
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --keep) KEEP="true"; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

MENHIR_URL="http://127.0.0.1:${SS_PORT}"
BOLT_URI="bolt://localhost:${SS_BOLT}"

log(){ printf '[scalar-e2e] %s\n' "$*" >&2; }
die(){ printf '[scalar-e2e] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- ensure the bolt read extra is installed (neo4j driver) ----
"${BENCH_PY}" -c "import neo4j" 2>/dev/null || {
  log "installing bolt read extra (.[menhir-scalar]) into the bench venv..."
  "${BENCH_PY}" -m pip install -e "${BENCH_DIR}[menhir-scalar]" >/dev/null
}

OPENAI_KEY="$("${MENHIR_MAIN_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"; [ -n "${OPENAI_KEY}" ] || die "no OPENAI_API_KEY in ${BENCH_DIR}/.env"

MENHIR_PID=""
cleanup(){
  if [ "${KEEP}" = "true" ]; then
    log "--keep set: leaving menhir (:${SS_PORT}) and Neo4j ${SS_NEO4J_NAME} (bolt ${SS_BOLT}) UP."
    return
  fi
  log "tearing down..."
  [ -n "${MENHIR_PID}" ] && kill "${MENHIR_PID}" 2>/dev/null || true
  docker rm -f "${SS_NEO4J_NAME}" >/dev/null 2>&1 || true   # ephemeral: no named volume, fully fresh
}
trap cleanup EXIT

# ---- FRESH ephemeral Neo4j (no named volume -> clean store every run; activation is fresh-only) ----
docker rm -f "${SS_NEO4J_NAME}" >/dev/null 2>&1 || true
log "creating FRESH throwaway Neo4j ${SS_NEO4J_NAME} (bolt ${SS_BOLT})..."
docker run -d --name "${SS_NEO4J_NAME}" -p ${SS_BOLT}:7687 -p ${SS_HTTP}:7474 \
  -e NEO4J_AUTH=neo4j/${SS_NEO4J_PW} -e NEO4J_server_memory_heap_max__size=2G \
  ${SS_NEO4J_IMAGE} >/dev/null
log "waiting for Neo4j HTTP (${SS_HTTP})..."
for _ in $(seq 1 60); do curl -sf "http://localhost:${SS_HTTP}" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://localhost:${SS_HTTP}" >/dev/null 2>&1 || die "Neo4j not ready"

# ---- throwaway menhir: scheduler ON (benchmark mode OFF) + scalar flag ON ----
EMPTY_ENV="$(mktemp)"; export ENV_FILE="${EMPTY_ENV}"
export MENHIR_LOG_DIR="${LME_RESULTS_DIR}/scalar-e2e-logs"; mkdir -p "${MENHIR_LOG_DIR}"
# ISOLATE the scheduler lease. The maintenance-scheduler lease is a cross-process SQLite lease
# keyed by lease_name in a SHARED telemetry DB (default .agent/mcp_telemetry.db), so a throwaway
# that shares it is BLOCKED by the operator's live menhir which already holds the lease -> the
# scheduler never starts -> no consolidation -> no scalar Views. A per-run telemetry DB gives this
# instance its own (empty) lease table so its scheduler acquires the lease and ticks.
# Decision-point evidence, same rationale as the LME builds (see config.sh): the scalar fold's
# drop reasons and vote distributions are emitted to telemetry lifecycle_events by
# consolidation_audit, not to the log, so without this an abstaining pass leaves no record of why.
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED="${SS_CONSOLIDATION_AUDIT:-1}"
export MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED="${SS_RECALL_AUDIT:-1}"
export MENHIR_LOG_LEVEL="${MENHIR_LOG_LEVEL:-${SS_LOG_LEVEL:-INFO}}"
export MENHIR_MCP_TELEMETRY_DB="${MENHIR_LOG_DIR}/scalar-e2e-telemetry-$$.db"
rm -f "${MENHIR_MCP_TELEMETRY_DB}" 2>/dev/null || true
export MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="${SS_PORT}"
# THE difference from build_graph.sh: scheduler must run, and the scalar path must be enabled.
export MENHIR_BENCHMARK_MODE=0
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED=1
export MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED=1
export MENHIR_PERSONAL_MEMORY_SCALAR_STATE_PERCEIVER_VERSION="${SS_PERCEIVER_VERSION}"
# Step 7 current-state View authority canary (default OFF; SS_VIEW_AUTHORITY=1 turns the live
# recall suppression hook ON for the SS_VIEW_AUTHORITY_AB branch).
export MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED="${SS_VIEW_AUTHORITY:-0}"
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_INTERVAL_S="${SS_INTERVAL_S}"
export MENHIR_MAX_LLM_CALLS_PER_JOB=20
export OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST=""
export MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY=""
export NEO4J_URI="${BOLT_URI}" NEO4J_USER="neo4j" NEO4J_PASSWORD="${SS_NEO4J_PW}" NEO4J_DATABASE="neo4j"
export GRAPHITI_LLM_PROVIDER="openai" MEMORY_GRAPHITI_PROVIDER="openai" LLM_CHAT_PROVIDER="openai" MEMORY_CHAT_PROVIDER="openai"
export GRAPHITI_RERANKER_PROVIDER="openai" MEMORY_GRAPHITI_RERANKER_PROVIDER="openai"
export GRAPHITI_EMBED_PROVIDER="openai" MEMORY_GRAPHITI_EMBED_PROVIDER="openai"
export MENHIR_PERSONAL_MEMORY_CHAT_MODEL="${SS_CHAT_MODEL}"
export OPENAI_API_KEY="${OPENAI_KEY}" OPENAI_CHAT_MODEL="${SS_CHAT_MODEL}" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}"

log "starting throwaway menhir on ${MENHIR_URL} (scheduler ON, scalar flag ON, interval ${SS_INTERVAL_S}s)..."
( cd "${MENHIR_MAIN}" && "${MENHIR_MAIN_BIN}" serve --port "${SS_PORT}" --host 127.0.0.1 ) & MENHIR_PID=$!
for _ in $(seq 1 90); do curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 || die "menhir not healthy"
if [ "${SS_DIAG:-0}" = "1" ]; then
  # Bounded provenance root-cause pass: per-call MENTIONS capture instead of the harness.
  log "menhir healthy. running the MENTIONS provenance diagnostic..."
  "${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/diagnose_mentions_provenance.py" \
    --menhir-url "${MENHIR_URL}" \
    --neo4j-uri "${BOLT_URI}" --neo4j-password "${SS_NEO4J_PW}" \
    "${EXTRA_ARGS[@]}"
  RC=$?
elif [ "${SS_LEADS:-0}" = "1" ]; then
  # LEADS measurement: seed via :TurnEvidence (declarant='user') so a View can trace to a real
  # declarant foundation (G14 bridge), then measure whether it LEADS in recall exactly when founded
  # (wrongful-authority = 0). Complements the Episodic A/B, which only exercises the advisory arm.
  log "menhir healthy. running the authority LEADS live measurement (flag=${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED})..."
  "${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/scalar_leads_authority_live.py" \
    --menhir-url "${MENHIR_URL}" \
    --neo4j-uri "${BOLT_URI}" --neo4j-password "${SS_NEO4J_PW}" \
    --label "flag-${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED}" \
    --max-wait-s "${SS_MAX_WAIT_S}" --out "${SS_OUT%.md}-leads-${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED}.json" \
    "${EXTRA_ARGS[@]}"
  RC=$?
elif [ "${SS_VIEW_AUTHORITY_AB:-0}" = "1" ]; then
  # Step 7c live A/B: drive PRODUCTION recall (suppression hook) with current->current fixtures.
  log "menhir healthy. running the live View-authority A/B (flag=${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED})..."
  "${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/scalar_view_authority_live.py" \
    --menhir-url "${MENHIR_URL}" \
    --neo4j-uri "${BOLT_URI}" --neo4j-password "${SS_NEO4J_PW}" \
    --label "flag-${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED}" \
    --max-wait-s "${SS_MAX_WAIT_S}" --out "${SS_OUT%.md}-view-authority-${MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED}.json" \
    "${EXTRA_ARGS[@]}"
  RC=$?
elif [ "${SS_PHASE_D:-0}" = "1" ]; then
  # Phase D counterfactual recall: seed stale->current, then baseline recall vs View-aware answer.
  log "menhir healthy. running the Phase D counterfactual recall evaluation..."
  "${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/scalar_phase_d.py" \
    --menhir-url "${MENHIR_URL}" \
    --neo4j-uri "${BOLT_URI}" --neo4j-password "${SS_NEO4J_PW}" \
    --max-wait-s "${SS_MAX_WAIT_S}" --out "${SS_OUT%.md}-phase-d.json" \
    "${EXTRA_ARGS[@]}"
  RC=$?
else
  log "menhir healthy. running the scalar-state harness..."

  # When --keep is set, also preserve the seeded namespace so the graph can be inspected over bolt.
  KEEP_NS_ARGS=()
  [ "${KEEP}" = "true" ] && KEEP_NS_ARGS=(--scalar-no-cleanup)

  # ---- run the harness ----
  "${BENCH_PY}" -m archolith_bench.cli harness menhir-scalar-state \
    --menhir-url "${MENHIR_URL}" \
    --neo4j-uri "${BOLT_URI}" --neo4j-password "${SS_NEO4J_PW}" \
    --scalar-max-wait-s "${SS_MAX_WAIT_S}" \
    --scalar-fixture "${SS_FIXTURE}" \
    --confirm-menhir-reset \
    --format "${SS_FORMAT}" --out "${SS_OUT}" \
    "${KEEP_NS_ARGS[@]}" "${EXTRA_ARGS[@]}"
  RC=$?
fi

log "done (harness rc=${RC}). evidence: ${SS_OUT}"
exit ${RC}
