#!/usr/bin/env bash
# Stratified answer-accuracy matrix: separate retrieval from generation across configs & types.
# Configs (menhir serve env):
#   node_plain : fact_edges off, frontier oracle OFF   -> boring node-only baseline
#   frontier   : fact_edges off, frontier defaults ON  -> oracle ranking + lens fix
#   pointer    : fact_edges on, mode=pointer (lens-gated), oracle ON
# For each config: answer-score (LME_ANSWER_MODEL + LME_JUDGE_MODEL judge, recall-only) across all 6
# question types, LME_PER_TYPE each = 90 questions/config. Graph already holds all 500 namespaces.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

PER="${LME_PER_TYPE:-15}"
PORT="${LME_PORT_MATRIX}"
URL="http://localhost:$PORT"
KEY="$("${MENHIR_MAIN_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"
TYPES="temporal-reasoning multi-session knowledge-update single-session-user single-session-assistant single-session-preference"
LOG="${BENCH_DIR}/results/lme-matrix/analysis.log"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
log(){ printf '[matrix] %s %s\n' "$(date '+%F %H:%M:%S')" "$*" | tee -a "$LOG"; }

serve(){ # <fact_edges> <mode> <oracle>
  local FE="$1" MODE="$2" ORACLE="$3"
  netstat -ano | grep ":$PORT" | grep -i listen | awk '{print $NF}' | sort -u | while read p; do taskkill //F //PID "$p" >/dev/null 2>&1; done
  docker start "${LME_NEO4J_NAME}" >/dev/null 2>&1
  for _ in $(seq 1 40); do docker exec "${LME_NEO4J_NAME}" cypher-shell -u neo4j -p "${LME_NEO4J_PW}" "RETURN 1" >/dev/null 2>&1 && break; sleep 2; done
  ( export LONGMEMEVAL_VARIANT=oracle MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="$PORT" \
      OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST="" \
      MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY="" \
      NEO4J_URI="bolt://localhost:${LME_BOLT}" NEO4J_USER=neo4j NEO4J_PASSWORD="${LME_NEO4J_PW}" NEO4J_DATABASE=neo4j \
      GRAPHITI_EMBED_PROVIDER=openai MEMORY_GRAPHITI_EMBED_PROVIDER=openai \
      OPENAI_API_KEY="$KEY" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}" PYTHONPATH="${MENHIR_FRONTIER}/src" \
      MENHIR_FRONTIER_FACT_EDGES="$FE" MENHIR_FRONTIER_FACT_EDGE_MODE="$MODE" \
      MENHIR_FRONTIER_ORACLE_RANKING="$ORACLE" ; \
    cd "${MENHIR_FRONTIER}" && "${MENHIR_FRONTIER_BIN}" serve --port "$PORT" ) >>"$LOG" 2>&1 & echo $!
}

config(){ # <cfg_label> <fact_edges> <mode> <oracle>
  local CFG="$1" FE="$2" MODE="$3" ORACLE="$4"
  local PID; PID=$(serve "$FE" "$MODE" "$ORACLE")
  for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -sf "$URL/api/health" >/dev/null 2>&1 || { log "$CFG: NOT healthy"; kill "$PID" 2>/dev/null; return 1; }
  log "$CFG: up (fe=$FE mode=$MODE oracle=$ORACLE)"
  for T in $TYPES; do
    local OUT="${BENCH_DIR}/results/lme-matrix-$CFG-$T"; mkdir -p "$OUT"
    ( cd "${BENCH_DIR}" && UPSTREAM_BASE_URL="https://api.openai.com/v1" UPSTREAM_API_KEY="$KEY" \
        BENCHMARK_MODEL="${LME_ANSWER_MODEL}" OPENAI_API_KEY="$KEY" LONGMEMEVAL_VARIANT=oracle \
        "${BENCH_BIN}" harness longmemeval-menhir --menhir-url "$URL" --recall-only \
          --subset "$T" --model "${LME_ANSWER_MODEL}" --scorer "${LME_SCORER}" --judge-model "${LME_JUDGE_MODEL}" \
          --format markdown --output-dir "$OUT" --out "$OUT/harness.md" --limit "$PER" ) >>"$LOG" 2>&1
    local S; S=$(grep -oE "menhir_recall \| [0-9]+ \| [0-9.]+" "$OUT/harness.md" 2>/dev/null | grep -oE "[0-9.]+$" | head -1)
    log "$CFG/$T: score=${S:-NA}"
  done
  kill "$PID" 2>/dev/null || true; sleep 2
}

log "===== ANSWER-ACCURACY MATRIX (3 configs x 6 types x $PER) ====="
config node_plain false standalone false
config frontier   false standalone true
config pointer    true  pointer    true
log "===== DONE ====="
echo "__MATRIX_COMPLETE__" | tee -a "$LOG"
