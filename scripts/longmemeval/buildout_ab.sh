#!/usr/bin/env bash
# Buildout A/B: ingest the SAME 30-question slice on main vs frontier into two FRESH graphs,
# then recall+QA each. Measures ingest-code quality (the temporal/belief work), which the
# recall-only A/B on a shared graph could not see. Never touches the backed-up menhir-lme-neo4j.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"   # LME_* models and call cap
LIMIT="${LIMIT:-30}"
ARCH=/c/Users/thron/IdeaProjects/projects/archolith
BENCH="$ARCH/archolith-bench"
BENCH_PY="$BENCH/.venv/Scripts/python.exe"
BENCH_BIN="$BENCH/.venv/Scripts/archolith-bench"
KEY_PY="$BENCH_PY"
OPENAI_KEY="$("$KEY_PY" - "$BENCH/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"
[ -n "$OPENAI_KEY" ] || { echo "no OPENAI_API_KEY in $BENCH/.env" >&2; exit 1; }
LOG="${LME_BUILDOUT_LOG:-$BENCH/results/lme-slice-ab.log}"; mkdir -p "$(dirname "$LOG")"
log(){ printf '[slice] %s %s\n' "$(date '+%F %H:%M:%S')" "$*" >> "$LOG"; }

# run_branch <label> <src_dir> <neo4j_name> <bolt> <http> <vol> <port>
run_branch(){
  local LABEL="$1" SRC="$2" NAME="$3" BOLT="$4" HTTP="$5" VOL="$6" PORT="$7"
  local URL="http://localhost:$PORT" PW="slice$BOLT"
  local BIN="$SRC/.venv/Scripts/menhir"
  local OUT="$BENCH/results/lme-slice-$LABEL"; mkdir -p "$OUT"
  local MANIFEST_INGEST="$OUT/ingest_manifest.json"
  log "==== $LABEL: fresh graph $NAME (bolt $BOLT), src $SRC ===="

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm "$VOL" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" -p "$BOLT":7687 -p "$HTTP":7474 \
    -e NEO4J_AUTH="neo4j/$PW" -e NEO4J_server_memory_heap_max__size=1G \
    -v "$VOL":/data neo4j:5.26-community >/dev/null || { log "$LABEL: neo4j create FAILED"; return 1; }
  for _ in $(seq 1 60); do curl -sf "http://localhost:$HTTP" >/dev/null 2>&1 && break; sleep 2; done
  curl -sf "http://localhost:$HTTP" >/dev/null 2>&1 || { log "$LABEL: neo4j not ready"; return 1; }

  # ---- serve this branch's menhir over the fresh graph (openai extraction, benchmark mode) ----
  local EMPTY; EMPTY="$(mktemp)"
  local PID=""
  ( export ENV_FILE="$EMPTY" \
      LONGMEMEVAL_VARIANT=oracle \
      MENHIR_LOG_DIR="$OUT/menhir-logs" \
      MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="$PORT" \
      MENHIR_MAX_LLM_CALLS_PER_JOB="$LME_MAX_LLM_CALLS_PER_JOB" MENHIR_INGEST_CONCURRENCY=8 \
      OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST="" \
      MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY="" \
      NEO4J_URI="bolt://localhost:$BOLT" NEO4J_USER=neo4j NEO4J_PASSWORD="$PW" NEO4J_DATABASE=neo4j \
      GRAPHITI_LLM_PROVIDER=openai MEMORY_GRAPHITI_PROVIDER=openai LLM_CHAT_PROVIDER=openai MEMORY_CHAT_PROVIDER=openai \
      GRAPHITI_RERANKER_PROVIDER=openai MEMORY_GRAPHITI_RERANKER_PROVIDER=openai \
      GRAPHITI_EMBED_PROVIDER=openai MEMORY_GRAPHITI_EMBED_PROVIDER=openai \
      OPENAI_API_KEY="$OPENAI_KEY" OPENAI_CHAT_MODEL="$LME_EXTRACT_MODEL" OPENAI_EMBED_MODEL="$LME_EMBED_MODEL" \
      PYTHONPATH="$SRC/src"; \
    mkdir -p "$OUT/menhir-logs"; cd "$SRC" && "$BIN" serve --port "$PORT" ) >"$OUT/serve.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf "$URL/api/health" >/dev/null 2>&1; then log "$LABEL: menhir not healthy (see $OUT/serve.log)"; kill "$PID" 2>/dev/null; return 1; fi
  log "$LABEL: menhir healthy on $PORT"

  # ---- ingest the slice (drains per-item; graph READY on return) ----
  log "$LABEL: ingesting $LIMIT items..."
  LME_NEO4J_CONTAINER="$NAME" LME_NEO4J_PW="$PW" LONGMEMEVAL_VARIANT=oracle \
    "$BENCH_PY" "$BENCH/scripts/longmemeval/lib/ingest.py" --limit "$LIMIT" --menhir-url "$URL" \
    --manifest "$MANIFEST_INGEST" >>"$LOG" 2>&1
  log "$LABEL: ingest done exit=$?"

  # ---- recall+QA against the just-built graph (LME_ANSWER_MODEL answer, llm-judge) ----
  log "$LABEL: recall+QA..."
  ( cd "$BENCH" && \
    UPSTREAM_BASE_URL="https://api.openai.com/v1" UPSTREAM_API_KEY="$OPENAI_KEY" BENCHMARK_MODEL="$LME_ANSWER_MODEL" \
    OPENAI_API_KEY="$OPENAI_KEY" LONGMEMEVAL_VARIANT=oracle \
    "$BENCH_BIN" harness longmemeval-menhir --menhir-url "$URL" --recall-only \
      --model "$LME_ANSWER_MODEL" --scorer llm-judge --judge-model "$LME_JUDGE_MODEL" --format markdown \
      --output-dir "$OUT" --out "$OUT/harness_recall_ab.md" --limit "$LIMIT" --resume ) >>"$LOG" 2>&1
  log "$LABEL: recall+QA done exit=$?"

  # ---- graph state + score snapshot ----
  local COUNTS; COUNTS="$(docker exec "$NAME" cypher-shell -u neo4j -p "$PW" --format plain \
    "MATCH (e:Episodic) WHERE e.processing_state IS NOT NULL RETURN e.processing_state AS s, count(*) AS n ORDER BY s" 2>/dev/null | tr '\n' '|')"
  log "$LABEL: graph state = $COUNTS"
  kill "$PID" 2>/dev/null || true   # stop menhir; keep neo4j $NAME for inspection
  log "$LABEL: menhir stopped (neo4j $NAME left up on bolt $BOLT)"
}

log "===== SLICE BUILDOUT A/B START (limit=$LIMIT) ====="
[ "${SLICE_SKIP_MAIN:-0}" = 1 ] || run_branch main     "$ARCH/menhir"          lme-slice-main     7690 7477 lme-slice-main-data     8105
[ "${SLICE_SKIP_FRONTIER:-0}" = 1 ] || run_branch frontier "$ARCH/menhir-frontier" lme-slice-frontier 7691 7478 lme-slice-frontier-data 8106
log "===== SLICE BUILDOUT A/B COMPLETE ====="
log "main     -> $BENCH/results/lme-slice-main/harness_recall_ab.md"
log "frontier -> $BENCH/results/lme-slice-frontier/harness_recall_ab.md"
