#!/usr/bin/env bash
# Answer A/B (confirm-a-change variant) — measure a surface/pass change end-to-end AND leave the
# perception pass SAVED. Single consolidation, ordered so the graph ends WITH the Views:
#   1. serve menhir (Views currently absent)         2. collect NO-VIEWS brief
#   3. run the perception consolidation (writes Views, new surface)   4. collect WITH-VIEWS brief
#   5. answer + judge both arms                       6. leave the Views in place (pass saved)
# Use after a View-surface / pipeline change to confirm the lift without a throwaway delete.
# Cost: one consolidation (~90 namespaces) + LME_ANSWER_MODEL answers and LME_JUDGE_MODEL judges (gpt-6-luna, Flex) over the counting slice. STOP on 429.
set -uo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../config.sh"

PORT="${AB_PORT:-8121}"; URL="http://localhost:${PORT}"
NAME="${LME_NEO4J_NAME}"; BOLT="${LME_BOLT}"; PW="${LME_NEO4J_PW}"
PY="${MENHIR_FRONTIER_PY}"; SERVE_BIN="${MENHIR_FRONTIER_BIN}"; SRC="${MENHIR_FRONTIER}"
OUT="${AB_OUT:-${LME_RESULTS_DIR%/lme-ingest}/lme-answer-ab}"; mkdir -p "$OUT"
SOURCE_TAG="perception-lme"
docker start "$NAME" >/dev/null 2>&1

KEY="$("$PY" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PYK'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PYK
)"
[ -n "$KEY" ] || { echo "no OPENAI key in ${BENCH_DIR}/.env" >&2; exit 1; }

count_views(){ "$PY" - "$BOLT" "$PW" "$SOURCE_TAG" <<'PYC'
import sys
from neo4j import GraphDatabase
bolt, pw, tag = sys.argv[1], sys.argv[2], sys.argv[3]
if not bolt.startswith("bolt://"): bolt=f"bolt://localhost:{bolt}"
d=GraphDatabase.driver(bolt, auth=("neo4j", pw))
with d.session() as s:
    print("perception-lme Views in graph:", s.run("MATCH (n:Entity {source:$t}) RETURN count(n) AS n", t=tag).single()["n"])
d.close()
PYC
}

# ---- serve menhir over the graph ----
netstat -ano | grep ":$PORT" | grep -i listen | awk '{print $NF}' | sort -u | while read p; do taskkill //F //PID "$p" >/dev/null 2>&1; done
( export LONGMEMEVAL_VARIANT="${LONGMEMEVAL_VARIANT}" MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="$PORT" \
    OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST="" \
    MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY="" \
    NEO4J_URI="bolt://localhost:$BOLT" NEO4J_USER=neo4j NEO4J_PASSWORD="$PW" NEO4J_DATABASE=neo4j \
    GRAPHITI_EMBED_PROVIDER=openai MEMORY_GRAPHITI_EMBED_PROVIDER=openai \
    OPENAI_API_KEY="$KEY" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}" PYTHONPATH="$SRC/src" ; \
  cd "$SRC" && "$SERVE_BIN" serve --port "$PORT" ) >"$OUT/serve_confirm.log" 2>&1 & PID=$!
trap '[ -n "${PID:-}" ] && kill "$PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "$URL/api/health" >/dev/null 2>&1 || { echo "menhir not healthy; see $OUT/serve_confirm.log" >&2; exit 1; }

export MENHIR_URL="$URL" OPENAI_API_KEY="$KEY" AB_OUT="$OUT" \
       ANSWER_MODEL="${LME_ANSWER_MODEL}" JUDGE_MODEL="${LME_JUDGE_MODEL}"

echo "== 1/5 views before (expect 0) =="; count_views
echo "== 2/5 collect NO-VIEWS brief =="; MODE=collect AB_TAG=noviews "$PY" "$HERE/answer_ab.py" || exit 1
echo "== 3/5 consolidate (save the pass, new surface) =="
PC_SCOPE=all PC_SOURCE="$SOURCE_TAG" PC_K=5 PC_THRESHOLD=1.0 PC_OUT="$OUT/perception-lme-write.json" \
  "$PY" "$HERE/perception_write.py" || exit 1
echo "== views after consolidation =="; count_views
echo "== 4/5 collect WITH-VIEWS brief =="; MODE=collect AB_TAG=withviews "$PY" "$HERE/answer_ab.py" || exit 1
echo "== 5/5 answer + judge both arms =="; MODE=score "$PY" "$HERE/answer_ab.py" || exit 1
echo; echo "(perception pass SAVED — Views remain in the graph, source=$SOURCE_TAG)"
