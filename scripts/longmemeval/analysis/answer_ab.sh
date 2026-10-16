#!/usr/bin/env bash
# Answer A/B — do the committed perception Views improve end-to-end answer accuracy on the counting
# slice? Serves menhir ONCE over the graph, collects the answer-facing brief WITH the Views present,
# DELETES them (source-tagged), collects the brief again WITHOUT, then answers+judges both arms.
# The Views are re-derivable (re-run the consolidation) — this leaves the graph WITHOUT them; restore
# with:  BENCH_DIR/scripts/longmemeval/analysis/perception_write.py PC_SCOPE=all PC_SOURCE=perception-lme
#
# Cost: LME_ANSWER_MODEL answers + LME_JUDGE_MODEL (gpt-6-luna, Flex) judges over the counting slice x2 arms (~14x4 calls). STOP on 429.
set -uo pipefail
HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../config.sh"

PORT="${AB_PORT:-8120}"; URL="http://localhost:${PORT}"
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

del_views(){ "$PY" - "$BOLT" "$PW" "$SOURCE_TAG" <<'PYC'
import sys
from neo4j import GraphDatabase
bolt, pw, tag = sys.argv[1], sys.argv[2], sys.argv[3]
if not bolt.startswith("bolt://"): bolt=f"bolt://localhost:{bolt}"
d=GraphDatabase.driver(bolt, auth=("neo4j", pw))
with d.session() as s:
    n=s.run("MATCH (n:Entity {source:$t}) DETACH DELETE n RETURN count(n) AS n", t=tag).single()["n"]
print(f"deleted {n} Views (source={tag})")
d.close()
PYC
}
count_views(){ "$PY" - "$BOLT" "$PW" "$SOURCE_TAG" <<'PYC'
import sys
from neo4j import GraphDatabase
bolt, pw, tag = sys.argv[1], sys.argv[2], sys.argv[3]
if not bolt.startswith("bolt://"): bolt=f"bolt://localhost:{bolt}"
d=GraphDatabase.driver(bolt, auth=("neo4j", pw))
with d.session() as s:
    print("perception-lme Views currently in graph:", s.run("MATCH (n:Entity {source:$t}) RETURN count(n) AS n", t=tag).single()["n"])
d.close()
PYC
}

# ---- serve menhir over the graph (benchmark mode; scheduler off) ----
netstat -ano | grep ":$PORT" | grep -i listen | awk '{print $NF}' | sort -u | while read p; do taskkill //F //PID "$p" >/dev/null 2>&1; done
( export LONGMEMEVAL_VARIANT="${LONGMEMEVAL_VARIANT}" MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="$PORT" \
    OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST="" \
    MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY="" \
    NEO4J_URI="bolt://localhost:$BOLT" NEO4J_USER=neo4j NEO4J_PASSWORD="$PW" NEO4J_DATABASE=neo4j \
    GRAPHITI_EMBED_PROVIDER=openai MEMORY_GRAPHITI_EMBED_PROVIDER=openai \
    OPENAI_API_KEY="$KEY" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}" PYTHONPATH="$SRC/src" ; \
  cd "$SRC" && "$SERVE_BIN" serve --port "$PORT" ) >"$OUT/serve.log" 2>&1 & PID=$!
trap '[ -n "${PID:-}" ] && kill "$PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 90); do curl -sf "$URL/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "$URL/api/health" >/dev/null 2>&1 || { echo "menhir not healthy; see $OUT/serve.log" >&2; exit 1; }

export MENHIR_URL="$URL" OPENAI_API_KEY="$KEY" AB_OUT="$OUT" \
       ANSWER_MODEL="${LME_ANSWER_MODEL}" JUDGE_MODEL="${LME_JUDGE_MODEL}"

echo "== views present? =="; count_views
echo "== 1/4 collect WITH views =="; MODE=collect AB_TAG=withviews "$PY" "$HERE/answer_ab.py" || exit 1
echo "== 2/4 delete Views =="; del_views
echo "== 3/4 collect WITHOUT views =="; MODE=collect AB_TAG=noviews "$PY" "$HERE/answer_ab.py" || exit 1
echo "== 4/4 answer + judge both arms =="; MODE=score "$PY" "$HERE/answer_ab.py" || exit 1
echo; echo "(graph now has NO perception-lme Views; re-run perception_write.py PC_SCOPE=all to restore)"
