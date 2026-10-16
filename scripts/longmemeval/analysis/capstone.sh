#!/usr/bin/env bash
# Arm C capstone — the honest end-to-end measurement of the Event->Fold->View architecture:
#   baseline D0  ->  gated perception WRITES real Views into the graph  ->  post D0  ->  delta
#
# Usage:
#   capstone.sh run              full sequence (baseline -> write -> post -> delta)
#   capstone.sh run --reuse-baseline   skip the baseline probe (reuse an existing rows file)
#   capstone.sh delta            re-print the delta report from existing files (free)
#   capstone.sh cleanup          delete every View this capstone wrote (source tag scoped)
#
# Cost: the two entropy probes are GPT-free (recall + query embeds); the write phase is the only
# LLM-billed step (~26 namespaces x k=5 PT_MODEL extractions + cross-checks). STOP on 429 per protocol.
set -uo pipefail

HERE="$(dirname "${BASH_SOURCE[0]}")"
source "$HERE/../config.sh"

OUT="${LME_RESULTS_DIR%/lme-ingest}/lme-capstone"; mkdir -p "$OUT"
BASE_ROWS="${PD_BASELINE:-$OUT/baseline-rows.json}"
POST_ROWS="${PD_POST:-$OUT/post-rows.json}"
WRITE_MANIFEST="${PD_WRITES:-$OUT/perception-write.json}"
PY="${MENHIR_FRONTIER_PY}"
SOURCE_TAG="perception-capstone"

cmd="${1:-run}"; shift || true

delta() {
  PD_BASELINE="$BASE_ROWS" PD_POST="$POST_ROWS" PD_WRITES="$WRITE_MANIFEST" \
    "$PY" "$HERE/perception_delta.py"
}

case "$cmd" in
  run)
    if [ "${1:-}" != "--reuse-baseline" ] || [ ! -f "$BASE_ROWS" ]; then
      echo "== 1/4 baseline D0 (GPT-free) =="
      LME_ENTROPY_OUT="$BASE_ROWS" bash "$HERE/entropy.sh" both || exit 1
    else
      echo "== 1/4 baseline D0: reusing $BASE_ROWS =="
    fi
    echo "== 2/4 gated perception writes (LLM-billed; stop on 429) =="
    PC_OUT="$WRITE_MANIFEST" "$PY" "$HERE/perception_write.py" || exit 1
    echo "== 3/4 post-write D0 (GPT-free) =="
    LME_ENTROPY_OUT="$POST_ROWS" bash "$HERE/entropy.sh" both || exit 1
    echo "== 4/4 delta =="
    delta
    echo
    echo "(views remain in the graph for inspection; 'capstone.sh cleanup' removes them)"
    ;;
  delta)
    delta
    ;;
  cleanup)
    "$PY" - "$LME_BOLT" "$LME_NEO4J_PW" "$SOURCE_TAG" <<'PYC'
import sys
from neo4j import GraphDatabase
bolt, pw, tag = sys.argv[1], sys.argv[2], sys.argv[3]
if not bolt.startswith("bolt://"): bolt = f"bolt://localhost:{bolt}"
d = GraphDatabase.driver(bolt, auth=("neo4j", pw))
with d.session() as s:
    n = s.run("MATCH (n:Entity {source:$t}) DETACH DELETE n RETURN count(n) AS n", t=tag).single()["n"]
print(f"deleted {n} capstone Views (source={tag})")
d.close()
PYC
    ;;
  *)
    grep '^#' "$0" | head -12; exit 2 ;;
esac
