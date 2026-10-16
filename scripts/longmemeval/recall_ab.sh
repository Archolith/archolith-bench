#!/usr/bin/env bash
# Recall-only LongMemEval Mode-B A/B against the PRE-BUILT menhir-lme-neo4j graph.
#
# Point it at any branch OR worktree path; it materializes the menhir source (auto
# `git worktree add` when given a bare branch, reusing an existing worktree if there is
# one), serves menhir from that source over the persistent LME Neo4j via PYTHONPATH
# (no per-variant rewiring), and runs the no_memory vs menhir_recall A/B in RECALL-ONLY
# mode — it never re-ingests and never resets the graph.
#
# Prereq: the graph must already be built once via scripts/longmemeval/build_graph.sh.
# Usage:  recall_ab.sh <branch|worktree-path> [limit]
#   recall_ab.sh main 30
#   recall_ab.sh claude/menhir-chain-handoff-doc-7iuat2 30
#   recall_ab.sh ../menhir-frontier 30
# Env overrides: LME_ANSWER_MODEL (default gpt-4o), LONGMEMEVAL_VARIANT (default oracle —
#   MUST match the variant the graph was built with), SCORER (containment|llm-judge).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SRC_ARG="${1:?usage: recall_ab.sh <branch|worktree-path> [limit]}"
LIMIT="${2:-30}"

# Pre-built LME graph (from build_graph.sh) — REUSE only, NEVER create or reset here.
MENHIR_PORT="${LME_PORT_RECALL}"
MENHIR_URL="http://localhost:${MENHIR_PORT}"
ANSWER_MODEL="${LME_ANSWER_MODEL}"
SCORER="${SCORER:-${LME_SCORER}}"   # framework default is llm-judge (config.sh); containment undercounts (paraphrase)
JUDGE_MODEL="${LME_JUDGE_MODEL}"   # llm-judge grader model (recorded in the manifest)
RECALL_TOP_K="${LME_RECALL_LIMIT}"              # per-question recall depth
# MUST match the variant the graph was ingested under, or question_ids (namespaces) won't line up.
export LONGMEMEVAL_VARIANT="${LONGMEMEVAL_VARIANT:-oracle}"

log(){ printf '[lme-recall] %s\n' "$*" >&2; }
die(){ printf '[lme-recall] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- resolve the menhir source to A/B (explicit worktree path, or auto worktree from a branch) ----
WT_CREATED=""
if [ -d "${SRC_ARG}/src/menhir" ]; then
  SRC_DIR="$(cd "${SRC_ARG}" && pwd)"                       # an explicit menhir checkout/worktree
  log "source = path ${SRC_DIR}"
else
  BRANCH="${SRC_ARG}"                                        # treat as a branch name
  SRC_DIR="$(git -C "${MENHIR_MAIN}" worktree list --porcelain \
            | awk -v b="refs/heads/${BRANCH}" '/^worktree /{p=substr($0,10)} /^branch /{if($2==b) print p}' \
            | head -1)"
  if [ -n "${SRC_DIR}" ]; then
    log "source = existing worktree for ${BRANCH}: ${SRC_DIR}"
  else
    SRC_DIR="${ARCH_DIR}/.lme-worktrees/$(printf '%s' "${BRANCH}" | tr '/:' '__')"
    log "source = NEW worktree for ${BRANCH} -> ${SRC_DIR}"
    git -C "${MENHIR_MAIN}" worktree add "${SRC_DIR}" "${BRANCH}" >/dev/null || die "git worktree add ${BRANCH} failed"
    WT_CREATED="${SRC_DIR}"
  fi
fi
[ -d "${SRC_DIR}/src/menhir" ] || die "no src/menhir under ${SRC_DIR}"
# Serve from the source's OWN venv when it has one — branches may add an interpreter-identity
# guard (e.g. frontier's runtime.py rejects being run under a foreign venv). Fall back to the
# main menhir venv for checkouts without their own .venv.
if [ -f "${SRC_DIR}/.venv/Scripts/menhir" ] || [ -f "${SRC_DIR}/.venv/Scripts/menhir.exe" ]; then
  SERVE_BIN="${SRC_DIR}/.venv/Scripts/menhir"
else
  SERVE_BIN="${MENHIR_MAIN_BIN}"
fi
VARIANT="$(basename "${SRC_DIR}")"
RUN_OUTPUT_DIR="${BENCH_DIR}/results/lme-recall-${VARIANT}"; mkdir -p "${RUN_OUTPUT_DIR}"

OPENAI_KEY="$("${MENHIR_MAIN_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get(sys.argv[2],""))
PY
)"; [ -n "${OPENAI_KEY}" ] || die "no OPENAI_API_KEY in ${BENCH_DIR}/.env"

# ---- ensure the PRE-BUILT Neo4j is up (reuse only; never create, never reset) ----
lme_container_running "${LME_NEO4J_NAME}" \
  || docker start "${LME_NEO4J_NAME}" >/dev/null 2>&1 \
  || die "pre-built ${LME_NEO4J_NAME} not found — run scripts/longmemeval/build_graph.sh first"
for _ in $(seq 1 60); do curl -sf "http://localhost:${LME_HTTP}" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "http://localhost:${LME_HTTP}" >/dev/null 2>&1 || die "Neo4j ${LME_NEO4J_NAME} not ready"

MENHIR_PID=""
cleanup(){
  [ -n "${MENHIR_PID}" ] && kill "${MENHIR_PID}" 2>/dev/null || true
  # Remove ONLY a worktree this script created; never an existing one, never the Neo4j.
  [ -n "${WT_CREATED}" ] && git -C "${MENHIR_MAIN}" worktree remove --force "${WT_CREATED}" 2>/dev/null || true
}
trap cleanup EXIT

# ---- serve menhir from the variant source over the pre-built Neo4j (recall path only) ----
export PYTHONPATH="${SRC_DIR}/src"          # load the variant's source over the installed menhir
EMPTY_ENV="$(mktemp)"; export ENV_FILE="${EMPTY_ENV}"
export MENHIR_LOG_DIR="${RUN_OUTPUT_DIR}/menhir-logs"; mkdir -p "${MENHIR_LOG_DIR}"
export MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="${MENHIR_PORT}"
# Per-run telemetry: recall menhir writes its own SQLite sidecar alongside the recall results.
export MENHIR_MCP_TELEMETRY_DB="${RUN_OUTPUT_DIR}/mcp_telemetry.db"
export OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST=""
export MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY=""
export NEO4J_URI="bolt://localhost:${LME_BOLT}" NEO4J_USER="neo4j" NEO4J_PASSWORD="${LME_NEO4J_PW}" NEO4J_DATABASE="neo4j"
# Recall embeds the query, so it needs the SAME embedder the graph was built with.
export GRAPHITI_EMBED_PROVIDER="openai" MEMORY_GRAPHITI_EMBED_PROVIDER="openai"
export OPENAI_API_KEY="${OPENAI_KEY}" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}"

log "serving menhir (${VARIANT}) on ${MENHIR_URL} over ${LME_NEO4J_NAME} (bolt ${LME_BOLT}) via ${SERVE_BIN}..."
( cd "${SRC_DIR}" && "${SERVE_BIN}" serve --port "${MENHIR_PORT}" ) & MENHIR_PID=$!
for _ in $(seq 1 90); do curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "${MENHIR_URL}/api/health" >/dev/null 2>&1 || die "menhir not healthy"

# ---- recall-only A/B: no ingest, no reset (so --confirm-menhir-reset is NOT needed) ----
export UPSTREAM_BASE_URL="https://api.openai.com/v1" UPSTREAM_API_KEY="${OPENAI_KEY}" BENCHMARK_MODEL="${ANSWER_MODEL}"
log "recall-only A/B (${VARIANT}) variant=${LONGMEMEVAL_VARIANT} limit=${LIMIT} scorer=${SCORER} judge=${JUDGE_MODEL}..."
RUN_STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
( cd "${BENCH_DIR}" && "${BENCH_BIN}" harness longmemeval-menhir \
    --menhir-url "${MENHIR_URL}" --recall-only \
    --model "${ANSWER_MODEL}" --scorer "${SCORER}" --judge-model "${JUDGE_MODEL}" --format markdown \
    --output-dir "${RUN_OUTPUT_DIR}" --out "${RUN_OUTPUT_DIR}/harness_recall_ab.md" \
    --limit "${LIMIT}" --resume )
log "done -> ${RUN_OUTPUT_DIR}/harness_recall_ab.md"

# ---- run manifest: record the FULL test surface for reproducibility ----------
# Every knob that shapes the result, resolved to its effective value, plus code
# commits, graph state, and the parsed scores. Reproduce a run from this file alone.
MANIFEST="${RUN_OUTPUT_DIR}/run_manifest.json"
MFT_MENHIR_COMMIT="$(git -C "${SRC_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
MFT_MENHIR_BRANCH="$(git -C "${SRC_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
MFT_MENHIR_DIRTY="$(git -C "${SRC_DIR}" status --porcelain 2>/dev/null | head -1 | grep -q . && echo true || echo false)"
MFT_BENCH_COMMIT="$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
MFT_BENCH_BRANCH="$(git -C "${BENCH_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
MFT_NEO4J_IMAGE="$(docker inspect "${LME_NEO4J_NAME}" --format '{{.Config.Image}}' 2>/dev/null || echo unknown)"
MFT_DATASET_SNAPSHOT="$(ls -1 "${HOME}/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots" 2>/dev/null | head -1 || echo unknown)"
MFT_GRAPH_COUNTS="$(docker exec "${LME_NEO4J_NAME}" cypher-shell -u neo4j -p "${LME_NEO4J_PW}" --format plain \
  "MATCH (e:Episodic) WHERE e.processing_state IS NOT NULL RETURN e.processing_state AS s, count(*) AS n ORDER BY s" 2>/dev/null | tr '\n' '|' || echo unknown)"
export MFT_MENHIR_COMMIT MFT_MENHIR_BRANCH MFT_MENHIR_DIRTY MFT_BENCH_COMMIT MFT_BENCH_BRANCH \
       MFT_NEO4J_IMAGE MFT_DATASET_SNAPSHOT MFT_GRAPH_COUNTS RUN_STARTED VARIANT SRC_DIR SERVE_BIN \
       ANSWER_MODEL SCORER JUDGE_MODEL RECALL_TOP_K LIMIT LONGMEMEVAL_VARIANT LME_NEO4J_NAME LME_BOLT
"${MENHIR_MAIN_PY}" - "${MANIFEST}" "${RUN_OUTPUT_DIR}/harness_recall_ab.md" <<'PY'
import json, os, re, sys, datetime
manifest_path, md_path = sys.argv[1], sys.argv[2]
def counts(raw):
    out = {}
    for cell in (raw or "").split("|"):
        cell = cell.strip().strip('"')
        m = re.match(r'"?(\w+)"?,\s*(\d+)', cell) or re.match(r'(\w+),\s*(\d+)', cell)
        if m and m.group(1) not in ("s",): out[m.group(1)] = int(m.group(2))
    return out
scores = []
try:
    for line in open(md_path, encoding="utf-8"):
        m = re.match(r'\|\s*(\w[\w ]*?)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|', line)
        if m and m.group(1) not in ("Arm",):
            scores.append({"arm": m.group(1), "n": int(m.group(2)), "score": float(m.group(3))})
except FileNotFoundError:
    pass
manifest = {
    "run_completed_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_started_utc": os.environ.get("RUN_STARTED"),
    "variant": os.environ.get("VARIANT"),
    "code": {
        "menhir": {"src_dir": os.environ.get("SRC_DIR"), "branch": os.environ.get("MFT_MENHIR_BRANCH"),
                    "commit": os.environ.get("MFT_MENHIR_COMMIT"), "dirty": os.environ.get("MFT_MENHIR_DIRTY") == "true",
                    "serve_venv": os.environ.get("SERVE_BIN")},
        "bench": {"branch": os.environ.get("MFT_BENCH_BRANCH"), "commit": os.environ.get("MFT_BENCH_COMMIT")},
    },
    "graph": {
        "neo4j_container": os.environ.get("LME_NEO4J_NAME"), "neo4j_image": os.environ.get("MFT_NEO4J_IMAGE"),
        "bolt_port": os.environ.get("LME_BOLT"), "lme_variant": os.environ.get("LONGMEMEVAL_VARIANT"),
        "dataset": "xiaowu0162/longmemeval", "dataset_snapshot": os.environ.get("MFT_DATASET_SNAPSHOT"),
        "episodic_state_counts": counts(os.environ.get("MFT_GRAPH_COUNTS")),
    },
    "recall": {
        "preset": "knowledge (menhir default; not set by harness)",
        "top_k": int(os.environ.get("RECALL_TOP_K", "10")),
        "top_k_configurable_from_launcher": False,
        "include_session": True,
        "embed_model": "text-embedding-3-small",
    },
    "answer": {"model": os.environ.get("ANSWER_MODEL"), "upstream": "https://api.openai.com/v1"},
    "scoring": {"scorer": os.environ.get("SCORER"),
                 "judge_model": os.environ.get("JUDGE_MODEL") if os.environ.get("SCORER") == "llm-judge" else None},
    "run": {"n_questions": int(os.environ.get("LIMIT", "0")), "namespace_template": "lme-{question_id}"},
    "results": scores,
    "results_md": os.path.basename(md_path),
}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {manifest_path}")
PY
log "manifest -> ${MANIFEST}"
