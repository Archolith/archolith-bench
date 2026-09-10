#!/usr/bin/env bash
# LongMemEval framework configuration — centralized, environment-overridable.
# Source this in every script; all values use ${VAR:-default} for override.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
#   echo "Build port: $LME_PORT_BUILD"
#   LME_ANSWER_MODEL=gpt-4-turbo ./recall_ab.sh main 30

# ---- Paths (derived from script location, not hardcoded absolutes) ----
# All path derivations should use git rev-parse --show-toplevel where possible.
# This file is sourced from scripts/longmemeval/, so $(dirname) gives us that dir.
# Parent of parent is archolith-bench root.
SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
_LONGMEMEVAL_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
BENCH_DIR="${BENCH_DIR:-$(git rev-parse --show-toplevel)}"
ARCH_DIR="${ARCH_DIR:-${BENCH_DIR%/*}}"  # parent of archolith-bench
MENHIR_MAIN="${MENHIR_MAIN:-${ARCH_DIR}/menhir}"
# menhir-frontier merged into main (2026-07); that checkout no longer exists. Default
# MENHIR_FRONTIER to MENHIR_MAIN so every consuming script (build_graph.sh, lme.sh
# backfill-dates/presence/ir-gate, retry.sh, analysis/*) works without a manual override.
# Still overridable if a real frontier fork ever returns.
MENHIR_FRONTIER="${MENHIR_FRONTIER:-${MENHIR_MAIN}}"

# ---- Neo4j (persistent LME graph) ----
LME_NEO4J_NAME="${LME_NEO4J_NAME:-menhir-lme-neo4j}"
LME_BOLT="${LME_BOLT:-7689}"
LME_HTTP="${LME_HTTP:-7476}"
LME_NEO4J_PW="${LME_NEO4J_PW:-lmedata123}"
LME_NEO4J_VOL="${LME_NEO4J_VOL:-menhir-lme-data}"
LME_NEO4J_IMAGE="${LME_NEO4J_IMAGE:-neo4j:5.26-community}"
# Heap for the build's Neo4j. This is sized against DOCKER'S VM limit, not host RAM -- Docker
# Desktop here caps the whole VM at ~3.8 GiB, and menhir-full + menhir-full-neo4j already hold
# ~0.8 GiB of it. Starting an additional 2G-heap Neo4j on top of a couple of existing graphs
# exhausted that budget and wedged the daemon (every API call returning HTTP 500) on 2026-07-27.
# A one-item smoke needs a tiny fraction of this; override it rather than paying full freight.
LME_NEO4J_HEAP="${LME_NEO4J_HEAP:-2G}"

# Avoid `docker ps | grep -q` for container detection. Every caller enables
# `set -o pipefail`; when the wanted container is near the start of a long list,
# grep exits after the match and Docker can receive SIGPIPE, turning a true match
# into status 1. Container inspect is exact and emits no multi-row pipeline.
lme_container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

lme_container_running() {
  local running
  running="$(docker container inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)"
  [ "${running}" = "true" ]
}

# ---- Menhir serve ports (one per workflow) ----
LME_PORT_BUILD="${LME_PORT_BUILD:-8102}"               # build_graph.sh
LME_PORT_RECALL="${LME_PORT_RECALL:-8103}"             # recall_ab.sh (shared persistent graph)
LME_PORT_BUILDOUT_MAIN="${LME_PORT_BUILDOUT_MAIN:-8105}"       # buildout_ab.sh (main branch, fresh graph)
LME_PORT_BUILDOUT_FRONTIER="${LME_PORT_BUILDOUT_FRONTIER:-8106}" # buildout_ab.sh (frontier, fresh graph)

# ---- Analysis ports (answer matrix, MSC sweep, ablation, retrieval quality) ----
LME_PORT_MATRIX="${LME_PORT_MATRIX:-8112}"             # analysis/answer_matrix.sh
LME_PORT_MSC="${LME_PORT_MSC:-8113}"                   # analysis/msc_sweep.sh
LME_PORT_ABL="${LME_PORT_ABL:-8114}"                   # analysis/ablation_sweep.sh
LME_PORT_RQ="${LME_PORT_RQ:-8109}"                     # analysis/lib/retrieval_quality.py
LME_PORT_BRIEF="${LME_PORT_BRIEF:-8118}"               # analysis/brief_ab.sh (BriefBuilder A/B)
LME_PORT_ENTROPY="${LME_PORT_ENTROPY:-8119}"           # analysis/entropy.sh (D0 retrieval entropy)

# ---- Models ----
# gpt-4o-mini (not nano): matches what Zep/Graphiti and Mem0 both use for LongMemEval
# extraction, so results are comparable to their published numbers. Confirmed via direct
# A/B test 2026-07-15 that nano measurably under-extracts vs mini under sparse context
# (see menhir/.agent/reviews/rca-lme-stale-fact-retention-2026-07-15.md) -- mini recovers
# more entities but does NOT fully fix the RELEVANT_SCHEMA_LIMIT=10 recency-window bug;
# see menhir/.agent/plans/menhir-extraction-prompt-recency-recall-research.md for the
# follow-on prompt-level research. Slower/costlier full-corpus builds than nano was chosen
# for; accept the tradeoff for comparability, same as Zep/Mem0 did.
LME_EXTRACT_MODEL="${LME_EXTRACT_MODEL:-gpt-4o-mini}"
# Extraction defaults to the canonical OpenAI path for benchmark comparability.  A local
# OpenAI-compatible endpoint can be selected for explicitly noncanonical model probes without
# changing the embedding provider used by the corpus build.
LME_EXTRACT_PROVIDER="${LME_EXTRACT_PROVIDER:-openai}"
LME_EXTRACT_BASE_URL="${LME_EXTRACT_BASE_URL:-}"
LME_EXTRACT_API_KEY="${LME_EXTRACT_API_KEY:-not-needed}"
LME_ANSWER_MODEL="${LME_ANSWER_MODEL:-gpt-4o}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-gpt-4o-mini}"
LME_EMBED_MODEL="${LME_EMBED_MODEL:-text-embedding-3-small}"
LME_SCORER="${LME_SCORER:-llm-judge}"

# ---- Ingest policy ----
# Legacy sentence splitting inflated knowledge-update items by roughly 8x and was
# abandoned in the full validation campaign. Adaptive is the frozen, evidenced
# default; override explicitly when reproducing an older arm.
LME_SEGMENTATION="${LME_SEGMENTATION:-adaptive}"
LME_REQUIRE_FRESH="${LME_REQUIRE_FRESH:-0}"
# Noncanonical: permit code-commit drift on resume. A canonical run refuses to resume when the
# menhir or bench commit changes; set this to 1 for development iteration where you expect to
# resume a partially-built graph after a code change. The provenance file and all output will be
# labelled noncanonical.
LME_NONCANONICAL="${LME_NONCANONICAL:-0}"
LME_REQUIRE_TURN_EVIDENCE="${LME_REQUIRE_TURN_EVIDENCE:-0}"

# ---- Evidence capture: ON for every run, not just the KU buildout ----
# WHY THESE DEFAULT ON. The decision points that explain a scalar result are recorded through the
# consolidation audit, NOT through the log. A 2026-09-09 date-smoke ran at MENHIR_LOG_LEVEL=DEBUG and
# produced 2,649 log lines that did not contain the answer: the typed-scalar extract drop reasons
# (`by_reason`), the per-claim vote distributions, and the gate verdicts are emitted to telemetry
# `lifecycle_events` by consolidation_audit, which was OFF because only
# run_knowledge_update_buildout.sh set it. The run reported "scalar ingest validation: PASS" while
# writing zero assertions, and the cause (hedged_value, then interpretation scatter) was invisible
# until the pass was re-run with the audit on.
#
# The audits are behavior-neutral by contract -- consolidation_audit's emit can never raise into the
# caller and changes only what is recorded -- so there is no measurement reason to keep them off.
# Set either to 0 to opt out of a specific run.
LME_CONSOLIDATION_AUDIT_ENABLED="${LME_CONSOLIDATION_AUDIT_ENABLED:-1}"
LME_RECALL_AUDIT_ENABLED="${LME_RECALL_AUDIT_ENABLED:-1}"

# Menhir log level for bench runs. INFO by default: the failure paths worth reading (View write
# refusals, budget refusals, slow enrichment) already log at INFO or WARNING, and DEBUG turns on the
# openai client's full request dump -- every prompt, including the episode text -- which is both
# enormous and a disclosure risk on any corpus that is not a public fixture. Set DEBUG deliberately
# while troubleshooting a specific run, not as a standing default.
LME_MENHIR_LOG_LEVEL="${LME_MENHIR_LOG_LEVEL:-INFO}"
LME_SCALAR_STATE_ENABLED="${LME_SCALAR_STATE_ENABLED:-0}"
LME_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED:-0}"
LME_SCALAR_CONSOLIDATION_K="${LME_SCALAR_CONSOLIDATION_K:-3}"
LME_SCALAR_CALL_BUDGET="${LME_SCALAR_CALL_BUDGET:-50}"
# Number of distinct item namespaces enriched concurrently. The general framework remains
# single-flight by default; guarded KU reingests opt into width 2.
LME_INGEST_CONCURRENCY="${LME_INGEST_CONCURRENCY:-1}"
# ---- Typed-scalar consistency gate (menhir-side, all default-off in menhir itself) ----
# The gate groups k samples by an exact identity label (subject/attribute/scope/kind/unit/op/value)
# and commits only what recurs in >= threshold*k samples. The model smears one fact's identity
# across those three free-text fields in arbitrary order, so exact-matching each independently
# turns one AGREED fact into several single-vote claims that then veto each other. Reconciliation
# votes on the identity TUPLE and picks the modal combination.
#
# Historical pre-binding measurement on the frozen LME panel
# (scripts/replay_fold_flags.py, correct current Views / 100):
#   threshold 1.0, no reconcile ...... 20   <- the shipped default; what a build gets with these off
#   threshold 2/3 + attribute ........ 65
#   + scope .......................... 70
#   + scope + subject ................ 72
# DO NOT use this grid as the current baseline. It predates menhir's resolved-twin non-self
# binding repair (1550c56) and the 2026-07-27 TurnEvidence/admission provenance work. A new run
# must rebuild the knowledge-update corpus on current Menhir before comparing these flag arms.
# THRESHOLD MUST BE THE RATIO `2/3`, never 0.67: 0.67 > 2/3, so at k=3 it still demands all three
# votes and silently buys nothing. menhir parses the ratio form exactly for this reason.
LME_SCALAR_THRESHOLD="${LME_SCALAR_THRESHOLD:-2/3}"
LME_SCALAR_RECONCILE_ATTRIBUTE="${LME_SCALAR_RECONCILE_ATTRIBUTE:-1}"
LME_SCALAR_RECONCILE_SCOPE="${LME_SCALAR_RECONCILE_SCOPE:-1}"
LME_SCALAR_RECONCILE_SUBJECT="${LME_SCALAR_RECONCILE_SUBJECT:-1}"
# Measured effect: ZERO cells in every configuration (the prompt already emits 'user' almost
# always). Off by default here; flip on only to re-measure.
LME_SCALAR_CANONICAL_SELF="${LME_SCALAR_CANONICAL_SELF:-0}"
# A one-item plumbing smoke may validly abstain at the consistency gate. Canonical/full builds keep
# this on so promotion refuses a corpus that produced no durable scalar assertion/View at all.
LME_REQUIRE_SCALAR_OUTPUT="${LME_REQUIRE_SCALAR_OUTPUT:-1}"
# ---- Temporal grounding repair ----
# DEFAULT OFF as of 2026-07-27. This used to be a MANDATORY post-build step, on the belief that
# graphiti-core ignored reference_time. It does not (graphiti.py:1110 sets valid_at=reference_time);
# the real defect was menhir's claim query never PROJECTING reference_time, fixed in menhir 27d9bad.
# Worse, the repair keys off e.session_id, which graphiti-written EpisodicNodes do not carry, so it
# silently skipped 62% of the 2026-07-22 corpus while reporting success -- masking the very bug it
# was compensating for. Leave this OFF so a build's dates are the INGEST's dates and a regression is
# visible. Verify instead of repairing: menhir scripts/_verify_valid_at_repair.py checks every
# episode against the fixture, including the ones this script cannot map. Set to 1 only to repair a
# corpus built by a menhir that predates 27d9bad.
LME_BACKFILL_DATES="${LME_BACKFILL_DATES:-0}"

# ---- Dataset and limits ----
LME_NS_PREFIX="${LME_NS_PREFIX:-lme-}"               # namespace prefix for all LME graphs (promote scope filter)
LME_DATASET="${LME_DATASET:-xiaowu0162/longmemeval}"
LONGMEMEVAL_VARIANT="${LONGMEMEVAL_VARIANT:-oracle}"
LME_LIMIT="${LME_LIMIT:-30}"
LME_RECALL_LIMIT="${LME_RECALL_LIMIT:-10}"            # recall top-k (menhir_client default is 10)
LME_PER_TYPE="${LME_PER_TYPE:-15}"                    # stratified sample size per question type
LME_BRIEF_PER_TYPE="${LME_BRIEF_PER_TYPE:-10}"       # brief_ab: questions per type
LME_BRIEF_TYPES="${LME_BRIEF_TYPES:-temporal-reasoning,knowledge-update,multi-session}"  # brief_ab: categories
LME_ENTROPY_PER_TYPE="${LME_ENTROPY_PER_TYPE:-15}"   # entropy: questions per type (GPT-free, run wide)
LME_ENTROPY_K="${LME_ENTROPY_K:-20}"                 # entropy: retriever depth for the delivered walk

# ---- Backup and results ----
LME_BACKUP_DIR="${LME_BACKUP_DIR:-C:/Users/thron/menhir-lme-backup}"
LME_RESULTS_DIR="${LME_RESULTS_DIR:-${BENCH_DIR}/results/lme-ingest}"

# Resume manifest, scoped by container name so a differently-named build (e.g. a
# throwaway smoke build with LME_NEO4J_NAME=menhir-lme-smoke) never shares dedup state
# with the canonical persistent graph's manifest.json, and a later full build against
# the real graph does not see the smoke items as "already done" (or vice versa).
# Default container name keeps the original unscoped filename for backward compat with
# existing manifests/status.sh history.
if [ "${LME_NEO4J_NAME}" = "menhir-lme-neo4j" ]; then
  LME_MANIFEST_PATH="${LME_MANIFEST_PATH:-${LME_RESULTS_DIR}/manifest.json}"
else
  LME_MANIFEST_PATH="${LME_MANIFEST_PATH:-${LME_RESULTS_DIR}/manifest-${LME_NEO4J_NAME}.json}"
fi

# Date-backfill revert snapshot, scoped the same way and for the same reason: two
# differently-named graphs both running backfill-dates must not clobber each other's
# only way to undo the mutation.
if [ "${LME_NEO4J_NAME}" = "menhir-lme-neo4j" ]; then
  LME_REVERT_SNAPSHOT_PATH="${LME_REVERT_SNAPSHOT_PATH:-${LME_RESULTS_DIR}/date-backfill-revert.json}"
else
  LME_REVERT_SNAPSHOT_PATH="${LME_REVERT_SNAPSHOT_PATH:-${LME_RESULTS_DIR}/date-backfill-revert-${LME_NEO4J_NAME}.json}"
fi

# ---- Virtual environment bins (derived from paths above) ----
MENHIR_MAIN_PY="${MENHIR_MAIN_PY:-${MENHIR_MAIN}/.venv/Scripts/python.exe}"
_MENHIR_MAIN_BIN_DEFAULT="${MENHIR_MAIN}/.venv/Scripts/menhir"
[ -f "${_MENHIR_MAIN_BIN_DEFAULT}.exe" ] && _MENHIR_MAIN_BIN_DEFAULT="${_MENHIR_MAIN_BIN_DEFAULT}.exe"
MENHIR_MAIN_BIN="${MENHIR_MAIN_BIN:-${_MENHIR_MAIN_BIN_DEFAULT}}"
MENHIR_FRONTIER_PY="${MENHIR_FRONTIER_PY:-${MENHIR_FRONTIER}/.venv/Scripts/python.exe}"
_MENHIR_FRONTIER_BIN_DEFAULT="${MENHIR_FRONTIER}/.venv/Scripts/menhir"
[ -f "${_MENHIR_FRONTIER_BIN_DEFAULT}.exe" ] && _MENHIR_FRONTIER_BIN_DEFAULT="${_MENHIR_FRONTIER_BIN_DEFAULT}.exe"
MENHIR_FRONTIER_BIN="${MENHIR_FRONTIER_BIN:-${_MENHIR_FRONTIER_BIN_DEFAULT}}"
BENCH_PY="${BENCH_PY:-${BENCH_DIR}/.venv/Scripts/python.exe}"
_BENCH_BIN_DEFAULT="${BENCH_DIR}/.venv/Scripts/archolith-bench"
[ -f "${_BENCH_BIN_DEFAULT}.exe" ] && _BENCH_BIN_DEFAULT="${_BENCH_BIN_DEFAULT}.exe"
BENCH_BIN="${BENCH_BIN:-${_BENCH_BIN_DEFAULT}}"
