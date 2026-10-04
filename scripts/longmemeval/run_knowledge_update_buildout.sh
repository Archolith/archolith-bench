#!/usr/bin/env bash
# Build a FRESH graph with ONLY the 78 knowledge-update items from LongMemEval (oracle variant),
# then run recall+QA scoring against it. This is the canonical current-code scalar reingest:
# scalar consolidation and TurnEvidence are mandatory, and recall explicitly exercises View authority.
#
# Two arms are intentionally pinned so the post-binding measurement has a contemporaneous control:
#   baseline  = current production gate (3/3, no identity reconciliation)
#   candidate = true 2/3 + attribute/scope/subject identity reconciliation
#
# Usage:
#   LME_KU_RUN_ID=scalar-current-baseline-20260728 LME_KU_ARM=baseline \
#     ./scripts/longmemeval/run_knowledge_update_buildout.sh
#   LME_KU_RUN_ID=scalar-current-candidate-20260728 LME_KU_ARM=candidate \
#     ./scripts/longmemeval/run_knowledge_update_buildout.sh
#   LME_KU_RUN_ID=... LME_KU_ARM=baseline \
#     ./scripts/longmemeval/run_knowledge_update_buildout.sh --preflight-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
ARCH_DIR="${BENCH_DIR%/*}"

log(){ printf '[lme-ku-buildout] %s %s\n' "$(date '+%F %H:%M:%S')" "$*" >&2; }
die(){ printf '[lme-ku-buildout] ERROR: %s\n' "$*" >&2; exit 1; }

MODE="${1:-run}"
[ "${MODE}" = "run" ] || [ "${MODE}" = "--preflight-only" ] ||
  die "unknown argument: ${MODE} (expected --preflight-only or no argument)"

RUN_ID="${LME_KU_RUN_ID:-}"
[ -n "${RUN_ID}" ] || die "LME_KU_RUN_ID is required; never reuse the historical default"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_ID contains unsafe characters"

ARM="${LME_KU_ARM:-}"
case "${ARM}" in
  baseline)
    export LME_SCALAR_THRESHOLD="1.0"
    export LME_SCALAR_RECONCILE_ATTRIBUTE="0"
    export LME_SCALAR_RECONCILE_SCOPE="0"
    export LME_SCALAR_RECONCILE_SUBJECT="0"
    ;;
  candidate)
    export LME_SCALAR_THRESHOLD="2/3"
    export LME_SCALAR_RECONCILE_ATTRIBUTE="1"
    export LME_SCALAR_RECONCILE_SCOPE="1"
    export LME_SCALAR_RECONCILE_SUBJECT="1"
    ;;
  *)
    die "LME_KU_ARM must be exactly 'baseline' or 'candidate'"
    ;;
esac

# This dedicated wrapper must never silently fall back to the ordinary non-scalar LME defaults.
export LME_COUNTER_STATE_ENABLED="${LME_COUNTER_STATE_ENABLED:-${MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED:-0}}"
export LME_SCALAR_STATE_ENABLED="1"
export LME_SCALAR_HISTORY_ENABLED="1"
export LME_REQUIRE_TURN_EVIDENCE="1"
export LME_BACKFILL_DATES="0"
export LME_SCALAR_VIEW_AUTHORITY_ENABLED="${LME_SCALAR_VIEW_AUTHORITY_ENABLED:-1}"
export LME_CONSOLIDATION_AUDIT_ENABLED="${LME_CONSOLIDATION_AUDIT_ENABLED:-1}"
export LME_RECALL_AUDIT_ENABLED="${LME_RECALL_AUDIT_ENABLED:-1}"
export MENHIR_PERSONAL_MEMORY_CONSOLIDATION_AUDIT_ENABLED="${LME_CONSOLIDATION_AUDIT_ENABLED}"
# These are Menhir runtime settings rather than benchmark-arm settings. Pin their effective
# values to Menhir's production defaults when callers omit them, then record them below so a
# buildout cannot accidentally claim Event History or deterministic routing after the fact.
export MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED="${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED:-0}"
for lane in "${LME_COUNTER_STATE_ENABLED}" "${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED}"; do
  case "${lane}" in 0|1) ;; *) die "processing lane settings must be 0 or 1" ;; esac
done
export MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_AUTHORITY_ENABLED="${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_AUTHORITY_ENABLED:-0}"
export MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_ROUTER="${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_ROUTER:-0}"
export MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_SHADOW="${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_SHADOW:-0}"
export LME_KU_ALLOW_RESUME="${LME_KU_ALLOW_RESUME:-0}"
export LME_KU_KEEP_NEO4J_UP="${LME_KU_KEEP_NEO4J_UP:-0}"
export LME_KU_ALLOW_DIRTY="${LME_KU_ALLOW_DIRTY:-0}"
export LME_INGEST_CONCURRENCY="${LME_KU_INGEST_CONCURRENCY:-2}"
export LME_KU_CHECKPOINT_ITEMS="${LME_KU_CHECKPOINT_ITEMS:-0}"
[ "${LME_SCALAR_HISTORY_ENABLED}" = "1" ] ||
  die "scalar history must be enabled for the knowledge-update buildout"
case "${LME_INGEST_CONCURRENCY}" in
  1|2|3|4) ;;
  *) die "LME_KU_INGEST_CONCURRENCY must be an integer from 1 through 4" ;;
esac
case "${LME_KU_CHECKPOINT_ITEMS}" in
  *[!0-9]*) die "LME_KU_CHECKPOINT_ITEMS must be a non-negative integer" ;;
esac
export LME_INGEST_STOP_AFTER_ITEMS="${LME_KU_CHECKPOINT_ITEMS}"
if [ "${LME_KU_ALLOW_RESUME}" = "1" ]; then
  export LME_REQUIRE_FRESH="0"
else
  export LME_REQUIRE_FRESH="1"
fi

export BENCH_DIR ARCH_DIR
export MENHIR_MAIN="${MENHIR_MAIN:-${ARCH_DIR}/menhir}"
export MENHIR_FRONTIER="${MENHIR_FRONTIER:-${MENHIR_MAIN}}"

# Fixture: pre-exported 78 knowledge-update items (oracle variant)
FIXTURE_PATH="${BENCH_DIR}/fixtures/longmemeval/knowledge_update_subset.json"
[ -f "${FIXTURE_PATH}" ] || die "fixture not found: ${FIXTURE_PATH}"
export LME_FIXTURE_PATH="${FIXTURE_PATH}"

# Container/volume/ports — isolated from the persistent graph
export LME_NEO4J_NAME="menhir-lme-${RUN_ID}"
export LME_NEO4J_VOL="menhir-lme-data-${RUN_ID}"
export LME_BOLT="${LME_BOLT:-7694}"
export LME_HTTP="${LME_HTTP:-7481}"
export LME_PORT_BUILD="${LME_PORT_BUILD:-8124}"
export LME_RESULTS_DIR="${BENCH_DIR}/results/lme-ku-buildout/${RUN_ID}"
export LME_MANIFEST_PATH="${LME_RESULTS_DIR}/manifest.json"
export LME_REVERT_SNAPSHOT_PATH="${LME_RESULTS_DIR}/date-backfill-revert.json"
export LONGMEMEVAL_VARIANT=oracle
# The fix under test: gpt-4o-mini (temp=0 is in the code)
export LME_EXTRACT_MODEL="${LME_EXTRACT_MODEL:-gpt-4o-mini}"

set -a
source "${SCRIPT_DIR}/config.sh"
set +a

command -v docker >/dev/null 2>&1 || die "docker not on PATH"
[ -f "${BENCH_PY}" ] || die "archolith-bench venv python missing: ${BENCH_PY}"
[ -f "${BENCH_BIN}" ] || die "archolith-bench CLI missing: ${BENCH_BIN}"
[ -f "${MENHIR_MAIN_PY}" ] || die "Menhir venv python missing: ${MENHIR_MAIN_PY}"
[ -f "${MENHIR_MAIN_BIN}" ] || die "Menhir CLI missing: ${MENHIR_MAIN_BIN}"
[ -d "${MENHIR_MAIN}/.git" ] || die "Menhir checkout missing: ${MENHIR_MAIN}"
docker info >/dev/null 2>&1 || die "Docker daemon is not available"
docker image inspect "${LME_NEO4J_IMAGE}" >/dev/null 2>&1 ||
  die "pinned Neo4j image is not available locally: ${LME_NEO4J_IMAGE}"

MENHIR_COMMIT="$(git -C "${MENHIR_MAIN}" rev-parse HEAD 2>/dev/null || echo unknown)"
BENCH_COMMIT="$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
MENHIR_DIRTY="$([ -n "$(git -C "${MENHIR_MAIN}" status --porcelain --untracked-files=no)" ] && echo true || echo false)"
BENCH_DIRTY="$([ -n "$(git -C "${BENCH_DIR}" status --porcelain --untracked-files=no)" ] && echo true || echo false)"
if [ "${LME_KU_ALLOW_DIRTY}" != "1" ] &&
   { [ "${MENHIR_DIRTY}" = "true" ] || [ "${BENCH_DIRTY}" = "true" ]; }; then
  die "tracked code is dirty (menhir=${MENHIR_DIRTY}, bench=${BENCH_DIRTY}); commit it or set LME_KU_ALLOW_DIRTY=1 for a non-canonical run"
fi

read -r KU_COUNT FIXTURE_SHA256 < <(
  "${BENCH_PY}" - "${FIXTURE_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
rows = json.loads(raw)
if not isinstance(rows, list) or not rows:
    raise SystemExit("fixture must be a non-empty JSON list")
ids = [str(row.get("question_id") or "") for row in rows]
if any(not question_id for question_id in ids) or len(set(ids)) != len(ids):
    raise SystemExit("fixture question_id values must be non-empty and unique")
if any(row.get("question_type") != "knowledge-update" for row in rows):
    raise SystemExit("fixture contains a non-knowledge-update item")
if any(not row.get("question") or not row.get("answer") for row in rows):
    raise SystemExit("fixture contains an item without a question or answer")
print(len(rows), hashlib.sha256(raw).hexdigest())
PY
)
# Windows Python writes CRLF even under Git Bash; `read` removes LF but leaves CR in the final
# field. Strip it before interpolating the digest into JSON provenance.
FIXTURE_SHA256="${FIXTURE_SHA256//$'\r'/}"
# Pass the fixture identity down so build_graph.sh can stamp its own phase record with the
# data it ingested, not just the settings it ingested under.
export LME_FIXTURE_SHA256="${FIXTURE_SHA256}"
export LME_FIXTURE_COUNT="${KU_COUNT}"
[ "${KU_COUNT}" = "78" ] || die "expected the frozen 78-item fixture, found ${KU_COUNT}"
[ "${LME_KU_CHECKPOINT_ITEMS}" -lt "${KU_COUNT}" ] ||
  die "LME_KU_CHECKPOINT_ITEMS must be smaller than the full fixture"
log "fixture: ${KU_COUNT} knowledge-update items; sha256=${FIXTURE_SHA256}"

# ---- guard against clobbering ----
if lme_container_exists "${LME_NEO4J_NAME}"; then
  if [ "${LME_KU_ALLOW_RESUME:-0}" = "1" ]; then
    log "RESUME: reusing existing container ${LME_NEO4J_NAME}"
    docker start "${LME_NEO4J_NAME}" 2>/dev/null || true
  else
    die "container ${LME_NEO4J_NAME} already exists; set LME_KU_ALLOW_RESUME=1 to resume or pick a new LME_KU_RUN_ID"
  fi
fi

if [ "${LME_KU_ALLOW_RESUME}" != "1" ]; then
  docker volume inspect "${LME_NEO4J_VOL}" >/dev/null 2>&1 &&
    die "fresh run refused: volume already exists: ${LME_NEO4J_VOL}"
  [ ! -e "${LME_RESULTS_DIR}" ] ||
    die "fresh run refused: results directory already exists: ${LME_RESULTS_DIR}"
  "${BENCH_PY}" - "${LME_BOLT}" "${LME_HTTP}" "${LME_PORT_BUILD}" "${LME_KU_RECALL_PORT:-8125}" <<'PY'
import socket
import sys

for raw in sys.argv[1:]:
    port = int(raw)
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"required port is unavailable: {port}: {exc}") from exc
PY
fi

OPENAI_KEY="$("${MENHIR_MAIN_PY}" - "${BENCH_DIR}/.env" OPENAI_API_KEY <<'PY'
import sys
from dotenv import dotenv_values

print(dotenv_values(sys.argv[1]).get(sys.argv[2], ""))
PY
)"
[ -n "${OPENAI_KEY}" ] || die "no OPENAI_API_KEY in ${BENCH_DIR}/.env"

log "preflight PASS: arm=${ARM} run=${RUN_ID} fresh=${LME_REQUIRE_FRESH} counter=${LME_COUNTER_STATE_ENABLED} scalar=1 scalar_history=${LME_SCALAR_HISTORY_ENABLED} threshold=${LME_SCALAR_THRESHOLD} reconcile=${LME_SCALAR_RECONCILE_ATTRIBUTE}/${LME_SCALAR_RECONCILE_SCOPE}/${LME_SCALAR_RECONCILE_SUBJECT} authority=${LME_SCALAR_VIEW_AUTHORITY_ENABLED} event_history=${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED}/${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_AUTHORITY_ENABLED} deterministic_router_shadow=${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_ROUTER}/${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_SHADOW} turn_evidence=${LME_REQUIRE_TURN_EVIDENCE} audits=${LME_CONSOLIDATION_AUDIT_ENABLED}/${LME_RECALL_AUDIT_ENABLED} ingest_concurrency=${LME_INGEST_CONCURRENCY} checkpoint_items=${LME_KU_CHECKPOINT_ITEMS}"

if [ "${MODE}" = "--preflight-only" ]; then
  exit 0
fi

mkdir -p "${LME_RESULTS_DIR}"

PROVENANCE_PATH="${LME_RESULTS_DIR}/run_provenance.json"
ATTEMPT_RECORD="${LME_RESULTS_DIR}/.attempt-record.json"

PROVENANCE_TOOL="${SCRIPT_DIR}/lib/run_provenance.py"
[ -f "${PROVENANCE_TOOL}" ] || die "provenance helper missing: ${PROVENANCE_TOOL}"

# The effective settings of a phase at the moment it starts -- not the values captured during
# preflight. The checkpoint continuation deliberately flips LME_REQUIRE_FRESH and
# LME_INGEST_STOP_AFTER_ITEMS, so a single preflight snapshot describes neither build phase
# correctly. Commits are re-read for the same reason: a resume can run against a different
# checkout than the attempt that ingested the earlier items. Everything a reader would need to
# reproduce the phase goes in, including the segmentation mode, the failure policy and the
# consolidation budget -- an arm is not defined by its threshold alone.
phase_settings(){
  printf -- '--setting\nmenhir_commit=%s\n--setting\nbench_commit=%s\n' \
    "$(git -C "${MENHIR_MAIN}" rev-parse HEAD 2>/dev/null || echo unknown)" \
    "$(git -C "${BENCH_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  printf -- '--setting\nmenhir_dirty=%s\n--setting\nbench_dirty=%s\n' \
    "$([ -n "$(git -C "${MENHIR_MAIN}" status --porcelain --untracked-files=no)" ] && echo true || echo false)" \
    "$([ -n "$(git -C "${BENCH_DIR}" status --porcelain --untracked-files=no)" ] && echo true || echo false)"
  printf -- '--setting\nrequire_fresh=%s\n--setting\ningest_stop_after_items=%s\n' \
    "${LME_REQUIRE_FRESH}" "${LME_INGEST_STOP_AFTER_ITEMS}"
  printf -- '--setting\ningest_concurrency=%s\n--setting\nsegmentation=%s\n' \
    "${LME_INGEST_CONCURRENCY}" "${LME_SEGMENTATION}"
  printf -- '--setting\nstrict_failure_policy=%s\n--setting\nallow_resume=%s\n' \
    "zero-failed-episodes-per-namespace" "${LME_KU_ALLOW_RESUME}"
  printf -- '--setting\ncounter_state_enabled=%s\n' "${LME_COUNTER_STATE_ENABLED}"
  printf -- '--setting\nscalar_threshold=%s\n--setting\nscalar_consolidation_k=%s\n' \
    "${LME_SCALAR_THRESHOLD}" "${LME_SCALAR_CONSOLIDATION_K}"
  printf -- '--setting\nscalar_consolidation_call_budget=%s\n--setting\nextract_model=%s\n' \
    "${LME_SCALAR_CALL_BUDGET}" "${LME_EXTRACT_MODEL}"
  printf -- '--setting\nscalar_reconcile_attribute=%s\n--setting\nscalar_reconcile_scope=%s\n' \
    "${LME_SCALAR_RECONCILE_ATTRIBUTE}" "${LME_SCALAR_RECONCILE_SCOPE}"
  printf -- '--setting\nscalar_reconcile_subject=%s\n--setting\nscalar_view_authority_enabled=%s\n' \
    "${LME_SCALAR_RECONCILE_SUBJECT}" "${LME_SCALAR_VIEW_AUTHORITY_ENABLED}"
  printf -- '--setting\nscalar_history_enabled=%s\n' \
    "${LME_SCALAR_HISTORY_ENABLED}"
  printf -- '--setting\nevent_history_enabled=%s\n--setting\nevent_history_authority_enabled=%s\n' \
    "${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED}" \
    "${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_AUTHORITY_ENABLED}"
  printf -- '--setting\nscalar_deterministic_router=%s\n--setting\nscalar_deterministic_shadow=%s\n' \
    "${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_ROUTER}" \
    "${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_SHADOW}"
  printf -- '--setting\nconsolidation_audit_enabled=%s\n--setting\nrecall_audit_enabled=%s\n' \
    "${LME_CONSOLIDATION_AUDIT_ENABLED}" "${LME_RECALL_AUDIT_ENABLED}"
  printf -- '--setting\nturn_evidence_required=%s\n--setting\nbackfill_dates=%s\n' \
    "${LME_REQUIRE_TURN_EVIDENCE}" "${LME_BACKFILL_DATES}"
}

# record_phase_start <phase-name>: appends a phase with the manifest state it inherits, so a
# resumed phase is distinguishable from a fresh one, plus its full effective settings.
record_phase_start(){
  local phase="$1"
  # One argument per line, read into an array: a settings value may contain spaces, so
  # unquoted word splitting would corrupt it.
  local settings=()
  local line
  while IFS= read -r line; do settings+=("${line}"); done < <(phase_settings)
  "${BENCH_PY}" "${PROVENANCE_TOOL}" phase-start "${PROVENANCE_PATH}" \
    --phase "${phase}" --manifest "${LME_MANIFEST_PATH}" "${settings[@]}"
}

# record_phase_end <phase-name> <completed|failed> [extra key=value ...]
record_phase_end(){
  local phase="$1" status="$2"
  shift 2
  local extra=()
  local pair
  for pair in "$@"; do extra+=(--setting "${pair}"); done
  "${BENCH_PY}" "${PROVENANCE_TOOL}" phase-end "${PROVENANCE_PATH}" \
    --phase "${phase}" --status "${status}" "${extra[@]}"
}

# ---- record run provenance ----
# Fingerprint the declared surface BEFORE writing the attempt record, so `begin` can treat the
# digest as immutable identity for a canonical run and refuse a resume that would mix two code
# states. Empty on failure, which `run_provenance.py` reads as "absent" rather than as a digest
# of "" -- fingerprinting stays non-fatal, consistent with the attach call further down.
SURFACE_DIGEST="$("${BENCH_PY}" "${SCRIPT_DIR}/lib/bench_surface.py" fingerprint --digest-only 2>/dev/null || true)"
cat > "${ATTEMPT_RECORD}" <<EOF
{
  "run_id": "${RUN_ID}",
  "arm": "${ARM}",
  "surface_digest": "${SURFACE_DIGEST}",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "resumed": $([ "${LME_KU_ALLOW_RESUME}" = "1" ] && echo true || echo false),
  "menhir_commit": "${MENHIR_COMMIT}",
  "menhir_dirty": ${MENHIR_DIRTY},
  "bench_commit": "${BENCH_COMMIT}",
  "bench_dirty": ${BENCH_DIRTY},
  "extract_model": "${LME_EXTRACT_MODEL}",
  "fixture": "${FIXTURE_PATH}",
  "fixture_count": ${KU_COUNT},
  "fixture_sha256": "${FIXTURE_SHA256}",
  "dataset": "${LME_DATASET}",
  "variant": "${LONGMEMEVAL_VARIANT}",
  "namespace_prefix": "${LME_NS_PREFIX}",
  "container": "${LME_NEO4J_NAME}",
  "volume": "${LME_NEO4J_VOL}",
  "bolt_port": ${LME_BOLT},
  "menhir_port": ${LME_PORT_BUILD},
  "recall_port": ${LME_KU_RECALL_PORT:-8125},
  "neo4j_heap": "${LME_NEO4J_HEAP}",
  "ingest_concurrency": ${LME_INGEST_CONCURRENCY},
  "checkpoint_items": ${LME_KU_CHECKPOINT_ITEMS},
  "require_fresh": ${LME_REQUIRE_FRESH},
  "counter_state_enabled": ${LME_COUNTER_STATE_ENABLED},
  "scalar_state_enabled": ${LME_SCALAR_STATE_ENABLED},
  "scalar_history_enabled": ${LME_SCALAR_HISTORY_ENABLED},
  "scalar_threshold": "${LME_SCALAR_THRESHOLD}",
  "scalar_reconcile_attribute": ${LME_SCALAR_RECONCILE_ATTRIBUTE},
  "scalar_reconcile_scope": ${LME_SCALAR_RECONCILE_SCOPE},
  "scalar_reconcile_subject": ${LME_SCALAR_RECONCILE_SUBJECT},
  "scalar_view_authority_enabled": ${LME_SCALAR_VIEW_AUTHORITY_ENABLED},
  "event_history_enabled": ${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_ENABLED},
  "event_history_authority_enabled": ${MENHIR_PERSONAL_MEMORY_EVENT_HISTORY_AUTHORITY_ENABLED},
  "scalar_deterministic_router": ${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_ROUTER},
  "scalar_deterministic_shadow": ${MENHIR_PERSONAL_MEMORY_SCALAR_DETERMINISTIC_SHADOW},
  "consolidation_audit_enabled": ${LME_CONSOLIDATION_AUDIT_ENABLED},
  "recall_audit_enabled": ${LME_RECALL_AUDIT_ENABLED},
  "turn_evidence_required": ${LME_REQUIRE_TURN_EVIDENCE},
  "backfill_dates": ${LME_BACKFILL_DATES},
  "purpose": "fresh current-code scalar knowledge-update ${ARM} arm"
}
EOF

# Merge this attempt into any provenance a previous attempt left behind. A resume must not
# `cat >` over that file: the earlier attempt ingested real items under its own commit and
# clock, and overwriting the record replaces what happened with what this attempt intends.
# `begin` also refuses a resume whose run_id/arm/fixture/container identity disagrees with
# the record, and marks phases the killed attempt left open as interrupted.
PROVENANCE_BEGIN_ARGS=()
[ "${LME_NONCANONICAL}" = "1" ] && PROVENANCE_BEGIN_ARGS+=("--noncanonical")
"${BENCH_PY}" "${PROVENANCE_TOOL}" begin "${PROVENANCE_PATH}" "${ATTEMPT_RECORD}" \
  "${PROVENANCE_BEGIN_ARGS[@]}" ||
  die "provenance refused this attempt (identity, commit, or surface mismatch); see above"
rm -f "${ATTEMPT_RECORD}"
log "provenance recorded: ${PROVENANCE_PATH}"

# Hash the declared benchmark-affecting surface into this attempt's record. Commit SHAs
# cannot do this job: a dirty tree or an untracked file leaves the SHA unchanged while the
# executed code differs, which is how the assistant-turn gate in claim_segmenter.py ended
# up attributed to a `chore: track untracked scripts` commit. Content hashes make a later
# `bench_surface.py blame A B` name the files that actually differ.
#
# Non-fatal: a fingerprint is evidence about a run, not a precondition for it. Refusing to
# launch here would trade a recorded gap for no run at all. The gap is recorded instead,
# and `blame` reports a run with no fingerprint as not-comparable rather than guessing.
SURFACE_TOOL="${SCRIPT_DIR}/lib/bench_surface.py"
if [ -f "${SURFACE_TOOL}" ]; then
  if "${BENCH_PY}" "${SURFACE_TOOL}" attach "${PROVENANCE_PATH}"; then
    :
  else
    log "WARNING: surface fingerprint failed; this run will not be attributable by file"
  fi
else
  log "WARNING: ${SURFACE_TOOL} missing; no surface fingerprint recorded"
fi

MENHIR_PID=""
cleanup_all(){
  [ -n "${MENHIR_PID}" ] && kill "${MENHIR_PID}" 2>/dev/null || true
  if [ "${LME_KU_KEEP_NEO4J_UP}" != "1" ] &&
     lme_container_running "${LME_NEO4J_NAME}"; then
    log "stopping ${LME_NEO4J_NAME}; container and volume are preserved for inspection"
    docker stop "${LME_NEO4J_NAME}" >/dev/null || true
  fi
}
trap cleanup_all EXIT

# ---- Phase 1: build graph ----
log "======== PHASE 1: BUILD GRAPH (${KU_COUNT} items) ========"
record_phase_start build-graph
"${SCRIPT_DIR}/build_graph.sh" "${KU_COUNT}"
record_phase_end build-graph completed
if [ "${LME_KU_CHECKPOINT_ITEMS}" -gt 0 ]; then
  CHECKPOINT_READY="${LME_RESULTS_DIR}/checkpoint-${LME_KU_CHECKPOINT_ITEMS}-ready.json"
  CHECKPOINT_CONTINUE="${LME_RESULTS_DIR}/continue-after-checkpoint-${LME_KU_CHECKPOINT_ITEMS}"
  "${BENCH_PY}" - "${LME_MANIFEST_PATH}" "${CHECKPOINT_READY}" \
    "${LME_KU_CHECKPOINT_ITEMS}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])
expected = int(sys.argv[3])
rows = json.loads(manifest_path.read_text(encoding="utf-8"))
if len(rows) != expected:
    raise SystemExit(
        f"checkpoint refused: manifest has {len(rows)} rows, expected {expected}"
    )
ready_path.write_text(
    json.dumps(
        {
            "status": "ready-for-independent-review",
            "manifest_items": len(rows),
            "question_ids": [row["question_id"] for row in rows],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
  log "HARD CHECKPOINT reached after ${LME_KU_CHECKPOINT_ITEMS} items."
  log "Waiting for independent review marker: ${CHECKPOINT_CONTINUE}"
  while [ ! -f "${CHECKPOINT_CONTINUE}" ]; do
    sleep 5
  done
  log "checkpoint review marker found; resuming remaining $((KU_COUNT-LME_KU_CHECKPOINT_ITEMS)) items"
  export LME_INGEST_STOP_AFTER_ITEMS=0
  export LME_REQUIRE_FRESH=0
  # A distinct phase with its own effective settings: this build runs with the freshness
  # requirement and the item cap both lifted, which the phase-1 record does not describe.
  record_phase_start build-graph-checkpoint-continuation
  "${SCRIPT_DIR}/build_graph.sh" "${KU_COUNT}"
  record_phase_end build-graph-checkpoint-continuation completed
fi
log "build complete"
"${BENCH_PY}" "${SCRIPT_DIR}/lib/summarize_llm_usage.py" \
  "${LME_RESULTS_DIR}/mcp_telemetry.db" \
  --run-id "${RUN_ID}" \
  --output "${LME_RESULTS_DIR}/ingest_llm_usage.json" >/dev/null
log "ingest LLM usage: ${LME_RESULTS_DIR}/ingest_llm_usage.json"

# ---- Phase 2: recall+QA scoring ----
log "======== PHASE 2: RECALL+QA SCORING ========"
record_phase_start recall-qa

# Start a fresh menhir for recall (the build menhir exited after build_graph.sh)
RECALL_PORT="${LME_KU_RECALL_PORT:-8125}"
RECALL_URL="http://localhost:${RECALL_PORT}"
EMPTY_ENV="$(mktemp)"

log "starting recall menhir on port ${RECALL_PORT}..."
( export ENV_FILE="${EMPTY_ENV}" \
    LONGMEMEVAL_VARIANT=oracle \
    MENHIR_LOG_DIR="${LME_RESULTS_DIR}/menhir-recall-logs" \
    MENHIR_BENCHMARK_MODE=1 MENHIR_API_HOST=127.0.0.1 MENHIR_API_PORT="${RECALL_PORT}" \
    MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED="${LME_SCALAR_STATE_ENABLED}" \
    MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED}" \
    MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED="${LME_SCALAR_VIEW_AUTHORITY_ENABLED}" \
    MENHIR_PERSONAL_MEMORY_RECALL_AUDIT_ENABLED="${LME_RECALL_AUDIT_ENABLED}" \
    MENHIR_BENCH_RESULTS_ROOT="${BENCH_DIR}/results" \
    MENHIR_BENCH_ACTIVE_RUN_ID="$(basename "$LME_RESULTS_DIR")" \
    MENHIR_MCP_TELEMETRY_DB="${LME_RESULTS_DIR}/mcp_telemetry.db" \
    OTEL_SDK_DISABLED=true LANGFUSE_TRACING_ENABLED=false LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" LANGFUSE_HOST="" \
    MENHIR_OPERATOR_KEY="" MENHIR_AGENT_KEY="" MENHIR_READONLY_KEY="" MENHIR_API_KEY="" \
    NEO4J_URI="bolt://localhost:${LME_BOLT}" NEO4J_USER=neo4j NEO4J_PASSWORD="${LME_NEO4J_PW}" NEO4J_DATABASE=neo4j \
    GRAPHITI_LLM_PROVIDER=openai MEMORY_GRAPHITI_PROVIDER=openai LLM_CHAT_PROVIDER=openai MEMORY_CHAT_PROVIDER=openai \
    GRAPHITI_RERANKER_PROVIDER=openai MEMORY_GRAPHITI_RERANKER_PROVIDER=openai \
    GRAPHITI_EMBED_PROVIDER=openai MEMORY_GRAPHITI_EMBED_PROVIDER=openai \
    OPENAI_API_KEY="${OPENAI_KEY}" OPENAI_CHAT_MODEL="${LME_EXTRACT_MODEL}" OPENAI_EMBED_MODEL="${LME_EMBED_MODEL}" \
    PYTHONPATH="${MENHIR_MAIN}/src"; \
  mkdir -p "${LME_RESULTS_DIR}/menhir-recall-logs"; \
  cd "${MENHIR_MAIN}" && "${MENHIR_MAIN_BIN}" serve --port "${RECALL_PORT}" ) \
  >"${LME_RESULTS_DIR}/recall_serve.log" 2>&1 &
MENHIR_PID=$!
for _ in $(seq 1 90); do curl -sf "${RECALL_URL}/api/health" >/dev/null 2>&1 && break; sleep 2; done
curl -sf "${RECALL_URL}/api/health" >/dev/null 2>&1 || die "recall menhir not healthy"
log "recall menhir healthy"

# Run the harness — score all knowledge-update items
log "scoring ${KU_COUNT} knowledge-update items..."
RECALL_OUT="${LME_RESULTS_DIR}/harness_recall"
mkdir -p "${RECALL_OUT}"

if ( cd "${BENCH_DIR}" && \
  UPSTREAM_BASE_URL="https://api.openai.com/v1" UPSTREAM_API_KEY="${OPENAI_KEY}" BENCHMARK_MODEL="gpt-4o" \
  OPENAI_API_KEY="${OPENAI_KEY}" LONGMEMEVAL_VARIANT=oracle \
  "${BENCH_BIN}" harness longmemeval-menhir --menhir-url "${RECALL_URL}" --recall-only \
    --model "${LME_ANSWER_MODEL}" --scorer "${LME_SCORER}" --judge-model "${LME_JUDGE_MODEL}" \
    --format markdown --subset knowledge-update \
    --output-dir "${RECALL_OUT}" --out "${RECALL_OUT}/results.md" \
    --limit "${KU_COUNT}" --recall-limit 10 --resume ) \
  >>"${LME_RESULTS_DIR}/recall_harness.log" 2>&1; then
  HARNESS_EXIT=0
else
  HARNESS_EXIT=$?
fi

log "harness exit code: ${HARNESS_EXIT}"

if [ -f "${RECALL_OUT}/results.md" ]; then
  log "======== RESULTS ========"
  cat "${RECALL_OUT}/results.md" >&2
fi

# Emit the score as machine-readable per-arm evidence alongside the rendered markdown.
# results.md is the only score artifact this harness used to write, which is why every
# LEDGER.md number was read by a human and retyped -- and why six runs with real scores
# never reached the scoreboard. score.json is computed from the harness checkpoint (one line
# per arm+task), so `ledger.py validate` can check the recorded number against the run's own
# evidence instead of taking it on trust.
#
# Non-fatal and deliberately after the provenance/`complete` bookkeeping: this is a
# derivation from artifacts already on disk, so it can be re-run later with
# `score_extract.py <run-dir>` and must never be able to fail a run that already scored.
SCORE_TOOL="${SCRIPT_DIR}/lib/score_extract.py"
if [ -f "${SCORE_TOOL}" ]; then
  if "${BENCH_PY}" "${SCORE_TOOL}" "${LME_RESULTS_DIR}" >&2; then
    log "per-arm scores: ${LME_RESULTS_DIR}/score.json"
  else
    log "WARNING: score.json not written; add this run's ledger row by hand"
  fi
fi

# Record completion
record_phase_end recall-qa "$([ "${HARNESS_EXIT}" = "0" ] && echo completed || echo failed)" \
  "harness_exit=${HARNESS_EXIT}"
"${BENCH_PY}" "${PROVENANCE_TOOL}" complete "${PROVENANCE_PATH}" --harness-exit "${HARNESS_EXIT}"

kill "${MENHIR_PID}" 2>/dev/null || true
MENHIR_PID=""
HARNESS_CHECKPOINT="$(find "${RECALL_OUT}" -maxdepth 1 -type f -name '.checkpoint_*.jsonl' -print -quit)"
if [ -n "${HARNESS_CHECKPOINT}" ]; then
  JUDGE_USAGE_ARG=()
  if [ "${LME_SCORER}" = "llm-judge" ]; then
    JUDGE_USAGE_ARG=(--require-judge-usage)
  fi
  "${BENCH_PY}" "${SCRIPT_DIR}/lib/summarize_llm_usage.py" \
    "${LME_RESULTS_DIR}/mcp_telemetry.db" \
    --run-id "${RUN_ID}" \
    --harness-checkpoint "${HARNESS_CHECKPOINT}" \
    "${JUDGE_USAGE_ARG[@]}" \
    --output "${LME_RESULTS_DIR}/run_llm_usage.json" >/dev/null
  log "combined LLM usage: ${LME_RESULTS_DIR}/run_llm_usage.json"
else
  # No checkpoint means the harness produced nothing to aggregate -- but the ingest usage is
  # already in the telemetry DB and build_graph.sh has written the ingest-only summary. Saying
  # "not written" here would be wrong, and discarding the ingest cost because the scoring half
  # failed is how the expensive half of a run becomes unknown.
  if [ -f "${LME_RESULTS_DIR}/run_llm_usage.json" ]; then
    log "no harness checkpoint; keeping the ingest-only usage summary from build_graph.sh"
  else
    log "WARNING: no harness checkpoint and no ingest usage summary; cost is unrecorded"
  fi
fi
log "done. Results in ${LME_RESULTS_DIR}"
if [ "${LME_KU_KEEP_NEO4J_UP}" = "1" ]; then
  log "Neo4j container ${LME_NEO4J_NAME} left up by request."
else
  log "Neo4j container ${LME_NEO4J_NAME} will be stopped; data remains in ${LME_NEO4J_VOL}."
fi
exit "${HARNESS_EXIT}"
