#!/usr/bin/env bash
# Build and verify the isolated Rachel/Chicago/suburbs LongMemEval ingestion fixture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
ARCH_DIR="${BENCH_DIR%/*}"
RUN_ID="${LME_FIXTURE_RUN_ID:-suburbs-fix-20260716-v1}"

log(){ printf '[lme-suburbs-fixture] %s\n' "$*" >&2; }
die(){ printf '[lme-suburbs-fixture] ERROR: %s\n' "$*" >&2; exit 1; }

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "LME_FIXTURE_RUN_ID contains unsafe characters"

export BENCH_DIR ARCH_DIR
export MENHIR_MAIN="${MENHIR_MAIN:-${ARCH_DIR}/menhir}"
export MENHIR_FRONTIER="${MENHIR_FRONTIER:-${MENHIR_MAIN}}"
export LME_FIXTURE_PATH="${BENCH_DIR}/fixtures/longmemeval/menhir_suburbs_extraction_regression.json"
export LME_NEO4J_NAME="${LME_NEO4J_NAME:-menhir-lme-${RUN_ID}}"
export LME_NEO4J_VOL="${LME_NEO4J_VOL:-menhir-lme-data-${RUN_ID}}"
export LME_BOLT="${LME_BOLT:-7693}"
export LME_HTTP="${LME_HTTP:-7480}"
export LME_PORT_BUILD="${LME_PORT_BUILD:-8123}"
export LME_RESULTS_DIR="${LME_RESULTS_DIR:-${BENCH_DIR}/results/lme-fixtures/${RUN_ID}}"
export LME_MANIFEST_PATH="${LME_MANIFEST_PATH:-${LME_RESULTS_DIR}/manifest.json}"
export LME_REVERT_SNAPSHOT_PATH="${LME_REVERT_SNAPSHOT_PATH:-${LME_RESULTS_DIR}/date-backfill-revert.json}"
export LONGMEMEVAL_VARIANT=oracle

set -a
source "${SCRIPT_DIR}/config.sh"
set +a

command -v docker >/dev/null 2>&1 || die "docker is not on PATH"
[ -f "${BENCH_PY}" ] || die "archolith-bench venv python missing: ${BENCH_PY}"
[ -d "${MENHIR_MAIN}/.git" ] || die "Menhir checkout missing: ${MENHIR_MAIN}"

"${BENCH_PY}" "${SCRIPT_DIR}/lib/verify_suburbs_fixture.py" \
  --fixture "${LME_FIXTURE_PATH}" \
  --menhir-root "${MENHIR_MAIN}" \
  --preflight-only >/dev/null
log "Menhir fix commit and fixture contract verified"

if [ "${LME_FIXTURE_ALLOW_RESUME:-0}" != "1" ]; then
  if lme_container_exists "${LME_NEO4J_NAME}"; then
    die "container ${LME_NEO4J_NAME} already exists; choose a new LME_FIXTURE_RUN_ID or set LME_FIXTURE_ALLOW_RESUME=1"
  fi
  if docker volume inspect "${LME_NEO4J_VOL}" >/dev/null 2>&1; then
    die "volume ${LME_NEO4J_VOL} already exists; choose a new LME_FIXTURE_RUN_ID or set LME_FIXTURE_ALLOW_RESUME=1"
  fi
  [ ! -e "${LME_MANIFEST_PATH}" ] \
    || die "manifest already exists: ${LME_MANIFEST_PATH}; choose a new run id"
fi

log "building one-item fixture in fresh graph ${LME_NEO4J_NAME}"
"${SCRIPT_DIR}/build_graph.sh" 1

log "verifying extracted current value and stale-value retirement"
"${BENCH_PY}" "${SCRIPT_DIR}/lib/verify_suburbs_fixture.py" \
  --fixture "${LME_FIXTURE_PATH}" \
  --menhir-root "${MENHIR_MAIN}" \
  --container "${LME_NEO4J_NAME}" \
  --output "${LME_RESULTS_DIR}/verification.json"

log "fixture passed; graph retained for inspection"
log "container=${LME_NEO4J_NAME} volume=${LME_NEO4J_VOL} results=${LME_RESULTS_DIR}"
