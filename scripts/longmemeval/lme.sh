#!/usr/bin/env bash
# LongMemEval dispatcher for the tools that have no replacement yet (streamlined-harness P5).
#
# Usage:
#   lme.sh backfill-dates [--dry-run]  # repair valid_at from real session dates (no re-ingest)
#   lme.sh status [watch [secs]]       # read-only progress tracker
#   lme.sh validate --expected N       # final acceptance report (provenance + manifest + telemetry)
#   lme.sh -h|--help                   # show this help
#
# Builds:  python -m archolith_bench.build --build builds/<x>.yaml  (promotion runs inside it)
# Retry:   python -m archolith_bench.build --build builds/<x>.yaml --retry-failed [--drain-timeout S]
# Recall:  python -m archolith_bench.staged run [--serve <menhir tree> --graph builds/<x>.yaml]|rerun|compare
# Research tools run directly: analysis/answer_matrix.sh, analysis/msc_sweep.sh, analysis/brief_ab.sh,
# analysis/entropy.sh, analysis/lib/retrieval_quality.py (presence / ir-gate).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

usage(){
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^#//; s/^ //'
  exit "${1:-0}"
}

COMMAND="${1:-}"
[ -n "$COMMAND" ] || usage 1

case "$COMMAND" in
  backfill-dates)
    [ "${2:-}" = "--dry-run" ] && export DRY_RUN=1
    LME_BOLT="${LME_BOLT}" LME_NEO4J_PW="${LME_NEO4J_PW}" LME_NS_PREFIX="${LME_NS_PREFIX}" \
      LME_REVERT_SNAPSHOT="${LME_REVERT_SNAPSHOT:-${LME_REVERT_SNAPSHOT_PATH}}" \
      "${MENHIR_FRONTIER_PY}" "${_LONGMEMEVAL_DIR}/lib/backfill_dates.py"
    ;;
  status)
    "${_LONGMEMEVAL_DIR}/status.sh" "${2:-}" "${3:-}"
    ;;
  validate)
    # Canonical acceptance: without an expected count any non-empty manifest would pass.
    if [ "${2:-}" != "--expected" ] || ! [[ "${3:-}" =~ ^[1-9][0-9]*$ ]]; then
      echo "lme.sh validate requires --expected N (a positive item count, e.g. --expected 500)" >&2
      exit 2
    fi
    VALIDATE_ARGS=(
      "${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json"
      "${LME_MANIFEST_PATH}"
      --telemetry-db "${LME_RESULTS_DIR}/mcp_telemetry.db"
      --require-fresh-clean
    )
    VALIDATE_ARGS+=(--expected-items "${3}")
    VALIDATE_OUTPUT="${LME_RESULTS_DIR}/acceptance-report.json"
    VALIDATE_ARGS+=(--output "${VALIDATE_OUTPUT}")
    "${BENCH_PY}" "${_LONGMEMEVAL_DIR}/lib/validate_run.py" "${VALIDATE_ARGS[@]}"
    ;;
  -h|--help)
    usage 0
    ;;
  *)
    echo "unknown command: $COMMAND" >&2
    usage 1
    ;;
esac
