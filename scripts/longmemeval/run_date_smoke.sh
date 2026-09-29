#!/usr/bin/env bash
# One-item build that PROVES menhir's ingest backdating works, before committing to a full re-ingest.
#
# WHY THIS EXISTS
# The 2026-07-22 scalar-ku corpus had 1707/2862 episodes stamped with their INGESTION time instead of
# the conversation time. Two defects stacked:
#   1. menhir's claim_pending_episode projected queued_at but NOT reference_time, so the enrichment
#      handoff silently fell back to queue time (fixed: menhir 27d9bad).
#   2. build_graph.sh then ran backfill_dates.py unconditionally to "repair" it -- but that script
#      keys off e.session_id, which graphiti-written EpisodicNodes do not carry, so it skipped
#      exactly the 62% that were broken and printed success anyway.
# The repair masked the bug. So this smoke runs with LME_BACKFILL_DATES=0: the dates in the resulting
# graph are the INGEST's dates, unassisted, and the verifier checks them against the fixture.
#
# THE ITEM: cc5ded98, a knowledge-update question whose answer depends entirely on valid_at ordering.
#   session 0  2023-05-25  "about an hour each day"
#   session 1  2023-05-27  "about two hours each day"   <- the correct answer
# In the old corpus 42 of its 45 episodes carried a wrong date. If backdating is broken, both
# sessions land at ingest time and "which is current" is decided by import order, not world time.
#
# ISOLATION: its own container, volume, ports and manifest. It cannot touch the canonical LME graph
# (menhir-lme-neo4j / menhir-lme-data) or the 7701 snapshot.
#
# Usage:  bash scripts/longmemeval/run_date_smoke.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LME_NEO4J_NAME="${LME_NEO4J_NAME:-menhir-lme-datesmoke}"
export LME_NEO4J_VOL="${LME_NEO4J_VOL:-menhir-lme-datesmoke-data}"
export LME_BOLT="${LME_BOLT:-7703}"
export LME_HTTP="${LME_HTTP:-7477}"
export LME_PORT_BUILD="${LME_PORT_BUILD:-8122}"
# 45 episodes need nothing like the canonical 2G, and Docker's VM ceiling (~3.8 GiB total, shared
# with menhir-full and any graph already up) is the binding constraint -- overshooting it here is
# what wedged the daemon on the first attempt.
export LME_NEO4J_HEAP="${LME_NEO4J_HEAP:-512M}"

export LME_FIXTURE_PATH="${LME_FIXTURE_PATH:-${HERE}/fixtures/date-smoke-cc5ded98.json}"
# Overridable so the same harness can run the 1-item date check or a wider multi-type smoke
# (fixtures/multi-smoke-12.json, 2 items x 6 question types) without a second script.
export LME_LIMIT="${LME_LIMIT:-1}"

# The whole point: no post-hoc date repair. If the graph's dates are right, the INGEST made them right.
export LME_BACKFILL_DATES=0

# Scalar path on, at the reconciling config the full run will use.
export LME_SCALAR_STATE_ENABLED="${LME_SCALAR_STATE_ENABLED:-1}"
# A single item may legitimately abstain at the consistency gate; do not fail the build on it.
export LME_REQUIRE_SCALAR_OUTPUT="${LME_REQUIRE_SCALAR_OUTPUT:-0}"
export LME_REQUIRE_TURN_EVIDENCE="${LME_REQUIRE_TURN_EVIDENCE:-1}"

# Keep the smoke's manifest and results away from the canonical ones.
export LME_MANIFEST_PATH="${LME_MANIFEST_PATH:-${HERE}/results/manifest-datesmoke.json}"

log(){ printf '[date-smoke] %s\n' "$*" >&2; }

# Sourced here, not only before the verify step below: the --clean block needs LME_RESULTS_DIR to
# remove this graph's provenance record, and under `set -u` an unsourced config aborted --clean
# AFTER it had already destroyed the container and volume. config.sh is side-effect free and every
# value it sets is ${VAR:-default}, so the exports above still win.
source "${HERE}/config.sh"

if [ "${1:-}" = "--clean" ] || [ "${LME_SMOKE_CLEAN:-0}" = "1" ]; then
  log "removing previous smoke container/volume/manifest..."
  docker rm -f "${LME_NEO4J_NAME}" >/dev/null 2>&1 || true
  docker volume rm "${LME_NEO4J_VOL}" >/dev/null 2>&1 || true
  rm -f "${LME_MANIFEST_PATH}" || true
  # The graph provenance too. Provenance is append-only and records failed attempts on purpose,
  # so a refused run leaves a record carrying that attempt's surface_digest. Keeping it across an
  # explicit --clean made the next run look like a mixed-code RESUME of a graph that no longer
  # exists, and the canonical surface gate refused it -- correctly, on stale evidence.
  rm -f "${LME_RESULTS_DIR}/graph-provenance-${LME_NEO4J_NAME}.json" || true
fi

log "container=${LME_NEO4J_NAME} bolt=${LME_BOLT} fixture=${LME_FIXTURE_PATH}"
log "BACKFILL_DATES=${LME_BACKFILL_DATES} (0 = dates come from the ingest, unassisted)"

# Pass the limit through -- build_graph.sh takes it positionally, and hardcoding 1 here silently
# ignored LME_LIMIT (the 12-item multi-type smoke ran a single item before this was caught).
bash "${HERE}/build_graph.sh" "${LME_LIMIT}"

log "build done. verifying valid_at against the fixture (writes nothing)..."
source "${HERE}/config.sh"
LME_BOLT="${LME_BOLT}" "${MENHIR_MAIN}/.venv/Scripts/python.exe" \
  "${MENHIR_MAIN}/scripts/_verify_valid_at_repair.py" \
  --uri "bolt://127.0.0.1:${LME_BOLT}" --password "${LME_NEO4J_PW}"
