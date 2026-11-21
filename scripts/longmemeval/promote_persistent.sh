#!/usr/bin/env bash
# Promote one build's LongMemEval memories from SESSION to PERSISTENT scope.
#
# WHY: menhir stamps freshly-extracted Entity nodes as scope=SESSION, and only the
# consolidation pass (lifecycle_service) promotes reinforced facts to PERSISTENT — but
# consolidation is off in benchmark mode AND it DELETES low-sharpness one-off facts, which
# is most LME answer entities. So without this step every benchmark memory stays SESSION.
#
# SESSION-scoped nodes are filtered out of recall unless include_session=True
# (recall_service.py:937), which means build_context returns empty briefs and any plain
# (non-session) recall path sees nothing. LME facts are durable knowledge the system should
# always recall, so we write them as regular memories: a blanket SESSION->PERSISTENT flip.
#
# This is idempotent and non-destructive (unlike consolidation -- nothing is deleted).
# Existing recall-only A/B results are unaffected: they already use include_session=True,
# so those nodes were always visible; this only ALSO makes them visible to build_context
# and plain recall.
#
# It promotes exactly the namespaces in this build's manifest, through Menhir's
# promote_namespace_scope backend op (operator tier), so the build's Menhir must still be serving.
#
# Usage: MENHIR_URL=<serving menhir> [LME_MANIFEST_PATH=<manifest>] promote_persistent.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

log(){ printf '[lme-promote] %s\n' "$*" >&2; }

[ -n "${MENHIR_URL:-}" ] || { log "ERROR: MENHIR_URL is not set (run from build_graph.sh)"; exit 1; }
[ -f "${LME_MANIFEST_PATH}" ] || { log "ERROR: manifest not found: ${LME_MANIFEST_PATH}"; exit 1; }

log "promoting SESSION -> PERSISTENT for the namespaces in ${LME_MANIFEST_PATH}..."
"${BENCH_PY}" "$(dirname "${BASH_SOURCE[0]}")/lib/promote.py" "${LME_MANIFEST_PATH}" "${MENHIR_URL}"
log "done. LME memories are now regular (PERSISTENT) scope."
