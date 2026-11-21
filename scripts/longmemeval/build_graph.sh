#!/usr/bin/env bash
# Build a PERSISTENT menhir/Neo4j with LongMemEval haystacks pre-ingested under
# stable namespaces, so recall-only A/B (main vs frontier) can run without re-ingesting.
# Neo4j stays up after; only the temp menhir (used for ingestion) is stopped.
# Usage: build_graph.sh [limit]
#
# python -m archolith_bench.build no longer runs this: it resolves the same settings in Python
# (archolith_bench/build/settings.py). This wrapper stays for the KU buildout (moving to the fixtures
# repo, plan P8) and the date/suburbs smokes.
#
# The build itself (checks, Neo4j, provenance, Menhir serve, ingest, promotion) runs in
# archolith_bench/build/pipeline.py. This wrapper resolves config.sh and passes its settings on
# stdin: config.sh exports only a few of them, and children such as ingest.py take defaults from
# the environment, so the exported set must stay exactly what it was.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

LIMIT="${1:-30}"

for _var in ${!LME_@} ${!MENHIR_@} ${!BENCH_@} LONGMEMEVAL_VARIANT; do
  printf '%s\0%s\0' "${_var}" "${!_var-}"
done | "${BENCH_PY}" "${_LONGMEMEVAL_DIR}/../../archolith_bench/build/pipeline.py" "${LIMIT}"
