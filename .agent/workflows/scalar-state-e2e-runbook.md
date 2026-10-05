# Scalar State E2E Runbook

End-to-end workflow for running a canonical scalar evidence benchmark with immutable provenance.

## Prerequisites

- Docker with Neo4j 5.26+ image
- `menhir` and `archolith-bench` venvs installed
- OpenAI API key in `menhir/.env`
- All code committed (canonical builds refuse untracked source files)

## Canonical Run

```bash
cd archolith-bench/scripts/longmemeval

# 1. Preflight (no spend)
LME_KU_RUN_ID=scalar-<arm>-<date> LME_KU_ARM=<baseline|candidate> \
  ./run_knowledge_update_buildout.sh --preflight-only

# 2. Build with checkpoint (optional: inspect after first 2 items)
LME_KU_RUN_ID=scalar-<arm>-<date> LME_KU_ARM=<baseline|candidate> \
  LME_KU_CHECKPOINT_ITEMS=2 \
  ./run_knowledge_update_buildout.sh

# 3. Validate the run
./lme.sh validate --expected 78

# 4. Inspect the acceptance report
cat results/lme-ingest/acceptance-report.json | python -m json.tool
```

## Provenance Gates

The build enforces these gates automatically:

| Gate | What it checks | Override |
|------|---------------|----------|
| Commit immutability | menhir_commit and bench_commit match across resumes | `LME_NONCANONICAL=1` |
| Untracked source files | No .py files in src/ or scripts/ shadow committed code | `LME_NONCANONICAL=1` |
| Fresh graph | Volume, container, and manifest must not pre-exist | `LME_REQUIRE_FRESH=0` |
| Zero failed episodes | Every namespace must complete enrichment | None (strict) |

## Per-run Telemetry

Both ingest and recall menhir processes write to a run-local SQLite sidecar at
`$LME_RESULTS_DIR/mcp_telemetry.db`. This DB contains:

- `lifecycle_events`: enrichment phases, scalar perception, gate/commit receipts
- Vote receipts correlated by `source_key`/`assertion_id` to graph assertions

The dashboard's `ScalarTaskReader` reads this path for audit provenance display.

## Acceptance Report

`lme.sh validate --expected N` (required) checks:

- Manifest cardinality matches expected item count
- Zero failed episodes
- All namespaces start with the configured prefix
- All attempts used the same menhir and bench commits
- Telemetry DB exists and has lifecycle events
- No interrupted phases

## Scalar History

When `MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED=1`:

- `scalar_history` Views materialize alongside `scalar_state` Views
- The dashboard shows both projections; history renders source-time-ordered delta entries
- The recall advisory lane surfaces history for PREVIOUS_VALUE/COMPARISON queries
- For delta-only slots (like the postcard regression), state abstains and history provides the
  ordered evidence — the answer is the latest delta (25), not the sum (42)

## Development Runs

For iterative development where code changes between resumes:

```bash
LME_NONCANONICAL=1 LME_KU_ALLOW_DIRTY=1 \
  LME_KU_RUN_ID=dev-probe-<date> LME_KU_ARM=candidate \
  ./run_knowledge_update_buildout.sh
```

All output is labelled `noncanonical`. These runs are NOT canonical benchmark evidence.
