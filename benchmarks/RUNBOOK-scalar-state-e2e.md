# Runbook: ScalarStateView (Piece C) e2e (throwaway menhir, scheduler ON)

How to run the `menhir-scalar-state` harness against a **throwaway** menhir + real LLM. This exercises
Menhir's REAL typed-scalar path end to end: `/api/memory` ingest -> background consolidation scheduler
(`enable_scalar_state=True`) -> `TypedScalarPerceptionService` -> `:TypedAssertion` log -> materialized
`scalar_state` Views, verified over bolt. Successor to the CLOSED lexical sidecar line
(`scripts/longmemeval/analysis/TYPED-VALUE-ARM.md`).

## Why this differs from the phase3 runbook

The numeric-counter phase3 benchmark triggers consolidation explicitly over HTTP (`POST /api/phase3/run`)
and runs menhir with the scheduler OFF. Typed-scalar perception runs ONLY inside the SCHEDULED
consolidation job gated by `MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED=1` -- the HTTP route does NOT
enable it. So this harness needs the **background scheduler ON** (benchmark mode OFF) with a short
consolidation interval; it ingests episodes, waits for a tick to materialize the Views, then verifies.

## Safety rules (non-negotiable)

- **Never touch the real menhir on `:8090`** or the real (prod) menhir Neo4j on **bolt `:7687`**. Use a
  throwaway only. The harness `ScalarBoltReader` refuses any bolt URI on port 7687 or the known prod host.
- Activation is fresh-only: `activate_scalar_state()` refuses a store with legacy/unstamped nodes. The
  script below uses a FRESH ephemeral Neo4j (no named volume) so every run is a clean store.

## Throwaway Neo4j via compose

`docker-compose.throwaway-neo4j.yml` is the committed, repeatable version of the ephemeral
Neo4j these runs need, so it does not have to be reconstructed by hand each time:

```bash
docker compose -f docker-compose.throwaway-neo4j.yml up -d
# ... run the harness ...
docker compose -f docker-compose.throwaway-neo4j.yml down -v
```

It publishes bolt on **7688** and the browser on **7475**, both bound to `127.0.0.1`, matching
the harness default `--neo4j-uri bolt://localhost:7688`. Storage is `tmpfs` with no named
volume, so `down -v` leaves nothing and a stale graph cannot survive into the next run.

If those ports are already taken -- a hand-run throwaway such as `menhir-neo4j-test` uses the
same ones -- either stop that container or publish elsewhere:

```bash
BENCH_NEO4J_BOLT_PORT=7699 BENCH_NEO4J_HTTP_PORT=7476   docker compose -f docker-compose.throwaway-neo4j.yml -p benchverify up -d
export ARCHOLITH_BENCH_ALLOW_PORTS=7699   # the guard allow-lists ports too
```

The port override does not weaken the safety rule above, and the guard is now stricter than
a port check: `assert_allowed_target` resolves the host **and the port a driver will actually
dial** (an omitted port is not "no port" -- the neo4j driver uses 7687), then requires both to
be allow-listed. 7687 and 8090 are reserved as real services and no opt-in can reach them, so
publishing a throwaway on a non-default port needs `ARCHOLITH_BENCH_ALLOW_PORTS`.

Verified 2026-09-08 on Docker 25.0.3 / Compose v2.24.6 with `neo4j:5.26-community`: healthy in
about 20s, bolt serving queries through the `neo4j` driver, and `down -v` leaving no residual
container, network, or volume.

## The one command (recommended)

`scripts/run_scalar_state_e2e.sh` automates the whole loop the same way the LongMemEval scripts do
(reuses `scripts/longmemeval/config.sh` for paths/venv/OpenAI key): fresh ephemeral Neo4j -> menhir
serve with the scheduler ON + scalar flag ON + short interval -> health wait -> run the harness ->
teardown.

```bash
cd <archolith-bench>
bash scripts/run_scalar_state_e2e.sh            # runs + tears down; evidence -> results/menhir_scalar_state_e2e.md
bash scripts/run_scalar_state_e2e.sh --keep     # leave menhir + Neo4j up afterwards for debugging
```

Overridable knobs (env): `SS_PORT` (menhir, 8098), `SS_BOLT` (7691), `SS_NEO4J_NAME`, `SS_INTERVAL_S`
(5), `SS_CHAT_MODEL` (gpt-4o-mini), `SS_MAX_WAIT_S` (120), `SS_OUT`, `SS_FORMAT`. Anything after `--`
is passed through to the harness. The script installs the `.[menhir-scalar]` bolt extra if missing.

Offline smoke of the harness itself (no menhir/Neo4j/LLM):

```bash
.venv/Scripts/python.exe -m archolith_bench.cli harness menhir-scalar-state --offline-fixture x
```

## Manual equivalent (only if you need to drive the pieces yourself)

The env profile the script exports -- the load-bearing bit is `MENHIR_BENCHMARK_MODE=0` (benchmark mode
disables the scheduler) plus the scalar flag:

```
MENHIR_BENCHMARK_MODE=0
MENHIR_PERSONAL_MEMORY_CONSOLIDATION_ENABLED=1
MENHIR_PERSONAL_MEMORY_SCALAR_STATE_ENABLED=1
MENHIR_PERSONAL_MEMORY_SCALAR_STATE_PERCEIVER_VERSION=v1
MENHIR_PERSONAL_MEMORY_CONSOLIDATION_INTERVAL_S=5
MENHIR_PERSONAL_MEMORY_CHAT_MODEL=gpt-4o-mini   # or a local Qwen via the OpenAI-compatible chat model env
```

then `menhir serve` against the throwaway bolt and run `archolith-bench harness menhir-scalar-state
--menhir-url ... --neo4j-uri ... --confirm-menhir-reset`. Auth: menhir disables bearer auth when no
`MENHIR_*_KEY` is set. The health log should NOT say `MENHIR_BENCHMARK_MODE=1 -- scheduler ... disabled`.

## 5. What "good" looks like

- Verdict PASS: >=1 current `scalar_state` View in the seeded namespace; zero duplicate current View
  keys; zero duplicate (subject, slot) pairs; every assertion `evidence_tier='agent'`; zero wrong-
  namespace Views; zero default-silo leak.
- Per-case: `view` fixtures MATCH their (kind, value) slot; the advisory case stays advisory/abstain
  (no concrete View); the non-scalar control produces nothing. Perception is stochastic, so a single
  `view` MISS is reported, not a hard failure -- the verdict gates on the structural invariants.

## Teardown

The script tears down automatically (stops menhir, `docker rm -f` the ephemeral Neo4j) unless `--keep`
was passed. To clean up a `--keep` run manually:

```bash
kill "$(pgrep -f 'menhir.*serve.*8098')" 2>/dev/null   # stop the throwaway server
docker rm -f menhir-scalar-neo4j                        # remove the ephemeral throwaway Neo4j
# verify: :8098 down, real :8090 + prod bolt :7687 untouched
```
