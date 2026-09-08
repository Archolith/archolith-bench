# Handoff — LME Scalar-History Candidate Reingest

## Compact Summary

Run candidate-only LME after scalar-history and provenance hardening; approve at 2 items, then finish 78 on one immutable code pair.

**Date:** 2026-07-29
**Status:** BLOCKED ON IMPLEMENTATION AND NO-SPEND PREFLIGHT
**Menhir repo:** `C:\Users\thron\IdeaProjects\projects\archolith\menhir`
**Bench repo:** `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench`
**Implementation plan:** `C:\Users\thron\IdeaProjects\projects\archolith\.agent\plans\menhir-scalar-history-projection-plan.md`

## Objective

Produce one trustworthy, fresh, candidate-only LongMemEval knowledge-update run under the current
typed-scalar code, including the new scalar-history projection. The run must prove not only that
ingest finished, but that:

- source/world time is preserved from the LME fixture;
- typed assertions are correctly gated and bound;
- delta-only histories remain advisory rather than becoming false absolute totals;
- every View can be traced to assertions, TurnEvidence, and source episodes;
- replay and lifecycle behavior are deterministic;
- the dashboard reads the same run-local evidence used by the run;
- all 78 knowledge-update items were built and scored on one immutable code/configuration pair.

There is no baseline arm. Use `LME_KU_ARM=candidate`, which pins 2-of-3 scalar agreement and
attribute/scope/subject reconciliation.

## What “Good Results” Means

Separate infrastructure validity from product quality.

A run is **valid** when its graph, manifest, settings, code identity, time semantics, telemetry, and
provenance satisfy every gate in this handoff. A valid run may still have a disappointing recall or
QA score. Report that result and its failed tasks honestly; do not discard it in favor of an older
or mixed-code run.

A run is **invalid** if it mixes commits, reuses a non-fresh graph, has residual failed episodes,
loses source time, lacks evidence chains, has incomplete manifest/provenance, or cannot correlate
the dashboard to run-local telemetry. Its score is not comparable evidence.

## Current Evidence and Quarantine

### v3 — do not resume

`results/lme-ku-buildout/scalar-current-candidate-v3-20260728` is quarantined. It is partial,
mixed-code, and includes a consolidated residual failed episode. Never resume it, score it, or use
it as the current result.

### v4 — diagnostic replay only

`results/lme-ku-buildout/scalar-current-candidate-v4-20260729` is useful but not canonical:

- manifest: 48 of 78 items;
- latest phase stopped fail-closed on a residual failed episode;
- Menhir and bench commits changed across attempts;
- source-time repair was applied and receipt-bearing;
- the graph contains useful typed assertions and scalar Views.

Preserve its stopped container, volume, logs, snapshots, and provenance. Use it only for LLM-free
scalar-history replay and dashboard validation. Do not continue it into a claimed final run.

## Hard Blockers Before Paid Work

Do not start the fresh 78-item wrapper until all of these are committed and tested:

1. The scalar-history implementation plan is complete.
2. A focused postcard fixture/selective run exists.
3. A durable LLM-free scalar-history replay command exists and is listed in
   `menhir/.agent/scripts-index.md`.
4. The dashboard displays scalar history separately from scalar state and shows the full evidence
   chain and source/world time.
5. `run_knowledge_update_buildout.sh` assigns a per-run telemetry database to both ingest and recall,
   preferably `${LME_RESULTS_DIR}/mcp_telemetry.db`, and preserves it in the results.
6. Canonical resume refuses a changed Menhir commit, bench commit, fixture SHA, effective settings,
   container, or volume. A code change requires a new run ID and fresh volume.
7. A final machine-readable provenance/acceptance validator exists.
8. Focused and full Menhir tests plus relevant archolith-bench tests pass.

The current wrapper records commit changes but can still resume across them. Recording drift is not
the same as preventing it; fix this before treating a run as canonical.

## Required Provenance Contract

For every scalar-history entry that appears in recall or the dashboard, retain and verify:

```text
scalar_history -[:HISTORY_ENTRY]-> TypedAssertion
TurnEvidence    -[:FOUNDS]->       TypedAssertion
Episodic        -[:ADMITTED_ON]->  TurnEvidence
Episodic        -[:MENTIONS]->     scalar_history
new View        -[:SUPERSEDES]->   old View, when versioned
```

For current scalar state, preserve the existing contributor meanings:

- `CURRENT_ANCHOR` — the absolute anchor used by the head;
- `CONTRIBUTED_TO` — applied post-anchor deltas;
- `SUPERSEDED_ANCHOR` — excluded prior anchors/pre-anchor values.

`HISTORY_ENTRY` is not an authority edge. A delta-only history must never pass the current-head
foundation gate merely because its assertion has user evidence.

Every run must preserve:

- original fixture SHA-256;
- Menhir and bench commit IDs;
- tracked-worktree cleanliness and explicit review of untracked files;
- effective scalar, evidence, audit, model, concurrency, and checkpoint settings;
- container and volume identity;
- attempt and phase start/end records;
- manifest rows and failure counts;
- source/world `valid_at` and separate learned/recorded timestamps;
- per-run consolidation and recall telemetry;
- replay receipts and final validation report;
- snapshot/backup identity;
- ledger entry for completed, failed, or manually stopped runs.

## Stage 0 — No-Spend Preflight

### 0.1 Verify repository state

Run from PowerShell:

```powershell
$archolith = 'C:\Users\thron\IdeaProjects\projects\archolith'
$menhir = Join-Path $archolith 'menhir'
$bench = Join-Path $archolith 'archolith-bench'

git -C $menhir status --short
git -C $bench status --short
git -C $menhir rev-parse HEAD
git -C $bench rev-parse HEAD
git -C $menhir log -1 --oneline
git -C $bench log -1 --oneline
```

Expected:

- no tracked modifications;
- every untracked file understood and confirmed unable to affect imports/configuration;
- scalar-history and harness-hardening commits present;
- the exact commit IDs copied into the run notes before launch.

Do not use `LME_KU_ALLOW_DIRTY=1` for canonical evidence.

### 0.2 Verify fixture identity

The canonical fixture is:

```text
C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\fixtures\longmemeval\knowledge_update_subset.json
```

Expected:

- 78 unique knowledge-update items;
- SHA-256
  `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.

Verify:

```powershell
Get-FileHash (Join-Path $bench 'fixtures\longmemeval\knowledge_update_subset.json') -Algorithm SHA256
```

If an intentional fixture change occurs, commit it, document the new hash, and do not compare the
run as though the fixture were unchanged.

### 0.3 Verify tests

Run the focused scalar-history, projection lifecycle, provenance, recall, and dashboard tests first,
then the full Menhir unit suite and relevant bench suite. Run online tests only against the
ephemeral test Neo4j instance.

Minimum areas:

- history builder/key/signature;
- scalar fold `no_anchor`;
- `HISTORY_ENTRY` atomic redraw;
- TurnEvidence/FOUNDS/ADMITTED_ON chain;
- source-time rendering;
- merge/unmerge/deletion/binding repair;
- recall authority exclusion;
- postcard focused fixture;
- dashboard APIs/UI;
- canonical-resume immutability and per-run telemetry.

Record exact commands, pass counts, skipped counts, and commit IDs in the run notes.

### 0.4 Verify local infrastructure

- Docker daemon healthy.
- Pinned `neo4j:5.26-community` image available locally.
- No production Neo4j URI/password in the environment.
- Selected Bolt, HTTP, build-Menhir, recall-Menhir, and dashboard ports are free.
- Existing v3/v4 containers and volumes remain preserved but need not be running.
- OpenAI key is available only to the benchmark process.

The usual defaults are Bolt 7694, Neo4j HTTP 7481, build Menhir 8124, recall Menhir 8125, and
dashboard 8200. Choose other free ports for the fresh run if any are occupied.

## Stage 1 — LLM-Free Replay on Preserved v4

Start only the preserved v4 Neo4j container. Do not start its old Menhir processes and do not invoke
extraction/consolidation.

Run the new scalar-history replay command against the v4 test Bolt port with:

- an explicit namespace first: `lme-01493427`;
- a receipt written under the v4 results directory;
- no LLM credentials in the replay environment if practical;
- dry-run before write;
- a second write replay to prove idempotence.

Exact acceptance for the postcard namespace:

- assertions are `delta 17` at `2023-08-11` and `delta 25` at `2023-11-30`;
- both assertions have their gate and source evidence;
- one current slot-keyed `scalar_history` View exists;
- exactly two ordered `HISTORY_ENTRY` edges exist;
- both chains reach TurnEvidence and source episodes;
- source/world time is displayed, not 2026 ingest/repair time;
- `scalar_state` remains absent;
- a `no_anchor` abstention receipt exists;
- no value of 42 is materialized;
- replay performs zero LLM calls;
- the second replay changes no semantic View, assertion, or contributor count.

Then replay all eligible v4 namespaces. This is a migration/replay validation only; it does not make
v4 canonical.

## Stage 2 — Focused Fresh Postcard Run

Use the focused fixture or question-ID selector in a brand-new isolated container and volume.

Require:

- original LME turns and dates;
- zero residual failed episodes;
- expected TurnEvidence and assertion votes;
- exactly two current delta assertions, 17 then 25;
- one current scalar-history View and no scalar-state View;
- complete namespace-local evidence chains;
- recall answer 25, not 42;
- structured recall labels it advisory/latest recorded delta, not absolute total;
- dashboard shows the same values, times, and provenance;
- run-local telemetry contains both consolidation and recall receipts.

If this fails, stop. Preserve the results/container/volume, append the ledger, fix the code, rerun
tests, and start a new focused run ID. Do not “repair” the run into canonical evidence after a code
change.

## Stage 3 — Fresh Candidate Run with Two-Item Checkpoint

Use a new descriptive run ID, for example:

```text
scalar-history-candidate-20260729
```

Do not reuse it if a code/configuration change forces a restart.

From PowerShell:

```powershell
$env:LME_KU_RUN_ID = 'scalar-history-candidate-20260729'
$env:LME_KU_ARM = 'candidate'
$env:LME_KU_ALLOW_RESUME = '0'
$env:LME_KU_CHECKPOINT_ITEMS = '2'
$env:LME_KU_INGEST_CONCURRENCY = '2'
$env:LME_KU_KEEP_NEO4J_UP = '1'
$env:LME_KU_ALLOW_DIRTY = '0'

# Set only if defaults are occupied:
# $env:LME_BOLT = '7696'
# $env:LME_HTTP = '7483'
# $env:LME_PORT_BUILD = '8128'
# $env:LME_KU_RECALL_PORT = '8129'

Set-Location $bench
& 'C:\Program Files\Git\bin\bash.exe' `
  './scripts/longmemeval/run_knowledge_update_buildout.sh' `
  '--preflight-only'
```

Read the preflight output. It must say candidate, threshold `2/3`, all three reconciliations enabled,
TurnEvidence required, audits enabled, concurrency 2, checkpoint 2, and fresh graph required.

Then launch the same command without `--preflight-only`. Foreground is safest. If it must run in the
background, redirect stdout/stderr to files inside the new results directory and record the PID and
exact command in run provenance.

Do not pipe progress output through `tail`; benchmark progress is written to stderr and piping can
hide the real state.

## Stage 4 — Independent Two-Item Review

The wrapper will create:

```text
results/lme-ku-buildout/<run-id>/checkpoint-2-ready.json
```

and wait for:

```text
results/lme-ku-buildout/<run-id>/continue-after-checkpoint-2
```

Do not create the continuation marker until a human/Codex review confirms all of the following.

### Run identity

- manifest has exactly 2 unique rows;
- run and graph provenance both name the expected run ID, container, and volume;
- Menhir commit, bench commit, fixture SHA, and settings match Stage 0;
- graph is recorded fresh;
- per-run telemetry DB exists and is growing;
- no production endpoint appears in logs or provenance.

### Ingest health

- both namespaces have `failed_remaining=0`;
- no episode remains queued, processing, or failed;
- both rows have `scalar_consolidated=true`;
- turn and episode counts match fixture expectations;
- TurnEvidence is present for expected turns;
- no unexpected retry loop or drain timeout occurred.

### Scalar correctness

- typed assertions agree with source text;
- scalar operations, normalized values, units, scopes, attributes, and subjects are credible;
- source/world dates match the LME fixture;
- every scalar-state View has valid current contributors;
- every scalar-history View has stable ordered `HISTORY_ENTRY` contributors;
- `no_anchor` slots do not acquire an absolute scalar-state View;
- no duplicate current View exists for a canonical slot.

### Provenance and isolation

- each sampled assertion reaches TurnEvidence by `FOUNDS`;
- TurnEvidence reaches its source episode through incoming `ADMITTED_ON`;
- Views have contributor episode `MENTIONS`;
- View contributor edges agree with receipt IDs;
- no assertion, entity edge, View edge, or recall result crosses namespaces;
- dashboard results match direct graph queries and run-local telemetry.

### Recall smoke

Before approving the remaining 76 items, run a bounded recall smoke against the two completed
namespaces if the wrapper/harness provides that checkpoint surface. Confirm structured recall
distinguishes scalar state from advisory history and includes inspectable contributor IDs.

After every check passes:

```powershell
$results = Join-Path $bench "results\lme-ku-buildout\$env:LME_KU_RUN_ID"
New-Item -ItemType File `
  -Path (Join-Path $results 'continue-after-checkpoint-2') `
  -Force
```

Record who approved it, at what time, and the validation receipt path.

## Stage 5 — Finish the 78 Items

Let the same process continue with the same code, settings, container, and volume.

Rules:

- no code edits during the canonical run;
- no dependency or model-setting changes;
- no manual graph mutations;
- no production Neo4j access;
- do not stop/restart merely because progress is slow;
- monitor manifest growth, phase logs, failed counts, container health, and telemetry;
- a same-code transient process failure may use the wrapper’s canonical resume only if provenance
  matches exactly;
- any code/config/fixture change ends the run’s canonical status and requires a new run ID/volume.

The wrapper must fail closed on a residual failed episode. Do not proceed to paid recall scoring
with failed ingest state.

## Dashboard During the Run

Once the graph is available and the dashboard hardening has landed:

```powershell
$results = Join-Path $bench "results\lme-ku-buildout\$env:LME_KU_RUN_ID"
$dashboardBolt = if ($env:LME_BOLT) { $env:LME_BOLT } else { '7694' }
$env:LME_DASHBOARD_NEO4J_URI = "bolt://127.0.0.1:$dashboardBolt"
$env:LME_DASHBOARD_NEO4J_USER = 'neo4j'
$env:LME_DASHBOARD_NEO4J_PASSWORD = 'lmedata123'
$env:LME_DASHBOARD_TELEMETRY_DB = Join-Path $results 'mcp_telemetry.db'

& (Join-Path $bench '.venv\Scripts\archolith-bench.exe') dashboard `
  --results-dir $results `
  --serve `
  --port 8200 `
  --scalar-task lme-01493427
```

If a nondefault Neo4j password is configured, use it instead of the example. The dashboard should
show no completed task until the manifest contains one; that is expected, not proof the ingest is
idle.

For each inspected task, compare browser output to a direct graph query and the telemetry receipt.
The UI is an inspection surface, not the source of truth.

## Failure Policy

If any stage fails:

1. stop paid work promptly;
2. do not delete the container, volume, result directory, logs, or telemetry;
3. record the exact failure, phase, namespace, episode, commits, and settings;
4. append/update `results/lme-ku-buildout/LEDGER.md`;
5. distinguish infrastructure failure, extraction failure, projection failure, recall failure, and
   score failure;
6. fix code only after the failed run is preserved;
7. rerun focused/no-spend validation;
8. create a new run ID and fresh volume for the next canonical attempt.

Do not mutate a failed graph into a “fresh” final graph. LLM-free replay is acceptable for migration
validation when explicitly labeled; it is not a substitute for a fresh canonical ingest after code
changes.

## Final Acceptance

Do not call the run complete until all conditions pass:

### Build

- 78 unique manifest rows;
- every row has `failed_remaining=0`;
- no drain timeout;
- no queued/processing/failed episode remains;
- every required row has `scalar_consolidated=true`;
- checkpoint and continuation are recorded;
- one immutable Menhir/bench commit pair spans every canonical phase.

### Data and projection

- expected fixture SHA and item count;
- original LME source/world time preserved at TurnEvidence/assertion/View/API/UI boundaries;
- typed assertions have valid vote, binding, namespace, and operation metadata;
- scalar-state Views obey anchor rules;
- scalar-history Views are slot-keyed, ordered, current, and advisory;
- no duplicate current View keys;
- no unanchored delta sum or false absolute materialization;
- postcard exact result: deltas 17 and 25, history current, state `no_anchor`, recall 25, never 42.

### Provenance

- run-local telemetry database present and readable;
- all phases have completed receipts;
- every sampled and validator-checked history entry has
  `HISTORY_ENTRY → assertion ← FOUNDS ← TurnEvidence ← ADMITTED_ON ← episode`;
- View `MENTIONS` contributors match source episodes;
- scalar-state contributor edges agree with its fold receipt;
- superseded Views remain inspectable and noncurrent;
- namespace-isolation validator reports zero violations;
- replay/repair receipts report zero unresolved failures.

### Recall and score

- recall/QA harness exits 0;
- `harness_recall/results.md` and structured results exist;
- all 78 items are represented in score output;
- task-level failures are retained;
- overall and scalar-relevant results are reported as measured;
- no score is substituted from v3, v4, or another code/configuration.

### Evidence package

- `manifest.json`;
- `run_provenance.json` and graph provenance;
- checkpoint and approval receipt;
- per-run telemetry DB;
- build, recall, and harness logs;
- replay and final acceptance reports;
- graph snapshot and/or backup with identity receipt;
- final dashboard validation notes/screenshots if retained;
- updated `results/lme-ku-buildout/LEDGER.md`;
- stopped benchmark containers after signoff unless an active inspection window is explicitly needed.

## Closing State

After signoff:

- stop the ephemeral benchmark Neo4j container;
- keep the volume and evidence package until the result is reviewed and backed up;
- do not touch production Neo4j;
- update the implementation wrapup with exact commits, tests, run ID, score, provenance findings,
  known failures, and follow-up work;
- only then decide whether scalar history should become default-on.
