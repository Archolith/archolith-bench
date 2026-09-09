# LongMemEval Framework

A self-contained testing framework for LongMemEval (LME) Mode-B evaluations of menhir's recall and ingest capabilities against extended conversation memory haystacks.

## What This Measures

### Recall-Only A/B
Compares `no_memory` vs `menhir_recall` on a **shared, pre-built persistent graph**. Tests whether a menhir variant can correctly rank relevant episodic facts during retrieval. This is **blind to ingest-code changes** — both variants retrieve from the same graph, so only the recall/ranking path is tested.

Workflow: `build` → `recall-ab` (across multiple branches)

### Buildout A/B
Compares main vs frontier branch **ingest code** by building two fresh graphs from the same question slice, then running recall+QA on each. Isolates the contribution of ingest-time temporal reasoning, belief gate, and other graph-construction features. This **reveals ingest-code quality** that recall-only cannot see.

Workflow: `buildout-ab` (self-contained; does not use the persistent graph)

### Analysis Harnesses
Measurement tools isolating specific aspects:
- **Answer matrix**: accuracy across 3 menhir configs × 6 question types (measures retrieval + generation separately)
- **MSC sweep**: minimal-sufficient-context (k vs accuracy curve) reveals the brief-construction bottleneck
- **Ablation sweep**: add-one-in breakdown of oracle components' marginal effects
- **Retrieval quality**: deterministic gold-presence-at-rank (no answer-model spend)
- **D0 retrieval entropy** (`analysis/entropy.sh [floor|delivered|both]`): distance-from-query-sufficient-state, GPT-free; FLOOR = evidence dispersion, DELIVERED = retriever walk to first gold hit
- **Perception gate tuning** (`analysis/perception_tune.py`, env `PT_*`): live threshold/veto sweep of menhir-frontier's perception boundary over the counting + held-out slices (dataset+LLM, no graph)
- **Acquisition window** (`analysis/acquisition_window.py`, env `AW_*`): Lever A σ WINDOW measurement — extract acquisitions → timeline → read-time windowed count (rolling vs calendar) vs gold (dataset+LLM, no graph)
- **Arm C capstone** (`analysis/capstone.sh run|delta|cleanup`): the honest end-to-end — baseline D0 → gated perception writes real Views into the graph (`perception_write.py`) → post D0 → delta report (`perception_delta.py`, counting collapse + held-out Goodhart guard). Writes are source-tagged `perception-capstone`; `cleanup` removes them exactly.

## Stratification ⚠️ Load-Bearing

The LongMemEval dataset is grouped by `question_type`:
1. `temporal-reasoning` — "what happened earlier"
2. `multi-session` — facts across multiple conversation sessions
3. `knowledge-update` — user-stated facts that change over time
4. `single-session-user` — facts the user stated in one session
5. `single-session-assistant` — facts the assistant inferred in one session
6. `single-session-preference` — user preferences stated once

**⚠️ Bare `--limit N` takes the first N items, which are 100% `temporal-reasoning`.** This is a sampling trap. All fair runs **MUST** sweep all 6 types via `--subset <type>` with `LME_PER_TYPE` items each (configurable, default 15/type).

The built-in analysis harnesses (`matrix`, `msc`, `ablation`) already loop across all 6 types. The runbook scripts (`recall-ab`, `buildout-ab`) accept `--subset` to run a single type.

## Prerequisites

- Docker (Neo4j 5.26, menhir serve)
- `archolith-bench` venv with `[longmemeval]` extra
- `menhir` venv (main branch, installed under `../../menhir` relative to bench root)
- `menhir-frontier` venv (optional, for frontier A/B). **As of 2026-07-15, `MENHIR_FRONTIER`
  defaults to `MENHIR_MAIN`** (the frontier branch merged to main, so `../../menhir-frontier` no
  longer exists) — only set `MENHIR_FRONTIER` explicitly if a real frontier fork checkout returns.
- OpenAI API key in `archolith-bench/.env` as `OPENAI_API_KEY`
- HuggingFace dataset: `xiaowu0162/longmemeval` (auto-cached)

## Quickstart

```bash
# 1. Build the persistent graph once (~1 day for 500-item oracle variant was measured with
#    LME_EXTRACT_MODEL=gpt-4.1-nano; now defaults to gpt-4o-mini (2026-07-15, comparability with
#    Zep/Mem0) which is slower/costlier -- this estimate is stale until re-measured)
./lme.sh build 500

# 2. Check graph status (read-only; safe to run mid-ingest)
./lme.sh status

# 3. Recall-only A/B: main vs frontier on the built graph
./lme.sh recall-ab main 30
./lme.sh recall-ab frontier 30

# 4. Buildout A/B: ingest-code comparison (fresh graphs)
./lme.sh buildout-ab 30

# 5. Analysis (stratified, all 6 types)
./lme.sh matrix      # answer accuracy
./lme.sh msc node_plain  # minimal sufficient context (main config)
./lme.sh ablation    # oracle component ablation

# 6. M1 IR gate (retrieval-only, no answer-model spend; needs a menhir server on LME_PORT_RQ)
LME_PER_TYPE=133 ./lme.sh ir-gate   # 133 = full 500-item corpus (max per-type count); default 15 = 90-item sample
```

For individual types:

```bash
LME_RECALL_LIMIT=5 ./lme.sh recall-ab main 30 --subset temporal-reasoning
```

## Graph Lifecycle

1. **Build**: `lme.sh build [N]` — ingest N items (default 30, full 500 in oracle mode takes ~1 day) into a fresh persistent Neo4j, then leave it up. Build finishes by promoting all memories to PERSISTENT scope (step 2), so builds end ready-to-recall.
2. **Promote**: `lme.sh promote` — flip every LME memory from `SESSION` to `PERSISTENT` scope (run automatically at the end of `build`; also runnable standalone, e.g. after a partial/legacy build). Idempotent and non-destructive. See [Memory scope](#memory-scope-regular-vs-session) below.
3. **Backup/Restore**: `lme.sh backup` / `lme.sh restore` — dump and restore for safe archival.
4. **Recall A/B**: `lme.sh recall-ab <branch> [N]` — read-only against the persistent graph; never resets it.
5. **Retry failed**: `lme.sh retry` — reset+drain episodic nodes in FAILED state.

**Rule**: Never reset a built graph. The manifest (`results/lme-ingest/manifest.json` for the
canonical `menhir-lme-neo4j` container; `manifest-<LME_NEO4J_NAME>.json` for any other container
name, e.g. a throwaway smoke build — see `config.sh:LME_MANIFEST_PATH`) tracks ingested items; a
fresh build needs all 500 questions re-ingested.

### Memory scope: regular vs. session

menhir stamps freshly-extracted `Entity` nodes as **`SESSION`** scope; only the consolidation
pass promotes reinforced facts to **`PERSISTENT`**. But consolidation is off in benchmark mode,
*and* it **deletes** low-sharpness one-off facts — which is most LME answer entities. So without
intervention every benchmark memory stays `SESSION` forever.

That matters because recall filters out `SESSION` nodes unless `include_session=True`
(`recall_service.py:937`). Consequences of leaving them `SESSION`:
- `build_context` returns **empty briefs** (it calls recall without `include_session`).
- Any plain (non-session) recall path sees **nothing**.

LME facts are durable knowledge the system should always recall, so we write them as **regular
memories**: `promote_persistent.sh` does a blanket `SESSION → PERSISTENT` flip (nodes +
`RELATES_TO` edges) at the end of every build. This is **non-destructive** (nothing deleted,
unlike consolidation) and does **not** change existing recall-only A/B results — those already
pass `include_session=True`, so the nodes were always visible; promotion only *also* exposes them
to `build_context` and plain recall.

### ⚠️ Temporal grounding: dates need a backfill (now automatic as of 2026-07-15)

**Known bug, still present in graphiti-core 0.29.2 (re-verified 2026-07-15):** the ingest path
does **not** honor `occurred_at` on graphiti's own Episodic nodes/`RELATES_TO` edges — an episode
ingested with the session's historical date still lands `valid_at = now()` even though menhir's
own Episodic nodes correctly carry the backdated `reference_time`. So a **freshly built graph has
fake temporal grounding on the graphiti side** (episodes/edges stamped with the build date), which
breaks temporal-reasoning and the BriefBuilder Timeline.

**As of 2026-07-15, `build_graph.sh` runs the backfill automatically as the last build step** —
no manual follow-up needed for a normal `lme.sh build`. It's still available standalone for a
partial/legacy build or to preview changes first:

```bash
./scripts/longmemeval/lme.sh backfill-dates --dry-run   # preview counts
./scripts/longmemeval/lme.sh backfill-dates             # apply (writes a revert snapshot first)
```

Only `valid_at` (world-time) is rewritten; `created_at`/`expired_at` (belief-time = ingestion) are
correct as-is. Genuine LLM-extracted edge dates are preserved (only ingestion-defaulted edges are
touched). Idempotent — safe to run again even if already backfilled (a fully-backfilled graph
reports `0 to backfill`). A revert snapshot is written to `results/lme-ingest/date-backfill-revert.json`
(or `date-backfill-revert-<LME_NEO4J_NAME>.json` for a non-canonical container — see
`config.sh:LME_REVERT_SNAPSHOT_PATH`) before any mutation.

## run_manifest.json Contract

Every `recall-ab` run writes `results/lme-recall-<variant>/run_manifest.json` — a reproducibility record that captures:
- Menhir source (branch, commit, dirty state)
- Bench branch/commit
- Neo4j container, image, bolt port
- LME variant (oracle/s/m) and dataset snapshot
- Recall settings (top_k, include_session)
- Answer model + scorer (containment or llm-judge)
- Episodic state counts in the graph at run time
- Parsed scores from the harness output

Reproduce a run from the manifest alone (modulo API randomness).

## Per-run Telemetry

Both `build_graph.sh` and `recall_ab.sh` export `MENHIR_MCP_TELEMETRY_DB` to a run-local SQLite
path (`$LME_RESULTS_DIR/mcp_telemetry.db` for builds, `$RUN_OUTPUT_DIR/mcp_telemetry.db` for
recall). This preserves vote receipts, lifecycle events, and scalar audit trails alongside results
so the dashboard's `ScalarTaskReader` can display them and the acceptance validator can check for
their presence.

Current Menhir builds also persist `llm_usage_events`, one terminal row per instrumented
provider-client call.
The row contains run/episode provenance, model and endpoint, latency, the raw provider usage object,
and exact input/output/total/cached/reasoning token fields when reported. Counts are not estimated:
`missing_usage_calls` identifies completed calls whose provider omitted usage. The knowledge-update
wrapper snapshots the ingest aggregate to `$LME_RESULTS_DIR/ingest_llm_usage.json` before recall,
including model/endpoint breakdowns. Recall uses the same run-local telemetry DB. After the harness,
the wrapper writes `$LME_RESULTS_DIR/run_llm_usage.json`, combining the final Menhir total with the
harness checkpoint's answer-model and optional LLM-judge totals, with per-arm and per-operation
breakdowns. Its `complete` flag is true only when both sources are present. Cost should be derived
later from these immutable token counts and the price schedule being evaluated.

## Acceptance Validation

`lme.sh validate [--expected N]` runs `lib/validate_run.py` against the current provenance file,
manifest, and telemetry DB. It emits a machine-readable JSON report covering:

- **manifest cardinality** (expected vs actual items)
- **zero failed episodes** (strict zero-tolerance policy)
- **projection counts** (assertions, scalar_state Views)
- **namespace isolation** (all namespaces start with the configured prefix)
- **commit immutability** (all attempts ran the same Menhir and bench code)
- **canonical label** (warns if the run is marked noncanonical)
- **telemetry presence** (DB exists and has lifecycle events)
- **no interrupted phases** (all phases completed cleanly)

Exit 0 on PASS; exit 1 on any FAIL check. The report is also written to
`$LME_RESULTS_DIR/acceptance-report.json`.

## run_provenance.json Contract (append-only)

A buildout is resumable, so its provenance file outlives the process that created it. Every write
is therefore **additive** — `lib/run_provenance.py` owns the file, and neither wrapper may `cat >`
it. This applies to `run_provenance.json` and to `graph-provenance-<container>.json`.

| Field | Meaning |
|---|---|
| *(top level)* | Frozen after the first attempt. Describes the graph on disk — `graph_fresh`, `volume_pre_existed`, dataset, and the settings the data was actually ingested under. `lme.sh ir-gate` reads `graph_fresh` from here. |
| `identity` | `run_id`, `arm`, fixture hash/count, container/volume, dataset, variant, namespace prefix, `menhir_commit`, and `bench_commit` (when supplied by that wrapper). A resume that disagrees on any recorded identity field is **refused**. Code commits are also immutable in canonical mode; set `LME_NONCANONICAL=1` to permit mixed-code resumes for development iteration (the output is labelled `noncanonical`). |
| `attempts[]` | Every attempt in full, first included, numbered, with `phases_interrupted`. |
| `latest_attempt` | Mirror of the most recent attempt, for readers that want current values. |
| `phases[]` | One entry per phase actually run. |
| `first_started_at` / `last_started_at` / `attempt_count` | Span of the whole run. |

Each phase entry records the state it inherited and the settings it truly ran under:

- `phase` — `build-graph`, `build-graph-checkpoint-continuation`, `recall-qa`, or `ingest-graph`
- `status` — `started`, `completed`, `failed`, or `interrupted`
- `manifest_items_at_start` / `manifest_question_ids_at_start` — what the phase inherited, which is
  what distinguishes a resumed phase from a fresh one
- `effective_settings` — read at phase start, not at preflight. The checkpoint continuation flips
  `require_fresh` and `ingest_stop_after_items`, so one preflight snapshot describes neither build
  phase. Includes commits/dirty state, `segmentation`, `strict_failure_policy`, the scalar
  threshold/`k`/call budget, the reconcile flags, the audit flags, and `turn_evidence_required`.

A phase is closed only on the success path, so a killed run leaves its phase `started`; the next
attempt's `begin` marks it `interrupted`. That is a fact worth seeing, not an error.

## Ledger: `ledger.csv` is the source, `LEDGER.md`'s table is generated

`LEDGER.md` carries real analysis — regression breakdowns, per-item tables, findings — and
that prose stays prose, hand-maintained. Its *scoreboard* was different: tabular data being
hand-typed, which meant nothing could join a score delta to a provenance delta without a
human reading both files.

So the scoreboard lives in `results/lme-ku-buildout/ledger.csv` and the markdown table is
generated from it. Edit the CSV, then render. Only the region between the
`BEGIN/END GENERATED SCOREBOARD` markers is rewritten; if those markers are missing the
generator refuses rather than guessing where the table is, because guessing would overwrite
analysis.

```bash
LIB=scripts/longmemeval/lib

python $LIB/ledger.py render     # CSV -> the table in LEDGER.md
python $LIB/ledger.py validate   # rows vs. the closed vocabulary and the filesystem
python $LIB/ledger.py join A B   # score delta + surface blame, in one command
```

`join` is the question the ledger could not answer before: a score moved, where do I look
first? The delta comes from the CSV, the attribution from the surface fingerprints in each
run's provenance.

### Schema

The columns refuse to launder messy reality into clean numbers:

- `score` is empty unless the run produced exactly one usable score. A row reporting two arms
  (`v2c 0.667 / v2h 0.679`) keeps both in `score_raw` and leaves `score` empty — any single
  float there would be a fabrication.
- `status` is a closed vocabulary (`scored`, `multi_arm`, `killed`, `aborted`, `invalid`,
  `partial`, `measure_only`, `offline`), so an empty score always has a stated reason and is
  never ambiguous between *unscored* and *scored zero*.
- `canonical` is `""`, `superseded`, or `current`, and **at most one row may be `current`** —
  two rows claiming to be the current canonical evidence is an unresolved claim, not a
  formatting detail.

### Severity model

`FAIL` means the ledger asserts something false; `WARN` means it is incomplete. Only `FAIL`
exits non-zero (use `--strict` to fail on warnings too), which keeps `validate` usable as a
gate instead of permanently red.

Validation runs in **both directions**. Rows → disk catches a row claiming a results directory
that is not there. Disk → rows catches the more dangerous case: a run that executed, wrote
provenance, and never reached the scoreboard, so its evidence exists but is invisible to
anyone reading the table. Note that `manifest.json` and `harness_recall/` are deliberately
*not* treated as score evidence — they prove ingest or recall ran, not that a number was
produced.

**Current state of the real ledger: 0 FAIL, 38 WARN.** The warnings are honest debt, not
noise: 24 rows cover 51 run directories, 9 directories hold a readable score with no row
(small-sample packet-shape and rescore panels — decide whether they belong in a *buildout*
scoreboard), and most pre-fingerprint runs cannot have their deltas attributed to files.

## Benchmark Surface Fingerprint (`bench-surface.yaml`)

A commit SHA does not establish what code a run executed. Two runs can share a SHA and
differ — a dirty tree, or an untracked file — and a file can change behavior in one commit
while `git log` attributes it to another. The assistant-turn gate in `claim_segmenter.py` is
the worked example: it was authored untracked, then swept into git by
`f49ab59 chore(bench): track 12 previously-untracked LME scripts`, so a commit-range diff
dates it wrongly and never names it.

`bench-surface.yaml` declares the paths that can move a score, tagged by pipeline stage
(`fixture` → `ingest` → `consolidate` → `recall` → `score` → `harness`).
`lib/bench_surface.py` hashes them into each run's provenance, so comparing two runs yields
file-level evidence instead of a commit-range guess.

```bash
# What is the surface right now?
python scripts/longmemeval/lib/bench_surface.py fingerprint

# Why do these two runs differ?
python scripts/longmemeval/lib/bench_surface.py blame \
  results/lme-ku-buildout/<run-a>/run_provenance.json \
  results/lme-ku-buildout/<run-b>/run_provenance.json
```

`run_knowledge_update_buildout.sh` calls `attach` after `begin`, recording the fingerprint
against the running attempt — per attempt, because separate attempts of one run may execute
different code.

Three properties it holds, because it is evidence and not a convenience:

- **A declared-but-absent file is recorded, not skipped.** Deleting a surface file changes
  the digest; a skipped path would leave it unchanged.
- **The manifest hashes itself** (`manifest_sha256` feeds `surface_digest`). Narrowing the
  surface is itself a change to what is tracked, so a shrunken declaration cannot go on
  looking stable.
- **A missing repo is an error, not a column of nulls** — an absent checkout would otherwise
  fingerprint as "every file deleted", which is wrong *and* stable across runs.

Two deliberate non-properties:

- **Fingerprinting is non-fatal at launch.** A fingerprint is evidence about a run, not a
  precondition for it; refusing to launch would trade a recorded gap for no run at all.
- **It does not gate canonical runs on surface drift.** `run_provenance.py` already refuses
  commit drift on a canonical resume; extending that to `surface_digest` would catch strictly
  more, and is deliberately left as a follow-up rather than bundled in here.

When editing the surface: bias **broad**. A path listed that turns out not to matter costs
one extra line in a blame report; a path missing makes a real behavior change invisible.
`tests/test_lme_bench_surface.py` fails if any checked-in glob matches nothing, which catches
a renamed or deleted module silently shrinking coverage.

**Historical runs predate this.** Of 51 run directories, 16 have no `run_provenance.json` at
all, 18 of the remaining 35 record no dirty flag, 14 carry no `fixture_sha256`, and 5 ran with
a dirty tree. `blame` reports a run with no fingerprint as not-comparable rather than guessing.
Fingerprints are reconstructible via `git show` only for the runs that were clean *and* recorded
a full SHA; the 5 dirty runs and the 16 with no provenance are permanently unattributable.

## Menhir Recall Lab integration

Both `build_graph.sh` and `run_knowledge_update_buildout.sh` now export
`MENHIR_BENCH_RESULTS_ROOT` and `MENHIR_BENCH_ACTIVE_RUN_ID` so the Menhir Recall Lab
(`/explorer/recall-lab/bench-runs/`) can discover the run's artifacts and (for the active
run) display live graph projections alongside manifest and checkpoint data.

- `MENHIR_BENCH_RESULTS_ROOT` — set to `$BENCH_DIR/results` so the catalog discovers all runs.
- `MENHIR_BENCH_ACTIVE_RUN_ID` — derived as `$(basename "$LME_RESULTS_DIR")`, which is the
  subdirectory name under `results/` (e.g. `lme-ingest` for the persistent graph, or the
  `RUN_ID` for a KU buildout).

The recall Menhir subprocess in `run_knowledge_update_buildout.sh` also receives these vars
so live graph queries work during the recall/QA scoring phase.

## Configuration Reference

All hardcoded values are centralized in `config.sh` and environment-overridable:

```bash
# Neo4j
LME_NEO4J_NAME=menhir-lme-neo4j
LME_BOLT=7689
LME_HTTP=7476
LME_NEO4J_PW=lmedata123
LME_NEO4J_VOL=menhir-lme-data
LME_NEO4J_IMAGE=neo4j:5.26-community

# Ports (one per workflow + analysis)
LME_PORT_BUILD=8102             # build_graph.sh
LME_PORT_RECALL=8103            # recall_ab.sh
LME_PORT_BUILDOUT_MAIN=8105     # buildout_ab.sh (main)
LME_PORT_BUILDOUT_FRONTIER=8106 # buildout_ab.sh (frontier)
LME_PORT_MATRIX=8112            # analysis/answer_matrix.sh
LME_PORT_MSC=8113               # analysis/msc_sweep.sh
LME_PORT_ABL=8114               # analysis/ablation_sweep.sh
LME_PORT_RQ=8109                # analysis/lib/retrieval_quality.py

# Models
LME_EXTRACT_MODEL=gpt-4o-mini  # matches Zep/Mem0's published LongMemEval extraction model (2026-07-15)
LME_ANSWER_MODEL=gpt-4o
LME_JUDGE_MODEL=gpt-4o-mini
LME_EMBED_MODEL=text-embedding-3-small
LME_SCORER=llm-judge

# Dataset
LONGMEMEVAL_VARIANT=oracle  # oracle|s|m (smaller variants for dev)
LME_LIMIT=30                # items to ingest
LME_RECALL_LIMIT=10         # recall top-k
LME_PER_TYPE=15             # stratified sample per question type (analysis)
LME_NONCANONICAL=0          # permit code-commit drift on resume (labels output noncanonical)

# Paths (auto-derived from git root, but overridable)
ARCH_DIR, BENCH_DIR, MENHIR_MAIN, MENHIR_FRONTIER  # MENHIR_FRONTIER now defaults to MENHIR_MAIN
LME_MANIFEST_PATH, LME_REVERT_SNAPSHOT_PATH        # derived, scoped by LME_NEO4J_NAME (config.sh)
```

Override via environment:

```bash
LME_ANSWER_MODEL=gpt-4-turbo LME_PER_TYPE=5 ./lme.sh matrix
```

## Interpreting Results

### Recall-Only A/B
Compares two menhir variants (`no_memory` vs `menhir_recall`, or frontier vs main). The output markdown shows:
- **Arms**: typically `no_memory` (baseline) vs `menhir_recall` (test)
- **N**: number of questions run
- **Score**: LLM-judge score (0.0 = no correct facts, 1.0 = fully correct answer)

Example:
```
| Arm           | N  | Score  |
|---------------|----|--------|
| no_memory     | 30 | 0.150  |
| menhir_recall | 30 | 0.367  |
```

Frontier's oracle ranking improves recall by ~0.2 over baseline.

### Noise and Variability
With N=15/type (90 total), expect ±~0.13 standard error on the score. Report per-type, not just an average. Cross-run reproducibility is strong (node-plain hit 0.367 on three independent runs).

### Analysis Outputs
- **Answer matrix**: 3 configs × 6 types × 15 items = 270 questions. Reveals which config is strongest per type.
- **MSC sweep**: accuracy vs k (1, 2, 3, 5, 10). If plateau at k=5, brief construction is the bottleneck, not retrieval.
- **Ablation**: marginal effects of oracle components (oracle ranking, intent lens, evidence anchor, etc.). Warden gate zeros the score on anecdotal evidence.
- **Retrieval quality**: gold-rank histogram (rank at which the answer's tokens first appear in top-k) — no answer-model cost.
- **Brief A/B** (`lme.sh brief-ab [--score]`): does the frontier BriefBuilder brief beat the flat
  relevance brief? Serves menhir-frontier twice over the pre-built graph
  (`MENHIR_FRONTIER_BRIEF_BUILDER` off vs on), collects `/api/context` briefs each time (free),
  then answers + judges both. **The recall-only harness can't measure this** — it feeds
  `/api/recall`, never `/api/context` where the BriefBuilder lives. Collection is free; add
  `--score` to spend OpenAI (answer + judge). Needs `tiktoken` in the menhir venv, else
  build_context runs in heuristic mode and halves the token budget, squeezing out the Timeline.

  The BriefBuilder keeps the relevance-ranked list as the brief and **appends** a temporal
  Timeline below it (an earlier Timeline-*first* design measured −0.10 by burying the answer;
  append measured neutral, +0.03 within noise). The Timeline is retained because it is still the
  right container for genuinely temporal/supersession questions — it just must not displace
  relevance order. Toggle the feature with `MENHIR_FRONTIER_BRIEF_BUILDER` (default off).

### Campaign Findings (2026-07-02)

- **node-only is strongest**: no oracle, no frontier bells → 0.367. Simpler is better.
- **Frontier EvidenceAnchorWarden is selective**: On anecdotal questions (unreliable user claims), it returns zero results → 0.000 score. On other oracles it's neutral.
- **MSC plateaus at k=5**: ~400 tokens = 95% of ceiling. Bottleneck is brief construction (what to pack), not retrieval (ranking).

See `menhir-frontier/.agent/plans/anecdotal-recall-oracle-ladder.md` for the oracle-routing roadmap those measurements feed.

## KU-Buildout Ledger (mandatory)

Every knowledge-update buildout run **must** be recorded in
`results/lme-ku-buildout/LEDGER.md` before the session ends. The ledger is the
single source of truth for what was tested, how it scored, and what fixture
produced the graph.

After any KU buildout run completes (or is killed):
1. Add a row to the **Scoreboard** table with run ID, date, N, segmentation
   mode, score (or `(killed)`), and extract model.
2. If the run completed scoring, add a **Per-Item Results** section listing
   PASS/FAIL question IDs.
3. If a new fixture or detector snapshot was used, record its SHA256 in the
   **Fixture** section and ensure the snapshot is saved in the run's results
   directory.
4. Update the **Run Infrastructure** table with container name and ports.
5. Note any new **Known Issues** discovered during the run.

The fixture file (`fixtures/longmemeval/knowledge_update_subset.json`) and any
frozen detector snapshots (`claim_segmenter_frozen.py`) must be preserved in the
results directory of each run that used them.

### Current-code scalar reingest

`run_knowledge_update_buildout.sh` is the guarded 78-item scalar reingest. It refuses to run without
an explicit unique run ID and one of two pinned arms:

- `baseline`: current production gate (`3/3`, no attribute/scope/subject reconciliation);
- `candidate`: true `2/3` plus attribute/scope/subject reconciliation.

Both arms require a fresh graph, scalar consolidation, TurnEvidence, and View authority during
recall scoring. The wrapper validates the frozen fixture, records its SHA-256 and all effective
flags, enables consolidation and recall audit trails, refuses tracked dirty code, checks
Docker/image/ports/container/volume/result isolation, and stops Neo4j after the run while preserving
the stopped container and volume for inspection. Ingest defaults to two concurrent item namespaces:
the driver keeps one episode active in each namespace and submits its next episode only after the
prior one reaches a terminal state. This preserves chronological processing inside each namespace
while Menhir enriches distinct namespaces concurrently. Set `LME_KU_INGEST_CONCURRENCY=1` for the
legacy serial path; the guarded wrapper accepts widths 1 through 4 and records the effective value
in run and graph provenance. Resume resets are fail-closed and remove the incomplete namespace's
graph partition, scalar state, and TurnEvidence before resubmission. Each manifest row records
enrichment LLM-task and processing-attempt totals; a scalar window with residual FAILED episodes
stops before further paid work or consolidation.

For an independently reviewed spend gate, set `LME_KU_CHECKPOINT_ITEMS=2`. The ingester hard-limits
the first build to the first two fixture items, validates and persists their manifest, and then the
wrapper waits without starting item 3. After inspecting the live graph and audit evidence, create
the marker named in the wrapper log (`continue-after-checkpoint-2` in that run's results directory).
The same process then resumes from the manifest and processes the remaining 76 items before recall.
The checkpoint count and target are recorded in run and graph provenance.

Run the no-spend preflights first:

```bash
LME_KU_RUN_ID=scalar-current-baseline-20260728 LME_KU_ARM=baseline \
  ./scripts/longmemeval/run_knowledge_update_buildout.sh --preflight-only
LME_KU_RUN_ID=scalar-current-candidate-20260728 LME_KU_ARM=candidate \
  LME_KU_CHECKPOINT_ITEMS=2 \
  ./scripts/longmemeval/run_knowledge_update_buildout.sh --preflight-only
```

Then run the two arms sequentially with the same commands minus `--preflight-only`. Do not compare
against the 2026-07-22 graph: it predates the resolved-twin binding and admission-provenance fixes.
Set `LME_KU_KEEP_NEO4J_UP=1` only when immediate live graph inspection is worth the Docker memory.
Set `LME_KU_ALLOW_RESUME=1` only to continue the same interrupted run ID; it deliberately disables
the fresh-storage guard for that resume. `LME_KU_ALLOW_DIRTY=1` is an explicit escape hatch for
development probes; results produced with it are not canonical benchmark evidence.

## Troubleshooting

### "pre-built menhir-lme-neo4j not found"
`recall-ab` requires a persistent graph. Run `lme.sh build [N]` first.

### "Neo4j not ready"
The Docker container takes 10-20s to start. If it persists, check `docker ps` and `docker logs menhir-lme-neo4j`.

### Graph has 0 episodic nodes
The ingest may not have finished. Check `lme.sh status` and wait for queue_depth==0.

### FAILED episodes
The ingest script retries FAILED episodes once. If episodes still fail, they hit the per-job LLM budget (`MENHIR_MAX_LLM_CALLS_PER_JOB=20`). Run `lme.sh retry` to reset+drain them.

A scalar build then **stops** on any residual FAILED episode. This gate is strict and has no
tolerance knob — a FAILED episode means a turn never reached the graph, so scalar views, user-founded
views and recall accuracy would all be computed over a namespace that does not match the transcript.
(`LME_KU_MAX_FAILURES_PER_NS` existed briefly and was reverted: it turned the stop into a log line
and let builds ship silently-lossy namespaces.) If the failures are phatic turns that legitimately
extract to zero edges, the fix is the evidence-only path below, not a tolerance.

### Adaptive segmentation: extraction vs. evidence
Under `LME_SEGMENTATION=adaptive`, a fact-free turn (`CONTEXT_ONLY`) is recorded as
`:TurnEvidence` and **not** sent to extraction, so it creates no episode that could fail. Turns are
never merged. Each user source turn has one stable evidence identity holding the original unsplit
text, and all episode segments from that turn cite the same UUID; repeated identical turns use
different source-position keys. CONTEXT_ONLY assistant turns carry their own `role`/`declarant`,
which prevents assistant words from being filed under `declarant="user"` (menhir's evidence
projection is fail-closed on `role='user' AND declarant='user'`).

If a knowledge update is missing from the graph, check whether its turn was classified
`CONTEXT_ONLY`: `segmentation_mode("user", text)` in `lib/claim_segmenter.py` is the decision, and
`is_context_only_user_turn` is the fail-open gate. It bypasses extraction only for provably phatic
acknowledgments and fact-free question frames; an unknown declarative or a first-person/possessive
clause inside a question is extracted. `has_durable_signals` provides additional positive signals:
state-change verbs (including a wh-question presupposition such as "Why did I move to Seattle?"),
correction markers, named subjects, possessive assertions ("my mom uses the same app"), frequencies
("three times a week"), and quantities. Bare yes/no questions remain conservative.

### Stale enrichment on resume
With `LME_KU_ALLOW_RESUME=1`, menhir's stale-lease recovery may already be enriching a previous
run's episodes into namespaces this build is about to own. The ingest waits for those
PENDING/ENRICHING rows to settle **before** resetting, then resets once. Order matters: a reset
deletes the rows that prove a worker is alive, so settlement checked afterwards reads zero against
a worker still mid-flight. If settlement cannot be confirmed — state unreadable, or still busy at
the timeout — the build **raises** rather than resetting anyway. Re-running costs time; resetting
under a live writer costs a corrupted namespace whose numbers still look plausible.

### Per-branch venv guard
If a menhir variant enforces interpreter identity (e.g., frontier's `runtime.py`), `recall-ab` auto-detects and uses the variant's own `.venv` if it has one. Explicitly pass a worktree path to override: `lme.sh recall-ab /path/to/worktree 30`.

### Empty recall / empty build_context
If recall returns 0 results (or `build_context` returns an empty brief) on a graph you know has data, the memories are almost certainly still `SESSION`-scoped — `recall_service.py:937` drops `SESSION` nodes unless `include_session=True`. Fix: `lme.sh promote` (blanket `SESSION → PERSISTENT`). See [Memory scope](#memory-scope-regular-vs-session). The recall-only harness sidesteps this by always passing `include_session=True`, but that flag only masks the scope issue for that one path — `promote` is the real fix and is now part of every `build`.

### Disk space for dumps
`lme.sh backup` dumps the full graph to disk. With 500 items in oracle mode (~10k episodic nodes), expect ~500MB.

## Future Work

- Cross-platform portability (scripts assume Git Bash on Windows + Docker Desktop)
- Menhir preset configuration (currently hardcoded `knowledge`; should be launcher-driven)
- `top_k` configurability in the harness (partially done via `LME_RECALL_LIMIT` env knob)

## Standalone dashboard status

The `:8200` standalone dashboard (`archolith_bench/dashboard.py`) that displays checkpoint
progress and scalar task details is **transitional**. Menhir Recall Lab (`/explorer/recall-lab/bench-runs/`)
is the canonical owner for benchmark task inspection. The standalone dashboard is retained
as a temporary fallback until the Menhir explorer covers all scalar viewer detail. Do not
add new features to the standalone dashboard; extend the Menhir explorer instead.

## See Also

- `.agent/README.md` — main project index
- `menhir-frontier/.agent/plans/anecdotal-recall-oracle-ladder.md` — oracle routing roadmap
- `archolith-bench/scripts/longmemeval/lib/` — Python support (ingest, retry, expected turns)
