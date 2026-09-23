# archolith-bench — Durable Script Index

Local counterpart to `projects/archolith/menhir/.agent/scripts-index.md`, which indexes every
durable script in **both** repos by the question it answers and defines the naming convention
(`_name.py` is a throwaway — delete it once its finding is written down, never index it; `name.py`
is a durable instrument — index it in the same commit that adds it). Keep the two indexes in sync:
this file holds the full per-script methodology; the menhir index holds a short cross-repo pointer
to each entry here.

## Scalar-identity / dependency-bridge research instruments

The durable offline typed-scalar instrument is
`scripts/measure_deterministic_scalar_shadow.py`. It consumes only JSON captures from Menhir's
`scripts/freeze_scalar_samples.py`, loads the real Menhir proposal/gate/extractor/comparator from an
explicit checkout, reruns the pure deterministic extractor over frozen episode text, and replays
captured LLM proposals through the real gate/comparator. It makes no new LLM, network, Neo4j, Docker,
or service calls. The first held-out input is the versioned non-LME
`fixtures/deterministic_scalar_heldout_v1.json`; its six `2 namespaces × k=3` calls are smoke
evidence only, not a promotion or population gate. The paired Menhir workflow owns the exact
capture/report commands. Run command: see `.agent/README.md` § "How to Run".

The research scalar-identity acceptance panel is deliberately smaller and has no runtime lane. It
feeds source-bound raw candidates through Menhir's pure research adapter and structural composer:

```bash
python scripts/measure_scalar_identity_acceptance_panel.py \
  fixtures/scalar_identity_acceptance_v1.json \
  --menhir-root ../menhir \
  --json-out results/scalar-identity-acceptance/report.json \
  --markdown-out results/scalar-identity-acceptance/report.md
```

The report is source-free and keeps `promotion_status=not_evaluable`; it is a regression panel, not
a population estimate or permission to route/persist deterministic candidates.

The paired noisy-language companion keeps clean and informal cases in separately reported slices:

```bash
python scripts/measure_scalar_identity_noisy_panel.py \
  fixtures/scalar_identity_noisy_v1.json \
  --menhir-root ../menhir \
  --json-out results/scalar-identity-noisy/report.json \
  --markdown-out results/scalar-identity-noisy/report.md
```

It reports paired invariance, perturbation coverage, and false-current-state errors. It remains a
bounded regression panel with `promotion_status=not_evaluable`, not a population estimate.

The opt-in mapped-isolation comparison runs the same fixture through both research paths and reports
identity differences and composition gains without source quotes:

```bash
python scripts/measure_scalar_identity_isolated_comparison.py \
  fixtures/scalar_identity_noisy_v1.json \
  --menhir-root ../menhir \
  --json-out results/scalar-identity-isolated/report.json \
  --markdown-out results/scalar-identity-isolated/report.md
```

The isolated adapter must be present in the selected Menhir checkout; the runner fails loudly rather
than falling back to the canonical adapter. This is research evidence only and never enables routing,
persistence, or scalar authority.

The cumulative-completion holdout reuses the same source-free comparison runner:

```bash
python scripts/measure_scalar_identity_isolated_comparison.py \
  fixtures/scalar_identity_cumulative_v1.json \
  --menhir-root ../menhir \
  --json-out results/scalar-identity-cumulative/report.json \
  --markdown-out results/scalar-identity-cumulative/report.md
```

`fixtures/scalar_identity_cumulative_v1.json` is a 24-case, 12-pair non-LME contract with six
train and six holdout pairs. Positive cases use only source-authored present-perfect completion
forms with strict `so far`/`to date` wording and typed operation `absolute`; adversarial cases
cover past, negation, modality/future, history, coordination, and subset/empty-target risks.
The validated result is baseline clean 12/12 (6 composed), baseline noisy 6/12 (0 composed), and
isolated clean/noisy 12/12 (6 composed each), with a +6 noisy composition gain and 0/0
false-current-state errors. It is a bounded regression instrument, not a production-promotion gate.

The dependency-evidence bridge has a separate offline measurement runner:

```bash
python scripts/measure_dependency_scalar_bridge.py --menhir-root ../menhir \
  --json-out results/dependency-bridge/report.json \
  --markdown-out results/dependency-bridge/report.md
```

`measure_dependency_scalar_bridge.py` is the **Bench-owned, offline** evidence-bridge validator. It
strictly verifies the 48-case fixture SHA/version, builds raw candidates from supplied subject,
operation/value, claim text, and locators, and loads the real Menhir adapter, immutable evidence
models, and Phase-A dependency rule. It has no LLM, network, runtime, Neo4j, cache, or
bridge-to-baseline fallback lane. Gold candidate locators are supplied explicitly, so span
recognition is reported as `not_measured_gold_locator_supplied`; role and operation
classification remain `not_measured`. Outputs refuse overwrite and create parent directories.

The authoritative r2 execution is recorded in
`results/dependency-scalar-bridge/phase-a-v1-20260806-r2/report.{json,md}` for fixture SHA
`bde118508cf55c94bbd10fc88fbc625a0f465859a545f3ca79deb391a25ba57b`: 48 cases, 5/6 supported
identities/provenance exact, 0/14 unsupported composed, 0/28 false-current, 45/45 evidence
provenance, 45/45 bridge provenance, 179/192 cue exact, and edges 135 TP / 180 predicted / 144
gold (P .75/R .9375/F1 .8333). Proposal parity was 45/48 (three `temporal_past_only` adapter
drops); composer replay was stable 45/45; baseline composition was 2/48 (43 abstain, 3 drops).
R1 was the historical first execution with incomplete git provenance; r2 changed instrumentation
only and aggregates/cases are identical. Promotion remains `not_evaluable`, and recognition,
role/operation, performance, cache, and full replay gates remain unmeasured. The one supported miss
is bare `retain`; no post-hoc holdout tuning is permitted — broader support requires an independent
policy panel. Report SHA256: JSON
`53aff17ecd1d758f2d808050e9ac8e2512f269a3254ae2fee99f25d2696fc412`; Markdown
`3284c6d81d47e22907c7fcd6fe79993ca65fa386a7206366d9d079ac78999287`.

## Recall-packet evaluation instruments

`scripts/longmemeval/analysis/run_typed_recall_packet_rescore.py` is a noncanonical,
recall-only comparison of Menhir Recall Lab's full or query-filtered typed packet against the
canonical KU78 recall checkpoint. It reuses an existing graph, performs no ingest or graph writes,
rejects incomplete answer/judge evidence, records code and fixture provenance, and writes task
transitions plus token totals under a new run directory. It requires a live read-only Recall Lab
endpoint and explicit answer/judge provider credentials.

`scripts/longmemeval/analysis/run_packet_shape_panel.py` is the paired preregistered held-out packet
shape panel. It excludes the five development cases, reuses the same production retrieval results
for both shapes, scores ten fixed strata, and labels every result noncanonical and recall-only. It
may make paid answer/judge calls and writes its registration, fixture, checkpoints, results, and
provenance under `results/lme-ku-buildout/`.

## menhir ScalarStateView instruments (live here, documented in menhir)

`scripts/scalar_state_coverage.py` and its siblings (`run_scalar_state_e2e.sh`,
`inspect_scalar_state_graph.py`, `scalar_view_authority_live.py`, `scalar_leads_authority_live.py`,
`scalar_phase_d.py`) measure menhir's scalar/View ingest path. They live here because they import
`archolith_bench.harness.menhir_scalar_state` fixtures and the `scalar_bolt` prod guard.

Anyone working in menhir looks for them there and does not find them, then re-derives the answers by
hand. The index that prevents that is
`projects/archolith/menhir/.agent/workflows/scalar_state_measurement.md` — **keep it in sync when
adding, renaming, or removing an instrument here.**

Frozen methodology: every measured run uses a FRESH ISOLATED menhir+neo4j stack. Stacking rounds on
one stack is invalid for comparative yield.

## Beacon agent-task evaluation (P3)

`archolith-bench beacon-eval plan|run` (`archolith_bench/beacon_eval/`) answers: does a Beacon
endpoint help a coding agent orient, compared with `AGENTS.md` and docs alone, and with the beacon
pasted into the prompt? Conditions A (docs), B (+ Beacon MCP, the only MCP server), C (+
`beacon.generated.yaml` in the prompt); same model (`deepseek/deepseek-v4-flash` via OpenCode), same
pinned commit (`repos.json`). Each run gets a fresh export and a private OpenCode config with every
MCP server disabled (a port of cth.harness `createStrippedConfig`); B adds Beacon through
`OPENCODE_CONFIG_CONTENT`. Answers end in a fixed JSON block scored deterministically against gold
answers in `tasks/<repo>/<task>.json` (only owner-`reviewed` tasks run unless
`--include-unreviewed`). Paid: token use is summed from OpenCode events; the matrix stops before a
`--budget-tokens` cap (default 20M) and at the first rate-limit error, never retrying. Plan:
`IdeaProjects/.agent/plans/beacon-p3-agent-task-evaluation-plan-2026-09-23.md`. Offline tests:
`tests/test_beacon_eval.py` (a stub replaces OpenCode).

