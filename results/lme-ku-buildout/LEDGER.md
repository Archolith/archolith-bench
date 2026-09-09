# LME Knowledge-Update Buildout Ledger

All runs use the same fixture: `fixtures/longmemeval/knowledge_update_subset.json` (78 items).
Frozen detector snapshots are saved in each run's results directory where applicable.

## Scoreboard

The scoreboard below is GENERATED from `ledger.csv`. Edit the CSV, then run
`python ../../scripts/longmemeval/lib/ledger.py render`. Do not hand-edit the table:
the next render overwrites it. Prose sections in this file are hand-maintained and are
never touched by the generator.

<!-- BEGIN GENERATED SCOREBOARD -- edit ledger.csv, then run ledger.py render -->

| Run ID | Date | N | Segmentation | Score | Status | Extract Model | Notes |
|--------|------|---|--------------|-------|--------|---------------|-------|
| matrix-frontier (prod graph) | 2026-07-12 | 15 | sentence (legacy) | 0.267 | scored | gpt-4.1-nano | Production 500-item graph, old extraction |
| matrix-node_plain (prod graph) | 2026-07-12 | 15 | sentence (legacy) | 0.267 | scored | gpt-4.1-nano | Same graph, node_plain retrieval |
| matrix-pointer (prod graph) | 2026-07-12 | 15 | sentence (legacy) | 0.200 | scored | gpt-4.1-nano | Same graph, pointer retrieval |
| abl-oracle (prod graph) | 2026-07-12 | 10 | sentence (legacy) | 0.300 | scored | gpt-4.1-nano | Same graph, oracle retrieval |
| recall-menhir (prod graph, all subsets) | 2026-07-12 | 100 | sentence (legacy) | 0.230 | scored | gpt-4.1-nano | All 6 subsets, not just knowledge-update |
| ku-fix-20260716 | 2026-07-17 | 7/78 | sentence | (killed) | killed | gpt-4o-mini | Killed — sentence splitting caused ~190 episodes/item (8x inflation) |
| ku-nosplit-20260716 | 2026-07-17 | 15 | none | 0.333 | scored | gpt-4o-mini | Fix stack: temp=0, gpt-4o-mini, dedup hardening |
| ku-split15-20260716 | 2026-07-17 | 8/15 | sentence | (killed) | killed | gpt-4o-mini | Killed — infeasible episode inflation |
| ku-adaptive-20260716 | 2026-07-17 | 15 | adaptive | 0.467 | scored | gpt-4o-mini | Frozen detector v1, +40% relative vs no-split |
| ku-adaptive-full-20260717 | 2026-07-17 | 78 | adaptive | 0.346 | scored | gpt-4o-mini | Full validation of adaptive segmentation |
| ku-nosplit-full-20260717 | 2026-07-17 | 53/78 | none | *(stopped partial)* | partial | gpt-4o-mini | 50 healthy, 3 drain timeouts; unscored; graph volume preserved |
| value-arm-verify-20260717 | 2026-07-17 | 78 | adaptive (reused) | 0.679 | scored | gpt-4o-mini | Three-arm value-recall verification on the ku-adaptive-full graph; see below |
| value-arm-v2-verify-20260718 | 2026-07-18 | 78 | adaptive (reused) | v2c 0.667 / v2h 0.679 | multi_arm | gpt-4o-mini | Pre-registered supersession arms; NEGATIVE (0/5 targets, no lift); see below |
| value-arm-v3-verify-20260718 | 2026-07-18 | 78 | adaptive (reused) | v3c 0.641 / v3a 0.679 (v1 0.705) | multi_arm | gpt-4o-mini | Authoritative composition; MIXED - recovers 4/5 targets but net-neg from over-merge; needs confidence-gating |
| v4 advisory (offline) | 2026-07-18 | 78 | n/a (offline) | *(no paid run)* | offline | n/a | Advise-don't-delete; clean-supersession tier fires 0/78; = additive v1 + candidate hints, predicted ~= v1 |
| v5 derived (offline) | 2026-07-18 | 78 | n/a (offline) | *(no paid run)* | offline | n/a | Delta-fold "assumptions" arm; fires 1/78 (69fee5aa -> ~38 correct); all offline gates pass; NOT benchmax, no paid run on n=1 |
| scalar-ku-20260722 | 2026-07-22 | 78 | adaptive | *(measure-only, no QA)* | measure_only | gpt-4o-mini | Scalar-consolidation MEASURE run (k=3). Materialization, not recall-scored: 18/78 (23%) scalar views, 20/78 typed. See section below + .agent/reviews/menhir-lme-scalar-ku-20260722-results.md |
| scalar-current-candidate-v3-20260728 | 2026-07-29 | 32/78 | adaptive | (killed / INVALID) | invalid | gpt-4o-mini | Mixed-code/provenance run; item 2133c1b5 was consolidated with one real FAILED episode under threshold 2. Do not resume, score, or compare. |
| scalar-canonical-ku78-v1-20260806 | 2026-08-06 | 78 | adaptive | 0.872 | scored | gpt-4o-mini | CANONICAL BENCHMARK EVIDENCE. Fresh candidate-arm ingest; 68/78 recall vs 6/78 (0.077) no-memory; harness exit 0. |
| scalar-event-activity-ku78-v2-20260809 | 2026-08-09 | 0/78 | adaptive | *(aborted pre-manifest)* | aborted | gpt-4o-mini | Launch attempt produced provenance only; no graph result or semantic evidence. |
| scalar-event-activity-ku78-v3-20260809 | 2026-08-09 | 0/78 | adaptive | *(aborted pre-manifest)* | aborted | gpt-4o-mini | Launch reached fresh-graph startup but produced no manifest or score; container/volume later absent. |
| scalar-event-activity-ku78-v4-20260809 | 2026-08-09 | 78 | adaptive | 0.885 | scored | gpt-4o-mini | Fresh clean run; 69/78 recall vs 6/78 (0.077) no-memory; harness exit 0. Superseded by v6. |
| scalar-event-activity-ku78-v5-20260809 | 2026-08-09 | 0/78 | adaptive | *(launch refused)* | aborted | gpt-4o-mini | No ingest: first wrapper call refused an existing result directory; replacement attempt left provenance only. No graph/container. |
| **scalar-event-activity-ku78-v6-20260809** | 2026-08-09 | 78 | adaptive | **0.910** | scored | gpt-4o-mini | CURRENT CANONICAL BENCHMARK EVIDENCE. Fresh clean run; 71/78 recall vs 6/78 (0.077) no-memory; harness exit 0. |

<!-- END GENERATED SCOREBOARD -->

## 2026-08-09 — Event/activity scalar KU78 campaign

The v2, v3, and v5 attempts ended before a manifest row was written and provide no semantic or
score evidence. V4 and v6 each used a fresh, non-resumed graph, the unchanged 78-item oracle
fixture (SHA256 `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`), a two-item
checkpoint, adaptive segmentation, concurrency 2, 2/3 scalar agreement, `k=3`, 1/1/1
attribute/scope/subject reconciliation, scalar state/history and View authority, TurnEvidence,
and a zero-failed-episodes-per-namespace policy.

V4 used Menhir `9d9675c9397770b5bc654ce6f15da315d15c616a` and Bench
`b7a275403d413f4c9a7f92cd2ac5df9eae38b3a0`. It completed 78/78 manifest rows with cumulative
`failed_remaining=0`, final PENDING/ENRICHING/FAILED counts all zero, and harness exit 0. Menhir
recall scored **69/78 (0.884615; displayed 0.885)** with 117,104 input tokens, 1,494 output tokens,
and `$0.307697` in the scored arm. Provider-reported combined run usage was 17,413,642 tokens.

V6 used Menhir `1fa57955b24f90d08550c911f26133e5b14cbb89` and Bench
`d5e97cc4fc322564c624a749e2cb25dccdf9c2ea`. Event History and Event History authority were enabled;
the deterministic scalar router and shadow paths were disabled. The two-item checkpoint passed
before release. The full run completed 78/78 manifest rows with cumulative `failed_remaining=0`,
final PENDING/ENRICHING/FAILED counts all zero, and harness exit 0. Menhir recall scored
**71/78 (0.910256; displayed 0.910)** with 117,933 input tokens, 1,376 output tokens, and
`$0.308592` in the scored arm. Provider-reported combined run usage was 17,516,332 tokens. This is
the current best canonical KU78 result: +2 correct versus v4 and +3 versus the 68/78 canonical
baseline. It is benchmark evidence, but it is not automatically an approved launch headline.

### Per-Item Results (`scalar-event-activity-ku78-v4-20260809`, `menhir_recall`)

PASS (69): 6aeb4375 830ce83f 852ce960 945e3d21 71315a70 89941a93 ce6d2d27 9ea5eabc
07741c44 a1eacc2a 184da446 031748ae 4d6b87c8 0f05491a 08e075c7 f9e8c073 41698283
2698e78f b6019101 45dc21b6 6071bd76 e493bb7c 618f13b2 72e3ee87 c4ea545c 01493427
6a27ffc2 2133c1b5 18bc8abd db467c8c 7a87bd0c e61a7584 1cea1afa ed4ddc30 8fb83627
b01defab 22d2cb42 0e4e4c46 4b24c848 7e974930 603deb26 59524333 5831f84d eace081b
affe2881 50635ada e66b632c 0ddfec37 f685340e cc5ded98 dfde3500 69fee5aa 7401057b
cf22b7bf 06db6396 3ba21379 9bbe84a2 10e09553 dad224aa ba61f0b9 42ec0761 5c40ec5b
c6853660 0977f2af 6aeb4375_abs 2698e78f_abs 2133c1b5_abs 0ddfec37_abs f685340e_abs

FAIL (9): 6a1eabeb d7c942c3 5a4f22c0 a2f3aa27 c7dc5443 26bdc477 031748ae_abs
89941a94 07741c45

### Per-Item Results (`scalar-event-activity-ku78-v6-20260809`, `menhir_recall`)

PASS (71): 6a1eabeb 6aeb4375 830ce83f 852ce960 945e3d21 d7c942c3 71315a70 89941a93
ce6d2d27 9ea5eabc 07741c44 a1eacc2a 184da446 031748ae 4d6b87c8 0f05491a 08e075c7
41698283 2698e78f b6019101 45dc21b6 5a4f22c0 6071bd76 e493bb7c 618f13b2 72e3ee87
01493427 6a27ffc2 2133c1b5 18bc8abd db467c8c 7a87bd0c 1cea1afa ed4ddc30 8fb83627
b01defab 22d2cb42 0e4e4c46 4b24c848 7e974930 603deb26 59524333 5831f84d eace081b
affe2881 50635ada e66b632c 0ddfec37 f685340e cc5ded98 dfde3500 69fee5aa 7401057b
cf22b7bf c7dc5443 06db6396 3ba21379 9bbe84a2 10e09553 dad224aa ba61f0b9 42ec0761
5c40ec5b c6853660 0977f2af 6aeb4375_abs 2698e78f_abs 2133c1b5_abs 0ddfec37_abs
f685340e_abs 89941a94

FAIL (7): f9e8c073 c4ea545c e61a7584 a2f3aa27 26bdc477 031748ae_abs 07741c45

V6 fixed five v4 misses (`5a4f22c0`, `6a1eabeb`, `89941a94`, `c7dc5443`, `d7c942c3`) and
regressed three v4 passes (`c4ea545c`, `e61a7584`, `f9e8c073`). Inspection found that the needed
evidence remained present for those three regressions, so they are not evidence for broad
production-rule changes. Of the seven v6 misses, the clearest deterministic defect is
`26bdc477`: `trip_count=3` and `trip_count=5` were both minted but remained `binding_pending`
because possessive `my camera` did not bind to the co-mentioned `Canon EOS 80D camera`. Keep this
as backlog until an unrelated, non-benchmark panel establishes the general alias pattern.

Key v6 artifact SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| `manifest.json` | `21c6a0b09d413d0b87dac5aa2d7e28d827db739f80240a1071e64cb32dce64ee` |
| `run_provenance.json` | `04eea619eccb2d4eb77ee0beb0684cff628c6a2199c81813298fec7a7e354e5e` |
| `run_llm_usage.json` | `0035ecea36804c3f743b0e108764724b3a0de3d4a1ef08956200c90ee7d736fe` |
| `harness_recall/.checkpoint_longmemeval-menhir_oracle_gpt-4o.jsonl` | `9d4ed6c7ae91b736a11e698b0c98ee09c239a46f5d64b494c7a8865b5b4adef3` |
| `harness_recall/results.md` | `bdd05e1444582bd07bf6035e62f2cbc14315863c86882f4820b0c4356de9b73d` |

## Invalidated Run (scalar-current-candidate-v3-20260728)

Stopped at 32/78 manifested items on 2026-07-29. The result directory, stopped container, volume,
logs, manifest, and graph are preserved for diagnosis, and the directory contains
`INVALID-RUN.md`.

This run is not benchmark evidence:

- its active provenance still claims Menhir `012b0c5` and Bench `04fbe9f`, while later windows ran
  through additional revisions including Menhir `8ceafe4` and Bench `02fed32`;
- namespace `lme-2133c1b5` was consolidated and manifested with `failed_remaining=1` after the log
  reported `tolerated 1 FAILED episode(s) under threshold 2`;
- the failed user episode contained a real personal fact about not exploring historical Harajuku,
  not an approved no-op;
- adaptive context folding could relabel assistant-authored text as user evidence;
- Menhir's repair receipt could overwrite the first extraction pass and falsely accept a real
  relationless under-extraction as policy-empty; and
- stale-worker cleanup did not prove that late writes after namespace reset were purged.

Replacement work must use fresh storage and immutable per-phase code/settings provenance after the
reviewed correctness fixes pass. No rows from this run may be repaired in place or included in a
score.

## Scalar Consolidation Materialization (scalar-ku-20260722) — MEASURE

MEASURE-stage run of scalar-state consolidation (phase3, k=3, call-budget 50) over all 78 KU
items. **This is a materialization measurement, NOT a recall/QA-scored run** — there is no
comparable accuracy Score. Fresh per-item graph, adaptive segmentation, gpt-4o-mini.
Manifest: `results/lme-ingest-scalar-ku-20260722/manifest-menhir-lme-scalar-ku-20260722.json`.
Full analysis: `.agent/reviews/menhir-lme-scalar-ku-20260722-results.md`.

Build integrity: 78/78 consolidated, 0 drain timeouts, all 78 `turn_evidence > 0`,
validation PASS, promote + backfill complete.

### Materialization funnel (all 78 items have a scalar answer by construction)

| Stage | Cases | Rate |
|-------|-------|------|
| turn_evidence > 0 (perception ran) | 78/78 | 100% |
| typed_assertions > 0 | 20/78 | 26% |
| scalar_views > 0 (durable current view) | 18/78 | **23%** |

Corpus sums: turn_evidence 963, typed_assertions 32, scalar_states_written 29, scalar_views 21,
all 21 user-founded.

### Key finding: the cliff is scalar-perception abstention, NOT episode loss

- Episode-collapse rate is statistically identical in view-producers (0.281) and non-producers
  (0.267) — extraction collapse does NOT explain the sparse view rate.
- The loss is at turn_evidence -> typed_assertion, inside the k=3 consistency gate
  (`gate_typed_scalars`, default threshold=1.0 unanimous). 58/78 abstain despite ample evidence.
- Follow-up plan: `.agent/plans/menhir-scalar-perception-abstention.md`.

### Separate systemic defect surfaced (not the cause of the above)

- ~25% of all episodes FAILED, 100% `combined_extraction_collapsed` (797/797, all 78 namespaces).
  Thins the graph; matters for recall. Plan: `.agent/plans/menhir-combined-extraction-collapse.md`.

### Ingest defects fixed to reach a clean build (harness, uncommitted WIP)

- Premature-settle drain race (gate on PENDING+ENRICHING, not just ENRICHING).
- Cross-namespace `session_id` collision (namespace-qualify session_id); menhir-side root cause in
  `.agent/plans/menhir-cross-namespace-session-id-contamination.md` (+ namespace-reset does not
  delete Episodic nodes). Both in `scripts/longmemeval/lib/ingest.py`.
- Note: `89941a94` carries reset pollution (episodes=124 vs ~40); view output unaffected (0).

### Infra

Container `menhir-lme-scalar-ku-20260722`, bolt 7701, HTTP 7488, build port 8137,
namespace prefix `lme-scalar-ku-20260722-`.

## Typed Value v2 Supersession Arms (value-arm-v2-verify-20260718) — NEGATIVE

Pre-registered 5-arm run (no_memory, menhir_recall, v1, v2_current, v2_history) on the
same `menhir-lme-ku-adaptive-full-20260717` graph. Artifacts:
`results/lme-ku-buildout/value-arm-v2-verify-20260718/` (frozen `value_nodes_v2_frozen.py`).

| Arm | Overall | 29-loss | Other 49 | Abstention (6) |
|-----|---------|---------|----------|----------------|
| menhir_value_recall (v1) | 0.679 (53/78) | 21/29 | 32/49 | 5/6 |
| menhir_value_recall_v2_current | 0.667 (52/78) | 20/29 | 32/49 | 5/6 |
| menhir_value_recall_v2_history | 0.679 (53/78) | 21/29 | 32/49 | 5/6 |

- **Acceptance NOT met**: 0/5 supersession targets recovered (needed >=3/5). No aggregate lift.
- v2 changed typed context on only 6/78 items; changed outcome on none. All 5 targets byte-identical to v1 (clusters never merged).
- Zero real regressions: the one flagged item (`5831f84d`) is answer-model variance (context identical to v1).
- Root causes: (1) full-sentence scope fragmentation prevents clustering on real turns; (2) untyped Menhir backfill reintroduces stale values (sidecar governs only ~3/10 slots); (3) `69fee5aa` is arithmetic inference, not supersession.
- Recommendation: do not commit a production supersession/schema design on this arm; resolve canonical entity/attribute in Menhir's View/episode layer and control the whole recalled context.

## Typed Value Recall Verification (value-arm-verify-20260717)

Independent verification of the bench-only `menhir_value_recall` arm. Recall-only,
three arms, reusing the completed `menhir-lme-ku-adaptive-full-20260717` graph
(restored over its preserved volume; container had been removed, volume intact).
gpt-4o answers, gpt-4o-mini judge, recall-limit 10, oracle variant, 78 KU items.
Artifacts: `results/lme-ku-buildout/value-arm-verify-20260717/`.

| Arm | Overall (78) | 29 loss-set | Other 49 | Abstention (6) |
|-----|--------------|-------------|----------|----------------|
| no_memory | 0.064 | 0/29 | 5/49 | 5/6 |
| menhir_recall | 0.333 | 0/29 | 26/49 | 4/6 |
| menhir_value_recall | **0.679** | **21/29** | **32/49** | 5/6 |

- Per-item menhir->value: 28 wins, 1 loss (`50635ada`), 49 unchanged. No 49-set regression.
- Recall-limit invariant held (max 3 typed snippets/item, merged capped at 10, input tokens +5.7%).
- Abstention safety held: all 6 `_abs` items answered "I don't know" under the value arm.
- Reproduced over two runs (value 0.679 both; menhir_recall 0.346/0.333).
- Residual misses concentrated in stale/current supersession ambiguity (5 of 8 remaining 29-set misses).
- The ku-adaptive-full graph carries 3 documented failed episodes (`945e3d21`, `2698e78f`,
  `2698e78f_abs`, truncation; none in the 29-set); they affect both recall arms equally.

## Fix Stack (applied to all ku-* runs)

- `temperature = 0` — deterministic extraction (eliminates stochastic entity conflation)
- `extract_model = gpt-4o-mini` — replaces gpt-4.1-nano for higher extraction quality
- Dedup identity gate hardening — prevents false entity merges
- Fresh per-item Neo4j graph (not the shared 500-item production graph)

## Adaptive Claim Segmentation (ku-adaptive-* runs)

Two-stage detector:
- **Stage A**: Deterministic gate — regex for correction markers, state-change verbs, buried updates, topic shifts
- **Stage B**: Heuristic claim extraction — extracts sentences with correction/state-change signals

Four modes: `SKIP`, `CONTEXT_ONLY`, `EXTRACT_WHOLE`, `SEGMENT_CLAIMS`

Assistant turns default to `CONTEXT_ONLY` (eliminates ~80% of episode inflation).
Detector frozen before full-78 run — snapshot at `ku-adaptive-full-20260717/claim_segmenter_frozen.py`.

Source: `scripts/longmemeval/lib/claim_segmenter.py`

## Per-Item Results (ku-adaptive-full-20260717)

PASS (27): 6a1eabeb d7c942c3 ce6d2d27 a1eacc2a 031748ae 41698283 45dc21b6 5a4f22c0
01493427 18bc8abd 8fb83627 22d2cb42 4b24c848 50635ada e66b632c c7dc5443 06db6396
10e09553 ba61f0b9 5c40ec5b 26bdc477 6aeb4375 2698e78f 2133c1b5 0ddfec37 f685340e 07741c45

FAIL (51): 6aeb4375 830ce83f 852ce960 945e3d21 71315a70 89941a93 9ea5eabc 07741c44
184da446 4d6b87c8 0f05491a 08e075c7 f9e8c073 2698e78f b6019101 6071bd76 e493bb7c
618f13b2 72e3ee87 c4ea545c 6a27ffc2 2133c1b5 db467c8c 7a87bd0c e61a7584 1cea1afa
ed4ddc30 b01defab 0e4e4c46 7e974930 603deb26 59524333 5831f84d eace081b affe2881
0ddfec37 f685340e cc5ded98 dfde3500 69fee5aa 7401057b cf22b7bf a2f3aa27 3ba21379
9bbe84a2 dad224aa 42ec0761 c6853660 0977f2af 031748ae 89941a94

Items with failed episodes: 945e3d21 (1 failed), plus 2 others from truncation errors

## Per-Item Results (ku-nosplit-20260716, 15-item)

PASS (5): 6a1eabeb 830ce83f d7c942c3 ce6d2d27 031748ae
FAIL (10): 6aeb4375 852ce960 945e3d21 71315a70 89941a93 9ea5eabc 07741c44 a1eacc2a 184da446 4d6b87c8

## Per-Item Results (ku-adaptive-20260716, 15-item)

PASS (7): 6a1eabeb 830ce83f d7c942c3 71315a70 ce6d2d27 a1eacc2a 4d6b87c8
FAIL (8): 6aeb4375 852ce960 945e3d21 89941a93 9ea5eabc 07741c44 184da446 031748ae

## Regression Analysis

### 031748ae (SEG_R5 — retrieval variance)
- No-split 15: PASS → Adaptive 15: FAIL → Adaptive full: PASS
- Classified SEG_R5: retrieval variance on identical input, not segmentation-caused
- The segmenter made zero changes to this fixture's episodes

### First-15 subset variance across runs
- No-split 15: 5/15 = 0.333
- Adaptive 15: 7/15 = 0.467
- Adaptive full (first-15 only): 6/15 = 0.400
- Items flip both directions between runs — retrieval and answer-model variance

## Run Infrastructure

| Run ID | Container | Bolt | HTTP | Build Port | Recall Port |
|--------|-----------|------|------|------------|-------------|
| ku-fix-20260716 | menhir-lme-ku-fix-20260716 | 7694 | 7481 | 8124 | 8125 |
| ku-nosplit-20260716 | menhir-lme-ku-nosplit-20260716 | 7695 | 7482 | 8126 | 8127 |
| ku-split15-20260716 | menhir-lme-ku-split15-20260716 | 7696 | 7483 | 8128 | 8129 |
| ku-adaptive-20260716 | menhir-lme-ku-adaptive-20260716 | 7697 | 7484 | 8130 | 8131 |
| ku-adaptive-full-20260717 | menhir-lme-ku-adaptive-full-20260717 | 7698 | 7485 | 8132 | 8133 |
| ku-nosplit-full-20260717 | menhir-lme-ku-nosplit-full-20260717 | 7699 | 7486 | 8134 | 8135 |
| scalar-current-candidate-v3-20260728 | menhir-lme-scalar-current-candidate-v3-20260728 *(stopped, preserved)* | 7694 | — | 8124 | 8125 |
| scalar-event-activity-ku78-v2-20260809 | none *(aborted before usable graph)* | 7720 | 7507 | 8160 | 8161 |
| scalar-event-activity-ku78-v3-20260809 | none *(aborted pre-manifest; removed)* | 7720 | 7507 | 8160 | 8161 |
| scalar-event-activity-ku78-v4-20260809 | menhir-lme-scalar-event-activity-ku78-v4-20260809 *(stopped, volume preserved)* | 7720 | 7507 | 8160 | 8161 |
| scalar-event-activity-ku78-v5-20260809 | none *(launch refused/pre-ingest)* | 7721 | 7508 | 8163 | 8164 |
| scalar-event-activity-ku78-v6-20260809 | menhir-lme-scalar-event-activity-ku78-v6-20260809 *(running, volume preserved)* | 7721 | 7508 | 8163 | 8164 |

The legacy July `ku-*` rows above used Menhir `6d37255` and Bench `8209c20`. Later scalar campaign
commits are recorded in their dedicated sections and run provenance.

## Fixture

- Path: `fixtures/longmemeval/knowledge_update_subset.json`
- SHA256: `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`
- 78 items, each with 24 turns across 2 sessions
- Oracle variant (ground-truth answers available)
- Extracted from full LongMemEval dataset (500 items, ~250k turns)
- Question type: knowledge-update (facts that change during the conversation)
- Keys per item: `question_id`, `question_type`, `question`, `answer`, `question_date`,
  `haystack_dates`, `haystack_session_ids`, `haystack_sessions`, `answer_session_ids`
- First item: `6a1eabeb`, last item: see manifest for ordering

## Known Issues

- **Event/activity KU78 residuals are heterogeneous**: v6's seven misses include answer-selection
  variance with relevant evidence present, approximate/current-value policy, a synthetic unsupported
  role premise, and plan-versus-observation semantics. Do not tune broad production behavior to the
  miss list.
- **Possessive-to-specific object binding remains unresolved**: in `lme-26bdc477`, `my camera` did
  not bind to the exact co-mentioned `Canon EOS 80D camera`, leaving both trip-count assertions
  pending. Require independent generic evidence before adding provenance-bound alias resolution.
- **Invalid mixed-code scalar candidate run**: `scalar-current-candidate-v3-20260728` stopped at
  32/78 and is explicitly quarantined. Its provenance is stale and it consolidated a namespace
  with one real failed episode. It must never be resumed or used as comparative evidence.
- **JSON truncation in extraction**: gpt-4o-mini truncates output at ~3100 chars on episodes
  with rich entity context. Hits ~4% of episodes. Fix implemented in
  `menhir/src/menhir/infrastructure/graphiti_patches.py` (truncation-aware retry escalation
  with max_tokens doubling) but not yet deployed during these runs.
- **Retrieval variance**: Same graph + same query can produce different results across runs
  due to embedding similarity ties and non-deterministic entity summary ordering.

## 2026-07-29 — `scalar-duration-candidate-v3-20260729` (diagnostic, failed closed)

- Purpose: fresh two-item candidate test of colon-duration normalization without changing
  the extraction prompt.
- Code used by the run: Menhir `667f569`; Bench `6a0d060`.
- Fixture SHA256: `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.
- Configuration: candidate arm, 2/3 agreement, 1/1/1 reconcile, checkpoint 2,
  concurrency 2, scalar history enabled.
- Ingest completed with 46/46 ingest episodes ready and no remaining failed episodes.
- Item `6a1eabeb`: cadence persisted, but the old and new race times each survived only
  1/3 extraction samples. Seven candidates failed validation, and the surviving new
  `25:50` personal-best candidate incorrectly used `expire` instead of `absolute`.
  This establishes that the parser normalization works when given a duration, but an
  extraction-prompt change is still needed for consistent classification and operation.
- Item `6aeb4375`: the current restaurant count (`4`) and rice status persisted, but the
  prior restaurant count (`3`) was not extracted in this stochastic pass.
- Scalar-history postflight failed because the REST Phase 3 handler did not forward the
  enabled scalar-history setting. The run aborted before checkpoint/continuation and
  recall scoring. This wiring defect was fixed afterward in Menhir `1b2efd2`; 58 focused
  API and scalar-history tests passed.
- Preserved container for inspection:
  `menhir-lme-scalar-duration-candidate-v3-20260729` (Bolt 7697, HTTP 7484).

## 2026-07-30 — `scalar-duration-prompt-candidate-v1-20260730` (checkpoint diagnostic)

- Purpose: fresh two-item candidate test after the contextual colon-time extraction prompt.
- Code used by the run: Menhir `be63b4f`; Bench `6a0d060`.
- Fixture SHA256: `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.
- Configuration: candidate arm, 2/3 agreement, 1/1/1 reconcile, checkpoint 2,
  concurrency 2, scalar history enabled.
- Docker Desktop paused during startup. The runner remained in its pre-write stale-settlement
  guard and began evidence/memory submissions only after Docker resumed, so the fresh graph
  boundary was preserved. The ingest then completed with 46/46 enrichments successful, zero
  failed episodes, zero drain timeouts, and scalar postflight PASS.
- Item `6a1eabeb`: the old `27:12` personal best was correctly classified as
  `duration|seconds|absolute|1632` by 2/3 samples and materialized at source time
  `2023-05-25T20:21:00Z`. The later `25:50` update was present in TurnEvidence at source time
  `2023-05-27T10:20:00Z` and one sample correctly proposed
  `duration|seconds|absolute|1550`, but the other two samples omitted it. The 2/3 gate therefore
  abstained, leaving the current View incorrectly at 1632 and the history with one entry.
- Item `6aeb4375`: both restaurant totals were unanimous (3/3), materialized as source-time
  absolute counts `3` then `4`, and produced a current View of 4 plus a two-entry scalar history.
- Conclusion: colon-duration normalization, source-time provenance, current/history projection,
  supersession, and the 2/3 gate are working. The prompt improved the old race observation from
  1/3 to 2/3 and corrected its operation, but did not make the newer personal-best observation
  reliable. This checkpoint must not continue to the remaining 76 items. The remaining defect is
  extraction recall when twelve user turns are presented in one scalar-perception batch; it is not
  a parser or projection defect.
- Preserved container for inspection:
  `menhir-lme-scalar-duration-prompt-candidate-v1-20260730` (Bolt 7698, HTTP 7485).

## 2026-07-30 — `scalar-duration-completeness-candidate-v1-20260730` (infrastructure failure)

- Purpose: first fresh two-item checkpoint after the batched-observation completeness prompt.
- Code used by the run: Menhir `d07ae8c`; Bench `6a0d060`.
- Fixture SHA256: `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.
- Docker Desktop froze after Neo4j startup. The host Bolt port accepted connections but did not
  return a handshake, and the ingest safety guard refused to reset either namespace while their
  state could not be read.
- The run produced no manifest and no checkpoint marker, so it supplies no semantic evidence and
  must not be resumed or compared. Its exited container, volume, results, and logs are preserved.
- Preserved container:
  `menhir-lme-scalar-duration-completeness-candidate-v1-20260730`
  (Bolt 7702, HTTP 7489, build 8138, recall 8139).

## 2026-07-30 — `scalar-duration-completeness-candidate-v2-20260730` (checkpoint PASS)

- Purpose: replacement fresh two-item candidate checkpoint after Docker restarted, testing the
  committed batched-observation completeness prompt at the original twelve-user-turn batch size.
- Code used by the run: Menhir `d07ae8c`; Bench `6a0d060`.
- Fixture SHA256: `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.
- Configuration: candidate arm, fresh graph and volume, 2/3 agreement, `k=3`, 1/1/1 reconcile,
  checkpoint 2, concurrency 2, scalar state/history and turn evidence enabled.
- Ingest completed with 46/46 enrichments successful, zero remaining failed episodes, zero drain
  timeouts, both namespaces consolidated, and the hard checkpoint reached.
- Item `6a1eabeb`: both race observations were unanimous (3/3):
  `duration|seconds|absolute|1632` at `2023-05-25T20:21:00Z`, then
  `duration|seconds|absolute|1550` at `2023-05-27T10:20:00Z`. The active scalar state is 1550,
  and its active history contains both values in source-time order.
- Item `6aeb4375`: restaurant counts `3` at `2023-08-11T09:00:00Z` and `4` at
  `2023-09-30T18:01:00Z` were each unanimous (3/3). The active state is 4 and its history has both
  entries. A separate unanimous `rice_type` status was also materialized without colliding with the
  restaurant-count slot.
- Graph postflight: five typed assertions, each with exactly one user TurnEvidence founder; no
  binding/projection/activation pending flags; no pending projection repairs; one active state and
  one active history per scalar slot; every active history's declared count matches its
  `HISTORY_ENTRY` edges; current anchors point to 1550, 4, and the independent rice status.
- Audit passes `cap-501974fabdb14f27` and `cap-a356309c1f874350` show unanimous gate commits,
  source-time binding, supersession, and two-entry history rebuilds for the target updates.
- Provenance records a fresh, non-resumed graph, exact commits and fixture hash. Both repositories
  were tracked-clean. The graph provenance's dirty booleans reflect only pre-existing untracked
  review documents and the hash-pinned fixture, not tracked code changes.
- The continuation marker is deliberately absent. The runner and Neo4j remain up at the hard
  checkpoint for independent review.
- Container:
  `menhir-lme-scalar-duration-completeness-candidate-v2-20260730`
  (Bolt 7703, HTTP 7490, build 8140, recall 8141).

## 2026-08-06 — `scalar-canonical-ku78-v1-20260806` (canonical PASS)

- **Evidence status: CANONICAL.** This is the authoritative 78-item scalar knowledge-update result
  and is eligible for future comparisons. “Candidate arm” names the evaluated benchmark arm; it
  does not make the run noncanonical. Earlier interrupted, resumed, diagnostic, and mixed-code
  attempts remain development evidence only and must not be substituted for this result.
- Purpose: fresh canonical 78-item candidate-arm ingest and recall validation of the current scalar
  state/history implementation.
- Result: Menhir recall **68/78 (0.872)**; no-memory baseline **6/78 (0.077)**; delta **+0.795**.
- Provenance: one fresh, non-resumed attempt; 78 manifested items; recall harness exit 0; both
  repositories tracked-clean at launch.
- Configuration: adaptive segmentation, 2/3 scalar threshold, `k=3`, 1/1/1 attribute/scope/subject
  reconciliation, concurrency 2, scalar state/history and View authority enabled. Deterministic
  router, deterministic classes, and shadow research paths were explicitly disabled.
- Code identity: Menhir `2825c8d66ea2ced646ee64fcb0d6d9f433633dc0`; Bench
  `9a2074bec9f97d225d0f97ad6a33c9fd739b9b15`.
- Fixture: `fixtures/longmemeval/knowledge_update_subset.json`, SHA256
  `bba252a302e7b257a0f7457fe97411f7de144aafd1c6b44c98a0e88ee8570907`.
- Artifacts: `results/lme-ku-buildout/scalar-canonical-ku78-v1-20260806/`.

### Per-Item Results (`menhir_recall`)

PASS (68): 6a1eabeb 6aeb4375 830ce83f 852ce960 945e3d21 71315a70 89941a93 ce6d2d27
9ea5eabc 07741c44 a1eacc2a 184da446 031748ae 4d6b87c8 0f05491a 08e075c7 2698e78f
b6019101 45dc21b6 5a4f22c0 6071bd76 e493bb7c 618f13b2 72e3ee87 c4ea545c 01493427
6a27ffc2 2133c1b5 18bc8abd db467c8c 7a87bd0c e61a7584 1cea1afa ed4ddc30 8fb83627
b01defab 22d2cb42 0e4e4c46 4b24c848 7e974930 603deb26 59524333 5831f84d eace081b
affe2881 50635ada e66b632c 0ddfec37 f685340e cc5ded98 dfde3500 7401057b cf22b7bf 06db6396
3ba21379 9bbe84a2 10e09553 dad224aa ba61f0b9 42ec0761 5c40ec5b c6853660 6aeb4375_abs
2698e78f_abs 2133c1b5_abs 0ddfec37_abs f685340e_abs 89941a94

FAIL (10): d7c942c3 f9e8c073 41698283 69fee5aa a2f3aa27 c7dc5443 26bdc477 0977f2af
031748ae_abs 07741c45
