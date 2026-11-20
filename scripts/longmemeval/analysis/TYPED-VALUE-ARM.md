# Typed Value Recall Arm

> **Code removed 2026-11-20 (streamlined-harness P2):** the rejected V2-V5 arms,
> `value_nodes_v2.py`, `tests/test_value_nodes_v2.py` and `oracle_entity_grouping_probe.py`
> were deleted; see git history before that date. V1 (`menhir_value_recall`) remains.

`menhir_value_recall` is a bench-only experiment for testing first-class `Value`
nodes before changing Menhir's graph schema.

The arm builds a deterministic sidecar graph from user haystack turns only. A
single `Value` abstraction carries an explicit kind: boolean, status, count,
duration, frequency, money, measurement, clock time, or weekday. Values are
immutable and assertion-scoped. Relevant value assertions are combined with
ordinary Menhir recall while preserving the same total recall limit.

Extraction is question-blind and gold-blind. It does not write to Neo4j or alter
Menhir. Equal values in unrelated facts remain separate nodes and skip entity
resolution. Boolean nodes require explicit polarity; absence or failure to
mention something remains unknown rather than becoming `false`.

## Fixture preflight

The original scalar-only prototype put an answer-supporting assertion in the
typed top-4 for 27 of the 29 adaptive-run cases with genuinely absent answer
support. The generalized arm adds explicit status and boolean values for the two
state misses: finishing a book and owning a spare screwdriver. With question-blind
extraction, a fixture rerun now puts answer support in the typed top-4 for all 29
cases.

Candidate-support coverage is not answer accuracy. A three-arm answer/judge run
is still required before proposing a production schema change.

## Run the answer comparison

Run the three-way comparison only against a completed, verified throwaway graph:

```powershell
$env:LONGMEMEVAL_VARIANT = "oracle"
archolith-bench harness longmemeval-menhir `
  --menhir-url http://localhost:8134 `
  --recall-only `
  --subset knowledge-update `
  --limit 78 `
  --arms no_memory,menhir_recall,menhir_value_recall `
  --recall-limit 10 `
  --scorer llm-judge `
  --judge-model gpt-4o-mini `
  --model gpt-4o
```

The experimental question is whether `menhir_value_recall` improves accuracy on
the 29 extraction-loss cases without reducing accuracy elsewhere. Treat this as
a representation simulation, not a production Menhir result: the sidecar uses
deterministic extraction and lexical retrieval rather than Menhir's eventual
value-node implementation.

## Answer-level results (2026-07-17)

Three-arm recall-only run against the completed adaptive graph
(`menhir-lme-ku-adaptive-full-20260717`, bolt 7698, recall server 8133), oracle
variant, gpt-4o answers + gpt-4o-mini judge, recall-limit 10, 78 knowledge-update
items. Independent verification run; artifacts in
`results/lme-ku-buildout/value-arm-verify-20260717/`.

| Arm | Overall (78) | 29 extraction-loss | Other 49 | Abstention (6) |
|-----|--------------|--------------------|----------|----------------|
| `no_memory` | 0.064 (5/78) | 0.000 (0/29) | 0.102 (5/49) | 5/6 |
| `menhir_recall` | 0.333 (26/78) | 0.000 (0/29) | 0.531 (26/49) | 4/6 |
| `menhir_value_recall` | **0.679 (53/78)** | **0.724 (21/29)** | **0.653 (32/49)** | 5/6 |

Per-item `menhir_recall` -> `menhir_value_recall`: 28 wrong->correct, 1 correct->wrong
(`50635ada`), 49 unchanged. No regression on the 49-set (26->32). `menhir_recall`
scored 0/29 on the loss set *with the same recall context*, so the gain is
attributable to the typed representation, not to more context (input tokens +5.7%
only; typed snippets displace ordinary snippets under the same top-10 cap, max 3
typed snippets/item). Reproduced across two runs (value arm 0.679 both times;
`menhir_recall` 0.346/0.333).

Abstention safety held: all 6 `_abs` items produced safe "I don't know" answers
under the value arm (zero confident-wrong); the near-miss typed values did not
flip abstentions to misses.

Residual value-arm failures are concentrated in stale/current ambiguity: when a
later value supersedes an earlier one for the same subject but the two source
sentences differ in wording, both survive as `current` (supersession keys on exact
subject text). 5 of the 8 remaining 29-set misses are this pattern (37 vs 38 coins,
4 vs 5 MCU films, 8:30 vs 7:30 wake time, three vs five sessions, and a
three-way-current gym clock time). Tightening supersession/recency labeling is the
top follow-on bench iteration before a production ingestion design commits schema.

## v2 supersession arm result (NEGATIVE, 2026-07-18)

A pre-registered supersession-aware iteration (`menhir_value_recall_v2_current`,
`menhir_value_recall_v2_history`; `SupersededValueGraph`, clustering by
entity/attribute/scope/kind with per-assertion correction markers then latest-learned)
was implemented (v1 untouched) and run as a 5-arm comparison on the same graph.
Artifacts: `results/lme-ku-buildout/value-arm-v2-verify-20260717/`.

| Arm | Overall | 29-loss | Other 49 | Abstention (6) |
|-----|---------|---------|----------|----------------|
| menhir_value_recall (v1) | 0.679 (53/78) | 21/29 | 32/49 | 5/6 |
| menhir_value_recall_v2_current | 0.667 (52/78) | 20/29 | 32/49 | 5/6 |
| menhir_value_recall_v2_history | 0.679 (53/78) | 21/29 | 32/49 | 5/6 |

**The hypothesis did not validate. Acceptance was not met (required >=3/5 supersession
targets recovered; got 0/5).**

- v2 clustering changed the typed context on only 6/78 items, and changed the outcome on
  none of them. On all 5 supersession targets the v2 typed context was byte-identical to
  v1 (clusters never merged).
- The one apparent regression (`5831f84d`, v2_current) is answer-model variance, not v2:
  the recalled context is byte-identical to v1; gpt-4o answered 12 vs 15 stochastically.
  Zero real v2-caused regressions. v2_history == v1 exactly (0 wins, 0 losses).

Root causes (from paired snippet evidence):
1. **Scope fragmentation.** Deterministic full-sentence scope derivation is too brittle for
   real multi-clause turns: incidental tokens (jog/coffee/yoga; bereavement/remember)
   fragment the cluster so the same attribute across turns does not merge. Unit tests on
   short clean sentences passed, but fixture sentences do not cluster.
2. **Structural ceiling.** The sidecar governs only ~3 of 10 slots; the untyped Menhir
   recall backfill reintroduces the stale value regardless of sidecar supersession.
3. **Target misclassification.** Not all "supersession" misses are supersession:
   `69fee5aa` needs arithmetic inference (37 stated + "added a new coin" -> 38; 38 is never
   stated literally), which no CURRENT-selection rule can fix.

Implication: canonical entity/attribute resolution needs the graph's real entity nodes
(Menhir's View/episode layer), not lexical sentence tokens, and a typed-value system must
control the whole recalled context (or rank authoritatively) rather than add a sidecar
alongside untyped recall. Do NOT commit a production supersession/schema design on this arm;
the supersession lever is unconfirmed at the answer level.

### Residual v1 miss reclassification (by mechanism)

Grounded in each item's question + typed context (checkpoint evidence):

| ID | Question | Gold | Mechanism |
|----|----------|------|-----------|
| `b6019101` | MCU films last 3mo | 5 | direct supersession (4 vs 5) |
| `f9e8c073` | sessions attended | five | direct supersession (three vs five) |
| `dad224aa` | wake time Saturdays | 7:30 | direct supersession (ambiguous: 7:30 preference vs 8:30 actual) |
| `69fee5aa` | coins owned | 38 | delta / arithmetic (37 + "added a new coin") |
| `c4ea545c` | gym more frequent than before? | Yes | comparative reasoning |
| `6071bd76` | coffee ratio more/less water | less (5 oz) | relation-direction (6->5 oz = less) |
| `6aeb4375` | Korean restaurants tried | four | elided referent ("four different ones") |
| `59524333` | usual gym time | 6:00 pm | answer-model/context competition (7:00/2:00 co-present) |

**True direct-supersession headroom is ~2-3 of 8.** The remaining 5-6 are delta/arithmetic,
comparative, relation-direction, elided-referent, or context-competition failures that no
CURRENT-selection rule can address. Conclusion: stop iterating on lexical supersession; a
production typed View must be authoritative during recall composition (suppress/demote
overlapping untyped snippets for an attribute family when the View has a confident current
value) and should carry two claim forms - **absolute** (`collection_size = 37`) and **delta**
(`collection_size += 1`) - so it can derive 38 while preserving both supporting episodes.
The narrowed production question: can Menhir derive entity-linked scalar state from episodes
and make that state authoritative during recall?

## v3 authoritative-composition result (MIXED - mechanism works, net-negative as-is; 2026-07-18)

Bench simulation of an authoritative typed View: coarse (near-oracle, within-item) grouping
to fix v2's clustering fragmentation, then `_v3_coarse` (current-only typed) and
`_v3_authoritative` (current-only + suppress untyped snippets carrying a superseded value).
Artifacts: `results/lme-ku-buildout/value-arm-v3-verify-20260718/`.

| Arm | Overall | 29-loss | Other 49 | Abstention (6) | 5 targets |
|-----|---------|---------|----------|----------------|-----------|
| menhir_recall | 0.359 | 0/29 | 28/49 | 5/6 | 0/5 |
| menhir_value_recall (v1, additive) | **0.705** | 22/29 | 33/49 | 5/6 | 1/5 |
| v3_coarse (grouping only) | 0.641 | 22/29 | 28/49 | 4/6 | 3/5 |
| v3_authoritative (grouping + suppression) | 0.679 | 23/29 | 30/49 | 5/6 | 4/5 |

(Same-run paired baseline; v1 ran 0.705 here vs 0.679 prior = answer/judge variance.)

**The mechanism is validated but nets negative as an unconditional rule:**
- v3 recovers the targeted misses: v3_coarse 3/5, v3_authoritative **4/5** supersession/competition
  targets (`b6019101`, `dad224aa`, `f9e8c073`, `59524333`) that additive v1 missed.
- The suppression lever is net-positive: v3_authoritative (0.679) > v3_coarse (0.641) - suppressing
  stale untyped snippets helped (also won `852ce960`, a money item).
- But coarse GROUPING over-merges distinct series, and authoritative single-pick then selects the
  WRONG current value and suppresses the right one. v3_authoritative: +4 wins, **-6 real regressions**
  (all v1-correct this run): `71315a70` (5-6h picked over gold 10-12h), `dfde3500` (Thursday/Maria
  over gold Wednesday/Juan), `e66b632c` (26:30 over gold 27:45), `89941a93`/`89941a94` (bike
  count/compose), `945e3d21`.

**Key lesson: authoritativeness is double-edged.** Additive v1 is robust because it shows multiple
candidates and lets the answer model choose; authoritative composition removes that safety net, so a
wrong current-pick is *worse* than additive (`71315a70` is the poster child: v1 showed 5-6h AND 10-12h
-> model answered 10-12; v3 showed only 5-6h -> model answered 5-6). `69fee5aa` (arithmetic) and the
bike-count items remain unreachable - selection can't derive unstated values.

**Refined design principle (next iteration): confidence-gated authoritativeness.** Collapse to a single
current value + suppress competitors ONLY when the View is confident (clean cluster, unambiguous recency
or explicit correction marker); otherwise fall back to additive candidate display. That keeps the 4 wins
while avoiding the 6 wrong-pick regressions. This is the same confidence-tiering the derivation
(assumptions/delta) layer needs. Arithmetic/filtered-count/coreference remain out of scope for any
selection mechanism and require reducer/aggregation/coref primitives.

## v4 advisory composition (OFFLINE prototype - advise, don't delete; 2026-07-18)

Direct implementation of the confidence-gated principle, resolved toward its safe extreme:
instead of *deciding* (collapse + suppress), the typed layer *advises* and the answer model
decides. Arm `menhir_value_recall_v4_advisory` (`SupersededValueGraph.advisory_recall`,
grouping="coarse"): additive candidate set (reuses v1 ranking, so answer-support >= v1 - it
can never drop the correct value), each multi-value cluster annotated, and the answer LLM makes
the final pick. Rationale from the v3 evidence: **every v3 regression came from DELETING the
correct candidate**, not from a bad label - so the fix is to stop deleting.

Confidence tiers (deterministic):
- **Delete + assert `current`** only for an unambiguous BINARY supersession: an explicit
  author rejection (`used to` / `no longer` / `previously` / `..., not <value>`) AND exactly one
  distinct value surviving to replace it. A bare present marker (`now`) is NOT enough (regex is
  noisy on multi-clause turns; `now` does not identify which sibling went stale - this drove the
  v3 wrong-`current` picks). Over-merged clusters (>1 surviving value, e.g. dfde3500's
  Juan/Wednesday merged with Maria/Thursday) fail the second condition and delete nothing.
- **`candidate` (advise only)** for every other multi-value cluster: alternatives for the same
  slot, no deletion, no unearned recency claim. The untyped Menhir backfill is left fully intact
  (unlike v3, advisory never suppresses untyped recall).

Offline validation (deterministic, no paid calls; `tests/test_value_nodes_v2.py`, real-fixture
probe over all 78 items):
- The three v3 deletion-regressions are structurally fixed - advisory RETAINS the correct value
  with zero mislabels: `71315a70` keeps 10-12h (all candidate), `dfde3500` keeps Wednesday (all
  candidate), `e66b632c` keeps its context (all candidate; the "previous"-value question is left
  to the model). `dad224aa` retains 7:30.
- **Corpus footprint (key finding): the clean-binary-supersession tier fires 0 times across all
  78 items - zero deletions, zero `current` labels.** 33 items get `candidate` annotations; the
  rest are single-value/plain. On this fixture advisory therefore reduces to *additive v1 + a
  candidate hint on 33 items, deleting nothing*, so its answer score is predicted ~= v1 (~0.70),
  with a strictly safer profile (no wrong-pick regressions possible) but no deletion-based wins
  (it does not reproduce v3's `b6019101`/`59524333`/`852ce960` suppression wins).

**Meta-conclusion (consistent v2 -> v3 -> v4): clean deterministic supersession is essentially
unreachable with lexical/coarse grouping on real multi-clause turns.** Coarse grouping over-merges
referents (a second value always survives -> never "binary"); lexical grouping fragments (never
merges). The deterministic tier needs the graph's real entity resolution, not sentence tokens -
exactly the v2/v3 finding. The robust, always-available lever is the advisory/candidate path: it
never regresses and hands the disambiguation to the strong answer model. A paid answer-level run
is NOT yet justified (predicted ~= v1); the offline result is the deliverable. Not wired into any
default config; opt-in via `--arms menhir_value_recall_v4_advisory`.

## v5 derivation / "assumptions" layer (OFFLINE prototype - advise a derived value; 2026-07-18)

Targets the residual miss class no *selection* mechanism can reach because the answer is never
stated literally: `69fee5aa` (coins owned, gold 38) = "37 coins" + "added a new coin". Arm
`menhir_value_recall_v5_derived` (`SupersededValueGraph.derived_recall`, grouping="coarse"):
the v4 advisory base plus a deterministic **delta fold** that surfaces one labeled
`[typed-value count derived] coin: ~38 (stated 37 +1: added a new coin)` hint - ADVISORY only,
never authoritative, never deleting a stated value (the v4 base is unchanged). Plan:
`.agent/plans/archolith-bench-typed-value-derivation-arm-plan.md`.

Scoped to ONE primitive (delta fold); comparative / relation-direction / filtered-count are
explicit follow-ons. Gated hard toward silence (the catastrophic failure for a derivation layer
is a confident-wrong number): a hint fires only when the counted noun is asked about; the noun
has exactly ONE distinct stated count across the whole item (a global anchor gate); >=1 signed
delta with the same noun occurs at-or-after the anchor; and the fold is integer-consistent,
non-negative, and not already a stated value. Delta magnitudes and temporal-reference years
("pre-1920") are excluded from anchors; bare "got N nouns" is NOT a delta (ambiguous with a new
absolute - it caused a confident-wrong ~45 on `4d6b87c8` in an early cut and was removed).

**Offline result (deterministic, no paid calls; all pre-registered gates pass):**
- **Fire rate 1/78.** Only `69fee5aa` derives -> `~38` (correct). All 3 offline acceptance gates
  pass: (1) `69fee5aa` -> 38; (2) zero derived hints on the 6 abstention fixtures; (3) zero
  derived hints on any of the 53 v1-correct items.
- The **bike items** (`89941a93`/`89941a94`, "compositional count", gold 4 stated directly) are
  the key negative check: an early per-cluster gate let a stray "my other two bikes" anchor fold
  to a wrong ~3; the **global** single-anchor gate correctly refuses to derive when a noun has
  multiple stated counts ({2,3,4}). The arm stays silent rather than guess - this is the
  anti-benchmax property, not a tuned win.

**Not benchmax (explicit):** the arm contains ZERO fixture-specific constants (no "coin", no
37/38, no item IDs); every gate is a general linguistic/extraction rule, and the unit tests
exercise the fold on non-fixture inputs (books, decrements, other numbers). The delta-fold is a
real, general memory capability (inventory/collection counts). **But with a 1/78 fire rate the
benchmark cannot measure its impact - a single-item delta is noise - so NO paid run is run and NO
accuracy claim is made.** The deliverable is: a general, provably-*safe*-on-this-fixture
derivation primitive (1 correct fire, 0 misfires across 77 items incl. all abstention and bike
traps), plus the honest finding that this KU subset barely exercises derivation. Real validation
needs a fixture with many delta-fold cases or production traffic. Not wired into any default
config; opt-in via `--arms menhir_value_recall_v5_derived`.

## Oracle entity-grouping probe (TERMINAL sidecar experiment; 2026-07-18)

The four sidecar variants (v2-v5) all hit the same wall from different angles: lexical/coarse
grouping fragments (v2) or over-merges (v3) or never fires its deterministic tier (v4/v5). To
isolate whether the blocker is truly **entity resolution** (vs attribute resolution or reasoning),
a read-only probe regrouped the typed assertions of the 8 residual-miss cases by a **hand-labeled
oracle (entity, attribute)** - bypassing lexical grouping entirely - and re-ran the REAL selection
(`_current_edge_ids`) and delta fold (`_derived_hints`). Offline, deterministic, no graph/cost.
Script: `scripts/longmemeval/analysis/oracle_entity_grouping_probe.py`.

| Item | Gold | Oracle-grouped outcome | Blocker class |
|------|------|------------------------|---------------|
| `dfde3500` | Wednesday | **FIXED** (Juan vs Maria separated) | entity resolution |
| `b6019101` | 5 | **FIXED** (MCU vs all films) | scope resolution |
| `59524333` | 6:00pm | **FIXED** (gym vs meeting time + recency) | entity resolution |
| `f9e8c073` | five | **FIXED** (recency in clean cluster) | (grouping ok; recency) |
| `dad224aa` | 7:30 | **FIXED** (7:30 supersedes 8:30) | (grouping ok; recency) |
| `69fee5aa` | 38 | **FIXED** (delta 37+1 after year-noise excluded) | entity resolution + delta |
| `71315a70` | 10-12h | **NOT ENTITY** (gold is the *earlier* mention) | reasoning: mention-order != truth |
| `e66b632c` | 27:45 | **NOT ENTITY** (asks for *previous* PB; both values present) | reasoning: previous-value question |

**Result: 6/8 recovered by perfect entity/scope grouping; 2/8 are not entity problems at all.**
- The 6 wins confirm the meta-conclusion mechanically: when assertions are grouped by resolved
  entity (and scope), supersession selection and delta derivation DO recover the gold value. The
  blocker for these was entity/scope resolution, not the selection logic - which already works.
- The 2 residual are reasoning, not attribute/entity resolution: `71315a70` needs to know the
  earlier value is authoritative (mention order is not truth order), and `e66b632c` explicitly
  asks for the *previous* (superseded) value, so any current-selection is wrong by construction.
  Both are exactly what the **v4 advisory** path already handles correctly - show both candidates
  and let the answer model reason - so they need NO new deterministic mechanism.

### Sidecar research: CLOSED

**Typed scalar representation is validated; lexical sidecar authority and lexical entity grouping
are rejected after four independent variants (v2 fragmentation, v3 over-merge/authority regression,
v4 zero-fire deterministic tier, v5 1/78 fire).** The oracle probe shows the remaining gains are
gated on real entity/scope resolution, which a lexical sidecar structurally cannot provide.
**Further progress requires integration with Menhir's entity and View layers**, not more sidecar
tuning. Design principle carried forward for the View: **authoritative selection when the value is
entity-resolved AND the question wants the current value; advisory (show candidates) when the
question needs reasoning (previous-value, ambiguous recency, comparison).** Arithmetic/filtered-
count/coreference remain separate primitives (reducer/aggregation/coref), out of scope for selection.