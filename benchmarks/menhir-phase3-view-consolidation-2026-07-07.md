# Menhir Phase 3 View Consolidation — 2026-07-07

<!-- archolith-evidence
product: menhir
command: archolith-bench harness menhir-phase3
commit: unknown
run_date: 2026-07-07
source: offline scenario suite (fixtures)
source_tracked: true
public_copy_allowed: false
note: Consumer-pipeline validation, not a proxy A/B and not a model score.
-->

**Type:** Consumer-pipeline validation (NOT a proxy A/B, NOT a model score). Validates the
selective-capture *consumer* end to end against a throwaway menhir:

```
:TurnEvidence -> Phase 3 consolidation -> View writes
             -> abstention receipts   -> supersession/currentness -> recall
```

The producer (the host `UserPromptSubmit` triage) is **frozen**; fixtures encode its verdicts,
so junk is dropped exactly as the live triage drops it. What is measured is the consumer: given
the producer's decisions, does consolidation write the right Views, abstain *visibly* (never
silently), supersede on correction, and stay idempotent.

**Harness:** `archolith-bench harness menhir-phase3` (in-repo; `archolith_bench/harness/menhir_phase3.py`).
Black-box HTTP — no menhir import, no Cypher. Mirrors the live `menhir/scripts/validate_phase3_realdata.py`
scenario (`b144381`).

**Under test (commits):**
- menhir `e12a0af` (Phase 3 HTTP endpoints) + `1965997` (docs) — `main`
- archolith-bench `72fea60` (on `claude/menhir-chain-handoff-doc-7iuat2`): `9cfe2ef` benchmark, `9633c33` bearer, `72fea60` robustness

**Menhir endpoints exercised** (added `e12a0af`): `POST /api/turn-evidence`, `POST /api/phase3/run`,
`GET /api/phase3/status`, `GET /api/views`, `POST /api/phase3/reset`.

## Scenario

One isolated namespace per run: `reset -> capture phase-A candidates -> run#1 -> idempotent
re-fold (run#2) -> capture phase-B correction -> run#3 -> inspect Views / receipts / history`.
Each run self-isolates (namespace-scoped `turn_key`) and self-cleans (teardown).

## Cases

| Case | Prompt | Expected consumer outcome |
|------|--------|---------------------------|
| stated measure | `I have 25 movies on my watch list.` | View `movies/watchlist = 25` |
| fold SUM | `I bought one bike for $50 and another for $75.` | View `bike_spend = 125` (canonical key) |
| correction / supersession | `Actually it is 20, not 25.` | current `= 20`, prior `25` superseded (`expired_at` set) |
| junk / drop | `write the handoff` | dropped by triage; no evidence, no View |

## Result

**Verdict: PASS — 3/3 consecutive live runs** against a throwaway menhir (`:8099`, `MENHIR_BENCHMARK_MODE=1`,
sharing `yawn-neo4j` but isolated namespace + teardown; the live WSL menhir on `:8090` was not touched).

| Case | Result |
|------|--------|
| stated measure | PASS |
| fold SUM | PASS |
| correction / supersession | PASS (25 -> 20, `expired_at` set) |
| junk / drop | PASS |

### Safety invariants (0 every run — these are the verdict gates)

| Invariant | Value |
|-----------|-------|
| `wrong_view_writes` | 0 |
| `silent_abstentions` | 0 |
| `duplicate_writes_on_rerun` | 0 |

## Known stochastic behavior (by design, not a defect)

The k-sample extractor (`gpt-4o-mini @ k=3`) commits the *derived* SUM ~8/10 and **fails closed**
otherwise — menhir's own extractor matrix (menhir CHANGELOG 2026-07-07). Observed live: the fold
abstained on some passes with **visible receipts** (`silent_abstentions=0`), then committed `125`
on another pass. The benchmark therefore validates the correct value across **all** consolidation
passes (batch re-fold is idempotent; committed Views persist), which strictly requires `125`/`25`
to have committed without penalizing a single stochastic miss. Counter-key drift
(`bike_purchase` vs canonical `bike_spend`) is reported as a **warning**, not a failure — the
value is the hard gate.

## Bug found and fixed (the reason the benchmark was worth running)

A live throwaway re-run — which a pure service-layer mock would likely have missed — found a
**persistence/idempotency bug**:

- menhir `MERGE`s `:TurnEvidence` on a **global** `turn_key` derived from prompt text (no namespace).
- A second run with the same fixture prompts under a fresh namespace bound to the *prior* run's
  node (`created=false`), so the new namespace captured nothing (`turn_evidence_created=0`,
  `phase3_selected=false`) and consolidation wrote nothing.
- **Fix (`72fea60`):** scope the posted `turn_key` by namespace (`{namespace}:{case_id}`), plus
  end-of-run teardown so no residue lingers between runs. Regression test added.

This validates benchmarking through the real HTTP/API path rather than service-layer tests only.

Two secondary corrections in the same commit:
- `duplicate_writes` false positive: a measure that abstained on run#1 then committed on the
  idempotent re-fold was miscounted as a divergent write. Now counts only value **changes** on
  measures present in both passes; new commits are tracked as `late_commits` (informational).
- Report model label: the LLM under test is menhir's **server-side** personal-memory model, not
  the harness `--model`.

## Expanded scenario suite (2026-07-07, offline-verified)

Beyond the core 4-case flow, `archolith_bench/harness/phase3_scenarios.py` adds a declarative
suite of consumer stress scenarios (`Scenario` = scripted `Post` phases + typed assertions,
with a `gate` vs **characterization** split). Gate scenarios fold into the run's exit code;
characterization scenarios document menhir's current behavior on genuinely-uncertain paths and
never fail the verdict.

| Scenario | Kind | Checks |
|----------|------|--------|
| `ambiguous-correction` | gate | two Views == 25 -> a bare `20, not 25` touches NEITHER (F2 abstain) |
| `currency-worded-sum` | gate | `50 dollars and 75 dollars` folds to `bike_spend = 125` |
| `multi-namespace` | gate | identical fixtures in two namespaces capture independently |
| `negative-correction` | gate | `Not 25 anymore, it is 20` supersedes via the connective rule |
| `arrow-correction` | gate | `changed it to 20 from 25` supersedes via the reverse from/to connective |
| `count-vs-spend` | characterization | `2 bikes for $125` -> count(2) and spend(125) do not merge |

`count-vs-spend` stays characterization (non-gate) because whether menhir co-extracts a count and
a sum from one compound sentence is still what the benchmark is there to *measure*.
`negative-correction` was **promoted to a gate** (2026-07-07) once menhir's correction resolver
gained the `not OLD anymore, it is NEW` pattern — it now guards that binding against regression.
`arrow-correction` was **promoted to a gate** (2026-07-08, consumer-quality-pack v1) once the
resolver gained the ASCII-arrow (`25 -> 20`) and reverse (`to 20 from 25`) connectives.

### Live results (2 runs against throwaway menhir)

All **4 then-existing gate scenarios PASS** both runs (ambiguous-correction abstains, currency-SUM
folds to 125, multi-namespace independent, negative-correction supersedes 25 -> 20); overall
**verdict PASS**, safety invariants clean. (The 5th gate, `arrow-correction`, was added later in
consumer-quality-pack v1 and is offline-verified only — pending a live 2x run.) The remaining
characterization scenario **DIVERGES** consistently — one
real, reproducible consumer gap the benchmark documents (verdict unaffected):

| Scenario | Live behavior | Gap |
|----------|---------------|-----|
| `count-vs-spend` | `2 bikes for $125 total` yields only the spend View (or abstains); no separate `count=2` View | menhir does not co-extract a count and a SUM from one compound sentence |

#### Fixed (2026-07-07): negative-correction

`Not 25 anymore, it is 20` previously did NOT supersede (the resolver's connective rule missed the
`not OLD anymore, ...` phrasing, so the stale View survived). Fixed in menhir by adding a targeted
`not OLD (anymore|any longer)[,] <conn> NEW` pattern (still precision-first: requires a connective,
and the unique-value-match safety net means a detection can only re-value a View that already holds
`old`). Confirmed live: the scenario flipped **DIVERGES -> PASS** on both runs, so it was promoted
to a gate. The remaining `count-vs-spend` gap is a candidate for future menhir consumer work;
producer stays frozen.

#### Consumer-quality-pack v1 (2026-07-08)

Menhir-side consumer changes landed under `feat/phase3-consumer-quality-pack-v1`, governed by the
invariant **a wrong current-state View is worse than a miss** (no change loosens precision):

- **count-vs-spend (safety-only, DECISION 1).** Menhir added a deterministic `count_spend_compound`
  detector and a `count_vs_spend_partial` observability receipt: when a `bought N <noun> for $M`
  clause commits only one of {count, spend}, the fail-closed is now *legible* instead of a silent
  miss. No View is written and nothing merges — so `count-vs-spend` **stays characterization**, not a
  gate. Whether the extractor co-extracts both remains the stochastic gap this scenario measures.
- **fold-SUM stochasticity.** Opt-in `verify_retries` (default 0 = unchanged) re-runs the full
  k-sample verifier vote up to `1+retries` times, committing as soon as one attempt clears — each
  attempt keeps the same unanimity bar, so a flaky-but-correct SUM gets more chances without lowering
  precision. Verifier **receipt clarity**: a fail-closed SUM now carries `verify_votes`/`verify_k`/
  `verify_attempts` (how close the audit was) into the decision trail.
- **correction phrasings.** Resolver gained arrow (`25 -> 20`), reverse (`to 20 from 25`), and
  replacement (`20 replaces 25` / `25 replaced by 20`) connectives — all precision-first behind the
  unique-value-match net. `arrow-correction` promoted to a gate (above).

**Verification status:** all menhir unit tests + this offline scenario suite (6 scenarios, gate PASS,
invariants clean) pass. Live characterization results below.

#### Live characterization (2026-07-08, throwaway menhir :8099, gpt-4o-mini)

Ran against a throwaway menhir on `:8099` (bench Neo4j on bolt 7688, `docker-compose.benchmark.yml`;
real `:8090` untouched, no 429s). Two questions: does `verify_retries` improve the fold-SUM commit
rate, and does the `count_vs_spend_partial` receipt fire live.

**Full phase3 suite — 3 live runs (2× at retries=0, 1× at retries=1):** verdict **PASS** every run;
core `phase3-bike-sum` committed; all 5 gate scenarios PASS (incl. the new `arrow-correction`);
`count-vs-spend` DIVERGES (characterization, as designed); safety invariants
`wrong_view_writes=0 silent_abstentions=0 duplicate_writes=0` every run.

**Focused fold-SUM commit-rate probe** (fresh namespace per iteration, 2-purchase `bike_spend`
fixture that should fold to SUM=125, N=10 each):

| verify_retries | commit rate (=125) | wrong writes | duplicate writes | abstention veto |
|----------------|--------------------|--------------|------------------|-----------------|
| 0 (baseline)   | 5/10 (50%)         | 0            | 0                | `cross_check` ×5 |
| 1              | 5/10 (50%)         | 0            | 0                | `cross_check` ×5 |

**Finding — `verify_retries` does NOT help this fixture, and the mechanism is decisive (not a
sample-size artifact):** every abstention fired on `perception_abstained_cross_check` (Lever B, the
holistic second derivation) with `llm_calls=4` — i.e. the pipeline stopped at the cross-check gate,
*before* the verifier (Lever C4) ever ran (a commit reaches `llm_calls=7`). Since `verify_retries`
only re-runs the verifier, it is structurally unable to rescue a `cross_check` abstention — there is
nothing downstream to retry. `retries=2` would be identical for the same reason. The fold-SUM
stochasticity here is dominated by holistic-cross-check variance, not verifier variance.

**Decision (per the pack's decision tree — "if retries mostly don't help, leave default 0"):**
`MENHIR_PERSONAL_MEMORY_VERIFY_RETRIES` default stays **0**. The knob is wired (opt-in) for future
manual tuning but is **not enabled**. The receipt-clarity fields (`verify_votes`/`verify_k`/
`verify_attempts`) and the `count_vs_spend_partial` receipt are retained — they are the durable value.

**`count_vs_spend_partial` receipt — confirmed live.** A `"bought 2 bikes for $125 total"` turn
consolidated with `views_written=0, abstained=1`; the namespace's receipts carried
`count_vs_spend_partial=1.0` alongside `perception_abstained_cross_check=1.0` — the compound was
detected, only the spend side was a candidate, and the fail-closed is now legible (no wrong write).

**Future consumer lane (documented, out of scope here):** if fold-SUM commit rate becomes a priority,
the lever to investigate is the **holistic cross-check** (Lever B) variance/agreement tolerance — not
verifier retries. Left as characterization; producer stays frozen; no consumer logic changed here.

#### Cross-check-quality pack v1 (2026-07-08) — deterministic SUM grounding

Acted on the lane above: menhir added deterministic SUM **arithmetic grounding**
(`MENHIR_PERSONAL_MEMORY_SUM_GROUNDING`). "Arithmetic is not a belief" — when a SUM's amounts are each
an explicit price literally in the source span, prove `50+75=125` deterministically and skip the noisy
holistic cross-check (the sharper verifier still audits membership). Measured with the extended probe's
SUM **phrasing matrix** (`scripts/probe_phase3_sum_rate.py --fixture sum --variant all`), OFF vs ON,
N=5/variant (ON run 2x):

| variant | OFF commit | ON commit (2x, n=10) | wrong (all runs) |
|---------|-----------|----------------------|------------------|
| two-episode  | 40%  | **100%** (10/10) | 0 |
| one-sentence | 40%  | **90%** (9/10)   | 0 |
| worded       | 100% | 90%              | 0 |
| sequential   | 100% | 100%             | 0 |
| list         | 60%  | 40%              | 0 |

**`wrong_view_writes=0` in every cell across OFF + 2x ON** — the promotion gate. The
cross-check-dominated variants (`two-episode`, `one-sentence`) jump from 40% to 90-100%; the win was
promoted to menhir's default. Mechanism confirmed via `llm_calls`: grounded commits skip the holistic
call (7->6) and rescue the OFF holistic-veto abstentions (llm_calls=4). The `worded`/`one-sentence`
single dips are `self_consistency` (veto-1, upstream of grounding — noise); `list` is high-variance on
`count_floor`/`verification` (a separate extraction gap grounding doesn't touch). Real `:8090`
untouched. Reproduce via `RUNBOOK-phase3-live-characterization.md` + `--variant all`.

## Offline smoke (CI-safe)

`archolith-bench harness menhir-phase3 --offline-fixture stub` runs the **full** driver + scenario
suite + report against a deterministic in-memory `StubPhase3Client` — no menhir, no Neo4j, no
network. It guards the harness itself (driver call sequence, validators, metrics, report) and is
collected by the existing CI `pytest tests/` step (`test_cli_offline_menhir_phase3_smoke`). It is
**not** a substitute for the live run: the stub models the happy consumer and cannot reveal
server-side extraction defects or the fold-SUM stochasticity.

## Verification

- menhir route tests: **21/21** (`tests/test_api_routes.py`, incl. `TestPhase3` +7).
- archolith-bench: full suite passing; core driver tests **9/9**
  (`tests/test_menhir_phase3.py`, incl. namespace-scoped `turn_key` regression) + scenario/offline
  tests **12/12** (`tests/test_phase3_scenarios.py`, incl. gate/characterization split, the shipped
  `StubPhase3Client`, and the offline CLI smoke; 6 scenarios after the `arrow-correction` gate).
  Ruff clean.

## Reproduce

```bash
# 1. Throwaway menhir (separate from any live instance), from the menhir repo venv:
ENV_FILE=./.env MENHIR_BENCHMARK_MODE=1 .venv/Scripts/python.exe -m menhir.main serve --port 8099

# 2. Benchmark (MENHIR_AGENT_KEY = menhir's agent-tier bearer if auth is enabled):
archolith-bench harness menhir-phase3 \
  --menhir-url http://127.0.0.1:8099 \
  --confirm-menhir-reset \
  --format markdown --out results/menhir_phase3_view_consolidation.md
```

The generated `results/` report is gitignored (local evidence); this file is the durable summary.
