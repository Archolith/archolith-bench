# Methodology

How archolith-bench measures things, and what its numbers do and do not mean.

> **DRAFT — not maintainer-reviewed.** Written 2026-09-08 against commit `2f9185e` by reading
> `suites/proxy.py`, `core/metrics.py`, and `core/public_claims.py`. The formulas and the
> cost-model bias direction are transcribed from source. The surrounding framing is inferred
> and may not match intent. Verify before treating this as authoritative or public-facing.

## What this suite measures

archolith-bench measures **middleware deltas**, not model quality. Every headline
comparison is an A/B between the same model answering the same scenario with and without an
archolith component in the path — proxy context assembly, client-side filtering, MCP audit
trimming, or Menhir retrieval.

It is not a leaderboard. "Model X scores N on benchmark Y" is not a claim this suite is built
to make, and a number from here should never be read as one. When an official external
benchmark is run (LongBench v2, SWE-bench, BigCodeBench, LongMemEval), it is run as a
direct-vs-proxy A/B, and the reportable result is the delta between the two arms plus any
quality regression — not the absolute score.

## The two savings numbers

This is the distinction most likely to be misread, so it is the one to get right.

The proxy suite reports two different reductions, computed from different quantities:

**Internal curation savings** (`overall_savings_ratio`)

```
total_savings_tokens / total_direct_input_tokens
```

What the proxy's curation avoided handling internally. It measures curation leverage — how
much of the raw context the assembly step decided not to carry forward.

**Upstream input reduction** (`upstream_input_reduction_ratio`)

```
(total_direct_input_tokens - total_arm_input_tokens) / total_direct_input_tokens
```

What actually changed at the upstream billing meter after the arm finished constructing its
final prompt.

**Only the second one is a cost claim.** The first is an internal efficiency measure. They
can diverge sharply, and they can diverge in opposite directions.

The figures below are **retired 2026-05-30 evidence, reproduced here as illustration only**.
They are not active claims and must not be copied into public material — see the
"Retired / Not For Copy" table in `HEADLINE-NUMBERS.md`.

<!-- archolith-claim-scan: ignore-start -->

| Scenario | Arm | Budget | Upstream input reduction | Internal curation savings |
|----------|-----|--------|--------------------------|---------------------------|
| code_review | proxy_only | 15000 | 25.8% | 58.6% |
| code_review | proxy_plus_filter | 15000 | **-2.4%** | 58.7% |
| debugging | proxy_plus_filter | default | **-244.5%** | 0.0% |

The second row is the whole point. Curation was doing substantial work (58.7%), and the bill
still went *up* (-2.4%), because prompt reconstruction added more than curation removed. A
report that cited only the curation number here would be describing a cost saving that did
not happen.

<!-- archolith-claim-scan: ignore-end -->

Rule: never present internal curation savings as billed-token reduction, and never quote one
without the other.

## Cost model

Costs are **cache-weighted effective costs**, not raw token counts times a headline rate.
Per turn (`core/metrics.py:compute_turn_cost`):

- When the provider reports cache data, input cost is split across
  `cache_hit x hit_rate + cache_miss x miss_rate + residual x full_rate`.
- When it does not, the whole prompt is priced at the full input rate and the turn is
  flagged `cache_data_available: false`.

Two deliberate choices:

- **Rates are static and dated, never looked up live.** Each `PricingModel` carries a
  `comment` with its source and date. A cost number is therefore only as current as its rate
  table, which is why run date is part of every evidence record.
- **On write-asymmetric providers the bias is against the proxy.** Anthropic charges more for
  cache creation than for full input. Where `input_cache_write` is set, the entire miss
  bucket is priced at the write rate. This slightly overstates proxy cost on purpose: a
  go/no-go verdict should fail safe, so the proxy has to win against a handicap.

Helper-model spend (a smaller model used inside context assembly) is priced separately and
included in `effective_cost_usd`. A cost claim that ignores helper spend is incomplete.

## Quality preservation

Savings mean nothing without the quality side, so every proxy arm reports
`recall_preservation`:

```
avg_arm_recall / avg_direct_recall
```

A ratio against the direct baseline, not an absolute score. `1.0` means the arm retained the
baseline's recall; below `1.0` is a regression the savings number has to justify.

Report both or neither. A savings figure published without its preservation figure is not a
result, it is half of one.

## Evidence labels

Not every number in this repo is a claim. Four categories, and only one of them is public:

| Label | Meaning | Public? |
|-------|---------|---------|
| **Tracked** | Lives in `benchmarks/`, has command, commit, model, date, caveats | Yes, if in `HEADLINE-NUMBERS.md` |
| **Fixture** | Generated from bundled `fixtures/`; demonstrates report format | Never |
| **Sample / historical** | A real but dated or single-scenario run | Methodology reference only |
| **Candidate** | An adapter exists; no run has happened | Never — say "deferred, no launch claim" |

`HEADLINE-NUMBERS.md` is the single source of truth for anything public. A number used in the
README, on archolith.dev, or in external material must appear in its active table in the same
commit. This is enforced mechanically, not by convention: `core/evidence_policy.py` validates
the active table and cross-checks evidence artifacts against it, and `core/public_claims.py`
scans docs for claim-shaped strings that no active headline supports.

Raw `results/` and `logs/` are local runtime output and stay gitignored. Curated evidence is
promoted into `benchmarks/` deliberately. A launch-facing document must never cite `results/`
as its only source.

## When a number expires

A tracked number stops being quotable when any of its inputs move:

- The measured code changed — a number is bound to the commit it was produced at.
- The provider's rates changed, invalidating the dated `PricingModel` behind any cost figure.
- The scenario, corpus, budget, or model changed.
- The arm definition changed, which makes the delta a different comparison.

Expiry is not automatic. Until a number is re-run or retired, it belongs in the
"Retired / Not For Copy" table with the reason recorded, which is where the 2026-05-30 proxy
and filter values currently sit.

## What this suite does not establish

- Absolute model capability on any external benchmark.
- Security posture. Adapters for CyberSecEval, AgentDojo, and OWASP checks are coverage
  scaffolding; no run has produced evidence, and "security benchmark-backed" is not a
  supportable phrase today.
- Menhir production memory performance from fixture runs. This is the easiest mistake to make
  here, because most of what runs offline runs on fixtures: the R1/R2/oracle/intent/L4
  ladders, **and LongMemEval itself**. `harness/longmemeval.py` reads
  `fixture_path` when the `longmemeval` extra is absent, so `menhir longmemeval` will
  complete offline against `fixtures/longmemeval/` and emit a full-looking result that
  measures the harness, not memory quality.

  A persistent-memory claim needs a Mode B run against a throwaway Menhir + Neo4j, never
  production — `harness/memory_ab.py:assert_not_production` refuses production-looking
  targets, and mutating runs require `--confirm-menhir-reset`. Absent that, an LME number
  from this repo is fixture output regardless of how complete it looks.
- Anything about the `stack` suite, which remains experimental pending a refreshed live run.
