# archolith-bench — Project Context

> **menhir R2 facet work lives here** (`archolith_bench/facet/`). For the cross-repo picture — how this
> ties into menhir's research ladder, what's owed, what not to re-litigate — read the menhir handoff:
> `menhir/.agent/archive/plans/chain-handoff.md`. Local R2 results + caveats: `.agent/benchmark-notes/facet-r2-demo-run.md`.

## Purpose
Unified benchmark suite for the archolith product family. Measures token savings, continuity, compression, and cross-product metrics across proxy, filter, audit, and stack scenarios.

## Suites
- **menhir bootstrap-hygiene**: Deterministic acceptance gate for structural/recent leakage, exact workspace isolation, scoped pins, stale advisories, and off-topic/token diagnostics
- **proxy** (Phase 1, critical path for June 30 launch): Multi-turn token savings + continuity across experiment arms
- **filter** (Phase 2): Compression-ratio product claim on real corpora via archolith-filter
- **stack** (Phase 3): Experimental four-way comparison (direct/filter/proxy/proxy+filter); pending refreshed live run before launch copy
- **audit** (Phase 4): MCP token-waste reduction before/after via archolith-audit
- **industry** (Launch gate): Product-to-benchmark coverage matrix tying Archolith claims to trusted external benchmark families

## Docs map

- **Full per-script methodology** (scalar-identity/dependency-bridge research instruments, menhir
  ScalarStateView instruments) -> [`.agent/scripts-index.md`](scripts-index.md)
- **System design, data flow, file layout** -> [`.agent/architecture.md`](architecture.md)
- **Dataclass/schema reference** -> [`.agent/data_models.md`](data_models.md)
- **Pre-launch checklist** -> [`.agent/launch-readiness-tracker.md`](launch-readiness-tracker.md)
- **User-facing install/quickstart** -> [`../README.md`](../README.md)

## Before writing a script, read the index

`projects/archolith/menhir/.agent/scripts-index.md` indexes **every durable script in both repos**
by the question it answers, and defines the naming convention: `_name.py` is a throwaway (delete it
once its finding is written down, never index it), `name.py` is a durable instrument (index it in
the same commit that adds it). **Keep the bench half of that index in sync** when you add, rename,
or remove a script here.

Before writing a new analysis script, check it there first. Two sessions in a row re-derived results
an existing instrument already produced. Full methodology for the bench-owned scalar-research and
ScalarStateView instruments (commands, fixtures, validated results) lives in
[`.agent/scripts-index.md`](scripts-index.md) — read that before touching `scripts/measure_*.py` or
`scripts/scalar_*.py`.

## How to Run
```bash
pip install -e .
python -m pip install -e ../archolith-filter
python -m pip install -e ../archolith-mcp-audit
pip install -e ".[all]"                 # optional filter/audit suite deps after sibling editable installs
archolith-bench proxy --list          # list scenarios (no proxy needed)
archolith-bench proxy --all --arms direct,proxy_only  # run against live proxy
archolith-bench industry --launch-only                # generate launch benchmark coverage matrix
archolith-bench menhir bootstrap-hygiene --offline   # deterministic startup hygiene gate
```

Offline deterministic scalar shadow measurement:
```bash
python scripts/measure_deterministic_scalar_shadow.py frozen.json \
  --menhir-root C:\path\to\projects\archolith\menhir \
  --json-out report.json --markdown-out report.md
```
The command defaults to exactly the candidate 2/3 threshold with span alignment enabled. Reconciliation
and canonical-self switches are opt-in and are recorded as effective report settings. A namespace
with any fallback episode retains its entire k-call batch in the conservative savings calculation.
The optional v1 sidecar is capture-bound by a required exact set of unique canonical SHA-256 hashes.
Its `false_positive` and `false_current` rows are human-labeled known-negative targets; the report
uses target-hit fields, never an overall/population false-positive or false-current rate, and does
not claim the plan's population precision/confidence-interval gate.

Offline compositional scalar semantic panel:
```bash
python scripts/measure_compositional_scalar_panel.py \
  fixtures/compositional_scalar_generic_v1.json \
  --menhir-root C:\path\to\projects\archolith\menhir \
  --json-out panel-report.json --markdown-out panel-report.md
```
This is the capture-independent, source-labeled lane. Its v1 generic holdout has 12 positive and 12
negative cases, four relation groups, and at least three perturbations per group. It reports exact
join and semantic quality separately and is permanently descriptive until a separately
preregistered population gate exists.

## LLM-call helpers

**Flex is the default for non-live answering/judging.** Batch-style benchmark
LLM calls use OpenAI flex processing via `archolith_bench.core.openai_flex.run_flex`
(synchronous `service_tier: "flex"`, ~50% of standard price) instead of the
OpenRouter Batch API, which sat queued for hours at 0 complete with no
cancellation. Key contract points:

- Pricing rates ((input, output) $/1M at flex prices) must be passed in
  explicitly — no catalog lookup.
- 429 policy: capacity 429 (`resource_unavailable`) retries with exponential
  backoff up to `max_capacity_retries` (default 5) then stops the run; any
  other 429, plus 402/401/403, stop immediately with no retry. 408/5xx and
  httpx transport errors retry up to 5 times, then become per-request failures
  (run continues). Other 4xx are per-request failures without retry.
- JSONL checkpoint gives crash-safe resume; a `max_usd` cost cap stops
  scheduling once the running total (including checkpointed rows) reaches it.
- API keys and request bodies are never logged.

### Progress on long runs
Long bench loops (R1/facet ladders, live recall) print a live heartbeat via
`archolith_bench/progress.py` — `ProgressReporter` / `track` / `run_ladder`. Progress goes
to **stderr**, so run the script directly (don't pipe stdout through `tail` — it buffers to
EOF and hides everything). Full usage + adoption guide:
[`.agent/workflows/bench-progress.md`](workflows/bench-progress.md).

## LongMemEval Framework

Memory-specific A/B testing for menhir's recall and ingest, consolidated under
`scripts/longmemeval/` (one `config.sh`, one `lme.sh` dispatcher, one runbook). See
`scripts/longmemeval/README.md` for the full runbook and `lme.sh -h` for all verbs.

Quickstart:
```bash
./scripts/longmemeval/lme.sh -h                 # all commands
./scripts/longmemeval/lme.sh status             # read-only graph/queue state
./scripts/longmemeval/lme.sh build 500          # persistent graph (~1 day for oracle)
./scripts/longmemeval/lme.sh recall-ab main 30  # recall-only A/B against that graph
./scripts/longmemeval/lme.sh matrix             # analysis: accuracy × config × question-type
```

**⚠️ Stratification (load-bearing):** the dataset is grouped by `question_type`, so a bare
`--limit N` samples only `temporal-reasoning` (the hardest type) — a fair run MUST sweep all 6
types via `--subset`. The `matrix`/`msc`/`ablation` verbs already do; see the runbook's
Stratification section. Every value in `config.sh` is env-overridable; nothing is hardcoded in
the scripts. OpenAI key is read at runtime from `menhir/.env`, never committed.

## Headline Numbers Policy

**`HEADLINE-NUMBERS.md` is the canonical source for any stat used in marketing copy or README headlines.**
Before writing any percentage or token count into archolith.dev or a product README:
1. Check `HEADLINE-NUMBERS.md` — if it isn't there, it isn't verified.
2. Run the benchmark, paste the result row into the table, note the commit.
3. Fixture data (from bundled `fixtures/`) is NOT a headline number — it demonstrates report format only.

## KU-Buildout Ledger

**`results/lme-ku-buildout/LEDGER.md` must be updated after every knowledge-update
buildout run.** This includes the scoreboard, per-item results, fixture provenance,
and infrastructure details. See `scripts/longmemeval/README.md` § "KU-Buildout Ledger"
for the full checklist. Do not end a session that ran a buildout without updating the
ledger.

## Launch Readiness

Use `.agent/launch-readiness-tracker.md` as the active pre-launch checklist.
The current posture is imminent pre-launch: fix Critical and High items first,
and defer polish unless it blocks installability, reproducibility, or trust.

## FOLLOW-UP
- GitHub remote `archolith/archolith-bench` still needs creating. Add with: `git remote add origin git@github.com:archolith/archolith-bench.git`
- `archolith-filter` and the `archolith-audit` distribution are optional extras (`filter`, `audit`, `all`) so base install remains usable before those sibling packages are published. For source checkouts, install sibling repos editable from `../archolith-filter` and `../archolith-mcp-audit` before `pip install -e ".[all]"`. The audit distribution imports as `archolith_mcp_audit`.
- The industry benchmark registry is launch-facing. Candidate benchmarks are gates, not completed evidence, until a tracked artifact exists under `benchmarks/`.
