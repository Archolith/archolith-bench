# archolith-bench

Benchmark suite for the [archolith&trade;](https://github.com/archolith/archolith-bench) family.
Measures proxy context assembly, client-side filtering, MCP audit savings, security benchmark
coverage, and Menhir memory/retrieval behavior across reproducible scenarios.

### Evidence status

No launch headline numbers are currently active. Historical and fixture runs remain in
[`benchmarks/`](benchmarks/) and generated reports for methodology review, but they should not be
used as public copy until refreshed against the current launch configuration and recorded in
[`HEADLINE-NUMBERS.md`](HEADLINE-NUMBERS.md).

See [BENCHMARKS.md](BENCHMARKS.md) for full tables and reproduction instructions,
and [`benchmarks/`](benchmarks/) for tracked evidence summaries.

## Suites

| Suite | Purpose |
|-------|---------|
| `proxy` | Multi-turn token savings and continuity measurement |
| `filter` | Compression-ratio measurement on real tool-output corpora |
| `audit` | MCP token-waste reduction before/after comparison |
| `stack` | Experimental four-way comparison; pending refreshed live run |
| `industry` | Launch coverage matrix mapping products to trusted external benchmark families |
| `harness` | Runs official external benchmarks (e.g. LongBench v2) as a direct-vs-proxy A/B |
| `extraction-bench` | Compare extraction models on the real backend pipeline (speed/quality/cache-aware cost) |
| `dashboard` | Live view of in-progress memory runs (terminal or `--serve` web); reads run checkpoints |
| `ports` | Index running stack processes by label + port (find stray menhir/neo4j/dashboard instances) |

### One-task scalar view explorer

The web dashboard can add a read-only drill-down for tasks present in both the active LongMemEval
ingest manifest and the connected graph. It walks one task through the actual scalar-state boundaries:

1. preserved `TurnEvidence` and its `FOUNDS` links;
2. the correlated k-sample vote receipt (including 2-of-3 distributions);
3. immutable `TypedAssertion` events;
4. superseded and current `scalar_state` Views from the deterministic fold;
5. recall/model/scorer output once the QA checkpoint exists.

Neo4j credentials stay server-side, and the API accepts only namespaces present in the loaded
manifest. Configure the optional telemetry DB to show the exact vote distribution; without it,
the graph stages remain available and the gate stage explains that the receipt is missing.

```bash
export LME_DASHBOARD_NEO4J_URI=bolt://127.0.0.1:7694
export LME_DASHBOARD_NEO4J_USER=neo4j
export LME_DASHBOARD_NEO4J_PASSWORD=...
export LME_DASHBOARD_TELEMETRY_DB=/path/to/mcp_telemetry.db
archolith-bench dashboard --results-dir results/run --serve --scalar-task lme-01493427
```

### Offline deterministic scalar shadow measurement

Use the durable Bench-side instrument to compare Menhir's pure deterministic extractor with
committed decisions replayed from `scripts/freeze_scalar_samples.py` captures. It imports Menhir's
real proposal dataclass, gate, extractor, and canonical comparator, reruns Menhir's pure
deterministic extractor on the frozen episode text, and replays the captured LLM proposals through
Menhir's real gate/comparator. It makes no new LLM, network, Neo4j, Docker, or service calls.

```bash
python scripts/measure_deterministic_scalar_shadow.py frozen.json \
  --menhir-root C:\path\to\projects\archolith\menhir \
  --json-out report.json --markdown-out report.md
```

The first held-out smoke input is the versioned non-LME fixture
`fixtures/deterministic_scalar_heldout_v1.json`. Menhir's freezer accepts it with
`--episodes-json` before any graph connection is made; it contains separate fully-covered and
fallback/adversarial namespaces, exact stable UUID/content rows, and only a one-off payment as a
negative control for categorical event-history scope. Freeze it with `-k 3` for exactly six calls
(`2 namespaces × 3 samples`), then run this offline instrument. Those six calls are smoke evidence
only, not a promotion or population gate; do not substitute an LME fixture or tune rules to LME
task text.

The default gate policy is exactly 2/3 with `align_spans` enabled; reconciliation and
`canonical_self` are explicit opt-in switches. A versioned, generic label sidecar is optional.
Every sidecar must carry a non-empty, unique, canonical lowercase `capture_sha256` list matching
the measured captures exactly. Each `false_positive` or `false_current` row is a human-labeled
known-negative target, so its report is a `known_negative_target_hit_rate` with
`hit_count` / `labeled_negative_targets` / `hit_rate`—not an overall or population false-positive
or false-current rate and not the plan's population precision/confidence-interval gate. Categories
with no labeled targets remain not measured with null values. Token/dollar savings remain unclaimed
unless measured fields exist in the capture.

### Offline compositional scalar semantic panel

Use the independent non-LME panel to evaluate the deterministic extractor plus structural composer
against source-authored semantic labels. It does not consume an LLM capture and never treats LLM
agreement as truth.

```bash
python scripts/measure_compositional_scalar_panel.py \
  fixtures/compositional_scalar_generic_v1.json \
  --menhir-root C:\path\to\projects\archolith\menhir \
  --json-out panel-report.json --markdown-out panel-report.md
```

The v1 panel contains 12 positive and 12 negative holdout cases across four relation groups. Every
label is bound to an exact unique source span and source hash; group/split leakage, duplicate
locators, unknown reason codes, and noncanonical expected identities fail closed. Reports separate
exact join coverage, correct/wrong/unresolved composition, strict composer abstention, extractor
omission, system non-admission, and false-current risk. The bounded panel always remains
`promotion_status=not_evaluable`; it is regression evidence, not a population acceptance gate.

### Offline historical scalar spend attribution

Use the durable Bench-owned instrument to measure a completed LongMemEval-style run from its
manifest, provenance record, read-only SQLite telemetry, and recall checkpoint. It records artifact
hashes, attempt/commit/dirty-state provenance, exact scalar and Graphiti call counts, per-arm
persisted tokens/latency/cost, and conservative state/history presentation signatures. It makes no
network, LLM, Neo4j, Docker, or input-artifact writes.

```bash
python scripts/measure_scalar_spend_attribution.py results/run \
  --json-out results/run/scalar-spend-attribution.json \
  --markdown-out results/run/scalar-spend-attribution.md \
  --input-usd-per-million 2.5 --output-usd-per-million 10 \
  --negative-control lme-6071bd76
```

The checkpoint is auto-discovered only when exactly one `.checkpoint_*.json`/`.jsonl` exists;
otherwise pass `--checkpoint`. Costs cover only persisted answer-model tokens at explicitly supplied
rates. Scalar spend, evaluator/judge spend, and scalar-caused corrections remain `not_measured`.
Full canonical acceptance is not evaluated by this instrument; resumed or mixed-code runs are
reported as descriptive evidence with observed canonicality concerns.

### Harness: official benchmarks as direct-vs-proxy A/B

Archolith is middleware, not a model, so a harness adapter never claims a standalone
benchmark score. It runs the official dataset + scorer twice (client pointed at the direct
upstream, then at the proxy) and reports the **delta**: official score preserved while input
tokens and cost drop. Local-analogue scenarios (RULER) are `-style` smoke tests only.

Adapters (all under one roof):

| Benchmark id | Kind | Extra |
|--------------|------|-------|
| `longbench-v2` | in-process | `longbench` |
| `bigcodebench-hard` | in-process (sandboxed exec) | `bigcodebench` |
| `longmemeval` | in-process (menhir memory capability) | `longmemeval` |
| `swe-bench` | external-cli wrapper | `swebench` |
| `cyberseceval-4` | external-cli wrapper | `cyberseceval` |
| `agentdojo` | external-cli wrapper | `agentdojo` |
| `mteb-retrieval` | external-cli wrapper (embeddings; see caveat) | `mteb` |

```bash
archolith-bench harness --list
pip install -e ".[longbench]"          # official LongBench v2 needs `datasets`
archolith-bench harness longbench-v2 --arms direct,proxy_only,proxy_plus_filter --subset single_document_qa --limit 50
```

In-process adapters drive each task via the chat client; external-cli adapters invoke the
official tool (per arm, direct vs proxy base URL) and parse its results. SWE-bench/CyberSecEval/
AgentDojo/MTEB are scaffolded — real runs (datasets, agent scaffolds, API budget) land at launch
step 3. **MTEB caveat:** it measures embeddings, which the chat proxy doesn't sit in front of, so
its proxy-arm delta is a no-op until an embeddings layer exists.

## Extraction model selection

Graph-memory extraction is a multi-call structured-JSON pipeline per episode, so the choice
of extraction model is a real speed / cost / quality decision. `archolith-bench
extraction-bench` replays that real pipeline against any OpenAI-compatible model and reports
latency, quality, and **cache-aware** cost. Full writeup, pricing, and findings:
**[EXTRACTION_MODELS.md](EXTRACTION_MODELS.md)**.

**TL;DR:** quality is a tie among capable small models, so optimize speed + cost.

| model | provider | call p50 | ent/fact recall | cache hit | $/1k ep |
|-------|----------|---------:|----------------:|----------:|--------:|
| gpt-oss-120b | Cerebras (wafer) | 0.30 s | 1.00 / **0.40** | 56% | $0.77 |
| llama-3.3-70b | Groq (LPU) | 0.35 s | 1.00 / 0.80 | – | $0.53 |
| **gpt-4.1-nano** | OpenAI | 0.54 s | 0.90 / 0.80 | 0% | **$0.10** |
| **gemini-3.1-flash-lite** | Google | 0.63 s | 1.00 / **0.90** | 0% | $0.20 |
| **deepseek-v4-flash** | DeepSeek | 1.19 s | 1.00 / 0.80 | 74% | **$0.05** |
| local qwen ~9B | LM Studio | 4.56 s | 1.00 / 0.80 | – | $0 |

- **Best value:** `gpt-4.1-nano` (~$0.10/1k). **Best quality:** `gemini-3.1-flash-lite` (best fact recall, but ~2× cost).
- **Best open-weight:** `qwen3-next-80b` via OpenRouter (0.85 fact recall, 100% JSON, $0.22/1k — gemini-3.1 tier, no Google).
- **Cheapest:** `deepseek-v4-flash` — caching makes it cheapest (50× cache discount).
- **Fastest:** Cerebras `gpt-oss-120b` (0.30 s) — but only 0.40 fact recall, so *not* recommended for graph memory. **Free/private:** local Qwen (slower, $0).

By default the bench runs only the two **blessed keepers** — `gpt-4.1-nano` (default) and
`qwen3-next-80b` (open-weight alternative, via OpenRouter). Pass `--all` for the full
candidate sweep.

```bash
# keys read from env or menhir/.env; fast providers auto-enable when their key is present
archolith-bench extraction-bench --repeats 2            # just nano + qwen3-next-80b
archolith-bench extraction-bench --repeats 2 --all      # full candidate sweep
archolith-bench extraction-bench --repeats 2 --all --exclude gpt-5   # sweep minus slow reasoning models
```

## Quick Start

```bash
# Base CLI install
pip install -e .

# List available scenarios (no live proxy needed)
archolith-bench proxy --list
```

Install optional suite dependencies when you need filter or audit runs. The launch-supported
source checkout workflow is to install the sibling repos editable first, then install the
bench extras:

```bash
python -m pip install -e ../archolith-filter
python -m pip install -e ../archolith-mcp-audit
pip install -e ".[all]"

# Run a single scenario against a live proxy
archolith-bench proxy --scenario scenarios/taskflow.json --arms direct,proxy_only

# Run all scenarios across multiple budgets
archolith-bench proxy --all --arms direct,proxy_only,proxy_plus_filter --budgets 4000,15000
```

Generate the industry benchmark coverage matrix before launch:

```bash
archolith-bench industry --launch-only --out benchmarks/industry-trusted-benchmark-coverage.md
```

The industry matrix is not a substitute for completed benchmark evidence. It
separates implemented local coverage from candidate-before-launch work such as
SWE-bench, LongBench v2, CyberSecEval, AgentDojo, OWASP security checks, and
real audit logs.

The base install supports the CLI, report generation, and proxy orchestration. Until
`archolith-filter` and the `archolith-audit` distribution are published to the package index used by
your environment, `pip install -e ".[all]"` is not a standalone public install command; use the
source checkout workflow above or install those packages from their eventual release artifacts first.
The audit source repo is `../archolith-mcp-audit`; its Python import package is `archolith_mcp_audit`.

## Experiment Arms

| Arm | Filter | Proxy Assembly | What It Measures |
|-----|--------|----------------|-----------------|
| `direct` | off | off | Baseline |
| `filter_only` | on | off | Filter token savings alone |
| `proxy_only` | off | on (baseline) | Context engine without filter |
| `proxy_plus_filter` | on | on (filter as engine) | Full-stack experimental arm |
| `proxy_typed_state` | on | on + typed work state | Typed state continuity |
| `proxy_state_snippets` | on | on + snippet injection | Snippet continuity |

## Configuration

Set environment variables in `.env`:

- `UPSTREAM_API_KEY` (required for runs)
- `PROXY_URL` (default: `http://localhost:9800/v1`)
- `UPSTREAM_BASE_URL` (default: `https://api.openai.com/v1`)
- `BENCHMARK_MODEL` (default: `gpt-4o-mini`)

## Checkpoints

Resumable benchmark runs may create `.checkpoint_*.json` files in the working directory. They contain local run state only and are safe to delete after a run finishes or when intentionally starting fresh.

## License

Licensed under the [Apache License 2.0](LICENSE).
Free for non-commercial use; commercial use requires permission from the licensor.
Contributions are subject to the [CLA](CLA.md).

archolith&trade; is a trademark of Charles Harvey. Use of the name in
derivative works or competing products requires explicit permission.
