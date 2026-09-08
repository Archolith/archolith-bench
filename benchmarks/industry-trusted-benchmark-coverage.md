# Industry-Trusted Benchmark Coverage

<!-- archolith-evidence
product: archolith-bench
command: archolith-bench industry --launch-only
commit: unknown
run_date: unknown
source: archolith_bench/core/industry.py
source_tracked: true
public_copy_allowed: false
note: Generated coverage matrix, not a measurement. Candidate rows are gates, not results.
-->

This matrix maps each Archolith product to external benchmark families that are trusted enough to anchor launch claims. `implemented-local` means archolith-bench has a local analogue or reporting path today. `candidate-before-launch` means the benchmark is relevant but must not be claimed until the listed gate has a tracked evidence artifact.

| Product | Suite | Benchmark | Status | Launch gate |
|---------|-------|-----------|--------|-------------|
| archolith-context | proxy | RULER | implemented-local | Run direct, proxy_only, and proxy_plus_filter on `ruler_recall`; publish recall preservation, upstream input reduction, and any quality regressions together. |
| archolith-context | proxy | LongBench v2 | candidate-before-launch | Run `archolith-bench harness longbench-v2` direct vs proxy on a real subset and publish the accuracy delta plus input-token and cost reduction as tracked evidence under benchmarks/. |
| archolith-context | proxy | SWE-bench Lite / Verified | candidate-before-launch | Run `archolith-bench harness swe-bench` on a Lite/Verified smoke subset, direct vs proxy, and report the resolution-rate delta + token/cost. Do not claim SWE-bench performance until this exists. |
| archolith-context | proxy | BigCodeBench-Hard | candidate-before-launch | Run `archolith-bench harness bigcodebench-hard` direct vs proxy and report the pass@1 delta plus token/cost reduction. Cheap proxy-overhead sanity check; report separately from multi-turn context claims. |
| archolith-filter | filter | HELM efficiency metrics | implemented-local | Run the filter suite on the launch corpus and publish per-category savings, total savings, sample count, corpus source, and known no-op categories. |
| archolith-filter | filter | SWE-bench-style agent traces | candidate-before-launch | Add at least one tracked corpus summary showing sample provenance and category balance, then rerun `archolith-bench filter`. |
| archolith-skree | audit | HELM-style token/cost accounting | implemented-local | Use real before/after session logs, not fixtures. Publish server-level deltas and note any new waste type regressions. |
| archolith-context | proxy | LongMemEval (in-context / proxy) | candidate-before-launch | Run `archolith-bench harness longmemeval` direct vs proxy and publish memory-QA accuracy preserved while input tokens drop (the proxy curates the long history). A context-curation claim, not a menhir memory claim. |
| menhir | memory | LongMemEval (persistent menhir memory) | candidate-before-launch (accuracy-lift claim); IR-gate PASS tracked 2026-07-15 (narrower claim, see detail) | Stand up a throwaway menhir+Neo4j, run LongMemEval Mode B (no-memory baseline vs menhir-recall), and publish the memory-QA accuracy lift. This is menhir's primary advertisable capability claim. |
| menhir | memory | Deep Memory Retrieval (DMR) | candidate-before-launch | Either wire a DMR adapter and report a direct-vs-proxy retrieval-accuracy delta, or keep DMR a documented future benchmark and lead menhir's memory claim with LongMemEval. |
| menhir | memory | MTEB retrieval/reranking slices | candidate-before-launch | Run `archolith-bench harness mteb-retrieval` against the embeddings model and publish the MTEB score as a menhir/embedding-model baseline (not a proxy claim). A proxy A/B requires an embeddings layer that does not exist yet. |
| archolith-security | security | CyberSecEval 4 | candidate-before-launch | Run `archolith-bench harness cyberseceval-4` on a scoped subset, direct vs proxy, and publish pass/fail, refusal, and false-refusal caveats before any security benchmark claim. |
| archolith-security | security | AgentDojo | candidate-before-launch | Run `archolith-bench harness agentdojo` direct vs proxy and publish utility delta + attack-success-rate before claiming hardening against malicious tool output. |
| archolith-context | security | OWASP LLM Top 10 + ASVS | candidate-before-launch | Complete a scoped OWASP LLM Top 10 + ASVS checklist for public proxy surfaces and track the findings/remediations before presenting the proxy as launch-safe. |

## Details

### RULER (ruler)

- Product: `archolith-context`
- Suite: `proxy`
- Authority: NVIDIA
- Type: long-context synthetic retrieval and aggregation
- Status: `implemented-local`
- Source: https://github.com/NVIDIA/RULER
- Paper: https://arxiv.org/abs/2404.06654
- Why relevant: RULER is the closest external methodology for testing whether long-context systems can retrieve and aggregate facts across very long distractor contexts. That maps directly to archolith-context continuity, recall preservation, and repeated-read reduction.
- Local coverage: `scenarios/ruler_recall.json` is a RULER-STYLE smoke test (seeded-fact recall with late-turn probes): local, reproducible, useful for regression checks — but NOT an official RULER run and must not be advertised as a RULER score. Real RULER would be wired as a harness adapter.
- Launch gate: Run direct, proxy_only, and proxy_plus_filter on `ruler_recall`; publish recall preservation, upstream input reduction, and any quality regressions together.
- Command: `archolith-bench proxy --scenario scenarios/ruler_recall.json --arms direct,proxy_only,proxy_plus_filter --budgets 15000`
- Evidence path: `benchmarks/proxy-ruler-recall-YYYY-MM-DD.md`

### LongBench v2 (longbench-v2)

- Product: `archolith-context`
- Suite: `proxy`
- Authority: THUDM / Tsinghua
- Type: realistic long-context multitask and code-repo understanding
- Status: `candidate-before-launch`
- Source: https://github.com/THUDM/LongBench
- Paper: https://arxiv.org/abs/2412.15204
- Why relevant: LongBench v2 includes long-document, long-dialogue, structured-data, and code-repository understanding. It is relevant to the proxy's claim that curated context can preserve useful state without replaying the whole transcript.
- Local coverage: Official adapter implemented (`harness/longbench_v2.py`) — runs the real THUDM/LongBench-v2 multiple-choice set as a direct-vs-proxy A/B via `harness.run_ab`. Awaiting a tracked paid run; the result is the delta (accuracy preserved + tokens/cost reduced), not a standalone score.
- Launch gate: Run `archolith-bench harness longbench-v2` direct vs proxy on a real subset and publish the accuracy delta plus input-token and cost reduction as tracked evidence under benchmarks/.
- Command: `archolith-bench harness longbench-v2 --arms direct,proxy_only,proxy_plus_filter --subset single_document_qa --limit 50`
- Evidence path: `benchmarks/proxy-longbench-v2-YYYY-MM-DD.md`

### SWE-bench Lite / Verified (swe-bench)

- Product: `archolith-context`
- Suite: `proxy`
- Authority: Princeton NLP / Stanford
- Type: real GitHub issue resolution
- Status: `candidate-before-launch`
- Source: https://github.com/SWE-bench/SWE-bench
- Paper: https://arxiv.org/abs/2310.06770
- Why relevant: SWE-bench is the trusted coding-agent benchmark for real issue resolution. It is relevant because archolith-context targets agent sessions that must retain repo state, decisions, and test feedback across multiple turns.
- Local coverage: Official adapter scaffolded (`harness/external.py` SweBenchAdapter) — wraps the SWE-bench evaluation harness per arm (client base_url = direct vs proxy) and parses its report into the A/B result. Real run needs an agent scaffold (e.g. SWE-agent) + Docker eval; deferred to step 3. `scenarios/{code_review,debugging,long_agent}.json` remain -style smoke tests, not SWE-bench scores.
- Launch gate: Run `archolith-bench harness swe-bench` on a Lite/Verified smoke subset, direct vs proxy, and report the resolution-rate delta + token/cost. Do not claim SWE-bench performance until this exists.
- Command: `archolith-bench harness swe-bench --subset princeton-nlp/SWE-bench_Lite --arms direct,proxy_only`
- Evidence path: `benchmarks/proxy-swe-bench-smoke-YYYY-MM-DD.md`

### BigCodeBench-Hard (bigcodebench-hard)

- Product: `archolith-context`
- Suite: `proxy`
- Authority: BigCode Project
- Type: function-level code generation with complex instructions
- Status: `candidate-before-launch`
- Source: https://github.com/bigcode-project/bigcodebench
- Paper: https://arxiv.org/abs/2406.15877
- Why relevant: BigCodeBench-Hard is relevant as a compact, reproducible coding workload when full SWE-bench runs are too expensive. It is less agentic than SWE-bench, so it should be a secondary signal for proxy overhead and answer preservation.
- Local coverage: Official adapter implemented (`harness/bigcodebench.py`) — runs the real bigcode/bigcodebench-hard pass@1 (generated code executed in a sandboxed subprocess) as a direct-vs-proxy A/B via `harness.run_ab`. Awaiting a tracked paid run.
- Launch gate: Run `archolith-bench harness bigcodebench-hard` direct vs proxy and report the pass@1 delta plus token/cost reduction. Cheap proxy-overhead sanity check; report separately from multi-turn context claims.
- Command: `archolith-bench harness bigcodebench-hard --arms direct,proxy_only,proxy_plus_filter --limit 50`
- Evidence path: `benchmarks/proxy-bigcodebench-hard-YYYY-MM-DD.md`

### HELM efficiency metrics (helm-efficiency)

- Product: `archolith-filter`
- Suite: `filter`
- Authority: Stanford CRFM
- Type: holistic model evaluation including efficiency
- Status: `implemented-local`
- Source: https://github.com/stanford-crfm/helm
- Paper: https://openreview.net/forum?id=iO4LZibEqW
- Why relevant: HELM is not a tool-output compression benchmark, but its efficiency framing is the trusted external evaluation lens for reporting token and cost effects alongside quality.
- Local coverage: `archolith-bench filter` reports raw tokens, filtered tokens, and savings by real tool-output category. This is the product-specific implementation of token-efficiency measurement.
- Launch gate: Run the filter suite on the launch corpus and publish per-category savings, total savings, sample count, corpus source, and known no-op categories.
- Command: `archolith-bench filter --corpora corpora/ --format markdown`
- Evidence path: `benchmarks/filter-YYYY-MM-DD.md`

### SWE-bench-style agent traces (swe-bench-traces-filter)

- Product: `archolith-filter`
- Suite: `filter`
- Authority: Derived workload from SWE-bench-style coding sessions
- Type: tool-output compression on coding-agent traces
- Status: `candidate-before-launch`
- Source: https://github.com/SWE-bench/SWE-bench
- Paper: https://arxiv.org/abs/2310.06770
- Why relevant: The filter product should be tested on the same kind of logs it will compress: file reads, test output, search output, diffs, and shell traces from coding-agent issue-resolution runs.
- Local coverage: The current `corpora/` directory has real session samples across 8 categories. It should be expanded with SWE-bench-like agent traces before stronger launch claims.
- Launch gate: Add at least one tracked corpus summary showing sample provenance and category balance, then rerun `archolith-bench filter`.
- Command: `archolith-bench filter --corpora corpora/ --format markdown`
- Evidence path: `benchmarks/filter-swe-style-traces-YYYY-MM-DD.md`

### HELM-style token/cost accounting (helm-token-accounting)

- Product: `archolith-skree`
- Suite: `audit`
- Authority: Stanford CRFM
- Type: efficiency, transparency, and reproducibility reporting
- Status: `implemented-local`
- Source: https://github.com/stanford-crfm/helm
- Paper: https://openreview.net/forum?id=iO4LZibEqW
- Why relevant: Archolith Skree is not a model-quality benchmark. Its trusted external analogue is HELM's transparent reporting of efficiency metrics, with reproducible inputs and clear caveats.
- Local coverage: `archolith-bench audit` compares before/after MCP audit JSON reports and writes aggregate token and waste deltas. Current bundled fixtures are examples only.
- Launch gate: Use real before/after session logs, not fixtures. Publish server-level deltas and note any new waste type regressions.
- Command: `archolith-bench audit --before <real-before.json> --after <real-after.json> --format markdown`
- Evidence path: `benchmarks/audit-live-before-after-YYYY-MM-DD.md`

### LongMemEval (in-context / proxy) (longmemeval)

- Product: `archolith-context`
- Suite: `proxy`
- Authority: Wu et al. (ICLR 2025)
- Type: long-term memory QA with the history IN-CONTEXT (Mode A: tests proxy curation)
- Status: `candidate-before-launch`
- Source: https://github.com/xiaowu0162/LongMemEval
- Paper: https://arxiv.org/abs/2410.10813
- Why relevant: MODE A — the LongMemEval history is placed in the prompt and the proxy curates/compresses it. This tests archolith-context (context curation over a long memory-QA history), NOT menhir's persistent graph memory. menhir's capability benchmark is the separate `longmemeval-menhir` (Mode B) entry.
- Local coverage: Official adapter implemented (`harness/longmemeval.py`, in-process) — runs LongMemEval QA as a direct (full history in context) vs proxy A/B. Deterministic normalized-containment scorer offline; the official GPT-4 judge can be added behind a flag. Awaiting a tracked run.
- Launch gate: Run `archolith-bench harness longmemeval` direct vs proxy and publish memory-QA accuracy preserved while input tokens drop (the proxy curates the long history). A context-curation claim, not a menhir memory claim.
- Command: `archolith-bench harness longmemeval --arms direct,proxy_only,proxy_plus_filter --limit 50`
- Evidence path: `benchmarks/longmemeval-proxy-YYYY-MM-DD.md`

### LongMemEval (persistent menhir memory) (longmemeval-menhir)

- Product: `menhir`
- Suite: `memory`
- Authority: Wu et al. (ICLR 2025)
- Type: long-term interactive memory QA via ingest-then-recall (Mode B: tests menhir end-to-end)
- Status: `candidate-before-launch` (accuracy-lift claim below) — **but see the 2026-07-15 IR-gate
  update**: a distinct, narrower claim (menhir retrieval beats vector-only retrieval) now has
  tracked PASS evidence. Do not conflate the two — see the note below before citing either.
- Source: https://github.com/xiaowu0162/LongMemEval
- Paper: https://arxiv.org/abs/2410.10813
- Why relevant: MODE B — menhir's CAPABILITY benchmark. Per question: ingest the haystack sessions into menhir's graph (extraction -> temporal KG), then query menhir's recall, feed the retrieved memory to the model, answer. menhir is built on Graphiti (the engine Zep reports on LongMemEval/DMR), so this is the apples-to-apples industry standard for what menhir is. Unlike Mode A it exercises the actual graph store, so it needs an isolated (throwaway) Neo4j and per-question `group_id` isolation.
- Local coverage: Adapter WIRED: `harness/longmemeval.py` LongMemEvalMemoryAdapter + `harness/memory_ab.py` run_memory_ab driver (ingest -> recall -> answer) + `harness/menhir_client.py` (Stub for offline, Http for real). Offline-runnable now with StubMenhirClient (no Neo4j). A real run needs a throwaway menhir (`--menhir-url`, prod-guarded) + Neo4j and per-item group_id isolation (menhir backend: `recall(...)` / `ingest_document(...)` with group_id). **Real answer-accuracy (LLM-judge) run still not tracked — this is the remaining gap for this row.** Plan: `archolith-bench-longmemeval-menhir-mode-b-plan.md`.
- Launch gate: Stand up a throwaway menhir+Neo4j, run LongMemEval Mode B (no-memory baseline vs menhir-recall), and publish the memory-QA accuracy lift. This is menhir's primary advertisable capability claim. **Not yet done — do not claim an accuracy-lift number.**
- Command: `archolith-bench harness longmemeval-menhir --limit 30   # (Mode-B driver pending)`
- Evidence path: `benchmarks/longmemeval-menhir-YYYY-MM-DD.md`

**2026-07-15 IR-gate update (distinct evidence, not the row's launch gate above):** the separate
`scripts/longmemeval/analysis/lib/retrieval_quality.py` harness (pure retrieval-quality — Hit@3,
MRR@10, explainability; no LLM-judge, no answer generation) was run against the full n=500 oracle
corpus for the first time and **PASSED** all 3 measured gates: menhir Hit@3(support)=4.60% vs
graphiti(vector-only)=0.40% (~11.5x), menhir MRR@10(support)=0.0466 vs graphiti=0.0033 (~14x),
explainability=100%. Evidence: `benchmarks/longmemeval-menhir-2026-07-15.md`. **This licenses a
comparative claim ("menhir's graph retrieval substantially beats vector-only search on this
corpus"), not the absolute "memory-QA accuracy lift" this row's launch gate asks for** — in
absolute terms menhir found supporting evidence for only 81/500 questions (16.2%). The Hit@3 gate
threshold itself was also recalibrated this run (see `menhir-mvp-roadmap.md` M1) from an
unvalidated absolute 0.80 to the relative bar above.

### Deep Memory Retrieval (DMR) (dmr)

- Product: `menhir`
- Suite: `memory`
- Authority: MemGPT / Letta; reported by Zep
- Type: memory retrieval accuracy over conversational history
- Status: `candidate-before-launch`
- Source: https://github.com/cpacker/MemGPT
- Paper: https://arxiv.org/abs/2310.08560
- Why relevant: The second memory benchmark Zep/Graphiti report on, complementing LongMemEval. Registered under one roof; an adapter is not yet wired.
- Local coverage: No adapter yet. Candidate — wire after LongMemEval lands, reusing the same in-process A/B pattern.
- Launch gate: Either wire a DMR adapter and report a direct-vs-proxy retrieval-accuracy delta, or keep DMR a documented future benchmark and lead menhir's memory claim with LongMemEval.
- Command: `TODO: add a DMR adapter under harness/ (in-process, same pattern as longmemeval)`
- Evidence path: `benchmarks/dmr-YYYY-MM-DD.md`

### MTEB retrieval/reranking slices (mteb-retrieval)

- Product: `menhir`
- Suite: `memory`
- Authority: Embeddings Benchmark / MTEB
- Type: embedding retrieval/reranking (COMPONENT diagnostic, not the memory capability)
- Status: `candidate-before-launch`
- Source: https://github.com/embeddings-benchmark/mteb
- Paper: https://arxiv.org/abs/2210.07316
- Why relevant: COMPONENT diagnostic only: MTEB measures the embedding model's retrieval quality, not the memory system end-to-end. Useful for an embedder-selection decision (e.g. free local nomic vs OpenAI text-embedding-3-small for menhir), NOT a memory capability claim. menhir's capability benchmark is LongMemEval (and DMR).
- Local coverage: Official adapter (`harness/external.py` MtebAdapter) runs mteb against an OpenAI-compatible embeddings endpoint, defaulting to a local LM Studio server (text-embedding-nomic-embed-text-v1.5) — so it runs for FREE, no API spend, once the `mteb` extra is installed. SINGLE-ARM BASELINE: the embedding model is a menhir / fact-retrieval dependency, not the chat proxy path, so there is no direct-vs-proxy A/B (a real A/B needs an embeddings proxy/caching layer). Measures the embedding model menhir depends on, not the proxy.
- Launch gate: Run `archolith-bench harness mteb-retrieval` against the embeddings model and publish the MTEB score as a menhir/embedding-model baseline (not a proxy claim). A proxy A/B requires an embeddings layer that does not exist yet.
- Command: `archolith-bench harness mteb-retrieval --subset SciFact --arms direct`
- Evidence path: `benchmarks/menhir-retrieval-YYYY-MM-DD.md`

### CyberSecEval 4 (cyberseceval-4)

- Product: `archolith-security`
- Suite: `security`
- Authority: Meta / PurpleLlama
- Type: LLM cybersecurity vulnerabilities and defensive capabilities
- Status: `candidate-before-launch`
- Source: https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks
- Paper: https://arxiv.org/abs/2408.01605
- Why relevant: CyberSecEval is the closest external benchmark family for archolith-security's LLM security posture: prompt injection, insecure code generation, code interpreter risk, vulnerability exploitation, autonomous offensive operations, and defensive SOC/autopatch tasks.
- Local coverage: Official adapter scaffolded (`harness/external.py` CyberSecEvalAdapter) — wraps Meta PurpleLlama CybersecurityBenchmarks per arm and parses its stat file into the A/B result. Real run pending (archolith-security-owned, step 3); no score may be claimed yet.
- Launch gate: Run `archolith-bench harness cyberseceval-4` on a scoped subset, direct vs proxy, and publish pass/fail, refusal, and false-refusal caveats before any security benchmark claim.
- Command: `archolith-bench harness cyberseceval-4 --subset mitre --arms direct,proxy_only`
- Evidence path: `benchmarks/security-cyberseceval-YYYY-MM-DD.md`

### AgentDojo (agentdojo)

- Product: `archolith-security`
- Suite: `security`
- Authority: ETH Zurich SPY Lab
- Type: prompt injection attacks and defenses for tool-using agents
- Status: `candidate-before-launch`
- Source: https://github.com/ethz-spylab/agentdojo
- Paper: https://arxiv.org/abs/2406.13352
- Why relevant: AgentDojo directly matches Archolith's agent/tool threat model: untrusted tool results, indirect prompt injection, data exfiltration attempts, and over-defense that breaks useful work.
- Local coverage: Official adapter scaffolded (`harness/external.py` AgentDojoAdapter) — wraps the AgentDojo runner per arm and parses utility-under-attack into the A/B result; attack-success-rate is reported alongside as the security signal. Real run pending (step 3).
- Launch gate: Run `archolith-bench harness agentdojo` direct vs proxy and publish utility delta + attack-success-rate before claiming hardening against malicious tool output.
- Command: `archolith-bench harness agentdojo --subset workspace --arms direct,proxy_only`
- Evidence path: `benchmarks/security-agentdojo-YYYY-MM-DD.md`

### OWASP LLM Top 10 + ASVS (owasp-llm-asvs)

- Product: `archolith-context`
- Suite: `security`
- Authority: OWASP Foundation
- Type: LLM application and API security control verification
- Status: `candidate-before-launch`
- Source: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Paper: https://owasp.org/www-project-application-security-verification-standard/
- Why relevant: The proxy exposes API, admin, trace, live-stream, memory, and plugin surfaces. OWASP LLM Top 10 covers LLM-specific risks such as prompt injection, data leakage, and excessive agency; ASVS anchors conventional application controls such as authentication, authorization, input validation, logging, and secure configuration.
- Local coverage: archolith-bench does not run an OWASP security audit. Recent launch audits identified admin/live stream exposure risks, but those are review findings rather than an automated benchmark.
- Launch gate: Complete a scoped OWASP LLM Top 10 + ASVS checklist for public proxy surfaces and track the findings/remediations before presenting the proxy as launch-safe.
- Command: `TODO: add archolith-bench security-checklist or link a tracked OWASP review artifact for archolith-context`
- Evidence path: `benchmarks/security-owasp-context-YYYY-MM-DD.md`

