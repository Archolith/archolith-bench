# archolith-bench — Benchmark Results
Generated: 2026-07-07 04:07 UTC

> **Launch-claim status:** this generated report may include historical, fixture, or local methodology runs. It is not a source of headline numbers by itself. Public claims must come from `HEADLINE-NUMBERS.md`, which currently has no active values until refreshed launch evidence exists.

<!-- archolith-claim-scan: ignore-start -->

## Filter Suite (archolith-filter)
Token-savings compression ratio measured on real tool-output corpora.
| Category | Samples | Raw Tokens | Filtered | Savings |
|----------|---------|------------|----------|--------|
| git_diff | 2 | 4,566 | 1,826 | 60.0% |
| git_log | 1 | 580 | 580 | 0.0% |
| git_status | 1 | 774 | 774 | 0.0% |
| json | 1 | 2,078 | 59 | 97.2% |
| logs | 2 | 977 | 835 | 14.5% |
| read_file | 1 | 1,296 | 568 | 56.2% |
| search | 2 | 5,588 | 2,353 | 57.9% |
| test | 2 | 1,851 | 1,499 | 19.0% |
| **Total** | 12 | 17,710 | 8,494 | **52.0%** |

## Proxy Suite
Multi-turn token savings, effective cost, and continuity metrics across proxy experiment arms.

> **Stale evidence caveat:** existing proxy result files include dated, single-scenario or single-model runs. Do not use these as refreshed launch headline numbers until P4 evidence refresh lands after the remediation sessions.

| Scenario | Arm | Budget | Direct In | Arm In | Upstream Input Reduction | Internal Curation Savings | Recall Pres. |
|----------|-----|--------|-----------|--------|--------------------------|---------------------------|-------------|
| code_review | direct | 15000 | 108,516 | 108,516 | 0.0% | 0.0% | 70% |
| code_review | direct | 4000 | 106,707 | 106,707 | 0.0% | 0.0% | 89% |
| code_review | proxy_only | 15000 | 108,516 | 80,520 | 25.8% | 58.6% | 57% |
| code_review | proxy_only | 4000 | 108,516 | 91,701 | 15.5% | 58.6% | 91% |
| code_review | proxy_plus_filter | 15000 | 108,376 | 111,021 | -2.4% | 58.7% | 60% |
| code_review | proxy_plus_filter | 4000 | 108,516 | 105,139 | 3.1% | 58.6% | 54% |
| debugging | proxy_plus_filter | default | 912 | 3,142 | -244.5% | 0.0% | N/A |
| long_agent | proxy_plus_filter | default | 15,077 | 17,387 | -15.3% | 41.4% | N/A |

## Audit Suite (Archolith Skree)
> **Note:** sample/fixture data, not a live audit run. The numbers below reflect the bundled `fixtures/` inputs and demonstrate the report format only. Run `archolith-bench audit` against real before/after session logs to produce measured results.

MCP token-waste reduction before vs after.
Source: `fixtures\audit_before.json` -> `fixtures\audit_after.json`

| Server | Before | After | Change | Pct | Status |
|--------|--------|-------|--------|-----|--------|
| gradle | 32,500 | 19,500 | -13,000 | -40.0% | improved |
| harness | 20,300 | 5,100 | -15,200 | -74.9% | improved |
| memory | 28,400 | 7,100 | -21,300 | -75.0% | improved |
| vps | 50,200 | 12,600 | -37,600 | -74.9% | improved |
| workspace-artifacts | 211,400 | 148,000 | -63,400 | -30.0% | improved |
| **Total** | 342,800 | 192,300 | -150,500 | -43.9% | - |

**Token reduction:** 150,500 (43.9%). **Waste reduction:** 83,500 (71.5%).

## Industry-Trusted Benchmark Coverage
Coverage matrix for external benchmark families that are relevant enough to anchor launch claims. Candidate benchmarks are not completed results.

| Product | Suite | Benchmark | Status | Evidence |
|---------|-------|-----------|--------|----------|
| archolith-context | proxy | RULER | implemented-local | `benchmarks/proxy-ruler-recall-YYYY-MM-DD.md` |
| archolith-context | proxy | LongBench v2 | candidate-before-launch | `benchmarks/proxy-longbench-v2-YYYY-MM-DD.md` |
| archolith-context | proxy | SWE-bench Lite / Verified | candidate-before-launch | `benchmarks/proxy-swe-bench-smoke-YYYY-MM-DD.md` |
| archolith-context | proxy | BigCodeBench-Hard | candidate-before-launch | `benchmarks/proxy-bigcodebench-hard-YYYY-MM-DD.md` |
| archolith-filter | filter | HELM efficiency metrics | implemented-local | `benchmarks/filter-YYYY-MM-DD.md` |
| archolith-filter | filter | SWE-bench-style agent traces | candidate-before-launch | `benchmarks/filter-swe-style-traces-YYYY-MM-DD.md` |
| archolith-skree | audit | HELM-style token/cost accounting | implemented-local | `benchmarks/audit-live-before-after-YYYY-MM-DD.md` |
| archolith-context | proxy | LongMemEval (in-context / proxy) | candidate-before-launch | `benchmarks/longmemeval-proxy-YYYY-MM-DD.md` |
| menhir | memory | LongMemEval (persistent menhir memory) | candidate-before-launch | `benchmarks/longmemeval-menhir-YYYY-MM-DD.md` |
| menhir | memory | Deep Memory Retrieval (DMR) | candidate-before-launch | `benchmarks/dmr-YYYY-MM-DD.md` |
| menhir | memory | MTEB retrieval/reranking slices | candidate-before-launch | `benchmarks/menhir-retrieval-YYYY-MM-DD.md` |
| archolith-security | security | CyberSecEval 4 | candidate-before-launch | `benchmarks/security-cyberseceval-YYYY-MM-DD.md` |
| archolith-security | security | AgentDojo | candidate-before-launch | `benchmarks/security-agentdojo-YYYY-MM-DD.md` |
| archolith-context | security | OWASP LLM Top 10 + ASVS | candidate-before-launch | `benchmarks/security-owasp-context-YYYY-MM-DD.md` |

## Stack Suite (Four-Way Comparison)
*Experimental and pending refreshed live-proxy run. Run `archolith-bench stack --all` to generate; do not use stack results as launch headlines until tracked evidence is added under `benchmarks/`.*

<!-- archolith-claim-scan: ignore-end -->
