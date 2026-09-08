# Benchmark Evidence

This directory contains curated, tracked benchmark evidence for launch-facing
numbers. Raw suite output still lands in `results/`, which is local runtime
scratch space and remains gitignored.

Use this directory for:

- stable summary tables used by `README.md`, `BENCHMARKS.md`, and
  `HEADLINE-NUMBERS.md`
- methodology notes and formulas
- links back to the source result artifact names

## Required: the `archolith-evidence` block

Every evidence artifact in this directory -- `.md` or `.json` -- must declare its
provenance mechanically. For Markdown, that means an HTML comment immediately after the
title. It does not render, and it is **enforced**: `scripts/check_evidence_policy.py` exits
non-zero without it, and `tests/test_evidence_policy.py::TestRepoEvidenceStaysCompliant`
fails the suite if a new artifact lands here unstamped.

```markdown
# LongMemEval Menhir M1 Gate Benchmark

<!-- archolith-evidence
product: menhir
command: archolith-bench menhir longmemeval
commit: cf13f8cb48c2bd308aeda171ca6af1f4404e2bc9
run_date: 2026-07-15
source: run m1-full-500-recalibrated-2026-07-15
source_tracked: false
public_copy_allowed: false
-->
```

| Key | Meaning |
|-----|---------|
| `product` | Which product the number describes (`archolith-context`, `archolith-filter`, `menhir`, ...) |
| `command` | The exact command that produced it |
| `commit` | Hex hash of the code under measurement, or `unknown` |
| `run_date` | `YYYY-MM-DD`, or `unknown` |
| `source` | Where the raw output lives |
| `source_tracked` | `true` only if that raw output is in the repository |
| `public_copy_allowed` | `true` only if this may be quoted publicly |
| `note` | Optional one-line caveat |

`unknown` is deliberate. Artifacts predating this convention keep it rather than carrying a
fabricated hash or date -- an honest gap is auditable, an invented value is not. It comes at
a price: `unknown` provenance is disqualifying for public copy.

Setting `public_copy_allowed: true` requires a real `commit`, a real `run_date`, and
`source_tracked: true`. A public claim cannot rest on data that is not in the repository.
Anything short of that is an error, not a warning.

"Real" is checked, not taken at your word. The validator asks git whether the `commit`
exists and whether `source` names a tracked path, and rejects a `run_date` in the future.
When a declaration fails verification it is an error for `public_copy_allowed: true` and a
warning otherwise -- artifacts predating this convention are reported, not broken. If git
cannot answer at all, that counts as unverified: still disqualifying for public copy.

`README.md` and `RUNBOOK-*.md` are documentation, not evidence, and are exempt.

Check your artifact before committing it:

```sh
python scripts/check_evidence_policy.py --evidence-dir benchmarks/
```

Do not use fixture-only or sample-only results as launch headlines. If a result
comes from fixtures, label it as format evidence only.

## Current Evidence

| File | Scope | Launch headline eligible |
|------|-------|--------------------------|
| `proxy-code-review-2026-05-30.md` | Historical proxy code-review run | Partial; actual upstream input deltas only |
| `filter-2026-05-30.md` | Filter compression run over 12 corpus samples | Yes |
| `audit-fixture-2026-06-06.md` | Fixture audit report format evidence | No |
| `industry-trusted-benchmark-coverage.md` | Product-to-industry-benchmark launch coverage matrix | No; coverage/gate artifact only |
| `mteb-embedding-baseline-2026-06-19.md` | Single-arm embedding-model component baseline | No; component diagnostic, not proxy or memory-system A/B |
| `menhir-phase3-view-consolidation-2026-07-07.md` | Menhir Phase 3 consumer-pipeline validation (TurnEvidence -> Views) | No; consumer-correctness validation, not a proxy or model score |

## Refresh TODO

- Re-run proxy benchmarks against the current launch proxy configuration.
- Cover `proxy_only` and `proxy_plus_filter`, 4K and 15K budgets, and the current launch model.
- Record actual upstream input reduction separately from internal context-curation savings.
- Replace or supplement the historical proxy evidence before using broader launch copy.
- Run `archolith-bench industry --launch-only --out benchmarks/industry-trusted-benchmark-coverage.md`
  after changing product scope or benchmark policy.
- Keep MTEB artifacts named as embedding-model baselines unless an embeddings
  proxy/cache layer exists and a true A/B run is added.
- `scripts/run_mteb_local.py` intentionally stays separate from `MtebAdapter`:
  it drives MTEB's in-process encoder API for local embedding-model measurement,
  while `MtebAdapter` is the archolith-bench harness wrapper for evidence parsing.
