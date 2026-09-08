# Evidence Manifest

Generated index of tracked benchmark evidence. **Do not edit by hand** --
regenerate with:

```sh
python scripts/check_evidence_policy.py --evidence-dir benchmarks/ \
    --manifest benchmarks/evidence-manifest.md
```

Every column is read from the artifact's own provenance block, so this file
cannot drift from what the artifacts declare. `Status` is the validator's
verdict: `ok`, `incomplete` (valid block, unknown provenance -- not promotable
to a headline claim), or `INVALID` (fails the evidence policy).

| Artifact | Product | Command | Commit | Run date | Source tracked | Public copy | Status |
|---|---|---|---|---|---|---|---|
| audit-fixture-2026-06-06.md | archolith-mcp-audit | `archolith-bench audit` | unknown | 2026-06-06 | false | false | incomplete |
| filter-2026-05-30.md | archolith-filter | `archolith-bench filter` | unknown | 2026-05-30 | false | false | incomplete |
| industry-trusted-benchmark-coverage.md | archolith-bench | `archolith-bench industry --launch-only` | unknown | unknown | true | false | incomplete |
| longmemeval-baseline.json | - | - | - | - | - | - | incomplete |
| longmemeval-menhir-2026-07-15.md | menhir | `archolith-bench menhir longmemeval` | cf13f8cb48c2 | 2026-07-15 | false | false | incomplete |
| menhir-phase3-view-consolidation-2026-07-07.md | menhir | `archolith-bench harness menhir-phase3` | unknown | 2026-07-07 | true | false | incomplete |
| mteb-embedding-baseline-2026-06-19.md | menhir | `python scripts/run_mteb_local.py` | unknown | 2026-06-19 | false | false | incomplete |
| proxy-code-review-2026-05-30.md | archolith-context | `archolith-bench proxy` | unknown | 2026-05-30 | false | false | incomplete |

8 artifact(s) indexed; 0 currently quotable as public copy.
