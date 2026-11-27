"""S4 + S5 Engineering Evidence Graph benchmark suite.

Measures the four structural/governance capabilities that conversational benchmarks
(LongMemEval, LoCoMo) cannot exercise:
  1. Coverage-aware blast_radius / affected_tests with fail-closed refusal on unindexed paths (S5)
  2. Stale code-anchor advisory labeling on post-anchor file edits (S1 + S5)
  3. L4 WorkArtifact trust invariants & non-destructive supersession (S4)
  4. Git-backed corpus reconciliation & fail-closed ambiguity handling (S4)

Runs 100% deterministically against real `menhir` modules with zero LLM calls and
zero external database required.
"""

from archolith_bench.engineering_evidence.runner import (
    CONDITIONS,
    PILLARS,
    run_engineering_evidence_benchmark,
)

__all__ = [
    "CONDITIONS",
    "PILLARS",
    "run_engineering_evidence_benchmark",
]
