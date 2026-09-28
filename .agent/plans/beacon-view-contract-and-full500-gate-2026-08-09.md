# Beacon View Contract and Full-500 Gate

**Status:** PLANNED — implementation and paid run not started
**Date:** 2026-08-09
**Owner:** Cross-repository follow-up (`beacon`, `menhir`, `archolith-bench`)
**Parent roadmap:** Beacon `docs/beacon-functional-product-roadmap.md` (v0.5 dynamic/temporal
release plus Menhir/evaluation workstreams)
**Starting anchors:** Beacon `77ad631`; Menhir
`1fa57955b24f90d08550c911f26133e5b14cbb89`; Bench
`d5e97cc4fc322564c624a749e2cb25dccdf9c2ea`

> **DEFERRED 2026-09-15 (ctharvey).** The runtime `MenhirBeaconProvider` this plan defines is a
> much-later item. Menhir integrates into Beacon at **build time** first — as a source adapter whose
> output is baked into the snapshot — per
> `beacon/.agent/plans/beacon-build-pipeline-and-source-adapters-plan-2026-09-15.md` §7 Q1. This plan is
> deferred, not rejected: the normalized envelope and View read boundary it specifies remain the
> intended contract, and the build-time adapter should consume the same envelope so the runtime provider
> can layer on later without rework. Do not start Phase 1 until the build-time path has shipped and the
> revisit conditions recorded in that plan's Q1 apply.

## Objective

Prove that Beacon can consume a small, generic Menhir View projection for real project knowledge
without changing Beacon's five v0 provider/tool contracts, then use that reviewed contract as the
go/no-go gate for a fresh paid LongMemEval full-500 run.

This is a child execution plan, not the complete Beacon product roadmap. It supplies the first
Menhir-backed current/superseded proof and one backend-scale validation stream for Beacon v0.5. The
parent roadmap separately owns maintainer setup, repository adapters, snapshots/refresh, task
evaluation, remote operations/security, compatibility, examples, and the v1 release gate.

This plan begins only after the cumulative-activity/KU78 plan is closed. The KU78 implementation and
71/78 result are prior evidence, not unfinished work in this plan.

## Non-Goals

- Do not rename or change the signatures or answer dataclasses of Beacon's five current
  `BeaconProvider` capabilities.
- Do not couple Beacon to scalar presentation strings, scalar-only View names, Neo4j labels, or raw
  Menhir persistence rows.
- Do not tune Menhir production rules to the seven KU78 v6 misses.
- Do not launch the full 500 before the View/provider gate is reviewed and a fresh cost estimate is
  accepted.
- Do not mutate KU78 v4/v6 artifacts or reuse their graph as the full-500 graph.
- Do not claim that completing this plan makes Beacon a functional v1 product; the parent roadmap's
  project-task, operability, compatibility, and maintainer-journey gates remain independent.

## Contract Decision Gate

Before implementation, choose one read boundary and record the decision in this plan:

1. a narrow Menhir projection DTO/API over View records; or
2. a direct provider adapter over an explicitly versioned Menhir View reader.

Prefer the projection DTO/API if direct reading would expose storage labels, scalar content strings,
or repository-specific fields. Whichever boundary wins must return a generic envelope with:

- stable `view_key` and `view_kind`;
- lifecycle state (`current` or `superseded`) and predecessor/successor identity when known;
- validity time and learned/source time when available;
- confidence and typed payload;
- source citations sufficient to populate Beacon `BeaconSource` values; and
- namespace/project isolation.

Unknown, malformed, cross-namespace, or citation-free records must fail closed to omission or an
explicit low-confidence/uncertain answer. They must never become current high-confidence knowledge.

## Phase 1 — Generic View Fixture

Create a small project-memory fixture independent of LongMemEval with at least:

- one current project decision;
- one superseded predecessor for the same decision lane;
- one current project-status fact;
- source citations for every record; and
- one cross-project or malformed negative control.

Rebuilding the fixture twice must be idempotent: stable keys, no duplicate current View, the same
current/superseded relation, and byte-stable projected envelopes after canonical ordering.

## Phase 2 — `MenhirBeaconProvider` Vertical Slice

Implement a thin provider in Beacon that satisfies the existing `BeaconProvider` Protocol:

- `project_overview`
- `agent_onboarding`
- `search`
- `explain_concept`
- `guardrails`

The first slice may combine the existing manifest/docs provider with Menhir-backed project Views,
but the MCP tools and answer-contract dataclasses remain unchanged. At minimum, prove through the
existing answer contract that:

- current-only output selects the current decision;
- history-aware output can cite the superseded predecessor without presenting it as current;
- every Menhir-derived answer includes source citations;
- uncertainty/confidence degrades when the projection is incomplete; and
- startup/provider selection is explicit, reversible, and leaves manifest-only mode working.

## Phase 3 — Review Gate

Before any full-500 spend:

1. review the DTO/reader boundary for storage coupling and namespace isolation;
2. run Beacon's complete offline suite plus provider contract tests;
3. run Menhir focused View/projection, authority, and namespace-isolation tests;
4. run deterministic fixture rebuild/idempotency tests twice;
5. record exact commits and a cost estimate for the full 500; and
6. obtain an explicit go/no-go decision for the paid run.

The gate fails if any Beacon tool contract changes, citations are missing, current/superseded state
is ambiguous, rebuild is non-idempotent, or the provider needs raw scalar presentation strings.

## Phase 4 — Fresh Full-500 Run

After the review gate passes:

- pin clean reviewed Menhir and Bench commits (or reviewed descendants of the starting anchors);
- create a fresh isolated graph and unique immutable result directory;
- enable Event History and Event History authority;
- disable deterministic scalar router and shadow;
- require TurnEvidence, scalar history/View authority, and zero failed episodes per namespace;
- use the unchanged full-500 fixture and record its SHA256;
- release only an approximately 10-item mixed checkpoint first;
- validate graph freshness, settings, View/current-history behavior, failure counts, and recall before
  releasing the remaining items; and
- stop on any failed namespace, dirty-code drift, provenance mismatch, or checkpoint regression.

Record provider-reported usage, scored-arm usage/cost, per-item outcomes, logs, commit/settings
provenance, artifact hashes, and final graph state. Do not promote a launch headline automatically.

## Verification Commands

Exact commands may be adjusted to repository tooling, but the review must record equivalent concrete
commands and outcomes:

```text
Beacon: python -m pytest -p no:cacheprovider -q
Beacon: python -m ruff check src tests
Menhir: python -m pytest -p no:cacheprovider -q <focused View/provider/authority tests>
Bench:  python -m pytest -p no:cacheprovider -q <provider contract and run-provenance tests>
Bench:  bash -n scripts/longmemeval/run_knowledge_update_buildout.sh
```

The full repository suites must be run where dependencies are available. Any collection blocker must
be named with exact missing modules rather than being reported as a pass.

## Acceptance Criteria

- A documented, versioned generic View boundary exists and does not expose storage-specific fields.
- The project fixture proves current, superseded, citation, negative-control, and idempotent rebuild
  behavior.
- `MenhirBeaconProvider` satisfies all five existing Beacon provider methods without contract changes.
- Manifest-only Beacon remains available and its existing tests stay green.
- The vertical slice passes an independent review before paid work.
- The full-500 run starts only after explicit cost/go-no-go approval and a passing mixed checkpoint.
- Final full-500 artifacts are immutable, hash-anchored, clean-commit-provenanced, and honest about
  failures, cost, and whether the result is headline-approved.
- Outcomes and unresolved product implications are recorded back in Beacon's canonical functional
  product roadmap or its linked execution-plan index.

## Deferred Backlog

- Possessive-to-specific alias binding (`my camera` to `Canon EOS 80D camera`) remains outside this
  plan until unrelated non-benchmark examples establish the general pattern.
- Cleanup of preserved KU78 containers/volumes is an explicit operational decision, not an implicit
  step of the full-500 launch.
