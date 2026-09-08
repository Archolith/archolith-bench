# WRAPUP — Cumulative activity scalars, KU78 v6, and Beacon View handoff

**Date:** 2026-08-10
**Agent:** Codex
**Model:** gpt-5
**Status:** PARTIAL
**Plan / Ticket:** `C:\Users\thron\IdeaProjects\projects\archolith\menhir\.agent\plans\menhir-cumulative-activity-scalars-2026-08-08.md`; `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\plans\beacon-view-contract-and-full500-gate-2026-08-09.md`
**Worktree:** Menhir `C:\Users\thron\IdeaProjects\projects\archolith\menhir`; Bench `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench`; Beacon reference `C:\Users\thron\IdeaProjects\projects\archolith\beacon`
**Branch:** Menhir `main`; Bench `master`; Beacon `master`
**Commits:** Menhir `ea6f2c81a4285e349f4930c07494e180aebc3b74`, `9d9675c9397770b5bc654ce6f15da315d15c616a`, `7e0ff6be4673a4ce54e1cb7cf12401ed6a4a3513`, `c54a8efb756952c995a0c20e9488b5de977e599a`, `72f58452aa05bac4659249bc68e3e2558788ede3`, `bbf116fc448728150b47b428906d8d0621036f15`, `c878188a1643a35a4d600724f9d92739a0e843f0`, `4439611900d6357b0d4b7c765ecc82689f7f239b`, `57ee277355ee00c3d051292e2408991aaa635928`, `11a6e413770892735087a1b7767501449b79deea`, `04d75297e3b6d5d9c4be62140f3b53e245d78dc3`, `bb436d3e7c710939cc39cb9851f77e9105e36992`; Bench `b7a275403d413f4c9a7f92cd2ac5df9eae38b3a0`, `83fac54301e5d325df009e790adb845c5e2b6b7f`, `55d333aa58cba11a1cddc427000eaf67f763690a`, `d8c2f2fe61ae6f29cb1187d856e5c91110d83657`, `33ff3f0bfc419dee23d3f343dffa6b30d6c76b50`, `3bc3c58940d361e4a21e66757435e2c11c295d28`, `1e3f4cdd03d28c8084328ce528f4acec06335a8c`, `c9d304e7bca9e4d3139090bb1ce6e80e4f0cc7fb`
**Verification Scope:** Menhir `main` at `bb436d3e7c710939cc39cb9851f77e9105e36992`; Bench `master` at `c9d304e7bca9e4d3139090bb1ce6e80e4f0cc7fb`; immutable KU78 v6 artifacts; preserved v6 Neo4j container
**Docs Updated:** Menhir cumulative-activity plan and changelog; Bench KU-buildout ledger, scalar run-lineage note, changelog, Beacon/full-500 gate plan, and this handoff
**Changelog Updated:** `C:\Users\thron\IdeaProjects\projects\archolith\menhir\CHANGELOG.md`; `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md`

---

## Before Writing

Working backwards from acceptance: the fresh KU78 run completed and is recorded in the mandatory Bench ledger; its effective settings and artifacts are preserved; Event History authority and cumulative scalar behavior are on the default branches; focused default-branch tests pass; and the paid full-500 run is gated behind the separate Beacon View/provider plan. The remaining seven misses were reviewed. Only the camera possessive-binding case is a clear deterministic Menhir defect, and it remains deferred until benchmark-independent evidence justifies a general fix.

## Summary

Menhir `main` and Bench `master` are now the source of truth for this work. Cumulative activity totals use the typed-scalar path, exact namespace-local fallback binding is available, unresolved acquisition anchors fail closed, Event History authority survives the Bench recall boundary, blocking advisories isolate conflicting context, and run provenance records the effective Event History/router/shadow settings.

Fresh run `scalar-event-activity-ku78-v6-20260809` built 78/78 namespaces with cumulative `failed_remaining=0`, no final PENDING/ENRICHING/FAILED episodes, and scored 71/78 (0.910256; displayed 0.910). This is the current canonical KU78 result, above v4 at 69/78 (0.885) and the previous canonical baseline at 68/78 (0.872). It is deliberately not promoted to `HEADLINE-NUMBERS.md` without separate launch-claim approval.

Do not tune broadly to the seven misses. Three were v4 passes whose evidence remained present, two reward approximate or planned statements as exact current facts, one contains an unsupported-role premise, and one exposes unresolved possessive subject binding (`my camera` versus `Canon EOS 80D camera`). The next milestone is the generic Beacon View contract and `MenhirBeaconProvider` vertical slice; the full-500 run remains gated behind that work, review, cost approval, and a mixed checkpoint.

## Files Changed

| File | Why |
|------|-----|
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\.agent\plans\menhir-cumulative-activity-scalars-2026-08-08.md` | Records design, anti-fitting constraints, implementation, verification, and KU78 v6 closure. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\.agent\research\menhir-research-execution-ladder.md` | Reconciles the research execution ladder with the completed write-side evidence. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\CHANGELOG.md` | Records cumulative scalar and Event History hardening. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\CHANGELOG-archive.md` | Receives older changelog entries displaced by the bounded current changelog. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\docs\research\README.md` | Repairs the research roadmap index and points to the acceptance record. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\domain\recall.py` | Extends the advisory contract for unresolved anchors. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\infrastructure\episode_lifecycle.py` | Adds namespace-scoped exact-name lookup while excluding derived Views. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\infrastructure\memory_graph_adapter.py` | Exposes exact entity lookup to scalar persistence and service paths. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\services\context_builder.py` | Prefers grounded Event History leads and isolates blocking advisories. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\services\event_history_authority.py` | Adds conservative unresolved acquisition-anchor guarding. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\services\typed_scalar_persistence.py` | Threads exact namespace lookup through write and repair binding. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\services\typed_scalar_rules.py` | Adds cumulative activity extraction/admission, hedge handling, and exact subject reconciliation. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\src\menhir\services\typed_scalar_service.py` | Wires namespace lookup into scalar processing. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_cumulative_activity_scalars.py` | Covers cumulative extraction, admission, binding, folding, and negatives. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_context_builder.py` | Covers Event History lead preference and blocking isolation. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_episode_lifecycle.py` | Covers namespace filtering and View exclusion. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_event_history_authority.py` | Covers acquisition-anchor guards and grounded silence. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_recall_event_authority_runtime.py` | Covers runtime authority transport. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_scalar_hedged_abstention.py` | Covers approximate versus non-quantifying language. |
| `C:\Users\thron\IdeaProjects\projects\archolith\menhir\tests\test_typed_scalar_bind_persist.py` | Covers exact fallback binding and repair persistence. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\event_history_acceptance.py` | Bounds and retries malformed Event History panel responses. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\menhir_client.py` | Preserves authority in generic recall and isolates blocking advisories. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\analysis\event_history_acceptance.py` | Applies the 512-token cap and records retry telemetry. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\run_knowledge_update_buildout.sh` | Records effective scalar/Event History settings in run provenance. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_event_history_acceptance.py` | Covers malformed-response recovery. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_event_history_acceptance_cli.py` | Covers cap, retries, telemetry, and provenance. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_longmemeval_ingest_fixture.py` | Covers effective phase settings and run provenance. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_menhir_client_contract.py` | Covers authority rendering and blocking isolation. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\results\lme-ku-buildout\LEDGER.md` | Labels v6 as current canonical evidence and records outcomes, usage, infrastructure, and hashes. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\benchmark-notes\scalar-lme-run-lineage-2026-08-05.md` | Advances the canonical comparison point to v6. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\plans\beacon-view-contract-and-full500-gate-2026-08-09.md` | Defines the Beacon View/provider and full-500 acceptance gates. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md` | Records canonical v6 evidence and its non-headline status. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\for-review\WRAPUP-2026-08-09-cumulative-activity-event-history-ku78-handoff.md` | Replaces the superseded preflight handoff with this default-branch-anchored closeout. |

## Verification

- `C:\Users\thron\IdeaProjects\projects\archolith\menhir\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q tests/test_cumulative_activity_scalars.py tests/test_memory_graph_adapter_methods.py tests/test_episode_lifecycle.py tests/test_event_history_authority.py tests/test_context_builder.py tests/test_recall_event_authority_runtime.py tests/test_scalar_hedged_abstention.py tests/test_typed_scalar_bind_persist.py` — `PASS` — 257 passed on Menhir `main` at `bb436d3`.
- `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.venv\Scripts\python.exe -m pytest tests/test_menhir_client_contract.py tests/test_longmemeval_ingest_fixture.py -q` — `PASS` — 63 passed on Bench `master` at `c9d304e`.
- Fresh KU78 `scalar-event-activity-ku78-v6-20260809` — `PASS` — 78/78 manifest rows, cumulative `failed_remaining=0`, final PENDING/ENRICHING/FAILED all zero, harness exit 0, 71/78 correct, score 0.910 versus baseline 0.872.
- KU78 v6 usage — `PASS` — provider-reported combined 17,516,332 tokens; scored Menhir arm 117,933 input and 1,376 output tokens; $0.308592.
- Existing cumulative scalar panel at `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\results\scalar-identity-isolated-comparison\cumulative-v1-menhir-9d9675c-20260809\report.md` — `PASS` — 24/24 isolated-path cases; zero false-current errors.
- Event History live panel at `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\results\event-history-acceptance\event-history-production-gate-v6-20260809\report.json` — `PASS` — 5/5; two malformed responses recovered; 14,478 tokens.
- `artifact_validate(artifact_type="wrapups", filename="WRAPUP-2026-08-09-cumulative-activity-event-history-ku78-handoff.md")` — `NOT RUN` — the validator is unavailable in this harness, so status remains `PARTIAL`.

## Claim Cross-Check

- Summary checked against actual code/diff: `yes`
- Files Changed checked against actual modified files: `yes`
- Commit list checked against actual commit hashes or working-tree state: `yes`
- Verification results copied from actual command output: `yes`

## Completion Checklist

- Plan / acceptance criteria completed: `yes` — cumulative scalar/KU78 acceptance is complete; Beacon View/provider/full-500 work is a separate gated plan.
- Docs updated as required: `yes`
- Changelog updated as required: `yes`
- Work committed: `yes` — implementation, evidence, gate plan, and this handoff are on Bench/Menhir default branches.

## Assumptions

1. The v6 artifacts remain immutable evidence and production logic will not be tuned to individual benchmark misses.
2. Beacon's Menhir provider will consume a generic View contract rather than scalar presentation strings.
3. A full-500 run will use reviewed descendants of the pinned code, Event History plus authority enabled, deterministic scalar router/shadow disabled, a fresh graph, and a mixed checkpoint.

## Risks / Gaps

1. `artifact_validate` is unavailable, so this handoff cannot honestly be marked `READY FOR REVIEW`.
2. Repository-wide suites were not rerun after default-branch integration; the focused affected suites pass.
3. The immutable v6 result directory remains in the task clone at `C:\Users\thron\Documents\Codex\2026-08-09\wrapup-cumulative-activity-scalars-and-ku78\work\bench-b7-lf\results\lme-ku-buildout\scalar-event-activity-ku78-v6-20260809`; the Bench ledger is the in-repository evidence index.
4. The completed v6 Neo4j container and volume remain preserved. The container is currently running on host Bolt port 7721 and HTTP port 7508.
5. Bench still has unrelated untracked packet-rescore experiments and historical review artifacts; Beacon has untracked Python bytecode caches. They were not silently deleted or mixed into this work.
6. KU78 v6 had seven misses: `f9e8c073`, `c4ea545c`, `e61a7584`, `a2f3aa27`, `26bdc477`, `031748ae_abs`, and `07741c45`.
7. The full-500 run requires the separate View/provider plan, independent review, explicit cost approval, and a mixed checkpoint.

## Follow-Up Tasks

1. Execute `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\plans\beacon-view-contract-and-full500-gate-2026-08-09.md`.
2. Keep the camera possessive-binding case in backlog until unrelated examples justify a general provenance-bound alias rule.
3. With explicit approval, retire the superseded task branches/clones and decide whether to remove the completed v6 container and volume.
4. Separately review the untracked Bench packet-rescore experiments before committing or discarding them.

## Notes

- Current canonical record: `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\results\lme-ku-buildout\LEDGER.md`.
- KU78 v6 failed task IDs: `f9e8c073`, `c4ea545c`, `e61a7584`, `a2f3aa27`, `26bdc477`, `031748ae_abs`, `07741c45`.
- Versus v4, v6 fixed `5a4f22c0`, `6a1eabeb`, `89941a94`, `c7dc5443`, and `d7c942c3`; `c4ea545c`, `e61a7584`, and `f9e8c073` regressed. Net: +2 versus v4 and +3 versus the prior canonical baseline.
- Bench `master` also contains independent local-extraction-provider support at `b7b16f77aa6c54f810497ac1a344fdf66b57c945`; it is not part of the scalar/KU78 acceptance claim.
