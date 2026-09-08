# WRAPUP — Archolith Bench LME suburbs ingestion fixture

**Date:** 2026-07-16
**Agent:** Codex
**Model:** GPT-5
**Status:** PARTIAL
**Plan / Ticket:** C:\Users\thron\IdeaProjects\.agent\plans\menhir-resolve-suburbs-extraction-failure-handoff.md
**Worktree:** C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench
**Branch:** master
**Commits:** 5b1b442
**Verification Scope:** commit 5b1b442 plus isolated live graph `menhir-lme-suburbs-fix-20260716-v1`
**Docs Updated:** C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\architecture.md; C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\data_models.md; C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\benchmark-notes\menhir-suburbs-ingestion-fixture-2026-07-16.md; C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\fixtures\README.md
**Changelog Updated:** C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md

---

## Before Writing

The requested end state was traced backwards from a repeatable fresh-ingestion gate through direct
graph assertions, fixture-aware ingestion, isolated graph creation, and the exact long utterance
that originally failed. The harness work is complete and committed. The live gate is intentionally
RED because the current Menhir commit still misbinds the suburb fact; a full-corpus rebuild is
therefore not justified yet.

---

## Summary

Archolith Bench now has a one-item LongMemEval-compatible regression fixture that runs through the
real persistent ingester on an isolated Neo4j container and volume. The runner verifies the required
Menhir fix commit before ingestion and then checks graph state directly. Unit and full Bench tests
pass. The first live run correctly demonstrated that the exact long Rachel/Chicago/suburbs utterance
still fails: no suburb entity was created, the current fact remained attached to Chicago, and a
current Chicago edge survived. It also exposed a separate `source="user"` admission-contract issue.

## Files Changed

| File | Why |
|------|-----|
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md` | Records the fixture and first RED live result. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\architecture.md` | Documents the persistent fixture path and rebuild gate. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\benchmark-notes\menhir-suburbs-ingestion-fixture-2026-07-16.md` | Preserves live graph evidence and follow-up criteria. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\data_models.md` | Defines the fixture contract and expectations metadata. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\fixtures\longmemeval\menhir_suburbs_extraction_regression.json` | Supplies the exact 14-turn regression item. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\fixtures\README.md` | Documents running, resuming, and cleaning up the fixture. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\lib\ingest.py` | Adds fixture input, namespace prefixes, correct default paths, and manifest provenance. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\lib\verify_suburbs_fixture.py` | Validates fixture shape, required Menhir commit, and live graph invariants. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\longmemeval\run_suburbs_fixture.sh` | Creates and runs the isolated fresh-ingestion graph. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_longmemeval_ingest_fixture.py` | Covers fixture loading, defaults, namespace behavior, and verifier queries. |

## Verification

- `.venv\Scripts\python.exe -m pytest -q tests/test_longmemeval_ingest_fixture.py -p no:cacheprovider` — `PASS` — 7 passed.
- `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` — `FAIL` — 459 passed, 7 skipped, and 54 setup errors because pytest could not access its external Windows temp root.
- `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .pytest_tmp_codex_suburbs` — `PASS` — 513 passed, 7 skipped.
- `.venv\Scripts\python.exe -m ruff check scripts/longmemeval/lib/ingest.py scripts/longmemeval/lib/verify_suburbs_fixture.py tests/test_longmemeval_ingest_fixture.py` — `PASS` — all checks passed.
- `C:\Program Files\Git\bin\bash.exe -n scripts/longmemeval/run_suburbs_fixture.sh` — `PASS` — shell syntax accepted.
- `.venv\Scripts\python.exe scripts/longmemeval/lib/verify_suburbs_fixture.py --preflight-only` — `PASS` — fixture contract and required Menhir commit were present.
- `C:\Program Files\Git\bin\bash.exe scripts/longmemeval/run_suburbs_fixture.sh` — `FAIL` — expected regression signal: 0 suburb entities, 0 current suburb edges, 1 expired stale edge, 1 current stale edge, and 1 target episode.
- `git diff --cached --check` — `PASS` — no staged whitespace errors before commit.
- `artifact_validate(artifact_type="wrapups", filename="WRAPUP-2026-07-16-archolith-bench-lme-suburbs-fixture.md")` — `NOT RUN` — the validator is unavailable in this harness.

## Claim Cross-Check

- Summary checked against actual code/diff: `yes`
- Files Changed checked against actual modified files: `yes`
- Commit list checked against actual commit hashes or working-tree state: `yes`
- Verification results copied from actual command output: `yes`

## Completion Checklist

- Plan / acceptance criteria completed: `partial` — the Bench fixture is complete, but the linked Menhir extraction outcome remains RED.
- Docs updated as required: `yes`
- Changelog updated as required: `yes`
- Work committed: `yes`

## Assumptions

1. The exact 14-turn LongMemEval utterance is the canonical reproducer; shortening it would hide the
   context-dependent extraction failure.
2. A fresh isolated Neo4j volume is required for each clean gate run.

## Risks / Gaps

1. Menhir commit `c949dfa5e87ba70e1d3a498f81b89b6af77c3980` does not make the exact long fixture pass.
2. The ingester's `source="user"` claim is denied without turn evidence, producing an admission
   artifact and indicating a second Bench-to-Menhir contract mismatch.
3. The failed live graph is deliberately retained as container `menhir-lme-suburbs-fix-20260716-v1`
   with volume `menhir-lme-data-suburbs-fix-20260716-v1` for inspection.
4. Pytest left `.pytest_tmp_codex_suburbs` with an ACL that the managed session could not reclaim;
   it is untracked and not part of commit `5b1b442`.
5. The branch was already 13 commits ahead of origin before this work, so commit `5b1b442` was not
   pushed to avoid publishing unrelated local commits.
6. Mechanical wrapup validation could not be run, so status remains `PARTIAL`.

## Follow-Up Tasks

1. Fix combined extraction for the exact long utterance and rerun `run_suburbs_fixture.sh` until
   all graph assertions pass.
2. Resolve the ingestion source/admission contract and add an assertion that no admission-denial
   entity is created.
3. Only after the fixture is green, run a fresh full-corpus ingest and the separate LongMemEval
   answer-scoring workflow.
4. Run the wrapup validator when available and promote this wrapup to `READY FOR REVIEW` if clean.

## Notes

- Live verification output is at
  `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\results\lme-fixtures\suburbs-fix-20260716-v1\verification.json`.
