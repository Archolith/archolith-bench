# WRAPUP — archolith-bench remediation

**Date:** 2026-06-21  
**Agent:** Codex  
**Model:** GPT-5 Codex  
**Status:** PARTIAL  
**Plan / Ticket:** `C:\Users\thron\IdeaProjects\projects\archolith\.agent\plans\archive\archolith-bench-remediation-plan-2026-06-20.md`; remaining follow-up plan `C:\Users\thron\IdeaProjects\projects\archolith\.agent\plans\archolith-bench-remaining-evidence-closeout-plan-2026-06-21.md`  
**Worktree:** `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench`  
**Branch:** `master`  
**Commits:** `aaa8a19`, `c3b1423`, `540c5de`, `5828364`  
**Verification Scope:** Bench-local remediation commits on `master` from base `0a56b76` through `5828364`; docs/artifact closeout staged separately in this wrapup commit  
**Docs Updated:** `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\README.md`, `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\AGENTS.md`, `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\BENCHMARKS.md`, `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\benchmarks\README.md`, `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\architecture.md`, `C:\Users\thron\IdeaProjects\projects\archolith\.agent\reviews\archolith-bench-outstanding-issues.md`, `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\plans\archolith-bench-industry-benchmark-rollout-plan.md`  
**Changelog Updated:** `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md`

---

## Summary

Implemented the approved local bench remediation sessions: Session A token-count bias fix, Session B memory A/B client threading, Session E proxy trace polling/schema fix, Session F external env hardening, Session G offline coverage, the approved safe-fix set, local Tier 2 metric-quality fixes, and the concrete Tier 3 backlog items. The missing industry rollout deferral plan was created, `BENCHMARKS.md` was regenerated with the stale-evidence caveat intact, and the outstanding-issues review now records implemented versus deferred status.

The local code/test remediation is committed in four scoped chunks. Deferred/blocking items remain outside the local remediation pass: P4 benchmark evidence refresh, official LongMemEval LLM-judge evidence mode, scenario representativeness and variance evidence, live external benchmark execution, distribution-plan execution, and PR/push closeout.

## Files Changed

| File | Why |
|------|-----|
| `C:\Users\thron\IdeaProjects\projects\archolith\.agent\reviews\archolith-bench-outstanding-issues.md` | Added uncommitted remediation status mapped back to audit IDs and verification evidence. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\CHANGELOG.md` | Recorded Session A, safe fixes, Session B, Session F, and Session E remediation entries. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\architecture.md` | Prior token-accounting doc update for shared maintenance helper. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\.agent\plans\archolith-bench-industry-benchmark-rollout-plan.md` | Created the Session D / P5 deferral rollout plan. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\AGENTS.md` | Documented checkpoint/config notes and shared token-counting location. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\BENCHMARKS.md` | Regenerated through the CLI with stale-evidence caveat and separate upstream/internal savings columns. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\benchmarks\README.md` | Clarified MTEB artifact nomenclature as a single-arm embedding-model component baseline. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\README.md` | Documented checkpoint lifecycle. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\arms.py` | Removed dead `filter_only` config override. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\cli.py` | Added `--poll-interval` and threaded it through proxy/experiment/stack entrypoints. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\core\api.py` | Removed unused timing local. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\core\display.py` | Removed unused imports/locals and placeholder-free f-string prefixes. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\core\metrics.py` | Fixed empty token counts and message-token floor; retained shared maintenance primitive use. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\core\report.py` | Loaded benchmark JSON once, added generated stale-evidence caveat, and separated upstream/internal savings columns. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\external.py` | Added env allowlist, configurable timeout, MTEB warning, and local fallback comment. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\bigcodebench.py` | Uses secure temp directories for generated-code subprocess execution. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\longmemeval.py` | Tightened deterministic scoring to reject obvious negated-answer false positives. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\longbench_v2.py` | Uses last-letter fallback extraction when no explicit answer format is present. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\memory_ab.py` | Threaded a real chat client through memory A/B and tightened production markers. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\menhir_client.py` | Added context-manager lifecycle support to `HttpMenhirClient` and removed O(n²) stub recall sorting. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\harness\tempfiles.py` | Added shared secure temporary-directory helper. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\suites\continuity.py` | Broadened path extraction for Windows, relative, dotfile, and common no-extension paths. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\suites\probes.py` | Added morphology-aware fact-probe keyword matching. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\suites\proxy.py` | Added poll interval parameter, 1-based trace selection, fallback warning, and upstream input reduction metric. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\suites\restart.py` | Removed unused import and changed restart scoring to flag re-read intent separately from fact recovery. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\archolith_bench\suites\stack.py` | Threaded poll interval through stack suite. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\pyproject.toml` | Removed deprecated license classifier, added ruff dev dependency/config, corrected project URLs, retained shared maintenance dependency. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_cost_model.py` | Removed unused imports. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_harness.py` | Added memory client lifecycle, env allowlist, and production-marker regressions. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_metric_quality.py` | Added Tier 2 metric-quality regressions for probe matching and path extraction. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_proxy_remediation.py` | Added Session E regressions for trace selection, fallback warning, upstream ratio, and poll-interval defaults. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_proxy_run_benchmark.py` | Added offline coverage for the main proxy loop, checkpoint resume/cleanup, and collapse abort. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_report.py` | Added report persistence and generated BENCHMARKS.md coverage. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_restart.py` | Added restart/bootstrap orientation scoring coverage. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_tempfiles.py` | Added secure tempdir helper coverage. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\tests\test_token_estimator.py` | Added Session A regressions for empty/None/whitespace and multipart content handling. |
| `C:\Users\thron\IdeaProjects\projects\archolith\archolith-bench\scripts\run_mteb_local.py` | Documented intentional separation from `MtebAdapter`. |

## Verification

- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest tests\test_proxy_remediation.py tests\test_token_estimator.py tests\test_harness.py -q -p no:cacheprovider` — `PASS` — `49 passed in 2.16s`.
- `python -m ruff check archolith_bench\cli.py archolith_bench\suites\proxy.py archolith_bench\suites\stack.py tests\test_proxy_remediation.py` — `PASS` — `All checks passed!`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest tests\test_proxy_run_benchmark.py tests\test_restart.py tests\test_report.py tests\test_cost_reporting.py -q -p no:cacheprovider` — `PASS` — `11 passed in 1.08s`.
- `python -m ruff check archolith_bench\core\report.py tests\test_proxy_run_benchmark.py tests\test_restart.py tests\test_report.py` — `PASS` — `All checks passed!`.
- `python -m archolith_bench report --results-dir results --out BENCHMARKS.md` — `PASS` — `Written to BENCHMARKS.md`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest -q -p no:cacheprovider` — `PASS` — `100 passed in 5.75s`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest tests\test_metric_quality.py tests\test_restart.py tests\test_harness.py -q -p no:cacheprovider` — `PASS` — `35 passed in 3.05s`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest -q -p no:cacheprovider` — `PASS` — `104 passed in 4.86s`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest tests\test_harness.py tests\test_cost_model.py tests\test_proxy_remediation.py tests\test_tempfiles.py -q -p no:cacheprovider` — `PASS` — `54 passed in 2.83s`.
- `$env:TMP='C:\tmp'; $env:TEMP='C:\tmp'; python -m pytest -q -p no:cacheprovider` — `PASS` — `111 passed in 7.41s`.
- `python -m ruff check .` — `PASS` — `All checks passed!`.
- `python -m pip install -e . --dry-run --no-deps` — `PASS` — `Would install archolith-bench-0.4.0`.
- `artifact_validate(artifact_type="wrapups", filename="WRAPUP-2026-06-21-archolith-bench-remediation.md")` — `NOT RUN` — validator tool is not exposed in the current toolset.

## Claim Cross-Check

- Summary checked against actual code/diff: `yes`
- Files Changed checked against actual modified files: `yes`
- Commit list checked against actual commit hashes or working-tree state: `yes`
- Verification results copied from actual command output: `yes`
- If any item is `no`, explain the mismatch here before claiming review readiness.

## Completion Checklist

- Plan / acceptance criteria completed: `partial`
- Docs updated as required: `yes`
- Changelog updated as required: `yes`
- Work committed: `yes`
- If uncommitted, explain why the work is still only anchored to the current worktree and why a commit was not made before wrapup.

The bench-local code/test remediation is now commit-anchored. Remaining open work is deliberately tracked in the follow-up plan because it requires live services, API budget, external tools, distribution decisions, or unavailable artifact validation.

## Assumptions

1. The shared token-accounting changes were intentional and are included in commit `aaa8a19`.
2. Session C is satisfied by the existing workspace-level `ARCHOLITH-DISTRIBUTION-PLAN.md`; executing that distribution plan remains out of scope for this bench remediation pass.
3. Session D is satisfied by creating the rollout plan, not by installing/running external benchmark tools.

## Risks / Gaps

1. The local remediation is committed, but the wrapup validator was unavailable, so this wrapup remains `PARTIAL` until validation can run.
2. No fresh P4 benchmark evidence was run; it needs live proxy/API budget and a deliberate benchmark schedule.
3. Official LongMemEval LLM-judge scoring remains deferred; the deterministic scorer was tightened locally, but official judge mode needs API budget and a flag/config decision.
4. Scenario representativeness and variance/error-bar evidence remain blocked on real workload characterization and multi-run benchmark execution.
5. External benchmark execution remains deferred; SWE-bench, CyberSecEval, AgentDojo, OWASP, DMR, and full LongMemEval evidence require external tooling or throwaway services.
6. Push/PR closeout remains open; no push or PR was requested in this pass.
7. Parent `projects\archolith` artifact commits are separate from these bench-local commits and should be handled with explicit paths only because that parent repo has unrelated dirty state.

## Follow-Up Tasks

1. Schedule P4 fresh evidence runs after live proxy/API budget is approved.
2. Decide whether official LongMemEval LLM judge mode should be implemented for paid evidence runs.
3. Define the scenario representativeness and variance policy before publishing broader launch claims.
4. Execute the distribution plan separately if packaging/publication is still in launch scope.
5. Run wrapup validation when the artifact validator is available, then move the wrapup to `READY FOR REVIEW` only if validation passes.
6. Push/open PR if that is the desired publication path for the four bench-local commits.
