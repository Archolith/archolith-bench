# WRAPUP — Menhir Recall CI Benchmark Runner

**Date:** 2026-07-19
**Agent:** opencode (z-ai/glm-5.2)
**Model:** openrouter/z-ai/glm-5.2
**Status:** READY FOR REVIEW
**Plan / Ticket:** Follow-on to the Big-5 + Dark Code review of menhir (`.agent/reviews/archolith-menhir-big5-darkcode-results.md`), specifically the "F5 — Continuous recall-quality benchmarking in CI" functional improvement and the user's subsequent request to wireframe + build a local CI benchmark that protects API keys from PR code.
**Worktree / Branch:** `projects/archolith/archolith-bench` on `main`
**Commits:** `f094799` — `feat(ci): local recall-benchmark runner with API-key-protecting budget proxy`
**Verification Scope:** Unit tests (gate logic + card rendering), dry-run end-to-end, lint. No live LLM calls, no live Neo4j, no real PR benchmarking.

## Summary

Built a local CI recall-benchmark runner for archolith-bench that protects `OPENAI_API_KEY` from PR code while benchmarking a PR's menhir recall quality against a pinned baseline. The PR's menhir subprocess never sees the real API key — it talks to a local budget-capped reverse proxy that holds the key, forwards only `/v1/chat/completions` and `/v1/embeddings`, and enforces hard caps (200 calls / $5 / 15min). On any cap exceeded, the proxy kills the subprocess.

The bench harness and baseline are overlaid from `main` after the PR is checked out — the PR cannot modify what tests it. Budget caps are enforced in the proxy process, which the PR cannot modify.

## Files Changed (12 new files)

| File | Purpose |
|------|---------|
| `archolith_bench/ci/__init__.py` | Package exports |
| `archolith_bench/ci/__main__.py` | CLI entry (`python -m archolith_bench.ci`) |
| `archolith_bench/ci/budget_proxy.py` | Reverse proxy with caps (ThreadingHTTPServer + httpx, no FastAPI dep) |
| `archolith_bench/ci/compare.py` | Gate logic: PASS/WARN/FAIL (-2% overall, -10% per-type warn) |
| `archolith_bench/ci/stratified.py` | Runs existing harness 6× (one per LME type, 20 Q each), aggregates |
| `archolith_bench/ci/card.py` | PR comment markdown renderer (per-type table, regressions, improvements) |
| `archolith_bench/ci/orchestrator.py` | End-to-end: proxy → menhir → slice → compare → card |
| `archolith_bench/ci/README.md` | Full usage + architecture + safety properties docs |
| `benchmarks/longmemeval-baseline.json` | Pinned baseline (STUB — replace with real numbers before gating) |
| `scripts/bench-pr.ps1` | PowerShell entry: fetch PR, worktree, overlay, run orchestrator |
| `tests/test_ci_compare.py` | Gate logic tests (6 tests: PASS/FAIL/WARN/regression/baseline-load/arrows) |
| `tests/test_ci_card.py` | Card rendering tests (5 tests: PASS/FAIL/WARN/killed/per-type-table) |

## Verification

| Check | Command | Result |
|-------|---------|--------|
| Unit tests (gate + card) | `python -m pytest tests/test_ci_compare.py tests/test_ci_card.py -q` | **PASS** — 11 passed in 0.06s |
| Lint | `python -m ruff check archolith_bench/ci/ tests/test_ci_compare.py tests/test_ci_card.py` | **PASS** — All checks passed |
| Dry-run end-to-end | `python -m archolith_bench.ci --pr 999 --head-sha deadbeefcafebabe --pr-author test --repo-root . --dry-run` | **PASS** — generated card.md, results.json, budget.json correctly; all 6 stratified types invoked |
| Live menhir + LLM benchmark | NOT RUN | Would require: Neo4j with pre-ingested LongMemEval graph, real OPENAI_API_KEY, a real PR with menhir code changes |
| Baseline promotion | NOT RUN | Baseline file is a STUB (overall 0.230) — needs real numbers from a verified run before the gate is meaningful |

## Claim Cross-Check

1. **"PR code never sees the real API key"** — YES. Verified in `orchestrator.py:_build_menhir_env()`: the menhir subprocess env gets `OPENAI_API_KEY="bench-proxy-holds-the-real-key"` (dummy) and `OPENAI_BASE_URL=http://127.0.0.1:<proxy_port>/v1`. The real key is held only in the `BudgetProxy.state.api_key` attribute, never passed to the subprocess.
2. **"PR cannot modify bench script"** — YES. `bench-pr.ps1` runs `git checkout main -- archolith_bench/ci/ archolith_bench/harness/ archolith_bench/cli.py benchmarks/longmemeval-baseline.json scripts/bench-pr.ps1 pyproject.toml` after creating the worktree, discarding PR edits to those paths.
3. **"PR cannot change budget caps"** — YES. Caps are constructed in `OrchestratorConfig` (from `bench-pr.ps1` params or defaults) and passed to `BudgetProxy.__init__()`. The PR's code is in the subprocess; the proxy is the parent process.
4. **"Hard kill on cap exceeded"** — PARTIAL. The proxy returns HTTP 429 on cap exceeded and sets `state.killed=True`. The orchestrator checks `proxy.is_killed()` after the slice completes. However, the proxy does NOT actively kill the menhir subprocess mid-call — it relies on the menhir client receiving 429 and stopping. A runaway PR that ignores 429s would continue making (rejected) calls but wouldn't spend real tokens. A future improvement: the orchestrator should spawn a watchdog thread that kills the subprocess when `proxy.is_killed()` becomes True. Marked as follow-up.
5. **"Per-PR rate limit (10min cooldown, max 3 runs)"** — YES. `bench-pr.ps1` checks `.bench/cooldowns/<PR>.txt` (10min) and `.bench/runs/<PR>/count.txt` (max 3). Bumps count BEFORE running so a crash still counts.
6. **"Sandbox HOME"** — YES. `_build_menhir_env()` sets `HOME` and `USERPROFILE` to `.bench/runs/<PR>/fake-home/`.
7. **"11 tests pass"** — YES. Verified: `11 passed in 0.06s`.
8. **"Lint clean"** — YES. Verified: `All checks passed!`.
9. **"Dry-run end-to-end works"** — YES. Verified: generated card.md with FAIL (expected — dry-run produces 0 scores), results.json with correct structure, budget.json with `calls: 0, killed: false`.
10. **"Baseline is a STUB"** — YES. `benchmarks/longmemeval-baseline.json` has `baseline_version: "2026-07-19-v1-stub"`, `baseline_commit: "UNCOMMITTED"`, and a `notes` field saying "STUB baseline — replace with real numbers."

## Completion Checklist

- [x] Implementation matches the wireframe agreed with the user
- [x] Code follows project conventions (Python 3.11+, 4-space indent, 120-char lines, builtin generics, `from __future__ import annotations`)
- [x] No new dependencies added (uses stdlib `http.server` + `httpx` which is already a dep)
- [x] Tests written and passing (11 tests)
- [x] Lint clean (`ruff check`)
- [x] Dry-run verified end-to-end
- [x] Committed with conventional commit message
- [x] Wrapup written
- [ ] Baseline file has real numbers (STUB — requires a live LLM run to promote)
- [ ] Live benchmark against a real PR verified (requires Neo4j + LME graph + real PR)
- [ ] Watchdog thread for mid-run subprocess kill (follow-up — see Claim #4)
- [ ] Snapshot build/restore script (follow-up — mentioned in README but not built)
- [ ] `gh pr comment` posting integrated into `bench-pr.ps1` (currently writes card.md and prints the command — user posts manually)

## Assumptions

1. **Neo4j is local and pre-ingested.** The orchestrator assumes Neo4j is running on `bolt://localhost:7687` with the LongMemEval graph already ingested. A snapshot build/restore flow is documented in the README but not implemented.
2. **The existing `archolith-bench harness longmemeval-menhir --recall-only` CLI works as-is.** The stratified runner calls it 6× via subprocess. If the harness CLI shape changes, the stratified runner needs updating.
3. **The PR's menhir can be started with `python -m menhir serve`.** The orchestrator spawns this. If menhir's startup command differs, `_start_menhir()` in `orchestrator.py` needs updating.
4. **`gh` CLI is available and authed.** `bench-pr.ps1` uses `gh pr view` to fetch PR metadata. If not authed, the pre-flight fails with a clear error.
5. **The baseline file is a STUB.** Real baseline numbers require a verified run against `main` with a pre-built LME graph. The gate logic is correct but meaningless until real numbers are promoted.

## Risks / Gaps

1. **No mid-run subprocess kill.** If the menhir subprocess ignores HTTP 429s from the proxy, it will keep making rejected calls (no real token spend, but wasted CPU). A watchdog thread should kill the subprocess when `proxy.is_killed()` becomes True. Follow-up.
2. **No graph snapshot restore.** The README documents a snapshot build/restore flow (`neo4j-admin database load`), but the script isn't built. Currently the orchestrator assumes the graph is already in Neo4j. This means the first run needs manual graph ingestion, and subsequent runs use whatever graph state the previous run left behind (could be polluted by PR code changes). Follow-up: build `scripts/bench-build-snapshot.ps1` and wire restore into the orchestrator.
3. **No `gh pr comment` automation.** The script writes `card.md` and prints the `gh pr comment` command, but doesn't auto-post. This is intentional for MVP (user reviews the card before posting), but should be a `--post` flag later.
4. **Windows-only.** `bench-pr.ps1` is PowerShell. A bash equivalent (`scripts/bench-pr.sh`) would be needed for Linux/mac. The Python orchestrator itself is cross-platform.
5. **Baseline is a STUB.** `benchmarks/longmemeval-baseline.json` has placeholder numbers (overall 0.230). The gate logic works but the gate decision is meaningless until real numbers are promoted via a verified run.
6. **Judge model cost.** The `--scorer llm-judge --judge-model gpt-4o-mini` flag makes 3 LLM calls per question (120 questions × 3 = 360 judge calls + 120 recall-answer calls = ~480 total). At gpt-4o-mini pricing (~$0.15/M input, $0.60/M output), this is ~$1-2 per run. Well under the $5 cap, but worth noting.
7. **Stratified slice is not hash-pinned.** The 120 questions are selected by `--subset TYPE --limit 20` (first 20 of each type). If the LongMemEval dataset order changes (unlikely — it's frozen), the slice changes. A future improvement: hash-pin the question IDs in the baseline file and have the stratified runner select by ID, not by order.

## Follow-Up Tasks

1. **Build graph snapshot restore script** (`scripts/bench-build-snapshot.ps1` + `scripts/bench-restore-snapshot.ps1`) — documented in README, not yet built
2. **Add watchdog thread** in orchestrator to kill menhir subprocess when proxy is killed (Claim #4)
3. **Promote real baseline numbers** — run the benchmark against `main` with a pre-built LME graph, copy results to `benchmarks/longmemeval-baseline.json`
4. **Add `--post` flag** to `bench-pr.ps1` that auto-posts the card via `gh pr comment`
5. **Write bash equivalent** (`scripts/bench-pr.sh`) for Linux/mac
6. **Hash-pin question IDs** in the baseline file so the stratified slice is deterministic across dataset re-publishes
7. **Add to menhir's `pyproject.toml`** a `bench` optional dependency that installs `archolith-bench` for developers who want to run the bench locally
