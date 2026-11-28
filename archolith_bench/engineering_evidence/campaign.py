"""The v2 sprint campaign: chains (arm x repeat) run in parallel under one shared budget.

Reuses the bench's existing pieces rather than a second implementation:

* ``beacon_eval.runner.Budget`` / ``BudgetExhausted`` and ``core.opencode.process.RateLimited``: one budget shared by
  every chain, checked before each session (``check`` holds a reserve per session in flight).
  Menhir's server-side spend is added with ``Budget.spend(0, usd)``.
* ``core.opencode.process._cancel_scope`` / ``_kill_live_children``: the interrupt handling of
  ``_run_parallel``, at chain level (its unit of work is one task run, so it cannot be called
  directly: a sprint chain is sequential and stateful).
* ``staged.rundir.RunDir`` / ``StageFile``: append-only JSONL rows, ``config.resolved.yaml``
  and the config-mismatch refusal.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from archolith_bench.beacon_eval.runner import Budget, BudgetExhausted
from archolith_bench.core.opencode.process import (
    DRAIN_TIMEOUT_S,
    RateLimited,
    _cancel_scope,
    _kill_live_children,
)
from archolith_bench.engineering_evidence.server_usage import ServerRates
from archolith_bench.engineering_evidence.sprint import (
    SessionOutcome,
    SprintFixture,
    compose_full_prompt,
    is_notes_path,
    score_execution,
)
from archolith_bench.engineering_evidence.sprint_rubric import (
    RUBRIC_VERSION,
    ChainView,
    build_verdict_report,
)
from archolith_bench.staged.rundir import RunConfigMismatch, RunDir, StageFile

PROGRESS_SCRIPT = Path.home() / ".claude" / "tools" / "progress.py"
SESSIONS_FILE = "sessions.jsonl"


class RunDirInUse(RunConfigMismatch):
    """The run directory already exists and --resume was not given; nothing is written."""


class ProgressFeed:
    """`python ~/.claude/tools/progress.py start|update|done` via subprocess (stderr visible).

    A missing script is skipped with one warning; a failing call never stops the campaign.
    """

    def __init__(self, job_id: str, label: str, total: int, *, ceiling_usd: float | None,
                 script: Path = PROGRESS_SCRIPT, run_fn: Callable[..., Any] = subprocess.run) -> None:
        self.job_id, self.label, self.total, self.ceiling = job_id, label, total, ceiling_usd
        self.script, self._run = script, run_fn
        self.enabled = script.is_file()
        self._lock = threading.Lock()
        if not self.enabled:
            print(f"warning: {script} not found; progress feed skipped", file=sys.stderr)

    def _call(self, *args: str) -> None:
        if not self.enabled:
            return
        try:
            self._run([sys.executable, str(self.script), *args], check=False)
        except OSError as exc:
            print(f"warning: progress feed failed: {exc}", file=sys.stderr)

    def start(self) -> None:
        extra = ["--ceiling", f"{self.ceiling}"] if self.ceiling is not None else []
        self._call("start", self.job_id, self.label, "--total", str(self.total), "--unit", "sessions", *extra)

    def update(self, done: int, spend_usd: float, note: str = "") -> None:
        with self._lock:
            self._call("update", self.job_id, "--done", str(done), "--spend", f"{spend_usd:.4f}",
                       *(["--note", note] if note else []))

    def finish(self, failed: bool = False) -> None:
        self._call("done", self.job_id, *(["--status", "failed"] if failed else []))


def prompt_manifest(fixture: SprintFixture, arms: tuple[str, ...],
                    cache_shaping: bool = False) -> dict[str, dict[str, str]]:
    """Every arm's exact prompts (namespace left as a placeholder), for the hash and the results."""
    return {
        arm: {s.session_id: compose_full_prompt(s, arm, "<namespace>", cache_shaping,
                                                 project=fixture.repo.name)
              for s in fixture.sessions}
        for arm in arms
    }


def prompt_hash(manifest: dict[str, dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def chain_namespace(fixture: SprintFixture, arm: str, repeat: int, run_id: str, attempt: int = 1) -> str:
    # Opaque on purpose: a namespace that resembles the checkout dir name
    # ("<fixture>-<arm>-r<k>") made Menhir-arm agents splice it into file paths, which OpenCode
    # refused, ending 4/15 sessions per repo in run v2-20261127. The chain row records the mapping.
    key = f"{fixture.name}|{arm}|{repeat}|{run_id}|{attempt}"
    return "sv2-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _outcome_from_row(row: dict[str, Any]) -> SessionOutcome:
    names = {f.name for f in fields(SessionOutcome)}
    return SessionOutcome(**{k: v for k, v in row.items() if k in names})


@dataclass
class _Chain:
    arm: str
    repeat: int
    attempt: int
    namespace: str
    outcomes: list[SessionOutcome] = field(default_factory=list)
    reused: bool = False
    error: str = ""


class Campaign:
    """One fixture's campaign. All shared mutable state is guarded by ``self.lock``."""

    def __init__(
        self,
        fixture: SprintFixture,
        *,
        executor: Any,
        run_dir: RunDir,
        run_id: str,
        arms: tuple[str, ...],
        repeats: int,
        parallel: int,
        budget: Budget,
        rates: ServerRates | None = None,
        progress: ProgressFeed | None = None,
        resume: bool = False,
        dry_run: bool = False,
        repo_root: Path | None = None,
        since_iso: str | None = None,
    ) -> None:
        self.fixture, self.executor, self.rd, self.run_id = fixture, executor, run_dir, run_id
        self.arms, self.repeats, self.parallel = arms, repeats, max(1, parallel)
        self.budget, self.rates, self.progress = budget, rates, progress
        self.resume, self.dry_run, self.repo_root = resume, dry_run, repo_root
        self.since_iso = since_iso or datetime.now(timezone.utc).isoformat()
        self.lock = threading.Lock()
        self.stop = ""
        self.done_sessions = 0
        self.server_usd_seen = 0.0
        self.namespaces: list[str] = []
        self.sessions_file = StageFile(run_dir.path / SESSIONS_FILE)

    # -- shared state ----------------------------------------------------------------

    def _first_stop(self, message: str) -> None:
        with self.lock:
            if not self.stop:
                self.stop = message

    def _append(self, chain: _Chain, outcome: SessionOutcome) -> None:
        row = {
            **asdict(outcome),
            "item_id": f"{chain.arm}/r{chain.repeat}/a{chain.attempt}/{outcome.session_id}",
            "repeat": chain.repeat,
            "attempt": chain.attempt,
            "namespace": chain.namespace,
        }
        with self.lock:
            self.sessions_file.append(row)
            self.done_sessions += 1
            done, spend = self.done_sessions, self.budget.used_usd
        if self.progress:
            self.progress.update(done, spend, f"{chain.arm} r{chain.repeat} {outcome.session_id}")

    def _server_spend(self) -> None:
        """Add Menhir's new server-side spend to the shared budget (called with the lock held)."""
        usage_fn = getattr(self.executor, "server_usage", None)
        if usage_fn is None or self.rates is None or not self.namespaces:
            return
        try:
            usage = usage_fn(list(self.namespaces), since_iso=self.since_iso, rates=self.rates)
        except Exception as exc:  # noqa: BLE001 - never lose a session to a metering failure
            print(f"warning: server usage unavailable for the budget: {exc}", file=sys.stderr)
            return
        if not usage:
            return
        # by_namespace costs already include each chain's even share of unattributed rows.
        total = sum(u["cost_usd"] for u in usage["by_namespace"].values())
        delta = total - self.server_usd_seen
        if delta > 0:
            self.budget.spend(0, delta)
            self.server_usd_seen = total

    # -- one chain -------------------------------------------------------------------

    def _not_run(self, session: Any, arm: str, repeat: int, reason: str) -> SessionOutcome:
        return SessionOutcome(
            session_id=session.session_id, phase=session.phase, arm=arm, scores={}, overall_score=0.0,
            total_tokens=0, input_tokens=0, output_tokens=0, turns=0, tool_calls=0, files_read=[],
            redundant_file_reads=0, memories_saved=0, todos_saved=0, seconds=0.0,
            not_run=True, not_run_reason=reason, repeat=repeat,
        )

    def _run_chain(self, chain: _Chain) -> None:
        seen: set[str] = set()
        for idx, session in enumerate(self.fixture.sessions):
            if chain.error or self.stop:
                reason = chain.error or self.stop
                outcome = self._not_run(session, chain.arm, chain.repeat, reason)
                chain.outcomes.append(outcome)
                self._append(chain, outcome)
                continue
            with self.lock:
                try:
                    self.budget.check()
                    cap = self.budget.cap_usd
                    if cap is not None and self.budget.used_usd >= cap:  # "at or over" never starts
                        raise BudgetExhausted(f"${self.budget.used_usd:.4f} spent is at the ${cap:.2f} cap")
                except BudgetExhausted as exc:
                    admitted = False
                    if not self.stop:
                        self.stop = str(exc)
                else:
                    admitted = True
                    self.budget.in_flight += 1
            if not admitted:
                outcome = self._not_run(session, chain.arm, chain.repeat, self.stop)
                chain.outcomes.append(outcome)
                self._append(chain, outcome)
                continue
            try:
                res = self.executor.run_session(
                    fixture=self.fixture, session=session, arm=chain.arm, session_index=idx,
                    namespace=chain.namespace, repeat=chain.repeat,
                )
            except RateLimited as exc:
                self._first_stop(f"rate_limited: {exc}")
                outcome = self._not_run(session, chain.arm, chain.repeat, self.stop)
                chain.outcomes.append(outcome)
                self._append(chain, outcome)
                continue
            except Exception as exc:  # noqa: BLE001 - one broken chain must not end the others
                chain.error = f"chain_error: {type(exc).__name__}: {exc}"[:300]
                outcome = self._not_run(session, chain.arm, chain.repeat, chain.error)
                chain.outcomes.append(outcome)
                self._append(chain, outcome)
                continue
            finally:
                with self.lock:
                    self.budget.in_flight -= 1

            scored, overall = score_execution(res, session, chain.arm, self.repo_root)
            if res.censored:
                scored, overall = {}, 0.0
            files = [f for f in dict.fromkeys(res.files_read) if not is_notes_path(f)]
            redundant = sum(1 for f in files if f in seen) if idx > 0 else 0
            seen.update(files)
            outcome = SessionOutcome(
                session_id=session.session_id, phase=session.phase, arm=chain.arm,
                scores={k: round(float(v), 4) for k, v in scored.items()},
                overall_score=round(overall, 4),
                total_tokens=res.total_tokens, input_tokens=res.input_tokens,
                output_tokens=res.output_tokens, reasoning_tokens=res.reasoning_tokens,
                cache_read_tokens=res.cache_read_tokens, cache_write_tokens=res.cache_write_tokens,
                uncached_tokens=res.uncached_tokens, cost_usd=res.cost_usd, turns=res.turns,
                tool_calls=res.tool_calls, files_read=files, redundant_file_reads=redundant,
                memories_saved=res.memories_saved, todos_saved=res.todos_saved,
                seconds=round(res.seconds, 3), error=res.error, censored=res.censored,
                notes_violations=list(res.notes_violations),
                refused_reads=getattr(res, "refused_reads", 0),
                enrichment_wait_s=res.enrichment_wait_s, enrichment_drained=res.enrichment_drained,
                repeat=chain.repeat,
            )
            with self.lock:
                self.budget.spend(res.total_tokens, res.cost_usd)
                if chain.arm.startswith("menhir"):
                    self._server_spend()
            chain.outcomes.append(outcome)
            self._append(chain, outcome)

    # -- planning, resume and the pool -----------------------------------------------

    def _saved_rows(self) -> list[dict[str, Any]]:
        return self.sessions_file.records()

    def _plan(self) -> list[_Chain]:
        saved = self._saved_rows() if self.resume else []
        chains: list[_Chain] = []
        expected = len(self.fixture.sessions)
        for repeat in range(1, self.repeats + 1):
            for arm in self.arms:
                rows = [r for r in saved if r.get("arm") == arm and r.get("repeat") == repeat]
                attempts = sorted({int(r.get("attempt", 1)) for r in rows})
                for r in rows:  # a replaced attempt still cost real money
                    self.budget.spend(int(r.get("total_tokens", 0)), float(r.get("cost_usd", 0.0)))
                    if arm.startswith("menhir") and r["namespace"] not in self.namespaces:
                        self.namespaces.append(r["namespace"])
                latest = max(attempts, default=0)
                last_rows = [r for r in rows if int(r.get("attempt", 1)) == latest]
                complete = len(last_rows) == expected and not any(r.get("not_run") for r in last_rows)
                if complete:
                    chain = _Chain(arm, repeat, latest, last_rows[0]["namespace"], reused=True)
                    chain.outcomes = [_outcome_from_row(r) for r in last_rows]
                else:
                    att = latest + 1
                    chain = _Chain(arm, repeat, att, chain_namespace(self.fixture, arm, repeat, self.run_id, att))
                    if arm.startswith("menhir"):
                        self.namespaces.append(chain.namespace)
                chains.append(chain)
        return chains

    def run(self) -> tuple[list[_Chain], str]:
        chains = self._plan()
        todo = [c for c in chains if not c.reused]
        self.done_sessions = sum(len(c.outcomes) for c in chains if c.reused)
        if self.progress:
            self.progress.start()
        interrupted: BaseException | None = None
        with _cancel_scope() as cancel, ThreadPoolExecutor(
            max_workers=self.parallel, thread_name_prefix="sprint-chain"
        ) as pool:
            pending: dict[Future[None], _Chain] = {pool.submit(self._run_chain, c): c for c in todo}
            try:
                while pending:
                    done, _ = wait(pending, timeout=1.0)
                    for fut in done:
                        chain = pending.pop(fut)
                        exc = fut.exception()
                        if exc is not None:
                            chain.error = f"chain_error: {type(exc).__name__}: {exc}"[:300]
                            self._first_stop(chain.error)
            except BaseException as exc:  # noqa: BLE001 - an interrupt ends the campaign promptly
                interrupted = exc
                self._first_stop("KeyboardInterrupt")
                cancel.set()
                for fut in pending:
                    fut.cancel()
                _kill_live_children()
                while pending:
                    done, _ = wait(pending, timeout=DRAIN_TIMEOUT_S)
                    for fut in done:
                        pending.pop(fut)
                    if not done:
                        break
        if interrupted is not None:
            raise interrupted
        return chains, self.stop

    # -- report ----------------------------------------------------------------------

    def final_server_usage(self, chains: list[_Chain]) -> dict[str, Any]:
        """Drain enrichment on every Menhir namespace, then attribute server usage per chain."""
        menhir = [c for c in chains if c.arm.startswith("menhir")]
        drains: dict[str, Any] = {}
        wait_fn = getattr(self.executor, "wait_for_enrichment", None)
        usage_fn = getattr(self.executor, "server_usage", None)
        if not menhir or usage_fn is None:
            return {"available": False, "reason": "no_menhir_chains_or_no_telemetry_db"}
        if wait_fn is not None:
            for c in menhir:
                if not c.reused:
                    drains[c.namespace] = wait_fn(c.namespace)
        usage = usage_fn([c.namespace for c in menhir], since_iso=self.since_iso, rates=self.rates)
        if usage is None:
            return {"available": False, "reason": "no_telemetry_db", "final_drain": drains}
        usage["available"] = True
        usage["final_drain"] = drains
        return usage

    def build_report(
        self, chains: list[_Chain], stop: str, *, provenance: dict[str, Any],
        structure_index: dict[str, Any] | None,
    ) -> dict[str, Any]:
        usage = self.final_server_usage(chains)
        by_ns = usage.get("by_namespace", {}) if usage.get("available") else {}
        views = [
            ChainView(c.arm, c.repeat, tuple(c.outcomes), by_ns.get(c.namespace, {})) for c in chains
        ]
        analysis = build_verdict_report(views, expected_sessions=len(self.fixture.sessions))
        not_run = sum(1 for c in chains for o in c.outcomes if o.not_run)
        return {
            "suite": "engineering_evidence_longitudinal_sprint_v2",
            "rubric_version": RUBRIC_VERSION,
            "dry_run": self.dry_run,
            "fixture": self.fixture.name,
            "repo": asdict(self.fixture.repo),
            "run_id": self.run_id,
            "provenance": provenance,
            "structure_index": structure_index,
            "campaign": {
                "stopped_early": bool(stop) or not_run > 0,
                "stop_reason": stop,
                "sessions_not_run": not_run,
                "max_usd": self.budget.cap_usd,
                "spent_usd": round(self.budget.used_usd, 6),
                "note": (
                    "The campaign stopped early: sessions marked not_run were never executed and "
                    "are excluded from accuracy, never scored as 0; chains missing a session are "
                    "left out of the verdict inputs." if (stop or not_run) else ""
                ),
            },
            "server_usage": {k: v for k, v in usage.items() if k != "by_namespace"},
            "chains": [
                {
                    "arm": c.arm, "repeat": c.repeat, "attempt": c.attempt, "namespace": c.namespace,
                    "reused_from_resume": c.reused, "error": c.error,
                    "server": by_ns.get(c.namespace, {}),
                    "sessions": [asdict(o) for o in c.outcomes],
                }
                for c in chains
            ],
            "analysis": analysis,
        }


def prepare_run_dir(root: Path, name: str, resolved: dict[str, Any], *, resume: bool) -> RunDir:
    """The run directory for *name*: refuses an existing one unless resuming with the same config.

    ``RunDir.write_config`` raises ``RunConfigMismatch`` when a different config is already there.
    """
    rd = RunDir(root, name)
    if rd.path.exists() and not resume:
        raise RunDirInUse(f"{rd.path} already exists; pick a new --run-id or pass --resume")
    rd.write_config(resolved)
    return rd
