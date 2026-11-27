"""Longitudinal frozen-repo multi-session coding-sprint benchmark (LLM-ready + dry-run).

Extends the single-shot frozen-repo harness in `archolith_bench.beacon_eval` into a
stateful 4-session sprint over a pinned external OSS repository (e.g. `simonw/llm`
at `764dc386`) comparing three arms:

  1. `no_menhir`           — built-in file/search tools only, fresh context per session.
  2. `menhir_structure`    — built-in tools + deterministic Menhir code graph
                             (`query_structure`, `blast_radius`, `affected_tests`; 0 LLM ingest).
  3. `menhir_longitudinal` — built-in tools + Menhir code graph + live `add_memory` /
                             `recall_memories` / `add_todo` persisted in one isolated
                             namespace across Sessions 1 -> 4.

Tracks deterministic accuracy (reusing `beacon_eval.scoring`), total & downstream
(Sessions 2-4) token usage, tool calls, and redundant file re-reads across sessions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from archolith_bench.beacon_eval.models import Gold, RepoPin
from archolith_bench.beacon_eval.scoring import extract_answer, score

SPRINT_ARMS: tuple[str, ...] = (
    "no_menhir",
    "menhir_structure",
    "menhir_longitudinal",
)


@dataclass(frozen=True)
class DriftPatch:
    """Deterministic file edit applied to the working checkout before a session."""

    path: str
    find: str
    replace: str
    description: str = ""


@dataclass(frozen=True)
class SprintSession:
    """One session in a chained multi-session coding sprint."""

    session_id: str
    phase: str  # investigate | impact_plan | drift_check | followup
    prompt: str
    gold: Gold
    memory_instruction: str = ""  # appended only in `menhir_longitudinal`
    drift_patch: DriftPatch | None = None  # applied before this session starts


@dataclass(frozen=True)
class SprintFixture:
    """Pinned external repo + ordered sequence of dependent coding sessions."""

    name: str
    description: str
    repo: RepoPin
    sessions: tuple[SprintSession, ...]

    @classmethod
    def from_file(cls, path: str | Path) -> "SprintFixture":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        repo = RepoPin(**raw["repo"])
        sessions: list[SprintSession] = []
        for s in raw["sessions"]:
            g = s.get("gold") or {}
            gold = Gold(
                docs=tuple(g.get("docs", ())),
                files=tuple(g.get("files", ())),
                acceptable_files=tuple(g.get("acceptable_files", ())),
                commands=tuple(g.get("commands", ())),
                guardrails=tuple(
                    tuple(x) if isinstance(x, list) else str(x)
                    for x in g.get("guardrails", ())
                ),
                verdict=str(g.get("verdict", "")),
                points=tuple(
                    tuple(x) if isinstance(x, list) else str(x)
                    for x in g.get("points", ())
                ),
                risky=tuple(g.get("risky", ())),
                allowed_flags=tuple(g.get("allowed_flags", ())),
            )
            dp_raw = s.get("drift_patch")
            drift_patch = DriftPatch(**dp_raw) if dp_raw else None
            sessions.append(
                SprintSession(
                    session_id=s["session_id"],
                    phase=s["phase"],
                    prompt=s["prompt"],
                    gold=gold,
                    memory_instruction=s.get("memory_instruction", ""),
                    drift_patch=drift_patch,
                )
            )
        return cls(
            name=raw["name"],
            description=raw.get("description", ""),
            repo=repo,
            sessions=tuple(sessions),
        )


@dataclass
class SessionExecution:
    """Raw output of executing one session under one arm."""

    response_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0
    files_read: list[str] = field(default_factory=list)
    memories_saved: int = 0
    todos_saved: int = 0
    seconds: float = 0.0
    error: str = ""
    repo_root: Path | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass
class SessionOutcome:
    """Scored outcome of one session in an arm."""

    session_id: str
    phase: str
    arm: str
    scores: dict[str, float]
    overall_score: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    turns: int
    tool_calls: int
    files_read: list[str]
    redundant_file_reads: int
    memories_saved: int
    todos_saved: int
    seconds: float
    error: str = ""


class SessionExecutor(Protocol):
    """Pluggable executor (DryRunExecutor for offline tests, OpenCode executor for live runs)."""

    def run_session(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        session_index: int,
        namespace: str,
    ) -> SessionExecution: ...


class DryRunExecutor:
    """Zero-LLM deterministic executor that validates the full sprint pipeline & scoring."""

    def run_session(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        session_index: int,
        namespace: str,
    ) -> SessionExecution:
        g = session.gold
        # Emit a valid fenced JSON answer matching ANSWER_KEYS for deterministic scoring validation.
        points_text = [p[0] if isinstance(p, tuple) else p for p in g.points]
        guardrails_text = [gr[0] if isinstance(gr, tuple) else gr for gr in g.guardrails]
        payload = {
            "docs": list(g.docs),
            "files": list(g.files),
            "commands": list(g.commands),
            "guardrails": guardrails_text,
            "verdict": g.verdict,
            "findings": points_text,
            "plan": points_text if session.phase in ("impact_plan", "followup") else [],
            "citations": [],
        }
        # In dry-run mode, simulate realistic structural/longitudinal token & file-read profiles
        # strictly for harness accounting verification (clearly marked `dry_run=True` in report).
        base_files = list(dict.fromkeys([*g.docs, *g.files]))
        if arm == "no_menhir":
            return SessionExecution(
                response_text=f"```json\n{json.dumps(payload, indent=2)}\n```",
                input_tokens=18_000 + 2_000 * session_index,
                output_tokens=1_200,
                turns=8,
                tool_calls=12,
                files_read=base_files + ["docs/changelog.md", "llm/default_plugins/openai_models.py"],
                seconds=0.01,
            )
        if arm == "menhir_structure":
            return SessionExecution(
                response_text=f"```json\n{json.dumps(payload, indent=2)}\n```",
                input_tokens=13_000 + 1_000 * session_index,
                output_tokens=1_100,
                turns=5,
                tool_calls=8,
                files_read=base_files,
                seconds=0.01,
            )
        # menhir_longitudinal: saves memories in S1, avoids re-reading anchored files in S2/S4
        files_read = base_files if session_index in (0, 2) else list(g.files[:1])
        return SessionExecution(
            response_text=f"```json\n{json.dumps(payload, indent=2)}\n```",
            input_tokens=14_500 if session_index == 0 else 8_500,
            output_tokens=1_100,
            turns=6 if session_index == 0 else 3,
            tool_calls=9 if session_index == 0 else 5,
            files_read=files_read,
            memories_saved=2 if session_index == 0 else (1 if session_index == 2 else 0),
            todos_saved=1 if session_index == 0 else 0,
            seconds=0.01,
        )


def build_session_prompt(session: SprintSession, arm: str) -> str:
    """Compose the exact session prompt for a given arm."""
    parts = [session.prompt.strip()]
    if arm == "menhir_structure":
        parts.append(
            "You have Menhir structural tools (`query_structure`, `blast_radius`, "
            "`affected_tests`) indexed for this repository. Prefer them over broad grep."
        )
    elif arm == "menhir_longitudinal":
        parts.append(
            "You have Menhir structural AND memory tools (`query_structure`, `blast_radius`, "
            "`affected_tests`, `recall_memories`, `add_memory`, `add_todo`, `list_todos`). "
            "Recall existing memories/TODOs at session start before re-reading files."
        )
        if session.memory_instruction:
            parts.append(session.memory_instruction.strip())
    return "\n\n".join(parts)


def run_longitudinal_sprint(
    fixture: SprintFixture,
    *,
    executor: SessionExecutor | None = None,
    arms: tuple[str, ...] = SPRINT_ARMS,
    repo_root: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Run all sessions in *fixture* across *arms* and aggregate accuracy + token metrics."""
    active_executor = executor or DryRunExecutor()
    arm_reports: dict[str, dict[str, Any]] = {}

    for arm in arms:
        namespace = f"sprint-{fixture.name}-{arm}"
        seen_files: set[str] = set()
        outcomes: list[SessionOutcome] = []

        for idx, session in enumerate(fixture.sessions):
            exec_res = active_executor.run_session(
                fixture=fixture,
                session=session,
                arm=arm,
                session_index=idx,
                namespace=namespace,
            )
            parsed = extract_answer(exec_res.response_text)
            if parsed is not None:
                scored = score(
                    parsed, session.gold, repo_root=exec_res.repo_root or repo_root
                )
                positive_vals = [
                    float(v) for k, v in scored.items() if k != "risky_false_positive"
                ]
                if "risky_false_positive" in scored:
                    positive_vals.append(1.0 - float(scored["risky_false_positive"]))
                overall = sum(positive_vals) / len(positive_vals) if positive_vals else 0.0
            else:
                scored = {}
                overall = 0.0

            # Count how many files read in this session were already read in earlier sessions.
            unique_this_session = list(dict.fromkeys(exec_res.files_read))
            redundant = sum(1 for f in unique_this_session if f in seen_files) if idx > 0 else 0
            seen_files.update(unique_this_session)

            outcomes.append(
                SessionOutcome(
                    session_id=session.session_id,
                    phase=session.phase,
                    arm=arm,
                    scores={k: round(float(v), 4) for k, v in scored.items()},
                    overall_score=round(overall, 4),
                    total_tokens=exec_res.total_tokens,
                    input_tokens=exec_res.input_tokens,
                    output_tokens=exec_res.output_tokens,
                    turns=exec_res.turns,
                    tool_calls=exec_res.tool_calls,
                    files_read=unique_this_session,
                    redundant_file_reads=redundant,
                    memories_saved=exec_res.memories_saved,
                    todos_saved=exec_res.todos_saved,
                    seconds=round(exec_res.seconds, 3),
                    error=exec_res.error,
                )
            )

        mean_accuracy = (
            round(sum(o.overall_score for o in outcomes) / len(outcomes), 4) if outcomes else 0.0
        )
        total_tokens = sum(o.total_tokens for o in outcomes)
        downstream_tokens = sum(o.total_tokens for o in outcomes[1:])
        total_turns = sum(o.turns for o in outcomes)
        downstream_turns = sum(o.turns for o in outcomes[1:])
        total_tool_calls = sum(o.tool_calls for o in outcomes)
        total_redundant_reads = sum(o.redundant_file_reads for o in outcomes)
        total_memories_saved = sum(o.memories_saved for o in outcomes)
        total_todos_saved = sum(o.todos_saved for o in outcomes)
        denom = max(mean_accuracy, 0.01)

        arm_reports[arm] = {
            "arm": arm,
            "namespace": namespace,
            "mean_accuracy": mean_accuracy,
            "total_tokens": total_tokens,
            "downstream_tokens_s2_to_end": downstream_tokens,
            "total_turns": total_turns,
            "downstream_turns_s2_to_end": downstream_turns,
            "total_tool_calls": total_tool_calls,
            "tokens_per_accuracy_point": round(total_tokens / denom, 1),
            "turns_per_accuracy_point": round(total_turns / denom, 2),
            "total_redundant_file_reads": total_redundant_reads,
            "total_memories_saved": total_memories_saved,
            "total_todos_saved": total_todos_saved,
            "sessions": [asdict(o) for o in outcomes],
        }

    baseline = arm_reports.get("no_menhir")
    comparisons: dict[str, dict[str, Any]] = {}
    if baseline and baseline["total_tokens"] > 0:
        base_turns = baseline["total_turns"] or 1
        base_tools = baseline["total_tool_calls"] or 1
        ds_base = baseline["downstream_tokens_s2_to_end"] or 1
        for arm, rep in arm_reports.items():
            if arm == "no_menhir":
                continue
            acc_delta = round(rep["mean_accuracy"] - baseline["mean_accuracy"], 4)
            tok_delta_pct = round(
                (rep["total_tokens"] - baseline["total_tokens"]) / baseline["total_tokens"] * 100.0,
                2,
            )
            ds_delta_pct = round(
                (rep["downstream_tokens_s2_to_end"] - ds_base) / ds_base * 100.0,
                2,
            )
            turns_delta_pct = round(
                (rep["total_turns"] - baseline["total_turns"]) / base_turns * 100.0,
                2,
            )
            tools_delta_pct = round(
                (rep["total_tool_calls"] - baseline["total_tool_calls"]) / base_tools * 100.0,
                2,
            )
            acc_ok = acc_delta >= -0.02
            eff_improved = (tok_delta_pct <= -10.0) or (turns_delta_pct <= -10.0)
            if not acc_ok:
                verdict = "TRADEOFF_ACCURACY_DROP"
            elif acc_delta >= 0.02 and eff_improved:
                verdict = "WIN_ACCURACY_AND_EFFICIENCY"
            elif eff_improved:
                verdict = "WIN_EFFICIENCY_AT_PARITY"
            elif acc_delta >= 0.02:
                verdict = "WIN_ACCURACY_ONLY"
            else:
                verdict = "NEUTRAL_OR_REGRESSION"

            comparisons[arm] = {
                "accuracy_delta": acc_delta,
                "accuracy_non_regression": acc_ok,
                "total_tokens_delta_pct": tok_delta_pct,
                "downstream_tokens_delta_pct": ds_delta_pct,
                "total_turns_delta_pct": turns_delta_pct,
                "total_tool_calls_delta_pct": tools_delta_pct,
                "redundant_reads_delta": rep["total_redundant_file_reads"]
                - baseline["total_redundant_file_reads"],
                "verdict": verdict,
            }

    return {
        "suite": "engineering_evidence_longitudinal_sprint",
        "fixture": fixture.name,
        "dry_run": dry_run,
        "repo": asdict(fixture.repo),
        "arms": arm_reports,
        "comparisons_vs_no_menhir": comparisons,
    }
