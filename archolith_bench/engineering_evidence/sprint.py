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

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from archolith_bench.beacon_eval.models import Gold, RepoPin
from archolith_bench.beacon_eval.runner import ANSWER_INSTRUCTIONS
from archolith_bench.beacon_eval.scoring import _clean_path, extract_answer, score

SPRINT_ARMS: tuple[str, ...] = (
    "no_menhir",
    "menhir_structure",
    "menhir_longitudinal",
)

#: Arms of the v2 rigorous campaign (rubric `sprint-v2`): the no-memory floor, the primary
#: notes-file baseline and Menhir. `menhir_structure` is not part of v2.
V2_ARMS: tuple[str, ...] = (
    "no_menhir",
    "notes_file",
    "menhir_longitudinal",
)

#: The one file the `notes_file` arm may create or edit (at the checkout root).
NOTES_FILE = "SPRINT_NOTES.md"

#: Appended after the answer instructions (which say "read-only") for the `notes_file` arm.
NOTES_EXCEPTION = (
    f"The one exception to the read-only rule is `{NOTES_FILE}` at the repository root: you "
    "may create and edit it. It is the only memory you keep between sessions. Do not list "
    f"`{NOTES_FILE}` in your JSON answer."
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
    notes_instruction: str = ""  # appended only in `notes_file` (parallels memory_instruction)
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
                    notes_instruction=s.get("notes_instruction", ""),
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
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    censored: bool = False
    notes_violations: list[str] = field(default_factory=list)
    #: OpenCode reads auto-rejected as outside the checkout (each ends the run).
    refused_reads: int = 0
    enrichment_wait_s: float = 0.0
    enrichment_drained: bool = True
    run_dir: Path | None = None

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def uncached_tokens(self) -> int:
        """input + output + reasoning + cache_write (everything except cache reads)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
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
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_tokens: int = 0
    cost_usd: float = 0.0
    censored: bool = False
    not_run: bool = False
    not_run_reason: str = ""
    notes_violations: list[str] = field(default_factory=list)
    #: OpenCode reads auto-rejected as outside the checkout (each ends the run).
    refused_reads: int = 0
    enrichment_wait_s: float = 0.0
    enrichment_drained: bool = True
    repeat: int | None = None


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
        repeat: int | None = None,
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
        repeat: int | None = None,
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
        jitter = 37 * (repeat or 0)  # 0 on the old single-run path
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
                input_tokens=13_000 + 1_000 * session_index + jitter,
                output_tokens=1_100,
                turns=5,
                tool_calls=8,
                files_read=base_files,
                seconds=0.01,
            )
        if arm == "notes_file":
            return SessionExecution(
                response_text=f"```json\n{json.dumps(payload, indent=2)}\n```",
                input_tokens=(15_500 if session_index == 0 else 10_500) + jitter,
                output_tokens=1_150,
                turns=6 if session_index == 0 else 4,
                tool_calls=10 if session_index == 0 else 6,
                files_read=(base_files if session_index in (0, 2) else list(g.files[:1]))
                + [NOTES_FILE],
                seconds=0.01,
            )
        # menhir_longitudinal: saves memories in S1, avoids re-reading anchored files in S2/S4
        files_read = base_files if session_index in (0, 2) else list(g.files[:1])
        return SessionExecution(
            response_text=f"```json\n{json.dumps(payload, indent=2)}\n```",
            input_tokens=(14_500 if session_index == 0 else 8_500) + jitter,
            output_tokens=1_100,
            turns=6 if session_index == 0 else 3,
            tool_calls=9 if session_index == 0 else 5,
            files_read=files_read,
            memories_saved=2 if session_index == 0 else (1 if session_index == 2 else 0),
            todos_saved=1 if session_index == 0 else 0,
            seconds=0.01,
        )


#: Version of the arm notes and fixture instructions (verdict rules are RUBRIC_VERSION, separate).
#: sprint-v2.1: the memory arm is told to save as the notes arm is, with real lean tool names and
#: the index project; runs of different prompt sets are never compared or resumed together.
PROMPT_SET = "sprint-v2.1"

#: Each arm's tool/memory note; the same text for every session of the arm. Tool names are the
#: Menhir lean profile's; blast radius and affected tests are `query_structure` query types, not
#: tools. The memory arm gets the same "only memory, read at start, keep it current" rule as
#: the notes arm (the sprint-v2 prompts told it to recall only, so it saved nothing after session 1).
_ARM_NOTES: dict[str, str] = {
    "menhir_structure": (
        "You have the Menhir structural tool `query_structure`, indexed for this repository: "
        "query_type `blast_radius` or `affected_tests` (path = comma-separated changed files) for "
        "impact, `context` for one file. Prefer it over broad grep."
    ),
    "menhir_longitudinal": (
        "You have Menhir memory and structural tools. Memory: `recall_memories`, "
        "`recall_context_memories`, `read_flagged_memories`, `add_memory`, and `get_provenance` "
        "for a memory's sources. TODOs: `list_todos`, `get_todo`, `add_todo`, `close_todo`. "
        "Structure: `query_structure` with query_type `blast_radius` or `affected_tests` "
        "(path = comma-separated changed files) for impact, `context` for one file. Menhir is the "
        "only memory you have between sessions: recall memories and TODOs at session start before "
        "re-reading files, and before finishing store what a later session would need with "
        "`add_memory` (follow-ups with `add_todo`; close finished ones with `close_todo`). If a "
        "recalled memory is wrong or stale, store a corrected one with `add_memory` that says what "
        "it replaces."
    ),
    "notes_file": (
        f"Keep your working notes in `{NOTES_FILE}` at the repository root: it is the only "
        "memory you have between sessions, so read it at session start before re-reading files."
    ),
}


def build_session_prompt(session: SprintSession, arm: str) -> str:
    """Compose the exact session prompt for a given arm."""
    parts = [session.prompt.strip()]
    if arm in _ARM_NOTES:
        parts.append(_ARM_NOTES[arm])
    if arm == "menhir_longitudinal" and session.memory_instruction:
        parts.append(session.memory_instruction.strip())
    elif arm == "notes_file" and session.notes_instruction:
        parts.append(session.notes_instruction.strip())
    return "\n\n".join(parts)


def _project_line(arm: str, project: str) -> str:
    if arm in MENHIR_PROMPT_ARMS and project:
        return f"The Menhir structural index for this repository is project `{project}`."
    return ""


#: Arms whose prompt names Menhir tools.
MENHIR_PROMPT_ARMS = frozenset({"menhir_structure", "menhir_longitudinal"})


def compose_full_prompt(session: SprintSession, arm: str, namespace: str,
                        cache_shaping: bool = False, *, project: str = "") -> str:
    """The exact prompt a live executor sends: session prompt, namespace line, answer rules.

    *project* (the structural index's project, the fixture repo name) is named for the Menhir
    arms so the agent does not have to discover or guess it.

    With *cache_shaping* the same blocks are reordered so text shared by every session of the arm
    (tool note, answer rules, notes exception) comes first and session text last, extending the
    prompt prefix the provider can serve from cache. The words are unchanged.
    """
    project_line = _project_line(arm, project)
    if cache_shaping:
        stable = [_ARM_NOTES[arm]] if arm in _ARM_NOTES else []
        stable.append(ANSWER_INSTRUCTIONS)
        if arm == "notes_file":
            stable.append(NOTES_EXCEPTION)
        if arm == "menhir_longitudinal":
            stable.append(f"Use Menhir namespace `{namespace}` for all memory/TODO operations.")
        if project_line:
            stable.append(project_line)
        varying = [session.prompt.strip()]
        if arm == "menhir_longitudinal" and session.memory_instruction:
            varying.append(session.memory_instruction.strip())
        if arm == "notes_file" and session.notes_instruction:
            varying.append(session.notes_instruction.strip())
        return "\n\n".join(stable + varying)
    text = build_session_prompt(session, arm)
    if arm == "menhir_longitudinal":
        text += f"\n\nUse Menhir namespace `{namespace}` for all memory/TODO operations."
    if project_line:
        text += "\n\n" + project_line
    text += "\n\n" + ANSWER_INSTRUCTIONS
    if arm == "notes_file":
        text += "\n\n" + NOTES_EXCEPTION
    return text


def is_notes_path(value: Any) -> bool:
    """True when *value* names the notes file (at any depth, after path cleanup)."""
    if not isinstance(value, str):
        return False
    return _clean_path(value).rsplit("/", 1)[-1] == NOTES_FILE


def strip_notes_from_answer(answer: dict[str, Any] | None) -> dict[str, Any] | None:
    """Copy of *answer* without the notes file in `docs`, `files` or `citations`."""
    if answer is None:
        return None
    out = dict(answer)
    for key in ("docs", "files"):
        if isinstance(out.get(key), list):
            out[key] = [v for v in out[key] if not is_notes_path(v)]
    if isinstance(out.get("citations"), list):
        out["citations"] = [
            c for c in out["citations"] if not (isinstance(c, dict) and is_notes_path(c.get("path")))
        ]
    return out


def snapshot_tree(root: Path, exclude: tuple[str, ...] = (NOTES_FILE,)) -> dict[str, str]:
    """sha256 of every file under *root* (skipping `.git` and *exclude* at the root)."""
    snap: dict[str, str] = {}
    base = Path(root)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(base).as_posix()
            if rel in exclude:
                continue
            try:
                snap[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
            except OSError:
                snap[rel] = "unreadable"
    return snap


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """`added:`, `deleted:` and `modified:` entries between two :func:`snapshot_tree` results."""
    out = [f"added:{p}" for p in sorted(after.keys() - before.keys())]
    out += [f"deleted:{p}" for p in sorted(before.keys() - after.keys())]
    out += [f"modified:{p}" for p in sorted(p for p in before.keys() & after.keys() if before[p] != after[p])]
    return out


def overall_from_scores(scored: dict[str, float]) -> float:
    positive_vals = [float(v) for k, v in scored.items() if k != "risky_false_positive"]
    if "risky_false_positive" in scored:
        positive_vals.append(1.0 - float(scored["risky_false_positive"]))
    return sum(positive_vals) / len(positive_vals) if positive_vals else 0.0


def score_execution(
    exec_res: SessionExecution, session: SprintSession, arm: str, repo_root: Path | None = None
) -> tuple[dict[str, float], float]:
    """Deterministic score of one execution; the notes file never counts for or against."""
    parsed = extract_answer(exec_res.response_text)
    if arm == "notes_file":
        parsed = strip_notes_from_answer(parsed)
    if parsed is None:
        return {}, 0.0
    scored = score(parsed, session.gold, repo_root=exec_res.repo_root or repo_root)
    return scored, overall_from_scores(scored)


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
            scored, overall = score_execution(exec_res, session, arm, repo_root)

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
                    reasoning_tokens=exec_res.reasoning_tokens,
                    cache_read_tokens=exec_res.cache_read_tokens,
                    cache_write_tokens=exec_res.cache_write_tokens,
                    uncached_tokens=exec_res.uncached_tokens,
                    cost_usd=exec_res.cost_usd,
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
