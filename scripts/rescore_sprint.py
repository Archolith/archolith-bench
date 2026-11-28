"""Rescore a saved sprint v2 run from its saved answers; no model calls, never overwrites.

The sprint counterpart of `beacon_eval.runner.rescore`: the saved answer text comes from each
session's `events.jsonl[.gz]` (parsed with the same `EventLog`), the checkout is rebuilt from the
fixture pin with `export_commit` (the call `rebuild_checkout` makes) and the fixture's drift
patches are applied cumulatively per chain, then `score_execution` scores it. Results go to a
NEW directory; the source run directory is only read, and an existing target is refused.

    python scripts/rescore_sprint.py --run-dir results/sprint-v2/<fixture>-<run-id> \
        --fixture fixtures/sprint_simonw_llm_764dc386.json --out-dir results/sprint-v2/<fixture>-<run-id>-rescore1
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from archolith_bench.beacon_eval.runner import EventLog, export_commit  # noqa: E402
from archolith_bench.engineering_evidence.live_executor import apply_drift_patch  # noqa: E402
from archolith_bench.engineering_evidence.sprint import (  # noqa: E402
    SessionExecution,
    SessionOutcome,
    SprintFixture,
    score_execution,
)
from archolith_bench.engineering_evidence.sprint_rubric import ChainView, build_verdict_report  # noqa: E402
from archolith_bench.staged.rundir import StageFile  # noqa: E402


class RescoreTargetExists(RuntimeError):
    """The rescore output directory already exists; nothing is written."""


def saved_answer_text(session_dir: Path) -> str | None:
    """The agent's reply text from a session's saved events, or None when none were kept."""
    gz, plain = session_dir / "events.jsonl.gz", session_dir / "events.jsonl"
    if gz.is_file():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    elif plain.is_file():
        lines = plain.read_text(encoding="utf-8").splitlines()
    else:
        return None
    log = EventLog()
    for line in lines:
        log.feed(line)
    return "\n".join(log.texts)


def latest_chains(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Rows of the latest attempt of every (arm, repeat), in session order."""
    best: dict[tuple[str, int], int] = {}
    for r in rows:
        key = (r["arm"], int(r["repeat"]))
        best[key] = max(best.get(key, 0), int(r.get("attempt", 1)))
    chains: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["arm"], int(r["repeat"]))
        if int(r.get("attempt", 1)) == best[key]:
            chains.setdefault(key, []).append(r)
    return chains


def rescore_run(run_dir: Path, fixture: SprintFixture, cache_dir: Path, out_dir: Path) -> dict[str, Any]:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    if out_dir.exists():
        raise RescoreTargetExists(f"{out_dir} already exists; rescore never overwrites")
    rows = StageFile(run_dir / "sessions.jsonl").records()
    if not rows:
        raise FileNotFoundError(f"{run_dir / 'sessions.jsonl'} has no saved sessions")
    original = run_dir / "report.json"
    server_by_chain: dict[tuple[str, int], dict[str, Any]] = {}
    if original.is_file():
        for c in json.loads(original.read_text(encoding="utf-8")).get("chains", []):
            server_by_chain[(c["arm"], int(c["repeat"]))] = c.get("server", {})

    out_dir.mkdir(parents=True, exist_ok=False)
    names = {s.session_id: i for i, s in enumerate(fixture.sessions)}
    scratch = Path(tempfile.mkdtemp(prefix="rescore-sprint-"))
    views: list[ChainView] = []
    saved_rows: list[dict[str, Any]] = []
    try:
        for (arm, repeat), chain_rows in sorted(latest_chains(rows).items()):
            checkout = export_commit(fixture.repo, scratch / f"{arm}-r{repeat}", cache_dir)
            by_session = {r["session_id"]: r for r in chain_rows}
            outcomes: list[SessionOutcome] = []
            for idx, session in enumerate(fixture.sessions):
                if session.drift_patch is not None:
                    apply_drift_patch(checkout, session.drift_patch)
                row = by_session.get(session.session_id)
                if row is None:
                    continue
                keep = {k: v for k, v in row.items() if k in SessionOutcome.__dataclass_fields__}
                outcome = SessionOutcome(**keep)
                if not outcome.not_run and not outcome.censored:
                    sdir = run_dir / "runs" / arm / f"r{repeat}" / f"{idx:02d}-{session.session_id}"
                    text = saved_answer_text(sdir)
                    if text is None:
                        raise FileNotFoundError(f"no saved events for {arm}/r{repeat}/{session.session_id}")
                    scored, overall = score_execution(
                        SessionExecution(response_text=text, repo_root=checkout), session, arm, checkout
                    )
                    outcome = replace(
                        outcome,
                        scores={k: round(float(v), 4) for k, v in scored.items()},
                        overall_score=round(overall, 4),
                    )
                outcomes.append(outcome)
                saved_rows.append(
                    {**asdict(outcome), "arm": arm, "repeat": repeat,
                     "old_overall_score": row.get("overall_score")}
                )
            views.append(ChainView(arm, repeat, tuple(outcomes), server_by_chain.get((arm, repeat), {})))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    report = {
        "suite": "engineering_evidence_longitudinal_sprint_v2_rescore",
        "source_run_dir": str(run_dir),
        "fixture": fixture.name,
        "analysis": build_verdict_report(views, expected_sessions=len(fixture.sessions)),
    }
    (out_dir / "rescore_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "sessions.rescored.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in saved_rows), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rescore a saved sprint v2 run (no model calls).")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "results" / "repo-cache"))
    parser.add_argument("--out-dir", required=True, help="A NEW directory; an existing one is refused.")
    args = parser.parse_args(argv)
    try:
        report = rescore_run(
            Path(args.run_dir), SprintFixture.from_file(args.fixture), Path(args.cache_dir), Path(args.out_dir)
        )
    except RescoreTargetExists as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for label, v in report["analysis"]["verdicts"].items():
        print(f"{label}: {v['combined_verdict']}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
