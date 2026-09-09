"""Compute per-arm scores from a run's harness checkpoint, as machine-readable evidence.

The buildout harness writes ``harness_recall/results.md`` -- rendered markdown. That is why
every score in ``LEDGER.md`` had to be read by a human and retyped, and why six runs with
real scores never reached the scoreboard at all. This reads the authoritative source
instead: ``harness_recall/.checkpoint_*.jsonl``, one line per (arm, task) carrying
``result.correct``.

Verified against the hand-typed history: it reproduces 0.467, 0.346, 0.333 and 0.872
exactly for the four runs whose ledger rows record a score and whose checkpoint survives.

**It never picks a primary arm.** A run carries up to six (``no_memory``,
``menhir_recall``, ``menhir_value_recall``, the v2/v3 variants), and which one a ledger row
quotes is a judgement: ``value-arm-verify-20260717`` records 0.679, its
``menhir_value_recall`` arm, while its ``menhir_recall`` arm scored 0.333. Auto-picking by
name order would have silently published the wrong number. So every arm is emitted and the
ledger row must declare which one it quotes, the same way ``score``/``score_raw`` refuses to
collapse a multi-arm result into one float.

Verbs::

    score_extract.py <run-dir> [--json OUT] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_GLOB = ".checkpoint_*.jsonl"
HARNESS_SUBDIR = "harness_recall"
OUTPUT_NAME = "score.json"


class ScoreError(RuntimeError):
    """A score cannot be computed honestly from this run directory."""


def find_checkpoints(run_directory: Path) -> list[Path]:
    """Checkpoints for a run, searching the harness output dir then the run root."""
    candidates = sorted((run_directory / HARNESS_SUBDIR).glob(CHECKPOINT_GLOB))
    if candidates:
        return candidates
    return sorted(run_directory.glob(CHECKPOINT_GLOB))


def extract(run_directory: Path) -> dict[str, Any]:
    """Per-arm scores for one run directory.

    Raises rather than returning zeros when there is no checkpoint: a run that never scored
    and a run that scored 0.0 are different facts, and conflating them is how an aborted
    launch ends up published as a result.
    """
    checkpoints = find_checkpoints(run_directory)
    if not checkpoints:
        raise ScoreError(
            f"no {CHECKPOINT_GLOB} under {run_directory} -- this run has no score to read "
            "(it may have aborted before the harness ran)"
        )

    # (arm, task_id) -> record. Last wins: --resume appends, so a re-scored task appears
    # again later in the file and the newer verdict is the one that counts. No existing
    # checkpoint actually contains a repeat, but resume makes it possible, and silently
    # double-counting one would move the score.
    verdicts: dict[tuple[str, str], bool] = {}
    duplicates = 0
    malformed = 0
    tokens: dict[str, dict[str, int]] = {}

    for checkpoint in checkpoints:
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            result = row.get("result")
            if not isinstance(result, dict) or "correct" not in result:
                continue
            arm = str(row.get("arm") or "?")
            task = str(result.get("task_id") or row.get("task_id") or "")
            if not task:
                malformed += 1
                continue
            key = (arm, task)
            if key in verdicts:
                duplicates += 1
            verdicts[key] = bool(result["correct"])
            bucket = tokens.setdefault(arm, {"input_tokens": 0, "output_tokens": 0})
            for field in ("input_tokens", "output_tokens"):
                value = result.get(field)
                if isinstance(value, int):
                    bucket[field] += value

    if not verdicts:
        raise ScoreError(
            f"checkpoint(s) under {run_directory} contain no scored results "
            f"({malformed} unparseable line(s))"
        )

    arms: dict[str, Any] = {}
    for (arm, _task), correct in verdicts.items():
        entry = arms.setdefault(arm, {"n": 0, "correct": 0})
        entry["n"] += 1
        entry["correct"] += int(correct)
    for arm, entry in arms.items():
        entry["score"] = round(entry["correct"] / entry["n"], 6)
        entry.update(tokens.get(arm, {}))

    return {
        "run_id": run_directory.name,
        "source": [str(path.relative_to(run_directory).as_posix()) for path in checkpoints],
        # Declared, not assumed: a reader that wants "the" score must choose an arm.
        "primary_arm": None,
        "arms": dict(sorted(arms.items())),
        "duplicate_entries_collapsed": duplicates,
        "malformed_lines": malformed,
    }


def write(run_directory: Path, payload: dict[str, Any]) -> Path:
    target = run_directory / OUTPUT_NAME
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def format_report(payload: dict[str, Any]) -> str:
    lines = [f"{payload['run_id']}  (from {', '.join(payload['source'])})"]
    width = max((len(a) for a in payload["arms"]), default=4)
    for arm, entry in payload["arms"].items():
        lines.append(
            f"  {arm:<{width}}  n={entry['n']:<4} correct={entry['correct']:<4} "
            f"score={entry['score']:.3f}"
        )
    if payload["duplicate_entries_collapsed"]:
        lines.append(
            f"  note: {payload['duplicate_entries_collapsed']} duplicate (arm, task) "
            "entr(ies) collapsed, last verdict kept"
        )
    if payload["malformed_lines"]:
        lines.append(f"  note: {payload['malformed_lines']} unparseable line(s) skipped")
    lines.append("  primary_arm is null -- the ledger row must declare which arm it quotes")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--json", type=Path, default=None,
        help=f"where to write the payload (default: <run-dir>/{OUTPUT_NAME})",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = extract(args.run_dir)
    target = args.json or (args.run_dir / OUTPUT_NAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(format_report(payload))
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScoreError as exc:
        print(f"score_extract: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
