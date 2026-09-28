"""Machine-readable scoreboard for LongMemEval buildouts, with on-disk validation.

``LEDGER.md`` carries real analysis -- regression breakdowns, per-item tables, findings --
and that prose should stay prose. Its *scoreboard* is different: it is tabular data that
was being hand-typed, which means nothing could join a score delta to a provenance delta
without a human reading both. So the scoreboard moves to ``ledger.csv`` and the markdown
table is generated from it. One source, no drift.

The schema refuses to launder messy reality into clean numbers:

* ``score`` is empty unless the run produced exactly one usable score. A row reporting two
  arms (``v2c 0.667 / v2h 0.679``) keeps both in ``score_raw`` and leaves ``score`` empty,
  because any single float there would be a fabrication.
* ``status`` is a closed vocabulary, so "no score" is never ambiguous between *unscored*,
  *killed*, *invalid*, and *scored zero*.
* ``score_raw`` preserves what the ledger actually said, so generation is lossless even
  where parsing is partial.

Verbs::

    ledger.py render   [--csv P] [--markdown P]   # CSV -> the table in LEDGER.md
    ledger.py validate [--csv P] [--results D]    # rows vs. what is on disk
    ledger.py join A B [--csv P] [--results D]    # score delta + surface blame
    ledger.py import-markdown <LEDGER.md> --csv P # one-time migration
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LME_DIR = SCRIPT_DIR.parent
BENCH_ROOT = LME_DIR.parents[1]

DEFAULT_RESULTS = BENCH_ROOT / "results" / "lme-ku-buildout"
DEFAULT_CSV = DEFAULT_RESULTS / "ledger.csv"
DEFAULT_MARKDOWN = DEFAULT_RESULTS / "LEDGER.md"
DEFAULT_EXCLUDED = DEFAULT_RESULTS / "ledger-excluded.txt"

BEGIN_MARKER = "<!-- BEGIN GENERATED SCOREBOARD -- edit ledger.csv, then run ledger.py render -->"
END_MARKER = "<!-- END GENERATED SCOREBOARD -->"

FIELDS = (
    "run_id",
    "date",
    "items_scored",
    "items_total",
    "segmentation",
    "score",
    "score_raw",
    # Which arm the `score` column quotes. A run carries up to six
    # (no_memory / menhir_recall / menhir_value_recall / the v2-v3 variants), and the choice
    # is a judgement, not something a reader can derive: value-arm-verify-20260717 records
    # 0.679, its menhir_value_recall arm, while its menhir_recall arm scored 0.333. Declaring
    # it is what lets `validate` check the recorded number against the run's own score.json.
    "primary_arm",
    "status",
    "extract_model",
    "canonical",
    "has_results_dir",
    "notes",
)

# A closed vocabulary. The point is that an empty `score` always has a stated reason.
STATUSES = {
    "scored": "produced one usable score",
    "multi_arm": "several arm scores in one row; see score_raw",
    "killed": "deliberately stopped mid-run",
    "aborted": "failed before producing a result",
    "invalid": "produced data that is not trustworthy; do not compare",
    "partial": "stopped partway, never scored",
    "measure_only": "materialization measured, no QA scoring by design",
    "offline": "analysis only, no paid run",
}

# Exactly one row may claim to be the current canonical benchmark evidence.
CANONICAL_VALUES = {"", "superseded", "current"}


class LedgerError(RuntimeError):
    """The ledger cannot be read or rendered honestly."""


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def read_rows(csv_path: Path = DEFAULT_CSV) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise LedgerError(f"ledger csv not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [field for field in FIELDS if rows and field not in rows[0]]
    if missing:
        raise LedgerError(f"ledger csv missing columns: {', '.join(missing)}")
    return rows


def write_rows(rows: list[dict[str, str]], csv_path: Path = DEFAULT_CSV) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})



def read_exclusions(path: Path | None = None) -> dict[str, str]:
    """Runs deliberately kept out of the scoreboard, as {run_id: reason}.

    A reason is required. An entry without one is refused rather than defaulted, because
    "excluded" with no stated ground is how a real result gets quietly hidden -- the exact
    failure the orphan check exists to catch. Exclusions suppress the per-run orphan finding
    only; the count is still reported, so they never become fully invisible.
    """
    target = DEFAULT_EXCLUDED if path is None else path
    if not target.exists():
        return {}
    excluded: dict[str, str] = {}
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        run_id, _, reason = stripped.partition("  ")
        if not reason.strip():
            raise LedgerError(
                f"{target.name}:{number}: {run_id!r} is excluded with no reason; "
                "every exclusion must state why (separate run_id and reason with two spaces)"
            )
        excluded[run_id.strip()] = reason.strip()
    return excluded

# ---------------------------------------------------------------------------
# One-time migration from the hand-typed markdown table
# ---------------------------------------------------------------------------

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*\((.+?)\)\*")
_SCORE = re.compile(r"\b(0\.\d+|1\.000|1\.0)\b")


def _strip_markup(text: str) -> str:
    text = _BOLD.sub(r"\1", text)
    text = text.replace("`", "")
    return text.strip()


def _parse_n(raw: str) -> tuple[str, str]:
    """"7/78" -> (7, 78); "15" -> (15, 15); anything else -> ("", "")."""
    raw = _strip_markup(raw)
    if "/" in raw:
        left, _, right = raw.partition("/")
        if left.strip().isdigit() and right.strip().isdigit():
            return left.strip(), right.strip()
        return "", ""
    if raw.isdigit():
        return raw, raw
    return "", ""


def _classify_score(raw: str) -> tuple[str, str]:
    """Return (score, status) from a scoreboard Score cell, never inventing a number."""
    cleaned = _strip_markup(raw)
    lowered = cleaned.lower()

    if "invalid" in lowered:
        return "", "invalid"
    if "killed" in lowered:
        return "", "killed"
    if "aborted" in lowered or "launch refused" in lowered:
        return "", "aborted"
    if "stopped partial" in lowered:
        return "", "partial"
    if "measure-only" in lowered:
        return "", "measure_only"
    if "no paid run" in lowered:
        return "", "offline"

    found = _SCORE.findall(cleaned)
    if len(found) == 1:
        return found[0], "scored"
    if len(found) > 1:
        # Two or more arm scores in one cell. Collapsing them to one float would assert a
        # result the run never produced.
        return "", "multi_arm"
    return "", "aborted"


def import_markdown(markdown_path: Path) -> list[dict[str, str]]:
    """Parse the existing hand-typed scoreboard into rows. Migration only."""
    text = markdown_path.read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Run ID"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if len(cells) != 7:
                continue
            run_id_raw, date, n_raw, segmentation, score_raw, model, notes = cells
            run_id = _strip_markup(run_id_raw)
            scored, total = _parse_n(n_raw)
            score, status = _classify_score(score_raw)
            notes_clean = _strip_markup(notes)
            canonical = ""
            if "CURRENT CANONICAL" in notes:
                canonical = "current"
            elif "CANONICAL" in notes:
                canonical = "superseded"
            rows.append({
                "run_id": run_id,
                "date": date,
                "items_scored": scored,
                "items_total": total,
                "segmentation": _strip_markup(segmentation),
                "score": score,
                "score_raw": _strip_markup(score_raw),
                "status": status,
                "extract_model": _strip_markup(model),
                "canonical": canonical,
                "has_results_dir": "",
                "notes": notes_clean,
            })
    if not rows:
        raise LedgerError(f"no scoreboard table found in {markdown_path}")
    return rows


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| Run ID | Date | N | Segmentation | Score | Status | Extract Model | Notes |",
        "|--------|------|---|--------------|-------|--------|---------------|-------|",
    ]
    for row in rows:
        scored, total = row.get("items_scored", ""), row.get("items_total", "")
        if scored and total:
            n = scored if scored == total else f"{scored}/{total}"
        else:
            n = "?"
        score = row.get("score") or row.get("score_raw") or "--"
        if row.get("canonical") == "current":
            score = f"**{score}**"
        run_id = row.get("run_id", "")
        if row.get("canonical") == "current":
            run_id = f"**{run_id}**"
        lines.append(
            f"| {run_id} | {row.get('date','')} | {n} | {row.get('segmentation','')} "
            f"| {score} | {row.get('status','')} | {row.get('extract_model','')} "
            f"| {row.get('notes','')} |"
        )
    return "\n".join(lines)


def render(
    csv_path: Path = DEFAULT_CSV, markdown_path: Path = DEFAULT_MARKDOWN
) -> str:
    """Replace the generated block in LEDGER.md with the table built from the CSV.

    Only the region between the markers is touched; every prose section is left exactly as
    written. If the markers are absent the caller is told rather than guessed at, because
    overwriting a hand-maintained table is how analysis gets silently destroyed.
    """
    rows = read_rows(csv_path)
    table = render_table(rows)
    block = f"{BEGIN_MARKER}\n\n{table}\n\n{END_MARKER}"

    text = markdown_path.read_text(encoding="utf-8")
    if BEGIN_MARKER in text and END_MARKER in text:
        head, _, rest = text.partition(BEGIN_MARKER)
        _, _, tail = rest.partition(END_MARKER)
        updated = head + block + tail
    else:
        raise LedgerError(
            f"generated-scoreboard markers not found in {markdown_path}; "
            "add them around the scoreboard table first (see ledger.py docstring)"
        )
    markdown_path.write_text(updated, encoding="utf-8")
    return block


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def _read_provenance(path: Path) -> dict[str, Any] | None:
    try:
        # strict=False: some historical provenance carries raw control characters in
        # free-text purpose fields.
        return json.loads(path.read_text(encoding="utf-8"), strict=False)
    except (OSError, json.JSONDecodeError):
        return None


# Files that can actually carry a score. `manifest.json` and `harness_recall/` deliberately
# are not here: they prove ingest or recall ran, not that a number was produced. Treating
# them as score evidence over-flags build-only and diagnostic runs.
SCORED_RESULT_FILES = ("results.json", "comparison.json")


def describe_score(run_directory: Path) -> str | None:
    """A short description of whatever score evidence a run directory holds, or None.

    Prefers ``score.json`` (written by ``score_extract.py`` from the harness checkpoint),
    which carries every arm. All arms are listed rather than reduced to one number, because
    picking "the" score is the judgement the ledger's ``primary_arm`` column exists to
    record -- a summary here that chose for the reader would be the same mistake.
    """
    score_path = run_directory / "score.json"
    if score_path.exists():
        try:
            payload = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "score.json present but unreadable"
        arms = payload.get("arms") or {}
        if arms:
            rendered = ", ".join(
                f"{arm}={entry.get('score')}" for arm, entry in sorted(arms.items())
            )
            return f"{len(arms)} arm(s): {rendered}"
    legacy = extract_score(run_directory)
    return None if legacy is None else f"{legacy:g}"


def extract_score(run_directory: Path) -> float | None:
    """Read a numeric score from a run directory, or None if none is recorded.

    Shallow search for a score-shaped key, because the harnesses that write these files do
    not agree on one schema. Returns None rather than guessing; a run whose score cannot be
    read is reported as unreadable, not as zero.
    """
    def dig(node: Any, depth: int = 0) -> float | None:
        if depth > 3 or not isinstance(node, dict):
            return None
        for key, value in node.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and any(
                token in key.lower() for token in ("score", "accuracy", "correct", "rate")
            ):
                return float(value)
            if isinstance(value, dict):
                found = dig(value, depth + 1)
                if found is not None:
                    return found
        return None

    for name in SCORED_RESULT_FILES:
        path = run_directory / name
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"), strict=False)
        except (OSError, json.JSONDecodeError):
            continue
        found = dig(document)
        if found is not None:
            return found
    return None


def validate(
    csv_path: Path = DEFAULT_CSV, results_dir: Path = DEFAULT_RESULTS
) -> list[dict[str, str]]:
    """Check each row against the closed vocabulary and against what is on disk.

    Returns findings, each with a level: ``FAIL`` for a row that contradicts itself or the
    filesystem, ``WARN`` for a gap that is real but expected of historical runs.
    """
    rows = read_rows(csv_path)
    findings: list[dict[str, str]] = []
    excluded = read_exclusions(results_dir / "ledger-excluded.txt")

    def report(level: str, run_id: str, message: str) -> None:
        findings.append({"level": level, "run_id": run_id, "message": message})

    # A fresh checkout carries some committed score.json summaries but not the full run
    # directories. A directory containing only that summary is not materialized run evidence.
    # An absent evidence tree is "cannot check", not "the claim is false"; schema and
    # self-consistency checks still run.
    evidence_present = results_dir.is_dir() and any(
        child.is_dir() and not (
            (child / "score.json").is_file()
            and len(list(child.iterdir())) == 1
        ) for child in results_dir.iterdir()
    )
    if not evidence_present:
        report(
            "WARN", "(ledger)",
            f"no full run directories under {results_dir} -- only score summaries may be "
            "checked in, so the "
            "on-disk cross-checks (results dir, provenance, score.json, orphan runs) are "
            "skipped here; schema checks still ran",
        )

    seen: set[str] = set()
    current_canonical: list[str] = []

    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id:
            report("FAIL", "(blank)", "row has no run_id")
            continue
        if run_id in seen:
            report("FAIL", run_id, "duplicate run_id")
        seen.add(run_id)

        status = row.get("status", "")
        if status not in STATUSES:
            report("FAIL", run_id, f"status {status!r} is not in the closed vocabulary")

        score = (row.get("score") or "").strip()
        if status == "scored" and not score:
            report("FAIL", run_id, "status is 'scored' but no score is recorded")
        if status != "scored" and score:
            report(
                "FAIL", run_id,
                f"status is {status!r} but a score ({score}) is recorded; "
                "a score implies status 'scored'",
            )
        if score:
            try:
                value = float(score)
            except ValueError:
                report("FAIL", run_id, f"score {score!r} is not a number")
            else:
                if not 0.0 <= value <= 1.0:
                    report("FAIL", run_id, f"score {value} is outside [0, 1]")

        scored_raw, total_raw = row.get("items_scored", ""), row.get("items_total", "")
        if scored_raw.isdigit() and total_raw.isdigit():
            if int(scored_raw) > int(total_raw):
                report(
                    "FAIL", run_id,
                    f"items_scored ({scored_raw}) exceeds items_total ({total_raw})",
                )
        if status == "scored" and scored_raw.isdigit() and total_raw.isdigit():
            if int(scored_raw) < int(total_raw):
                report(
                    "WARN", run_id,
                    f"scored only {scored_raw}/{total_raw} items; the score is partial "
                    "and not comparable to a full run",
                )

        canonical = row.get("canonical", "")
        if canonical not in CANONICAL_VALUES:
            report(
                "FAIL", run_id,
                f"canonical {canonical!r} must be one of {sorted(CANONICAL_VALUES)}",
            )
        if canonical == "current":
            current_canonical.append(run_id)
            if status != "scored":
                report(
                    "FAIL", run_id,
                    f"claims current canonical evidence but status is {status!r}",
                )

        # On-disk cross-check. A row whose directory is absent is not automatically wrong:
        # prod-graph and offline rows never had one. It is wrong only when the row also
        # claims a results directory exists.
        run_directory = results_dir / run_id
        exists = run_directory.is_dir()
        claimed = (row.get("has_results_dir") or "").strip().lower()
        if not evidence_present:
            # Nothing on disk to check this row against; see the note above.
            continue
        if claimed in {"true", "yes", "1"} and not exists:
            report("FAIL", run_id, f"claims a results directory but {run_directory} is absent")
        if claimed in {"false", "no", "0"} and exists:
            report("FAIL", run_id, f"claims no results directory but {run_directory} exists")

        if exists:
            provenance_path = run_directory / "run_provenance.json"
            if not provenance_path.exists():
                report("WARN", run_id, "results directory has no run_provenance.json")
            else:
                document = _read_provenance(provenance_path)
                if document is None:
                    report("FAIL", run_id, "run_provenance.json is unreadable")
                else:
                    latest = document.get("latest_attempt") or {}
                    if latest.get("menhir_dirty") or latest.get("bench_dirty"):
                        report(
                            "WARN", run_id,
                            "ran with a DIRTY tree; its commit SHA does not describe the "
                            "code that executed",
                        )
                    if not document.get("surface_fingerprints"):
                        report(
                            "WARN", run_id,
                            "no surface fingerprint; score deltas cannot be attributed to "
                            "files for this run",
                        )
                    recorded_id = document.get("run_id")
                    if recorded_id and recorded_id != run_id:
                        report(
                            "FAIL", run_id,
                            f"provenance records run_id {recorded_id!r}",
                        )

            # Cross-check the recorded score against the run's own per-arm evidence. This is
            # what makes the ledger's number checkable rather than asserted -- a typo, a
            # stale copy, or a number quoted from the wrong arm all show up here.
            score_path = run_directory / "score.json"
            if score_path.exists():
                try:
                    evidence = json.loads(score_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    report("FAIL", run_id, "score.json is unreadable")
                    evidence = None
                if isinstance(evidence, dict):
                    arms = evidence.get("arms") or {}
                    declared = (row.get("primary_arm") or "").strip()
                    if score and not declared:
                        report(
                            "WARN", run_id,
                            f"records a score but no primary_arm; score.json has "
                            f"{len(arms)} arm(s) ({', '.join(sorted(arms))}) and the "
                            "number cannot be checked against the right one",
                        )
                    elif declared and declared not in arms:
                        report(
                            "FAIL", run_id,
                            f"primary_arm {declared!r} is not in score.json "
                            f"(has: {', '.join(sorted(arms)) or 'none'})",
                        )
                    elif declared and score:
                        measured = arms[declared].get("score")
                        try:
                            recorded_value = float(score)
                        except ValueError:
                            recorded_value = None
                        if (
                            isinstance(measured, (int, float))
                            and recorded_value is not None
                            and abs(float(measured) - recorded_value) > 0.0005
                        ):
                            report(
                                "FAIL", run_id,
                                f"records score {recorded_value} for arm {declared!r} but "
                                f"score.json measured {measured}",
                            )
                        arm_n = arms[declared].get("n")
                        if (
                            isinstance(arm_n, int)
                            and total_raw.isdigit()
                            and arm_n != int(total_raw)
                        ):
                            report(
                                "WARN", run_id,
                                f"arm {declared!r} scored {arm_n} items but the row says "
                                f"items_total={total_raw}",
                            )
            elif status == "scored":
                report(
                    "WARN", run_id,
                    "no score.json; the recorded score cannot be checked against the run's "
                    "own evidence (run score_extract.py on this directory)",
                )

    # Disk -> rows. Checking only rows -> disk would miss the more dangerous direction: a
    # run that executed, wrote provenance, and never reached the ledger. That run's evidence
    # exists but is invisible to anyone reading the scoreboard, which is how a result gets
    # silently dropped from a comparison.
    #
    # These are completeness findings, not contradictions, so they are WARN: the scoreboard
    # covers buildout runs, and some of these directories are recall-panel rescores or
    # diagnostics that may not belong in it. The distinction recorded is whether a score is
    # actually readable, because "a scored result is missing" needs following up and "an
    # aborted launch left provenance" does not.
    excluded_seen: list[str] = []
    if evidence_present:
        for candidate in sorted(results_dir.iterdir()):
            if not candidate.is_dir() or candidate.name in seen:
                continue
            if not (candidate / "run_provenance.json").exists():
                # No provenance: a survey or analysis output directory, not a run.
                continue
            if candidate.name in excluded:
                # Deliberately out of the scoreboard; see ledger-excluded.txt. Counted below
                # so the decision stays visible rather than silently swallowing a result.
                excluded_seen.append(candidate.name)
                continue
            described = describe_score(candidate)
            if described is not None:
                report(
                    "WARN", candidate.name,
                    f"a scored result exists on disk with no ledger row [{described}]; "
                    "decide whether it belongs in the scoreboard",
                )
            else:
                report(
                    "WARN", candidate.name,
                    "has provenance but no readable score and no ledger row "
                    "(aborted launch, build-only, or diagnostic run)",
                )

    # A run cannot be both excluded and recorded -- one of the two is wrong.
    for run_id in sorted(set(excluded) & seen):
        report(
            "FAIL", run_id,
            "is listed in ledger-excluded.txt but also has a ledger row; "
            "remove it from one of the two",
        )

    if evidence_present:
        if excluded_seen:
            findings.append({
                "level": "WARN", "run_id": "(ledger)",
                "message": (
                    f"{len(excluded_seen)} run(s) deliberately excluded from the scoreboard "
                    "per ledger-excluded.txt; their scores remain in each run's score.json"
                ),
            })
        # An exclusion naming a directory that is not there is stale bookkeeping, not a
        # suppression that is doing any work.
        stale = sorted(
            run_id for run_id in excluded
            if run_id not in seen and not (results_dir / run_id).is_dir()
        )
        for run_id in stale:
            report(
                "WARN", run_id,
                "is listed in ledger-excluded.txt but no such run directory exists; "
                "the exclusion is stale",
            )

    if len(current_canonical) > 1:
        for run_id in current_canonical:
            findings.append({
                "level": "FAIL",
                "run_id": run_id,
                "message": (
                    "more than one row claims current canonical evidence: "
                    + ", ".join(current_canonical)
                ),
            })
    if not current_canonical:
        findings.append({
            "level": "WARN",
            "run_id": "(ledger)",
            "message": "no row claims current canonical benchmark evidence",
        })

    return findings


# ---------------------------------------------------------------------------
# Join: score delta + surface blame in one command
# ---------------------------------------------------------------------------

def join(
    run_a: str,
    run_b: str,
    csv_path: Path = DEFAULT_CSV,
    results_dir: Path = DEFAULT_RESULTS,
) -> str:
    """Report the score delta between two runs, then what changed underneath them.

    This is the question the ledger could not answer before: a score moved, so where do I
    look first? The score comes from the CSV, the attribution from the surface fingerprints
    in each run's provenance.
    """
    rows = {row["run_id"]: row for row in read_rows(csv_path)}
    lines: list[str] = []

    for run_id in (run_a, run_b):
        if run_id not in rows:
            raise LedgerError(f"no ledger row for {run_id!r}")

    row_a, row_b = rows[run_a], rows[run_b]
    lines.append(f"score: {run_a}  ->  {run_b}")
    for label, row in ((run_a, row_a), (run_b, row_b)):
        shown = row.get("score") or row.get("score_raw") or "--"
        lines.append(f"  {label}: {shown}  [{row.get('status')}]")

    if row_a.get("status") == "scored" and row_b.get("status") == "scored":
        delta = float(row_b["score"]) - float(row_a["score"])
        lines.append(f"  delta: {delta:+.3f}")
    else:
        lines.append(
            "  delta: not computed -- a run that is not 'scored' has no comparable number"
        )

    if row_a.get("items_total") != row_b.get("items_total"):
        lines.append(
            f"  !! different item counts ({row_a.get('items_total')} vs "
            f"{row_b.get('items_total')}); these scores are not directly comparable"
        )

    lines.append("")
    provenance_a = results_dir / run_a / "run_provenance.json"
    provenance_b = results_dir / run_b / "run_provenance.json"
    if not provenance_a.exists() or not provenance_b.exists():
        missing = [
            run_id
            for run_id, path in ((run_a, provenance_a), (run_b, provenance_b))
            if not path.exists()
        ]
        lines.append(f"  no provenance for {', '.join(missing)}; cannot attribute the delta")
        return "\n".join(lines)

    surface_tool = SCRIPT_DIR / "bench_surface.py"
    if not surface_tool.exists():
        lines.append(f"  {surface_tool} missing; cannot attribute the delta")
        return "\n".join(lines)

    result = subprocess.run(
        [sys.executable, str(surface_tool), "blame", str(provenance_a), str(provenance_b)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    lines.append(result.stdout.rstrip() or "  (blame produced no output)")
    if result.returncode != 0 and result.stderr.strip():
        lines.append(f"  blame error: {result.stderr.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    render_parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    validate_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    validate_parser.add_argument(
        "--strict", action="store_true", help="exit non-zero on WARN as well as FAIL"
    )

    join_parser = subparsers.add_parser("join")
    join_parser.add_argument("run_a")
    join_parser.add_argument("run_b")
    join_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    join_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)

    import_parser = subparsers.add_parser("import-markdown")
    import_parser.add_argument("markdown", type=Path)
    import_parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    import_parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.verb == "render":
        render(args.csv, args.markdown)
        print(f"rendered {len(read_rows(args.csv))} rows into {args.markdown}")
        return 0

    if args.verb == "validate":
        findings = validate(args.csv, args.results)
        failures = [f for f in findings if f["level"] == "FAIL"]
        warnings = [f for f in findings if f["level"] == "WARN"]
        for finding in findings:
            print(f"  {finding['level']:<5} {finding['run_id']}: {finding['message']}")
        print(f"\n{len(failures)} FAIL, {len(warnings)} WARN")
        if failures:
            return 1
        return 1 if (args.strict and warnings) else 0

    if args.verb == "join":
        print(join(args.run_a, args.run_b, args.csv, args.results))
        return 0

    rows = import_markdown(args.markdown)
    for row in rows:
        row["has_results_dir"] = (
            "true" if (args.results / row["run_id"]).is_dir() else "false"
        )
    write_rows(rows, args.csv)
    print(f"imported {len(rows)} rows into {args.csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LedgerError as exc:
        print(f"ledger: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
