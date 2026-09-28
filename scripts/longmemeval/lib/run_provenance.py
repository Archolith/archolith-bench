"""Append-only run provenance for LongMemEval buildouts.

A buildout is resumable, so its provenance file outlives the process that created it.
The rule that follows from that: **a later attempt may add to the record, never replace
it.** Overwriting is how a run ends up claiming it was built fresh, at one commit, in one
sitting, when in fact half its namespaces were ingested days earlier by a killed attempt
running different code. The file is the only evidence of which; once overwritten, the
numbers derived from that graph are unattributable.

So every write here appends: attempts accumulate, phases accumulate, and the identity a
resume must match is checked before anything is written.

Verbs (all offline, no services touched)::

    run_provenance.py begin       <path> <attempt-record.json>
    run_provenance.py phase-start <path> --phase NAME [--manifest P] [--setting k=v ...]
    run_provenance.py phase-end   <path> --phase NAME --status STATUS [--setting k=v ...]
    run_provenance.py complete    <path> --harness-exit N

Types are coerced from ``--setting k=v``: ``true``/``false`` become booleans, digit strings
become integers, everything else stays a string -- so the JSON stays machine-readable
instead of stringly-typed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Facts about *which graph this is*. They are fixed when the first attempt creates the run;
# a resume that disagrees on any of them is not resuming this run, it is quietly writing a
# second experiment's history into the first one's file.
IMMUTABLE_KEYS = (
    "run_id",
    "arm",
    "fixture_sha256",
    "fixture_count",
    "container",
    "volume",
    "dataset",
    "variant",
    "namespace_prefix",
)

# Code-commit keys are immutable in canonical mode. A resume that changes the Menhir or bench
# commit is building a mixed-code graph, which makes the result unattributable. Canonical runs
# refuse this; a ``noncanonical`` flag relaxes it for development iteration.
COMMIT_KEYS = (
    "menhir_commit",
    "bench_commit",
)

# The content fingerprint of the benchmark-affecting surface (see bench_surface.py). Immutable
# in canonical mode for the same reason the commits are, and it catches strictly more: a dirty
# tree or an untracked edit changes behavior while leaving both SHAs identical, which is how
# the assistant-turn gate in claim_segmenter.py came to be attributed to the wrong commit.
#
# Treated as absent when empty or missing, not as a digest of "". Fingerprinting is non-fatal
# at launch by design -- evidence about a run, not a precondition for it -- so a run that could
# not compute one must still be resumable. The cost is that such a run is not gated; `validate`
# and `blame` both report a missing fingerprint rather than letting it pass as agreement.
SURFACE_KEYS = (
    "surface_digest",
)

# Phase statuses. "interrupted" is assigned retroactively: a phase recorded as started that
# never recorded an end means the process died inside it.
STATUS_STARTED = "started"
STATUS_INTERRUPTED = "interrupted"


class ProvenanceMismatch(RuntimeError):
    """A resume's immutable identity does not match the existing record."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, document: dict[str, Any]) -> None:
    """Write via a temp file + replace so a crash mid-write cannot truncate the record."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def coerce(raw: str) -> Any:
    """Coerce a shell-supplied scalar to bool/int/str."""
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    candidate = raw.strip()
    if candidate.lstrip("-").isdigit():
        return int(candidate)
    return raw


def parse_settings(pairs: list[str] | None) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator:
            raise ValueError(f"setting must be key=value, got {pair!r}")
        settings[key.strip()] = coerce(value)
    return settings


def identity_of(document: dict[str, Any]) -> dict[str, Any]:
    """The immutable identity carried by *document*, ignoring keys it never had."""
    stored = document.get("identity")
    source = stored if isinstance(stored, dict) else document
    return {key: source[key] for key in IMMUTABLE_KEYS if key in source}


def commits_of(document: dict[str, Any]) -> dict[str, Any]:
    """The commit identity carried by *document*, ignoring keys it never had."""
    stored = document.get("identity")
    source = stored if isinstance(stored, dict) else document
    return {key: source[key] for key in COMMIT_KEYS if key in source}


def surface_of(document: dict[str, Any]) -> dict[str, Any]:
    """The surface fingerprint carried by *document*, ignoring empty or missing values."""
    stored = document.get("identity")
    source = stored if isinstance(stored, dict) else document
    return {key: source[key] for key in SURFACE_KEYS if source.get(key)}


def assert_same_run(
    previous: dict[str, Any],
    attempt: dict[str, Any],
    *,
    noncanonical: bool = False,
) -> None:
    """Refuse a resume whose identity differs from the record it would extend.

    Only keys the earlier record actually carried are compared -- different wrappers write
    different subsets -- but a key that was present and is now absent counts as a mismatch,
    because it means the new attempt cannot demonstrate it is the same run.

    In canonical mode (default), code commits are also immutable: a resume that changes
    ``menhir_commit`` or ``bench_commit`` is building a mixed-code graph. Pass
    ``noncanonical=True`` to relax commit checking for development iteration.
    """
    established = identity_of(previous)
    incoming = identity_of(attempt)
    mismatched = {
        key: (value, incoming.get(key))
        for key, value in established.items()
        if incoming.get(key) != value
    }
    if mismatched:
        details = ", ".join(
            f"{key}: recorded={recorded!r} attempted={attempted!r}"
            for key, (recorded, attempted) in sorted(mismatched.items())
        )
        raise ProvenanceMismatch(
            "refusing to resume: this attempt does not match the recorded run identity "
            f"({details})"
        )

    if not noncanonical:
        established_commits = commits_of(previous)
        incoming_commits = commits_of(attempt)
        commit_mismatched = {
            key: (value, incoming_commits.get(key))
            for key, value in established_commits.items()
            if incoming_commits.get(key) != value
        }
        if commit_mismatched:
            details = ", ".join(
                f"{key}: recorded={recorded!r} attempted={attempted!r}"
                for key, (recorded, attempted) in sorted(commit_mismatched.items())
            )
            raise ProvenanceMismatch(
                "refusing canonical resume: code commits changed since the original attempt "
                f"({details}). Set LME_NONCANONICAL=1 to permit mixed-code development runs."
            )

        # Surface drift. Checked after the commits so the clearer diagnosis wins when both
        # moved; when only this one moved, the commits are identical and the cause is an
        # uncommitted or untracked edit -- which no SHA comparison can see.
        established_surface = surface_of(previous)
        incoming_surface = surface_of(attempt)
        for key, recorded in established_surface.items():
            attempted = incoming_surface.get(key)
            if attempted is None:
                raise ProvenanceMismatch(
                    f"refusing canonical resume: the original attempt recorded {key} "
                    f"{recorded[:12]}... but this attempt recorded none, so it cannot show it "
                    "is running the same code. Set LME_NONCANONICAL=1 to permit it."
                )
            if attempted != recorded:
                same_commits = commits_of(previous) == commits_of(attempt)
                cause = (
                    " The commits are identical, so this is an uncommitted or untracked edit "
                    "to a declared surface file -- run `bench_surface.py fingerprint` to see "
                    "the current surface, and `git status` to find the change."
                    if same_commits else
                    " The commits also differ; see the commit error above."
                )
                raise ProvenanceMismatch(
                    f"refusing canonical resume: {key} changed since the original attempt "
                    f"(recorded={recorded[:12]}... attempted={attempted[:12]}...). Resuming "
                    "would build one graph from two different code states, making its score "
                    f"unattributable.{cause} Set LME_NONCANONICAL=1 to permit it."
                )


def mark_interrupted(phases: list[dict[str, Any]], at: str) -> int:
    """Close out phases left in ``started`` by a previous attempt. Returns how many."""
    interrupted = 0
    for phase in phases:
        if phase.get("status") == STATUS_STARTED:
            phase["status"] = STATUS_INTERRUPTED
            phase["interrupted_at"] = at
            interrupted += 1
    return interrupted


def manifest_state(manifest_path: Path | None) -> dict[str, Any]:
    """Manifest count and question ids as a phase starts.

    This is what makes a phase record answer "which items did this phase inherit, and which
    did it add" -- without it a resumed phase is indistinguishable from a fresh one.
    """
    if manifest_path is None:
        return {}
    if not manifest_path.exists():
        return {"manifest_items_at_start": 0, "manifest_question_ids_at_start": []}
    rows = _read(manifest_path)
    if not isinstance(rows, list):
        raise ValueError(f"manifest is not a JSON list: {manifest_path}")
    return {
        "manifest_items_at_start": len(rows),
        "manifest_question_ids_at_start": [
            str(row.get("question_id") or "") for row in rows if isinstance(row, dict)
        ],
    }


def begin(
    path: Path,
    attempt: dict[str, Any],
    *,
    noncanonical: bool = False,
) -> dict[str, Any]:
    """Start (or resume) a run. On resume this ADDS ONLY -- it never restates the top level.

    The top-level fields belong to the attempt that created the run, and they describe the
    graph on disk: ``graph_fresh``, ``volume_pre_existed``, the dataset, the settings the data
    was actually ingested under. A resume is by definition not fresh, so copying the resuming
    attempt over the top level would flip ``graph_fresh`` true -> false and rewrite the
    settings -- and ``lme.sh ir-gate`` reads exactly that field to decide whether the data is
    fresh. It would report a graph as stale that is not, or (with the values reversed) vouch
    for one that is.

    So: the top level is written once and frozen. Every attempt, the first included, is stored
    whole in ``attempts``; ``latest_attempt`` mirrors the most recent for readers that want
    current values without walking the list.

    When *noncanonical* is true, code-commit drift between attempts is permitted but the
    document is labelled ``noncanonical`` so downstream consumers (ir-gate, the acceptance
    report) can distinguish development iteration from an attributable canonical run.
    """
    started_at = attempt.get("started_at") or _now()
    snapshot = dict(attempt)

    if not path.exists():
        document = dict(attempt)
        document["identity"] = identity_of(attempt)
        document["identity"].update(commits_of(attempt))
        document["identity"].update(surface_of(attempt))
        document["first_started_at"] = started_at
        document["phases"] = []
        document["attempts"] = [{**snapshot, "attempt": 1, "phases_interrupted": 0}]
        document["latest_attempt"] = document["attempts"][0]
        # Present from the first write, so a reader never has to special-case attempt one.
        document["last_started_at"] = started_at
        document["attempt_count"] = 1
        if noncanonical:
            document["noncanonical"] = True
        return document

    document = _read(path)
    assert_same_run(document, attempt, noncanonical=noncanonical)
    if noncanonical and not document.get("noncanonical"):
        document["noncanonical"] = True
    phases = document.setdefault("phases", [])
    interrupted = mark_interrupted(phases, started_at)
    attempts = document.setdefault("attempts", [])
    attempts.append(
        {**snapshot, "attempt": len(attempts) + 1, "phases_interrupted": interrupted}
    )
    document["latest_attempt"] = attempts[-1]
    # Additive breadcrumbs only. Everything else at the top level stays exactly as the first
    # attempt wrote it.
    document["last_started_at"] = started_at
    document["attempt_count"] = len(attempts)
    return document


def phase_start(
    document: dict[str, Any],
    *,
    phase: str,
    settings: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "phase": phase,
        "status": STATUS_STARTED,
        "started_at": _now(),
        **manifest,
        "effective_settings": settings,
    }
    document.setdefault("phases", []).append(entry)
    return document


def phase_end(
    document: dict[str, Any],
    *,
    phase: str,
    status: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    phases = document.setdefault("phases", [])
    for entry in reversed(phases):
        if entry.get("phase") == phase and entry.get("status") == STATUS_STARTED:
            entry.update({"status": status, "completed_at": _now(), **extra})
            return document
    # No open phase: still record the end rather than dropping it on the floor.
    phases.append({"phase": phase, "status": status, "completed_at": _now(), **extra})
    return document


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)

    begin_parser = subparsers.add_parser("begin")
    begin_parser.add_argument("path", type=Path)
    begin_parser.add_argument("record", type=Path)
    begin_parser.add_argument(
        "--noncanonical", action="store_true",
        help="Permit mixed-code resumes (commit drift). Labels the run noncanonical.",
    )

    start_parser = subparsers.add_parser("phase-start")
    start_parser.add_argument("path", type=Path)
    start_parser.add_argument("--phase", required=True)
    start_parser.add_argument("--manifest", type=Path, default=None)
    start_parser.add_argument("--setting", action="append", default=[])

    end_parser = subparsers.add_parser("phase-end")
    end_parser.add_argument("path", type=Path)
    end_parser.add_argument("--phase", required=True)
    end_parser.add_argument("--status", required=True)
    end_parser.add_argument("--setting", action="append", default=[])

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("path", type=Path)
    complete_parser.add_argument("--harness-exit", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.verb == "begin":
        document = begin(
            args.path, _read(args.record), noncanonical=args.noncanonical,
        )
    elif args.verb == "phase-start":
        document = phase_start(
            _read(args.path),
            phase=args.phase,
            settings=parse_settings(args.setting),
            manifest=manifest_state(args.manifest),
        )
    elif args.verb == "phase-end":
        document = phase_end(
            _read(args.path),
            phase=args.phase,
            status=args.status,
            extra=parse_settings(args.setting),
        )
    else:
        document = _read(args.path)
        document["completed_at"] = _now()
        document["harness_exit"] = args.harness_exit

    _write_atomic(args.path, document)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceMismatch as exc:
        print(f"run_provenance: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
