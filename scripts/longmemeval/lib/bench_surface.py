"""Content fingerprint of the benchmark-affecting surface, and A/B attribution over it.

A commit SHA does not establish what code a run executed. Two runs can share a SHA and
differ (dirty tree, untracked file), and a file can change behavior in one commit while
being attributed to another -- the assistant-turn gate in ``claim_segmenter.py`` was
authored untracked and swept into git by a ``chore: track untracked scripts`` commit, so
``git log`` dates it wrongly and a commit-range diff never names it.

What does establish it is the content of the files that can move a score. This module
hashes the surface declared in ``bench-surface.yaml`` and records it per attempt, so
comparing two runs yields file-level evidence instead of a commit-range guess.

Three properties the fingerprint must have, because it is evidence:

* **A declared-but-absent file is recorded, not skipped.** Deleting a surface file must
  change the digest; a skipped path would leave it unchanged.
* **The manifest hashes itself.** Narrowing the surface is itself a change to what is
  being tracked. Without ``manifest_sha256`` in the digest, someone could delete a glob
  and the digest would go on looking stable while tracking less.
* **A missing repo is an error, not a column of nulls.** An absent checkout would
  otherwise fingerprint as "every menhir file deleted", which is both wrong and, because
  it is internally consistent, stable across runs.

Verbs (all offline, no services touched)::

    bench_surface.py fingerprint [--manifest P] [--json OUT]
    bench_surface.py attach <provenance.json> [--manifest P]
    bench_surface.py blame <run-a.json> <run-b.json> [--stage NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent            # lib/
LME_DIR = SCRIPT_DIR.parent                             # scripts/longmemeval/
BENCH_ROOT = LME_DIR.parents[1]                         # archolith-bench/
DEFAULT_MANIFEST = LME_DIR / "bench-surface.yaml"

# Stage order is pipeline order. Blame output follows it so a reader can apply the obvious
# causal filter: a recall-stage change cannot explain an ingest-stage episode-count delta.
STAGE_ORDER = ("fixture", "ingest", "consolidate", "recall", "score", "harness")


class SurfaceError(RuntimeError):
    """The surface cannot be fingerprinted honestly."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path, *, normalize_line_endings: bool) -> str:
    payload = path.read_bytes()
    if normalize_line_endings:
        # CRLF -> LF only. Collapses checkout differences between Windows and Linux, which
        # are not behavior differences in Python. Nothing else is normalized, so two files
        # that differ in any way that matters still hash differently.
        payload = payload.replace(b"\r\n", b"\n")
    return _sha256_bytes(payload)


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    """Return the parsed manifest and its own content hash."""
    if not manifest_path.exists():
        raise SurfaceError(f"surface manifest not found: {manifest_path}")
    raw = manifest_path.read_bytes()
    document = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(document, dict) or "stages" not in document:
        raise SurfaceError(f"surface manifest has no 'stages': {manifest_path}")
    return document, _sha256_bytes(raw.replace(b"\r\n", b"\n"))


def resolve_repos(document: dict[str, Any]) -> dict[str, Path]:
    """Resolve each declared repo root, honouring env overrides.

    A declared repo that is not on disk raises: fingerprinting it as a set of missing
    files would be a self-consistent lie, stable across runs and indistinguishable from
    a repo whose files were genuinely deleted.
    """
    resolved: dict[str, Path] = {}
    for name, spec in (document.get("repos") or {}).items():
        spec = spec or {}
        override_var = spec.get("env_override")
        override = os.environ.get(override_var) if override_var else None
        root = Path(override) if override else (BENCH_ROOT / spec.get("root", ".")).resolve()
        if not root.is_dir():
            hint = f" (set {override_var})" if override_var else ""
            raise SurfaceError(f"declared repo {name!r} not found at {root}{hint}")
        resolved[name] = root
    return resolved


def fingerprint(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Hash every file in the declared surface.

    Returns a document carrying per-file hashes grouped by stage, a per-stage digest, and
    a rolled-up ``surface_digest`` that also covers the manifest itself.
    """
    document, manifest_sha = load_manifest(manifest_path)
    repos = resolve_repos(document)
    normalize = bool(document.get("normalize_line_endings", True))

    stages: dict[str, Any] = {}
    # Accumulates "<stage>\0<repo>/<path>\0<hash>" lines, sorted, for the rolled digest.
    digest_lines: list[str] = []

    for stage_name, stage_spec in (document.get("stages") or {}).items():
        files: dict[str, str | None] = {}
        missing: list[str] = []
        for entry in (stage_spec or {}).get("paths") or []:
            repo_name = entry.get("repo")
            pattern = entry.get("glob")
            if repo_name not in repos:
                raise SurfaceError(
                    f"stage {stage_name!r} references undeclared repo {repo_name!r}"
                )
            root = repos[repo_name]
            matches = sorted(root.glob(pattern))
            if not matches:
                # A glob matching nothing is recorded. It may be a deleted file or a
                # renamed module; either way the surface shrank and that must be visible.
                key = f"{repo_name}:{pattern}"
                files[key] = None
                missing.append(key)
                digest_lines.append(f"{stage_name}\0{key}\0MISSING")
                continue
            for match in matches:
                if not match.is_file():
                    continue
                key = f"{repo_name}:{match.relative_to(root).as_posix()}"
                file_hash = _hash_file(match, normalize_line_endings=normalize)
                files[key] = file_hash
                digest_lines.append(f"{stage_name}\0{key}\0{file_hash}")

        stage_payload = "\n".join(
            sorted(line for line in digest_lines if line.startswith(f"{stage_name}\0"))
        )
        stages[stage_name] = {
            "files": dict(sorted(files.items())),
            "file_count": sum(1 for value in files.values() if value is not None),
            "unmatched_globs": missing,
            "stage_digest": _sha256_bytes(stage_payload.encode("utf-8")),
        }

    rolled = "\n".join(sorted(digest_lines))
    return {
        "surface_version": document.get("version", 1),
        "manifest": str(manifest_path.relative_to(BENCH_ROOT).as_posix())
        if manifest_path.is_relative_to(BENCH_ROOT)
        else str(manifest_path),
        # Part of the rolled digest: narrowing the surface is a change to the surface.
        "manifest_sha256": manifest_sha,
        "normalize_line_endings": normalize,
        "repos": {name: str(root) for name, root in sorted(repos.items())},
        "stages": stages,
        "surface_digest": _sha256_bytes(
            (manifest_sha + "\n" + rolled).encode("utf-8")
        ),
        "total_files": sum(stage["file_count"] for stage in stages.values()),
    }


# ---------------------------------------------------------------------------
# Attach to provenance
# ---------------------------------------------------------------------------

def attach(provenance_path: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Record the current surface fingerprint on a provenance document.

    Append-only, matching ``run_provenance.py``: the fingerprint is stored against the
    attempt that is running, because separate attempts of one run may execute different
    code. ``surface_digest`` is mirrored at the top level for cheap reading, but the
    per-attempt list is the record.
    """
    if not provenance_path.exists():
        raise SurfaceError(f"provenance file not found: {provenance_path}")
    document = json.loads(provenance_path.read_text(encoding="utf-8"))
    snapshot = fingerprint(manifest_path)

    attempt_number = document.get("attempt_count", len(document.get("attempts", [])) or 1)
    history = document.setdefault("surface_fingerprints", [])
    history.append({"attempt": attempt_number, **snapshot})
    document["surface_digest"] = snapshot["surface_digest"]
    document["surface_digest_attempt"] = attempt_number

    temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(provenance_path)
    return snapshot


# ---------------------------------------------------------------------------
# Blame
# ---------------------------------------------------------------------------

def _latest_fingerprint(document: dict[str, Any]) -> dict[str, Any] | None:
    history = document.get("surface_fingerprints")
    if isinstance(history, list) and history:
        return history[-1]
    # A bare fingerprint file (from `fingerprint --json`) is also accepted.
    if "surface_digest" in document and "stages" in document:
        return document
    return None


def _settings_of(document: dict[str, Any]) -> dict[str, Any]:
    """Effective settings for a run, preferring the last phase that recorded them."""
    for phase in reversed(document.get("phases") or []):
        settings = phase.get("effective_settings")
        if isinstance(settings, dict) and settings:
            return settings
    latest = document.get("latest_attempt")
    return latest if isinstance(latest, dict) else {}


def _flatten_files(snapshot: dict[str, Any]) -> dict[str, tuple[str, str | None]]:
    """Map "repo:path" -> (stage, hash)."""
    flat: dict[str, tuple[str, str | None]] = {}
    for stage, payload in (snapshot.get("stages") or {}).items():
        for key, value in (payload.get("files") or {}).items():
            flat[key] = (stage, value)
    return flat


def _commits_touching(repo_root: Path, rel_paths: list[str], a: str, b: str) -> list[str]:
    """Commits between two SHAs that touch the given paths. Best-effort."""
    if not rel_paths or not a or not b:
        return []
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"{a}..{b}", "--"] + rel_paths,
            cwd=repo_root, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def blame(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    *,
    stage_filter: str | None = None,
) -> dict[str, Any]:
    """Attribute the difference between two runs to specific surface files and settings."""
    report: dict[str, Any] = {
        "run_a": run_a.get("run_id") or "?",
        "run_b": run_b.get("run_id") or "?",
        "warnings": [],
    }

    for label, document in (("run_a", run_a), ("run_b", run_b)):
        latest = document.get("latest_attempt") or {}
        if latest.get("menhir_dirty") or latest.get("bench_dirty"):
            report["warnings"].append(
                f"{label} ({document.get('run_id')}) ran with a DIRTY tree; "
                "its commit SHA does not describe the code that executed"
            )
        if latest.get("menhir_dirty") is None and "menhir_commit" in latest:
            report["warnings"].append(
                f"{label} ({document.get('run_id')}) recorded no dirty flag; "
                "cleanliness is unknown"
            )

    fa, fb = _latest_fingerprint(run_a), _latest_fingerprint(run_b)
    if fa is None or fb is None:
        missing = [n for n, f in (("run_a", fa), ("run_b", fb)) if f is None]
        report["warnings"].append(
            f"no surface fingerprint for {', '.join(missing)}; "
            "falling back to settings and commit-range evidence only"
        )
        report["surface_comparable"] = False
    else:
        report["surface_comparable"] = True
        report["surface_digest_a"] = fa.get("surface_digest")
        report["surface_digest_b"] = fb.get("surface_digest")
        report["surface_identical"] = fa.get("surface_digest") == fb.get("surface_digest")

        if fa.get("manifest_sha256") != fb.get("manifest_sha256"):
            report["warnings"].append(
                "the surface manifest ITSELF changed between these runs; "
                "the two runs were not measured against the same declared surface"
            )

        flat_a, flat_b = _flatten_files(fa), _flatten_files(fb)
        changed: list[dict[str, Any]] = []
        for key in sorted(set(flat_a) | set(flat_b)):
            stage_a, hash_a = flat_a.get(key, (None, None))
            stage_b, hash_b = flat_b.get(key, (None, None))
            if hash_a == hash_b and key in flat_a and key in flat_b:
                continue
            stage = stage_b or stage_a or "?"
            if stage_filter and stage != stage_filter:
                continue
            if key not in flat_a:
                kind = "added to surface"
            elif key not in flat_b:
                kind = "removed from surface"
            elif hash_a is None:
                kind = "appeared"
            elif hash_b is None:
                kind = "deleted"
            else:
                kind = "modified"
            changed.append({"file": key, "stage": stage, "change": kind})

        changed.sort(key=lambda row: (
            STAGE_ORDER.index(row["stage"]) if row["stage"] in STAGE_ORDER else 99,
            row["file"],
        ))
        report["changed_files"] = changed
        report["changed_stages"] = sorted(
            {row["stage"] for row in changed},
            key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99,
        )

        # Commit attribution is secondary: it explains files already identified as changed,
        # and is skipped for dirty runs where the SHA range is not trustworthy.
        repos = {**(fb.get("repos") or {}), **(fa.get("repos") or {})}
        commits: dict[str, list[str]] = {}
        la, lb = run_a.get("latest_attempt") or {}, run_b.get("latest_attempt") or {}
        for repo_name, root in repos.items():
            sha_a = la.get(f"{repo_name}_commit")
            sha_b = lb.get(f"{repo_name}_commit")
            rels = [
                row["file"].split(":", 1)[1]
                for row in changed
                if row["file"].startswith(f"{repo_name}:") and ":" in row["file"]
            ]
            found = _commits_touching(Path(root), rels, sha_a or "", sha_b or "")
            if found:
                commits[repo_name] = found
        report["commits_touching_changed_files"] = commits

    settings_a, settings_b = _settings_of(run_a), _settings_of(run_b)
    setting_changes = []
    for key in sorted(set(settings_a) | set(settings_b)):
        if settings_a.get(key) != settings_b.get(key):
            setting_changes.append(
                {"setting": key, "a": settings_a.get(key), "b": settings_b.get(key)}
            )
    report["changed_settings"] = setting_changes

    if run_a.get("fixture_sha256") and run_b.get("fixture_sha256"):
        if run_a["fixture_sha256"] != run_b["fixture_sha256"]:
            report["warnings"].append(
                "FIXTURE DIFFERS between these runs; their scores are not comparable"
            )

    return report


def format_blame(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"surface blame: {report['run_a']}  ->  {report['run_b']}")
    lines.append("")

    for warning in report.get("warnings") or []:
        lines.append(f"  !! {warning}")
    if report.get("warnings"):
        lines.append("")

    if report.get("surface_comparable"):
        if report.get("surface_identical"):
            lines.append("  surface digest IDENTICAL - no declared-surface file changed.")
            lines.append("  Look at settings below, or at behavior outside the declared")
            lines.append("  surface (then widen bench-surface.yaml).")
        else:
            changed = report.get("changed_files") or []
            stages = ", ".join(report.get("changed_stages") or [])
            lines.append(f"  {len(changed)} surface file(s) changed, stages: {stages}")
            lines.append("  (pipeline order - an earlier stage can explain a later symptom,")
            lines.append("   not the reverse)")
            lines.append("")
            current = None
            for row in changed:
                if row["stage"] != current:
                    current = row["stage"]
                    lines.append(f"    [{current}]")
                lines.append(f"      {row['change']:<20} {row['file']}")
            commits = report.get("commits_touching_changed_files") or {}
            if commits:
                lines.append("")
                lines.append("  commits touching those files:")
                for repo, entries in sorted(commits.items()):
                    lines.append(f"    {repo}:")
                    for entry in entries:
                        lines.append(f"      {entry}")
    lines.append("")

    settings = report.get("changed_settings") or []
    if settings:
        lines.append(f"  {len(settings)} setting(s) differ:")
        for row in settings:
            lines.append(f"    {row['setting']}: {row['a']!r} -> {row['b']!r}")
    else:
        lines.append("  no recorded setting differs.")

    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="verb", required=True)

    fingerprint_parser = subparsers.add_parser("fingerprint")
    fingerprint_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    fingerprint_parser.add_argument("--json", type=Path, default=None)
    fingerprint_parser.add_argument("--quiet", action="store_true")

    attach_parser = subparsers.add_parser("attach")
    attach_parser.add_argument("path", type=Path)
    attach_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    blame_parser = subparsers.add_parser("blame")
    blame_parser.add_argument("run_a", type=Path)
    blame_parser.add_argument("run_b", type=Path)
    blame_parser.add_argument("--stage", default=None)
    blame_parser.add_argument("--json", action="store_true")

    return parser


def _read_json(path: Path) -> dict[str, Any]:
    # strict=False: some historical provenance files carry raw control characters in
    # free-text purpose fields.
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.verb == "fingerprint":
        snapshot = fingerprint(args.manifest)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"surface_digest: {snapshot['surface_digest']}")
            print(f"files hashed:   {snapshot['total_files']}")
            for stage in STAGE_ORDER:
                payload = snapshot["stages"].get(stage)
                if not payload:
                    continue
                note = ""
                if payload["unmatched_globs"]:
                    note = f"  !! {len(payload['unmatched_globs'])} glob(s) matched nothing"
                print(f"  {stage:<12} {payload['file_count']:>4} files{note}")
        return 0

    if args.verb == "attach":
        snapshot = attach(args.path, args.manifest)
        print(f"attached surface_digest {snapshot['surface_digest']} to {args.path}")
        return 0

    report = blame(_read_json(args.run_a), _read_json(args.run_b), stage_filter=args.stage)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_blame(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SurfaceError as exc:
        print(f"bench_surface: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
