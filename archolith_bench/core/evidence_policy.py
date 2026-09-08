"""Offline evidence policy validator for archolith-bench evidence safety.

Validates HEADLINE-NUMBERS.md and benchmark evidence artifacts against the
repo's public-claim policy.  No network, no model calls, no Docker.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

REJECTED_ACTIVE_TERMS: tuple[str, ...] = (
    "fixture",
    "sample",
    "demo",
    "historical",
    "candidate",
    "pending",
    "not for copy",
    "internal only",
    "smoke only",
    "offline stub",
)

PLACEHOLDER_TERMS: tuple[str, ...] = (
    "pending",
    "none",
    "tbd",
    "unknown",
    "_none_",
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")

REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "title",
    "command",
    "commit",
    "product",
    "ability",
    "fixture_or_live_source",
    "model_provider",
    "environment_caveats",
    "metric_rows",
    "artifact",
    "public_copy_allowed",
)

HEADLINE_COLUMNS = ("Product", "Claim", "Value", "Source", "Commit", "Run date", "Notes")


@dataclass
class PolicyIssue:
    severity: Literal["error", "warning"]
    code: str
    path: str
    message: str


@dataclass
class PolicyResult:
    ok: bool
    errors: list[PolicyIssue] = field(default_factory=list)
    warnings: list[PolicyIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_error(code: str, path: str, message: str) -> PolicyIssue:
    return PolicyIssue(severity="error", code=code, path=path, message=message)


def _make_warning(code: str, path: str, message: str) -> PolicyIssue:
    return PolicyIssue(severity="warning", code=code, path=path, message=message)


def _is_commit_like(s: str) -> bool:
    return bool(_COMMIT_RE.match(s.strip()))


def _is_date_like(s: str) -> bool:
    return bool(_DATE_RE.match(s.strip()))


def _is_placeholder(s: str) -> bool:
    return s.strip().lower() in PLACEHOLDER_TERMS


def _contains_rejected_term(s: str) -> bool:
    """Check if *s* contains any of the REJECTED_ACTIVE_TERMS."""
    lower = s.strip().lower()
    for term in REJECTED_ACTIVE_TERMS:
        if term in lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Provenance verification
# ---------------------------------------------------------------------------
#
# Well-formed is not the same as true. `commit: deadbeef` matches the hex
# regex, `source_tracked: true` is just a word the author typed, and neither
# was ever checked against the repository -- so a fabricated block validated
# clean. These helpers ask git instead of trusting the assertion.
#
# Each returns True (verified), False (verified false) or None (could not
# check: no git, no repo, timeout). None is never treated as success; callers
# fail closed on it for public copy.

POLICY_REPO_ROOT = Path(__file__).resolve().parents[2]

_GIT_TIMEOUT_S = 10


def _git(args: list[str], repo_root: Path) -> tuple[int, str] | None:
    """Run a git command in *repo_root*. None when git cannot be run at all."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.returncode, proc.stdout.strip()


def commit_exists(commit: str, repo_root: Path = POLICY_REPO_ROOT) -> bool | None:
    """Whether *commit* is an object that exists in this repository.

    The `^{commit}` suffix forces the object to actually be a commit, so a
    hash that happens to name a blob or tree is not accepted as provenance.
    """
    value = (commit or "").strip()
    if not value:
        return False
    out = _git(["cat-file", "-e", f"{value}^{{commit}}"], repo_root)
    if out is None:
        return None
    return out[0] == 0


def source_is_tracked(source: str, repo_root: Path = POLICY_REPO_ROOT) -> bool | None:
    """Whether *source* names a path tracked by git.

    `source` is deliberately free-form -- existing artifacts carry run names
    and prose ("offline scenario suite (fixtures)") as well as paths. Prose is
    not a tracked path, so it verifies False rather than raising: asserting
    source_tracked=true over an unresolvable source is the thing being caught.
    """
    value = (source or "").strip()
    if not value:
        return False
    out = _git(["ls-files", "--error-unmatch", "--", value], repo_root)
    if out is None:
        return None
    return out[0] == 0


def _add_provenance_issue(
    result: PolicyResult,
    pc_allowed: bool,
    code: str,
    path: Path,
    message: str,
) -> None:
    """Record a failed provenance check at the severity public copy warrants."""
    if pc_allowed:
        result.ok = False
        result.errors.append(_make_error(code, str(path), message))
    else:
        result.warnings.append(_make_warning(code, str(path), message))


def _is_future_date(run_date: str, today: date | None = None) -> bool:
    """Whether *run_date* is after today. A run cannot have happened later."""
    try:
        parsed = date.fromisoformat(run_date.strip())
    except ValueError:
        return False
    return parsed > (today or date.today())


# ---------------------------------------------------------------------------
# Markdown table parsing
# ---------------------------------------------------------------------------

def _parse_headline_table(text: str) -> list[dict[str, str]]:
    """Return parsed rows from the ``## Active Headline Numbers`` table.

    Each row maps column name → cell value (stripped).
    """
    start = text.find("## Active Headline Numbers")
    if start == -1:
        return []

    after_header = text.index("\n", start)
    rest = text[after_header:]
    end = rest.find("\n## ")
    if end == -1:
        section = rest
    else:
        section = rest[:end]

    lines = section.strip().splitlines()
    rows: list[dict[str, str]] = []
    in_header = True
    columns: list[str] = []

    for line in lines:
        raw = line.strip()
        if not raw.startswith("|") or not raw.endswith("|"):
            continue
        if re.match(r"^\|[-\s|]+\|$", raw):
            continue
        cells = [c.strip().strip("`_").strip() for c in raw.split("|")[1:-1]]

        if in_header:
            columns = cells
            in_header = False
            continue

        if not cells or all(c in ("", "_none_") for c in cells):
            rows.append({col: "" for col in columns})
            continue

        row = {}
        for i, col in enumerate(columns):
            row[col] = cells[i] if i < len(cells) else ""
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# HEADLINE-NUMBERS.md validation
# ---------------------------------------------------------------------------

def validate_headline_numbers(headline_path: Path) -> PolicyResult:
    """Validate *HEADLINE-NUMBERS.md* active claims."""
    result = PolicyResult(ok=True, summary={
        "headline_active_claims": 0,
    })

    if not headline_path.exists():
        result.ok = False
        result.errors.append(_make_error(
            "file_not_found", str(headline_path),
            "HEADLINE-NUMBERS.md not found",
        ))
        return result

    text = headline_path.read_text(encoding="utf-8")
    rows = _parse_headline_table(text)

    if not rows:
        # No table found at all — treat as a warning but not a hard error
        result.warnings.append(_make_warning(
            "no_active_table", str(headline_path),
            "No ``## Active Headline Numbers`` table found",
        ))
        return result

    # Single placeholder row with _none_ → pass
    # After stripping underscores, "_none_" becomes "none"; also accept "none" as product.
    first = rows[0]
    product_val = first.get("Product", "").strip().lower()
    is_none = product_val in ("_none_", "none") or _is_placeholder(product_val)
    if is_none:
        return result

    # Validate each active claim row
    active_rows = 0
    for i, row in enumerate(rows):
        active_rows += 1
        ridx = i + 1  # 1-based row number

        for col in HEADLINE_COLUMNS:
            val = row.get(col, "")
            if col in ("Product", "Claim", "Value"):
                if _is_placeholder(val) or not val.strip():
                    result.ok = False
                    result.errors.append(_make_error(
                        "missing_claim_field", str(headline_path),
                        f"Active headline row {ridx} column '{col}' is blank or placeholder",
                    ))

            elif col == "Source":
                if not val.strip() or _is_placeholder(val):
                    result.ok = False
                    result.errors.append(_make_error(
                        "missing_source", str(headline_path),
                        f"Active headline row {ridx}: source is blank or placeholder",
                    ))

            elif col == "Commit":
                if not val.strip() or _is_placeholder(val) or not _is_commit_like(val):
                    result.ok = False
                    result.errors.append(_make_error(
                        "missing_commit", str(headline_path),
                        f"Active headline row {ridx}: commit is missing or not commit-like",
                    ))

            elif col == "Run date":
                if not val.strip() or _is_placeholder(val) or not _is_date_like(val):
                    result.ok = False
                    result.errors.append(_make_error(
                        "invalid_run_date", str(headline_path),
                        f"Active headline row {ridx}: run date is missing or not YYYY-MM-DD",
                    ))

            elif col == "Notes":
                if not val.strip() or _is_placeholder(val):
                    result.ok = False
                    result.errors.append(_make_error(
                        "missing_notes", str(headline_path),
                        f"Active headline row {ridx}: notes are blank or placeholder",
                    ))

        for col in ("Product", "Claim", "Value", "Source", "Notes"):
            if _contains_rejected_term(row.get(col, "")):
                result.ok = False
                result.errors.append(_make_error(
                    "rejected_term_in_active_claim", str(headline_path),
                    f"Active headline row {ridx} column '{col}' contains rejected term: "
                    f"'{row.get(col, '')}'",
                ))

    result.summary["headline_active_claims"] = active_rows
    return result


# ---------------------------------------------------------------------------
# Evidence artifact validation
# ---------------------------------------------------------------------------

def validate_evidence_artifact(artifact_path: Path) -> PolicyResult:
    """Validate a single evidence JSON artifact."""
    result = PolicyResult(ok=True, summary={
        "artifact_path": str(artifact_path),
        "public_copy_allowed": False,
    })

    if not artifact_path.exists():
        result.ok = False
        result.errors.append(_make_error(
            "file_not_found", str(artifact_path),
            "Evidence file not found",
        ))
        return result

    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        result.ok = False
        result.errors.append(_make_error(
            "malformed_json", str(artifact_path),
            f"Malformed JSON: {e}",
        ))
        return result

    if not isinstance(data, dict):
        result.ok = False
        result.errors.append(_make_error(
            "malformed_json", str(artifact_path),
            "Expected a JSON object at top level",
        ))
        return result

    pc_allowed = _get_bool(data, "public_copy_allowed")
    result.summary["public_copy_allowed"] = pc_allowed

    if pc_allowed:
        # Hard failures for missing fields
        for field in REQUIRED_EVIDENCE_FIELDS:
            if field == "public_copy_allowed":
                continue
            if field not in data or data[field] is None:
                result.ok = False
                result.errors.append(_make_error(
                    "missing_field", str(artifact_path),
                    f"public_copy_allowed=true but required field '{field}' is missing",
                ))

        if not _is_commit_like(data.get("commit", "")):
            result.ok = False
            result.errors.append(_make_error(
                "missing_commit", str(artifact_path),
                "public_copy_allowed=true but commit is missing or not commit-like",
            ))

        if not data.get("command", "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_command", str(artifact_path),
                "public_copy_allowed=true but command is missing",
            ))

        if not data.get("product", "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_product", str(artifact_path),
                "public_copy_allowed=true but product is missing",
            ))

        if not data.get("ability", "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_ability", str(artifact_path),
                "public_copy_allowed=true but ability is missing",
            ))

        if not data.get("fixture_or_live_source", "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_fixture_or_live_source", str(artifact_path),
                "public_copy_allowed=true but fixture_or_live_source is missing",
            ))

        if not data.get("model_provider", "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_model_provider", str(artifact_path),
                "public_copy_allowed=true but model_provider is missing",
            ))

        metric_rows = data.get("metric_rows", [])
        if not metric_rows:
            result.ok = False
            result.errors.append(_make_error(
                "empty_metric_rows", str(artifact_path),
                "public_copy_allowed=true but metric_rows is empty",
            ))

        caveats = data.get("environment_caveats", [])
        if not caveats:
            result.ok = False
            result.errors.append(_make_error(
                "empty_caveats", str(artifact_path),
                "public_copy_allowed=true but environment_caveats is empty",
            ))

        artifact_val = data.get("artifact")
        if artifact_val is None:
            result.ok = False
            result.errors.append(_make_error(
                "missing_artifact", str(artifact_path),
                "public_copy_allowed=true but artifact is missing",
            ))

        # Reject fixture/sample/demo language in fixture_or_live_source and caveats
        source = data.get("fixture_or_live_source", "")
        if _contains_rejected_term(source):
            result.ok = False
            result.errors.append(_make_error(
                "fixture_source_rejected", str(artifact_path),
                f"public_copy_allowed=true but fixture_or_live_source contains rejected term: "
                f"'{source}'",
            ))

        for ci, caveat in enumerate(caveats):
            if _contains_rejected_term(caveat):
                result.ok = False
                result.errors.append(_make_error(
                    "caveat_rejected_term", str(artifact_path),
                    f"public_copy_allowed=true but environment_caveats[{ci}] contains "
                    f"rejected term: '{caveat}'",
                ))

    else:
        # public_copy_allowed=false: missing fields are warnings, not errors
        for field in REQUIRED_EVIDENCE_FIELDS:
            if field == "public_copy_allowed":
                continue
            if field not in data or data[field] is None:
                result.warnings.append(_make_warning(
                    "missing_field", str(artifact_path),
                    f"public_copy_allowed=false: field '{field}' is missing",
                ))

        if not _is_commit_like(data.get("commit", "")):
            result.warnings.append(_make_warning(
                "missing_commit", str(artifact_path),
                "public_copy_allowed=false: commit is missing or not commit-like",
            ))

    return result


def _get_bool(data: dict, key: str) -> bool:
    val = data.get(key)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    return False


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

def _cross_check_evidence_against_headline(
    evidence_data: dict,
    headline_rows: list[dict[str, str]],
    artifact_path: Path,
) -> list[PolicyIssue]:
    """Check that ``public_copy_allowed=true`` evidence has a matching headline claim."""
    issues: list[PolicyIssue] = []
    if not evidence_data.get("public_copy_allowed"):
        return issues

    product = evidence_data.get("product", "").strip().lower()
    title = evidence_data.get("title", "").strip().lower()
    command = evidence_data.get("command", "").strip().lower()
    path_str = str(artifact_path).strip().lower()

    matched = False
    for row in headline_rows:
        row_product = row.get("Product", "").strip().lower()
        row_source = row.get("Source", "").strip().lower()
        row_claim = row.get("Claim", "").strip().lower()

        if not row_product or _is_placeholder(row_product):
            continue

        if row_product != product:
            continue

        source_match = (
            title in row_source
            or command in row_source
            or path_str in row_source
        )
        if not source_match:
            # Also check if claim text overlaps with title
            if title and row_claim:
                if title in row_claim or row_claim in title:
                    source_match = True

        if source_match:
            matched = True
            break

    if not matched:
        issues.append(_make_warning(
            "no_matching_headline_claim", str(artifact_path),
            f"public_copy_allowed=true evidence for product '{product}' has no matching "
            f"active headline claim in HEADLINE-NUMBERS.md",
        ))

    return issues


# ---------------------------------------------------------------------------
# Markdown evidence
# ---------------------------------------------------------------------------

# Files under benchmarks/ that are documentation, not evidence artifacts.
# evidence-manifest.md is the generated index of this directory: it describes
# the evidence rather than being evidence, so it does not carry a block of its
# own (and must not, or generating it would invalidate it).
MD_NON_EVIDENCE: tuple[str, ...] = ("README.md", "evidence-manifest.md")
MD_NON_EVIDENCE_PREFIXES: tuple[str, ...] = ("RUNBOOK-",)

# Every Markdown evidence artifact must carry this block. It is an HTML comment
# so it does not render, and it is required rather than optional: evidence whose
# provenance cannot be read mechanically is evidence nobody can audit.
#
#   <!-- archolith-evidence
#   product: filter
#   command: archolith-bench filter
#   commit: 1aec8f3
#   run_date: 2026-05-30
#   source: results/filter_results.json
#   source_tracked: false
#   public_copy_allowed: false
#   -->
#
# `commit` and `run_date` accept the literal `unknown` for artifacts predating
# this convention, which keeps historical files honest instead of forcing a
# fabricated value. `unknown` is still disqualifying for public copy.
MD_EVIDENCE_BLOCK_RE = re.compile(
    r"<!--\s*archolith-evidence\s*\n(?P<body>.*?)-->", re.DOTALL
)

MD_REQUIRED_EVIDENCE_KEYS: tuple[str, ...] = (
    "product",
    "command",
    "commit",
    "run_date",
    "source",
    "source_tracked",
    "public_copy_allowed",
)

_MD_KV_RE = re.compile(r"^(?P<key>[a-z][a-z0-9_]*)\s*:\s*(?P<value>.*)$")
_TRUE_VALUES = {"true", "yes"}
_FALSE_VALUES = {"false", "no"}


def is_markdown_evidence(path: Path) -> bool:
    """Whether a .md file under benchmarks/ is an evidence artifact."""
    name = path.name
    if name in MD_NON_EVIDENCE:
        return False
    return not any(name.startswith(p) for p in MD_NON_EVIDENCE_PREFIXES)


def parse_md_evidence_block(text: str) -> dict[str, str] | None:
    """Parse the ``archolith-evidence`` block, or None when absent."""
    m = MD_EVIDENCE_BLOCK_RE.search(text)
    if not m:
        return None
    meta: dict[str, str] = {}
    for line in m.group("body").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kv = _MD_KV_RE.match(stripped)
        if kv:
            meta.setdefault(kv.group("key"), kv.group("value").strip())
    return meta


def validate_markdown_evidence(
    artifact_path: Path,
    *,
    repo_root: Path = POLICY_REPO_ROOT,
) -> PolicyResult:
    """Validate a Markdown evidence artifact against the required block.

    A missing or incomplete block is an ERROR, so the convention is enforced on
    every new artifact rather than merely encouraged. `unknown` provenance is
    accepted for historical files but never for public copy.

    Declared provenance is checked against git, not taken at its word. The
    checks are errors when public_copy_allowed=true and warnings otherwise:
    an unverifiable claim is disqualifying for public copy, but historical
    internal artifacts predate the convention and are reported, not broken.
    """
    result = PolicyResult(ok=True, summary={
        "artifact_path": str(artifact_path),
        "public_copy_allowed": False,
        "format": "markdown",
    })

    if not artifact_path.exists():
        result.ok = False
        result.errors.append(_make_error(
            "file_not_found", str(artifact_path), "Evidence file not found",
        ))
        return result

    try:
        text = artifact_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        result.ok = False
        result.errors.append(_make_error(
            "unreadable_markdown", str(artifact_path), f"Unreadable: {e}",
        ))
        return result

    meta = parse_md_evidence_block(text)
    if meta is None:
        result.ok = False
        result.errors.append(_make_error(
            "missing_evidence_block", str(artifact_path),
            "Markdown evidence must open with an <!-- archolith-evidence ... --> "
            "block declaring " + ", ".join(MD_REQUIRED_EVIDENCE_KEYS),
        ))
        return result

    result.summary["metadata_keys"] = sorted(meta)

    for key in MD_REQUIRED_EVIDENCE_KEYS:
        if not meta.get(key, "").strip():
            result.ok = False
            result.errors.append(_make_error(
                "missing_field", str(artifact_path),
                f"archolith-evidence block is missing required key '{key}'",
            ))

    pc_raw = meta.get("public_copy_allowed", "").strip().lower()
    if pc_raw and pc_raw not in _TRUE_VALUES | _FALSE_VALUES:
        result.ok = False
        result.errors.append(_make_error(
            "invalid_field", str(artifact_path),
            f"public_copy_allowed must be true or false, got {pc_raw!r}",
        ))
    pc_allowed = pc_raw in _TRUE_VALUES
    result.summary["public_copy_allowed"] = pc_allowed

    tracked_raw = meta.get("source_tracked", "").strip().lower()
    if tracked_raw and tracked_raw not in _TRUE_VALUES | _FALSE_VALUES:
        result.ok = False
        result.errors.append(_make_error(
            "invalid_field", str(artifact_path),
            f"source_tracked must be true or false, got {tracked_raw!r}",
        ))
    source_tracked = tracked_raw in _TRUE_VALUES

    commit = meta.get("commit", "").strip()
    run_date = meta.get("run_date", "").strip()
    commit_known = _is_commit_like(commit)
    date_known = _is_date_like(run_date)

    # A value that is neither valid nor the explicit `unknown` sentinel is a
    # typo, not a disclosure -- always an error.
    if commit and not commit_known and not _is_placeholder(commit):
        result.ok = False
        result.errors.append(_make_error(
            "invalid_field", str(artifact_path),
            f"commit must be a hex hash or 'unknown', got {commit!r}",
        ))
    if run_date and not date_known and not _is_placeholder(run_date):
        result.ok = False
        result.errors.append(_make_error(
            "invalid_field", str(artifact_path),
            f"run_date must be YYYY-MM-DD or 'unknown', got {run_date!r}",
        ))

    # A run cannot have happened in the future. This is a fabricated or
    # mistyped value regardless of who may quote it, so it is always an error.
    if date_known and _is_future_date(run_date):
        result.ok = False
        result.errors.append(_make_error(
            "future_run_date", str(artifact_path),
            f"run_date {run_date} is in the future; a run cannot postdate today",
        ))

    if pc_allowed:
        if not commit_known:
            result.ok = False
            result.errors.append(_make_error(
                "missing_provenance", str(artifact_path),
                "public_copy_allowed=true requires a real commit, not 'unknown'",
            ))
        if not date_known:
            result.ok = False
            result.errors.append(_make_error(
                "missing_provenance", str(artifact_path),
                "public_copy_allowed=true requires a real run_date, not 'unknown'",
            ))
        if not source_tracked:
            result.ok = False
            result.errors.append(_make_error(
                "untracked_public_source", str(artifact_path),
                "public_copy_allowed=true but source_tracked=false: a public claim "
                "cannot rest on data that is not in the repository",
            ))

    # Verify the declarations rather than trusting them. `verified is None`
    # means git could not answer; for public copy that is a refusal, because
    # an unverifiable claim is exactly what this gate exists to stop.
    source = meta.get("source", "").strip()

    if commit_known:
        verified = commit_exists(commit, repo_root)
        if verified is False:
            _add_provenance_issue(
                result, pc_allowed, "unverifiable_commit", artifact_path,
                f"commit {commit} does not exist in this repository",
            )
        elif verified is None:
            _add_provenance_issue(
                result, pc_allowed, "unverified_commit", artifact_path,
                f"commit {commit} could not be checked (git unavailable)",
            )

    if source_tracked:
        verified = source_is_tracked(source, repo_root)
        if verified is False:
            _add_provenance_issue(
                result, pc_allowed, "untracked_source", artifact_path,
                f"source_tracked=true but {source!r} is not a git-tracked path",
            )
        elif verified is None:
            _add_provenance_issue(
                result, pc_allowed, "unverified_source", artifact_path,
                f"source_tracked=true but {source!r} could not be checked "
                "(git unavailable)",
            )

    if not pc_allowed:
        missing = [
            label for label, known in (("commit", commit_known), ("run_date", date_known))
            if not known
        ]
        if missing:
            result.warnings.append(_make_warning(
                "incomplete_provenance", str(artifact_path),
                "provenance is incomplete (" + ", ".join(missing)
                + "); this artifact cannot be promoted to a headline claim as written",
            ))
        if not source_tracked:
            result.warnings.append(_make_warning(
                "untracked_raw_source", str(artifact_path),
                "source_tracked=false: the underlying data is not in the repository",
            ))

    return result


# ---------------------------------------------------------------------------
# Aggregate validator
# ---------------------------------------------------------------------------


def validate_policy(
    headline_path: Path,
    evidence_paths: list[Path],
) -> PolicyResult:
    """Run all policy validations and return an aggregate result."""
    headline_result = validate_headline_numbers(headline_path)

    all_errors: list[PolicyIssue] = list(headline_result.errors)
    all_warnings: list[PolicyIssue] = list(headline_result.warnings)
    evidence_checked = 0
    public_copy_allowed = 0
    public_copy_rejected = 0

    headline_rows: list[dict[str, str]] = []
    if headline_path.exists():
        headline_rows = _parse_headline_table(headline_path.read_text(encoding="utf-8"))

    for ep in evidence_paths:
        suffix = ep.suffix.lower()
        if suffix == ".json":
            art_result = validate_evidence_artifact(ep)
        elif suffix == ".md":
            # README/RUNBOOK files live alongside evidence but are documentation.
            if not is_markdown_evidence(ep):
                continue
            art_result = validate_markdown_evidence(ep)
        else:
            continue

        evidence_checked += 1
        all_errors.extend(art_result.errors)
        all_warnings.extend(art_result.warnings)

        if art_result.summary.get("public_copy_allowed"):
            public_copy_allowed += 1
            # Cross-check against the headline table. The checker reads dict
            # fields, so Markdown supplies its evidence block rather than
            # being parsed as JSON.
            try:
                if suffix == ".md":
                    data = parse_md_evidence_block(ep.read_text(encoding="utf-8")) or {}
                else:
                    data = json.loads(ep.read_text(encoding="utf-8"))
                cross_issues = _cross_check_evidence_against_headline(data, headline_rows, ep)
                all_warnings.extend(cross_issues)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                pass
        else:
            public_copy_rejected += 1

    ok = len(all_errors) == 0

    return PolicyResult(
        ok=ok,
        errors=all_errors,
        warnings=all_warnings,
        summary={
            "headline_active_claims": headline_result.summary.get("headline_active_claims", 0),
            "evidence_files_checked": evidence_checked,
            "public_copy_allowed": public_copy_allowed,
            "public_copy_rejected": public_copy_rejected,
        },
    )


# ---------------------------------------------------------------------------
# Evidence manifest
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = (
    "Artifact", "Product", "Command", "Commit", "Run date", "Source tracked",
    "Public copy", "Status",
)


def _manifest_row(path: Path) -> dict[str, str]:
    """Build one manifest row from an artifact's own declared provenance."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        result = validate_markdown_evidence(path)
        meta = parse_md_evidence_block(path.read_text(encoding="utf-8")) or {}
    else:
        result = validate_evidence_artifact(path)
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            meta = {}

    def cell(key: str) -> str:
        value = meta.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return "-"
        return str(value).strip().replace("|", "\\|")

    if not result.ok:
        status = "INVALID"
    elif result.warnings:
        status = "incomplete"
    else:
        status = "ok"

    run_date = cell("run_date")
    if run_date == "-":
        run_date = cell("timestamp")

    return {
        "Artifact": path.name,
        "Product": cell("product"),
        "Command": f"`{cell('command')}`" if cell("command") != "-" else "-",
        "Commit": cell("commit")[:12],
        "Run date": run_date[:10],
        "Source tracked": cell("source_tracked"),
        "Public copy": cell("public_copy_allowed"),
        "Status": status,
    }


def build_evidence_manifest(evidence_paths: list[Path]) -> str:
    """Render a Markdown index of tracked evidence.

    Generated from each artifact's own declared provenance, so it cannot drift
    from the artifacts the way a hand-maintained manifest would. Regenerate it
    rather than editing it.
    """
    rows: list[dict[str, str]] = []
    for ep in sorted(evidence_paths, key=lambda p: p.name):
        suffix = ep.suffix.lower()
        if suffix not in (".md", ".json"):
            continue
        if suffix == ".md" and not is_markdown_evidence(ep):
            continue
        if not ep.exists():
            continue
        rows.append(_manifest_row(ep))

    lines = [
        "# Evidence Manifest",
        "",
        "Generated index of tracked benchmark evidence. **Do not edit by hand** --",
        "regenerate with:",
        "",
        "```sh",
        "python scripts/check_evidence_policy.py --evidence-dir benchmarks/ \\",
        "    --manifest benchmarks/evidence-manifest.md",
        "```",
        "",
        "Every column is read from the artifact's own provenance block, so this file",
        "cannot drift from what the artifacts declare. `Status` is the validator's",
        "verdict: `ok`, `incomplete` (valid block, unknown provenance -- not promotable",
        "to a headline claim), or `INVALID` (fails the evidence policy).",
        "",
        "| " + " | ".join(MANIFEST_COLUMNS) + " |",
        "|" + "|".join("---" for _ in MANIFEST_COLUMNS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[c] for c in MANIFEST_COLUMNS) + " |")

    # Case-insensitive: a JSON artifact's public_copy_allowed is a bool, which
    # str() renders as "True", so an exact "true" match never counted one.
    promotable = [
        r for r in rows
        if r["Status"] == "ok" and r["Public copy"].strip().lower() == "true"
    ]
    lines += [
        "",
        f"{len(rows)} artifact(s) indexed; {len(promotable)} currently quotable as public copy.",
        "",
    ]
    return "\n".join(lines)
