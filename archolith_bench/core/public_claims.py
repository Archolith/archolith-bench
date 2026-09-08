"""Offline public-claim scanner for archolith-bench.

Detects benchmark/statistic claims in public-facing docs and cross-checks
them against active rows in HEADLINE-NUMBERS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .evidence_policy import _parse_headline_table


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HeadlineClaim:
    product: str
    claim: str
    value: str
    source: str
    commit: str
    run_date: str
    notes: str


@dataclass
class DetectedClaim:
    path: str
    line: int
    text: str
    reason: str


@dataclass
class ClaimScanResult:
    ok: bool
    files_scanned: int
    claims_detected: int
    unapproved_claims: list[DetectedClaim] = field(default_factory=list)
    ignored_claims: list[DetectedClaim] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_SCAN_PATHS: tuple[str, ...] = (
    "README.md",
    "BENCHMARKS.md",
    "docs/",
    # Public-facing prose that carries figures and already writes ignore-*
    # pragmas. Outside the scan those pragmas were decorative: they suppressed
    # nothing, because nothing looked.
    "METHODOLOGY.md",
    # The card file only. `corpora/` itself is raw benchmark INPUT -- captured
    # pytest output and diffs, full of percentages that are data, not claims.
    "corpora/CORPUS-CARDS.md",
)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".agent/",
    "results/",
    "fixtures/",
    "tests/",
    "scripts/",
    "__pycache__/",
    "benchmarks/",
)


# ---------------------------------------------------------------------------
# Claim detection patterns
# ---------------------------------------------------------------------------

# Percentage: "75%", "40.5%", "~60%"
_PCT_RE = re.compile(r"(?:\b(?:about|around|~|approximately|over|under)\s+)?\d+(?:\.\d+)?\s*%")

# Score: "score 0.367", "accuracy 0.82", "pass@1 45%", "F1 0.91"
_SCORE_RE = re.compile(
    r"(?:score|accuracy|pass@\d+|f1|precision|recall|exact_match|em|bleu|rouge)\s+"
    r"(?:of\s+)?\d+(?:\.\d+)?\s*%?"
    r"|"
    r"\d+\.\d{2,}\s+(?:score|accuracy|f1|precision|recall)"
)

# Comparative language near numbers
_COMPARE_RE = re.compile(
    r"(?:beats?|improves?|reduces?|saves?|cuts?|lowers?|exceeds?|outperforms?|"
    r"increases?|decreases?|reduction|savings|gain|boost)\s+"
    r"(?:by\s+)?\d+(?:\.\d+)?\s*%?"
)

# Benchmark-specific signals: benchmark name near a number
_BENCHMARK_NAMES = (
    "ruler", "longbench", "swe-bench", "helm", "agentdojo",
    "cyberseceval", "dmr", "mteb", "longmemeval", "bigcodebench",
    "humaneval", "gsm8k", "mmlu", "bbh", "drop", "nq", "triviaqa",
    "hotpotqa", "2wiki", "musique",
)

_BENCH_RE = re.compile(
    r"(?:" + "|".join(_BENCHMARK_NAMES) + r")\s*[-:;]?\s*\d+(?:\.\d+)?\s*%?"
    r"|"
    r"\d+(?:\.\d+)?\s*%?\s+(?:on|in|for)\s+(?:" + "|".join(_BENCHMARK_NAMES) + r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Comment-based ignore markers
# ---------------------------------------------------------------------------

_IGNORE_NEXT_LINE_RE = re.compile(r"<!--\s*archolith-claim-scan:\s*ignore-next-line\s*-->")
_IGNORE_START_RE = re.compile(r"<!--\s*archolith-claim-scan:\s*ignore-start\s*-->")
_IGNORE_END_RE = re.compile(r"<!--\s*archolith-claim-scan:\s*ignore-end\s*-->")


# ---------------------------------------------------------------------------
# Claim detection
# ---------------------------------------------------------------------------

def _detect_claims_in_line(line: str) -> list[str]:
    """Return a list of distinct claim snippets found in *line*."""
    claims: list[str] = []
    for m in _PCT_RE.finditer(line):
        claims.append(m.group().strip())
    for m in _SCORE_RE.finditer(line):
        claims.append(m.group().strip())
    for m in _COMPARE_RE.finditer(line):
        claims.append(m.group().strip())
    for m in _BENCH_RE.finditer(line):
        claims.append(m.group().strip())
    return claims


def scan_file_for_claims(path: Path) -> list[DetectedClaim]:
    """Scan a single file for unignored benchmark/statistic claims.

    Respects ``<!-- archolith-claim-scan: ignore-* -->`` comments.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    lines = text.splitlines()
    detected: list[DetectedClaim] = []
    ignore_block = False
    skip_this_line = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Consumed here, but it does not short-circuit the pragma handling
        # below: an ignore-next-line sitting immediately above an ignore-start
        # used to swallow the block marker, leaving the whole block scanned.
        covered_by_previous = skip_this_line
        skip_this_line = False

        is_start = bool(_IGNORE_START_RE.search(stripped))
        is_end = bool(_IGNORE_END_RE.search(stripped))
        is_next = bool(_IGNORE_NEXT_LINE_RE.search(stripped))

        was_in_block = ignore_block
        if is_start:
            ignore_block = True
        if is_end:
            ignore_block = False
        if is_next:
            # Re-armed even on a line the previous pragma covered, so
            # consecutive ignore-next-line pragmas chain instead of cancelling.
            skip_this_line = True

        if covered_by_previous:
            reason = "ignored (archolith-claim-scan: ignore-next-line)"
        elif is_start or is_end or is_next:
            # A pragma sharing a line with a claim used to drop that claim from
            # BOTH lists, making the suppression invisible. Suppressions have to
            # stay auditable, so it is recorded rather than discarded.
            reason = "ignored (archolith-claim-scan: pragma on claim line)"
        elif was_in_block:
            reason = "ignored (archolith-claim-scan: ignore-start/ignore-end block)"
        else:
            reason = ""

        for claim_text in _detect_claims_in_line(line):
            detected.append(DetectedClaim(
                path=str(path), line=i + 1,
                text=claim_text,
                reason=reason or "potential benchmark/statistic claim",
            ))

    ignored = [c for c in detected if c.reason.startswith("ignored")]
    unignored = [c for c in detected if not c.reason.startswith("ignored")]
    return unignored + ignored


# ---------------------------------------------------------------------------
# Load active headline claims
# ---------------------------------------------------------------------------

def load_active_headline_claims(headline_path: Path) -> list[HeadlineClaim]:
    """Parse active headline rows from *HEADLINE-NUMBERS.md*."""
    if not headline_path.exists():
        return []

    text = headline_path.read_text(encoding="utf-8")
    rows = _parse_headline_table(text)

    if not rows:
        return []

    first = rows[0]
    product_val = first.get("Product", "").strip().lower()
    if product_val in ("_none_", "none"):
        return []

    claims: list[HeadlineClaim] = []
    for row in rows:
        p = row.get("Product", "").strip()
        if not p or p.lower() in ("_none_", "none"):
            continue
        claims.append(HeadlineClaim(
            product=p,
            claim=row.get("Claim", "").strip(),
            value=row.get("Value", "").strip(),
            source=row.get("Source", "").strip(),
            commit=row.get("Commit", "").strip(),
            run_date=row.get("Run date", "").strip(),
            notes=row.get("Notes", "").strip(),
        ))
    return claims


# ---------------------------------------------------------------------------
# Allowance check
# ---------------------------------------------------------------------------

def _get_surrounding_text(lines: list[str], line_idx: int, window: int = 5) -> str:
    start = max(0, line_idx - window)
    end = min(len(lines), line_idx + window + 1)
    return " ".join(lines[start:end]).lower()


def _product_matches(text: str, product: str) -> bool:
    """Check if *text* mentions the product, with partial prefix matching."""
    text_lower = text.lower()
    product_lower = product.lower()
    if product_lower in text_lower:
        return True
    # Partial match: "archolith-context" also matches plain "archolith"
    parts = product_lower.split("-")
    for i in range(1, len(parts) + 1):
        prefix = "-".join(parts[:i])
        if prefix in text_lower:
            return True
    return False


def is_claim_allowed(
    claim: DetectedClaim,
    active_claims: list[HeadlineClaim],
    file_lines: list[str] | None = None,
) -> bool:
    """Check if a detected claim is backed by an active headline claim.

    V1 matching: the detected claim text must appear in the value of an
    active claim row, AND the surrounding text must contain the row's
    product or claim phrase.
    """
    if not active_claims:
        return False

    claim_text_lower = claim.text.lower()

    surrounding = ""
    if file_lines is not None:
        surrounding = _get_surrounding_text(file_lines, claim.line - 1)

    for ac in active_claims:
        value_lower = ac.value.lower()
        claim_phrase_lower = ac.claim.lower()

        # Check if claim text overlaps with active row value
        value_ok = (
            value_lower in claim_text_lower
            or claim_text_lower in value_lower
            or _values_overlap(claim_text_lower, value_lower)
        )
        if not value_ok:
            continue

        # Check surrounding text for product or claim phrase
        if not surrounding:
            return True

        if _product_matches(surrounding, ac.product):
            return True

        if claim_phrase_lower in surrounding:
            return True

        # Also check the claim.line text itself
        if file_lines and claim.line - 1 < len(file_lines):
            line_text = file_lines[claim.line - 1].lower()
            if _product_matches(line_text, ac.product):
                return True
            if claim_phrase_lower in line_text:
                return True

    return False


def _values_overlap(a: str, b: str) -> bool:
    """Check if two value strings share a numeric token."""
    a_tokens = {t for t in a.split() if any(c.isdigit() for c in t)}
    b_tokens = {t for t in b.split() if any(c.isdigit() for c in t)}
    return bool(a_tokens & b_tokens)


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

def _is_excluded(rel_path: str, excludes: tuple[str, ...]) -> bool:
    for ex in excludes:
        if ex.endswith("/"):
            if rel_path.startswith(ex) or f"/{ex}" in rel_path:
                return True
        elif rel_path == ex or rel_path.startswith(ex + "/"):
            return True
    return False


def _collect_scan_paths(
    scan_targets: list[Path],
    excludes: tuple[str, ...],
    repo_root: Path,
) -> list[Path]:
    paths: list[Path] = []
    for target in scan_targets:
        if not target.exists():
            continue
        if target.is_file():
            paths.append(target)
        elif target.is_dir():
            for fpath in sorted(target.rglob("*")):
                if not fpath.is_file():
                    continue
                try:
                    rel = fpath.relative_to(repo_root).as_posix()
                except ValueError:
                    rel = fpath.name
                if _is_excluded(rel, excludes):
                    continue
                paths.append(fpath)
    return paths


DEFAULT_IGNORED_PATHS: tuple[str, ...] = (
    ".agent/",
    "results/",
    "fixtures/",
    "tests/",
    "scripts/",
    "__pycache__/",
    "benchmarks/",
    ".git/",
    ".github/",
)


def scan_public_claims(
    headline_path: Path,
    scan_paths: list[Path],
    *,
    repo_root: Path | None = None,
    extra_excludes: tuple[str, ...] = (),
) -> ClaimScanResult:
    """Scan public-facing docs for unapproved benchmark claims."""
    if repo_root is None:
        repo_root = Path.cwd()

    excludes = DEFAULT_IGNORED_PATHS + extra_excludes

    active_claims = load_active_headline_claims(headline_path)
    files = _collect_scan_paths(scan_paths, excludes, repo_root)

    unapproved: list[DetectedClaim] = []
    ignored: list[DetectedClaim] = []
    total_detected = 0

    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        file_lines = text.splitlines()

        raw_claims = scan_file_for_claims(fp)

        for dc in raw_claims:
            total_detected += 1
            if dc.reason.startswith("ignored"):
                ignored.append(dc)
                continue
            if is_claim_allowed(dc, active_claims, file_lines):
                continue
            unapproved.append(dc)

    return ClaimScanResult(
        ok=len(unapproved) == 0,
        files_scanned=len(files),
        claims_detected=total_detected,
        unapproved_claims=unapproved,
        ignored_claims=ignored,
    )
