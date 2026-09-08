"""Offline tests for the public-claim scanner (no network, no model calls)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from archolith_bench.core.public_claims import (
    DetectedClaim,
    is_claim_allowed,
    load_active_headline_claims,
    scan_file_for_claims,
    scan_public_claims,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "public_claims"


# ===================================================================
# load_active_headline_claims
# ===================================================================

class TestLoadActiveHeadlineClaims:
    def test_none_placeholder_returns_empty(self):
        path = FIXTURES / "headline_numbers_none.md"
        claims = load_active_headline_claims(path)
        assert claims == []

    def test_valid_claims_loaded(self):
        path = FIXTURES / "headline_numbers_with_claim.md"
        claims = load_active_headline_claims(path)
        assert len(claims) == 3
        assert claims[0].product == "archolith-context"
        assert claims[0].value == "75%"
        assert claims[1].product == "archolith-filter"
        assert claims[1].value == "60%"

    def test_missing_file_returns_empty(self):
        claims = load_active_headline_claims(Path("/nonexistent"))
        assert claims == []


# ===================================================================
# scan_file_for_claims
# ===================================================================

class TestScanFileForClaims:
    def test_detects_percentage_claims(self):
        path = FIXTURES / "README_with_unapproved_claim.md"
        claims = scan_file_for_claims(path)
        texts = [c.text for c in claims]
        assert any("75%" in t for t in texts)
        assert any("40.5%" in t for t in texts)
        assert any("0.367" in t for t in texts)

    def test_missing_file_returns_empty(self):
        claims = scan_file_for_claims(Path("/nonexistent"))
        assert claims == []


# ===================================================================
# is_claim_allowed
# ===================================================================

class TestIsClaimAllowed:
    def _make_claim(self, text: str, line: int = 1) -> DetectedClaim:
        return DetectedClaim(path="test.md", line=line, text=text, reason="test")

    def _load_claims(self) -> list:
        return load_active_headline_claims(FIXTURES / "headline_numbers_with_claim.md")

    def test_approved_75pct_allowed(self):
        claims = self._load_claims()
        dc = self._make_claim("75%")
        lines = ["Archolith-context Token savings on single-turn code review 75%"]
        assert is_claim_allowed(dc, claims, lines) is True

    def test_approved_60pct_allowed(self):
        claims = self._load_claims()
        dc = self._make_claim("60%")
        lines = ["archolith-filter Compression on tool-output corpora 60%"]
        assert is_claim_allowed(dc, claims, lines) is True

    def test_approved_0850_allowed(self):
        claims = self._load_claims()
        dc = self._make_claim("0.850")
        lines = ["archolith-context RULER accuracy 0.850"]
        assert is_claim_allowed(dc, claims, lines) is True

    def test_unrelated_number_not_allowed(self):
        claims = self._load_claims()
        dc = self._make_claim("99%")
        lines = ["Some random 99% number with no relation"]
        assert is_claim_allowed(dc, claims, lines) is False

    def test_no_active_claims_returns_false(self):
        dc = self._make_claim("75%")
        assert is_claim_allowed(dc, []) is False

    def test_unapproved_40pct_not_allowed(self):
        claims = self._load_claims()
        dc = self._make_claim("40.5%")
        lines = ["archolith-filter compression 40.5%"]
        assert is_claim_allowed(dc, claims, lines) is False


# ===================================================================
# scan_public_claims — integration
# ===================================================================

class TestScanPublicClaims:
    def test_no_active_claims_no_public_numbers_passes(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nNo numbers here.\n")
        result = scan_public_claims(hl, [readme])
        assert result.ok is True
        assert result.claims_detected == 0

    def test_no_active_claims_readme_claim_fails(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nReduces tokens by 75%.\n")
        result = scan_public_claims(hl, [readme])
        assert result.ok is False
        assert len(result.unapproved_claims) >= 1

    def test_active_claim_allows_matching(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("""## Active Headline Numbers
| Product | Claim | Value | Source | Commit | Run date | Notes |
|---------|-------|-------|--------|--------|----------|-------|
| archolith-context | Token savings | 75% | proxy run | a1b2c3d | 2026-07-01 | Real run. |
""")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nArcholith-context reduces tokens by 75%.\n")
        result = scan_public_claims(hl, [readme])
        assert result.ok is True, [c.text for c in result.unapproved_claims]

    def test_active_claim_does_not_allow_unrelated(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("""## Active Headline Numbers
| Product | Claim | Value | Source | Commit | Run date | Notes |
|---------|-------|-------|--------|--------|----------|-------|
| archolith-context | Token savings | 75% | proxy run | a1b2c3d | 2026-07-01 | Real run. |
""")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nOur filter achieves 40.5% compression.\n")
        result = scan_public_claims(hl, [readme])
        assert result.ok is False
        assert any("40.5%" in c.text for c in result.unapproved_claims)

    def test_internal_excluded_paths_ignored(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        benchmarks = tmp_path / "benchmarks"
        benchmarks.mkdir()
        (benchmarks / "result.json").write_text('{"score": 0.9}')
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nNo numbers.\n")
        result = scan_public_claims(hl, [tmp_path])
        assert result.files_scanned >= 1  # README discovered
        # benchmarks/ result should be excluded
        bench_claims = [c for c in result.unapproved_claims if "benchmarks" in c.path]
        assert len(bench_claims) == 0

    def test_explicit_scan_excluded_path(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        results_file = tmp_path / "results" / "out.json"
        results_file.parent.mkdir()
        results_file.write_text('score 50')
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nNo numbers.\n")
        result = scan_public_claims(hl, [readme, results_file])
        assert result.files_scanned >= 2
        assert any("out.json" in c.path for c in result.unapproved_claims)

    def test_ignore_next_line_suppresses(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text(
            "This example says 75% but is not a public claim.\n"
            "<!-- archolith-claim-scan: ignore-next-line -->\n"
            "This line has a 66% token savings claim.\n"
            "Unapproved 80% claim.\n"
        )
        result = scan_public_claims(hl, [readme])
        unapproved_texts = [c.text for c in result.unapproved_claims]
        # The pragma covers only the line after it: 66% is suppressed, while
        # the 75% above it and the 80% below it are still reported. The
        # percentages must stay distinct -- the earlier version of this test
        # used 75% on both lines, so it passed whether or not the pragma
        # suppressed anything.
        assert not any("66%" in t for t in unapproved_texts)
        assert any("75%" in t for t in unapproved_texts)
        assert any("80%" in t for t in unapproved_texts)

    def test_ignore_start_end_suppresses_block(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Project\n"
            "<!-- archolith-claim-scan: ignore-start -->\n"
            "This block has 40.5% compression numbers.\n"
            "And 0.367 accuracy score.\n"
            "<!-- archolith-claim-scan: ignore-end -->\n"
            "Unapproved 85% claim.\n"
        )
        result = scan_public_claims(hl, [readme])
        unapproved_texts = [c.text for c in result.unapproved_claims]
        assert any("85%" in t for t in unapproved_texts), unapproved_texts
        # 40.5% and 0.367 are inside the ignore block
        inside = [t for t in unapproved_texts if "40.5%" in t or "0.367" in t]
        assert len(inside) == 0, inside

    def test_json_output_parseable(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_public_claims",
             "--headline", str(hl), "--scan", str(readme), "--json"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "ok" in data
        assert "files_scanned" in data
        assert "unapproved_claims" in data

    def test_human_output_includes_file_and_line(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\n75% token savings.\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_public_claims",
             "--headline", str(hl), "--scan", str(readme)],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 1
        assert "README.md" in result.stdout
        assert "75%" in result.stdout
        assert ":1" in result.stdout or ":2" in result.stdout

    def test_benchmarks_not_scanned_by_default(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        readme = tmp_path / "README.md"
        readme.write_text("# Project\nNo numbers.\n")
        bench_dir = tmp_path / "benchmarks"
        bench_dir.mkdir()
        (bench_dir / "evidence.json").write_text('{"score": 90}')
        result = scan_public_claims(hl, [tmp_path])
        bench_claims = [c for c in result.unapproved_claims if "benchmarks" in c.path]
        assert len(bench_claims) == 0

    def test_missing_optional_paths_no_crash(self):
        result = scan_public_claims(
            FIXTURES / "headline_numbers_none.md",
            [Path("/nonexistent/path.md"), Path("/nonexistent/dir/")],
        )
        assert result.files_scanned == 0
        assert result.ok is True


# ===================================================================
# CLI
# ===================================================================

class TestCli:
    def test_exit_code_2_on_missing_headline(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_public_claims",
             "--headline", str(tmp_path / "nonexistent.md")],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 2
        assert "ERROR" in result.stderr

    def test_default_scan_no_crash(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_public_claims",
             "--headline", str(FIXTURES / "headline_numbers_none.md")],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        # May or may not find claims in the real repo docs — just check it doesn't crash
        assert result.returncode in (0, 1)
