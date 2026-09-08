"""Offline tests for the evidence policy validator (no network, no model calls)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from archolith_bench.core.evidence_policy import (
    _contains_rejected_term,
    _is_commit_like,
    _is_date_like,
    _is_placeholder,
    _parse_headline_table,
    MANIFEST_COLUMNS,
    build_evidence_manifest,
    count_md_evidence_blocks,
    is_markdown_evidence,
    parse_md_evidence_block,
    validate_evidence_artifact,
    validate_headline_numbers,
    validate_markdown_evidence,
    validate_policy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _head_commit() -> str:
    """HEAD of this repo -- a commit that provably exists, for honest-path tests."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        pytest.skip("git unavailable")
    return out.stdout.strip()


COMPLETE_MD = """# Example Evidence

<!-- archolith-evidence
product: archolith-filter
command: archolith-bench filter
commit: 1aec8f3
run_date: 2026-05-30
source: results/filter_results.json
source_tracked: true
public_copy_allowed: false
-->

Body text.
"""

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "evidence_policy"


# ===================================================================
# Unit: helpers
# ===================================================================

class TestHelpers:
    def test_is_commit_like(self):
        assert _is_commit_like("a1b2c3d")
        assert _is_commit_like("abcdef1234567890abcdef1234567890abcdef12")
        assert not _is_commit_like("")
        assert not _is_commit_like("unknown")
        assert not _is_commit_like("TBD")
        assert not _is_commit_like("not a commit")

    def test_is_date_like(self):
        assert _is_date_like("2026-07-01")
        assert _is_date_like("2024-01-31")
        assert not _is_date_like("")
        assert not _is_date_like("2026-13-01")
        assert not _is_date_like("pending")
        assert not _is_date_like("07-01-2026")

    def test_is_placeholder(self):
        assert _is_placeholder("_none_")
        assert _is_placeholder("pending")
        assert _is_placeholder("none")
        assert _is_placeholder("TBD")
        assert _is_placeholder("unknown")
        assert not _is_placeholder("real value")
        assert not _is_placeholder("")

    def test_contains_rejected_term(self):
        assert _contains_rejected_term("this is a fixture result")
        assert _contains_rejected_term("sample data only")
        assert _contains_rejected_term("demo run")
        assert _contains_rejected_term("historical evidence")
        assert _contains_rejected_term("candidate numbers")
        assert _contains_rejected_term("pending review")
        assert _contains_rejected_term("not for copy")
        assert _contains_rejected_term("internal only")
        assert _contains_rejected_term("smoke only test")
        assert _contains_rejected_term("offline stub result")
        assert not _contains_rejected_term("Live run against production")
        assert not _contains_rejected_term("Real benchmark output")

    def test_parse_headline_table_no_section(self):
        assert _parse_headline_table("no table here") == []

    def test_parse_headline_table_none_placeholder(self):
        text = """## Active Headline Numbers
| Product | Claim |
|---------|-------|
| _none_ | _placeholder_ |
"""
        rows = _parse_headline_table(text)
        assert len(rows) == 1
        # Parser strips surrounding underscores
        assert rows[0]["Product"] == "none"


# ===================================================================
# HEADLINE-NUMBERS.md validation
# ===================================================================

class TestValidateHeadlineNumbers:
    def test_no_claims_placeholder_passes(self):
        path = FIXTURES / "headline_numbers_no_claims.md"
        result = validate_headline_numbers(path)
        assert result.ok is True
        assert result.summary["headline_active_claims"] == 0

    def test_valid_active_claim_passes(self):
        path = FIXTURES / "headline_numbers_valid_claim.md"
        result = validate_headline_numbers(path)
        assert result.ok is True
        assert result.summary["headline_active_claims"] == 1

    def test_active_claim_missing_commit_fails(self):
        path = FIXTURES / "headline_numbers_invalid_claim.md"
        result = validate_headline_numbers(path)
        assert result.ok is False
        assert any(e.code == "missing_commit" for e in result.errors), (
            [e.code for e in result.errors]
        )

    def test_active_claim_invalid_run_date_fails(self):
        path = FIXTURES / "headline_numbers_invalid_claim.md"
        result = validate_headline_numbers(path)
        assert result.ok is False
        assert any(e.code == "invalid_run_date" for e in result.errors), (
            [e.code for e in result.errors]
        )

    def test_active_claim_with_fixture_language_fails(self):
        path = FIXTURES / "headline_numbers_invalid_claim.md"
        result = validate_headline_numbers(path)
        assert result.ok is False
        assert any(e.code == "rejected_term_in_active_claim" for e in result.errors), (
            [e.code for e in result.errors]
        )

    def test_retired_section_with_fixture_language_does_not_fail(self):
        """Retired section uses 'fixture' and 'sample' — those should not trigger errors."""
        path = FIXTURES / "headline_numbers_valid_claim.md"
        result = validate_headline_numbers(path)
        # The retired section says 'fixture' and 'historical' but only active rows are checked.
        assert result.ok is True

    def test_file_not_found(self):
        result = validate_headline_numbers(Path("/nonexistent/HEADLINE-NUMBERS.md"))
        assert result.ok is False
        assert any(e.code == "file_not_found" for e in result.errors)


# ===================================================================
# Evidence artifact validation
# ===================================================================

class TestValidateEvidenceArtifact:
    def test_public_copy_false_missing_fields_gives_warnings(self):
        path = FIXTURES / "evidence_missing_provenance.json"
        result = validate_evidence_artifact(path)
        assert result.ok is True
        assert len(result.warnings) > 0
        assert any(w.code == "missing_field" for w in result.warnings)

    def test_public_copy_true_full_provenance_passes(self):
        path = FIXTURES / "evidence_valid_public.json"
        result = validate_evidence_artifact(path)
        assert result.ok is True, [e.message for e in result.errors]

    def test_public_copy_true_fixture_source_fails(self):
        path = FIXTURES / "evidence_fixture_public_invalid.json"
        result = validate_evidence_artifact(path)
        assert result.ok is False
        assert any(e.code == "fixture_source_rejected" for e in result.errors), (
            [e.code for e in result.errors]
        )

    def test_public_copy_true_empty_caveats_fails(self):
        data = {
            "title": "Test",
            "command": "bench",
            "commit": "a1b2c3d",
            "product": "test",
            "ability": "test",
            "fixture_or_live_source": "Live run",
            "model_provider": "gpt-4o-mini",
            "environment_caveats": [],
            "public_copy_allowed": True,
            "metric_rows": [{"n": 1}],
            "artifact": {},
        }
        p = _write_tmp_json(data)
        result = validate_evidence_artifact(p)
        assert result.ok is False
        assert any(e.code == "empty_caveats" for e in result.errors)
        p.unlink()

    def test_public_copy_true_empty_metric_rows_fails(self):
        data = {
            "title": "Test",
            "command": "bench",
            "commit": "a1b2c3d",
            "product": "test",
            "ability": "test",
            "fixture_or_live_source": "Live run",
            "model_provider": "gpt-4o-mini",
            "environment_caveats": ["One caveat"],
            "public_copy_allowed": True,
            "metric_rows": [],
            "artifact": {},
        }
        p = _write_tmp_json(data)
        result = validate_evidence_artifact(p)
        assert result.ok is False
        assert any(e.code == "empty_metric_rows" for e in result.errors)
        p.unlink()

    def test_malformed_json(self):
        p = _write_tmp_str("not json")
        result = validate_evidence_artifact(p)
        assert result.ok is False
        assert any(e.code == "malformed_json" for e in result.errors)
        p.unlink()

    def test_non_dict_json(self):
        p = _write_tmp_str('["list", "not", "dict"]')
        result = validate_evidence_artifact(p)
        assert result.ok is False
        assert any(e.code == "malformed_json" for e in result.errors)
        p.unlink()

    def test_file_not_found(self):
        result = validate_evidence_artifact(Path("/nonexistent/evidence.json"))
        assert result.ok is False
        assert any(e.code == "file_not_found" for e in result.errors)


# ===================================================================
# Cross-check
# ===================================================================

class TestCrossCheck:
    def test_public_copy_true_no_matching_headline_warns(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("""## Active Headline Numbers
| Product | Claim | Value | Source | Commit | Run date | Notes |
|---------|-------|-------|--------|--------|----------|-------|
| archolith-filter | compression | 50% | filter suite run | a1b2c3d | 2026-07-01 | Real run. |
""")
        ev = tmp_path / "evidence.json"
        ev.write_text(json.dumps({
            "title": "Proxy suite evidence",
            "command": "archolith-bench proxy",
            "commit": "a1b2c3d",
            "product": "archolith-context",
            "ability": "curation",
            "fixture_or_live_source": "Live run",
            "model_provider": "gpt-4o-mini",
            "environment_caveats": ["One caveat"],
            "public_copy_allowed": True,
            "metric_rows": [{"n": 1}],
            "artifact": {},
        }))
        result = validate_policy(hl, [ev])
        cross_warnings = [w for w in result.warnings if w.code == "no_matching_headline_claim"]
        assert len(cross_warnings) == 1, [w.message for w in result.warnings]

    def test_public_copy_true_matching_headline_ok(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("""## Active Headline Numbers
| Product | Claim | Value | Source | Commit | Run date | Notes |
|---------|-------|-------|--------|--------|----------|-------|
| archolith-context | token savings | 60% | Proxy suite: `archolith-bench proxy` | a1b2c3d | 2026-07-01 | Real run. |
""")
        ev = tmp_path / "evidence.json"
        ev.write_text(json.dumps({
            "title": "Proxy suite evidence",
            "command": "archolith-bench proxy",
            "commit": "a1b2c3d",
            "product": "archolith-context",
            "ability": "curation",
            "fixture_or_live_source": "Live run",
            "model_provider": "gpt-4o-mini",
            "environment_caveats": ["One caveat"],
            "public_copy_allowed": True,
            "metric_rows": [{"n": 1}],
            "artifact": {},
        }))
        result = validate_policy(hl, [ev])
        cross_warnings = [w for w in result.warnings if w.code == "no_matching_headline_claim"]
        assert len(cross_warnings) == 0, [w.message for w in cross_warnings]


# ===================================================================
# Aggregate validate_policy
# ===================================================================

class TestValidatePolicy:
    def test_directory_mode_discovers_json_files(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        ev_dir = tmp_path / "evidence"
        ev_dir.mkdir()
        (ev_dir / "a.json").write_text("{}")
        (ev_dir / "b.json").write_text("{}")
        (ev_dir / "c.txt").write_text("not json")
        result = validate_policy(hl, sorted(ev_dir.glob("*.json")))
        assert result.summary["evidence_files_checked"] == 2

    def test_no_errors_passes(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        result = validate_policy(hl, [])
        assert result.ok is True


# ===================================================================
# CLI script
# ===================================================================

class TestCli:
    def test_json_mode_emits_parseable_json(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_evidence_policy",
             "--headline", str(hl), "--json"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        # Output must be parseable JSON only (no extra text in json mode)
        data = json.loads(result.stdout)
        assert isinstance(data, dict)
        assert "ok" in data
        assert "errors" in data
        assert "warnings" in data
        assert "summary" in data

    def test_human_mode_prints_summary(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_evidence_policy",
             "--headline", str(hl)],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "Evidence policy: PASS" in result.stdout
        assert "headline_active_claims=0" in result.stdout

    def test_human_mode_fail_prints_fail(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("""## Active Headline Numbers
| Product | Claim | Value | Source | Commit | Run date | Notes |
|---------|-------|-------|--------|--------|----------|-------|
| archolith-context | savings | pending | pending | pending | pending | pending |
""")
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_evidence_policy",
             "--headline", str(hl)],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 1
        assert "Evidence policy: FAIL" in result.stdout

    def test_exit_code_2_on_missing_headline(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_evidence_policy",
             "--headline", str(tmp_path / "nonexistent.md")],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 2
        assert "ERROR" in result.stderr

    def test_directory_mode_discovery(self, tmp_path):
        hl = tmp_path / "HEADLINE-NUMBERS.md"
        hl.write_text("## Active Headline Numbers\n| Product | Claim |\n|---------|-------|\n| _none_ | _none_ |\n")
        ev_dir = tmp_path / "ev"
        ev_dir.mkdir()
        (ev_dir / "a.json").write_text('{"public_copy_allowed": false}')
        (ev_dir / "b.json").write_text('{"public_copy_allowed": false}')
        result = subprocess.run(
            [sys.executable, "-m", "scripts.check_evidence_policy",
             "--headline", str(hl), "--evidence-dir", str(ev_dir), "--json"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        data = json.loads(result.stdout)
        assert data["summary"]["evidence_files_checked"] == 2


# ===================================================================
# Markdown evidence convention
# ===================================================================

class TestMarkdownEvidence:
    """The archolith-evidence block is mandatory on Markdown evidence."""

    def _write(self, tmp_path, name, body):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        return p

    def test_missing_block_is_an_error(self, tmp_path):
        p = self._write(tmp_path, "e.md", "# Some Evidence\n\nNo block here.\n")
        r = validate_markdown_evidence(p)
        assert not r.ok
        assert any(e.code == "missing_evidence_block" for e in r.errors)

    def test_complete_block_passes(self, tmp_path):
        p = self._write(tmp_path, "e.md", COMPLETE_MD)
        r = validate_markdown_evidence(p)
        assert r.ok, [e.message for e in r.errors]
        assert r.summary["public_copy_allowed"] is False

    def test_missing_required_key_is_an_error(self, tmp_path):
        body = COMPLETE_MD.replace("source_tracked: true\n", "")
        p = self._write(tmp_path, "e.md", body)
        r = validate_markdown_evidence(p)
        assert not r.ok
        assert any("source_tracked" in e.message for e in r.errors)

    def test_unknown_provenance_warns_but_passes(self, tmp_path):
        body = COMPLETE_MD.replace("commit: 1aec8f3", "commit: unknown")
        p = self._write(tmp_path, "e.md", body)
        r = validate_markdown_evidence(p)
        assert r.ok
        assert any(w.code == "incomplete_provenance" for w in r.warnings)

    def test_public_copy_rejects_unknown_commit(self, tmp_path):
        body = COMPLETE_MD.replace("commit: 1aec8f3", "commit: unknown").replace(
            "public_copy_allowed: false", "public_copy_allowed: true")
        p = self._write(tmp_path, "e.md", body)
        r = validate_markdown_evidence(p)
        assert not r.ok
        assert any(e.code == "missing_provenance" for e in r.errors)

    def test_public_copy_rejects_untracked_source(self, tmp_path):
        body = COMPLETE_MD.replace("source_tracked: true", "source_tracked: false").replace(
            "public_copy_allowed: false", "public_copy_allowed: true")
        p = self._write(tmp_path, "e.md", body)
        r = validate_markdown_evidence(p)
        assert not r.ok
        assert any(e.code == "untracked_public_source" for e in r.errors)

    def test_malformed_commit_is_an_error_not_a_disclosure(self, tmp_path):
        body = COMPLETE_MD.replace("commit: 1aec8f3", "commit: not-a-hash")
        p = self._write(tmp_path, "e.md", body)
        r = validate_markdown_evidence(p)
        assert not r.ok
        assert any(e.code == "invalid_field" for e in r.errors)

    def test_readme_and_runbook_are_not_evidence(self):
        assert not is_markdown_evidence(Path("benchmarks/README.md"))
        assert not is_markdown_evidence(Path("benchmarks/RUNBOOK-scalar-state-e2e.md"))
        assert is_markdown_evidence(Path("benchmarks/filter-2026-05-30.md"))


class TestProvenanceIsVerifiedNotDeclared:
    """Provenance must be checked against git, not taken at its word.

    Before this, a block reading `commit: deadbeef`, `run_date: 2099-12-31`,
    `source: does-not-exist.json`, `source_tracked: true` and
    `public_copy_allowed: true` validated clean with zero warnings -- the
    validator only checked that the values were well-formed. That is precisely
    the input this gate exists to stop reaching public copy.
    """

    FABRICATED = """# Totally Real Results

<!-- archolith-evidence
product: menhir
command: archolith-bench menhir r1
commit: deadbeef
run_date: 2099-12-31
source: benchmarks/this-file-does-not-exist.json
source_tracked: true
public_copy_allowed: true
-->

Menhir improves recall by 99%.
"""

    def _write(self, tmp_path, body, name="e.md"):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        return p

    def test_fabricated_public_block_is_rejected(self, tmp_path):
        p = self._write(tmp_path, self.FABRICATED)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert not r.ok
        codes = {e.code for e in r.errors}
        assert "future_run_date" in codes
        assert "unverifiable_commit" in codes
        assert "untracked_source" in codes

    def test_nonexistent_commit_is_rejected_for_public_copy(self, tmp_path):
        body = self.FABRICATED.replace("run_date: 2099-12-31", "run_date: 2026-01-02")
        body = body.replace(
            "source: benchmarks/this-file-does-not-exist.json",
            "source: archolith_bench/core/evidence_policy.py")
        p = self._write(tmp_path, body)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert not r.ok
        assert any(e.code == "unverifiable_commit" for e in r.errors)

    def test_source_tracked_lie_is_rejected_for_public_copy(self, tmp_path):
        body = self.FABRICATED.replace("run_date: 2099-12-31", "run_date: 2026-01-02")
        body = body.replace("commit: deadbeef", f"commit: {_head_commit()}")
        p = self._write(tmp_path, body)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert not r.ok
        assert any(e.code == "untracked_source" for e in r.errors)

    def test_honest_public_block_still_passes(self, tmp_path):
        body = self.FABRICATED.replace("run_date: 2099-12-31", "run_date: 2026-01-02")
        body = body.replace("commit: deadbeef", f"commit: {_head_commit()}")
        body = body.replace(
            "source: benchmarks/this-file-does-not-exist.json",
            "source: archolith_bench/core/evidence_policy.py")
        p = self._write(tmp_path, body)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert r.ok, [e.message for e in r.errors]

    def test_failed_verification_is_only_a_warning_when_not_public(self, tmp_path):
        """Historical internal artifacts predate the convention: report, do not break."""
        body = self.FABRICATED.replace("run_date: 2099-12-31", "run_date: 2026-01-02")
        body = body.replace("public_copy_allowed: true", "public_copy_allowed: false")
        p = self._write(tmp_path, body)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert r.ok, [e.message for e in r.errors]
        codes = {w.code for w in r.warnings}
        assert "unverifiable_commit" in codes
        assert "untracked_source" in codes

    def test_future_run_date_is_an_error_even_when_not_public(self, tmp_path):
        body = self.FABRICATED.replace("public_copy_allowed: true", "public_copy_allowed: false")
        p = self._write(tmp_path, body)
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert not r.ok
        assert any(e.code == "future_run_date" for e in r.errors)


class TestPolicyCannotBeBypassed:
    """Each of these was a live bypass or corruption found in review."""

    FENCED_ONLY = (
        "# How to stamp evidence\n"
        "\n"
        "```\n"
        "<!-- archolith-evidence\n"
        "product: example\n"
        "commit: deadbeef\n"
        "-->\n"
        "```\n"
    )

    REAL_PLUS_SAMPLE = (
        "<!-- archolith-evidence\n"
        "product: real\n"
        "-->\n"
        "\n"
        "```\n"
        "<!-- archolith-evidence\n"
        "product: sample\n"
        "-->\n"
        "```\n"
    )

    @staticmethod
    def _block(product: str) -> str:
        return (
            "<!-- archolith-evidence\n"
            f"product: {product}\n"
            "command: c\n"
            "commit: unknown\n"
            "run_date: unknown\n"
            "source: s\n"
            "source_tracked: false\n"
            "public_copy_allowed: false\n"
            "-->\n"
        )

    def test_block_inside_a_code_fence_is_not_provenance(self):
        """F4: a doc that SHOWS the format must not be read as declaring it."""
        assert parse_md_evidence_block(self.FENCED_ONLY) is None
        assert count_md_evidence_blocks(self.FENCED_ONLY) == 0

    def test_real_block_still_parses_alongside_a_fenced_sample(self):
        meta = parse_md_evidence_block(self.REAL_PLUS_SAMPLE)
        assert meta is not None and meta["product"] == "real"
        assert count_md_evidence_blocks(self.REAL_PLUS_SAMPLE) == 1

    def test_two_blocks_are_ambiguous_not_first_wins(self, tmp_path):
        """F4: silently taking the first is how a stale block outlives a fix."""
        p = tmp_path / "e.md"
        p.write_text(self._block("a") + "\n" + self._block("b"), encoding="utf-8")
        r = validate_markdown_evidence(p, repo_root=REPO_ROOT)
        assert not r.ok
        assert any(e.code == "ambiguous_evidence_block" for e in r.errors)

    def test_an_unlisted_runbook_name_does_not_exempt_itself(self):
        """F5: exemption is a roster, so renaming cannot opt out of the policy."""
        assert is_markdown_evidence(Path("benchmarks/RUNBOOK-my-new-evidence.md"))
        assert not is_markdown_evidence(Path("benchmarks/RUNBOOK-scalar-state-e2e.md"))

    def test_manifest_cell_survives_newlines_and_backticks(self, tmp_path):
        """F13: a newline split the row and corrupted every row below it."""
        art = {
            "title": "t",
            "command": "cmd `with` backtick",
            "commit": "aaaaaaa",
            "product": "line1\nline2",
            "ability": "a",
            "fixture_or_live_source": "s",
            "model_provider": "m",
            "environment_caveats": [],
            "metric_rows": [],
            "artifact": {},
            "public_copy_allowed": False,
        }
        p = tmp_path / "hostile.json"
        p.write_text(json.dumps(art), encoding="utf-8")
        out = build_evidence_manifest([p])
        rows = [ln for ln in out.splitlines() if ln.startswith("| hostile.json")]
        assert len(rows) == 1, "row must not split across lines"
        # The Command column is wrapped in a code span, so exactly two
        # backticks are expected: its delimiters. A third would close the span
        # early and spill raw text into the table.
        assert rows[0].count("`") == 2, f"unbalanced code span: {rows[0]}"
        assert "line1 line2" in rows[0], "the embedded newline must be flattened"


class TestRepoEvidenceStaysCompliant:
    """Guards the convention against new unstamped artifacts landing in benchmarks/."""

    def test_every_tracked_markdown_artifact_has_a_valid_block(self):
        bench = Path(__file__).resolve().parent.parent / "benchmarks"
        if not bench.is_dir():
            pytest.skip("benchmarks/ not present")
        offenders = []
        for md in sorted(bench.glob("*.md")):
            if not is_markdown_evidence(md):
                continue
            r = validate_markdown_evidence(md)
            if not r.ok:
                offenders.append((md.name, [e.code for e in r.errors]))
        assert not offenders, (
            "Markdown evidence missing a valid archolith-evidence block: "
            f"{offenders}. See benchmarks/README.md for the required keys."
        )


# ===================================================================
# Evidence manifest
# ===================================================================

class TestEvidenceManifest:
    def test_manifest_indexes_declared_provenance(self, tmp_path):
        (tmp_path / "a.md").write_text(COMPLETE_MD, encoding="utf-8")
        out = build_evidence_manifest([tmp_path / "a.md"])
        assert "| a.md |" in out
        assert "archolith-filter" in out
        assert "1aec8f3" in out
        assert "1 artifact(s) indexed" in out

    def test_manifest_skips_documentation(self, tmp_path):
        (tmp_path / "README.md").write_text("# Docs\n", encoding="utf-8")
        (tmp_path / "RUNBOOK-scalar-state-e2e.md").write_text("# Runbook\n", encoding="utf-8")
        (tmp_path / "evidence-manifest.md").write_text("# Manifest\n", encoding="utf-8")
        out = build_evidence_manifest(list(tmp_path.glob("*.md")))
        assert "0 artifact(s) indexed" in out

    def test_manifest_is_not_its_own_evidence(self):
        """Regenerating the manifest must not make the directory invalid."""
        assert not is_markdown_evidence(Path("benchmarks/evidence-manifest.md"))

    def test_unknown_provenance_marked_incomplete(self, tmp_path):
        body = COMPLETE_MD.replace("commit: 1aec8f3", "commit: unknown")
        (tmp_path / "b.md").write_text(body, encoding="utf-8")
        out = build_evidence_manifest([tmp_path / "b.md"])
        assert "incomplete" in out
        assert "0 currently quotable" in out

    def test_invalid_artifact_marked_invalid(self, tmp_path):
        (tmp_path / "c.md").write_text("# No Block\n", encoding="utf-8")
        out = build_evidence_manifest([tmp_path / "c.md"])
        assert "INVALID" in out

    def test_pipes_in_values_do_not_break_the_table(self, tmp_path):
        body = COMPLETE_MD.replace(
            "command: archolith-bench filter",
            "command: archolith-bench filter | tee out.txt")
        (tmp_path / "d.md").write_text(body, encoding="utf-8")
        out = build_evidence_manifest([tmp_path / "d.md"])
        row = next(ln for ln in out.splitlines() if ln.startswith("| d.md |"))
        # The escaped pipe still contains a "|" character, so count only the
        # cell separators -- pipes not preceded by a backslash.
        separators = len(re.findall(r"(?<!\\)\|", row))
        assert separators == len(MANIFEST_COLUMNS) + 1
        assert "\\|" in row

# ===================================================================
# Helpers
# ===================================================================

def _write_tmp_json(data: dict) -> Path:
    import tempfile
    p = Path(tempfile.mktemp(suffix=".json"))
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _write_tmp_str(s: str) -> Path:
    import tempfile
    p = Path(tempfile.mktemp(suffix=".json"))
    p.write_text(s, encoding="utf-8")
    return p
