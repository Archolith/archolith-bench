"""Evidence Status section of the generated BENCHMARKS.md report."""

from __future__ import annotations

from archolith_bench.core.report import _evidence_status_section, _group_by_artifact

COMPLETE_MD = """# Example Evidence

<!-- archolith-evidence
product: archolith-filter
command: archolith-bench filter
commit: 1aec8f3
run_date: 2026-05-30
source: benchmarks/filter_results.json
source_tracked: true
public_copy_allowed: false
-->
"""

NO_CLAIMS_HEADLINE = (
    "## Active Headline Numbers\n"
    "| Product | Claim |\n"
    "|---------|-------|\n"
    "| _none_ | _none_ |\n"
)


class TestGroupByArtifact:
    class _Issue:
        def __init__(self, path, code, message):
            self.path, self.code, self.message = path, code, message

    def test_missing_fields_collapse_to_one_line(self):
        issues = [
            self._Issue("b/x.json", "missing_field", "public_copy_allowed=false: field 'title' is missing"),
            self._Issue("b/x.json", "missing_field", "public_copy_allowed=false: field 'command' is missing"),
        ]
        grouped = _group_by_artifact(issues)
        assert list(grouped) == ["x.json"]
        assert len(grouped["x.json"]) == 1
        assert "missing 2 required field(s): title, command" in grouped["x.json"][0]

    def test_distinct_issues_are_kept(self):
        issues = [
            self._Issue("b/y.md", "incomplete_provenance", "provenance is incomplete (commit)"),
            self._Issue("b/y.md", "untracked_raw_source", "source_tracked=false"),
        ]
        grouped = _group_by_artifact(issues)
        assert len(grouped["y.md"]) == 2

    def test_output_is_sorted_by_artifact(self):
        issues = [
            self._Issue("b/z.md", "x", "m"),
            self._Issue("b/a.md", "x", "m"),
        ]
        assert list(_group_by_artifact(issues)) == ["a.md", "z.md"]


class TestEvidenceStatusSection:
    def _repo(self, tmp_path, artifacts: dict[str, str]):
        (tmp_path / "HEADLINE-NUMBERS.md").write_text(NO_CLAIMS_HEADLINE, encoding="utf-8")
        bench = tmp_path / "benchmarks"
        bench.mkdir()
        for name, body in artifacts.items():
            (bench / name).write_text(body, encoding="utf-8")
        return tmp_path

    def test_reports_pass_and_counts(self, tmp_path):
        root = self._repo(tmp_path, {"good.md": COMPLETE_MD})
        out = "".join(_evidence_status_section(root))
        assert "## Evidence Status" in out
        assert "**PASS**" in out
        assert "1 artifact(s) checked" in out
        assert "0 cleared for public copy" in out

    def test_clean_evidence_says_so(self, tmp_path):
        root = self._repo(tmp_path, {"good.md": COMPLETE_MD})
        out = "".join(_evidence_status_section(root))
        assert "All tracked evidence carries complete provenance." in out

    def test_unknown_provenance_listed_as_stale(self, tmp_path):
        body = COMPLETE_MD.replace("commit: 1aec8f3", "commit: unknown")
        root = self._repo(tmp_path, {"old.md": body})
        out = "".join(_evidence_status_section(root))
        assert "Stale or incomplete evidence" in out
        assert "old.md" in out

    def test_invalid_artifact_reported_as_error(self, tmp_path):
        root = self._repo(tmp_path, {"bad.md": "# No Block\n"})
        out = "".join(_evidence_status_section(root))
        assert "**FAIL**" in out
        assert "fail the evidence policy" in out

    def test_documentation_is_not_counted(self, tmp_path):
        root = self._repo(tmp_path, {
            "good.md": COMPLETE_MD,
            "README.md": "# Docs\n",
            "RUNBOOK-x.md": "# Runbook\n",
        })
        out = "".join(_evidence_status_section(root))
        assert "1 artifact(s) checked" in out

    def test_missing_repo_files_degrade_gracefully(self, tmp_path):
        out = "".join(_evidence_status_section(tmp_path))
        assert "Evidence policy not evaluated" in out

    def test_section_is_ascii(self, tmp_path):
        root = self._repo(tmp_path, {"good.md": COMPLETE_MD})
        out = "".join(_evidence_status_section(root))
        assert all(ord(c) < 128 for c in out)
