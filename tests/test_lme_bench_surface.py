"""Tests for benchmark-surface fingerprinting and blame.

This is evidence tooling, so the cases here are the ways it could lie: a digest that
stays stable while tracked code changed, a surface that silently shrinks, a deleted file
that reads as "no change", a missing checkout that fingerprints as a consistent-looking
set of absences, and a blame report that points at the wrong stage or vouches for a dirty
run's commit SHA.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


surface = _load_script("lme_bench_surface_test", "scripts/longmemeval/lib/bench_surface.py")


# ---------------------------------------------------------------------------
# Fixtures: a self-contained fake surface, so tests never depend on the real tree
# ---------------------------------------------------------------------------

def _make_surface(tmp_path: Path, *, extra_stage: str = "") -> tuple[Path, Path]:
    """Build a throwaway repo + manifest. Returns (manifest_path, repo_root)."""
    repo = tmp_path / "fakerepo"
    (repo / "ingest").mkdir(parents=True)
    (repo / "recall").mkdir(parents=True)
    (repo / "ingest" / "segmenter.py").write_text("def seg(): return 1\n", encoding="utf-8")
    (repo / "recall" / "ranker.py").write_text("def rank(): return 2\n", encoding="utf-8")

    manifest = tmp_path / "surface.yaml"
    manifest.write_text(
        "version: 1\n"
        "normalize_line_endings: true\n"
        "repos:\n"
        "  fake:\n"
        f"    root: {repo.as_posix()}\n"
        "stages:\n"
        "  ingest:\n"
        "    paths:\n"
        "      - repo: fake\n"
        "        glob: 'ingest/*.py'\n"
        "  recall:\n"
        "    paths:\n"
        "      - repo: fake\n"
        "        glob: 'recall/*.py'\n"
        + extra_stage,
        encoding="utf-8",
    )
    return manifest, repo


# ---------------------------------------------------------------------------
# The core property: a content change must move the digest
# ---------------------------------------------------------------------------

def test_editing_a_surface_file_changes_the_digest(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("def seg(): return 999\n", encoding="utf-8")
    after = surface.fingerprint(manifest)
    assert before["surface_digest"] != after["surface_digest"]


def test_unrelated_stage_digest_is_unaffected(tmp_path: Path) -> None:
    """Stage isolation is what makes blame output readable in pipeline order."""
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("changed\n", encoding="utf-8")
    after = surface.fingerprint(manifest)
    assert before["stages"]["ingest"]["stage_digest"] != after["stages"]["ingest"]["stage_digest"]
    assert before["stages"]["recall"]["stage_digest"] == after["stages"]["recall"]["stage_digest"]


def test_digest_is_stable_when_nothing_changes(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    assert surface.fingerprint(manifest)["surface_digest"] == (
        surface.fingerprint(manifest)["surface_digest"]
    )


def test_line_ending_difference_alone_does_not_move_the_digest(tmp_path: Path) -> None:
    """A Windows checkout and a Linux checkout of identical content must agree."""
    manifest, repo = _make_surface(tmp_path)
    target = repo / "ingest" / "segmenter.py"
    target.write_bytes(b"def seg():\n    return 1\n")
    lf = surface.fingerprint(manifest)["surface_digest"]
    target.write_bytes(b"def seg():\r\n    return 1\r\n")
    crlf = surface.fingerprint(manifest)["surface_digest"]
    assert lf == crlf


def test_real_content_change_survives_line_ending_normalization(tmp_path: Path) -> None:
    """Normalization must not be a hash collision for changes that matter."""
    manifest, repo = _make_surface(tmp_path)
    target = repo / "ingest" / "segmenter.py"
    target.write_bytes(b"def seg():\r\n    return 1\r\n")
    before = surface.fingerprint(manifest)["surface_digest"]
    target.write_bytes(b"def seg():\r\n    return 2\r\n")
    assert surface.fingerprint(manifest)["surface_digest"] != before


# ---------------------------------------------------------------------------
# Absence must be recorded, not skipped
# ---------------------------------------------------------------------------

def test_deleting_a_surface_file_changes_the_digest(tmp_path: Path) -> None:
    """A skipped path would leave the digest unchanged, hiding a deletion."""
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").unlink()
    after = surface.fingerprint(manifest)
    assert before["surface_digest"] != after["surface_digest"]
    assert after["stages"]["ingest"]["file_count"] == 0
    assert after["stages"]["ingest"]["unmatched_globs"] == ["fake:ingest/*.py"]


def test_adding_a_file_to_an_existing_glob_changes_the_digest(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "extra.py").write_text("def extra(): pass\n", encoding="utf-8")
    assert surface.fingerprint(manifest)["surface_digest"] != before["surface_digest"]


# ---------------------------------------------------------------------------
# The manifest must hash itself
# ---------------------------------------------------------------------------

def test_narrowing_the_surface_changes_the_digest(tmp_path: Path) -> None:
    """Removing a glob shrinks what is tracked; that must not look like 'no change'.

    Without manifest_sha256 folded into the rolled digest, deleting the recall stage
    would leave the remaining per-file hashes untouched and the digest could be made to
    look stable while tracking strictly less.
    """
    manifest, _ = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").split("  recall:")[0], encoding="utf-8"
    )
    after = surface.fingerprint(manifest)
    assert after["manifest_sha256"] != before["manifest_sha256"]
    assert after["surface_digest"] != before["surface_digest"]


# ---------------------------------------------------------------------------
# A missing repo is an error, not a stable set of absences
# ---------------------------------------------------------------------------

def test_missing_repo_root_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "surface.yaml"
    manifest.write_text(
        "version: 1\nrepos:\n  ghost:\n    root: /nonexistent/path/xyz\n"
        "stages:\n  ingest:\n    paths:\n      - repo: ghost\n        glob: '*.py'\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.SurfaceError, match="not found"):
        surface.fingerprint(manifest)


def test_env_override_redirects_a_repo_root(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "elsewhere"
    (real / "ingest").mkdir(parents=True)
    (real / "ingest" / "segmenter.py").write_text("x\n", encoding="utf-8")
    manifest = tmp_path / "surface.yaml"
    manifest.write_text(
        "version: 1\nrepos:\n  fake:\n    root: /nonexistent\n"
        "    env_override: FAKE_ROOT\n"
        "stages:\n  ingest:\n    paths:\n      - repo: fake\n        glob: 'ingest/*.py'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAKE_ROOT", str(real))
    assert surface.fingerprint(manifest)["total_files"] == 1


def test_undeclared_repo_reference_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "surface.yaml"
    manifest.write_text(
        "version: 1\nrepos:\n  fake:\n    root: .\n"
        "stages:\n  ingest:\n    paths:\n      - repo: typo\n        glob: '*.py'\n",
        encoding="utf-8",
    )
    with pytest.raises(surface.SurfaceError, match="undeclared repo"):
        surface.fingerprint(manifest)


def test_manifest_without_stages_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "surface.yaml"
    manifest.write_text("version: 1\nrepos: {}\n", encoding="utf-8")
    with pytest.raises(surface.SurfaceError, match="no 'stages'"):
        surface.fingerprint(manifest)


# ---------------------------------------------------------------------------
# Attach: append-only, per attempt
# ---------------------------------------------------------------------------

def test_attach_records_per_attempt_and_appends(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    provenance = tmp_path / "run_provenance.json"
    provenance.write_text(json.dumps({"run_id": "r1", "attempt_count": 1}), encoding="utf-8")

    surface.attach(provenance, manifest)
    (repo / "ingest" / "segmenter.py").write_text("changed\n", encoding="utf-8")
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["attempt_count"] = 2
    provenance.write_text(json.dumps(document), encoding="utf-8")
    surface.attach(provenance, manifest)

    document = json.loads(provenance.read_text(encoding="utf-8"))
    history = document["surface_fingerprints"]
    assert [entry["attempt"] for entry in history] == [1, 2]
    # Two attempts of one run executed different code; both records survive.
    assert history[0]["surface_digest"] != history[1]["surface_digest"]
    assert document["surface_digest"] == history[1]["surface_digest"]


def test_attach_refuses_a_missing_provenance_file(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    with pytest.raises(surface.SurfaceError, match="provenance file not found"):
        surface.attach(tmp_path / "absent.json", manifest)


# ---------------------------------------------------------------------------
# Blame
# ---------------------------------------------------------------------------

def _run_document(run_id: str, snapshot: dict, **attempt) -> dict:
    return {
        "run_id": run_id,
        "latest_attempt": {"menhir_dirty": False, "bench_dirty": False, **attempt},
        "surface_fingerprints": [{"attempt": 1, **snapshot}],
    }


def test_blame_names_the_changed_file_and_stage(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("def seg(): return 999\n", encoding="utf-8")
    after = surface.fingerprint(manifest)

    report = surface.blame(_run_document("a", before), _run_document("b", after))
    assert report["surface_identical"] is False
    assert report["changed_stages"] == ["ingest"]
    assert [row["file"] for row in report["changed_files"]] == ["fake:ingest/segmenter.py"]
    assert report["changed_files"][0]["change"] == "modified"


def test_blame_reports_identical_surface(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    report = surface.blame(_run_document("a", snapshot), _run_document("b", snapshot))
    assert report["surface_identical"] is True
    assert report["changed_files"] == []


def test_blame_orders_stages_in_pipeline_order(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("a\n", encoding="utf-8")
    (repo / "recall" / "ranker.py").write_text("b\n", encoding="utf-8")
    after = surface.fingerprint(manifest)
    report = surface.blame(_run_document("a", before), _run_document("b", after))
    assert report["changed_stages"] == ["ingest", "recall"]


def test_blame_can_filter_to_one_stage(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("a\n", encoding="utf-8")
    (repo / "recall" / "ranker.py").write_text("b\n", encoding="utf-8")
    after = surface.fingerprint(manifest)
    report = surface.blame(
        _run_document("a", before), _run_document("b", after), stage_filter="recall"
    )
    assert [row["stage"] for row in report["changed_files"]] == ["recall"]


def test_blame_flags_a_dirty_run(tmp_path: Path) -> None:
    """A dirty run's SHA does not describe what executed; the report must say so."""
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    report = surface.blame(
        _run_document("a", snapshot, menhir_dirty=True),
        _run_document("b", snapshot),
    )
    assert any("DIRTY" in warning for warning in report["warnings"])


def test_blame_flags_unknown_cleanliness(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    run_a = _run_document("a", snapshot)
    run_a["latest_attempt"] = {"menhir_commit": "abc1234"}  # no dirty flag recorded
    report = surface.blame(run_a, _run_document("b", snapshot))
    assert any("no dirty flag" in warning for warning in report["warnings"])


def test_blame_flags_a_changed_manifest(tmp_path: Path) -> None:
    """Two runs measured against different declared surfaces are not comparable."""
    manifest, _ = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").split("  recall:")[0], encoding="utf-8"
    )
    after = surface.fingerprint(manifest)
    report = surface.blame(_run_document("a", before), _run_document("b", after))
    assert any("manifest ITSELF changed" in warning for warning in report["warnings"])


def test_blame_flags_a_fixture_change(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    run_a = _run_document("a", snapshot) | {"fixture_sha256": "aaa"}
    run_b = _run_document("b", snapshot) | {"fixture_sha256": "bbb"}
    report = surface.blame(run_a, run_b)
    assert any("FIXTURE DIFFERS" in warning for warning in report["warnings"])


def test_blame_degrades_when_a_run_has_no_fingerprint(tmp_path: Path) -> None:
    """35 of 51 historical runs predate fingerprinting; blame must say so, not guess."""
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    legacy = {"run_id": "old", "latest_attempt": {"menhir_commit": "abc", "menhir_dirty": False}}
    report = surface.blame(legacy, _run_document("b", snapshot))
    assert report["surface_comparable"] is False
    assert any("no surface fingerprint" in warning for warning in report["warnings"])


def test_blame_reports_setting_differences(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    run_a = _run_document("a", snapshot)
    run_b = _run_document("b", snapshot)
    run_a["phases"] = [{"phase": "build", "effective_settings": {"scalar_threshold": "2/3"}}]
    run_b["phases"] = [{"phase": "build", "effective_settings": {"scalar_threshold": "3/3"}}]
    report = surface.blame(run_a, run_b)
    assert report["changed_settings"] == [
        {"setting": "scalar_threshold", "a": "2/3", "b": "3/3"}
    ]


def test_blame_uses_the_last_phase_that_recorded_settings(tmp_path: Path) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    run_a = _run_document("a", snapshot)
    run_b = _run_document("b", snapshot)
    run_a["phases"] = [
        {"phase": "build", "effective_settings": {"k": 1}},
        {"phase": "recall", "effective_settings": {}},  # empty: must fall back
    ]
    run_b["phases"] = [{"phase": "build", "effective_settings": {"k": 1}}]
    assert surface.blame(run_a, run_b)["changed_settings"] == []


def test_format_blame_renders_without_error(tmp_path: Path) -> None:
    manifest, repo = _make_surface(tmp_path)
    before = surface.fingerprint(manifest)
    (repo / "ingest" / "segmenter.py").write_text("x\n", encoding="utf-8")
    after = surface.fingerprint(manifest)
    rendered = surface.format_blame(
        surface.blame(_run_document("a", before), _run_document("b", after))
    )
    assert "fake:ingest/segmenter.py" in rendered
    assert "[ingest]" in rendered


# ---------------------------------------------------------------------------
# The real manifest must stay valid
# ---------------------------------------------------------------------------

def test_real_manifest_fingerprints_with_every_glob_matching() -> None:
    """An unmatched glob in the checked-in manifest means a surface file was renamed or
    deleted and the declaration was not updated - a silent loss of coverage."""
    snapshot = surface.fingerprint(surface.DEFAULT_MANIFEST)
    assert snapshot["total_files"] > 0
    unmatched = {
        stage: payload["unmatched_globs"]
        for stage, payload in snapshot["stages"].items()
        if payload["unmatched_globs"]
    }
    assert not unmatched, f"globs matching nothing: {unmatched}"


def test_real_manifest_declares_every_pipeline_stage() -> None:
    snapshot = surface.fingerprint(surface.DEFAULT_MANIFEST)
    assert set(snapshot["stages"]) == set(surface.STAGE_ORDER)


def test_blame_finds_dirty_flags_recorded_only_in_phase_settings(tmp_path: Path) -> None:
    """build_graph.sh records dirty state in the phase, not the attempt record.

    Reading only latest_attempt reported "cleanliness unknown" for every graph build even
    though the flags were right there in effective_settings -- withholding a dirty-tree
    warning the record could support. Found by running the real date-smoke build, not by
    the synthetic fixtures above.
    """
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    graph_build = {
        "container": "menhir-lme-datesmoke",
        "latest_attempt": {"menhir_commit": "a" * 40, "bench_commit": "b" * 40},
        "phases": [{
            "phase": "ingest-graph",
            "effective_settings": {
                "menhir_commit": "a" * 40, "bench_commit": "b" * 40,
                "menhir_dirty": False, "bench_dirty": True,
            },
        }],
        "surface_fingerprints": [{"attempt": 1, **snapshot}],
    }
    report = surface.blame(graph_build, _run_document("b", snapshot))
    assert any("DIRTY tree (bench)" in w for w in report["warnings"])
    assert not any("cleanliness is unknown" in w for w in report["warnings"])


def test_blame_still_reports_unknown_cleanliness_when_nothing_recorded_it(
    tmp_path: Path,
) -> None:
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    document = {
        "run_id": "old",
        "latest_attempt": {"menhir_commit": "a" * 40},
        "surface_fingerprints": [{"attempt": 1, **snapshot}],
    }
    report = surface.blame(document, _run_document("b", snapshot))
    assert any("cleanliness is unknown" in w for w in report["warnings"])


def test_blame_names_a_run_by_container_when_it_has_no_run_id(tmp_path: Path) -> None:
    """Graph provenance carries `container`, not `run_id`; "?" would be unactionable."""
    manifest, _ = _make_surface(tmp_path)
    snapshot = surface.fingerprint(manifest)
    document = {
        "container": "menhir-lme-datesmoke",
        "latest_attempt": {"menhir_commit": "a" * 40, "menhir_dirty": True},
        "surface_fingerprints": [{"attempt": 1, **snapshot}],
    }
    report = surface.blame(document, _run_document("b", snapshot))
    assert any("menhir-lme-datesmoke" in w for w in report["warnings"])
