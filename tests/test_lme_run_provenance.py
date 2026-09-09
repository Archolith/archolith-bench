"""Tests for append-only buildout provenance.

Every case here is a way a resumed run could end up claiming something untrue about the
graph it produced: history overwritten, a killed phase still shown as running, two runs'
records fused, or a phase whose recorded settings are not the ones it ran under.
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


provenance = _load_script(
    "lme_run_provenance_test", "scripts/longmemeval/lib/run_provenance.py"
)


def _attempt(**overrides) -> dict:
    record = {
        "run_id": "ku-baseline-20260729",
        "arm": "baseline",
        "started_at": "2026-07-29T00:00:00Z",
        "resumed": False,
        "menhir_commit": "aaaa111",
        "bench_commit": "bbbb222",
        "fixture_count": 78,
        "fixture_sha256": "deadbeef",
        "container": "menhir-lme-ku-baseline-20260729",
        "volume": "menhir-lme-data-ku-baseline-20260729",
        "dataset": "longmemeval_s",
        "variant": "oracle",
        "namespace_prefix": "lme-",
    }
    record.update(overrides)
    return record


def _begin(path: Path, **overrides) -> dict:
    document = provenance.begin(path, _attempt(**overrides))
    provenance._write_atomic(path, document)
    return document


# ---------------------------------------------------------------------------
# Append, never replace
# ---------------------------------------------------------------------------

def test_first_attempt_is_stored_whole(tmp_path: Path) -> None:
    document = _begin(tmp_path / "run_provenance.json")

    assert document["first_started_at"] == "2026-07-29T00:00:00Z"
    assert document["phases"] == []
    # The first attempt is an attempt too -- it must appear in the list, not only at top level.
    assert len(document["attempts"]) == 1
    assert document["attempts"][0]["attempt"] == 1
    assert document["latest_attempt"]["menhir_commit"] == "aaaa111"
    assert document["attempt_count"] == 1
    assert document["last_started_at"] == "2026-07-29T00:00:00Z"


def test_resume_preserves_phases_and_original_start(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)
    document = provenance.phase_start(
        provenance._read(path), phase="build-graph", settings={"require_fresh": 1}, manifest={}
    )
    provenance._write_atomic(path, document)

    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True)
    resumed = provenance._read(path)

    # The earlier attempt's work is still on record, dated when it actually happened.
    assert resumed["first_started_at"] == "2026-07-29T00:00:00Z"
    assert [phase["phase"] for phase in resumed["phases"]] == ["build-graph"]
    assert [attempt["menhir_commit"] for attempt in resumed["attempts"]] == ["aaaa111", "aaaa111"]
    assert resumed["latest_attempt"]["menhir_commit"] == "aaaa111"
    assert resumed["last_started_at"] == "2026-07-30T09:00:00Z"


def test_resume_never_restates_the_top_level_record(tmp_path: Path) -> None:
    """The top level describes the graph on disk and is frozen after the first attempt.

    ``lme.sh ir-gate`` reads ``graph_fresh`` from this file. A resume is never fresh, so
    letting the resuming attempt write the top level would flip a true graph_fresh to false
    and silently restate the settings the data was actually ingested under.
    """
    path = tmp_path / "graph-provenance.json"
    _begin(
        path,
        graph_fresh=True,
        volume_pre_existed=False,
        segmentation="adaptive",
        requested_items=78,
    )
    original = provenance._read(path)

    # The resume reports the opposite of every one of those, as a real resume would.
    _begin(
        path,
        started_at="2026-07-30T09:00:00Z",
        resumed=True,
        graph_fresh=False,
        volume_pre_existed=True,
        segmentation="none",
        requested_items=30,
    )
    after = provenance._read(path)

    for field in ("graph_fresh", "volume_pre_existed", "segmentation", "requested_items",
                  "started_at", "menhir_commit", "bench_commit", "resumed"):
        assert after[field] == original[field], f"{field} was restated by the resume"
    # ...and the resuming attempt's values are still recorded, just not at the top level.
    assert after["latest_attempt"]["graph_fresh"] is False
    assert after["latest_attempt"]["segmentation"] == "none"


def test_every_attempt_is_kept_in_full(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, menhir_commit="aaaa111")
    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True, menhir_commit="aaaa111")
    _begin(path, started_at="2026-07-31T09:00:00Z", resumed=True, menhir_commit="aaaa111")
    document = provenance._read(path)

    assert document["attempt_count"] == 3
    assert [attempt["attempt"] for attempt in document["attempts"]] == [1, 2, 3]
    # All attempts share the same commit (canonical mode enforces this).
    assert all(a["menhir_commit"] == "aaaa111" for a in document["attempts"])
    # Full snapshots, not a four-field summary.
    assert document["attempts"][0]["fixture_sha256"] == "deadbeef"
    assert document["attempts"][0]["container"] == "menhir-lme-ku-baseline-20260729"


# ---------------------------------------------------------------------------
# Identity is immutable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, value",
    [
        ("run_id", "some-other-run"),
        ("arm", "candidate"),
        ("fixture_sha256", "0badc0de"),
        ("fixture_count", 30),
        ("container", "menhir-lme-other"),
        ("volume", "menhir-lme-data-other"),
        ("dataset", "other-dataset"),
        ("variant", "other-variant"),
        ("namespace_prefix", "other-prefix-"),
    ],
)
def test_resume_refuses_a_different_run_identity(tmp_path: Path, field: str, value) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)

    with pytest.raises(provenance.ProvenanceMismatch, match=field):
        provenance.begin(path, _attempt(**{field: value, "resumed": True}))


def test_resume_refuses_when_an_identity_field_disappears(tmp_path: Path) -> None:
    """A resume that cannot show it is the same run is not the same run."""
    path = tmp_path / "run_provenance.json"
    _begin(path)
    incomplete = _attempt()
    del incomplete["fixture_sha256"]

    with pytest.raises(provenance.ProvenanceMismatch, match="fixture_sha256"):
        provenance.begin(path, incomplete)


def test_matching_resume_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)
    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True)

    assert provenance._read(path)["identity"]["run_id"] == "ku-baseline-20260729"


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def test_open_phases_are_marked_interrupted_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)
    provenance._write_atomic(
        path,
        provenance.phase_start(
            provenance._read(path), phase="build-graph", settings={}, manifest={}
        ),
    )

    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True)
    document = provenance._read(path)

    interrupted = document["phases"][0]
    assert interrupted["status"] == "interrupted"
    assert interrupted["interrupted_at"] == "2026-07-30T09:00:00Z"
    # Recorded against the attempt that found the abandoned phase, i.e. the resume.
    assert document["attempts"][1]["phases_interrupted"] == 1
    assert document["attempts"][0]["phases_interrupted"] == 0


def test_completed_phases_are_not_reopened_or_reinterrupted(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)
    document = provenance.phase_start(
        provenance._read(path), phase="build-graph", settings={}, manifest={}
    )
    document = provenance.phase_end(document, phase="build-graph", status="completed", extra={})
    provenance._write_atomic(path, document)

    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True)

    assert provenance._read(path)["phases"][0]["status"] == "completed"


def test_phase_records_the_manifest_it_inherited(tmp_path: Path) -> None:
    """A resumed phase must be distinguishable from a fresh one."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"question_id": "a"}, {"question_id": "b"}]), encoding="utf-8"
    )

    state = provenance.manifest_state(manifest_path)

    assert state["manifest_items_at_start"] == 2
    assert state["manifest_question_ids_at_start"] == ["a", "b"]


def test_missing_manifest_records_an_empty_start(tmp_path: Path) -> None:
    state = provenance.manifest_state(tmp_path / "absent.json")

    assert state == {"manifest_items_at_start": 0, "manifest_question_ids_at_start": []}


def test_each_phase_keeps_its_own_effective_settings(tmp_path: Path) -> None:
    """The checkpoint continuation runs with different settings than phase 1 did."""
    path = tmp_path / "run_provenance.json"
    _begin(path)
    document = provenance.phase_start(
        provenance._read(path),
        phase="build-graph",
        settings={"require_fresh": 1, "ingest_stop_after_items": 30},
        manifest={},
    )
    document = provenance.phase_end(document, phase="build-graph", status="completed", extra={})
    document = provenance.phase_start(
        document,
        phase="build-graph-checkpoint-continuation",
        settings={"require_fresh": 0, "ingest_stop_after_items": 0},
        manifest={},
    )
    provenance._write_atomic(path, document)

    phases = provenance._read(path)["phases"]
    assert phases[0]["effective_settings"]["require_fresh"] == 1
    assert phases[0]["effective_settings"]["ingest_stop_after_items"] == 30
    assert phases[1]["effective_settings"]["require_fresh"] == 0
    assert phases[1]["effective_settings"]["ingest_stop_after_items"] == 0


def test_phase_end_without_an_open_phase_is_still_recorded(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path)
    document = provenance.phase_end(
        provenance._read(path), phase="recall-qa", status="failed", extra={"harness_exit": 1}
    )

    assert document["phases"][-1]["status"] == "failed"
    assert document["phases"][-1]["harness_exit"] == 1


# ---------------------------------------------------------------------------
# CLI surface used by the shell wrappers
# ---------------------------------------------------------------------------

def test_settings_are_typed_not_stringly(tmp_path: Path) -> None:
    parsed = provenance.parse_settings(
        ["require_fresh=1", "menhir_dirty=false", "segmentation=adaptive", "commit=aaa111"]
    )

    assert parsed == {
        "require_fresh": 1,
        "menhir_dirty": False,
        "segmentation": "adaptive",
        "commit": "aaa111",
    }


def test_malformed_setting_is_rejected() -> None:
    with pytest.raises(ValueError, match="key=value"):
        provenance.parse_settings(["not-a-pair"])


def test_cli_round_trip_records_a_full_run(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    record = tmp_path / "attempt.json"
    record.write_text(json.dumps(_attempt()), encoding="utf-8")

    assert provenance.main(["begin", str(path), str(record)]) == 0
    assert provenance.main(
        ["phase-start", str(path), "--phase", "build-graph", "--setting", "segmentation=adaptive"]
    ) == 0
    assert provenance.main(
        ["phase-end", str(path), "--phase", "build-graph", "--status", "completed"]
    ) == 0
    assert provenance.main(["complete", str(path), "--harness-exit", "0"]) == 0

    document = provenance._read(path)
    assert document["harness_exit"] == 0
    assert document["phases"][0]["status"] == "completed"
    assert document["phases"][0]["effective_settings"]["segmentation"] == "adaptive"
    assert not path.with_suffix(".json.tmp").exists()


# ---------------------------------------------------------------------------
# Commit immutability (canonical vs noncanonical)
# ---------------------------------------------------------------------------

def test_canonical_resume_refuses_menhir_commit_drift(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, menhir_commit="aaaa111")

    with pytest.raises(provenance.ProvenanceMismatch, match="menhir_commit"):
        provenance.begin(path, _attempt(menhir_commit="dddd444", resumed=True))


def test_canonical_resume_refuses_bench_commit_drift(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, bench_commit="bbbb222")

    with pytest.raises(provenance.ProvenanceMismatch, match="bench_commit"):
        provenance.begin(path, _attempt(bench_commit="eeee555", resumed=True))


def test_noncanonical_permits_commit_drift(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, menhir_commit="aaaa111", bench_commit="bbbb222")

    document = provenance.begin(
        path,
        _attempt(menhir_commit="dddd444", bench_commit="eeee555", resumed=True),
        noncanonical=True,
    )

    assert document["noncanonical"] is True
    assert document["latest_attempt"]["menhir_commit"] == "dddd444"


def test_noncanonical_labels_even_first_attempt(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    document = provenance.begin(path, _attempt(), noncanonical=True)

    assert document["noncanonical"] is True


def test_canonical_resume_with_same_commits_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, menhir_commit="aaaa111", bench_commit="bbbb222")

    # Same commits — no error
    _begin(path, started_at="2026-07-30T09:00:00Z", resumed=True,
           menhir_commit="aaaa111", bench_commit="bbbb222")
    document = provenance._read(path)
    assert document["attempt_count"] == 2
    assert "noncanonical" not in document


def test_commits_stored_in_identity(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, menhir_commit="aaaa111", bench_commit="bbbb222")
    document = provenance._read(path)

    assert document["identity"]["menhir_commit"] == "aaaa111"
    assert document["identity"]["bench_commit"] == "bbbb222"


def test_cli_noncanonical_flag(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    record = tmp_path / "attempt.json"
    record.write_text(json.dumps(_attempt()), encoding="utf-8")

    assert provenance.main(["begin", str(path), str(record), "--noncanonical"]) == 0
    document = provenance._read(path)
    assert document["noncanonical"] is True


# ---------------------------------------------------------------------------
# Surface-drift gate
#
# Commit immutability cannot see a dirty tree or an untracked edit: both leave the SHAs
# identical while the executed code differs. The surface digest closes that, so these cases
# are the ways a resume could still fuse two code states into one graph.
# ---------------------------------------------------------------------------

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_canonical_resume_refuses_surface_drift(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    with pytest.raises(provenance.ProvenanceMismatch, match="surface_digest changed"):
        provenance.begin(path, _attempt(surface_digest=DIGEST_B))


def test_drift_with_identical_commits_names_the_dirty_tree(tmp_path: Path) -> None:
    """The actionable half: same SHAs means the change is uncommitted or untracked."""
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    with pytest.raises(provenance.ProvenanceMismatch) as caught:
        provenance.begin(path, _attempt(surface_digest=DIGEST_B))
    assert "uncommitted or untracked edit" in str(caught.value)


def test_matching_surface_digest_resumes(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    document = provenance.begin(path, _attempt(surface_digest=DIGEST_A))
    assert document["attempt_count"] == 2


def test_resume_refuses_when_the_digest_disappears(tmp_path: Path) -> None:
    """A resume that records no digest cannot demonstrate it is running the same code."""
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    with pytest.raises(provenance.ProvenanceMismatch, match="recorded none"):
        provenance.begin(path, _attempt(surface_digest=""))


def test_a_baseline_without_a_digest_stays_resumable(tmp_path: Path) -> None:
    """Runs predating the fingerprint must not become unresumable; only keys the earlier
    record actually carried are compared."""
    path = tmp_path / "p.json"
    _begin(path, surface_digest="")
    document = provenance.begin(path, _attempt(surface_digest=DIGEST_A))
    assert document["attempt_count"] == 2


def test_noncanonical_permits_surface_drift(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    document = provenance.begin(
        path, _attempt(surface_digest=DIGEST_B), noncanonical=True
    )
    assert document["noncanonical"] is True
    assert document["attempt_count"] == 2


def test_surface_digest_is_stored_in_identity(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    document = _begin(path, surface_digest=DIGEST_A)
    assert document["identity"]["surface_digest"] == DIGEST_A


def test_an_empty_digest_is_absent_not_a_digest_of_empty(tmp_path: Path) -> None:
    """Fingerprinting is non-fatal at launch, so "" means could-not-compute. Storing it as a
    value would make two such runs look like they agreed on a code state."""
    path = tmp_path / "p.json"
    document = _begin(path, surface_digest="")
    assert "surface_digest" not in document["identity"]


def test_commit_drift_is_reported_before_surface_drift(tmp_path: Path) -> None:
    """When both moved, the commit message is the clearer diagnosis."""
    path = tmp_path / "p.json"
    _begin(path, surface_digest=DIGEST_A)
    with pytest.raises(provenance.ProvenanceMismatch, match="code commits changed"):
        provenance.begin(
            path, _attempt(surface_digest=DIGEST_B, menhir_commit="f" * 40)
        )
