"""Tests for the final acceptance report validator."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_script(
    "lme_validate_run_test", "scripts/longmemeval/lib/validate_run.py"
)
provenance_mod = _load_script(
    "lme_run_provenance_for_validate", "scripts/longmemeval/lib/run_provenance.py"
)


def _provenance(tmp_path: Path, **overrides) -> Path:
    """Write a minimal valid provenance file and return its path."""
    doc = {
        "run_id": "ku-test",
        "arm": "baseline",
        "menhir_commit": "aaaa111",
        "bench_commit": "bbbb222",
        "namespace_prefix": "lme-",
        "identity": {
            "run_id": "ku-test",
            "arm": "baseline",
            "menhir_commit": "aaaa111",
            "bench_commit": "bbbb222",
            "namespace_prefix": "lme-",
        },
        "attempts": [{
            "attempt": 1,
            "menhir_commit": "aaaa111",
            "bench_commit": "bbbb222",
        }],
        "phases": [{
            "phase": "ingest-graph",
            "status": "completed",
        }],
    }
    doc.update(overrides)
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, items: list[dict] | None = None) -> Path:
    if items is None:
        items = [
            {"namespace": "lme-postcards", "question_id": "postcards",
             "typed_assertions": 2, "scalar_views": 1},
        ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def _telemetry(tmp_path: Path, n_events: int = 3) -> Path:
    db_path = tmp_path / "mcp_telemetry.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE lifecycle_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                phase TEXT NOT NULL,
                event TEXT NOT NULL,
                status TEXT NOT NULL,
                episode_uuid TEXT,
                details_json TEXT
            )
        """)
        for i in range(n_events):
            conn.execute(
                "INSERT INTO lifecycle_events (recorded_at, phase, event, status) "
                "VALUES (?, 'test', 'test', 'ok')",
                (f"2026-07-29T00:0{i}:00Z",),
            )
    return db_path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_checks_pass(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path),
        telemetry_db=_telemetry(tmp_path),
        expected_items=1,
    )

    assert report["verdict"] == "PASS"
    assert report["failures"] == 0
    statuses = {c["check"]: c["status"] for c in report["checks"]}
    assert statuses["commit_immutability"] == "PASS"
    assert statuses["manifest_cardinality"] == "PASS"
    assert statuses["telemetry_presence"] == "PASS"
    assert statuses["zero_failed_episodes"] == "PASS"
    assert statuses["namespace_isolation"] == "PASS"


# ---------------------------------------------------------------------------
# Commit immutability
# ---------------------------------------------------------------------------

def test_commit_drift_detected(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path, attempts=[
            {"attempt": 1, "menhir_commit": "aaaa111", "bench_commit": "bbbb222"},
            {"attempt": 2, "menhir_commit": "cccc333", "bench_commit": "bbbb222"},
        ]),
        _manifest(tmp_path),
    )

    immutability = next(c for c in report["checks"] if c["check"] == "commit_immutability")
    assert immutability["status"] == "FAIL"
    assert "menhir" in immutability["detail"]


def test_noncanonical_label_is_warning(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path, noncanonical=True),
        _manifest(tmp_path),
    )

    label = next(c for c in report["checks"] if c["check"] == "canonical_label")
    assert label["status"] == "WARN"
    assert "noncanonical" in label["detail"]


# ---------------------------------------------------------------------------
# Manifest checks
# ---------------------------------------------------------------------------

def test_manifest_cardinality_mismatch(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path, items=[{"namespace": "lme-a"}, {"namespace": "lme-b"}]),
        expected_items=78,
    )

    card = next(c for c in report["checks"] if c["check"] == "manifest_cardinality")
    assert card["status"] == "FAIL"
    assert "78" in card["detail"]


def test_failed_episodes_detected(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path, items=[
            {"namespace": "lme-a", "status": "OK"},
            {"namespace": "lme-b", "status": "FAILED"},
        ]),
    )

    failed = next(c for c in report["checks"] if c["check"] == "zero_failed_episodes")
    assert failed["status"] == "FAIL"


def test_namespace_isolation_violation(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path, items=[
            {"namespace": "lme-a"},
            {"namespace": "other-b"},  # wrong prefix
        ]),
    )

    ns = next(c for c in report["checks"] if c["check"] == "namespace_isolation")
    assert ns["status"] == "FAIL"
    assert "other-b" in ns["detail"]


# ---------------------------------------------------------------------------
# Telemetry presence
# ---------------------------------------------------------------------------

def test_missing_telemetry_db_fails(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path),
        telemetry_db=tmp_path / "nonexistent.db",
    )

    telem = next(c for c in report["checks"] if c["check"] == "telemetry_presence")
    assert telem["status"] == "FAIL"


def test_empty_telemetry_db_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE lifecycle_events (
                id INTEGER PRIMARY KEY, recorded_at TEXT, phase TEXT,
                event TEXT, status TEXT, episode_uuid TEXT, details_json TEXT
            )
        """)

    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path),
        telemetry_db=db_path,
    )

    telem = next(c for c in report["checks"] if c["check"] == "telemetry_presence")
    assert telem["status"] == "FAIL"
    assert "no events" in telem["detail"]


def test_no_telemetry_path_is_warning(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path),
        _manifest(tmp_path),
        telemetry_db=None,
    )

    telem = next(c for c in report["checks"] if c["check"] == "telemetry_presence")
    assert telem["status"] == "WARN"


# ---------------------------------------------------------------------------
# Interrupted phases
# ---------------------------------------------------------------------------

def test_interrupted_phases_detected(tmp_path: Path) -> None:
    report = validator.validate(
        _provenance(tmp_path, phases=[
            {"phase": "ingest-graph", "status": "interrupted"},
        ]),
        _manifest(tmp_path),
    )

    phases = next(c for c in report["checks"] if c["check"] == "no_interrupted_phases")
    assert phases["status"] == "FAIL"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_exit_codes(tmp_path: Path) -> None:
    prov = _provenance(tmp_path)
    mfst = _manifest(tmp_path)
    output = tmp_path / "report.json"

    rc = validator.main([str(prov), str(mfst), "--output", str(output)])
    assert rc == 0
    assert output.exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"


def test_cli_fails_on_cardinality_mismatch(tmp_path: Path) -> None:
    prov = _provenance(tmp_path)
    mfst = _manifest(tmp_path, items=[{"namespace": "lme-a"}])

    rc = validator.main([str(prov), str(mfst), "--expected-items", "99"])
    assert rc == 1
