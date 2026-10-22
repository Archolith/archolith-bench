"""validate_run: an interrupted ingest-graph phase is resolved only by a later completed one."""

from __future__ import annotations

import importlib.util
import json
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
    "lme_validate_run_resume_test", "scripts/longmemeval/lib/validate_run.py"
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


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text("[]", encoding="utf-8")
    return path


def _attempts(n: int) -> list[dict]:
    return [{"attempt": i + 1, "menhir_commit": "aaaa111", "bench_commit": "bbbb222"}
            for i in range(n)]


def _phase_check(tmp_path: Path, phases: list, **overrides) -> tuple[dict, dict]:
    report = validator.validate(
        _provenance(tmp_path, phases=phases, **overrides), _manifest(tmp_path))
    check = next(c for c in report["checks"] if c["check"] == "no_interrupted_phases")
    return report, check


def _p(phase: str, status: str) -> dict:
    return {"phase": phase, "status": status}


def test_interrupted_then_completed_ingest_graph_passes(tmp_path: Path) -> None:
    report, check = _phase_check(
        tmp_path,
        [_p("ingest-graph", "interrupted"), _p("ingest-graph", "completed")],
        attempts=_attempts(2))
    assert check["status"] == "PASS"
    assert "1" in check["detail"] and "resolved" in check["detail"]
    assert report["attempt_count"] == 2
    assert report["resumed"] is True


def test_two_interrupted_then_completed_reports_two_resolved(tmp_path: Path) -> None:
    _, check = _phase_check(
        tmp_path,
        [_p("ingest-graph", "interrupted"), _p("ingest-graph", "interrupted"),
         _p("ingest-graph", "completed")],
        attempts=_attempts(3))
    assert check["status"] == "PASS"
    assert "2 interrupted" in check["detail"]


def test_completed_before_interrupted_fails(tmp_path: Path) -> None:
    _, check = _phase_check(
        tmp_path, [_p("ingest-graph", "completed"), _p("ingest-graph", "interrupted")])
    assert check["status"] == "FAIL"


def test_interrupted_retry_failed_resolved_by_retry_failed_still_fails(tmp_path: Path) -> None:
    _, check = _phase_check(
        tmp_path,
        [_p("ingest-graph", "completed"), _p("retry-failed", "interrupted"),
         _p("retry-failed", "completed")])
    assert check["status"] == "FAIL"


def test_interrupted_ingest_graph_resolved_only_by_other_phase_fails(tmp_path: Path) -> None:
    _, check = _phase_check(
        tmp_path, [_p("ingest-graph", "interrupted"), _p("retry-failed", "completed")])
    assert check["status"] == "FAIL"


def test_lone_nonresolvable_phases_still_fail(tmp_path: Path) -> None:
    for phases in ([_p("ingest-graph", "interrupted")],
                   [_p("ingest-graph", "failed")],
                   [_p("ingest-graph", "started")],
                   [_p("ingest-graph", "failed"), _p("ingest-graph", "completed")],
                   []):
        _, check = _phase_check(tmp_path, phases)
        assert check["status"] == "FAIL", phases


def test_single_attempt_is_not_resumed(tmp_path: Path) -> None:
    report = validator.validate(_provenance(tmp_path), _manifest(tmp_path))
    assert report["attempt_count"] == 1
    assert report["resumed"] is False


def test_missing_attempts_does_not_crash_and_is_not_resumed(tmp_path: Path) -> None:
    for attempts in ([], None):
        report = validator.validate(
            _provenance(tmp_path, attempts=attempts), _manifest(tmp_path))
        assert report["resumed"] is False
        assert report["attempt_count"] in (None, 0)
