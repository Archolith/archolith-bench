"""Build resume gate: LME_RESUME=1 reuses only the graph an earlier attempt built."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archolith_bench.build import pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "scripts/longmemeval/config.sh").read_text()

NAME = "menhir-lme-test"
VOL = "lme-test-vol"


def _gates(tmp_path: Path, *, resume: str, fresh: str = "0", provenance=True, identity=None,
           container=True, volume=True, manifest_dir=True) -> None:
    results = tmp_path / "results"
    results.mkdir()
    if provenance:
        ident = {"container": NAME, "volume": VOL} if identity is None else identity
        (results / f"graph-provenance-{NAME}.json").write_text(
            json.dumps({"identity": ident}), encoding="utf-8")
    manifest_parent = results / "manifests"
    if manifest_dir:
        manifest_parent.mkdir()
    settings = {
        "LME_RESUME": resume, "LME_REQUIRE_FRESH": fresh,
        "LME_RESULTS_DIR": results.as_posix(),
        "LME_MANIFEST_PATH": (manifest_parent / "manifest.json").as_posix(),
        "LME_NEO4J_NAME": NAME, "LME_NEO4J_VOL": VOL,
    }
    pipeline.check_graph_gates(settings, container_exists=container, volume_exists=volume)


def _ingest_settings(**over: str) -> dict[str, str]:
    s = {
        "LIMIT": "1", "MENHIR_URL": "http://unused", "LME_MANIFEST_PATH": "unused",
        "LME_SEGMENTATION": "adaptive", "LME_INGEST_CONCURRENCY": "1",
        "LME_INGEST_STOP_AFTER_ITEMS": "0", "LME_SCALAR_CONSOLIDATION_K": "3",
        "LME_SCALAR_CALL_BUDGET": "50", "LME_COUNTER_STATE_ENABLED": "0",
        "LME_SCALAR_STATE_ENABLED": "0",
    }
    s.update(over)
    return s


def test_resume_with_matching_identity_passes(tmp_path: Path) -> None:
    _gates(tmp_path, resume="1", fresh="1")


@pytest.mark.parametrize("kwargs,message", [
    ({"provenance": False}, "provenance file not found"),
    ({"container": False}, "container does not exist"),
    ({"volume": False}, "volume does not exist"),
    ({"manifest_dir": False}, "manifest directory not found"),
    ({"identity": {"container": "other", "volume": VOL}}, "provenance container"),
    ({"identity": {"container": NAME, "volume": "other"}}, "provenance volume"),
])
def test_resume_refusals(tmp_path: Path, kwargs, message) -> None:
    with pytest.raises(pipeline.Die) as exc:
        _gates(tmp_path, resume="1", **kwargs)
    assert exc.value.code == 1
    assert "resume refused" in exc.value.message and message in exc.value.message


def test_fresh_path_unchanged_when_not_resuming(tmp_path: Path) -> None:
    with pytest.raises(pipeline.Die) as exc:
        _gates(tmp_path, resume="0", fresh="1", provenance=False)
    assert "fresh build refused: volume already exists" in exc.value.message
    assert "resume refused" not in exc.value.message


def test_resume_off_without_require_fresh_does_not_gate(tmp_path: Path) -> None:
    _gates(tmp_path, resume="0", fresh="0", provenance=False, container=False, volume=False)


def test_static_resume_wiring() -> None:
    assert ("resumed", "${LME_RESUME}", "int") in pipeline.ATTEMPT_RECORD
    assert ("resumed", "${LME_RESUME}") in pipeline.PHASE_SETTINGS
    assert ("LME_NEO4J_HTTP_URL", "http://localhost:${LME_HTTP}") in pipeline.MENHIR_ENV
    # --reset-file only inside the non-empty guard, and nowhere else.
    assert Path(pipeline.__file__).read_text(encoding="utf-8").count("--reset-file") == 1
    for unset in ({}, {"LME_RESET_FILE": ""}):
        assert "--reset-file" not in pipeline.ingest_args(_ingest_settings(**unset))
    args = pipeline.ingest_args(_ingest_settings(LME_RESET_FILE="ns.txt"))
    assert args.count("--reset-file") == 1
    assert args[args.index("--reset-file") + 1] == "ns.txt"
    assert 'LME_RESUME="${LME_RESUME:-0}"' in CONFIG
