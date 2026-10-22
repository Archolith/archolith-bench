"""build_graph.sh resume gate: LME_RESUME=1 reuses only the graph an earlier attempt built."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/longmemeval/build_graph.sh").read_text()
CONFIG = (ROOT / "scripts/longmemeval/config.sh").read_text()


def _bash() -> str:
    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    return bash


def _bench_py() -> str:
    candidate = ROOT / ".venv/Scripts/python.exe"
    if candidate.exists():
        return candidate.resolve().as_posix()
    return Path(sys.executable).resolve().as_posix()


def _slice() -> str:
    start = SCRIPT.index('CONTAINER_PRE_EXISTED="false"')
    elif_at = SCRIPT.index('elif [ "${LME_REQUIRE_FRESH}" = "1" ]', start)
    end = SCRIPT.index("\nfi\n", elif_at) + len("\nfi\n")
    return SCRIPT[start:end]


NAME = "menhir-lme-test"
VOL = "lme-test-vol"


def _run(tmp_path: Path, *, resume: str, fresh: str = "0", provenance=True, identity=None,
         container=True, volume=True, manifest_dir=True):
    results = tmp_path / "results"
    results.mkdir()
    if provenance:
        ident = {"container": NAME, "volume": VOL} if identity is None else identity
        (results / f"graph-provenance-{NAME}.json").write_text(
            json.dumps({"identity": ident}), encoding="utf-8")
    manifest_parent = results / "manifests"
    if manifest_dir:
        manifest_parent.mkdir()
    prelude = (
        "set -u\n"
        'die(){ echo "DIE: $*"; exit 1; }\n'
        'lme_container_exists(){ [ "${STUB_CONTAINER}" = "1" ]; }\n'
        'VOLUME_PRE_EXISTED="$STUB_VOLUME"\n'
    )
    command = prelude + _slice() + '\necho REACHED_END\n'
    env = dict(
        os.environ,
        LME_RESUME=resume, LME_REQUIRE_FRESH=fresh,
        LME_RESULTS_DIR=results.as_posix(),
        LME_MANIFEST_PATH=(manifest_parent / "manifest.json").as_posix(),
        LME_NEO4J_NAME=NAME, LME_NEO4J_VOL=VOL, BENCH_PY=_bench_py(),
        STUB_CONTAINER="1" if container else "0",
        STUB_VOLUME="true" if volume else "false",
    )
    return subprocess.run([_bash(), "-c", command], env=env, capture_output=True, text=True)


def test_resume_with_matching_identity_passes(tmp_path: Path) -> None:
    r = _run(tmp_path, resume="1", fresh="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REACHED_END" in r.stdout


@pytest.mark.parametrize("kwargs,message", [
    ({"provenance": False}, "provenance file not found"),
    ({"container": False}, "container does not exist"),
    ({"volume": False}, "volume does not exist"),
    ({"manifest_dir": False}, "manifest directory not found"),
    ({"identity": {"container": "other", "volume": VOL}}, "provenance container"),
    ({"identity": {"container": NAME, "volume": "other"}}, "provenance volume"),
])
def test_resume_refusals(tmp_path: Path, kwargs, message) -> None:
    r = _run(tmp_path, resume="1", **kwargs)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "resume refused" in r.stdout and message in r.stdout
    assert "REACHED_END" not in r.stdout


def test_fresh_path_unchanged_when_not_resuming(tmp_path: Path) -> None:
    r = _run(tmp_path, resume="0", fresh="1", provenance=False)
    assert r.returncode == 1
    assert "fresh build refused: volume already exists" in r.stdout
    assert "resume refused" not in r.stdout


def test_resume_off_without_require_fresh_does_not_gate(tmp_path: Path) -> None:
    r = _run(tmp_path, resume="0", fresh="0", provenance=False, container=False, volume=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_static_resume_wiring() -> None:
    assert '"resumed": ${LME_RESUME}' in SCRIPT
    assert '--setting "resumed=${LME_RESUME}"' in SCRIPT
    assert 'export LME_NEO4J_HTTP_URL="http://localhost:${LME_HTTP}"' in SCRIPT
    # --reset-file only inside the non-empty guard, and nowhere else.
    assert SCRIPT.count("--reset-file") == 1
    guard = 'if [ -n "${LME_RESET_FILE:-}" ]; then\n  INGEST_ARGS+=(--reset-file "${LME_RESET_FILE}")\nfi'
    assert guard in SCRIPT
    assert 'LME_RESUME="${LME_RESUME:-0}"' in CONFIG
