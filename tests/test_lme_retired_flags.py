"""retired_flags.py: refuse a Menhir launch that sets a flag the serving tree has retired.

Menhir ignores a retired env var, so an arm or export setting one to 0 would run the feature on
while the run's settings say it was off.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LME = ROOT / "scripts" / "longmemeval"
TOOL = LME / "lib" / "retired_flags.py"

REGISTRY = '''
from dataclasses import dataclass
FEATURES = {}
RETIRED = {
    "frontier_source_memories": "2026-10-27: always on (#168 freeze)",
    "MENHIR_FRONTIER_SOURCE_MEMORIES": "2026-10-27: always on (#168 freeze)",
    "MENHIR_ANCHORED_TIME_EXPIRY": "2026-10-27: folded into MENHIR_ANCHORED_TIME_RESOLVER",
}
'''


def _tree(tmp_path: Path, body: str | None = REGISTRY) -> Path:
    if body is not None:
        path = tmp_path / "src" / "menhir" / "config" / "feature_flags.py"
        path.parent.mkdir(parents=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def _check(tree: Path, **env: str) -> subprocess.CompletedProcess[str]:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("MENHIR_")}
    clean.update(env)
    return subprocess.run([sys.executable, "-I", str(TOOL), "check", str(tree)],
                          env=clean, capture_output=True, text=True)


def test_refuses_a_retired_flag_even_when_set_off(tmp_path: Path) -> None:
    result = _check(_tree(tmp_path), MENHIR_ANCHORED_TIME_EXPIRY="0")
    assert result.returncode == 1
    assert "MENHIR_ANCHORED_TIME_EXPIRY=0 is retired" in result.stderr
    assert "folded into MENHIR_ANCHORED_TIME_RESOLVER" in result.stderr


def test_names_every_retired_flag_set(tmp_path: Path) -> None:
    result = _check(_tree(tmp_path), MENHIR_ANCHORED_TIME_EXPIRY="1",
                    MENHIR_FRONTIER_SOURCE_MEMORIES="true")
    assert result.returncode == 1
    assert "MENHIR_ANCHORED_TIME_EXPIRY" in result.stderr
    assert "MENHIR_FRONTIER_SOURCE_MEMORIES" in result.stderr


def test_allows_live_flags_and_settings_keys(tmp_path: Path) -> None:
    # The settings-field key ("frontier_source_memories") is not an env var Menhir reads.
    result = _check(_tree(tmp_path), MENHIR_ANCHORED_TIME_RESOLVER="1",
                    frontier_source_memories="0")
    assert result.returncode == 0, result.stderr


def test_a_tree_without_the_registry_passes(tmp_path: Path) -> None:
    assert _check(_tree(tmp_path, None), MENHIR_ANCHORED_TIME_EXPIRY="0").returncode == 0


def test_a_tree_without_retired_passes(tmp_path: Path) -> None:
    assert _check(_tree(tmp_path, "FEATURES = {}\n"), MENHIR_ANCHORED_TIME_EXPIRY="0").returncode == 0


@pytest.mark.parametrize("body", ["RETIRED = dict(a=1)\n", "RETIRED = ['MENHIR_X']\n", "RETIRED = {\n"])
def test_an_unreadable_registry_fails_closed(tmp_path: Path, body: str) -> None:
    assert _check(_tree(tmp_path, body)).returncode == 2


def test_reads_the_registry_without_importing_menhir(tmp_path: Path) -> None:
    tree = _tree(tmp_path, "import menhir_not_installed\n" + REGISTRY)
    assert _check(tree, MENHIR_FRONTIER_SOURCE_MEMORIES="1").returncode == 1


@pytest.mark.parametrize("script, serve", [
    ("build_graph.sh", '"${MENHIR_FRONTIER_BIN}" serve'),
    ("retry.sh", '"${MENHIR_FRONTIER_BIN}" serve'),
])
def test_launchers_check_after_their_exports_and_before_serving(script: str, serve: str) -> None:
    text = (LME / script).read_text(encoding="utf-8")
    check = text.index('lib/retired_flags.py" check "${MENHIR_FRONTIER}"')
    assert check < text.index(serve)
    last_menhir_export = max(i for i, line in enumerate(text.splitlines())
                             if line.startswith("export MENHIR_"))
    assert text[:check].count("\n") > last_menhir_export


def test_module_loads() -> None:
    spec = importlib.util.spec_from_file_location("lme_retired_flags_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["bogus"]) == 2
