"""P8a.3: a served graph records the build's commits beside the serving tree's commit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from archolith_bench.build import pipeline
from archolith_bench.build.config import apply_env_overrides, resolve_yaml, resolved_record
from archolith_bench.staged import __main__ as cli
from archolith_bench.staged.config import resolve, run_id
from archolith_bench.staged.serve import graph_record

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = REPO / "builds" / "default.yaml"
FIXTURE = REPO / "fixtures" / "longmemeval_sample.json"
ARM = REPO / "arms" / "no_memory.yaml"


def _name() -> str:
    config, _, _ = apply_env_overrides(resolve_yaml(DEFAULT_BUILD), {})
    return str(config["container"]["neo4j_name"])


def _provenance(results: Path, **identity) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"graph-provenance-{_name()}.json"
    path.write_text(json.dumps({"identity": identity, "attempts": []}), encoding="utf-8")
    return path


def test_build_records_the_serving_trees_commit(tmp_path, monkeypatch) -> None:
    seen = []

    def facts(repo, env, specs):
        seen.append(repo)
        return ("c-" + Path(repo).name, False, 0, [])
    monkeypatch.setattr(pipeline, "git_facts", facts)
    monkeypatch.setattr(pipeline, "_out", lambda *a, **k: (0, "digest"))
    s = {"BENCH_PY": "py", "BENCH_DIR": (tmp_path / "bench").as_posix(),
         "MENHIR_MAIN": (tmp_path / "main").as_posix(), "MENHIR_FRONTIER": (tmp_path / "frontier").as_posix()}
    pipeline.code_facts(s, {})
    assert seen[0] == s["MENHIR_FRONTIER"] and s["MENHIR_COMMIT"] == "c-frontier"


def test_graph_record_reads_the_builds_provenance(tmp_path) -> None:
    results = tmp_path / "lme-ingest"
    _provenance(results, menhir_commit="m" * 40, bench_commit="b" * 40, surface_digest="s" * 64, container=_name())
    rec = graph_record(DEFAULT_BUILD, {}, results)
    config, overrides, _ = apply_env_overrides(resolve_yaml(DEFAULT_BUILD), {})
    assert rec == {"build_id": resolved_record(DEFAULT_BUILD, config, overrides)["build_id"], "neo4j_name": _name(),
                   "provenance": f"graph-provenance-{_name()}.json", "build_menhir_commit": "m" * 40,
                   "build_bench_commit": "b" * 40, "build_surface_digest": "s" * 64}


@pytest.mark.parametrize("setup", ["none", "missing", "corrupt"])
def test_graph_record_unknown_build_is_none_not_guessed(tmp_path, setup) -> None:
    results = tmp_path / "lme-ingest"
    if setup == "corrupt":
        results.mkdir()
        (results / f"graph-provenance-{_name()}.json").write_text("{not json", encoding="utf-8")
    rec = graph_record(DEFAULT_BUILD, {}, None if setup == "none" else results)
    assert rec["provenance"] is None and rec["build_menhir_commit"] is None and rec["build_bench_commit"] is None


def _resolved(**kw):
    from archolith_bench.staged.config import load_arm
    return resolve(load_arm(ARM), benchmark_id="lme", fixture_path="f.json", fixture_sha256="0" * 64,
                   prompt_fingerprint="p", renderer_sha256="r", menhir_commit="serve", namespace_template="t",
                   model="m", judge={}, **kw)


def test_graph_changes_the_run_id_and_absence_keeps_it() -> None:
    assert "graph" not in _resolved()
    g = {"build_id": "x", "neo4j_name": "n", "provenance": "p", "build_menhir_commit": "a",
         "build_bench_commit": "b", "build_surface_digest": "s"}
    ids = {run_id(_resolved()), run_id(_resolved(graph=g)), run_id(_resolved(graph={**g, "build_menhir_commit": "z"}))}
    assert len(ids) == 3


def test_served_run_config_carries_both_commits(tmp_path, monkeypatch) -> None:
    results = tmp_path / "lme-ingest"
    _provenance(results, menhir_commit="m" * 40, bench_commit="b" * 40, surface_digest="s" * 64)
    tree = tmp_path / "menhir"
    tree.mkdir()
    monkeypatch.setattr(cli, "_menhir_commit", lambda root: "e" * 40)
    monkeypatch.setattr(cli, "assert_not_production", lambda url: None)
    runs = tmp_path / "runs"
    rc = cli.main(["run", "--arm", str(ARM), "--fixture", str(FIXTURE), "--limit", "1", "--runs-dir", str(runs),
                   "--stages", "", "--serve", str(tree), "--graph", str(DEFAULT_BUILD),
                   "--graph-results-dir", str(results)])
    assert rc == 0
    (config_file,) = runs.glob("*/config.resolved.yaml")
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert cfg["menhir_commit"] == "e" * 40
    assert cfg["graph"]["build_menhir_commit"] == "m" * 40 and cfg["graph"]["neo4j_name"] == _name()
