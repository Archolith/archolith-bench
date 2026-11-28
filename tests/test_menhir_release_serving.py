"""Sprint serving of a released Menhir, and cross-run comparisons that flag build mismatches."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from archolith_bench.staged.release_tree import (
    LEAN_PROFILE_FILE,
    READY_MARKER,
    ReleaseError,
    has_lean_profile,
    newest_tag,
    release_tree,
    resolve_release,
)

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "menhir"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.invalid")
    _git(r, "config", "user.name", "t")
    for i, tag in enumerate(("v0.3.9", "v0.3.20", "v0.3.10-rc1")):
        (r / "f.txt").write_text(str(i), encoding="utf-8")
        _git(r, "add", "f.txt")
        _git(r, "commit", "-q", "-m", tag)
        _git(r, "tag", tag)
    return r


def _ok_sync(cmd, **kwargs):
    assert cmd[:4] == ["uv", "sync", "--frozen", "--no-dev"]
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_newest_tag_is_numeric_and_ignores_non_release_tags() -> None:
    assert newest_tag(["v0.3.9", "v0.3.20", "v0.3.10", "v1.0.0-rc1", "nightly"]) == "v0.3.20"
    assert newest_tag(["nightly"]) is None


def test_resolve_release_latest_and_exact(repo: Path) -> None:
    tag, commit = resolve_release(repo, "latest", fetch=False)
    assert tag == "v0.3.20" and commit == _git(repo, "rev-parse", "v0.3.20^{commit}")
    assert resolve_release(repo, "v0.3.9", fetch=False)[0] == "v0.3.9"
    with pytest.raises(ReleaseError):
        resolve_release(repo, "main", fetch=False)
    with pytest.raises(ReleaseError):
        resolve_release(repo, "v9.9.9", fetch=False)


def test_release_tree_installs_once_then_reuses(repo: Path, tmp_path: Path) -> None:
    calls = []

    def sync(cmd, **kw):
        calls.append(cmd)
        return _ok_sync(cmd, **kw)

    cache = tmp_path / "cache"
    first = release_tree(repo, "latest", cache_root=cache, fetch=False, run=sync)
    assert first.path == cache / "v0.3.20" and _git(first.path, "rev-parse", "HEAD") == first.commit
    assert (first.path / READY_MARKER).read_text(encoding="utf-8") == first.commit
    again = release_tree(repo, "latest", cache_root=cache, fetch=False, run=sync)
    assert again == first and len(calls) == 1


def test_release_tree_refuses_a_directory_at_another_commit(repo: Path, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    tree = release_tree(repo, "v0.3.9", cache_root=cache, fetch=False, run=_ok_sync)
    (cache / "v0.3.20").mkdir()
    with pytest.raises(ReleaseError, match="not v0.3.20"):
        release_tree(repo, "v0.3.20", cache_root=cache, fetch=False, run=_ok_sync)
    _git(tree.path, "checkout", "-q", "--detach", "v0.3.20")
    with pytest.raises(ReleaseError, match="not v0.3.9"):
        release_tree(repo, "v0.3.9", cache_root=cache, fetch=False, run=_ok_sync)


def test_failed_install_is_not_marked_ready(repo: Path, tmp_path: Path) -> None:
    def bad(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="no solution")

    cache = tmp_path / "cache"
    with pytest.raises(ReleaseError, match="no solution"):
        release_tree(repo, "latest", cache_root=cache, fetch=False, run=bad)
    assert not (cache / "v0.3.20" / READY_MARKER).exists()
    # The next run retries the install instead of serving a half-built tree.
    assert release_tree(repo, "latest", cache_root=cache, fetch=False, run=_ok_sync).tag == "v0.3.20"


def test_has_lean_profile(tmp_path: Path) -> None:
    assert not has_lean_profile(tmp_path)
    (tmp_path / LEAN_PROFILE_FILE).parent.mkdir(parents=True)
    (tmp_path / LEAN_PROFILE_FILE).write_text("CORE_TOOLS = set()\n", encoding="utf-8")
    assert not has_lean_profile(tmp_path)
    (tmp_path / LEAN_PROFILE_FILE).write_text("LEAN_TOOLS = frozenset()\n", encoding="utf-8")
    assert has_lean_profile(tmp_path)


def test_live_sprint_refuses_a_served_tree_without_the_lean_profile(tmp_path: Path, capsys) -> None:
    cli = _load("sprint_cli_rel", "scripts/run_longitudinal_sprint_bench.py")
    rc = cli.main(["--fixture", str(ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json"), "--repeats", "1",
                   "--live", "--max-usd", "5", "--est-usd-per-session", "0.01", "--server-rates", "0.1,0.01,0.5",
                   "--serve", str(tmp_path), "--graph", str(ROOT / "builds" / "sprint-v2.yaml"),
                   "--out-root", str(tmp_path / "out")])
    assert rc == 2 and "lean" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def _report(release, commit, prompt_set="sprint-v2.1", acc=0.8) -> dict:
    row = {"repeat": 1, "accuracy": acc, "complete": True, "primary_efficiency_tokens": 1000,
           "client_cost_usd": 0.01, "server_cost_usd": 0.002}
    prov = {"menhir_commit": commit, "prompt_set": prompt_set, "model": "m", "agent": "archolith-harness"}
    if release is not ...:
        prov["menhir_release"] = release
    return {"fixture": "fx", "run_id": f"r-{commit}", "rubric_version": "sprint-v2", "provenance": prov,
            "analysis": {"per_chain": {"menhir_longitudinal": [row]}}}


def test_compare_flags_different_menhir_builds(tmp_path: Path, capsys) -> None:
    cmp = _load("compare_cli", "scripts/compare_sprint_runs.py")
    dirs = []
    for i, rep in enumerate((_report("v0.3.20", "aaa"), _report("v0.3.21", "bbb", acc=0.9))):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "report.json").write_text(json.dumps(rep), encoding="utf-8")
        dirs.append(str(d))
    assert cmp.main([*dirs, "--strict"]) == 3
    out = capsys.readouterr().out
    assert "WARNING: these runs served different Menhir builds" in out
    assert "differs: menhir_release" in out and "differs: menhir_commit" in out
    assert cmp.main(dirs) == 0  # without --strict it reports and succeeds


def test_compare_same_build_reports_only_intended_differences(tmp_path: Path) -> None:
    sc = _load("sprint_compare_t", "archolith_bench/engineering_evidence/sprint_compare.py")
    a, b = _report("v0.3.20", "aaa", "sprint-v2"), _report("v0.3.20", "aaa", "sprint-v2.1")
    assert sc.mismatches([a, b]) == {"prompt_set": ["sprint-v2", "sprint-v2.1"]}
    # A run from before menhir_release existed is never silently matched to a recorded release.
    assert "menhir_release" in sc.mismatches([_report(..., "aaa"), _report("v0.3.20", "aaa")])
