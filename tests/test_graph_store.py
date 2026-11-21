"""Offline tests for the layered graph store (P8a.1): scratch git repos and a fake docker."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from archolith_bench.graphs import refs
from archolith_bench.graphs.store import GraphStore, LayerIdentity, StoreError
from archolith_bench.graphs.surface import SurfaceError, read_surface

DECL = """version = 1
[layers.ingest]
entries = ["src/menhir/ingest.py"]
globs = ["src/menhir/ingest_*.py"]
files = ["src/menhir/shared.py"]
paths = ["uv.lock"]
[layers.lifecycle]
parent = "ingest"
entries = ["src/menhir/lifecycle.py"]
[layers.recall]
stored = false
entries = ["src/menhir/recall.py"]
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def menhir(tmp_path: Path) -> Path:
    repo = tmp_path / "menhir"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _commit(repo, {"benchmarks/layer-surface.toml": DECL, "src/menhir/ingest.py": "a", "src/menhir/ingest_x.py": "b",
                   "src/menhir/shared.py": "c", "src/menhir/lifecycle.py": "d", "src/menhir/recall.py": "e",
                   "uv.lock": "f"}, "base")
    return repo


# ------------------------------------------------------------------- surface
def test_layer_commit_ignores_commits_outside_the_layer(menhir):
    base = _git(menhir, "rev-parse", "HEAD")
    later = _commit(menhir, {"src/menhir/recall.py": "e2"}, "recall only (P7-like)")
    s0, s1 = read_surface(menhir, base, "ingest"), read_surface(menhir, later, "ingest")
    assert s1.commit == later and s1.layer_commit == base == s0.layer_commit
    assert s1.digest == s0.digest


@pytest.mark.parametrize("path", ["src/menhir/ingest.py", "src/menhir/ingest_x.py", "src/menhir/shared.py", "uv.lock"])
def test_layer_commit_moves_when_any_layer_file_changes(menhir, path):
    base = read_surface(menhir, "HEAD", "ingest")
    later = _commit(menhir, {path: "changed"}, f"touch {path}")
    s = read_surface(menhir, later, "ingest")
    assert s.layer_commit == later and s.digest != base.digest


def test_declaration_change_changes_the_digest(menhir):
    base = read_surface(menhir, "HEAD", "ingest")
    _commit(menhir, {"src/menhir/extra.py": "x"}, "add file")  # not yet declared
    later = _commit(menhir, {"benchmarks/layer-surface.toml": DECL.replace(
        'files = ["src/menhir/shared.py"]', 'files = ["src/menhir/shared.py", "src/menhir/extra.py"]')}, "declare")
    s = read_surface(menhir, later, "ingest")
    assert "src/menhir/extra.py" in s.files and s.digest != base.digest


def test_commit_without_declaration_or_with_dead_glob_is_refused(menhir, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    _git(old, "init", "-q")
    _git(old, "config", "user.email", "t@example.com")
    _git(old, "config", "user.name", "t")
    with pytest.raises(SurfaceError, match="has no benchmarks/layer-surface.toml"):
        read_surface(old, _commit(old, {"README": "x"}, "no decl"), "ingest")
    gone = _commit(menhir, {"benchmarks/layer-surface.toml": DECL.replace("ingest_*.py", "intake_*.py")}, "bad glob")
    with pytest.raises(SurfaceError, match="matches no tracked file"):
        read_surface(menhir, gone, "ingest")


# ------------------------------------------------------------------- identity
def _ident(**kw) -> LayerIdentity:
    base = dict(fixture="F5", corpus_sha256="c" * 64, kind="ingest", parent_id=None, layer_commit="a" * 40,
                surface_digest="d" * 64, settings={"model": "gpt-6-luna", "tier": "flex"})
    base.update(kw)
    return LayerIdentity(**base)


def test_identity_changes_with_every_field():
    a = _ident()
    variants = [replace(a, fixture="F6"), replace(a, corpus_sha256="e" * 64), replace(a, kind="lifecycle"),
                replace(a, parent_id="p" * 64), replace(a, layer_commit="b" * 40), replace(a, surface_digest="f" * 64),
                replace(a, settings={"model": "gpt-6-luna", "tier": "default"}), replace(a, neo4j_image="neo4j:6")]
    assert len({a.id, *(v.id for v in variants)}) == len(variants) + 1
    assert a.volume.startswith("graph-f5-ingest-") and len(a.volume.split("-")[-1]) == 12


def test_identity_refuses_unstored_kind_and_parent_mismatch(menhir):
    with pytest.raises(StoreError, match="stored = false"):
        LayerIdentity.for_surface(read_surface(menhir, "HEAD", "recall"), fixture="F5", corpus_sha256="c",
                                  parent_id=None, settings={})
    with pytest.raises(StoreError, match="declared parent"):
        LayerIdentity.for_surface(read_surface(menhir, "HEAD", "lifecycle"), fixture="F5", corpus_sha256="c",
                                  parent_id=None, settings={})


# ------------------------------------------------------------------- store with a fake docker
class FakeDocker:
    def __init__(self, context: str = "ubuntu-server"):
        self.volumes: dict[str, str] = {}  # name -> content marker
        self.in_use: set[str] = set()
        self.ro_mounts: list[str] = []
        self.rw_mounts: list[str] = []
        self.context = context
        self.calls: list[list[str]] = []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        ok = subprocess.CompletedProcess(args, 0, "", "")
        if args[:3] == ["docker", "context", "show"]:
            return subprocess.CompletedProcess(args, 0, self.context + "\n", "")
        if args[:3] == ["docker", "volume", "inspect"]:
            return ok if args[3] in self.volumes else subprocess.CompletedProcess(args, 1, "", "no such volume")
        if args[:3] == ["docker", "volume", "create"]:
            self.volumes.setdefault(args[-1], "")
            return ok
        if args[:3] == ["docker", "volume", "rm"]:
            self.volumes.pop(args[3])
            return ok
        if args[:3] == ["docker", "ps", "-q"]:
            vol = args[-1].split("=", 1)[1]
            return subprocess.CompletedProcess(args, 0, "abc\n" if vol in self.in_use else "", "")
        if args[:3] == ["docker", "run", "--rm"]:
            src, dst = args[4].split(":")[0], args[6].split(":")[0]
            assert args[4].endswith(":/from:ro")
            self.ro_mounts.append(src)
            self.volumes[dst] = self.volumes[src]
            return ok
        if args[:3] == ["docker", "run", "-d"]:
            self.rw_mounts.append(args[args.index("-v") + 1].split(":")[0])
            return ok
        raise AssertionError(f"unexpected docker call {args}")


@pytest.fixture
def store(tmp_path):
    return GraphStore(tmp_path / "store", runner=FakeDocker())


def test_build_complete_then_reuse(store):
    ident = _ident()
    assert store.decide(ident) == "build"
    m = store.begin(ident, build_commit="h" * 40)
    store.run.volumes[m["work_volume"]] = "ingested"
    done = store.complete(ident.id, summary={"episodes": 3})
    assert done["status"] == "complete" and store.run.volumes[ident.volume] == "ingested"
    assert store.decide(ident) == "reuse"
    with pytest.raises(StoreError, match="nothing to build"):
        store.begin(ident, build_commit="h" * 40)


def test_dead_build_resumes_on_the_same_work_volume_after_explicit_release(store):
    ident = _ident()
    m = store.begin(ident, build_commit="h1" * 20)
    store.run.volumes[m["work_volume"]] = "half"
    with pytest.raises(StoreError, match="locked by another build"):
        store.begin(ident, build_commit="h2" * 20)
    store.release(ident.id)  # operator: the first build is dead
    assert store.decide(ident) == "resume"
    m2 = store.begin(ident, build_commit="h2" * 20)
    assert m2["work_volume"] == m["work_volume"] and store.run.volumes[m["work_volume"]] == "half"
    assert [a["build_commit"] for a in m2["attempts"]] == ["h1" * 20, "h2" * 20]


def test_complete_refuses_running_work_volume_existing_layer_and_missing_lock(store):
    ident = _ident()
    m = store.begin(ident, build_commit="h" * 40)
    store.run.in_use.add(m["work_volume"])
    with pytest.raises(StoreError, match="stop the containers"):
        store.complete(ident.id)
    store.run.in_use.clear()
    store.run.volumes[ident.volume] = "someone else's"
    with pytest.raises(StoreError, match="refusing to overwrite"):
        store.complete(ident.id)
    del store.run.volumes[ident.volume]
    store.release(ident.id)
    with pytest.raises(StoreError, match="no build lock"):
        store.complete(ident.id)


def test_child_build_needs_a_complete_parent_and_copies_it_read_only(store):
    parent = _ident()
    child = _ident(kind="lifecycle", parent_id=parent.id)
    with pytest.raises(StoreError, match="not a complete stored layer"):
        store.begin(child, build_commit="h" * 40)
    store.begin(parent, build_commit="h" * 40)
    store.run.volumes[f"{parent.volume}-work"] = "base"
    store.complete(parent.id)
    m = store.begin(child, build_commit="h" * 40)
    assert store.run.volumes[m["work_volume"]] == "base" and parent.volume in store.run.ro_mounts


def test_serving_uses_a_copy_and_never_mounts_a_layer(store):
    ident = _ident()
    store.begin(ident, build_commit="h" * 40)
    store.run.volumes[f"{ident.volume}-work"] = "base"
    store.complete(ident.id)
    work = store.serve_volume(ident.id, "P7 run 1")
    assert work.endswith("-serve-p7-run-1") and store.run.volumes[work] == "base"
    with pytest.raises(StoreError, match="already exists"):
        store.serve_volume(ident.id, "P7 run 1")
    with pytest.raises(StoreError, match="is a stored layer"):
        store.start_neo4j(ident.volume, name="n", port=17999, password="pw")
    with pytest.raises(StoreError, match="reserved"):
        store.start_neo4j(work, name="n", port=7687, password="pw")
    store.start_neo4j(work, name="n", port=17999, password="pw")
    assert store.run.rw_mounts == [work] and ident.volume not in store.run.rw_mounts


def test_remove_needs_the_volume_name_and_manifest_must_match_identity(store):
    ident = _ident()
    store.begin(ident, build_commit="h" * 40)
    store.complete(ident.id)
    with pytest.raises(StoreError, match="confirm with the volume name"):
        store.remove(ident.id, confirm="yes")
    m = store.manifest(ident.id)
    m["identity"]["layer_commit"] = "z" * 40
    store._write(m)
    with pytest.raises(StoreError, match="does not match its identity"):
        store.decide(ident)


def test_complete_layer_with_missing_volume_is_an_error_not_a_rebuild(store):
    ident = _ident()
    store.begin(ident, build_commit="h" * 40)
    store.complete(ident.id)
    del store.run.volumes[ident.volume]
    with pytest.raises(StoreError, match="volume .* is missing"):
        store.decide(ident)


def test_docker_context_guard(tmp_path):
    with pytest.raises(StoreError, match="docker context is 'default'"):
        GraphStore(tmp_path, runner=FakeDocker(context="default")).check_context()


# ------------------------------------------------------------------- refs
def test_menhir_worktree_is_detached_per_commit_and_refuses_dirty_or_moved(menhir, tmp_path):
    wt = tmp_path / "wt"
    commit, path = refs.menhir_worktree(menhir, "HEAD", wt)
    assert path.name == f"menhir-{commit[:12]}" and _git(path, "rev-parse", "HEAD") == commit
    assert refs.menhir_worktree(menhir, commit, wt) == (commit, path)  # reused
    (path / "src/menhir/ingest.py").write_text("local edit", encoding="utf-8")
    with pytest.raises(SurfaceError, match="local changes"):
        refs.menhir_worktree(menhir, commit, wt)
    _git(path, "checkout", "-q", "--", ".")
    later = _commit(menhir, {"src/menhir/recall.py": "z"}, "later")
    _git(path, "checkout", "-q", "--detach", later)
    with pytest.raises(SurfaceError, match="not " + commit[:12]):
        refs.menhir_worktree(menhir, commit, wt)
    with pytest.raises(SurfaceError, match="no .venv"):
        refs.venv_python(path)
