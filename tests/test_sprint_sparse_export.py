"""Sparse blobless export for huge-repo sprint fixtures (live_executor._export_sparse_commit)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from archolith_bench.core.agent_task.models import RepoPin
from archolith_bench.engineering_evidence.live_executor import LiveSprintConfig, OpenCodeSprintExecutor
from archolith_bench.engineering_evidence.sprint import SprintFixture


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, body in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(body, encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", message, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture()
def origin(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "origin"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    # GitHub serves any reachable commit by id; a local origin must opt in.
    _git("config", "uploadpack.allowReachableSHA1InWant", "true", cwd=repo)
    old = _commit(repo, {"fs/ext4/inode.c": "old\n", "Documentation/filesystems/ext4/index.rst": "doc\n",
                         "drivers/net/x.c": "net\n"}, "one")
    new = _commit(repo, {"fs/ext4/inode.c": "new\n"}, "two")
    return repo, old, new


def _executor(tmp_path: Path, exports: list[Path] | None = None) -> OpenCodeSprintExecutor:
    def fake_export(pin: RepoPin, dest: Path, cache: Path) -> Path:
        (exports if exports is not None else []).append(dest)
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    return OpenCodeSprintExecutor(
        LiveSprintConfig(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache",
                         source_config=tmp_path / "unused.json"),
        export_fn=fake_export,
        resolve_cmd_fn=lambda: ["opencode"],
        check_memory_fn=lambda url: None,
    )


def _fixture(origin_repo: Path, commit: str, paths: tuple[str, ...]) -> SprintFixture:
    pin = RepoPin(name="linux-ext4", url=origin_repo.as_uri(), commit=commit)
    return SprintFixture(name="linux-ext4", description="", repo=pin, sessions=(), sparse_paths=paths)


def test_sparse_export_copies_only_the_sparse_paths_at_the_pinned_commit(tmp_path, origin):
    repo, old, _new = origin
    ex = _executor(tmp_path)
    dest = ex._export_sparse_commit(_fixture(repo, old, ("fs/ext4", "Documentation/filesystems/ext4")), tmp_path / "out")
    assert (dest / "fs/ext4/inode.c").read_text(encoding="utf-8") == "old\n"
    assert (dest / "Documentation/filesystems/ext4/index.rst").read_text(encoding="utf-8") == "doc\n"
    assert not (dest / "drivers").exists()
    assert not (dest / ".git").exists()


def test_sparse_export_uses_its_own_cache_clone(tmp_path, origin):
    repo, old, _new = origin
    _executor(tmp_path)._export_sparse_commit(_fixture(repo, old, ("fs/ext4",)), tmp_path / "out")
    assert (tmp_path / "cache" / "linux-ext4.sparse" / ".git").exists()
    assert not (tmp_path / "cache" / "linux-ext4").exists()  # export_commit's shared clone is untouched


def test_a_cached_clone_still_gets_a_new_fixtures_paths(tmp_path, origin):
    repo, old, _new = origin
    ex = _executor(tmp_path)
    ex._export_sparse_commit(_fixture(repo, old, ("fs/ext4",)), tmp_path / "a")
    dest = ex._export_sparse_commit(_fixture(repo, old, ("drivers/net",)), tmp_path / "b")
    assert (dest / "drivers/net/x.c").read_text(encoding="utf-8") == "net\n"


def test_a_missing_sparse_path_fails_instead_of_exporting_less(tmp_path, origin):
    repo, old, _new = origin
    with pytest.raises(FileNotFoundError, match="fs/xfs"):
        _executor(tmp_path)._export_sparse_commit(_fixture(repo, old, ("fs/ext4", "fs/xfs")), tmp_path / "out")


def test_ensure_checkout_routes_sparse_fixtures_away_from_export_commit(tmp_path, origin):
    repo, old, _new = origin
    exports: list[Path] = []
    ex = _executor(tmp_path, exports)
    checkout = ex._ensure_checkout(_fixture(repo, old, ("fs/ext4",)), "no_menhir")
    assert exports == []
    assert (checkout / "fs/ext4/inode.c").exists()
