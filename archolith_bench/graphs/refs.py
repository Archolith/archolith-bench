"""Resolve a Menhir ref to a clean, detached worktree at that exact commit (one per commit)."""

from __future__ import annotations

from pathlib import Path

from archolith_bench.graphs.surface import SurfaceError, git, resolve_commit


def menhir_worktree(repo: Path, ref: str, worktrees: Path) -> tuple[str, Path]:
    """Return (commit, path) for a detached worktree of ``repo`` at ``ref``, creating it if needed.

    An existing worktree is reused only if it is still at that commit and clean; a checkout someone
    else is working in is never used (the path is keyed by commit, and created detached).
    """
    commit = resolve_commit(repo, ref)
    path = Path(worktrees) / f"menhir-{commit[:12]}"
    if path.exists():
        head = git(path, "rev-parse", "HEAD").strip()
        if head != commit:
            raise SurfaceError(f"{path} is at {head[:12]}, not {commit[:12]}; remove it or pick another directory")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", str(path), commit)
    dirty = git(path, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise SurfaceError(f"{path} has local changes; a stored layer needs a clean tree:\n{dirty[:300]}")
    return commit, path


def venv_python(path: Path) -> Path:
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (path / rel).is_file():
            return path / rel
    raise SurfaceError(f"{path} has no .venv; create it once for this commit (python -m venv .venv && "
                       f".venv/Scripts/python -m pip install -e .) and rerun")
