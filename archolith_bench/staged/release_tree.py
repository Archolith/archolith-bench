"""Serve a released Menhir: resolve a ``v*`` tag and keep one locked worktree per release.

A campaign resolves its release once, records the tag and commit, and serves that tree for the
whole run. Trees live under a cache root, one directory per tag, each with its own ``uv sync
--frozen`` venv; a tree is reused only when its HEAD is the tag's commit and its install finished.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
#: Written after a tree's venv is installed; holds the commit it was installed for.
READY_MARKER = ".bench-release-ready"
#: The sprint serves MENHIR_LOOPBACK_PROFILE=lean; a tree without it would serve every tool.
LEAN_PROFILE_FILE = Path("src/menhir/mcp/local_profile.py")


class ReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseTree:
    tag: str
    commit: str
    path: Path


def default_cache_root() -> Path:
    env = os.environ.get("ARCHOLITH_BENCH_RELEASE_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "archolith-bench" / "menhir-releases"


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ReleaseError(f"git {' '.join(args)} failed in {repo}: {detail.strip()}") from exc


def newest_tag(tags: list[str]) -> str | None:
    """The highest ``vX.Y.Z`` (numeric, not lexical: v0.3.20 > v0.3.9); other tags are ignored."""
    parsed = [(tuple(int(p) for p in m.groups()), t) for t in tags if (m := _TAG.match(t))]
    return max(parsed)[1] if parsed else None


def resolve_release(repo: Path, spec: str = "latest", *, fetch: bool = True) -> tuple[str, str]:
    """(tag, commit) for ``spec``: ``latest`` or an exact ``vX.Y.Z`` tag."""
    if fetch:
        _git(repo, "fetch", "-q", "--tags", "origin")
    if spec == "latest":
        tag = newest_tag(_git(repo, "tag", "--list", "v*").splitlines())
        if tag is None:
            raise ReleaseError(f"{repo} has no vX.Y.Z tags")
    elif _TAG.match(spec):
        tag = spec
    else:
        raise ReleaseError(f"--menhir-release must be 'latest' or vX.Y.Z, not {spec!r}")
    return tag, _git(repo, "rev-parse", f"{tag}^{{commit}}")


def has_lean_profile(tree: Path) -> bool:
    try:
        return "LEAN_TOOLS" in (tree / LEAN_PROFILE_FILE).read_text(encoding="utf-8")
    except OSError:
        return False


def release_tree(repo: Path, spec: str = "latest", *, cache_root: Path | None = None,
                 fetch: bool = True, run=subprocess.run) -> ReleaseTree:
    """The worktree for ``spec``'s release, created and installed on first use.

    An existing directory is reused only when its HEAD is the tag's commit and the ready marker
    names that commit; anything else is refused rather than deleted.
    """
    tag, commit = resolve_release(repo, spec, fetch=fetch)
    path = (cache_root or default_cache_root()) / tag
    marker = path / READY_MARKER
    if path.exists():
        head = _git(path, "rev-parse", "HEAD") if (path / ".git").exists() else ""
        if head != commit:
            raise ReleaseError(f"{path} is not {tag} ({commit[:12]}); HEAD is {head[:12] or 'not a worktree'}. "
                               "Remove it (git worktree remove) and rerun.")
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
            return ReleaseTree(tag, commit, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _git(repo, "worktree", "add", "-q", "--detach", str(path), commit)
    # Locked install from the release's own uv.lock: the served dependencies are the released ones.
    done = run(["uv", "sync", "--frozen", "--no-dev", "--project", str(path)],
               capture_output=True, text=True)
    if done.returncode != 0:
        raise ReleaseError(f"uv sync for {tag} failed: {(done.stderr or done.stdout).strip()[-800:]}")
    marker.write_text(commit, encoding="utf-8")
    return ReleaseTree(tag, commit, path)
