"""Export a pinned commit into a plain directory, and the git argv for sealed checkouts."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

from archolith_bench.core.agent_task.models import RepoPin


def _ensure_source(pin: RepoPin, cache: Path) -> Path:
    """The local repository holding *pin*'s commit: cloned into *cache* once, and fetched
    only when the commit is missing, so concurrent exports never write to it."""
    source = Path(pin.local_path) if pin.local_path else cache / pin.name
    if not pin.local_path and not (source / ".git").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", pin.url, str(source)],
            check=True,
        )
    present = subprocess.run(
        ["git", "-C", str(source), "cat-file", "-e", f"{pin.commit}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if present.returncode != 0:
        subprocess.run(
            ["git", "-C", str(source), "fetch", "--quiet", "origin", pin.commit],
            check=False,
            capture_output=True,
        )
    return source


def export_commit(pin: RepoPin, dest: Path, cache: Path, source: Path | None = None) -> Path:
    """Write the tree of *pin* at its commit into *dest* (no .git), from *source* if given."""
    source = source if source is not None else _ensure_source(pin, cache)
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", pin.commit],
        check=True,
        capture_output=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


def _checkout_git(checkout: Path, autocrlf_off: bool = True) -> list[str]:
    git = [
        "git", "-C", str(checkout),
        "-c", "user.name=beacon-eval", "-c", "user.email=beacon-eval@example.invalid",
        "-c", "commit.gpgsign=false",
    ]
    return [*git, "-c", "core.autocrlf=false"] if autocrlf_off else git
