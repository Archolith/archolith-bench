"""Menhir layer surfaces read at an exact commit (Menhir `benchmarks/layer-surface.toml`, P8a.0).

A layer is marked by the last commit, up to the build commit, that changed any of its declared files.
Everything here reads git objects at that commit, never the working tree.
"""

from __future__ import annotations

import fnmatch
import hashlib
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

DECLARATION = "benchmarks/layer-surface.toml"


class SurfaceError(RuntimeError):
    pass


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SurfaceError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()[:300]}")
    return proc.stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


def load_declaration(repo: Path, commit: str) -> dict:
    try:
        text = git(repo, "show", f"{commit}:{DECLARATION}")
    except SurfaceError as exc:
        raise SurfaceError(f"{commit[:12]} has no {DECLARATION}; it cannot build a stored layer") from exc
    decl = tomllib.loads(text)
    if decl.get("version") != 1:
        raise SurfaceError(f"{DECLARATION} at {commit[:12]}: unsupported version {decl.get('version')!r}")
    return decl


def tracked_files(repo: Path, commit: str) -> list[str]:
    return git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()


@dataclass(frozen=True)
class LayerSurface:
    kind: str
    commit: str          # build commit the surface was read at
    files: tuple[str, ...]
    layer_commit: str    # last commit <= commit that changed any of files
    digest: str          # sha256 over the files' contents at commit (CRLF -> LF)
    parent: str | None
    stored: bool


def layer_files(decl: dict, kind: str, tracked: list[str]) -> list[str]:
    layers = decl.get("layers", {})
    if kind not in layers:
        raise SurfaceError(f"layer {kind!r} is not declared (have: {', '.join(sorted(layers))})")
    layer = layers[kind]
    files = set(layer.get("entries", [])) | set(layer.get("files", [])) | set(layer.get("paths", []))
    for pattern in layer.get("globs", []):
        hits = [f for f in tracked if fnmatch.fnmatchcase(f, pattern)]
        if not hits:
            raise SurfaceError(f"layer {kind}: glob {pattern} matches no tracked file")
        files.update(hits)
    missing = sorted(files - set(tracked))
    if missing:
        raise SurfaceError(f"layer {kind}: declared files not tracked at this commit: {', '.join(missing[:5])}")
    return sorted(files)


def content_digest(repo: Path, commit: str, files: list[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
        blob = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{rel}"], capture_output=True).stdout
        h.update(rel.encode() + b"\0" + blob.replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def read_surface(repo: Path, commit: str, kind: str) -> LayerSurface:
    decl = load_declaration(repo, commit)
    files = layer_files(decl, kind, tracked_files(repo, commit))
    last = git(repo, "log", "-1", "--format=%H", commit, "--", *files).strip()
    if not last:
        raise SurfaceError(f"layer {kind}: no commit touches its files up to {commit[:12]}")
    layer = decl["layers"][kind]
    return LayerSurface(kind=kind, commit=commit, files=tuple(files), layer_commit=last,
                        digest=content_digest(repo, commit, files), parent=layer.get("parent"),
                        stored=layer.get("stored", True) is not False)
