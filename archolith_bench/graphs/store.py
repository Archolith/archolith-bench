"""Layered, reusable Menhir graph store (plan P8a).

A layer is a Docker volume holding a stopped Neo4j ``/data`` plus a JSON manifest. Layers are
immutable once complete and are kept until removed by name. A completed layer volume is only ever
mounted read-only as the source of a copy: Menhir startup and recall both write the graph, so every
serve and every child build runs on a throwaway work volume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from archolith_bench.graphs.surface import LayerSurface

NEO4J_IMAGE = "neo4j:5.26-community"
RESERVED_PORTS = {7474, 7687, 7688}
DOCKER_CONTEXT = "ubuntu-server"
LABEL = "archolith.graph-layer"
NAME_RE = re.compile(r"[^a-z0-9-]+")

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def run_cmd(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True)


class StoreError(RuntimeError):
    pass


def default_root() -> Path:
    return Path(os.environ.get("ARCHOLITH_GRAPH_STORE", Path.home() / ".archolith" / "graph-store"))


@dataclass(frozen=True)
class LayerIdentity:
    fixture: str
    corpus_sha256: str
    kind: str
    parent_id: str | None
    layer_commit: str
    surface_digest: str
    settings: dict = field(default_factory=dict)
    neo4j_image: str = NEO4J_IMAGE

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def id(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @property
    def volume(self) -> str:
        return f"graph-{NAME_RE.sub('-', self.fixture.lower()).strip('-')}-{self.kind}-{self.id[:12]}"

    @classmethod
    def for_surface(cls, surface: LayerSurface, *, fixture: str, corpus_sha256: str,
                    parent_id: str | None, settings: dict) -> LayerIdentity:
        if not surface.stored:
            raise StoreError(f"layer kind {surface.kind!r} is declared stored = false")
        if bool(surface.parent) != bool(parent_id):
            raise StoreError(f"layer {surface.kind}: declared parent {surface.parent!r} but parent_id={parent_id!r}")
        return cls(fixture=fixture, corpus_sha256=corpus_sha256, kind=surface.kind, parent_id=parent_id,
                   layer_commit=surface.layer_commit, surface_digest=surface.digest, settings=dict(settings))


class GraphStore:
    def __init__(self, root: Path | None = None, runner: Runner = run_cmd) -> None:
        self.root = Path(root) if root else default_root()
        self.run = runner

    # ------------------------------------------------------------------ manifests
    def _path(self, layer_id: str) -> Path:
        return self.root / "layers" / f"{layer_id}.json"

    def manifest(self, layer_id: str) -> dict | None:
        p = self._path(layer_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def manifests(self) -> list[dict]:
        d = self.root / "layers"
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []

    def _write(self, m: dict) -> None:
        p = self._path(m["id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(p)

    # ------------------------------------------------------------------ decisions
    def decide(self, ident: LayerIdentity) -> str:
        """'reuse' (complete layer with this identity), 'resume' (an unfinished build), or 'build'."""
        m = self.manifest(ident.id)
        if m is None:
            return "build"
        if m.get("identity") != json.loads(ident.canonical()):
            raise StoreError(f"manifest {ident.id[:12]} does not match its identity; refusing to use it")
        if m["status"] == "complete":
            if not self._volume_exists(m["volume"]):
                raise StoreError(f"layer {ident.id[:12]} is complete but volume {m['volume']} is missing")
            return "reuse"
        return "resume"

    def _lock_path(self, layer_id: str) -> Path:
        return self.root / "layers" / f"{layer_id}.lock"

    def _acquire(self, layer_id: str) -> None:
        # One writer per layer: `docker volume create` silently reuses an existing name, so two
        # builders (or a resumer racing a builder) would otherwise ingest into the same work volume.
        p = self._lock_path(layer_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise StoreError(f"layer {layer_id[:12]} is locked by another build ({p.read_text(encoding='utf-8')}); "
                             f"if that build is dead, release it explicitly") from None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "host": os.environ.get("COMPUTERNAME", ""), "at": _now()}))

    def release(self, layer_id: str) -> None:
        """Drop a build lock left by a dead build (operator action)."""
        self._lock_path(layer_id).unlink(missing_ok=True)

    def begin(self, ident: LayerIdentity, *, build_commit: str, extra: dict | None = None) -> dict:
        """Start or resume a build under the layer's lock; returns the manifest, whose ``work_volume``
        the producer writes. The lock is held until ``complete`` (or an explicit ``release``)."""
        self._acquire(ident.id)
        try:
            return self._begin_locked(ident, build_commit=build_commit, extra=extra)
        except BaseException:
            self.release(ident.id)
            raise

    def _begin_locked(self, ident: LayerIdentity, *, build_commit: str, extra: dict | None) -> dict:
        state = self.decide(ident)
        if state == "reuse":
            raise StoreError(f"layer {ident.id[:12]} is complete; nothing to build")
        if ident.parent_id:
            parent = self.manifest(ident.parent_id)
            if not parent or parent["status"] != "complete":
                raise StoreError(f"parent layer {ident.parent_id[:12]} is not a complete stored layer")
        if state == "build":
            m = {"id": ident.id, "identity": json.loads(ident.canonical()), "volume": ident.volume,
                 "work_volume": f"{ident.volume}-work", "status": "building", "attempts": [],
                 "created_at": _now(), **(extra or {})}
            self._create_volume(m["work_volume"], ident, work=True)
            if ident.parent_id:
                self.copy_volume(self.manifest(ident.parent_id)["volume"], m["work_volume"])
        else:
            m = self.manifest(ident.id)
            if not self._volume_exists(m["work_volume"]):
                raise StoreError(f"unfinished layer {ident.id[:12]} lost its work volume {m['work_volume']}")
        m["attempts"].append({"started_at": _now(), "build_commit": build_commit})
        self._write(m)
        return m

    def complete(self, layer_id: str, *, summary: dict | None = None) -> dict:
        """Snapshot the (stopped) work volume into the immutable layer volume and mark it complete."""
        m = self.manifest(layer_id)
        if not m or m["status"] != "building":
            raise StoreError(f"layer {layer_id[:12]} is not being built")
        if not self._lock_path(layer_id).exists():
            raise StoreError(f"layer {layer_id[:12]} has no build lock; complete it from the build that began it")
        if self._volume_in_use(m["work_volume"]):
            raise StoreError(f"stop the containers using {m['work_volume']} before completing the layer")
        if self._volume_exists(m["volume"]):
            raise StoreError(f"layer volume {m['volume']} already exists; refusing to overwrite it")
        ident = LayerIdentity(**m["identity"])
        self._create_volume(m["volume"], ident, work=False)
        self.copy_volume(m["work_volume"], m["volume"])
        m.update(status="complete", finished_at=_now(), summary=summary or {})
        self._write(m)
        self.release(layer_id)
        return m

    # ------------------------------------------------------------------ serving
    def serve_volume(self, layer_id: str, run_name: str) -> str:
        """A fresh throwaway copy of a complete layer, for one serve. Never the layer itself."""
        m = self.manifest(layer_id)
        if not m or m["status"] != "complete":
            raise StoreError(f"layer {layer_id[:12]} is not a complete stored layer")
        work = f"{m['volume']}-serve-{NAME_RE.sub('-', run_name.lower()).strip('-')}"
        if self._volume_exists(work):
            raise StoreError(f"serve volume {work} already exists; remove it or pick another run name")
        self._create_volume(work, LayerIdentity(**m["identity"]), work=True)
        self.copy_volume(m["volume"], work)
        return work

    def start_neo4j(self, volume: str, *, name: str, port: int, password: str) -> None:
        if port in RESERVED_PORTS:
            raise StoreError(f"port {port} is reserved for shared instances")
        if any(m["volume"] == volume for m in self.manifests() if m["status"] == "complete"):
            raise StoreError(f"{volume} is a stored layer; serve a copy (serve_volume) instead")
        self._ok(["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{port}:7687",
                  "-e", f"NEO4J_AUTH=neo4j/{password}", "-v", f"{volume}:/data", NEO4J_IMAGE])

    # ------------------------------------------------------------------ docker
    def check_context(self) -> None:
        out = self._ok(["docker", "context", "show"]).stdout.strip()
        if out != DOCKER_CONTEXT:
            raise StoreError(f"docker context is {out!r}; graph layers live on {DOCKER_CONTEXT!r}")

    def copy_volume(self, src: str, dst: str) -> None:
        self._ok(["docker", "run", "--rm", "-v", f"{src}:/from:ro", "-v", f"{dst}:/to", NEO4J_IMAGE,
                  "sh", "-c", "cp -a /from/. /to/"])

    def remove(self, layer_id: str, *, confirm: str) -> None:
        """Delete a layer. ``confirm`` must equal its volume name; nothing calls this implicitly."""
        m = self.manifest(layer_id)
        if not m:
            raise StoreError(f"no layer {layer_id[:12]}")
        if confirm != m["volume"]:
            raise StoreError(f"confirm with the volume name {m['volume']!r}")
        for vol in (m["volume"], m["work_volume"]):
            if self._volume_exists(vol):
                self._ok(["docker", "volume", "rm", vol])
        self._path(layer_id).unlink()

    def _create_volume(self, name: str, ident: LayerIdentity, *, work: bool) -> None:
        if self._volume_exists(name):
            raise StoreError(f"volume {name} already exists")
        self._ok(["docker", "volume", "create", "--label", f"{LABEL}={ident.id}",
                  "--label", f"{LABEL}.kind={ident.kind}", "--label", f"{LABEL}.commit={ident.layer_commit}",
                  "--label", f"{LABEL}.role={'work' if work else 'layer'}", name])

    def _volume_exists(self, name: str) -> bool:
        return self.run(["docker", "volume", "inspect", name]).returncode == 0

    def _volume_in_use(self, name: str) -> bool:
        out = self._ok(["docker", "ps", "-q", "--filter", f"volume={name}"]).stdout.strip()
        return bool(out)

    def _ok(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        proc = self.run(args)
        if proc.returncode != 0:
            raise StoreError(f"{' '.join(args[:4])} ... failed: {(proc.stderr or '').strip()[:300]}")
        return proc


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
