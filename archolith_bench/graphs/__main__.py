"""python -m archolith_bench.graphs: inspect layer surfaces and the stored-layer manifests.

  surface --menhir DIR --ref REF --kind ingest      layer commit + digest at a ref (read-only)
  plan    --menhir DIR --ref REF --kind ingest --fixture F5 --corpus FILE [--parent ID] [--settings JSON]
                                                    identity and reuse/resume/build decision (no writes)
  list | show ID | release ID | rm ID --confirm VOLUME

Building and serving layers for a fixture is done by that fixture's adapter (plan P8a.2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from archolith_bench.graphs.store import GraphStore, LayerIdentity, StoreError
from archolith_bench.graphs.surface import SurfaceError, read_surface, resolve_commit


def _surface(args):
    commit = resolve_commit(Path(args.menhir), args.ref)
    return read_surface(Path(args.menhir), commit, args.kind)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m archolith_bench.graphs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=None, help="store root (default $ARCHOLITH_GRAPH_STORE or "
                                                             "~/.archolith/graph-store)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("surface", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--menhir", required=True)
        p.add_argument("--ref", required=True)
        p.add_argument("--kind", required=True)
        if name == "plan":
            p.add_argument("--fixture", required=True)
            p.add_argument("--corpus", required=True, type=Path)
            p.add_argument("--parent", default=None)
            p.add_argument("--settings", default="{}")
    sub.add_parser("list")
    for name in ("show", "release"):
        sub.add_parser(name).add_argument("layer_id")
    p = sub.add_parser("rm")
    p.add_argument("layer_id")
    p.add_argument("--confirm", required=True, help="the layer's volume name")
    args = ap.parse_args(argv)
    store = GraphStore(args.store)

    try:
        if args.cmd == "surface":
            s = _surface(args)
            print(json.dumps({"kind": s.kind, "commit": s.commit, "layer_commit": s.layer_commit,
                              "digest": s.digest, "files": len(s.files), "stored": s.stored}, indent=1))
        elif args.cmd == "plan":
            s = _surface(args)
            ident = LayerIdentity.for_surface(
                s, fixture=args.fixture, corpus_sha256=hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
                parent_id=_full_id(store, args.parent), settings=json.loads(args.settings))
            print(json.dumps({"id": ident.id, "volume": ident.volume, "decision": store.decide(ident),
                              "identity": json.loads(ident.canonical())}, indent=1))
        elif args.cmd == "list":
            for m in store.manifests():
                i = m["identity"]
                print(f"{m['id'][:12]}  {m['status']:9} {i['fixture']:10} {i['kind']:10} "
                      f"layer@{i['layer_commit'][:12]}  {m['volume']}")
        elif args.cmd == "show":
            print(json.dumps(store.manifest(_full_id(store, args.layer_id)), indent=1))
        elif args.cmd == "release":
            store.release(_full_id(store, args.layer_id))
        elif args.cmd == "rm":
            store.remove(_full_id(store, args.layer_id), confirm=args.confirm)
    except (StoreError, SurfaceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _full_id(store: GraphStore, prefix: str | None) -> str | None:
    if not prefix:
        return None
    hits = [m["id"] for m in store.manifests() if m["id"].startswith(prefix)]
    if len(hits) != 1:
        raise StoreError(f"layer id {prefix!r} matches {len(hits)} layers")
    return hits[0]


if __name__ == "__main__":
    raise SystemExit(main())
