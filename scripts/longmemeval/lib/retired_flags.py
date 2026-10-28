"""Refuse a Menhir launch whose environment sets a flag the serving tree has retired.

Menhir ignores a retired env var (its ``RETIRED`` registry is read only by tests), so an arm or
export that sets one to ``0`` silently runs the feature on and no longer means what it says.

    retired_flags.py check <menhir_root>

Reads ``RETIRED`` from ``<menhir_root>/src/menhir/config/feature_flags.py`` without importing
Menhir. Exit 0: no retired ``MENHIR_*`` name is set (or the tree predates the registry).
Exit 1: one or more are set; each is printed with its retirement note. Exit 2: the registry
exists but cannot be read, so the check fails closed.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

REGISTRY = Path("src") / "menhir" / "config" / "feature_flags.py"


def retired_env_names(menhir_root: Path) -> dict[str, str] | None:
    """``MENHIR_*`` name -> note from the tree's RETIRED dict; {} when the tree has no registry."""
    path = menhir_root / REGISTRY
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "RETIRED" for t in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                return None
            return {str(k): str(v) for k, v in value.items() if str(k).startswith("MENHIR_")}
    return {}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "check":
        print("usage: retired_flags.py check <menhir_root>", file=sys.stderr)
        return 2
    try:
        retired = retired_env_names(Path(argv[1]))
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"cannot read RETIRED from {argv[1]}: {exc}", file=sys.stderr)
        return 2
    if retired is None:
        print(f"RETIRED in {argv[1]} is not a dict literal", file=sys.stderr)
        return 2
    found = sorted(name for name in retired if name in os.environ)
    for name in found:
        print(f"{name}={os.environ[name]} is retired in {argv[1]}: {retired[name]}", file=sys.stderr)
    if found:
        print("remove these settings; Menhir ignores them and runs the shipped behavior",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
