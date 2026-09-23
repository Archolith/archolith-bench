"""Per-run OpenCode isolation: no MCP servers, or only Beacon.

Ports cth.harness ``OpencodeAdapter.createStrippedConfig``: the OpenCode config
directory is copied to a private temp directory (``node_modules`` linked, not copied)
and every MCP server in the copy is disabled, so the user's memory and other servers
never reach a trial. The run points at the copy through ``OPENCODE_CONFIG_DIR``.
Condition B adds exactly one server, Beacon, through ``OPENCODE_CONFIG_CONTENT``.

The copy holds the user's provider configuration, credentials included, exactly as
the harness's does; it lives only for one run and is removed afterwards. Nothing
here reads, prints or logs a credential.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def default_config_source() -> Path:
    return Path.home() / ".config" / "opencode"


def _link_dir(source: Path, target: Path) -> None:
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(source), str(target))
    else:
        os.symlink(source, target, target_is_directory=True)


def _unlink_dir(path: Path) -> None:
    """Remove a junction or symlink without touching what it points at."""
    if path.is_symlink():
        path.unlink()
    elif os.path.isdir(path):
        os.rmdir(path)  # removes a junction itself, never its target's contents


@contextmanager
def stripped_config(source: Path, run_dir: Path) -> Iterator[Path]:
    """Yield a private OpenCode config dir with every MCP server disabled."""
    target = run_dir / "opencode-config"
    target.mkdir(parents=True, exist_ok=True)
    try:
        for entry in source.iterdir():
            if entry.name == "node_modules":
                continue
            if entry.is_dir():
                shutil.copytree(entry, target / entry.name, symlinks=True)
            else:
                shutil.copy2(entry, target / entry.name)
        modules = source / "node_modules"
        if modules.is_dir():
            _link_dir(modules, target / "node_modules")
        config_path = target / "opencode.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for server in (config.get("mcp") or {}).values():
                if isinstance(server, dict):
                    server["enabled"] = False
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        yield target
    finally:
        modules_link = target / "node_modules"
        if modules_link.exists() or modules_link.is_symlink():
            _unlink_dir(modules_link)
        shutil.rmtree(target, ignore_errors=True)


def beacon_overlay(beacon_python: str, manifest: Path, beacon_src: str | None) -> str:
    """``OPENCODE_CONFIG_CONTENT`` adding only the Beacon MCP server (condition B)."""
    server: dict[str, object] = {
        "type": "local",
        "command": [beacon_python, "-m", "beacon", "serve", "--manifest", str(manifest)],
        "enabled": True,
    }
    if beacon_src:
        server["environment"] = {"PYTHONPATH": beacon_src}
    return json.dumps({"mcp": {"beacon": server}})
