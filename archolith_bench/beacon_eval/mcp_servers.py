"""``mcp`` config blocks for the Beacon server in conditions B and R."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def beacon_server(beacon_python: str, manifest: Path, beacon_src: str | None) -> dict[str, Any]:
    """The ``mcp`` block for condition B: exactly one server, Beacon."""
    server: dict[str, Any] = {
        "type": "local",
        "command": [beacon_python, "-m", "beacon", "serve", "--manifest", str(manifest)],
        "enabled": True,
    }
    if beacon_src:
        server["environment"] = {"PYTHONPATH": beacon_src}
    return {"beacon": server}


def beacon_remote_server(url: str) -> dict[str, Any]:
    """The ``mcp`` block for condition R: Beacon MCP over Streamable HTTP.

    ``memory_server``'s shape without headers: the loopback Beacon server needs no key.
    """
    return {"beacon": {"type": "remote", "url": url, "enabled": True}}
