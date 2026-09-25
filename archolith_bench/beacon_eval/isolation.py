"""Per-run OpenCode isolation: a private config home holding only the model's provider.

OpenCode merges its global config directory (``opencode.json``, ``AGENTS.md``,
plugins, agents, skills) into every run and falls back to ``~/.claude/CLAUDE.md``.
Copying that directory and disabling servers still let the global instructions and
plugins through. Each run instead gets ``XDG_CONFIG_HOME`` pointing at a fresh temp
directory whose ``opencode/opencode.json`` holds only ``$schema``, ``model`` and the
one provider the model needs, plus the Beacon MCP server in condition B. The caller's
``OPENCODE_*`` variables are dropped and the Claude Code fallbacks are disabled.
OpenCode's data and state (its sessions database) also go in that temp directory, so runs
never write into the user's own OpenCode history and parallel runs never share a database;
only the downloaded ripgrep is copied in, so a run need not fetch it.

OpenCode also searches parent directories for project config up to the git root, so
the agent's checkout is made its own git repository (see ``runner.seal_checkout``).

The provider block may hold a credential. The temp directory lives in the system temp
directory (never the results tree) for one run and is removed afterwards; nothing here
reads a credential field, prints or logs one.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class IsolationError(RuntimeError):
    """The per-run config could not be built as specified."""


def default_config_source() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def minimal_config(
    source_config: Path,
    model: str,
    mcp: dict[str, Any] | None = None,
    builtin_provider: bool = False,
    disabled_tools: tuple[str, ...] = (),
) -> dict[str, Any]:
    """``$schema``, ``model`` and the model's provider only, plus *mcp* when given.

    With *builtin_provider*, a provider missing from the user's config is left to
    OpenCode's built-in catalog, with its key supplied through the environment.
    """
    real = json.loads(source_config.read_text(encoding="utf-8"))
    provider_id = model.split("/", 1)[0]
    providers = real.get("provider") or {}
    config: dict[str, Any] = {"model": model}
    if provider_id in providers:
        config["provider"] = {provider_id: providers[provider_id]}
    elif not builtin_provider:
        raise IsolationError(f"provider {provider_id!r} is not defined in {source_config}")
    if "$schema" in real:
        config["$schema"] = real["$schema"]
    if mcp:
        config["mcp"] = mcp
    if disabled_tools:
        # Named, not "*": a wildcard would also hide the MCP server's tools.
        config["tools"] = {name: False for name in disabled_tools}
    return config


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


#: Environment variable carrying condition M's memory key to the OpenCode process only.
MEMORY_KEY_ENV = "BEACON_EVAL_MEMORY_KEY"


def memory_stdio_server(
    command: list[str], environment: dict[str, str], key_var: str
) -> dict[str, Any]:
    """The ``mcp`` block for condition M over stdio: Menhir's stdio bridge as a local server.

    *key_var* names the variable the bridge reads its backend key from; its value is an
    ``{env:...}`` reference, so the key never lands in the written config.
    """
    return {
        "menhir": {
            "type": "local",
            "command": list(command),
            "enabled": True,
            "environment": {**environment, key_var: "{env:" + MEMORY_KEY_ENV + "}"},
        }
    }


def memory_server(url: str) -> dict[str, Any]:
    """The ``mcp`` block for condition M: exactly one server, Menhir's remote MCP.

    The key is referenced as ``{env:...}`` so it never lands in the written config.
    """
    return {
        "menhir": {
            "type": "remote",
            "url": url,
            "enabled": True,
            "headers": {"Authorization": "Bearer {env:" + MEMORY_KEY_ENV + "}"},
        }
    }


#: Per-run OpenCode data and state (sessions database, logs), inside the run's temp home.
DATA_DIR = ".data"
STATE_DIR = ".state"


def isolated_env(base: Mapping[str, str], config_home: Path) -> dict[str, str]:
    """*base* without ``OPENCODE_*`` overrides, pointed at *config_home*.

    Data and state also live under *config_home*: runs never write into the user's own
    OpenCode sessions database, and concurrent runs never share one.
    """
    env = {key: value for key, value in base.items() if not key.upper().startswith("OPENCODE_")}
    env["XDG_CONFIG_HOME"] = str(config_home)
    env["XDG_DATA_HOME"] = str(config_home / DATA_DIR)
    env["XDG_STATE_HOME"] = str(config_home / STATE_DIR)
    env["OPENCODE_DISABLE_CLAUDE_CODE"] = "1"
    return env


def _seed_tools(data_home: Path, source_data_home: Path) -> None:
    """Copy OpenCode's downloaded ripgrep into a fresh data home, so a run doesn't fetch it."""
    source_bin = source_data_home / "opencode" / "bin"
    target_bin = data_home / "opencode" / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    for name in ("rg.exe", "rg"):
        found = source_bin / name
        if found.is_file():
            shutil.copy2(found, target_bin / name)


def default_data_source() -> Path:
    """The user's OpenCode data home (read only: tools are copied from it)."""
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "share"


def load_api_keys(env_file: Path) -> dict[str, str]:
    """``*_API_KEY`` entries of a ``.env`` file, for the OpenCode process only (never logged)."""
    keys: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().partition("=")
        name = name.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if sep and name.endswith("_API_KEY") and value:
            keys[name] = value
    return keys


@contextmanager
def isolated_config_home(
    source_config: Path,
    model: str,
    mcp: dict[str, Any] | None = None,
    builtin_provider: bool = False,
    disabled_tools: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield a temp ``XDG_CONFIG_HOME`` for one run; removed on exit."""
    config = minimal_config(source_config, model, mcp, builtin_provider, disabled_tools)
    home = Path(tempfile.mkdtemp(prefix="beacon-eval-oc-"))
    try:
        (home / "opencode").mkdir()
        (home / "opencode" / "opencode.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (home / STATE_DIR).mkdir()
        _seed_tools(home / DATA_DIR, default_data_source())
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)
        if home.exists():
            print(f"warning: could not remove per-run OpenCode config {home}", file=sys.stderr)
