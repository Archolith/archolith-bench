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

import hashlib
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


def beacon_remote_server(url: str) -> dict[str, Any]:
    """The ``mcp`` block for condition R: Beacon MCP over Streamable HTTP.

    ``memory_server``'s shape without headers: the loopback Beacon server needs no key.
    """
    return {"beacon": {"type": "remote", "url": url, "enabled": True}}


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


def remove_tree(path: Path) -> None:
    """Remove *path*, including read-only files (git objects), which Windows refuses to delete."""
    def clear_readonly(func: Any, target: str, _exc: Any) -> None:
        try:
            os.chmod(target, 0o700)
            func(target)
        except FileNotFoundError:
            pass  # already gone

    if not path.exists():
        return
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=clear_readonly)
    else:
        shutil.rmtree(path, onerror=clear_readonly)


def _seed_tools(data_home: Path, source_data_home: Path) -> None:
    """Link OpenCode's downloaded ripgrep into a fresh data home, so a run doesn't fetch it.

    A hard link writes no file data; it falls back to a copy across volumes.
    """
    source_bin = source_data_home / "opencode" / "bin"
    target_bin = data_home / "opencode" / "bin"
    target_bin.mkdir(parents=True, exist_ok=True)
    for name in ("rg.exe", "rg"):
        found = source_bin / name
        if found.is_file():
            try:
                os.link(found, target_bin / name)
            except OSError:
                shutil.copy2(found, target_bin / name)


#: The files OpenCode writes next to the ``node_modules`` it installs in its config dir.
DEP_MANIFESTS = ("package.json", "package-lock.json")


def deps_template_dir(opencode_cmd: list[str]) -> Path:
    """Where the shared OpenCode config-dir dependencies live for this OpenCode build.

    Keyed by the executable's path, size and mtime, so an upgraded OpenCode (which pins a
    new plugin version) gets its own template. The ``beacon-eval-oc-`` prefix matches
    the per-run config homes, so the same Defender exclusion covers it.
    """
    exe = Path(opencode_cmd[-1] if len(opencode_cmd) > 1 else opencode_cmd[0])
    try:
        stat = exe.stat()
        ident = f"{exe.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        ident = " ".join(opencode_cmd)
    key = hashlib.sha256(ident.encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"beacon-eval-oc-deps-{key}"


def _link_dir(target: Path, link: Path) -> None:
    """A directory junction on Windows (no admin needed), a symlink elsewhere."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def _complete_install(root: Path) -> bool:
    """npm writes the lock file last; the plugin's manifest shows the package landed."""
    return (root / "package-lock.json").is_file() and (
        root / "node_modules" / "@opencode-ai" / "plugin" / "package.json"
    ).is_file()


def _use_deps(opencode_dir: Path, template: Path) -> bool:
    """Point *opencode_dir* at the shared install; False when there is no whole one."""
    if not _complete_install(template):
        return False
    for name in DEP_MANIFESTS:
        if (template / name).is_file():
            shutil.copy2(template / name, opencode_dir / name)
    try:
        _link_dir(template / "node_modules", opencode_dir / "node_modules")
    except OSError:
        return False
    return True


def _keep_deps(opencode_dir: Path, template: Path) -> None:
    """Make this run's finished install the shared template, if there is none yet.

    Only a complete install is kept (npm writes the lock file last). The template appears
    in one atomic rename, so a concurrent run sees either no template or a whole one.
    """
    installed = opencode_dir / "node_modules"
    if template.exists() or not _complete_install(opencode_dir):
        return
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=template.name + ".staging-", dir=template.parent))
        os.rename(installed, staging / "node_modules")
        for name in DEP_MANIFESTS:
            if (opencode_dir / name).is_file():
                shutil.copy2(opencode_dir / name, staging / name)
        os.rename(staging, template)
    except OSError:
        pass  # another run kept its install first (or temp is full); removed with the home
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


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
    deps_template: Path | None = None,
) -> Iterator[Path]:
    """Yield a temp ``XDG_CONFIG_HOME`` for one run; removed on exit.

    With *deps_template*, the plugin dependency OpenCode installs into its config dir
    (~3,700 files) is shared: the run links the template's ``node_modules`` (OpenCode
    then installs nothing), and the first run to finish an install becomes the template.
    """
    config = minimal_config(source_config, model, mcp, builtin_provider, disabled_tools)
    home = Path(tempfile.mkdtemp(prefix="beacon-eval-oc-"))
    opencode_dir = home / "opencode"
    linked = False
    try:
        opencode_dir.mkdir()
        (opencode_dir / "opencode.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if deps_template is not None:
            linked = _use_deps(opencode_dir, deps_template)
        (home / STATE_DIR).mkdir()
        _seed_tools(home / DATA_DIR, default_data_source())
        yield home
    finally:
        safe = True
        if linked:
            # Remove the link itself first: the tree walk must never reach the shared install.
            try:
                if sys.platform == "win32":
                    os.rmdir(opencode_dir / "node_modules")
                else:
                    os.unlink(opencode_dir / "node_modules")
            except OSError:
                safe = False
        elif deps_template is not None:
            _keep_deps(opencode_dir, deps_template)
        if safe:
            # OpenCode's snapshot store is read-only git objects; a plain rmtree leaks them.
            try:
                remove_tree(home)
            except OSError:
                pass  # e.g. a file still held by an exiting process; warned below
        if home.exists():
            print(f"warning: could not remove per-run OpenCode config {home}", file=sys.stderr)
