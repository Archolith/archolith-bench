"""Serve a Menhir tree over an already-built graph for one staged run (replaces recall_ab.sh).

The tree's own source is served (``PYTHONPATH=<tree>/src``), with the tree's venv when it has one,
over the Neo4j a build YAML names. It never creates, resets or writes the graph; the server is
stopped when the run ends.
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..build.config import apply_env_overrides, resolve_yaml
from ..build.pipeline import http_ok, stop, wait_http

# config.sh's default for LME_NEO4J_PW (a local throwaway graph password, not a credential).
_DEFAULT_NEO4J_PW = "lmedata123"


class ServeError(RuntimeError):
    pass


def graph_settings(build_yaml: str | Path, environ: dict[str, str]) -> dict[str, str]:
    """The graph a build YAML (plus its LME_* env overrides) built: bolt port and embedder."""
    config, _, _ = apply_env_overrides(resolve_yaml(build_yaml), environ)
    return {
        "bolt": str(config["container"]["bolt"]),
        "neo4j_name": str(config["container"]["neo4j_name"]),
        "embed_model": str(config["extraction"]["embed_model"]),
        "service_tier": str(config["extraction"]["service_tier"]),
        "neo4j_password": environ.get("LME_NEO4J_PW") or _DEFAULT_NEO4J_PW,
    }


def serve_bin(tree: Path, fallback: str | None) -> str:
    """The tree's own venv ``menhir`` when it has one (a branch may guard interpreter identity),
    else ``fallback``."""
    for name in ("menhir.exe", "menhir"):
        for sub in ("Scripts", "bin"):
            candidate = tree / ".venv" / sub / name
            if candidate.is_file():
                return str(candidate)
    if not fallback:
        raise ServeError(f"{tree} has no .venv menhir; pass --menhir-bin (or set MENHIR_MAIN_BIN)")
    return fallback


def serve_env(
    tree: Path,
    *,
    port: int,
    graph: dict[str, str],
    openai_key: str,
    log_dir: Path,
    telemetry_db: Path,
    env_file: Path,
    base: dict[str, str],
) -> dict[str, str]:
    """The recall serve's environment: recall_ab.sh's exports, read path only."""
    env = dict(base)
    env.update({
        "PYTHONPATH": str(tree / "src"),
        "ENV_FILE": str(env_file),
        "MENHIR_LOG_DIR": str(log_dir),
        "MENHIR_BENCHMARK_MODE": "1",
        "MENHIR_API_HOST": "127.0.0.1",
        "MENHIR_API_PORT": str(port),
        "MENHIR_MCP_TELEMETRY_DB": str(telemetry_db),
        "MENHIR_OPENAI_SERVICE_TIER": graph["service_tier"],
        "OTEL_SDK_DISABLED": "true",
        "LANGFUSE_TRACING_ENABLED": "false",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SECRET_KEY": "",
        "LANGFUSE_HOST": "",
        "MENHIR_OPERATOR_KEY": "",
        "MENHIR_AGENT_KEY": "",
        "MENHIR_READONLY_KEY": "",
        "MENHIR_API_KEY": "",
        "NEO4J_URI": f"bolt://localhost:{graph['bolt']}",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": graph["neo4j_password"],
        "NEO4J_DATABASE": "neo4j",
        # Recall embeds the query, so it needs the embedder the graph was built with.
        "GRAPHITI_EMBED_PROVIDER": "openai",
        "MEMORY_GRAPHITI_EMBED_PROVIDER": "openai",
        "OPENAI_API_KEY": openai_key,
        "OPENAI_EMBED_MODEL": graph["embed_model"],
    })
    return env


def _bolt_open(port: str) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=5):
            return True
    except OSError:
        return False


@contextmanager
def served_menhir(
    tree: Path,
    *,
    port: int,
    graph: dict[str, str],
    openai_key: str,
    out_dir: Path,
    menhir_bin: str | None,
    base: dict[str, str] | None = None,
    health_tries: int = 90,
) -> Iterator[str]:
    """Serve ``tree`` on 127.0.0.1:``port`` over ``graph``; yields the URL, stops on exit."""
    tree = Path(tree)
    if not (tree / "src" / "menhir").is_dir():
        raise ServeError(f"no src/menhir under {tree}")
    if not openai_key:
        raise ServeError("serving needs OPENAI_API_KEY (recall embeds the query)")
    url = f"http://127.0.0.1:{port}"
    # Something already answering would pass our health check while our serve fails to bind.
    if http_ok(f"{url}/api/health"):
        raise ServeError(f"something already serves {url}; stop it or pick another --serve-port")
    if not _bolt_open(graph["bolt"]):
        raise ServeError(f"no Neo4j on bolt 127.0.0.1:{graph['bolt']} ({graph['neo4j_name']}); "
                         "start the built graph (and its tunnel) first")
    binary = serve_bin(tree, menhir_bin)
    log_dir = out_dir / "menhir-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="staged-empty-env-", delete=False) as handle:
        env_file = Path(handle.name)
    env = serve_env(tree, port=port, graph=graph, openai_key=openai_key, log_dir=log_dir,
                    telemetry_db=out_dir / "mcp_telemetry.db", env_file=env_file,
                    base=dict(os.environ if base is None else base))
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen([binary, "serve", "--port", str(port)], cwd=str(tree), env=env)
        if not wait_http(f"{url}/api/health", health_tries, alive=lambda: proc.poll() is None):
            raise ServeError(f"menhir from {tree} did not become healthy on {url}")
        if proc.poll() is not None:
            raise ServeError(f"our menhir exited, yet {url} answers; refusing")
        yield url
    finally:
        if proc is not None:
            stop(proc)
        env_file.unlink(missing_ok=True)
