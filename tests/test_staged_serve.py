"""staged run --serve: serve a Menhir tree over a built graph for the search stage."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from archolith_bench.staged import __main__ as cli
from archolith_bench.staged import serve

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = REPO / "builds" / "default.yaml"

# Stands in for `menhir serve --port N`: run as `python serve --port N` from the tree.
FAKE_SERVE = """\
import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/api/health" else 404)
        self.end_headers()
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tree(tmp_path: Path) -> Path:
    tree = tmp_path / "menhir"
    (tree / "src" / "menhir").mkdir(parents=True)
    (tree / "serve").write_text(FAKE_SERVE, encoding="utf-8")
    return tree


def _graph(bolt: int) -> dict[str, str]:
    return {"bolt": str(bolt), "neo4j_name": "g", "embed_model": "text-embedding-3-small",
            "service_tier": "flex", "neo4j_password": "pw"}


def test_graph_settings_from_build_yaml_and_overrides():
    g = serve.graph_settings(DEFAULT_BUILD, {})
    assert g["bolt"] == "7689"
    assert g["embed_model"] == "text-embedding-3-small"
    assert g["service_tier"] == "flex"
    assert g["neo4j_password"] == "lmedata123"
    g = serve.graph_settings(DEFAULT_BUILD, {"LME_BOLT": "17971", "LME_NEO4J_PW": "x"})
    assert g["bolt"] == "17971"
    assert g["neo4j_password"] == "x"


def test_serve_bin_prefers_tree_venv(tmp_path: Path):
    tree = _tree(tmp_path)
    with pytest.raises(serve.ServeError):
        serve.serve_bin(tree, None)
    assert serve.serve_bin(tree, "fallback") == "fallback"
    venv = tree / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "menhir.exe").write_text("", encoding="utf-8")
    assert serve.serve_bin(tree, "fallback") == str(venv / "menhir.exe")


def test_serve_env_points_at_tree_and_graph(tmp_path: Path):
    tree = _tree(tmp_path)
    env = serve.serve_env(tree, port=8199, graph=_graph(17971), openai_key="k", log_dir=tmp_path / "l",
                          telemetry_db=tmp_path / "t.db", env_file=tmp_path / "e",
                          base={"MENHIR_OPERATOR_KEY": "inherited", "KEEP": "1"})
    assert env["PYTHONPATH"] == str(tree / "src")
    assert env["NEO4J_URI"] == "bolt://localhost:17971"
    assert env["NEO4J_PASSWORD"] == "pw"
    assert env["MENHIR_API_HOST"] == "127.0.0.1"
    assert env["MENHIR_API_PORT"] == "8199"
    assert env["OPENAI_EMBED_MODEL"] == "text-embedding-3-small"
    assert env["MENHIR_OPENAI_SERVICE_TIER"] == "flex"
    assert env["MENHIR_OPERATOR_KEY"] == ""
    assert env["KEEP"] == "1"


def test_served_menhir_serves_then_stops(tmp_path: Path):
    tree = _tree(tmp_path)
    port = _free_port()
    with socket.socket() as bolt:
        bolt.bind(("127.0.0.1", 0))
        bolt.listen()
        with serve.served_menhir(tree, port=port, graph=_graph(bolt.getsockname()[1]), openai_key="k",
                                 out_dir=tmp_path / "run", menhir_bin=sys.executable) as url:
            assert url == f"http://127.0.0.1:{port}"
            assert serve.http_ok(f"{url}/api/health")
    assert not serve.http_ok(f"http://127.0.0.1:{port}/api/health")
    assert (tmp_path / "run" / "menhir-logs").is_dir()


def test_served_menhir_refuses_without_graph(tmp_path: Path):
    tree = _tree(tmp_path)
    with pytest.raises(serve.ServeError, match="no Neo4j on bolt"):
        with serve.served_menhir(tree, port=_free_port(), graph=_graph(_free_port()), openai_key="k",
                                 out_dir=tmp_path / "run", menhir_bin=sys.executable):
            pass


def test_served_menhir_refuses_busy_port(tmp_path: Path, monkeypatch):
    tree = _tree(tmp_path)
    monkeypatch.setattr(serve, "http_ok", lambda url: True)
    with pytest.raises(serve.ServeError, match="already serves"):
        with serve.served_menhir(tree, port=_free_port(), graph=_graph(1), openai_key="k",
                                 out_dir=tmp_path / "run", menhir_bin=sys.executable):
            pass


def test_served_menhir_refuses_non_menhir_tree(tmp_path: Path):
    with pytest.raises(serve.ServeError, match="no src/menhir"):
        with serve.served_menhir(tmp_path, port=_free_port(), graph=_graph(1), openai_key="k",
                                 out_dir=tmp_path / "run", menhir_bin=sys.executable):
            pass


def test_cli_serve_refuses_unlisted_port_before_serving(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ARCHOLITH_BENCH_ALLOW_PORTS", raising=False)
    monkeypatch.setattr(cli, "served_menhir", lambda *a, **k: pytest.fail("served before the port check"))
    argv = ["run", "--arm", "a.yaml", "--fixture", "f.json", "--serve", str(tmp_path),
            "--graph", str(DEFAULT_BUILD), "--serve-port", "8193"]
    with pytest.raises(SystemExit):
        cli.main(argv)


@pytest.mark.parametrize("extra",[["--menhir-url", "http://127.0.0.1:1"], ["--menhir-root", "x"], []])
def test_cli_serve_argument_conflicts(tmp_path: Path, extra: list[str], capsys):
    argv = ["run", "--arm", "a.yaml", "--fixture", "f.json", "--serve", str(tmp_path), *extra]
    if extra:
        argv += ["--graph", str(DEFAULT_BUILD)]
    assert cli.main(argv) == 2
    assert "--serve" in capsys.readouterr().err
