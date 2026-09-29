"""End-to-end budget proxy: real HTTP through the proxy to a fake local upstream."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from archolith_bench.ci.budget_proxy import BudgetProxy


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # noqa: ANN002
        pass

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._reply({"data": [{"id": "openai/gpt-6-luna"}]})

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._reply({"choices": [{"message": {"content": "ok"}}],
                     "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}})


@pytest.fixture()
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", _free_port()), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_proxy_forwards_counts_lists_models_and_stops_at_the_cap(upstream, tmp_path):
    proxy = BudgetProxy(
        api_key="k", upstream=upstream, port=_free_port(),
        trace_file=tmp_path / "trace.jsonl", budget_file=tmp_path / "budget.json",
        max_usd=0.2,  # default input price 0.15/M: one 1M-token call costs $0.15
    )
    proxy.start()
    try:
        base = proxy.base_url
        models = httpx.get(f"{base}/v1/models", timeout=10)
        assert models.status_code == 200 and models.json()["data"][0]["id"] == "openai/gpt-6-luna"
        assert proxy.state.calls == 0  # model listing is free and uncounted

        assert httpx.get(f"{base}/v1/files", timeout=10).status_code == 403
        assert httpx.post(f"{base}/v1/files", json={}, timeout=10).status_code == 403

        first = httpx.post(f"{base}/v1/chat/completions", json={"model": "m"}, timeout=10)
        assert first.status_code == 200 and proxy.state.calls == 1
        second = httpx.post(f"{base}/v1/chat/completions", json={"model": "m"}, timeout=10)
        assert second.status_code == 200  # $0.15 < $0.20 when it started
        assert proxy.is_killed()          # $0.30 after it: cap reached
        third = httpx.post(f"{base}/v1/chat/completions", json={"model": "m"}, timeout=10)
        assert third.status_code == 429 and proxy.state.calls == 2
    finally:
        proxy.stop()
