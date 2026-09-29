"""Budget-capped reverse proxy for OpenAI-compatible APIs.

Sits between a PR's menhir subprocess and the real OpenAI API. The PR code
talks to ``http://127.0.0.1:<port>/v1/...`` and never sees the real
``OPENAI_API_KEY``. The proxy:

1. Holds the real key in its own process.
2. Forwards only ``/v1/chat/completions`` and ``/v1/embeddings`` — rejects
   everything else with 403.
3. Counts every call and every token spent.
4. Enforces three hard caps: max calls, max USD, max wall-clock seconds.
5. On any cap exceeded, returns 429 and signals the parent to kill the
   subprocess.
6. Logs every call to a trace file for audit.

Uses only stdlib + httpx (already a dependency). No FastAPI/uvicorn needed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

ALLOWED_PATHS = frozenset({"/v1/chat/completions", "/v1/embeddings"})
ALLOWED_GET_PATHS = frozenset({"/v1/models"})

# Per-1M-token pricing for cost cap. Defaults to gpt-4o-mini; override via env.
# These are conservative — actual pricing may vary by deployment.
DEFAULT_PRICE_INPUT_PER_1M = float(os.getenv("BENCH_PRICE_INPUT_PER_1M", "0.150"))
DEFAULT_PRICE_OUTPUT_PER_1M = float(os.getenv("BENCH_PRICE_OUTPUT_PER_1M", "0.600"))


class BudgetState:
    """Shared mutable state for the budget proxy. Thread-safe via a lock."""

    def __init__(
        self,
        *,
        api_key: str,
        upstream: str,
        trace_file: Path,
        budget_file: Path,
        max_calls: int,
        max_usd: float,
        max_seconds: float,
        price_input_per_1m: float = DEFAULT_PRICE_INPUT_PER_1M,
        price_output_per_1m: float = DEFAULT_PRICE_OUTPUT_PER_1M,
        on_killed: Callable[[], None] | None = None,
        body_overrides: dict | None = None,
    ) -> None:
        self.api_key = api_key
        # Merged into every forwarded chat-completion body (e.g. {"reasoning": {"effort": "low"}}).
        self.body_overrides = dict(body_overrides or {})
        self.upstream = upstream.rstrip("/")
        self.trace_file = trace_file
        self.budget_file = budget_file
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.max_seconds = max_seconds
        self.price_input_per_1m = price_input_per_1m
        self.price_output_per_1m = price_output_per_1m
        self.calls = 0
        self.usd = 0.0
        self.started_at = time.time()
        self.killed = False
        self.kill_reason: str | None = None
        self._on_killed = on_killed
        self._lock = threading.Lock()
        self._trace_fp = trace_file.open("a", encoding="utf-8")

    def check_caps(self) -> str | None:
        """Return a kill reason if any cap is exceeded, else None."""
        with self._lock:
            return self._check_caps_unlocked()

    def _check_caps_unlocked(self) -> str | None:
        if self.calls >= self.max_calls:
            return f"LLM call cap exceeded ({self.calls} / {self.max_calls})"
        return self._check_runtime_caps_unlocked()

    def check_runtime_caps(self) -> str | None:
        """Check guards that may safely be polled while a request is in flight."""
        with self._lock:
            return self._check_runtime_caps_unlocked()

    def _check_runtime_caps_unlocked(self) -> str | None:
        if self.usd >= self.max_usd:
            return f"USD cap exceeded (${self.usd:.4f} / ${self.max_usd:.2f})"
        elapsed = time.time() - self.started_at
        if elapsed >= self.max_seconds:
            return f"Wall-clock cap exceeded ({elapsed:.0f}s / {self.max_seconds:.0f}s)"
        return None

    def try_begin_call(self) -> str | None:
        """Reserve one call before forwarding it, preventing concurrent overruns."""
        with self._lock:
            reason = self._check_caps_unlocked()
            if reason:
                return reason
            self.calls += 1
            return None

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            cost = (
                input_tokens / 1_000_000 * self.price_input_per_1m
                + output_tokens / 1_000_000 * self.price_output_per_1m
            )
            self.usd += cost

    def write_trace(self, entry: dict) -> None:
        entry["ts"] = time.time()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            self._trace_fp.write(line)
            self._trace_fp.flush()

    def write_budget(self) -> None:
        with self._lock:
            payload = {
                "calls": self.calls,
                "usd": round(self.usd, 6),
                "max_calls": self.max_calls,
                "max_usd": self.max_usd,
                "max_seconds": self.max_seconds,
                "elapsed_seconds": round(time.time() - self.started_at, 1),
                "killed": self.killed,
                "kill_reason": self.kill_reason,
            }
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        self.budget_file.write_text(json.dumps(payload, indent=2))

    def mark_killed(self, reason: str) -> None:
        callback: Callable[[], None] | None = None
        with self._lock:
            if self.killed:
                return
            self.killed = True
            self.kill_reason = reason
            callback = self._on_killed
        self.write_budget()
        if callback is not None:
            callback()

    def is_killed(self) -> bool:
        with self._lock:
            return self.killed

    def close(self) -> None:
        self._trace_fp.close()


class _ProxyHandler(BaseHTTPRequestHandler):
    """Per-request handler. Reads state from server instance."""

    state: BudgetState  # set by BudgetProxy before serve_forever
    client: httpx.Client  # set by BudgetProxy

    def log_message(self, format: str, *args) -> None:  # noqa: A002 — silence default logging
        pass

    def _send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 — stdlib expects this name
        self._handle()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ALLOWED_GET_PATHS:
            # Free metadata (model listing for client startup checks): forwarded, never counted.
            self._forward("GET", b"", path=path)
            return
        self.state.write_trace({"blocked": self.path, "method": "GET", "reason": "method not allowed"})
        self._send_json(403, {"error": f"only POST to {sorted(ALLOWED_PATHS)} allowed by bench proxy"})

    def _handle(self) -> None:
        path = self.path.split("?")[0]
        if path not in ALLOWED_PATHS:
            self.state.write_trace({"blocked": path, "method": "POST", "reason": "path not allowed"})
            self._send_json(403, {"error": f"only {sorted(ALLOWED_PATHS)} allowed by bench proxy"})
            return

        if self.state.is_killed():
            self._send_json(429, {"error": f"budget cap exceeded: {self.state.kill_reason}"})
            return

        cap_reason = self.state.try_begin_call()
        if cap_reason:
            self.state.mark_killed(cap_reason)
            self._send_json(429, {"error": f"budget cap exceeded: {cap_reason}"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b""
        self._call_meta = {}
        if path == "/v1/chat/completions" and body:
            try:
                parsed = json.loads(body)
                schema = ((parsed.get("response_format") or {}).get("json_schema") or {}).get("name")
                self._call_meta = {"schema": schema or (parsed.get("response_format") or {}).get("type") or "text"}
                if self.state.body_overrides:
                    parsed.update(self.state.body_overrides)
                    body = json.dumps(parsed).encode("utf-8")
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        self._forward("POST", body, path=path)

    def _forward(self, method: str, body: bytes, *, path: str | None = None) -> None:
        upstream_path = path or self.path.split("?")[0]
        url = f"{self.state.upstream}{upstream_path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.state.api_key}",
        }
        try:
            resp = self.client.request(method, url, content=body, headers=headers, timeout=120.0)
        except httpx.HTTPError as e:
            self.state.write_trace({"error": str(e), "path": upstream_path, "method": method})
            self._send_json(502, {"error": f"upstream failed: {e}"})
            return

        # Try to extract usage for cost accounting
        in_tok = out_tok = reasoning_tok = 0
        try:
            resp_json = json.loads(resp.content)
            usage = resp_json.get("usage", {})
            in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            reasoning_tok = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            self.state.record_usage(in_tok, out_tok)
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            # The call was already reserved before forwarding it. Unknown usage
            # remains a zero-cost observation, but still consumes a call slot.
            pass

        self.state.write_trace({
            "path": upstream_path,
            "method": method,
            "status": resp.status_code,
            "calls": self.state.calls,
            "usd": round(self.state.usd, 6),
            **getattr(self, "_call_meta", {}),
            "in_tok": in_tok,
            "out_tok": out_tok,
            "reasoning_tok": reasoning_tok,
        })
        self.state.write_budget()

        # Re-check caps after the call
        cap_reason = self.state.check_caps()
        if cap_reason:
            self.state.mark_killed(cap_reason)

        # Forward the response back
        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() not in {"transfer-encoding", "content-encoding", "content-length"}:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(resp.content)))
        self.end_headers()
        self.wfile.write(resp.content)


class BudgetProxy:
    """Runs the budget-capped proxy in a background thread."""

    def __init__(
        self,
        *,
        api_key: str,
        upstream: str = "https://api.openai.com",
        port: int = 8765,
        trace_file: str | Path,
        budget_file: str | Path,
        max_calls: int = 200,
        max_usd: float = 5.0,
        max_seconds: float = 900.0,
        on_killed: Callable[[], None] | None = None,
        body_overrides: dict | None = None,
    ) -> None:
        self.state = BudgetState(
            api_key=api_key,
            upstream=upstream,
            trace_file=Path(trace_file),
            budget_file=Path(budget_file),
            max_calls=max_calls,
            max_usd=max_usd,
            max_seconds=max_seconds,
            on_killed=on_killed,
            body_overrides=body_overrides,
        )
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._client = httpx.Client(timeout=120.0)

    def start(self) -> None:
        # BaseHTTPRequestHandler serves the request inside __init__, so state must exist on the
        # class before construction; assigning it afterwards left every request without it.
        handler_cls = type(
            "_BoundProxyHandler", (_ProxyHandler,), {"state": self.state, "client": self._client}
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"budget-proxy-{self.port}",
            daemon=True,
        )
        self._thread.start()
        self.state.write_budget()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._client.close()
        self.state.close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def is_killed(self) -> bool:
        return self.state.is_killed()

    def enforce_caps(self) -> str | None:
        """Trip the proxy for elapsed-time or post-response spend exhaustion."""
        reason = self.state.check_runtime_caps()
        if reason:
            self.state.mark_killed(reason)
        return reason

    def kill_reason(self) -> str | None:
        return self.state.kill_reason
