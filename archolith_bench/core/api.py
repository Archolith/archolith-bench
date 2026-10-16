"""HTTP API helpers for communicating with the archolith proxy and upstream."""

from __future__ import annotations

import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

_proxy_port = os.getenv("PROXY_PORT", "9800")
PROXY_URL = os.getenv("PROXY_URL", f"http://localhost:{_proxy_port}/v1")
DIRECT_URL = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("UPSTREAM_API_KEY", "")
MODEL = os.getenv("BENCHMARK_MODEL", "gpt-6-luna")

# OpenAI reasoning models reject max_tokens and any non-default temperature, and count their
# reasoning tokens inside max_completion_tokens -- so they get a far larger ceiling (billing is
# on actual tokens). Menhir measured a Luna sample truncating at 2048 with 1969 reasoning tokens.
_REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
REASONING_MAX_COMPLETION_TOKENS = 16384
# Opt-in service tier (e.g. "flex"), sent only to reasoning models on the OpenAI endpoint.
SERVICE_TIER_ENV = "BENCHMARK_SERVICE_TIER"
FLEX_TIMEOUT_S = 900
_CAPACITY_MARKERS = ("resource_unavailable", "resource unavailable")
FLEX_CAPACITY_RETRIES = 8


class ChatCallFailed(RuntimeError):
    """A chat call produced no usable completion (raised only when ``strict``)."""


def is_reasoning_model(model: str | None) -> bool:
    return (model or "").strip().lower().startswith(_REASONING_PREFIXES)


def build_chat_body(model: str, messages: list[dict], base_url: str) -> tuple[dict, float]:
    """Request body and per-request timeout for ``model`` at ``base_url``."""
    body: dict = {"model": model, "messages": messages}
    timeout = 300.0
    if is_reasoning_model(model):
        body["max_completion_tokens"] = REASONING_MAX_COMPLETION_TOKENS
        tier = os.getenv(SERVICE_TIER_ENV, "").strip().lower()
        if tier and "api.openai.com" in (base_url or "").lower():
            body["service_tier"] = tier
            if tier == "flex":
                timeout = float(FLEX_TIMEOUT_S)
    else:
        body["max_tokens"] = 2048
        body["temperature"] = 0.3
    return body, timeout


def openai_chat_kwargs(
    model: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    default_tier: str = "flex",
) -> dict:
    """Model-dependent kwargs for an OpenAI SDK ``chat.completions.create`` call.

    For analysis tools (not latency measurements), so the tier defaults to ``flex``;
    ``BENCHMARK_SERVICE_TIER`` overrides it, and an empty value sends none. A reasoning model
    gets no temperature (it rejects one) and ``max_completion_tokens`` of at least
    ``REASONING_MAX_COMPLETION_TOKENS``, since reasoning is counted inside that ceiling.
    ``base_url=None`` means the SDK default, api.openai.com.
    """
    kwargs: dict = {}
    if is_reasoning_model(model):
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max(int(max_tokens), REASONING_MAX_COMPLETION_TOKENS)
        tier = os.getenv(SERVICE_TIER_ENV, default_tier).strip().lower()
        if tier and (base_url is None or "api.openai.com" in str(base_url).lower()):
            kwargs["service_tier"] = tier
            if tier == "flex":
                kwargs["timeout"] = float(FLEX_TIMEOUT_S)
    else:
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
    return kwargs


def is_flex_capacity_error(exc: BaseException) -> bool:
    """True for a 429 meaning Flex has no capacity right now (retry later), not a rate limit.

    Accepts an OpenAI SDK error (``status_code``/``body``) or an ``httpx.HTTPStatusError``
    (``response``), whose message omits the body.
    """
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status != 429:
        return False
    try:
        response_text = getattr(response, "text", "") or ""
    except Exception:  # noqa: BLE001 - an unread streamed body has no text
        response_text = ""
    text = f"{exc} {getattr(exc, 'body', '')} {getattr(exc, 'code', '')} {response_text}".lower()
    return any(marker in text for marker in _CAPACITY_MARKERS)


def call_with_flex_retry(fn, *, retries: int = FLEX_CAPACITY_RETRIES, sleep=None):  # noqa: ANN001, ANN201
    """Call ``fn()``, waiting out Flex capacity 429s; every other error is raised unchanged."""
    sleep = sleep or time.sleep
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_flex_capacity_error(exc) or attempt >= retries:
                raise
            wait = min(300, 15 * 2 ** attempt)
            attempt += 1
            print(f"  [429] Flex capacity unavailable, waiting {wait}s ({attempt}/{retries})...")
            sleep(wait)


async def acall_with_flex_retry(fn, *, retries: int = FLEX_CAPACITY_RETRIES, sleep=None):  # noqa: ANN001, ANN201
    """Async ``call_with_flex_retry``: ``fn()`` returns an awaitable."""
    import asyncio

    sleep = sleep or asyncio.sleep
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if not is_flex_capacity_error(exc) or attempt >= retries:
                raise
            wait = min(300, 15 * 2 ** attempt)
            attempt += 1
            print(f"  [429] Flex capacity unavailable, waiting {wait}s ({attempt}/{retries})...")
            await sleep(wait)


def send_chat(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    messages: list[dict],
    model: str,
    max_retries: int = 5,
    session_id: str | None = None,
    session_config: dict | None = None,
    strict: bool = False,
) -> tuple[str, float, dict]:
    """Send a chat completion request. Returns (response_text, latency_ms, usage_dict).

    Retries on 429 (rate limit) with exponential backoff. A Flex capacity 429
    (``resource_unavailable``) gets its own longer budget; ``insufficient_quota`` is never
    retried.

    By default a failure comes back as a bracketed error string in place of the text. With
    ``strict=True`` it raises ``ChatCallFailed`` instead, so a caller that scores the text
    cannot grade an error message (or an empty, length-truncated reasoning reply) as an answer.

    When session_id is provided it is sent as the X-Session-ID header so the proxy
    pins this conversation to a known session id (which the trace query then reads
    directly, instead of guessing sessions[0]).

    When session_config is provided it is sent as the X-Session-Config header (JSON
    object) so the proxy scopes those config overrides to this session only, without
    mutating global config. The proxy merges + persists it on the session, so it is
    safe (idempotent) to send on every turn.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if session_id:
        headers["X-Session-ID"] = session_id
    if session_config:
        headers["X-Session-Config"] = json.dumps(session_config)
    body, timeout = build_chat_body(model, messages, base_url)
    # deepseek-v4-flash "thinks" by default (~2.3x latency, ~3x output tokens). The
    # benchmark answers do not need chain-of-thought, so disable it for deepseek
    # targets. This is the ONLY accepted disable form for the DeepSeek API
    # (boolean / enable_thinking / reasoning_effort are rejected or ignored).
    if "deepseek" in (base_url or "").lower() or "deepseek" in (model or "").lower():
        body["thinking"] = {"type": "disabled"}

    def _fail(message: str) -> tuple[str, float, dict]:
        latency_ms = (time.monotonic() - total_start) * 1000
        if strict:
            raise ChatCallFailed(f"{model}: {message}")
        return message, latency_ms, {}

    total_start = time.monotonic()
    attempt = 0
    capacity_attempt = 0
    while True:
        try:
            resp = client.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            return _fail(f"[TIMEOUT after {(time.monotonic() - total_start):.0f}s]")
        except httpx.HTTPError as e:
            return _fail(f"[HTTP ERROR]: {e}")

        if resp.status_code == 429:
            body_text = (resp.text or "").lower()
            if "insufficient_quota" in body_text:
                break
            if any(marker in body_text for marker in _CAPACITY_MARKERS):
                if capacity_attempt >= FLEX_CAPACITY_RETRIES:
                    break
                wait = min(300, 15 * 2 ** capacity_attempt)
                capacity_attempt += 1
                print(f"  [429] Flex capacity unavailable, waiting {wait}s "
                      f"({capacity_attempt}/{FLEX_CAPACITY_RETRIES})...")
                time.sleep(wait)
                continue
            if attempt < max_retries:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = int(retry_after)
                    except ValueError:
                        wait = 2 ** attempt * 10
                else:
                    wait = 2 ** attempt * 10
                wait = min(wait, 300)
                print(f"  [429] Rate limited, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                attempt += 1
                continue

        break

    latency_ms = (time.monotonic() - total_start) * 1000

    if resp.status_code != 200:
        return _fail(f"[ERROR {resp.status_code}]: {resp.text[:300]}")

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "") or ""
    usage = data.get("usage", {})
    if strict and not text.strip():
        # A reasoning model that spends its whole budget thinking returns empty content with
        # finish_reason "length"; scoring that as an answer is a silent wrong grade.
        raise ChatCallFailed(f"{model}: empty completion (finish_reason={choice.get('finish_reason')!r})")
    return text, latency_ms, usage


def _proxy_base(proxy_url: str) -> str:
    return proxy_url.rstrip("/").removesuffix("/v1")


def get_proxy_trace(client: httpx.Client, proxy_url: str, session_id: str | None = None) -> dict:
    base = _proxy_base(proxy_url)
    # A session_id is REQUIRED. The old sessions[0] fallback read whatever session
    # happened to be first in the proxy's store (often a stale/unrelated run), which
    # silently reported the wrong session's assembly_mode/savings. Fail loudly instead.
    if not session_id:
        return {"error": "get_proxy_trace requires an explicit session_id (no sessions[0] fallback)"}
    try:
        resp2 = client.get(f"{base}/trace/sessions/{session_id}", timeout=10)
        if resp2.status_code != 200:
            return {"error": f"trace session detail {resp2.status_code}"}
        return resp2.json()
    except Exception as e:
        return {"error": str(e)}


def set_proxy_budget(
    client: httpx.Client, proxy_url: str, budget: int, persist: bool = False
) -> bool:
    """Set the proxy context token budget. Returns True if successful.

    Applies ephemerally by default so a benchmark run does not mutate the proxy's
    persisted config_overrides.json.
    """
    base = _proxy_base(proxy_url)
    try:
        resp = client.post(
            f"{base}/admin/config",
            params={"persist": str(persist).lower()},
            json={"context_token_budget": budget},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def apply_arm_config(
    client: httpx.Client, proxy_url: str, config_overrides: dict, persist: bool = False
) -> bool:
    """Apply experiment-arm config overrides to the proxy. Returns True if successful.

    Applies ephemerally by default (persist=False): the proxy holds the override
    in memory for its current process but does NOT write config_overrides.json, so a
    benchmark run never mutates the proxy's persisted config. (Older proxies that do
    not support ?persist ignore the query param and persist as before.)
    """
    base = _proxy_base(proxy_url)
    try:
        resp = client.post(
            f"{base}/admin/config",
            params={"persist": str(persist).lower()},
            json=config_overrides,
            timeout=5,
        )
        if resp.status_code == 200:
            result = resp.json()
            if result.get("rejected"):
                print(f"  WARNING: rejected overrides: {result['rejected']}")
            else:
                _mode = "persisted" if result.get("persisted", True) else "ephemeral"
                print(f"  Config overrides applied ({_mode}): {result.get('updated', {})}")
            return True
        print(f"  WARNING: Failed to apply config overrides: {resp.status_code}")
        return False
    except Exception as e:
        print(f"  WARNING: Failed to apply config overrides: {e}")
        return False


def snapshot_proxy_config(client: httpx.Client, proxy_url: str) -> dict:
    """Capture the proxy runtime config for experiment recording."""
    base = _proxy_base(proxy_url)
    try:
        resp = client.get(f"{base}/admin/config", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def check_proxy_health(proxy_url: str) -> dict | None:
    """Check proxy health. Returns health dict on success, None on failure."""
    base = _proxy_base(proxy_url)
    try:
        with httpx.Client() as client:
            resp = client.get(f"{base}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None
