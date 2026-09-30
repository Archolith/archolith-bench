---
artifact_schema: 1
artifact_type: plan
artifact_status: ACTIVE
---

# archolith-bench -- OpenAI flex client for non-live LLM calls

Project Scope: `Archolith/archolith-bench`, worktree `C:/Users/thron/IdeaProjects/.agent/worktrees/bench-openai-flex`,
branch `feat/openai-flex-client`, base `origin/master` `32ec8ea`. Nothing outside this worktree.

## Why
Owner decision 2026-09-30: batch-style benchmark calls (answering, judging) use OpenAI **flex
processing** (`service_tier: "flex"`, synchronous, ~50% of standard price) instead of the OpenRouter
Batch API. A 3,030-request OpenRouter judge batch sat queued 2+ hours at 0 complete and cannot be
cancelled; the same work via flex finished in ~4 minutes for $0.09. PR #12's batch client also died on
one poll ConnectTimeout. This module supersedes that client for Bench's use.

## Behavior (contract)
New module `archolith_bench/core/openai_flex.py`:

```python
@dataclass(frozen=True)
class FlexResult:
    custom_id: str
    content: str | None      # message content, None if the request ultimately failed
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    error: str | None = None

class FlexError(RuntimeError): ...        # stop conditions (cap, non-capacity 429, 402, auth)

def run_flex(
    requests: list[dict],                 # [{"custom_id": str, "body": {chat.completions body WITHOUT "model"}}]
    *,
    model: str,                           # e.g. "gpt-6-luna" (OpenAI model id, no provider prefix)
    checkpoint: Path,                     # JSONL, one line per finished request
    api_key: str,
    max_usd: float,
    rates: tuple[float, float],           # (input $/1M, output $/1M) at flex prices; required, no catalog lookup
    workers: int = 16,
    base_url: str = "https://api.openai.com/v1",
    timeout_s: float = 900.0,             # OpenAI recommends 15 min for flex
    max_capacity_retries: int = 5,
    client: httpx.Client | None = None,   # injectable for tests
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, FlexResult], dict]:
```

1. **Request**: POST `{base_url}/chat/completions` with `{"model": model, "service_tier": "flex", **body}`.
   Do not override fields the caller put in `body` (e.g. `reasoning_effort`, `max_completion_tokens`,
   `response_format`), except that `model` and `service_tier` are always set by the client.
2. **Unique ids**: duplicate `custom_id` -> `ValueError` before any call.
3. **Checkpoint/resume**: on start, read the JSONL; requests already present are not re-sent. Each
   finished request (success or terminal per-request failure) is appended as one JSON line
   immediately, under a lock. Resume must never double-send a checkpointed id.
4. **Cost**: cost per request = `prompt_tokens*rates[0]/1e6 + completion_tokens*rates[1]/1e6`
   (from the response `usage`). Running total includes checkpointed rows. Before sending each
   request, if the running total >= `max_usd`, stop scheduling new requests and raise `FlexError`
   after in-flight requests finish (their results are still checkpointed).
5. **Retry policy** (per request):
   - HTTP 429 whose body indicates capacity (`resource_unavailable` / "resource unavailable",
     case-insensitive): exponential backoff `min(120, 5 * 2**attempt)` seconds, up to
     `max_capacity_retries`; then `FlexError` (stop the run).
   - Any other 429, any 402, 401, 403: `FlexError` immediately (stop the run; no retry).
   - 408, 500, 502, 503, 504, and `httpx.TimeoutException` / `httpx.TransportError` (connect errors):
     backoff `min(60, 5 * 2**attempt)`, up to 5 retries; then record a per-request failure
     (`content=None`, `error=...`) and continue with other requests (does not stop the run).
   - Other 4xx (e.g. 400 bad request): per-request failure, no retry, continue.
   - A `FlexError` in one worker stops scheduling new requests; the function raises it after
     in-flight requests finish.
6. **Summary dict**: `{"requests": total, "completed": n_ok, "failed": n_failed, "resumed": n_from_checkpoint,
   "cost_usd": total, "seconds": elapsed, "model": model}`.
7. **Logging**: progress line every 250 finished requests (`done N/total $cost`), via `log`.
8. Never print or log the API key or request bodies.

## Tests (new `tests/test_openai_flex.py`, all offline with `httpx.MockTransport`, `sleep` stubbed)
- success path: model + service_tier injected, caller body fields preserved, cost computed, summary.
- duplicate custom_id -> ValueError, no HTTP calls.
- checkpoint resume: pre-written lines are skipped (no HTTP for them) and counted as resumed.
- capacity 429 retried then succeeds; capacity 429 beyond max -> FlexError.
- rate-limit 429 (non-capacity body) -> FlexError on first occurrence, no retry.
- 402 -> FlexError.
- connect error / 503 retried then succeeds; persistent 503 -> per-request failure, run continues.
- 400 -> per-request failure, no retry.
- cost cap: stops scheduling and raises FlexError once the cap is reached; checkpoint has the finished rows.
- key never appears in `log` output.

## Docs
`.agent/CHANGELOG.md` dated entry; a short section in `.agent/README.md` or the most relevant existing
doc that lists Bench's LLM-call helpers (find it; if none, add a subsection to `.agent/README.md`):
flex is the default for non-live answering/judging; the 429 policy above; rates must be passed in.

## Test policy
Executor: `uv run pytest -q tests/test_openai_flex.py` and `uv run ruff check archolith_bench/core/openai_flex.py tests/test_openai_flex.py`,
at most once more after a fix. No full suite (reviewer runs it once). No network calls in tests.

## Non-goals
No changes to other modules, no migration of existing scripts in this PR, no dependency changes.
