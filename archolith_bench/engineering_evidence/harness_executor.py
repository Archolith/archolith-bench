"""Sprint sessions run by archolith-harness (in-process agent loop) instead of OpenCode.

Everything before the agent runs is shared with `OpenCodeSprintExecutor` (checkouts, drift
patches, enrichment waits, the Menhir readiness check, prompts, notes snapshots); only
`_run_agent` differs. Results from this agent are reported separately from OpenCode's, never
pooled, since the agents differ.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Callable

from archolith_bench.beacon_eval.isolation import load_api_keys
from archolith_bench.beacon_eval.runner import RateLimited
from archolith_bench.core.metrics import PRICING_DEFAULTS, PricingModel
from archolith_bench.engineering_evidence.live_executor import (
    OpenCodeSprintExecutor,
    _notes_violations,
    _with_notes_error,
)
from archolith_bench.engineering_evidence.sprint import (
    NOTES_FILE,
    SessionExecution,
    SprintFixture,
    SprintSession,
)

AGENT_NAME = "archolith-harness"
#: Menhir tools the memory arms count as saves (the harness logs tool names, not arguments).
_MEMORY_SAVE_TOOLS = frozenset({"add_memory"})
_TODO_SAVE_TOOLS = frozenset({"add_todo"})
#: cth_flex error kinds that halt the campaign: quota/billing and auth. Every other 429 (Flex
#: capacity, rate_limit_exceeded) is retried until the session timer runs out and at worst
#: ends that session as a timeout.
_HALTING_KINDS = frozenset({"quota", "auth"})


def _halts_campaign(exc: BaseException) -> bool:
    from cth_flex import TierMismatch

    if isinstance(exc, TierMismatch):  # billed at another tier: the Flex spend guard no longer holds
        return True
    info = getattr(exc, "info", None)
    kind = getattr(getattr(info, "kind", None), "value", None)
    return kind in _HALTING_KINDS


def _bare_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _rates(key: str) -> PricingModel:
    """Rates for a pricing key; unpriced models fail closed (the spend guard reads cost_usd)."""
    rates = PRICING_DEFAULTS.get(key)
    if rates is None:
        raise ValueError(f"no pricing for {key!r} in core.metrics.PRICING_DEFAULTS")
    return rates


def _steps_cost_usd(steps: list[Any], flex: PricingModel, standard: PricingModel | None) -> float:
    """Each step at the rates of the tier that served it (fallback steps are Standard)."""
    total = 0.0
    for step in steps:
        rates = flex if step.served_tier == "flex" else standard
        if rates is None:  # cth_flex refuses other tiers in open mode; never price one as Flex
            raise ValueError(f"step served on {step.served_tier!r} with no rates for it")
        total += ((step.input_tokens - step.cached_tokens) * rates.input_cache_miss
                  + step.cached_tokens * rates.input_cache_hit + step.output_tokens * rates.output)
    return total / 1_000_000


def _tool_counts(events_path: Path) -> tuple[int, int]:
    """(memories_saved, todos_saved) from successful tool events."""
    memories = todos = 0
    if not events_path.is_file():
        return 0, 0
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") != "tool" or not event.get("ok"):
            continue
        memories += event.get("name") in _MEMORY_SAVE_TOOLS
        todos += event.get("name") in _TODO_SAVE_TOOLS
    return memories, todos


class HarnessSprintExecutor(OpenCodeSprintExecutor):
    """`OpenCodeSprintExecutor` with the agent swapped for archolith-harness."""

    def __init__(self, config: Any, *, flex_mode: str = "fallback", send_fn: Callable[..., dict] | None = None,
                 mcp_connect_fn: Callable[..., Any] | None = None, **kwargs: Any) -> None:
        from archolith_harness.flex_send import FLEX_MODES

        if flex_mode not in FLEX_MODES:
            raise ValueError(f"flex_mode must be one of {FLEX_MODES}, got {flex_mode!r}")
        super().__init__(config, **kwargs)
        model = _bare_model(config.model)
        self.flex_mode = flex_mode
        self._flex_rates = _rates(f"{model}-flex")
        # Fallback mode may be served on Standard; open mode never is.
        self._standard_rates = _rates(model) if flex_mode == "fallback" else None
        self._send = send_fn
        self._mcp_connect = mcp_connect_fn

    def _sender(self) -> Callable[..., dict]:
        with self._lock:
            if self._send is None:
                from archolith_harness.flex_send import flex_sender

                keys = load_api_keys(self.config.env_file) if self.config.env_file else {}
                self._send = flex_sender(self.flex_mode, api_key=keys.get("OPENAI_API_KEY"))
            return self._send

    def _connect_mcp(self) -> Any:
        if self._mcp_connect is None:
            from archolith_harness.mcp_tools import McpTools

            self._mcp_connect = McpTools.connect
        key = self.config.menhir_api_key
        headers = {"Authorization": f"Bearer {key}"} if key else None
        # None = every tool the server lists, the same set OpenCode is given.
        return self._mcp_connect(self.config.menhir_mcp_url, None, headers=headers)

    def _run_agent(
        self,
        *,
        fixture: SprintFixture,
        session: SprintSession,
        arm: str,
        checkout: Path,
        mcp: dict[str, Any] | None,
        run_dir: Path,
        full_prompt: str,
        notes_before: dict[str, str] | None,
        wait_info: dict[str, Any],
    ) -> SessionExecution:
        from archolith_harness.loop import AgentConfig, run_session
        from archolith_harness.tools import ToolSet

        model = _bare_model(self.config.model)
        # Open mode has no session clock: Flex waits are part of the run, not a censoring limit.
        config = AgentConfig(model=model, max_total_tokens=self.config.reserve_tokens,
                             timeout_s=None if self.flex_mode == "open" else self.config.timeout_s)
        tools = ToolSet(checkout, writable=frozenset({NOTES_FILE}) if arm == "notes_file" else frozenset())
        events_path = run_dir / "events.jsonl"

        # run_session records send errors as text; keep the exception to classify it exactly.
        send, send_errors = self._sender(), []

        def recording_send(request: dict, **kwargs: Any) -> dict:
            try:
                return send(request, **kwargs)
            except BaseException as exc:
                send_errors.append(exc)
                raise

        recording_send.takes_timeout = getattr(send, "takes_timeout", False)  # type: ignore[attr-defined]

        t0 = time.monotonic()
        mcp_tools = self._connect_mcp() if mcp else None
        try:
            res = run_session(full_prompt, checkout, config=config, send=recording_send, tools=tools,
                              mcp=mcp_tools, events_path=events_path)
        finally:
            if mcp_tools is not None:
                mcp_tools.close()
        elapsed = time.monotonic() - t0

        if any(_halts_campaign(exc) for exc in send_errors):
            raise RateLimited(f"{res.error} during {arm}/{session.session_id}; stopping fail-closed")

        memories_saved, todos_saved = _tool_counts(events_path)
        if self.config.gzip_events and events_path.is_file():
            with gzip.open(run_dir / "events.jsonl.gz", "wt", encoding="utf-8") as gz:
                gz.write(events_path.read_text(encoding="utf-8"))
            events_path.unlink()
        (run_dir / "agent.json").write_text(json.dumps({
            "agent": AGENT_NAME, "cache_key": res.cache_key, "system_prompt_sha256": res.system_prompt_sha256,
            "cache_hit_rate": res.cache_hit_rate, "served_tiers": sorted({s.served_tier or "" for s in res.steps}),
            "flex_mode": self.flex_mode, "standard_steps": sum(s.served_tier != "flex" for s in res.steps),
        }, indent=2), encoding="utf-8")

        notes_violations = _notes_violations(notes_before, checkout)
        # OpenAI reports output_tokens including reasoning; SessionExecution keeps them apart.
        output_visible = max(res.output_tokens - res.reasoning_tokens, 0)
        return SessionExecution(
            response_text=res.final_text,
            input_tokens=res.uncached_input_tokens,
            output_tokens=output_visible,
            reasoning_tokens=res.reasoning_tokens,
            cache_read_tokens=res.cached_tokens,
            cache_write_tokens=0,  # OpenAI bills no cache writes; uncached input is input_tokens
            cost_usd=_steps_cost_usd(res.steps, self._flex_rates, self._standard_rates),
            censored=res.censored,
            turns=len(res.steps),
            tool_calls=res.tool_calls,
            files_read=list(res.files_read),
            memories_saved=memories_saved,
            todos_saved=todos_saved,
            seconds=elapsed,
            error=_with_notes_error(res.error, notes_violations),
            repo_root=checkout,
            notes_violations=notes_violations,
            refused_reads=res.refused,
            enrichment_wait_s=float(wait_info.get("waited_s", 0.0)),
            enrichment_drained=bool(wait_info.get("drained", True)),
            run_dir=run_dir,
        )
