"""Metric dataclasses for the archolith-bench proxy suite."""

from __future__ import annotations

from dataclasses import dataclass

from archolith_maintenance.token_accounting import count_text_tokens

from archolith_bench.core.pricing import FLEX_RATES_PER_1M, RATES_PER_1M, ModelRates


def estimate_tokens(text: str | None) -> int:
    """Estimate content token count from text using cl100k_base when available."""
    if not text or not text.strip():
        return 0
    return count_text_tokens(text, minimum=1)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total content token count from a list of OpenAI-style message dicts."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or (part.get("content") if isinstance(part.get("content"), str) else "")
                    total += estimate_tokens(text)
                elif isinstance(part, str):
                    total += estimate_tokens(part)
    return total


# ---------------------------------------------------------------------------
# Pricing model — cache-aware effective cost
# ---------------------------------------------------------------------------


@dataclass
class PricingModel:
    """Per-provider pricing rates per 1M tokens.

    All rates in USD per 1M tokens.  Rates are static (dated in `comment`)
    — no live pricing lookups.

    `input_cache_write` is optional because Anthropic charges MORE than the
    full input rate for cache writes (write-vs-read asymmetry).
    """

    provider: str
    input_full: float
    input_cache_hit: float
    input_cache_miss: float
    output: float
    helper_input: float = 0.0
    helper_output: float = 0.0
    helper_cache_hit: float = 0.0
    input_cache_write: float | None = None
    comment: str = ""


_HELPER = RATES_PER_1M["gpt-4.1-mini"]
_FLEX_LUNA = FLEX_RATES_PER_1M["gpt-6-luna"]


def _pricing(
    provider: str,
    rates: ModelRates,
    *,
    comment: str,
    helper: bool = False,
    cache_write: bool = False,
) -> PricingModel:
    return PricingModel(
        provider=provider,
        input_full=rates.input,
        input_cache_hit=rates.cached_input,
        input_cache_miss=rates.input,
        output=rates.output,
        helper_input=_HELPER.input if helper else 0.0,
        helper_output=_HELPER.output if helper else 0.0,
        helper_cache_hit=_HELPER.cached_input if helper else 0.0,
        input_cache_write=rates.cache_write if cache_write else None,
        comment=comment,
    )


PRICING_DEFAULTS: dict[str, PricingModel] = {
    "deepseek-v4-flash": _pricing(
        "deepseek-v4-flash",
        RATES_PER_1M["deepseek-v4-flash"],
        helper=True,
        comment="api-docs.deepseek.com/quick_start/pricing (2026-06-13). "
                "Helper = gpt-4.1-mini ($0.40/$1.60, cached $0.10).",
    ),
    "deepseek-v4-pro": _pricing(
        "deepseek-v4-pro",
        RATES_PER_1M["deepseek-v4-pro"],
        helper=True,
        comment="api-docs.deepseek.com/quick_start/pricing (2026-06-13). "
                "Helper = gpt-4.1-mini ($0.40/$1.60, cached $0.10).",
    ),
    "opus-4-8": _pricing(
        "opus-4-8",
        RATES_PER_1M["claude-opus-4-8"],
        helper=True,
        cache_write=True,
        comment="claude-opus-4-8 (Anthropic API skill, 2026-06): $5/$25, "
                "cache read $0.50, cache write 5m $6.25. Helper = gpt-4.1-mini.",
    ),
    "openai": _pricing(
        "openai",
        RATES_PER_1M["gpt-4o"],
        comment="Rates as of 2026-06: gpt-4o",
    ),
    # 2026-10-18: official gpt-6-luna rates. input_cache_write stays None on purpose:
    # compute_turn_cost would price the whole miss bucket at it (Anthropic semantics).
    "gpt-6-luna": _pricing(
        "gpt-6-luna",
        RATES_PER_1M["gpt-6-luna"],
        comment="OpenAI gpt-6-luna Standard, short context (2026-10-18).",
    ),
    "gpt-6-luna-flex": _pricing(
        "gpt-6-luna-flex",
        _FLEX_LUNA,
        comment="OpenAI gpt-6-luna Flex = 50% of Standard (2026-10-18).",
    ),
    "anthropic": _pricing(
        "anthropic",
        RATES_PER_1M["claude-sonnet-4-20250514"],
        cache_write=True,
        comment="Rates as of 2026-06: claude-sonnet-4-20250514",
    ),
}


@dataclass
class TurnCost:
    """Cache-weighted effective cost for a single turn."""

    turn: int
    effective_cost_usd: float
    cache_data_available: bool
    upstream_input_cost: float
    upstream_output_cost: float
    helper_cost: float = 0.0


@dataclass
class ArmCost:
    """Cache-weighted effective cost aggregated across turns."""

    total_effective_cost_usd: float
    cache_data_available: bool
    upstream_input_cost: float
    upstream_output_cost: float
    helper_cost: float = 0.0


_PER_M = 1_000_000.0


def compute_turn_cost(trace_turn: dict, pricing: PricingModel) -> TurnCost:
    """Compute cache-weighted effective cost for a single trace turn.

    Uses cache_hit_tokens / cache_miss_tokens when available; falls back to
    full-rate pricing on prompt_tokens_actual (or input_tokens) when the
    provider does not report cache data.
    """
    # Coerce with `or 0`: a trace key may be PRESENT but None (e.g. an early
    # cold-start proxy turn with prompt_tokens_actual=None), in which case
    # dict.get(key, default) returns None, not the default. None tokens == 0.
    cache_hit = trace_turn.get("cache_hit_tokens") or 0
    cache_miss = trace_turn.get("cache_miss_tokens") or 0
    prompt_actual = (
        trace_turn.get("prompt_tokens_actual")
        or trace_turn.get("input_tokens")
        or 0
    )
    output = trace_turn.get("output_tokens") or 0

    if cache_hit + cache_miss > 0:
        residual = max(0, prompt_actual - cache_hit - cache_miss)
        # On write-asymmetric providers (Anthropic), cache misses are
        # cache-creation events priced ABOVE the full input rate. When
        # input_cache_write is set, price the whole miss bucket at the write
        # rate. This slightly overstates cost (some misses are never cached),
        # biasing AGAINST the proxy -- the safe direction for a go/no-go verdict.
        miss_rate = (
            pricing.input_cache_write
            if pricing.input_cache_write is not None
            else pricing.input_cache_miss
        )
        upstream_input = (
            cache_hit * pricing.input_cache_hit
            + cache_miss * miss_rate
            + residual * pricing.input_full
        ) / _PER_M
        cache_available = True
    else:
        upstream_input = prompt_actual * pricing.input_full / _PER_M
        cache_available = False

    upstream_output = output * pricing.output / _PER_M

    helper = _compute_helper_spend(trace_turn, pricing)

    return TurnCost(
        turn=trace_turn.get("turn_number", trace_turn.get("turn", 0)),
        effective_cost_usd=round(upstream_input + upstream_output + helper, 6),
        cache_data_available=cache_available,
        upstream_input_cost=round(upstream_input, 6),
        upstream_output_cost=round(upstream_output, 6),
        helper_cost=round(helper, 6),
    )


def compute_arm_cost(turns: list[dict], pricing: PricingModel) -> ArmCost:
    """Aggregate effective cost across trace turns for one experiment arm."""
    total_input = 0.0
    total_output = 0.0
    total_helper = 0.0
    # An arm has cache telemetry if ANY turn reports cache activity. The first
    # turn of any session is unavoidably a cold-start cache miss (nothing is
    # cached yet), so AND-ing availability across turns would mark every proxy
    # arm "no cache data" and force an INCONCLUSIVE verdict. Per-turn pricing is
    # still correct either way (cold-start turns price at full rate).
    cache_available = False

    for t in turns:
        tc = compute_turn_cost(t, pricing)
        total_input += tc.upstream_input_cost
        total_output += tc.upstream_output_cost
        total_helper += tc.helper_cost
        if tc.cache_data_available:
            cache_available = True

    return ArmCost(
        total_effective_cost_usd=round(total_input + total_output + total_helper, 6),
        cache_data_available=cache_available,
        upstream_input_cost=round(total_input, 6),
        upstream_output_cost=round(total_output, 6),
        helper_cost=round(total_helper, 6),
    )


def _compute_helper_spend(trace_turn: dict, pricing: PricingModel) -> float:
    """Sum helper-LLM token costs (extractor / curator / embedding).

    Helper fields are expected to be present only when the archolith-context
    telemetry plan has landed.  Until then this returns 0.0.
    """
    hi = pricing.helper_input if pricing.helper_input else pricing.input_full
    ho = pricing.helper_output if pricing.helper_output else pricing.output
    # Cached helper prompt tokens (OpenAI prompt-cache hits) bill at a lower
    # rate. Falls back to the full helper-input rate when helper_cache_hit is
    # unset, so this is a no-op until archolith-context emits *_cached_tokens.
    hc = pricing.helper_cache_hit if pricing.helper_cache_hit else hi
    total = 0.0
    for prefix in ("extractor", "curator", "embedding"):
        # `or 0`: keys may be present-but-None on direct/cold-start trace turns.
        h_in = trace_turn.get(f"{prefix}_prompt_tokens") or 0
        h_out = trace_turn.get(f"{prefix}_completion_tokens") or 0
        h_cached = trace_turn.get(f"{prefix}_cached_tokens") or 0
        h_uncached = max(0, h_in - h_cached)
        total += h_uncached * hi / _PER_M
        total += h_cached * hc / _PER_M
        total += h_out * ho / _PER_M
    return total


# ---------------------------------------------------------------------------
# Continuity metrics
# ---------------------------------------------------------------------------


@dataclass
class ContinuityMetrics:
    repeat_file_reads: int = 0
    repeat_diagnostics: int = 0
    decision_retention: float = 0.0
    verification_continuity: float = 0.0
    turn_one_orientation_score: float = 0.0
    snippet_hit_rate: float = 0.0
