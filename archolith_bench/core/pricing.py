"""One model-rate table for the bench. Stdlib only; imports no benchmark.

All rates are USD per 1M tokens, static and dated in the group comments (no live
pricing lookups). Consumers derive their own table shape from ``RATES_PER_1M`` and keep
their own key set; ``cache_write`` is ``None`` where no write rate is recorded.

Where providers publish no separate cache rate, ``cached_input`` equals ``input``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRates:
    input: float
    cached_input: float
    output: float
    cache_write: float | None = None


RATES_PER_1M: dict[str, ModelRates] = {
    # OpenAI (June 2026 published rates; chat rows' cache_write from the ingest cost tool,
    # rates as of 2026-09, openai.com/api/pricing)
    "gpt-5.2-pro": ModelRates(21.00, 21.00, 168.00),
    "gpt-5.2": ModelRates(1.75, 0.175, 14.00),
    "gpt-5.1": ModelRates(1.25, 0.125, 10.00),
    "gpt-5-nano": ModelRates(0.05, 0.005, 0.40),
    "gpt-5-mini": ModelRates(0.25, 0.025, 2.00),
    "gpt-5-pro": ModelRates(15.00, 15.00, 120.00),
    "gpt-5": ModelRates(1.25, 0.125, 10.00),
    "gpt-4.1-nano": ModelRates(0.10, 0.025, 0.40, 0.10),
    "gpt-4.1-mini": ModelRates(0.40, 0.10, 1.60, 0.40),
    "gpt-4.1": ModelRates(2.00, 0.50, 8.00),
    "gpt-4o-mini": ModelRates(0.15, 0.075, 0.60, 0.15),
    "gpt-4o": ModelRates(2.50, 1.25, 10.00, 2.50),
    # DeepSeek (api-docs.deepseek.com/quick_start/pricing, 2026-06-13)
    "deepseek-v4-flash": ModelRates(0.14, 0.0028, 0.28),
    "deepseek-v4-pro": ModelRates(0.435, 0.003625, 0.87),
    # Google (cloud.google.com/.../generative-ai/pricing) -- Global standard tier
    "gemini-3.1-flash-lite": ModelRates(0.25, 0.025, 1.50),
    "gemini-3.5-flash": ModelRates(1.50, 0.15, 9.00),
    "gemini-2.5-flash-lite": ModelRates(0.10, 0.01, 0.40),
    "gemini-2.5-flash": ModelRates(0.30, 0.03, 2.50),
    # Groq (groq.com/pricing) -- cache hit is 50% off input
    "llama-3.1-8b": ModelRates(0.05, 0.025, 0.08),
    "llama-3.3-70b": ModelRates(0.59, 0.295, 0.79),
    "gpt-oss-20b": ModelRates(0.075, 0.0375, 0.30),
    "gpt-oss-120b": ModelRates(0.15, 0.075, 0.60),
    # Cerebras (cerebras.ai/pricing) -- no separate cache rate. Provider-scoped because
    # Cerebras charges more than Groq for the same open weights. Free tier = $0, but the
    # list-price estimate is what cost columns report.
    "cerebras/gpt-oss-120b": ModelRates(0.35, 0.35, 0.75),
    "cerebras/zai-glm-4.7": ModelRates(1.20, 1.20, 1.20),  # preview; provisional flat estimate
    "cerebras/qwen-3-32b": ModelRates(0.40, 0.40, 0.80),
    "cerebras/llama-3.3-70b": ModelRates(0.85, 0.85, 1.20),
    "cerebras/llama3.1-8b": ModelRates(0.10, 0.10, 0.10),
    # OpenRouter (openrouter.ai/api/v1) -- paid routes from the live model catalog; no
    # separate cache rate. Provider-scoped to avoid colliding with the Groq entries.
    "openrouter/gpt-oss-120b": ModelRates(0.03, 0.03, 0.15),
    "openrouter/gpt-oss-20b": ModelRates(0.029, 0.029, 0.14),
    "openrouter/llama-3.3-70b-instruct": ModelRates(0.10, 0.10, 0.32),
    "openrouter/llama-3.2-3b-instruct": ModelRates(0.051, 0.051, 0.335),
    "openrouter/qwen3-next-80b": ModelRates(0.09, 0.09, 1.10),
    "openrouter/gemma-4-31b": ModelRates(0.12, 0.12, 0.35),
    # OpenAI gpt-6-luna Standard, short context (<=272K input), developers.openai.com/api/docs/pricing
    # and /models/gpt-6-luna, 2026-10-15
    "gpt-6-luna": ModelRates(0.10, 0.01, 0.50, 0.125),
    # OpenRouter slugs, openrouter.ai/api/v1/models, 2026-09-08: prompt $0.20/M, completion
    # $1.20/M, cache read $0.02/M. The :batch variants are exactly 50% off (asynchronous API).
    "openai/gpt-5.6-luna": ModelRates(0.20, 0.02, 1.20, 0.20),
    "openai/gpt-5.6-luna-pro": ModelRates(0.20, 0.02, 1.20, 0.20),
    "openai/gpt-5.6-luna:batch": ModelRates(0.10, 0.01, 0.60, 0.10),
    "openai/gpt-5.6-luna-pro:batch": ModelRates(0.10, 0.01, 0.60, 0.10),
    # Embeddings -- no output tokens, so the output rate is 0.0 rather than unknown
    "text-embedding-3-small": ModelRates(0.02, 0.02, 0.0, 0.02),
    "text-embedding-3-large": ModelRates(0.13, 0.13, 0.0, 0.13),
    # Anthropic (rates as of 2026-06): opus-4-8 $5/$25, cache read $0.50, cache write 5m $6.25
    "claude-opus-4-8": ModelRates(5.00, 0.50, 25.00, 6.25),
    "claude-sonnet-4-20250514": ModelRates(3.00, 0.30, 15.00, 3.75),
}

# Flex tier (50% of Standard for gpt-6-luna), only where OpenAI publishes one; 2026-10-15.
FLEX_RATES_PER_1M: dict[str, ModelRates] = {
    "gpt-6-luna": ModelRates(0.05, 0.005, 0.25, 0.0625),
}
