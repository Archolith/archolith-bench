from __future__ import annotations

import pytest

from archolith_bench.core.api import SERVICE_TIER_ENV
from archolith_bench.core.metrics import PRICING_DEFAULTS
from archolith_bench.harness.base import _pick_pricing


@pytest.fixture(autouse=True)
def _no_tier(monkeypatch):
    monkeypatch.delenv(SERVICE_TIER_ENV, raising=False)


@pytest.mark.parametrize("model", ["gpt-6-luna", "openai/gpt-6-luna", "GPT-6-Luna"])
def test_luna_uses_luna_standard_rates(model) -> None:
    pricing = _pick_pricing(model, None)
    assert pricing is PRICING_DEFAULTS["gpt-6-luna"]
    assert (pricing.input_full, pricing.input_cache_hit, pricing.output) == (0.10, 0.01, 0.50)
    assert pricing.input_cache_write is None


@pytest.mark.parametrize("model", ["gpt-6-luna", "openai/gpt-6-luna"])
def test_luna_uses_flex_rates_when_tier_is_flex(monkeypatch, model) -> None:
    monkeypatch.setenv(SERVICE_TIER_ENV, "flex")
    pricing = _pick_pricing(model, None)
    assert pricing is PRICING_DEFAULTS["gpt-6-luna-flex"]
    assert (pricing.input_full, pricing.input_cache_hit, pricing.output) == (0.05, 0.005, 0.25)


def test_non_flex_tier_prices_luna_at_standard(monkeypatch) -> None:
    monkeypatch.setenv(SERVICE_TIER_ENV, "default")
    assert _pick_pricing("gpt-6-luna", None) is PRICING_DEFAULTS["gpt-6-luna"]


def test_explicit_pricing_wins(monkeypatch) -> None:
    explicit = PRICING_DEFAULTS["opus-4-8"]
    assert _pick_pricing("gpt-6-luna", explicit) is explicit


def test_other_models_are_unchanged(monkeypatch) -> None:
    monkeypatch.setenv(SERVICE_TIER_ENV, "flex")
    assert _pick_pricing("gpt-4o", None) is PRICING_DEFAULTS["openai"]
    assert _pick_pricing("claude-sonnet-4", None) is PRICING_DEFAULTS["anthropic"]
