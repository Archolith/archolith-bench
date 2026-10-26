"""Shared A/B harness foundation for real external benchmarks.

`run_ab` runs an official benchmark adapter across experiment arms (direct vs
proxy family), reusing the existing chat primitive (`core.api.send_chat`), proxy
config control (`core.api.apply_arm_config`), and the cache-aware cost model
(`core.metrics`). The result is a per-arm official score plus token/cost deltas
versus the direct baseline — the only honest, advertisable Archolith claim.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol, runtime_checkable

from ..arms import ARMS
from ..core.api import (
    API_KEY,
    DIRECT_URL,
    MODEL,
    PROXY_URL,
    SERVICE_TIER_ENV,
    apply_arm_config,
    send_chat,
)
from ..core.metrics import PRICING_DEFAULTS, PricingModel, compute_arm_cost

# send_fn signature mirrors core.api.send_chat:
#   (client, base_url, api_key, messages, model) -> (text, latency_ms, usage)
SendFn = Callable[..., tuple[str, float, dict]]


@dataclass
class Task:
    """One official benchmark item, normalized to a chat request + gold answer."""

    task_id: str
    prompt_messages: list[dict]
    answer: str
    meta: dict = field(default_factory=dict)


@dataclass
class TaskResult:
    """Outcome of running one Task through one arm."""

    task_id: str
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    correct: bool
    raw_usage: dict = field(default_factory=dict)
    # Memory-benchmark transparency: the question asked, the memory recalled for it,
    # and (for scoring/debug) the gold answer. Lets the dashboard show question ->
    # retrieval -> response and makes recall failures inspectable (e.g. a temporal
    # question where one of the two compared events was never recalled).
    question: str = ""
    recalled: str = ""
    gold: str = ""
    scorer_input_tokens: int = 0
    scorer_output_tokens: int = 0
    scorer_raw_usage: dict = field(default_factory=dict)
    unofficial_correct: bool | None = None
    unofficial_scorer_raw_usage: dict = field(default_factory=dict)
    # Opt-in (--recall-trace) compact per-recall candidate/rank/lane rows; empty when off.
    recall_trace: dict = field(default_factory=dict)


@dataclass
class ArmResult:
    """Aggregated result for one experiment arm over all tasks."""

    arm: str
    n: int
    score: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    results: list[TaskResult] = field(default_factory=list)
    unofficial_score: float | None = None
    unofficial_n: int = 0


@dataclass
class ABResult:
    """Full A/B result: per-arm scores + proxy-vs-direct deltas."""

    benchmark_id: str
    name: str
    model: str
    subset: str | None
    arms: dict[str, ArmResult]
    deltas: dict[str, dict]


@runtime_checkable
class ExternalBenchmarkAdapter(Protocol):
    """In-process adapter: we drive each official task via send_chat (the one roof).

    Suited to benchmarks whose items are single chat requests we score ourselves
    (LongBench v2, BigCodeBench). Driven by `run_ab`.
    """

    benchmark_id: str
    name: str

    def load_tasks(
        self,
        subset: str | None = None,
        limit: int | None = None,
        fixture_path: str | Path | None = None,
    ) -> list[Task]:
        """Return official tasks (or a bundled fixture when offline)."""
        ...

    def score(self, task: Task, response_text: str) -> bool:
        """Apply the official scoring rule to one response."""
        ...


@runtime_checkable
class HarnessBenchmarkAdapter(Protocol):
    """External-harness adapter: the official tool has its own runner.

    We invoke the official harness once per arm with the inference client pointed
    at the arm's base_url (direct vs proxy), then parse its results into an
    ArmResult. Suited to SWE-bench, CyberSecEval, AgentDojo, MTEB. Driven by
    `run_external_ab`. `results_fixture` lets tests parse a sample results file
    offline without invoking the real tool.
    """

    benchmark_id: str
    name: str

    def run_arm(
        self,
        arm: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        subset: str | None = None,
        limit: int | None = None,
        results_fixture: str | Path | None = None,
    ) -> ArmResult:
        """Run the official harness for one arm and parse it into an ArmResult."""
        ...


def _usage_tokens(usage: dict) -> tuple[int, int]:
    """Extract (input, output) tokens from an OpenAI-style usage dict."""
    inp = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    out = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    return int(inp), int(out)


def _pick_pricing(model: str, pricing: PricingModel | None) -> PricingModel:
    if pricing is not None:
        return pricing
    name = (model or "").lower()
    if "gpt-6-luna" in name:
        # Same tier source the chat primitive sends (core.api.build_chat_body).
        tier = os.getenv(SERVICE_TIER_ENV, "").strip().lower()
        return PRICING_DEFAULTS["gpt-6-luna-flex" if tier == "flex" else "gpt-6-luna"]
    if "claude" in name or "anthropic" in name:
        return PRICING_DEFAULTS["anthropic"]
    return PRICING_DEFAULTS["openai"]


def _run_arm(
    adapter: ExternalBenchmarkAdapter,
    arm: str,
    tasks: Sequence[Task],
    *,
    base_url: str,
    api_key: str,
    model: str,
    send_fn: SendFn,
    client,
    pricing: PricingModel,
) -> ArmResult:
    results: list[TaskResult] = []
    turn_dicts: list[dict] = []
    for task in tasks:
        text, latency_ms, usage = send_fn(client, base_url, api_key, task.prompt_messages, model)
        inp, out = _usage_tokens(usage)
        correct = adapter.score(task, text)
        results.append(
            TaskResult(
                task_id=task.task_id,
                response_text=text,
                input_tokens=inp,
                output_tokens=out,
                latency_ms=latency_ms,
                correct=correct,
                raw_usage=usage,
            )
        )
        turn_dicts.append({"input_tokens": inp, "output_tokens": out})

    score = mean(1.0 if r.correct else 0.0 for r in results) if results else 0.0
    arm_cost = compute_arm_cost(turn_dicts, pricing)
    return ArmResult(
        arm=arm,
        n=len(results),
        score=round(score, 4),
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cost_usd=round(arm_cost.total_effective_cost_usd, 6),
        results=results,
    )


def arm_result_from_summary(
    arm: str,
    *,
    n: int,
    score: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = MODEL,
    pricing: PricingModel | None = None,
) -> ArmResult:
    """Build an ArmResult (with cost) from an external harness's parsed summary.

    External tools report a score and often token totals but not per-task detail;
    this fills the same ArmResult shape so evidence/deltas are uniform. Cost is
    computed from the token totals via the cache-aware pricing model.
    """
    pricing = _pick_pricing(model, pricing)
    arm_cost = compute_arm_cost([{"input_tokens": input_tokens, "output_tokens": output_tokens}], pricing)
    return ArmResult(
        arm=arm,
        n=n,
        score=round(score, 4),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(arm_cost.total_effective_cost_usd, 6),
        results=[],
    )


def _pct_reduction(direct: float, arm: float) -> float:
    if direct <= 0:
        return 0.0
    return round((direct - arm) / direct * 100.0, 2)


def run_ab(
    adapter: ExternalBenchmarkAdapter,
    *,
    arms: Sequence[str] = ("direct", "proxy_only", "proxy_plus_filter"),
    subset: str | None = None,
    limit: int | None = None,
    model: str = MODEL,
    direct_url: str = DIRECT_URL,
    proxy_url: str = PROXY_URL,
    api_key: str = API_KEY,
    send_fn: SendFn = send_chat,
    configure_proxy: bool = True,
    fixture_path: str | Path | None = None,
    pricing: PricingModel | None = None,
    client=None,
) -> ABResult:
    """Run an official benchmark adapter across arms and compute proxy-vs-direct deltas.

    Offline/testing: pass `fixture_path` (load_tasks reads a bundled sample),
    `send_fn` (a deterministic stub), and `configure_proxy=False` (skip proxy
    admin calls). No network is required in that mode.
    """
    tasks = adapter.load_tasks(subset=subset, limit=limit, fixture_path=fixture_path)
    pricing = _pick_pricing(model, pricing)

    own_client = client is None
    if own_client and configure_proxy:
        import httpx

        client = httpx.Client()

    arm_results: dict[str, ArmResult] = {}
    try:
        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown arm: {arm!r} (known: {sorted(ARMS)})")
            base_url = direct_url if arm == "direct" else proxy_url
            if configure_proxy and arm != "direct":
                apply_arm_config(client, proxy_url, ARMS[arm].get("config_overrides", {}))
            arm_results[arm] = _run_arm(
                adapter,
                arm,
                tasks,
                base_url=base_url,
                api_key=api_key,
                model=model,
                send_fn=send_fn,
                client=client,
                pricing=pricing,
            )
    finally:
        if own_client and client is not None:
            client.close()

    return ABResult(
        benchmark_id=adapter.benchmark_id,
        name=adapter.name,
        model=model,
        subset=subset,
        arms=arm_results,
        deltas=_compute_deltas(arm_results),
    )


def _compute_deltas(arm_results: dict[str, ArmResult]) -> dict[str, dict]:
    """Proxy-arm deltas vs the direct baseline (score, input-token, cost)."""
    deltas: dict[str, dict] = {}
    direct = arm_results.get("direct")
    if direct is None:
        return deltas
    for arm, res in arm_results.items():
        if arm == "direct":
            continue
        deltas[arm] = {
            "score_delta": round(res.score - direct.score, 4),
            "input_token_reduction_pct": _pct_reduction(direct.input_tokens, res.input_tokens),
            "cost_reduction_pct": _pct_reduction(direct.cost_usd, res.cost_usd),
        }
    return deltas


def run_external_ab(
    adapter: HarnessBenchmarkAdapter,
    *,
    arms: Sequence[str] = ("direct", "proxy_only", "proxy_plus_filter"),
    subset: str | None = None,
    limit: int | None = None,
    model: str = MODEL,
    direct_url: str = DIRECT_URL,
    proxy_url: str = PROXY_URL,
    api_key: str = API_KEY,
    configure_proxy: bool = True,
    results_fixtures: dict[str, str | Path] | None = None,
    client=None,
) -> ABResult:
    """Run an external-harness adapter across arms and compute proxy-vs-direct deltas.

    The adapter shells out to the official tool per arm (base_url = direct, then
    proxy). Offline: pass `results_fixtures={arm: path}` so each arm parses a
    sample results file instead of invoking the real harness, and
    `configure_proxy=False` to skip proxy admin calls.
    """
    results_fixtures = results_fixtures or {}
    own_client = client is None
    if own_client and configure_proxy:
        import httpx

        client = httpx.Client()

    arm_results: dict[str, ArmResult] = {}
    try:
        for arm in arms:
            if arm not in ARMS:
                raise ValueError(f"unknown arm: {arm!r} (known: {sorted(ARMS)})")
            base_url = direct_url if arm == "direct" else proxy_url
            if configure_proxy and arm != "direct":
                apply_arm_config(client, proxy_url, ARMS[arm].get("config_overrides", {}))
            arm_results[arm] = adapter.run_arm(
                arm,
                base_url=base_url,
                api_key=api_key,
                model=model,
                subset=subset,
                limit=limit,
                results_fixture=results_fixtures.get(arm),
            )
    finally:
        if own_client and client is not None:
            client.close()

    return ABResult(
        benchmark_id=adapter.benchmark_id,
        name=adapter.name,
        model=model,
        subset=subset,
        arms=arm_results,
        deltas=_compute_deltas(arm_results),
    )


def ab_result_to_dict(ab: ABResult) -> dict:
    """Serialize an ABResult to a stable JSON-compatible dict."""
    arms: dict[str, dict] = {}
    for arm, r in ab.arms.items():
        entry = {
            "arm": r.arm,
            "n": r.n,
            "score": r.score,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "results": [asdict(t) for t in r.results],
        }
        if r.unofficial_n > 0:
            entry["unofficial_score"] = r.unofficial_score
            entry["unofficial_n"] = r.unofficial_n
        arms[arm] = entry
    return {
        "benchmark_id": ab.benchmark_id,
        "name": ab.name,
        "model": ab.model,
        "subset": ab.subset,
        "arms": arms,
        "deltas": ab.deltas,
    }


def write_harness_evidence(ab: ABResult, out_path: str | Path, output_format: str = "markdown") -> Path:
    """Write A/B evidence (markdown or JSON) for a harness run."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ab_result_to_dict(ab), f, indent=2, ensure_ascii=False)
        return out_path

    lines: list[str] = [
        f"# {ab.name} — direct vs proxy A/B\n\n",
        f"- Benchmark: `{ab.benchmark_id}`\n",
        f"- Model: `{ab.model}`\n",
        f"- Subset: `{ab.subset}`\n\n",
        "Archolith is middleware: the advertisable result is the delta versus the "
        "direct baseline (official score preserved, tokens/cost reduced), not a standalone score.\n\n",
        "| Arm | N | Score | Input tokens | Output tokens | Cost (USD) |\n",
        "|-----|---|-------|--------------|---------------|------------|\n",
    ]
    for arm, r in ab.arms.items():
        lines.append(
            f"| {arm} | {r.n} | {r.score:.3f} | {r.input_tokens:,} | "
            f"{r.output_tokens:,} | {r.cost_usd:.6f} |\n"
        )
    if ab.deltas:
        lines.append("\n## Delta vs direct\n\n")
        lines.append("| Arm | Score delta | Input token reduction | Cost reduction |\n")
        lines.append("|-----|-------------|-----------------------|----------------|\n")
        for arm, d in ab.deltas.items():
            lines.append(
                f"| {arm} | {d['score_delta']:+.3f} | "
                f"{d['input_token_reduction_pct']:+.2f}% | {d['cost_reduction_pct']:+.2f}% |\n"
            )
    if any(r.unofficial_n > 0 for r in ab.arms.values()):
        lines.append("\n## Unofficial judge (substance; not the LongMemEval score)\n\n")
        lines.append("| Arm | N judged | Unofficial score |\n")
        lines.append("|-----|----------|------------------|\n")
        for arm, r in ab.arms.items():
            if r.unofficial_n > 0:
                lines.append(f"| {arm} | {r.unofficial_n} | {r.unofficial_score:.3f} |\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path
