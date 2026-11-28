"""External benchmark harness — real official benchmarks run as direct-vs-proxy A/B.

Archolith is middleware, not a model, so these adapters never claim a standalone
benchmark score. They run the official dataset + scorer twice (client base_url =
direct, then = proxy) and report the delta the proxy/filter produces: official
score preserved + tokens/cost reduced.

All benchmarks live under one roof here behind ExternalBenchmarkAdapter.
"""

from __future__ import annotations

from .base import (
    ABResult,
    ArmResult,
    ExternalBenchmarkAdapter,
    HarnessBenchmarkAdapter,
    Task,
    TaskResult,
    ab_result_to_dict,
    arm_result_from_summary,
    run_ab,
    run_external_ab,
    write_harness_evidence,
)
from .bigcodebench import BigCodeBenchHardAdapter
from .external import (
    AgentDojoAdapter,
    CyberSecEvalAdapter,
    ExternalCliAdapter,
    MtebAdapter,
    SweBenchAdapter,
)
from .longbench_v2 import LongBenchV2Adapter
from .longmemeval import LongMemEvalAdapter, LongMemEvalMemoryAdapter
from .checkpoint import MemoryCheckpoint, checkpoint_path_for, checkpoint_prompt_tag
from .memory_ab import (
    DEFAULT_MEMORY_ARMS,
    NO_MEMORY,
    VALUE_RECALL,
    MemoryQAAdapter,
    MenhirClient,
    run_memory_ab,
)
from .menhir_client import (
    HttpMenhirClient,
    Phase3MenhirClient,
    StubMenhirClient,
    StubPhase3Client,
    StubScalarStateClient,
)
from .menhir_scalar_state import (
    SCALAR_FIXTURE_SETS,
    MenhirScalarStateAdapter,
    ScalarStateCase,
    ScalarStateResult,
    default_scalar_state_cases,
    is_scalar_state,
    run_scalar_state,
    scalar_state_result_to_dict,
    third_party_scalar_state_cases,
    write_scalar_state_evidence,
)
from .scalar_bolt import ScalarBoltReader
from .menhir_phase3 import (
    MenhirPhase3Adapter,
    Phase3Case,
    Phase3Result,
    default_phase3_cases,
    is_phase3,
    phase3_result_to_dict,
    run_phase3,
    write_phase3_evidence,
)
from .phase3_scenarios import (
    Assertion,
    Post,
    Scenario,
    ScenarioResult,
    default_scenarios,
    run_scenario,
    run_scenario_suite,
    scenario_result_to_dict,
    suite_verdict,
)
from .scoring import LLMJudgeScorer, UnofficialJudgeScorer

# Registry of available real-harness adapters, keyed by benchmark_id (the one roof).
# In-process: run_ab. ExternalCliAdapter: run_external_ab. MemoryQAAdapter: run_memory_ab.
ADAPTERS: dict[str, object] = {
    a.benchmark_id: a
    for a in (
        LongBenchV2Adapter(),
        BigCodeBenchHardAdapter(),
        LongMemEvalAdapter(),
        LongMemEvalMemoryAdapter(),
        SweBenchAdapter(),
        CyberSecEvalAdapter(),
        AgentDojoAdapter(),
        MtebAdapter(),
        MenhirPhase3Adapter(),
        MenhirScalarStateAdapter(),
    )
}


def get_adapter(benchmark_id: str):
    """Return a registered adapter by benchmark_id, or raise with the known set."""
    try:
        return ADAPTERS[benchmark_id]
    except KeyError as e:
        raise KeyError(
            f"unknown benchmark_id: {benchmark_id!r} (available: {sorted(ADAPTERS)})"
        ) from e


def is_external(adapter: object) -> bool:
    """True if the adapter wraps an external CLI (run via run_external_ab)."""
    return isinstance(adapter, ExternalCliAdapter)


def is_memory(adapter: object) -> bool:
    """True if the adapter is an ingest-then-recall memory benchmark (run via run_memory_ab)."""
    return hasattr(adapter, "sessions") and hasattr(adapter, "load_items")


__all__ = [
    "ABResult",
    "ADAPTERS",
    "AgentDojoAdapter",
    "ArmResult",
    "BigCodeBenchHardAdapter",
    "CyberSecEvalAdapter",
    "DEFAULT_MEMORY_ARMS",
    "ExternalBenchmarkAdapter",
    "ExternalCliAdapter",
    "HarnessBenchmarkAdapter",
    "HttpMenhirClient",
    "LLMJudgeScorer",
    "UnofficialJudgeScorer",
    "LongBenchV2Adapter",
    "LongMemEvalAdapter",
    "LongMemEvalMemoryAdapter",
    "MemoryCheckpoint",
    "MemoryQAAdapter",
    "MenhirClient",
    "Assertion",
    "MenhirPhase3Adapter",
    "MenhirScalarStateAdapter",
    "MtebAdapter",
    "Phase3Case",
    "Phase3MenhirClient",
    "Phase3Result",
    "Post",
    "ScalarBoltReader",
    "ScalarStateCase",
    "ScalarStateResult",
    "Scenario",
    "ScenarioResult",
    "SCALAR_FIXTURE_SETS",
    "default_phase3_cases",
    "default_scalar_state_cases",
    "default_scenarios",
    "third_party_scalar_state_cases",
    "is_phase3",
    "is_scalar_state",
    "phase3_result_to_dict",
    "run_phase3",
    "run_scalar_state",
    "run_scenario",
    "run_scenario_suite",
    "scalar_state_result_to_dict",
    "scenario_result_to_dict",
    "suite_verdict",
    "write_phase3_evidence",
    "write_scalar_state_evidence",
    "checkpoint_path_for",
    "checkpoint_prompt_tag",
    "NO_MEMORY",
    "VALUE_RECALL",
    "StubMenhirClient",
    "StubPhase3Client",
    "StubScalarStateClient",
    "SweBenchAdapter",
    "Task",
    "TaskResult",
    "ab_result_to_dict",
    "arm_result_from_summary",
    "get_adapter",
    "is_external",
    "is_memory",
    "run_ab",
    "run_external_ab",
    "run_memory_ab",
    "write_harness_evidence",
]
