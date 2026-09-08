"""First-class Menhir capability runners for the archolith-bench CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..core.evidence import EvidenceRecord, current_commit, publish_evidence


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"
MENHIR_FRONTIER_SRC = REPO_ROOT.parent / "menhir-frontier" / "src"


def ensure_menhir_frontier_path() -> None:
    for path in (REPO_ROOT, MENHIR_FRONTIER_SRC):
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_r1(fixture_path: Path | None = None, *, k: int = 5) -> dict:
    from ..r1.models import R1Fixture
    from ..r1.runner import R1BenchmarkRunner, build_stub_conditions

    fixture = R1Fixture.from_file(fixture_path or FIXTURES / "r1_demo.json")
    return R1BenchmarkRunner(fixture, build_stub_conditions(fixture), k=k).run()


def run_bootstrap_hygiene(
    fixture_path: Path | None = None,
    *,
    menhir_url: str | None = None,
    api_key: str = "",
) -> dict:
    """Run the deterministic policy gate or a live black-box throwaway probe."""
    from ..bootstrap_hygiene import BootstrapFixture, BootstrapHygieneRunner

    fixture = BootstrapFixture.from_file(
        fixture_path or FIXTURES / "menhir_bootstrap_hygiene.json"
    )
    if menhir_url is None:
        return BootstrapHygieneRunner(fixture).run()

    from ..harness.memory_ab import assert_not_production
    from ..harness.menhir_client import HttpMenhirClient

    # Live probe against a real instance would read the operator's memories.
    # Guarded here because the check belongs at every client construction, not
    # only at the CLI call sites.
    assert_not_production(menhir_url)

    print(f"bootstrap-hygiene live target: {menhir_url}")
    with HttpMenhirClient(menhir_url, api_key=api_key) as client:
        return BootstrapHygieneRunner(fixture, client=client).run()


def run_r2_facet(
    fixture_path: Path | None = None,
    *,
    no_traces: bool = False,
    facet_scope: str = "all",
) -> dict:
    from ..facet.models import FacetFixture
    from ..facet.runner import FacetBenchmarkRunner

    fixture = FacetFixture.from_file(fixture_path or FIXTURES / "facet_demo.json")
    if facet_scope == "regular":
        fixture = _strip_structural_facets(fixture)
    return FacetBenchmarkRunner(fixture).run(include_traces=not no_traces)


def run_r3_belief(fixture_path: Path | None = None) -> dict:
    ensure_menhir_frontier_path()
    from ..r3.models import BeliefFixture
    from ..r3.runner import R3BenchmarkRunner

    fixture = BeliefFixture.from_file(fixture_path or FIXTURES / "r3_ce_willow.json")
    return R3BenchmarkRunner(fixture).run()


def run_oracle(fixture_path: Path | None = None, *, no_traces: bool = False) -> dict:
    from ..oracle.models import OracleFixture
    from ..oracle.runner import OracleBenchmarkRunner
    from ..oracle.validate import has_errors, validate_oracle_fixture

    fixture = OracleFixture.from_file(fixture_path or FIXTURES / "oracle_demo.json")
    findings = validate_oracle_fixture(fixture)
    if has_errors(findings):
        raise ValueError("oracle fixture has validation errors: " + "; ".join(str(f) for f in findings))
    return OracleBenchmarkRunner(fixture).run(include_traces=not no_traces)


def run_intent(fixture_path: Path | None = None) -> dict:
    from ..intent.models import IntentFixture
    from ..intent.runner import IntentBenchmarkRunner
    from ..intent.validate import has_errors, validate_intent_fixture

    fixture = IntentFixture.from_file(fixture_path or FIXTURES / "intent_floor_corpus.json")
    findings = validate_intent_fixture(fixture)
    if has_errors(findings):
        raise ValueError("intent fixture has validation errors: " + "; ".join(str(f) for f in findings))
    return IntentBenchmarkRunner(fixture).run()


def run_l4_artifacts(fixture_path: Path | None = None) -> dict:
    from ..l4.models import ArtifactFixture
    from ..l4.runner import run_l4_benchmark

    fixture = ArtifactFixture.from_file(fixture_path or FIXTURES / "l4_failure_demo.json")
    return run_l4_benchmark(fixture)


def run_r5_structure_temporal(fixture_path: Path | None = None, *, k: int = 3) -> dict:
    ensure_menhir_frontier_path()
    from ..r5.runner import R5BenchRunner, R5Fixture

    fixture = R5Fixture.from_file(fixture_path or FIXTURES / "r5_seed_blast_radius.json")
    return R5BenchRunner(fixture, k=k).run()


def write_json_artifact(artifact: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return out_path


def print_summary(name: str, artifact: dict) -> None:
    print(f"\n{name}: {artifact.get('fixture', 'artifact')}")
    rows = metric_rows_for(name, artifact)
    if not rows:
        print("  (no compact metric rows available)")
        return
    keys = list(rows[0].keys())
    print("  " + " | ".join(keys))
    print("  " + " | ".join("---" for _ in keys))
    for row in rows:
        print("  " + " | ".join(_fmt(row.get(k)) for k in keys))


def publish_menhir_evidence(
    *,
    title: str,
    command: str,
    ability: str,
    fixture_or_live_source: str,
    artifact: dict,
    out_path: Path,
    model_provider: str = "offline deterministic fixture",
    public_copy_allowed: bool = False,
    caveats: list[str] | None = None,
) -> Path:
    record = EvidenceRecord(
        title=title,
        command=command,
        commit=current_commit(REPO_ROOT),
        product="menhir",
        ability=ability,
        fixture_or_live_source=fixture_or_live_source,
        model_provider=model_provider,
        environment_caveats=caveats or [
            "Offline fixture output demonstrates harness behavior only.",
            "Do not use as public launch evidence unless HEADLINE-NUMBERS.md records it as active.",
        ],
        public_copy_allowed=public_copy_allowed,
        metric_rows=metric_rows_for(ability, artifact),
        artifact=artifact,
    )
    return publish_evidence(record, out_path)


def metric_rows_for(kind: str, artifact: dict) -> list[dict[str, Any]]:
    key = kind.lower()
    if "bootstrap" in key or "hygiene" in key:
        return [{"passed": artifact.get("passed"), **artifact.get("metrics", {})}]
    if "r1" in key or "hybrid retrieval" in key:
        return _condition_rows(artifact, "conditions")
    if "facet" in key:
        rows: list[dict[str, Any]] = []
        for mode, conditions in artifact.get("modes", {}).items():
            for condition, data in conditions.items():
                rows.append({"mode": mode, "condition": condition, **_select_metrics(data.get("metrics", {}))})
        return rows
    if "r3" in key or "belief" in key:
        return _condition_rows(artifact, "conditions")
    if "oracle" in key:
        return _condition_rows(artifact, "conditions")
    if "intent" in key:
        ic = artifact.get("intent_correct_at_1", {})
        gate = artifact.get("promotion_gate", {})
        return [{
            "semantic_only": ic.get("semantic_only"),
            "oracle_default_no_intent": ic.get("oracle_default_no_intent"),
            "intent_on": ic.get("intent_on"),
            "shuffle_ablation": ic.get("shuffle_ablation"),
            "graduates": gate.get("graduates"),
        }]
    if "l4" in key or "artifact" in key:
        rows = []
        for task in artifact.get("tasks", []):
            for condition, data in task.get("conditions", {}).items():
                rows.append({"task": task.get("task"), "condition": condition, **data.get("metrics", {})})
        return rows
    if "r5" in key or "structure" in key:
        return _condition_rows(artifact, "conditions")
    return []


def run_smoke(output_dir: Path, *, publish_dir: Path | None = None) -> dict[str, dict]:
    runs = {
        "r1": ("hybrid retrieval tuning", run_r1, "r1_run.json"),
        "r2-facet": ("facet retrieval", run_r2_facet, "facet_run.json"),
        "oracle": ("oracle combiner", run_oracle, "oracle_run.json"),
        "intent": ("intent-aware retrieval", run_intent, "intent_run.json"),
        "l4-artifacts": ("institutional artifact memory", run_l4_artifacts, "l4_run.json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for name, (ability, fn, filename) in runs.items():
        artifact = fn()
        results[name] = artifact
        write_json_artifact(artifact, output_dir / filename)
        print_summary(name, artifact)
        if publish_dir:
            publish_menhir_evidence(
                title=f"Menhir {name} smoke evidence",
                command=f"archolith-bench menhir {name}",
                ability=ability,
                fixture_or_live_source=str(output_dir / filename),
                artifact=artifact,
                out_path=publish_dir / f"menhir-{name}-smoke.md",
            )
    return results


def _condition_rows(artifact: dict, key: str) -> list[dict[str, Any]]:
    return [
        {"condition": condition, **_select_metrics(data.get("metrics", {}))}
        for condition, data in artifact.get(key, {}).items()
    ]


def _select_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "recall_at_5",
        "exact_string_recall",
        "symbol_recall",
        "stale_hit_rate",
        "wrong_scope_injection_rate",
        "support_sufficiency",
        "current_truth_suppression_accuracy",
        "historical_context_preservation",
        "stale_current_assertion_rate",
        "poisoned_context_injection_rate",
        "failed_approach_surfaced",
        "first_action_quality",
        "culprit_at_1",
        "noise_at_1",
        "latency_ms",
    )
    return {k: metrics[k] for k in keep if k in metrics}


def _strip_structural_facets(fixture):
    import copy

    fx = copy.deepcopy(fixture)
    for memory in fx.memories:
        memory.facets.file, memory.facets.symbol, memory.facets.test = set(), set(), set()
    for query in fx.queries:
        query.facets.file, query.facets.symbol, query.facets.test = set(), set(), set()
    return fx


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
