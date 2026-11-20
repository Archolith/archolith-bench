"""Staged arm config: loading, strictness, resolved config and run id."""

from __future__ import annotations

from pathlib import Path

import pytest

from archolith_bench.staged.config import ArmConfig, arm_from_dict, load_arm, resolve, run_id

ARMS = Path(__file__).resolve().parent.parent / "arms"


def _resolved(arm: ArmConfig | None = None, **over) -> dict:
    kw = dict(
        benchmark_id="b", fixture_path="f.json", fixture_sha256="aa", prompt_fingerprint="pp",
        renderer_sha256="rr", menhir_commit="c1", namespace_template="lme-{question_id}",
        model="m", judge={"scorer": "llm-judge", "model": "j"},
    )
    kw.update(over)
    return resolve(arm or ArmConfig(name="a", context="legacy"), **kw)


def test_arm_files_load() -> None:
    none = load_arm(ARMS / "no_memory.yaml")
    rec = load_arm(ARMS / "menhir_recall.yaml")
    assert none.context == "none"
    assert (rec.context, rec.recall_limit, rec.source_memories) == ("legacy", 20, True)
    assert not rec.counted_authority_only and rec.counted_verdict_limit is None and not rec.drill_down


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValueError, match="bogus"):
        arm_from_dict({"name": "a", "context": "none", "bogus": 1})
    with pytest.raises(ValueError, match="limt"):
        arm_from_dict({"name": "a", "context": "legacy", "recall": {"limt": 3}})


def test_drill_down_rejected() -> None:
    with pytest.raises(ValueError, match="drill-down"):
        arm_from_dict({"name": "a", "context": "legacy", "render": {"drill_down": True}})


def test_only_builtin_prompt_accepted() -> None:
    with pytest.raises(ValueError):
        arm_from_dict({"name": "a", "context": "none", "answer": {"prompt": "other"}})


def test_run_id_stable_and_sensitive() -> None:
    base = _resolved()
    assert run_id(base) == run_id(_resolved())
    assert run_id(base).startswith("a-")
    variants = [
        _resolved(ArmConfig(name="a", context="legacy", recall_limit=5)),
        _resolved(benchmark_id="x"),
        _resolved(fixture_path="g.json"),
        _resolved(fixture_sha256="bb"),
        _resolved(prompt_fingerprint="qq"),
        _resolved(renderer_sha256="ss"),
        _resolved(menhir_commit="c2"),
        _resolved(namespace_template="x-{question_id}"),
        _resolved(model="m2"),
        _resolved(judge={"scorer": "llm-judge", "model": "j2"}),
    ]
    ids = {run_id(v) for v in variants} | {run_id(base)}
    assert len(ids) == len(variants) + 1
