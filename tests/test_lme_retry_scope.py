"""retry.py must only touch Episodic nodes inside the LME namespace prefix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


retry = _load("lme_retry_scope_test", ROOT / "scripts" / "longmemeval" / "lib" / "retry.py")


@pytest.fixture
def queries(monkeypatch):
    seen: list[str] = []

    def fake(query, timeout=300):
        seen.append(query)
        return []

    monkeypatch.setattr(retry, "_cypher", fake)
    return seen


def _run_both():
    retry.all_failed_uuids()
    retry.global_counts()


def test_default_prefix_scopes_both_queries(queries, monkeypatch) -> None:
    monkeypatch.delenv("LME_NS_PREFIX", raising=False)
    _run_both()
    assert len(queries) == 2
    for q in queries:
        assert "e.namespace STARTS WITH 'lme-'" in q
    assert "processing_state='FAILED'" in queries[0]


def test_custom_prefix_replaces_default(queries, monkeypatch) -> None:
    monkeypatch.setenv("LME_NS_PREFIX", "lme-zz9-")
    _run_both()
    for q in queries:
        assert "e.namespace STARTS WITH 'lme-zz9-'" in q
        assert "STARTS WITH 'lme-'" not in q
    assert "processing_state='FAILED'" in queries[0]


@pytest.mark.parametrize("prefix", ["lme-'x", "lme-\\x", "' OR 1=1 //"])
def test_quote_or_backslash_prefix_raises_before_any_query(queries, monkeypatch, prefix) -> None:
    monkeypatch.setenv("LME_NS_PREFIX", prefix)
    with pytest.raises(SystemExit):
        retry.all_failed_uuids()
    with pytest.raises(SystemExit):
        retry.global_counts()
    assert queries == []
