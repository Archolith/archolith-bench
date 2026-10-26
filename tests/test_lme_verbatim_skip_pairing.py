"""`ready_episodes_time_unpaired` accepts a verbatim-skipped client only through its READY projection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_ingest():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "ingest.py"
    spec = importlib.util.spec_from_file_location("lme_ingest_verbatim_skip_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_ingest_verbatim_skip_test"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


def _time_unpaired_query(monkeypatch: pytest.MonkeyPatch) -> str:
    queries: list[str] = []

    def fake_counts(query: str, width: int) -> list[int]:
        queries.append(query)
        return [0] * width

    monkeypatch.setattr(ingest, "_cypher_counts", fake_counts)
    ingest._integrity_counts("lme-x", [])
    matches = [q for q in queries if "q.resolved_episode_uuid" in q and "valid_at" in q]
    assert len(matches) == 1
    return matches[0]


def test_skip_needs_both_substage_and_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _time_unpaired_query(monkeypatch)
    assert (
        "q.processing_substage = 'verbatim_skipped' AND q.extraction_skipped_for IS NOT NULL AS skipped"
        in query
    )


def test_skip_passes_only_through_ready_projection_in_namespace_at_same_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _time_unpaired_query(monkeypatch)
    assert "OPTIONAL MATCH (p:Episodic {namespace:'lme-x'})" in query
    assert "WHERE skipped AND p.uuid = q.extraction_skipped_for" in query
    assert "coalesce(p.is_evidence_projection, false) AND p.processing_state = 'READY'" in query
    assert (
        "datetime(toString(ps[0].reference_time)).epochMillis "
        "= datetime(toString(q.reference_time)).epochMillis THEN 0 ELSE 1 END" in query
    )


def test_non_skipped_nodes_keep_graphiti_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _time_unpaired_query(monkeypatch)
    assert "OPTIONAL MATCH (g:Episodic {group_id:'lme-x'})" in query
    assert "WHEN size(gs) <> 1 OR q.reference_time IS NULL THEN 1" in query
    assert (
        "WHEN datetime(toString(gs[0].valid_at)).epochMillis "
        "<> datetime(toString(q.reference_time)).epochMillis THEN 1 ELSE 0 END" in query
    )
    # Store-only nodes stay out entirely (#44).
    assert "coalesce(q.extraction_mode, '') <> 'store_only'" in query
