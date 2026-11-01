"""Offline tests for retrieval_quality.py's arms, gates and evidence (no live graph, no spend).

Counterexamples for the 2026-11 review findings: a vacuous explainability gate, a silently
empty baseline passing the gates, a 0 = 0 MRR tie passing, and a menhir arm that ignores the
source-memory block the S4 harness renders.
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import retrieval_quality as rq  # noqa: E402  -- real shipped code under test


def _row(**kw):
    base = {"qid": "q", "qtype": "t", "errors": {}, "m_results_n": 1, "m_attributed_n": 1,
            "m_supp": None, "gv_supp": None, "m_rank": None, "gv_rank": None}
    base.update(kw)
    return base


def test_score_attribution_needs_a_backend_number():
    assert rq.has_score_attribution({"retrieval_score": 0.4}) is True
    assert rq.has_score_attribution({"retrieval_score": 0}) is True
    assert rq.has_score_attribution({"retrieval_score": None}) is False
    assert rq.has_score_attribution({"final_score": 0.9, "relevance_basis": "x",
                                     "retrieval_score_kind": "graphiti_rrf"}) is False
    assert rq.has_score_attribution({"retrieval_score": True}) is False
    assert rq.has_score_attribution("x") is False


def test_route_defaults_alone_do_not_pass_gate3():
    # The fields the /api/recall route fills by default pass the old presence check...
    defaulted = {"final_score": 0.0, "retrieval_score_kind": "graphiti_rrf",
                 "relevance_basis": "legacy_rrf_threshold_unvalidated", "retrieval_score": None}
    assert rq.has_explainability(defaulted) is True
    # ...but not the gate.
    assert rq.has_score_attribution(defaulted) is False
    gates = rq.compute_gates([_row(m_supp=1, gv_supp=None, m_results_n=2, m_attributed_n=1)])
    assert gates["gate3"] is False and gates["overall"] is False


def test_mrr_tie_at_zero_fails_gate2():
    gates = rq.compute_gates([_row(), _row(qid="q2")])
    assert gates["m_mrr"] == 0.0 == gates["b_mrr"]
    assert gates["gate2"] is False and gates["overall"] is False


def test_hit3_must_strictly_beat_baseline():
    gates = rq.compute_gates([_row(m_supp=1, gv_supp=1)])
    assert gates["gate1"] is False
    gates = rq.compute_gates([_row(m_supp=1, gv_supp=4)])
    assert gates["gate1"] is True and gates["gate2"] is True and gates["overall"] is True


def test_an_errored_row_makes_the_run_fail_even_when_clean_rows_pass():
    rows = [_row(m_supp=1, gv_supp=4), _row(qid="q2", errors={"graphiti_vector": "boom"})]
    gates = rq.compute_gates(rows)
    assert gates["n_clean"] == 1 and gates["n_incomplete"] == 1
    assert gates["gate1"] is True
    assert gates["overall"] is False


class _FailingGraphiti:
    async def search_(self, *a, **k):
        raise RuntimeError("neo4j auth failed")

    async def search(self, *a, **k):
        raise RuntimeError("neo4j auth failed")


class _FakeMenhir:
    def recall(self, ns, q, limit=20):
        return ["the pizza had extra cheese"]


ITEM = {"question_id": "abc", "question_type": "single-session-user", "question": "what pizza?",
        "answer": "cheese pizza",
        "haystack_sessions": [[{"role": "user", "content": "pizza with cheese", "has_answer": True}]]}


def test_baseline_failure_is_recorded_not_an_empty_arm(monkeypatch):
    monkeypatch.setattr(rq, "RETRY_DELAY_S", 0.0)
    captured = {}

    def fake_recall(mc, cap, q, ns):
        return (["pizza with cheese"], ["pizza with cheese"], [{"retrieval_score": 0.5}])

    monkeypatch.setattr(rq, "menhir_recall", fake_recall)

    async def _go():
        return await rq.score_item(ITEM, _FakeMenhir(), captured, _FailingGraphiti())

    row = asyncio.run(_go())
    assert set(row["errors"]) == {"graphiti_vector", "graphiti_hybrid"}
    assert "neo4j auth failed" in row["errors"]["graphiti_vector"]
    assert rq.compute_gates([row])["overall"] is False


def test_call_with_retry_retries_then_reports():
    calls = []

    def flaky():
        calls.append(1)
        raise httpx.ConnectError("refused")

    value, err = rq.call_with_retry(flaky, attempts=3, delay=0)
    assert value is None and "ConnectError" in err and len(calls) == 3

    seq = iter([RuntimeError("once"), "ok"])

    def second_time():
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v

    assert rq.call_with_retry(second_time, attempts=3, delay=0) == ("ok", None)


def _recall_transport(body):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/recall"
        sent = json.loads(request.content)
        assert sent["include_session"] is True and sent["include_invalidated"] is True
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


def test_menhir_arm_is_the_s4_context_with_source_memories():
    body = {
        "results": [{"uuid": "u1", "name": "dinner", "content": "ate out", "memory_type": "SEMANTIC",
                     "final_score": 0.7, "retrieval_score": 0.9}],
        "source_memories": [{"uuid": "s1", "reference_time": "2023-05-01",
                             "content": "I had a pizza with extra cheese"}],
    }
    mc, captured = rq.make_menhir_client("http://127.0.0.1:9", transport=_recall_transport(body))
    packet, texts, raw = rq.menhir_recall(mc, captured, "what pizza?", "lme-abc")
    assert texts == ["dinner ate out"]
    assert raw[0]["retrieval_score"] == 0.9
    assert len(packet) == 2 and "pizza with extra cheese" in packet[-1]
    support = rq.evidence_tokens(ITEM)
    # results-only misses the evidence; the S4 context finds it in the source-memory block.
    assert rq.support_rank(support, texts) is None
    r = rq.support_rank(support, list(packet))
    assert r == 2 and rq.snippet_kind(packet, r) == "source_memories"


def test_menhir_arm_raises_when_body_not_captured():
    mc, captured = rq.make_menhir_client("http://127.0.0.1:9",
                                         transport=_recall_transport({"results": []}))
    captured["data"] = "stale"
    mc.recall = lambda ns, q, limit=20: []  # bypass HTTP: nothing is captured
    with pytest.raises(RuntimeError):
        rq.menhir_recall(mc, captured, "q", "ns")


def test_resume_redoes_errored_rows(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(_row(qid="a")) + "\n"
                 + json.dumps(_row(qid="b", errors={"menhir": "x"})) + "\n", encoding="utf-8")
    assert set(rq.load_done(p)) == {"a"}


def test_require_env_refuses_missing_provenance(monkeypatch):
    for k in ("MENHIR_URL", "OPENAI_API_KEY", "MENHIR_SERVED_SRC", "LME_IR_OUT"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="MENHIR_SERVED_SRC"):
        rq._require_env()


def test_artifacts_carry_rows_gates_and_provenance(tmp_path):
    rows = [_row(m_supp=1, gv_supp=4, m_rank=2, gv_rank=None, mr_supp=None, mr_rank=None,
                 gh_supp=3, gh_rank=None, m_supp_kind="recall",
                 m_relevance_basis={"legacy_rrf_threshold_unvalidated": 1})]
    gates = rq.compute_gates(rows)
    meta = {"run_id": "r", "timestamp": "t", "menhir_commit": "c", "menhir_dirty": False,
            "bench_commit": "b", "bench_dirty": False, "snapshot": "s", "menhir_profile": "ship",
            "topk": 20}
    rq.emit_artifacts(tmp_path, rows, gates, meta)
    data = json.loads((tmp_path / "ir-gate.json").read_text(encoding="utf-8"))
    assert data["rows"] == rows and data["gates"]["overall"] is True
    assert data["menhir_commit"] == "c" and data["snapshot"] == "s"
    assert data["menhir_relevance_basis"] == {"legacy_rrf_threshold_unvalidated": 1}
    assert "graphiti_vector" in (tmp_path / "ir-gate.md").read_text(encoding="utf-8")
