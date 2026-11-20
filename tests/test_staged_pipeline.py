"""Staged pipeline: byte-identical context and messages vs today's driver, resume, report."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from archolith_bench.core.metrics import PRICING_DEFAULTS
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter
from archolith_bench.harness.memory_ab import _run_memory_arm
from archolith_bench.harness.menhir_client import HttpMenhirClient, render_recall_snippets
from archolith_bench.staged.config import ArmConfig
from archolith_bench.staged.pipeline import (
    build_report,
    render_context,
    run_answer,
    run_judge,
    run_search,
)
from archolith_bench.staged.rundir import RunDir

_ROOT = Path(__file__).resolve().parent
_GOLDEN = json.loads((_ROOT / "fixtures" / "legacy_recall_render_golden.json").read_text(encoding="utf-8"))
_CASES = _GOLDEN["cases"]
_RESPONSES = _GOLDEN["responses"]
_ITEMS = json.loads((_ROOT.parent / "fixtures" / "longmemeval_sample.json").read_text(encoding="utf-8"))
_ADAPTER = LongMemEvalMemoryAdapter()
_DATE = "2023/02/15 (Wed) 19:47"


def _arm(case: dict) -> ArmConfig:
    return ArmConfig(
        name="menhir_recall", context="legacy", recall_limit=case["limit"],
        counted_authority_only=case["counted_authority_only"],
        source_memories=case["source_memories"],
        counted_verdict_limit=case["counted_verdict_limit"],
    )


def test_context_byte_identical_for_every_golden_case() -> None:
    checked = 0
    for case in _CASES:
        response = json.loads(json.dumps(_RESPONSES[case["response"]]))
        if case["drill_down"]:
            out = render_recall_snippets(
                response, case["limit"], drill_down=True,
                counted_authority_only=case["counted_authority_only"],
                source_memories=case["source_memories"],
                counted_verdict_limit=case["counted_verdict_limit"],
            )
            assert "\n".join(out) == "\n".join(case["out"])
            continue
        assert render_context(_arm(case), response) == "\n".join(case["out"])
        checked += 1
    assert checked > 100


def test_menhir_and_none_context() -> None:
    assert render_context(ArmConfig(name="m", context="menhir"), {"context": "x"}) == "x"
    assert render_context(ArmConfig(name="m", context="menhir"), {}) == ""
    assert render_context(ArmConfig(name="n", context="none"), None) == ""


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return json.loads(json.dumps(self._data))


class _Http:
    def __init__(self, data):
        self.data = data

    def post(self, url, json=None, headers=None):  # noqa: A002
        return _Resp(self.data)


class _FakeSystem:
    def __init__(self, response):
        self.response = response

    async def search(self, ns, query, **kw):
        return json.loads(json.dumps(self.response))

    async def context(self, ns, query, **kw):
        return self.response


def _capture(store: list):
    def send(client, base_url, api_key, messages, model, strict=False):  # noqa: ANN001
        store.append(messages)
        return "ans", 1.0, {"input_tokens": 1, "output_tokens": 1}

    return send


def _items() -> list[dict]:
    dated = dict(_ITEMS[0], question_date=_DATE)
    undated = {k: v for k, v in _ITEMS[1].items() if k != "question_date"}
    return [dated, undated, dict(_ITEMS[2], question_date=_DATE)]


def _old_messages(arm: str, case: dict | None, items: list[dict]) -> list:
    sent: list = []
    client = None
    if case is not None:
        client = HttpMenhirClient(
            "http://menhir.test", counted_authority_only=case["counted_authority_only"],
            source_memories=case["source_memories"],
        )
        client._client = _Http(_RESPONSES[case["response"]])
    _run_memory_arm(
        _ADAPTER, arm, items, client=client, chat_client=None, send_fn=_capture(sent),
        chat_base_url="u", api_key="k", model="m", recall_limit=case["limit"] if case else 20,
        pricing=PRICING_DEFAULTS["anthropic"], reset_memory=False, reset_confirmed=False,
        dry_run_reset=False, recall_only=True,
    )
    return sent


def _new_messages(tmp_path, arm: ArmConfig, response, items: list[dict]) -> list:
    sent: list = []
    rd = RunDir(tmp_path, "r")
    asyncio.run(run_search(rd, arm, items, _ADAPTER, _FakeSystem(response)))
    run_answer(rd, arm, _ADAPTER, send_fn=_capture(sent), chat_client=None,
               chat_base_url="u", api_key="k", model="m")
    return sent


@pytest.mark.parametrize("idx", [0, 5, 20, 60, 100])
def test_answer_messages_identical_to_today_driver(tmp_path, idx: int) -> None:
    case = [c for c in _CASES if not c["drill_down"]][idx]
    items = _items()
    old = _old_messages("menhir_recall", case, items)
    new = _new_messages(tmp_path, _arm(case), _RESPONSES[case["response"]], items)
    assert len(old) == len(new) == len(items)
    assert old == new


def test_no_memory_messages_identical(tmp_path) -> None:
    items = _items()
    old = _old_messages("no_memory", None, items)
    new = _new_messages(tmp_path, ArmConfig(name="no_memory", context="none"), None, items)
    assert old == new
    assert "(no relevant memory found)" in new[0][1]["content"]


def test_resume_judge_and_report(tmp_path) -> None:
    items = _items()
    arm = ArmConfig(name="no_memory", context="none")
    rd = RunDir(tmp_path, "r")
    asyncio.run(run_search(rd, arm, items, _ADAPTER, None))
    asyncio.run(run_search(rd, arm, items, _ADAPTER, None))
    assert len(rd.stage("search").records()) == 3
    sent: list = []
    kw = dict(chat_client=None, chat_base_url="u", api_key="k", model="m")
    run_answer(rd, arm, _ADAPTER, send_fn=_capture(sent), **kw)
    assert len(sent) == 3
    run_answer(rd, arm, _ADAPTER, send_fn=_capture(sent), **kw)
    assert len(sent) == 3
    ans = rd.stage("answers").records()[0]
    assert len(ans["messages_sha256"]) == 64 and ans["response_text"] == "ans"

    def score(item, text):
        return item["question_id"] != "lme-sample-003"

    score.last_usage = {"input_tokens": 2}
    by_id = {it["question_id"]: it for it in items}
    run_judge(rd, by_id, score_fn=score)
    run_judge(rd, by_id, score_fn=score)
    judged = rd.stage("judged").records()
    assert len(judged) == 3 and judged[0]["judge_usage"] == {"input_tokens": 2}
    report = build_report(rd)
    assert report["n"] == 3 and report["accuracy"] == pytest.approx(2 / 3)
    assert sum(v["n"] for v in report["per_type"].values()) == 3
    assert (rd.path / "report.json").exists() and (rd.path / "report.md").exists()
