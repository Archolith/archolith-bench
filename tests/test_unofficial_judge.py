"""Tests for the unofficial (substance) LLM judge layered after the official LME judge."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from archolith_bench.harness import MemoryCheckpoint, run_memory_ab
from archolith_bench.harness.base import (
    ABResult,
    ArmResult,
    TaskResult,
    ab_result_to_dict,
    write_harness_evidence,
)
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter
from archolith_bench.harness.scoring import (
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_ABSTENTION,
    _UNOFFICIAL_JUDGE_SYSTEM,
    LLMJudgeScorer,
    UnofficialJudgeScorer,
    _judge_messages,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = "2024/05/01 (Wed) 10:00"


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score_extract = _load_script(
    "unofficial_score_extract_test", "scripts/longmemeval/lib/score_extract.py"
)


# ---------------------------------------------------------------- (a) scorer

def _scorer(reply: str, seen: list):
    def send_fn(client, base_url, api_key, messages, model, **kwargs):  # noqa: ANN001
        seen.append((messages, kwargs))
        return reply, 1.0, {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}

    return UnofficialJudgeScorer(base_url="http://x", api_key="k", model="m", send_fn=send_fn)


def test_unofficial_scorer_prompt_layout_strict_and_usage() -> None:
    seen: list = []
    s = _scorer("Yes.", seen)
    item = {"question": "How many?", "answer": "3", "question_type": "multi-session"}
    assert s(item, "At least 3.") is True
    messages, kwargs = seen[0]
    assert kwargs.get("strict") is True
    assert messages[0]["content"] == _UNOFFICIAL_JUDGE_SYSTEM
    assert messages[0]["content"] != _JUDGE_SYSTEM
    assert messages[1] == _judge_messages(item, "At least 3.")[1]
    assert "Question: How many?" in messages[1]["content"]
    assert "Gold answer: 3" in messages[1]["content"]
    assert "Model response: At least 3." in messages[1]["content"]
    assert s.last_usage["total_tokens"] == 6
    s.close()


def test_unofficial_scorer_no_verdict() -> None:
    assert _scorer("no", [])({"question": "q", "answer": "3"}, "4") is False


def test_unofficial_scorer_abstention_reuses_official_messages() -> None:
    seen: list = []
    s = _scorer("yes", seen)
    item = {"question": "q?", "answer": "", "question_type": "abstention"}
    s(item, "I do not know")
    assert seen[0][0] == _judge_messages(item, "I do not know")
    assert seen[0][0][0]["content"] == _JUDGE_SYSTEM_ABSTENTION


def test_official_scorer_unchanged_by_unofficial() -> None:
    assert LLMJudgeScorer.__call__ is not UnofficialJudgeScorer.__call__
    msgs = _judge_messages({"question": "q", "answer": "a"}, "r")
    assert msgs[0]["content"] == _JUDGE_SYSTEM


# ---------------------------------------------------------------- (b)(c) run loop

class _Client:
    def new_group(self) -> str:
        return "ns-1"

    def record_turn_evidence(self, namespace, text, **kwargs):  # noqa: ANN001
        return {"turn_id": "t"}

    def ingest(self, group_id, role, content, **kwargs):  # noqa: ANN001
        pass

    def recall(self, group_id, query, limit=10):  # noqa: ANN001
        return ["fact"]

    def reset(self, group_id):  # noqa: ANN001
        pass


_ITEMS = [
    {"question_id": "good", "question": "q good", "answer": "1", "question_date": DATE,
     "haystack_sessions": [[{"role": "user", "content": "x"}]]},
    {"question_id": "bad", "question": "q bad", "answer": "2", "question_date": DATE,
     "haystack_sessions": [[{"role": "user", "content": "x"}]]},
]


class _LmeAdapter(LongMemEvalMemoryAdapter):
    def load_items(self, subset=None, limit=None, fixture_path=None):  # noqa: ANN001
        return list(_ITEMS)


class _OtherAdapter(_LmeAdapter):
    benchmark_id = "other-memory"


def _send_fn(chat_client, base_url, api_key, messages, model, **kw):  # noqa: ANN001
    return "ans", 1.0, {"prompt_tokens": 3, "completion_tokens": 1}


class _Official:
    last_usage: dict = {}

    def __call__(self, item, text):
        self.last_usage = {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}
        return item["question_id"] == "good"


class _Unofficial:
    def __init__(self, verdict=True):
        self.calls: list[str] = []
        self.verdict = verdict
        self.last_usage: dict = {}
        self.closed = False

    def __call__(self, item, text):
        self.calls.append(item["question_id"])
        self.last_usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        return self.verdict

    def close(self):
        self.closed = True


def _run(adapter, **kw):
    return run_memory_ab(
        adapter, arms=("menhir_recall",), client=_Client(), send_fn=_send_fn,
        reset_memory=False, score_fn=_Official(), **kw,
    )


def test_run_loop_calls_unofficial_only_on_official_wrong() -> None:
    un = _Unofficial(verdict=True)
    base = _run(_LmeAdapter())
    ab = _run(_LmeAdapter(), unofficial_score_fn=un)
    arm = ab.arms["menhir_recall"]
    by_id = {r.task_id: r for r in arm.results}
    assert un.calls == ["bad"]
    assert un.closed
    assert by_id["good"].unofficial_correct is True
    assert by_id["good"].unofficial_scorer_raw_usage == {}
    assert by_id["bad"].unofficial_correct is True
    assert by_id["bad"].unofficial_scorer_raw_usage["total_tokens"] == 5
    assert by_id["bad"].correct is False
    assert arm.unofficial_n == 2 and arm.unofficial_score == 1.0
    base_arm = base.arms["menhir_recall"]
    assert arm.score == base_arm.score == 0.5
    assert [r.correct for r in arm.results] == [r.correct for r in base_arm.results]
    assert base_arm.unofficial_n == 0 and base_arm.unofficial_score is None


def test_run_loop_unofficial_negative_verdict() -> None:
    ab = _run(_LmeAdapter(), unofficial_score_fn=_Unofficial(verdict=False))
    arm = ab.arms["menhir_recall"]
    assert arm.unofficial_score == 0.5 and arm.unofficial_n == 2
    assert arm.score == 0.5


def test_unofficial_failure_leaves_item_unjudged_and_official_recorded(tmp_path) -> None:
    class _Failing(_Unofficial):
        def __call__(self, item, text):
            raise RuntimeError("judge down")

    ck = MemoryCheckpoint(tmp_path / "ck.jsonl")
    ab = _run(_LmeAdapter(), unofficial_score_fn=_Failing(), checkpoint=ck)
    arm = ab.arms["menhir_recall"]
    by_id = {r.task_id: r for r in arm.results}
    assert by_id["bad"].correct is False and by_id["bad"].unofficial_correct is None
    assert by_id["good"].unofficial_correct is True
    assert arm.score == 0.5 and arm.unofficial_n == 1
    assert MemoryCheckpoint(tmp_path / "ck.jsonl").done_count() == 2


def test_non_lme_adapter_never_calls_unofficial() -> None:
    un = _Unofficial()
    ab = _run(_OtherAdapter(), unofficial_score_fn=un)
    arm = ab.arms["menhir_recall"]
    assert un.calls == []
    assert all(r.unofficial_correct is None and r.unofficial_scorer_raw_usage == {} for r in arm.results)
    assert arm.unofficial_n == 0 and arm.unofficial_score is None
    arm_dict = ab_result_to_dict(ab)["arms"]["menhir_recall"]
    assert "unofficial_score" not in arm_dict and "unofficial_n" not in arm_dict


# ---------------------------------------------------------------- (d) checkpoint

def test_checkpoint_roundtrip_preserves_unofficial_fields(tmp_path) -> None:
    path = tmp_path / "ck.jsonl"
    un = _Unofficial(verdict=False)
    first = _run(_LmeAdapter(), unofficial_score_fn=un, checkpoint=MemoryCheckpoint(path))
    un2 = _Unofficial()
    second = _run(_LmeAdapter(), unofficial_score_fn=un2, checkpoint=MemoryCheckpoint(path))
    assert un2.calls == []  # resumed: no backfill, no re-judge
    a, b = first.arms["menhir_recall"], second.arms["menhir_recall"]
    assert [r.unofficial_correct for r in b.results] == [True, False]
    assert b.results[1].unofficial_scorer_raw_usage["total_tokens"] == 5
    assert b.unofficial_score == a.unofficial_score == 0.5
    assert b.unofficial_n == a.unofficial_n == 2


def test_checkpoint_old_row_loads_with_none(tmp_path) -> None:
    path = tmp_path / "ck.jsonl"
    row = {"arm": "menhir_recall", "task_id": "t1", "ts": 0, "result": {
        "task_id": "t1", "response_text": "x", "input_tokens": 1, "output_tokens": 1,
        "latency_ms": 1.0, "correct": False, "raw_usage": {}}}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tr = MemoryCheckpoint(path).get("menhir_recall", "t1")
    assert tr is not None
    assert tr.unofficial_correct is None
    assert tr.unofficial_scorer_raw_usage == {}


# ---------------------------------------------------------------- (e) serialization

def _ab(unofficial_n: int, unofficial_score: float | None) -> ABResult:
    tr = TaskResult("t", "r", 1, 1, 1.0, False)
    arm = ArmResult("a", 1, 0.0, 1, 1, 0.0, [tr],
                    unofficial_score=unofficial_score, unofficial_n=unofficial_n)
    return ABResult("longmemeval-menhir", "LME", "m", None, {"a": arm}, {})


def test_ab_result_to_dict_keys_only_when_judged() -> None:
    plain = ab_result_to_dict(_ab(0, None))["arms"]["a"]
    assert "unofficial_score" not in plain and "unofficial_n" not in plain
    judged = ab_result_to_dict(_ab(1, 0.5))["arms"]["a"]
    assert judged["unofficial_score"] == 0.5 and judged["unofficial_n"] == 1


def test_evidence_section_only_when_judged(tmp_path) -> None:
    p0 = write_harness_evidence(_ab(0, None), tmp_path / "a.md")
    p1 = write_harness_evidence(_ab(1, 0.5), tmp_path / "b.md")
    plain, judged = p0.read_text(encoding="utf-8"), p1.read_text(encoding="utf-8")
    assert "Unofficial" not in plain
    assert judged.startswith(plain)
    tail = judged[len(plain):]
    assert "## Unofficial judge (substance; not the LongMemEval score)" in tail
    assert "| Arm | N judged | Unofficial score |" in tail
    assert "| a | 1 | 0.500 |" in tail


# ---------------------------------------------------------------- (f) score_extract

def _row(arm, task, correct, **extra):
    return {"arm": arm, "task_id": task, "result": {"task_id": task, "correct": correct, **extra}}


def _write_ck(run: Path, rows: list[dict]) -> None:
    d = run / "harness_recall"
    d.mkdir(parents=True)
    (d / ".checkpoint_x.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_score_extract_unofficial_present_and_absent(tmp_path) -> None:
    run = tmp_path / "r1"
    _write_ck(run, [
        _row("a", "t1", False, unofficial_correct=True),
        _row("a", "t2", False, unofficial_correct=False),
        _row("a", "t3", True, unofficial_correct=True),
        _row("b", "t1", True),
    ])
    payload = score_extract.extract(run)
    a, b = payload["arms"]["a"], payload["arms"]["b"]
    assert (a["unofficial_n"], a["unofficial_correct"]) == (3, 2)
    assert a["unofficial_score"] == round(2 / 3, 6)
    assert "unofficial_n" not in b and "unofficial_score" not in b
    report = score_extract.format_report(payload).splitlines()
    a_line = next(x for x in report if x.strip().startswith("a "))
    b_line = next(x for x in report if x.strip().startswith("b "))
    assert a_line.endswith("  unofficial=0.667 (n=3)")
    assert "unofficial" not in b_line


def test_score_extract_unofficial_last_wins(tmp_path) -> None:
    run = tmp_path / "r2"
    _write_ck(run, [
        _row("a", "t1", False, unofficial_correct=False),
        _row("a", "t1", False, unofficial_correct=True),
    ])
    a = score_extract.extract(run)["arms"]["a"]
    assert a["n"] == 1 and a["unofficial_n"] == 1 and a["unofficial_correct"] == 1


# ---------------------------------------------------------------- (g) usage summary

def test_usage_judge_unofficial_is_separate_operation(tmp_path) -> None:
    from scripts.longmemeval.lib.summarize_llm_usage import summarize_harness_usage

    u = lambda n: {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}  # noqa: E731
    rows = [
        {"arm": "a", "result": {"raw_usage": u(10), "scorer_raw_usage": u(20),
                                "unofficial_scorer_raw_usage": u(30)}},
        {"arm": "a", "result": {"raw_usage": u(10), "scorer_raw_usage": u(20)}},
    ]
    p = tmp_path / "ck.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ops = {o["operation"]: o for o in summarize_harness_usage(p)["by_operation"]}
    assert ops["judge"]["calls"] == 2 and ops["judge"]["input_tokens"] == 40
    assert ops["judge_unofficial"]["calls"] == 1 and ops["judge_unofficial"]["input_tokens"] == 30
    assert ops["answer"]["calls"] == 2
    # require_judge_usage never demands unofficial usage
    ops2 = {o["operation"]: o for o in
            summarize_harness_usage(p, require_judge_usage=True)["by_operation"]}
    assert ops2["judge_unofficial"]["calls"] == 1


# ---------------------------------------------------------------- (h) CLI wiring

def _cli_run(monkeypatch, tmp_path, **overrides):
    from archolith_bench import cli as cli_module

    class FakeClient:
        def __init__(self, url, *, api_key="", **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    captured: dict = {}

    class MockABResult:
        model = "test"
        arms = {}
        deltas = {}

    def fake_run(*a, **kw):
        captured.update(kw)
        return MockABResult()

    monkeypatch.setattr("archolith_bench.harness.HttpMenhirClient", FakeClient)
    monkeypatch.setattr("archolith_bench.harness.assert_not_production", lambda url: None)
    monkeypatch.setattr("archolith_bench.harness.run_memory_ab", fake_run)
    monkeypatch.setattr("archolith_bench.harness.write_harness_evidence", lambda *a, **kw: None)
    monkeypatch.setattr("archolith_bench.harness.ab_result_to_dict", lambda *a, **kw: {})
    monkeypatch.setattr("archolith_bench.cli._publish_cli_evidence", lambda *a, **kw: None)
    args = dict(
        benchmark_id="longmemeval-menhir", list_adapters=False, arms="menhir_recall",
        offline_fixture=None, menhir_url="http://localhost:8099", model="test-model",
        recall_limit=10, confirm_menhir_reset=True, dry_run_menhir_reset=False,
        recall_only=False, namespace_template="lme-{question_id}", format="json",
        output_dir=tmp_path, out=None, publish_evidence=None, public_copy=False,
        command_text="test", scorer="llm-judge", judge_api_key="jk",
        judge_model="judge-m", judge_url="http://judge", resume=False, subset=None, limit=None,
    )
    args.update(overrides)
    cli_module._run_harness(argparse.Namespace(**args))
    return captured


def test_cli_passes_unofficial_scorer_for_lme_llm_judge(monkeypatch, tmp_path) -> None:
    kw = _cli_run(monkeypatch, tmp_path)
    un = kw["unofficial_score_fn"]
    assert isinstance(un, UnofficialJudgeScorer)
    assert isinstance(kw["score_fn"], LLMJudgeScorer)
    assert (un.base_url, un.api_key, un.model) == ("http://judge", "jk", "judge-m")


def test_cli_opt_out_and_containment_pass_none(monkeypatch, tmp_path) -> None:
    assert _cli_run(monkeypatch, tmp_path, no_unofficial_judge=True)["unofficial_score_fn"] is None
    assert _cli_run(monkeypatch, tmp_path, scorer="containment")["unofficial_score_fn"] is None
