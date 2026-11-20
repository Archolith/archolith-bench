"""Prompt files, rerun-from-stage (derive_run), unofficial judge and run comparison."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from archolith_bench import prompts
from archolith_bench.harness import longmemeval, scoring
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter
from archolith_bench.staged.__main__ import main as staged_main
from archolith_bench.staged.config import ArmConfig, arm_from_resolved, resolve, run_id
from archolith_bench.staged.ops import compare_runs, derive_run, format_compare, load_resolved
from archolith_bench.staged.pipeline import (
    accuracy_summary,
    build_report,
    run_answer,
    run_judge,
    run_search,
)
from archolith_bench.staged.rundir import RunDir

_ROOT = Path(__file__).resolve().parent
_ITEMS = json.loads((_ROOT.parent / "fixtures" / "longmemeval_sample.json").read_text(encoding="utf-8"))
_ADAPTER = LongMemEvalMemoryAdapter()
_ARM = ArmConfig(name="menhir_ctx", context="menhir")


class _FakeSystem:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, ns, query, **kw):
        self.calls += 1
        return {}

    async def context(self, ns, query, **kw):
        self.calls += 1
        return {"context": f"ctx for {ns}"}


def _capture(store: list):
    def send(client, base_url, api_key, messages, model, strict=False):  # noqa: ANN001
        store.append(messages)
        return f"ans{len(store)}", 1.0, {"input_tokens": 1, "output_tokens": 1}

    return send


def _items() -> list[dict]:
    return [dict(it) for it in _ITEMS[:3]]


def _by_id(items: list[dict]) -> dict[str, dict]:
    return {it["question_id"]: it for it in items}


def _resolve(prompt_files, *, model="m1", judge=None, fingerprint="fp1") -> dict:
    return resolve(
        _ARM, benchmark_id="lme", fixture_path="f.json", fixture_sha256="0" * 64,
        prompt_fingerprint=fingerprint, renderer_sha256="1" * 64, menhir_commit=None,
        namespace_template="lme-{question_id}", model=model,
        judge=judge or {"model": "j1"}, prompt_files=prompt_files,
    )


def _real_prompt_files() -> dict[str, str]:
    return prompts.hashes(prompts.LONGMEMEVAL_RUN_PROMPTS)


def _disk_text(name: str) -> str:
    return (prompts.PROMPT_DIR / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


# 1. prompt files vs module constants -------------------------------------------------------------


def test_answer_prompt_constant_matches_file() -> None:
    assert longmemeval._MEMORY_SYSTEM_PROMPT == _disk_text(prompts.ANSWER_LONGMEMEVAL)
    assert longmemeval._MEMORY_SYSTEM_PROMPT == prompts.load(prompts.ANSWER_LONGMEMEVAL)


@pytest.mark.parametrize("qtype", prompts.LME_QUESTION_TYPES)
def test_lme_judge_templates_match_files(qtype: str) -> None:
    assert scoring._LME_JUDGE_TEMPLATES[qtype] == _disk_text(prompts.lme_judge(qtype))
    assert scoring._LME_JUDGE_TEMPLATES[qtype] == prompts.load(prompts.lme_judge(qtype))


def test_other_judge_constants_match_files() -> None:
    assert scoring._UNOFFICIAL_JUDGE_SYSTEM == _disk_text(prompts.JUDGE_UNOFFICIAL)
    assert scoring._JUDGE_SYSTEM == _disk_text(prompts.JUDGE_GENERIC)
    assert scoring._JUDGE_SYSTEM_ABSTENTION == _disk_text(prompts.JUDGE_GENERIC_ABSTENTION)
    assert scoring._LME_ABSTENTION_TEMPLATE == _disk_text(prompts.JUDGE_LME_ABSTENTION)


def test_load_normalises_crlf(tmp_path, monkeypatch) -> None:
    (tmp_path / "p.txt").write_bytes(b"line1\r\nline2\r\n")
    (tmp_path / "q.txt").write_bytes(b"line1\nline2\n")
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    prompts.load.cache_clear()
    try:
        assert prompts.load("p.txt") == "line1\nline2\n"
        assert prompts.sha256("p.txt") == prompts.sha256("q.txt")
    finally:
        prompts.load.cache_clear()


def test_hashes_sorted_and_hex() -> None:
    h = prompts.hashes(list(reversed(prompts.LONGMEMEVAL_RUN_PROMPTS)))
    assert list(h) == sorted(prompts.LONGMEMEVAL_RUN_PROMPTS)
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in h.values())


# 2. resolve ---------------------------------------------------------------------------------------


def test_resolve_records_prompt_files_and_run_id_tracks_them() -> None:
    pf = _real_prompt_files()
    resolved = _resolve(pf)
    assert resolved["prompt_files"] == pf
    changed = dict(pf, **{prompts.JUDGE_GENERIC: "f" * 64})
    assert run_id(_resolve(changed)) != run_id(resolved)
    assert run_id(_resolve(dict(pf))) == run_id(resolved)


# helpers to build a source run -------------------------------------------------------------------


def _wrong_id(items) -> str:
    return items[1]["question_id"]


def _build_source(tmp_path):
    items = _items()
    pf = _real_prompt_files()
    resolved = _resolve(pf)
    src = RunDir(tmp_path, run_id(resolved))
    src.write_config(resolved)
    system = _FakeSystem()
    sent: list = []
    asyncio.run(run_search(src, _ARM, items, _ADAPTER, system))
    run_answer(src, _ARM, _ADAPTER, send_fn=_capture(sent), chat_client=None,
               chat_base_url="u", api_key="k", model="m1")
    wrong = _wrong_id(items)

    def score(item, text):
        return item["question_id"] != wrong

    run_judge(src, _by_id(items), score_fn=score)
    return src, items, pf, system, sent, score


# 3. acceptance: judge-only rerun ---------------------------------------------------------------


def test_judge_rerun_reproduces_labels_without_search_or_answer(tmp_path) -> None:
    src, items, pf, system, sent, score = _build_source(tmp_path)
    assert system.calls == 3 and len(sent) == 3

    rd = derive_run(src.path, tmp_path, "judge", prompt_files=pf, judge={"model": "j2"}, tag="t")
    assert rd.run_id != src.run_id
    assert load_resolved(rd.path)["derived_from"] == {"run_id": src.run_id, "from": "judge", "tag": "t"}
    for name in ("search", "answers"):
        assert (rd.path / f"{name}.jsonl").read_bytes() == (src.path / f"{name}.jsonl").read_bytes()
    assert not (rd.path / "judged.jsonl").exists()

    run_judge(rd, _by_id(items), score_fn=score)
    old = {r["item_id"]: r["correct"] for r in src.stage("judged").records()}
    new = {r["item_id"]: r["correct"] for r in rd.stage("judged").records()}
    assert new == old and len(new) == 3 and False in new.values()
    assert system.calls == 3 and len(sent) == 3


def test_derived_from_has_no_tag_key_when_untagged(tmp_path) -> None:
    src, _, pf, *_ = _build_source(tmp_path)
    rd = derive_run(src.path, tmp_path, "judge", prompt_files=pf, judge={"model": "j2"})
    assert load_resolved(rd.path)["derived_from"] == {"run_id": src.run_id, "from": "judge"}


# 4. prompt hash handling per stage ------------------------------------------------------------


def test_judge_rerun_keeps_answer_hash_refreshes_judge_hashes(tmp_path) -> None:
    src, _, pf, *_ = _build_source(tmp_path)
    new_pf = dict(pf)
    new_pf[prompts.ANSWER_LONGMEMEVAL] = "a" * 64
    new_pf[prompts.JUDGE_GENERIC] = "b" * 64
    rd = derive_run(src.path, tmp_path, "judge", prompt_files=new_pf, judge={"model": "j2"},
                    model="ignored", answer_prompt_fingerprint="ignored")
    got = load_resolved(rd.path)
    assert got["prompt_files"][prompts.ANSWER_LONGMEMEVAL] == pf[prompts.ANSWER_LONGMEMEVAL]
    assert got["prompt_files"][prompts.JUDGE_GENERIC] == "b" * 64
    assert got["judge"] == {"model": "j2"}
    # answer-stage settings are the source's when answers are reused
    assert got["model"] == "m1" and got["answer_prompt_fingerprint"] == "fp1"


def test_answer_rerun_takes_all_new_hashes_and_copies_only_search(tmp_path) -> None:
    src, _, pf, *_ = _build_source(tmp_path)
    new_pf = dict(pf)
    new_pf[prompts.ANSWER_LONGMEMEVAL] = "a" * 64
    new_pf[prompts.JUDGE_GENERIC] = "b" * 64
    rd = derive_run(src.path, tmp_path, "answer", prompt_files=new_pf, judge={"model": "j2"},
                    model="m2", answer_prompt_fingerprint="fp2")
    got = load_resolved(rd.path)
    assert got["prompt_files"] == new_pf
    assert got["model"] == "m2" and got["answer_prompt_fingerprint"] == "fp2"
    assert got["derived_from"] == {"run_id": src.run_id, "from": "answer"}
    assert (rd.path / "search.jsonl").read_bytes() == (src.path / "search.jsonl").read_bytes()
    assert not (rd.path / "answers.jsonl").exists()
    assert arm_from_resolved(got) == _ARM


# 5. derive_run details ------------------------------------------------------------------------


def test_derive_is_idempotent_and_tag_changes_id(tmp_path) -> None:
    src, _, pf, *_ = _build_source(tmp_path)
    kw = dict(prompt_files=pf, judge={"model": "j2"})
    a = derive_run(src.path, tmp_path, "judge", **kw)
    a_answers = (a.path / "answers.jsonl").read_bytes()
    b = derive_run(src.path, tmp_path, "judge", **kw)
    assert (a.run_id, a.path) == (b.run_id, b.path)
    assert (b.path / "answers.jsonl").read_bytes() == a_answers
    c = derive_run(src.path, tmp_path, "judge", tag="x", **kw)
    assert c.run_id != a.run_id


def test_derive_invalid_stage_and_missing_answers(tmp_path) -> None:
    src, _, pf, *_ = _build_source(tmp_path)
    with pytest.raises(ValueError):
        derive_run(src.path, tmp_path, "search", prompt_files=pf, judge={})

    bare = RunDir(tmp_path, "bare")
    bare.write_config(_resolve(pf, model="other"))
    (bare.path / "search.jsonl").write_bytes((src.path / "search.jsonl").read_bytes())
    with pytest.raises(FileNotFoundError):
        derive_run(bare.path, tmp_path, "judge", prompt_files=pf, judge={})


# 6. unofficial judge --------------------------------------------------------------------------


def _answered_run(tmp_path):
    items = _items()
    arm = ArmConfig(name="no_memory", context="none")
    rd = RunDir(tmp_path, "r")
    asyncio.run(run_search(rd, arm, items, _ADAPTER, None))
    run_answer(rd, arm, _ADAPTER, send_fn=_capture([]), chat_client=None,
               chat_base_url="u", api_key="k", model="m")
    return rd, items


def test_unofficial_judge_only_runs_on_official_wrong_and_survives_failure(tmp_path) -> None:
    rd, items = _answered_run(tmp_path)
    right, rescued, broken = (it["question_id"] for it in items)
    seen: list[str] = []

    def score(item, text):
        return item["question_id"] == right

    def unofficial(item, text):
        seen.append(item["question_id"])
        if item["question_id"] == broken:
            raise RuntimeError("judge down")
        unofficial.last_usage = {"input_tokens": 7}
        return True

    run_judge(rd, _by_id(items), score_fn=score, unofficial_fn=unofficial)
    assert seen == [rescued, broken]  # never called for the official-correct item
    rows = {r["item_id"]: r for r in rd.stage("judged").records()}
    assert rows[right]["correct"] is True and rows[right]["unofficial_correct"] is True
    assert rows[right]["unofficial_usage"] == {}
    assert rows[rescued]["correct"] is False and rows[rescued]["unofficial_correct"] is True
    assert rows[rescued]["unofficial_usage"] == {"input_tokens": 7}
    assert rows[broken]["correct"] is False and rows[broken]["unofficial_correct"] is None

    summary = accuracy_summary(list(rows.values()))
    assert summary["unofficial"] == {"n": 2, "accuracy": 1.0}  # the failed item is excluded
    assert summary["accuracy"] == pytest.approx(1 / 3)
    report = build_report(rd)
    assert report["unofficial"] == {"n": 2, "accuracy": 1.0}
    assert "unofficial (substance) judge: n=2 accuracy=1.0" in (rd.path / "report.md").read_text(encoding="utf-8")


def test_no_unofficial_fn_means_no_unofficial_keys(tmp_path) -> None:
    rd, items = _answered_run(tmp_path)
    run_judge(rd, _by_id(items), score_fn=lambda item, text: False)
    rows = rd.stage("judged").records()
    assert len(rows) == 3 and all("unofficial_correct" not in r for r in rows)
    assert "unofficial" not in build_report(rd)
    assert "unofficial" not in (rd.path / "report.md").read_text(encoding="utf-8")


# 7/8. compare ---------------------------------------------------------------------------------


def _judged_run(root: Path, name: str, labels: dict[str, bool], types: dict[str, str]) -> Path:
    rd = RunDir(root, name)
    rd.path.mkdir(parents=True)
    for iid, ok in labels.items():
        rd.stage("judged").append({"item_id": iid, "correct": ok, "question_type": types[iid]})
    return rd.path


_TYPES = {"i1": "t1", "i2": "t1", "i3": "t2", "i4": "t2", "i5": "t2"}


def _compare_fixture(tmp_path):
    # common = i2, i3, i4; i1 only in A, i5 only in B (i1 right / i5 wrong must not count)
    a = _judged_run(tmp_path, "runA", {"i1": True, "i2": True, "i3": False, "i4": True}, _TYPES)
    b = _judged_run(tmp_path, "runB", {"i2": False, "i3": True, "i4": True, "i5": False}, _TYPES)
    return a, b


def test_compare_runs_common_items_and_flips(tmp_path) -> None:
    a, b = _compare_fixture(tmp_path)
    c = compare_runs(a, b)
    assert (c["a"], c["b"]) == ("runA", "runB")
    assert c["n_common"] == 3
    assert c["only_in_a"] == ["i1"] and c["only_in_b"] == ["i5"]
    assert c["summary_a"]["n"] == c["summary_b"]["n"] == 3
    assert c["summary_a"]["accuracy"] == pytest.approx(2 / 3)
    assert c["summary_b"]["accuracy"] == pytest.approx(2 / 3)
    assert c["summary_a"]["per_type"]["t1"] == {"n": 1, "accuracy": 1.0}
    assert c["summary_b"]["per_type"]["t1"] == {"n": 1, "accuracy": 0.0}
    assert c["a_correct_b_wrong"] == ["i2"]
    assert c["b_correct_a_wrong"] == ["i3"]

    text = format_compare(c)
    assert "runA" in text and "runB" in text
    assert "i2" in text.split("A right, B wrong")[1].splitlines()[0]
    assert "i3" in text.split("B right, A wrong")[1]


def test_cli_compare_writes_json(tmp_path, capsys) -> None:
    a, b = _compare_fixture(tmp_path)
    out = tmp_path / "cmp.json"
    assert staged_main(["compare", str(a), str(b), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(compare_runs(a, b)))
    assert "runA" in capsys.readouterr().out
