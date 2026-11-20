"""Staged run directory: resumable JSONL stages and config pinning."""

from __future__ import annotations

import pytest

from archolith_bench.staged.rundir import RunConfigMismatch, RunDir


def test_append_done_ids_records(tmp_path) -> None:
    st = RunDir(tmp_path, "r").stage("search")
    assert st.done_ids() == set() and st.records() == []
    st.append({"item_id": "a", "t": "é"})
    st.append({"item_id": "b"})
    assert st.done_ids() == {"a", "b"}
    assert st.records()[0]["t"] == "é"


def test_truncated_last_line_ignored(tmp_path) -> None:
    st = RunDir(tmp_path, "r").stage("answers")
    st.append({"item_id": "a"})
    with open(st.path, "a", encoding="utf-8") as f:
        f.write('{"item_id": "b", "x"')
    assert st.done_ids() == {"a"}
    assert len(st.records()) == 1
    st.append({"item_id": "c"})
    assert st.done_ids() == {"a", "c"}


def test_config_mismatch_and_noop(tmp_path) -> None:
    rd = RunDir(tmp_path, "r")
    rd.write_config({"a": 1})
    rd.write_config({"a": 1})
    with pytest.raises(RunConfigMismatch):
        rd.write_config({"a": 2})


def test_unknown_stage_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        RunDir(tmp_path, "r").stage("nope")
