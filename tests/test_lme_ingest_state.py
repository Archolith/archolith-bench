"""Tests for the append-only resumable-ingest state log.

Each case is a way a resume could redo or contradict work: retrying twice, resetting a
namespace unprompted, accepting a changed turn plan, or choking on a crash-torn log.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


state = _load_script("lme_ingest_state_test", "scripts/longmemeval/lib/ingest_state.py")


def _rec(rtype: str, namespace: str = "ns1", **fields) -> str:
    return json.dumps({"v": 1, "type": rtype, "namespace": namespace, **fields})


def _write(path: Path, *lines: str, trailing_newline: bool = True) -> None:
    text = "\n".join(lines) + ("\n" if trailing_newline else "")
    path.write_bytes(text.encode("utf-8"))


def _plan(digest: str = "d1", segments: int = 3, namespace: str = "ns1") -> str:
    return _rec("plan", namespace, digest=digest, segments=segments)


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "ingest-state-m.jsonl"


def test_round_trip_survives_reload(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 4)
    log.record_retry("ns1", "ep-1", "episode")
    log.record_retry("ns1", "pj-1", "projection")
    log.record_plan("ns2", "d2", 2)
    log.record_done("ns1", "q1")

    reloaded = state.IngestStateLog(log_path)
    p1 = reloaded.progress("ns1")
    assert p1.plan_digest == "d1"
    assert p1.segments == 4
    assert p1.retried_episodes == {"ep-1"}
    assert p1.retried_projections == {"pj-1"}
    assert p1.done is True
    p2 = reloaded.progress("ns2")
    assert (p2.plan_digest, p2.done, p2.retried_episodes) == ("d2", False, set())
    assert reloaded.progress("never-seen") == state.NamespaceProgress()


def test_progress_is_a_copy(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 1)
    log.record_retry("ns1", "ep-1", "episode")
    snapshot = log.progress("ns1")
    snapshot.retried_episodes.add("bogus")
    snapshot.done = True
    snapshot.plan_digest = "other"
    fresh = log.progress("ns1")
    assert fresh.retried_episodes == {"ep-1"}
    assert fresh.done is False
    assert fresh.plan_digest == "d1"


def test_retry_recorded_before_crash_is_still_retried_after_reload(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 1)
    log.record_retry("ns1", "ep-9", "episode")
    del log  # "crash": no done record, new process
    after = state.IngestStateLog(log_path).progress("ns1")
    assert "ep-9" in after.retried_episodes
    assert after.done is False


def test_second_plan_with_different_digest_rejected_and_nothing_written(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 2)
    before = log_path.read_bytes()
    with pytest.raises(state.IngestStateError):
        log.record_plan("ns1", "d2", 2)
    assert log_path.read_bytes() == before
    assert log.progress("ns1").plan_digest == "d1"
    # same digest again is idempotent, not an error
    log.record_plan("ns1", "d1", 2)


def test_hand_written_contradiction_names_line(log_path):
    _write(log_path, _plan("d1"), _rec("retry", episode_uuid="e", kind="episode"), _plan("d2"))
    with pytest.raises(state.IngestStateError) as exc:
        state.IngestStateLog(log_path)
    assert ":3" in str(exc.value)


def test_reset_clears_everything_and_new_digest_accepted(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 2)
    log.record_retry("ns1", "ep-1", "episode")
    log.record_retry("ns1", "pj-1", "projection")
    log.record_done("ns1", "q1")
    log.record_reset("ns1")
    assert log.progress("ns1") == state.NamespaceProgress()
    log.record_plan("ns1", "d2", 5)

    reloaded = state.IngestStateLog(log_path).progress("ns1")
    assert reloaded.plan_digest == "d2"
    assert reloaded.segments == 5
    assert reloaded.retried_episodes == set()
    assert reloaded.retried_projections == set()
    assert reloaded.done is False


def test_reset_only_affects_its_namespace(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("a", "da", 1)
    log.record_plan("b", "db", 1)
    log.record_reset("a")
    assert log.progress("b").plan_digest == "db"


def test_torn_final_line_ignored_then_truncated_on_append(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 2)
    log.record_retry("ns1", "ep-1", "episode")
    fragment = b'{"v": 1, "type": "retry", "namespace": "ns1", "episode_u'
    with open(log_path, "ab") as handle:
        handle.write(fragment)

    torn = state.IngestStateLog(log_path)  # does not raise
    assert torn.progress("ns1").retried_episodes == {"ep-1"}
    assert log_path.read_bytes().endswith(fragment)  # load alone does not modify

    torn.record_retry("ns1", "ep-2", "episode")
    data = log_path.read_bytes()
    assert fragment not in data
    assert data.endswith(b"\n")
    assert len(data.splitlines()) == 3
    reloaded = state.IngestStateLog(log_path).progress("ns1")
    assert reloaded.retried_episodes == {"ep-1", "ep-2"}
    assert reloaded.plan_digest == "d1"


def test_unparsable_terminated_middle_line_is_error(log_path):
    _write(log_path, _plan("d1"), "{not json", _rec("done", question_id="q"))
    with pytest.raises(state.IngestStateError) as exc:
        state.IngestStateLog(log_path)
    assert ":2" in str(exc.value)


@pytest.mark.parametrize(
    "line",
    [
        json.dumps({"v": 2, "type": "done", "namespace": "ns1"}),
        json.dumps({"type": "done", "namespace": "ns1"}),
        json.dumps({"v": 1, "type": "bogus", "namespace": "ns1"}),
        json.dumps({"v": 1, "type": "done"}),
        json.dumps({"v": 1, "type": "done", "namespace": ""}),
        "[1, 2]",
    ],
)
def test_bad_records_rejected_on_load(log_path, line):
    _write(log_path, _plan("d1"), line, _rec("done"))
    with pytest.raises(state.IngestStateError):
        state.IngestStateLog(log_path)


def test_unterminated_final_line_that_is_valid_json_is_kept(log_path):
    _write(log_path, _plan("d1"), _rec("retry", episode_uuid="ep-1", kind="episode"),
           trailing_newline=False)
    log = state.IngestStateLog(log_path)
    assert log.progress("ns1").retried_episodes == {"ep-1"}  # kept, not dropped

    log.record_retry("ns1", "ep-2", "projection")
    lines = log_path.read_bytes().split(b"\n")
    assert lines[-1] == b""
    parsed = [json.loads(line) for line in lines[:-1]]
    assert len(parsed) == 3
    assert parsed[1]["episode_uuid"] == "ep-1"
    assert parsed[2]["episode_uuid"] == "ep-2"
    reloaded = state.IngestStateLog(log_path).progress("ns1")
    assert reloaded.retried_episodes == {"ep-1"}
    assert reloaded.retried_projections == {"ep-2"}


def test_record_retry_bad_kind_raises_and_writes_nothing(log_path):
    log = state.IngestStateLog(log_path)
    log.record_plan("ns1", "d1", 1)
    before = log_path.read_bytes()
    with pytest.raises(ValueError):
        log.record_retry("ns1", "ep-1", "turn")
    assert log_path.read_bytes() == before
    assert log.progress("ns1").retried_episodes == set()


def test_path_for_places_state_next_to_manifest(tmp_path):
    manifest = tmp_path / "sub" / "manifest-foo.json"
    assert state.IngestStateLog.path_for(manifest) == tmp_path / "sub" / "ingest-state-manifest-foo.jsonl"
