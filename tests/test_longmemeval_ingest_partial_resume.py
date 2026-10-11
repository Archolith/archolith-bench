"""Partial resume for LongMemEval ingest (--resume-partial): literals, settle timeout, decisions."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_ingest():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "ingest.py"
    spec = importlib.util.spec_from_file_location("lme_ingest_partial_resume_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_ingest_partial_resume_test"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


# ---------------------------------------------------------------- 1. Cypher literal


def _decode_cypher_literal(lit: str) -> str:
    """Reference decoder: Neo4j-style single-quoted string with \\' \\\\ \\n \\r \\t \\uXXXX."""
    assert lit[0] == "'" and lit[-1] == "'"
    body = lit[1:-1]
    units: list[int] = []  # UTF-16 code units; plain chars are appended as their own units
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "'":
            raise AssertionError("unescaped quote inside literal")
        if ch != "\\":
            units.extend(int.from_bytes(ch.encode("utf-16-le")[j:j + 2], "little")
                         for j in range(0, len(ch.encode("utf-16-le")), 2))
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in ("'", "\\"):
            units.append(ord(nxt))
            i += 2
        elif nxt == "n":
            units.append(10)
            i += 2
        elif nxt == "r":
            units.append(13)
            i += 2
        elif nxt == "t":
            units.append(9)
            i += 2
        elif nxt == "u":
            units.append(int(body[i + 2:i + 6], 16))
            i += 6
        else:
            raise AssertionError(f"unknown escape \\{nxt}")
    raw = b"".join(u.to_bytes(2, "little") for u in units)
    return raw.decode("utf-16-le")


HOSTILE = (
    "it's a \\ back \"dq\" line\nnext\rcr\ttab \u00e9 \u4e2d "
    "\U0001F600 and literal text \\u0027 and \\u005C end"
)


def test_cypher_literal_roundtrips_hostile_text_without_unicode_quote_escapes() -> None:
    lit = ingest._cypher_string_literal(HOSTILE)
    assert lit.startswith("'") and lit.endswith("'")
    # A real ' / \ escape is a backslash NOT itself escaped (odd run of backslashes);
    # the hostile text legitimately contains the characters "'" as data, which are
    # emitted as an escaped backslash followed by plain "u0027".
    assert not re.search(r"(?<!\\)(?:\\\\)*\\u(?:0027|005[Cc])", lit)
    assert "\\\\u0027" in lit  # the data text survived as escaped-backslash + u0027
    assert _decode_cypher_literal(lit) == HOSTILE
    # The astral char must be a surrogate pair of escapes.
    assert "\\uD83D\\uDE00" in lit


def test_cypher_literal_apostrophe_and_backslash_use_backslash_escapes() -> None:
    assert ingest._cypher_string_literal("a'b\\c") == "'a\\'b\\\\c'"


def test_cypher_literal_plain_ascii_passes_through() -> None:
    assert ingest._cypher_string_literal("Hello, world 123 ~!") == "'Hello, world 123 ~!'"


# ---------------------------------------------------------------- 2. settle timeout


def test_settle_timeout_flag_off_is_60_and_never_reads_neo4j(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(ns):
        raise AssertionError("Neo4j must not be read without --resume-partial")

    monkeypatch.setattr(ingest, "_lease_seconds_remaining", boom)
    monkeypatch.setattr(ingest, "_cypher_counts", lambda *a, **k: boom(None))
    args = SimpleNamespace(resume_partial=False, drain_timeout=7200)
    assert ingest._settle_timeout_s(["a", "b"], args) == 60.0


def test_settle_timeout_flag_on_outlasts_longest_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    leases = {"a": 0, "b": 120}
    monkeypatch.setattr(ingest, "_lease_seconds_remaining", lambda ns: leases[ns])
    args = SimpleNamespace(resume_partial=True, drain_timeout=7200)
    assert ingest._settle_timeout_s(["a", "b"], args) == 7320.0


def test_settle_timeout_unreadable_lease_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    leases = {"a": 0, "b": -1}
    monkeypatch.setattr(ingest, "_lease_seconds_remaining", lambda ns: leases[ns])
    args = SimpleNamespace(resume_partial=True, drain_timeout=7200)
    with pytest.raises(RuntimeError):
        ingest._settle_timeout_s(["a", "b"], args)


# ---------------------------------------------------------------- 3. stored reader


def test_stored_reader_count_unreadable_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "_cypher_count", lambda q: -1)
    monkeypatch.setattr(ingest, "_cypher", lambda *a, **k: pytest.fail("rows must not be read"))
    assert ingest._stored_submitted_episodes("ns", ["x"]) is None


def test_stored_reader_zero_count_is_empty_without_row_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest, "_cypher_count", lambda q: 0)
    monkeypatch.setattr(ingest, "_cypher", lambda *a, **k: pytest.fail("row query issued"))
    assert ingest._stored_submitted_episodes("ns", ["x"]) == []


def _stub_rows(monkeypatch: pytest.MonkeyPatch, count: int, rows: list[list[str]]) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(ingest, "_cypher_count", lambda q: count)

    def fake_cypher(query, *, stdin=False):
        calls.append({"query": query, "stdin": stdin})
        return rows

    monkeypatch.setattr(ingest, "_cypher", fake_cypher)
    return calls


def test_stored_reader_row_count_mismatch_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_rows(monkeypatch, 2, [["0", "u1", "READY", "1", "TRUE"]])
    assert ingest._stored_submitted_episodes("ns", ["a", "b"]) is None


def test_stored_reader_bad_flag_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_rows(monkeypatch, 1, [["0", "u1", "READY", "1", "MAYBE"]])
    assert ingest._stored_submitted_episodes("ns", ["a"]) is None


def test_stored_reader_short_row_and_bad_int_are_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_rows(monkeypatch, 1, [["0", "u1", "READY", "1"]])
    assert ingest._stored_submitted_episodes("ns", ["a"]) is None
    _stub_rows(monkeypatch, 1, [["0", "u1", "READY", "x", "TRUE"]])
    assert ingest._stored_submitted_episodes("ns", ["a"]) is None


def test_stored_reader_parses_rows_and_sends_bodies_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _stub_rows(
        monkeypatch,
        2,
        [["0", "u1", "ready", "2", "true"], ["1", "u2", "FAILED", "", "FALSE"]],
    )
    stored = ingest._stored_submitted_episodes("ns", ["it's", "assistant: b"])
    assert stored == [
        ingest._StoredEpisode("u1", "READY", 2, True),
        ingest._StoredEpisode("u2", "FAILED", 0, False),
    ]
    assert len(calls) == 1 and calls[0]["stdin"] is True
    assert "['it\\'s', 'assistant: b']" in calls[0]["query"]


# ---------------------------------------------------------------- 4. resume decisions


class _Adapter:
    def sessions(self, item):
        return item["sessions"]


def _item() -> dict:
    return {
        "question_id": "q1",
        "sessions": [
            [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}],
            [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}],
        ],
    }


EXPECTED = ["u1", "assistant: a1", "u2", "assistant: a2"]


class Resume:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, stored) -> None:
        self.resets: list[str] = []
        self.evidence: list[str] = []
        self.expected_seen: list[list[str]] = []

        def fake_stored(ns, expected):
            self.expected_seen.append(list(expected))
            return stored

        monkeypatch.setattr(ingest, "_stored_submitted_episodes", fake_stored)
        monkeypatch.setattr(
            ingest, "_reset_namespace", lambda admin, url, ns: self.resets.append(ns)
        )
        monkeypatch.setattr(
            ingest, "_ensure_turn_evidence",
            lambda client, ns, role, **kw: self.evidence.append(kw["evidence_body"]),
        )
        self.state = ingest.WindowItem(item=_item(), question_id="q1", namespace="ns1")

    def run(self):
        return ingest._resume_or_reset(
            object(), object(), "http://x", self.state,
            adapter=_Adapter(), segmentation="none", evidence_by_turn={},
        )


def _ep(uuid: str, state: str = "READY", matches: bool = True):
    return ingest._StoredEpisode(uuid, state, 1, matches)


def _contents(turns) -> list[str]:
    return [t.content for t in turns]


def test_resume_unreadable_stored_raises_and_does_not_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = Resume(monkeypatch, None)
    with pytest.raises(RuntimeError):
        r.run()
    assert r.resets == []


def test_resume_empty_stored_resets_and_starts_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [])
    point = r.run()
    assert r.resets == ["ns1"]
    assert point.resumed is None and point.active is None
    assert _contents(point.turns) == ["u1", "a1", "u2", "a2"]
    assert r.state.resumed is None


def test_resume_mismatching_episode_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [_ep("e1"), _ep("e2", matches=False)])
    point = r.run()
    assert r.resets == ["ns1"]
    assert point.resumed is None
    assert _contents(point.turns) == ["u1", "a1", "u2", "a2"]
    assert r.state.turns == 0 and r.evidence == []


def test_resume_more_stored_than_expected_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [_ep(f"e{i}") for i in range(5)])
    point = r.run()
    assert r.resets == ["ns1"]
    assert point.resumed is None


@pytest.mark.parametrize("bad_state", ["ENRICHING", "PENDING"])
def test_resume_in_flight_stored_episode_resets(
    monkeypatch: pytest.MonkeyPatch, bad_state: str
) -> None:
    r = Resume(monkeypatch, [_ep("e1"), _ep("e2", bad_state)])
    point = r.run()
    assert r.resets == ["ns1"]
    assert point.resumed is None
    assert r.state.turns == 0


def test_resume_happy_path_skips_k_and_positions_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = Resume(monkeypatch, [_ep("e1"), _ep("e2"), _ep("e3")])
    point = r.run()
    assert r.resets == []
    assert _contents(point.turns) == ["a2"]  # exactly the 4th extract turn
    assert r.state.turns == 3
    assert r.evidence == ["u1", "u2"]  # consumed user turns only, not assistant a1
    assert point.active is None
    expected_resumed = {"skipped_episodes": 3, "failed_in_prefix": 0, "adopted_failed": False}
    assert point.resumed == expected_resumed
    assert r.state.resumed == expected_resumed


def test_resume_last_failed_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [_ep("e1", "FAILED"), _ep("e2"), _ep("e3", "FAILED")])
    point = r.run()
    assert r.resets == []
    assert point.active == ("e3", False)
    assert point.resumed == {
        "skipped_episodes": 3, "failed_in_prefix": 2, "adopted_failed": True,
    }


def test_resume_failed_not_last_is_not_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [_ep("e1", "FAILED"), _ep("e2")])
    point = r.run()
    assert point.active is None
    assert point.resumed["failed_in_prefix"] == 1
    assert point.resumed["adopted_failed"] is False


def test_resume_all_stored_leaves_nothing_to_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    r = Resume(monkeypatch, [_ep(f"e{i}") for i in range(4)])
    point = r.run()
    assert r.resets == []
    assert _contents(point.turns) == []
    assert r.state.turns == 4
    assert point.resumed["skipped_episodes"] == 4


def test_resume_expected_bodies_raw_for_user_prefixed_for_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = Resume(monkeypatch, [])
    r.run()
    assert r.expected_seen == [EXPECTED]


# ---------------------------------------------------------------- 5. _ingest_window adoption


def test_ingest_window_monitors_adopted_episode_instead_of_submitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    outcomes = iter(["wait", "next", "next"])
    submits = iter(["ep-y", None])

    def fake_advance(admin, url, uuid, already_retried):
        outcome = next(outcomes)
        events.append(f"advance:{uuid}:{outcome}")
        return outcome

    def fake_submit(client, state, turns, evidence):
        events.append("submit")
        return next(submits)

    monkeypatch.setattr(ingest, "_advance_active_episode", fake_advance)
    monkeypatch.setattr(ingest, "_submit_next_turn", fake_submit)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)

    window = [ingest.WindowItem(item={}, question_id="q", namespace="ns")]
    ingest._ingest_window(
        window, [iter(())], object(), object(), "http://x",
        timeout_s=1000.0, initial_active={0: ("ep-x", False)},
    )

    assert events[0] == "advance:ep-x:wait"  # nothing submitted before ep-x is monitored
    assert events[1] == "advance:ep-x:next"
    assert events[2] == "submit"  # only after ep-x advanced
    assert events[3] == "advance:ep-y:next"
    assert events.count("submit") == 2


# ---------------------------------------------------------------- 6. _complete_item


def _complete(monkeypatch: pytest.MonkeyPatch, resumed) -> dict:
    monkeypatch.setattr(ingest, "_integrity_counts", lambda ns, epochs: {"cross_namespace_links": 0})
    state = ingest.WindowItem(item=_item(), question_id="q1", namespace="ns1", turns=4,
                              resumed=resumed)

    class A(_Adapter):
        def question(self, item):
            return "Q?"

    args = SimpleNamespace(
        consolidate_scalar=False, fixture=None, namespace_window=1, rolling_window=False,
        consolidation_k=1, segmentation="none", menhir_url="http://x",
    )
    return ingest._complete_item(
        state, {"total": 4, "ready": 4}, 0,
        args=args, adapter=A(), admin=object(), requested_lanes={"scalar": False},
    )


def test_complete_item_includes_resumed_only_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "resumed" not in _complete(monkeypatch, None)
    info = {"skipped_episodes": 2, "failed_in_prefix": 0, "adopted_failed": False}
    assert _complete(monkeypatch, info)["resumed"] == info
