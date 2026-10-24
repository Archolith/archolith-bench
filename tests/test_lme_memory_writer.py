"""Tests for the LongMemEval agent memory writer prepass (#168 P3).

The failures these guard against are silent: a failed or unparseable writer reply treated as
"no items", a stale checkpoint row replayed after a prompt/model change, or an item cited to the
wrong (or a user) turn so it grounds on evidence it does not come from.
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


mw = _load_script("lme_memory_writer_test", "scripts/longmemeval/lib/memory_writer.py")
from archolith_bench.core.openai_flex import FlexResult  # noqa: E402

T = mw.SessionTurn


def _session(turns, *, ns="ns1", sid="s1", date="2024-05-01"):
    return mw.WriterSession(namespace=ns, session_id=sid, date=date, turns=tuple(turns))


def _ok(cid, content, pt=10, ct=5, cost=0.25):
    return FlexResult(cid, content, pt, ct, cost)


class FakeFlex:
    """Stands in for run_flex: skips checkpointed ids, appends rows in the real format."""

    def __init__(self, reply):
        self.reply = reply  # (cid) -> FlexResult | None (None = omit from results)
        self.calls: list[list[str]] = []

    def __call__(self, requests, *, model, checkpoint, **kw):
        ids = [r["custom_id"] for r in requests]
        done = {}
        if checkpoint.exists():
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[row["custom_id"]] = row
        pending = [c for c in ids if c not in done]
        self.calls.append(pending)
        results = {}
        for cid in pending:
            res = self.reply(cid)
            if res is None:
                continue
            row = {"custom_id": res.custom_id, "content": res.content,
                   "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                   "cost_usd": res.cost_usd, "error": res.error}
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            results[cid] = res
        for cid, row in done.items():
            if cid in ids:
                results[cid] = FlexResult(cid, row["content"], row["prompt_tokens"],
                                          row["completion_tokens"], row["cost_usd"], row["error"])
        return results, {}


def _ckpt_ids(path):
    return [json.loads(x)["custom_id"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _run(sessions, fake, tmp_path, granularity="turn", model="m1"):
    return mw.write_memories(sessions, granularity=granularity, model=model,
                             checkpoint=tmp_path / "w.jsonl", run_flex=fake, flex_kwargs={})


@pytest.mark.parametrize(
    ("granularity", "items"),
    [
        ("turn", ["just a string", {"text": "   "}]),
        ("session", [{"turn": 0, "text": "cites the user turn"}, {"turn": "t1", "text": "x"},
                     {"text": "no turn"}, {"turn": 1.0, "text": "float turn"}]),
    ],
)
def test_reply_with_only_unusable_items_fails_and_is_not_replayed(tmp_path, granularity, items):
    """Items written but none usable is a broken reply, never "nothing to record"."""
    session = _session([T(0, "user", "q"), T(1, "assistant", "Try Dune.")])
    fake = FakeFlex(lambda cid: _ok(cid, json.dumps({"items": items})))

    with pytest.raises(mw.WriterError, match="malformed or mis-cited"):
        _run([session], fake, tmp_path, granularity=granularity)

    assert _ckpt_ids(tmp_path / "w.jsonl") == []  # a rerun asks again
    rerun = FakeFlex(lambda cid: _ok(cid, json.dumps({"items": []})))
    items_out, report = _run([session], rerun, tmp_path, granularity=granularity)
    assert len(rerun.calls[0]) == 1 and items_out == {} and report.items == 0  # empty is valid


def test_reply_with_some_usable_items_keeps_them_and_counts_the_rest(tmp_path):
    session = _session([T(0, "user", "q"), T(1, "assistant", "Try Dune.")])
    fake = FakeFlex(lambda cid: _ok(cid, json.dumps({"items": [{"text": "kept"}, "junk"]})))

    items, report = _run([session], fake, tmp_path)

    assert [i.text for i in items[("ns1", "s1", 1)]] == ["kept"] and report.dropped == 1


@pytest.mark.parametrize("granularity", mw.GRANULARITIES)
def test_changed_session_text_never_replays_checkpoint_row(tmp_path, granularity):
    """Same namespace/session id, different text: the old row must not be replayed."""
    def reply(text):
        def _reply(cid):
            item = {"text": text} if granularity == "turn" else {"turn": 1, "text": text}
            return _ok(cid, json.dumps({"items": [item]}))
        return _reply

    old = _session([T(0, "user", "q"), T(1, "assistant", "Try Dune.")])
    new = _session([T(0, "user", "q"), T(1, "assistant", "Try Hyperion.")])
    first = FakeFlex(reply("recommended Dune"))
    items, _ = _run([old], first, tmp_path, granularity=granularity)
    assert [i.text for i in items[("ns1", "s1", 1)]] == ["recommended Dune"]

    second = FakeFlex(reply("recommended Hyperion"))
    items, _ = _run([new], second, tmp_path, granularity=granularity)
    assert len(second.calls[0]) == 1  # re-asked, not replayed
    assert [i.text for i in items[("ns1", "s1", 1)]] == ["recommended Hyperion"]


# 1 ---------------------------------------------------------------------------------------------

def test_prompt_sha_per_granularity_stable_and_in_cid():
    assert mw.prompt_sha("turn") != mw.prompt_sha("session")
    assert mw.prompt_sha("turn") == mw.prompt_sha("turn")
    for g in mw.GRANULARITIES:
        cid = mw.custom_id("k", granularity=g, model="gpt-x")
        assert cid.startswith(f"{g}|gpt-x|{mw.prompt_sha(g)[:16]}|")
    assert mw.custom_id("k", granularity="turn", model="a") != mw.custom_id("k", granularity="turn", model="b")
    with pytest.raises(ValueError):
        mw.prompt_sha("bogus")


# 2 ---------------------------------------------------------------------------------------------

def test_turn_mode_requests_and_preceding_user_block():
    s = _session([T(0, "user", "Q one"), T(1, "assistant", "A one"),
                  T(2, "assistant", "A two"), T(3, "user", "Q three"), T(4, "assistant", "A three")])
    reqs = mw.build_requests([s], granularity="turn", model="m")
    assert len(reqs) == 3
    msgs = [r["body"]["messages"][1]["content"] for r in reqs]
    assert all("Date: 2024-05-01" in m for m in msgs)
    assert "[user]\nQ one" in msgs[0] and "[assistant]\nA one" in msgs[0]
    assert "[user]" not in msgs[1] and "[assistant]\nA two" in msgs[1]  # preceded by assistant
    assert "[user]\nQ three" in msgs[2] and "Q one" not in msgs[2]
    assert reqs[0]["custom_id"].endswith("|t1") and reqs[2]["custom_id"].endswith("|t4")


def test_session_mode_uses_raw_indices_and_skips_assistantless_sessions():
    s = _session([T(0, "user", "hi"), T(1, "assistant", "yo"), T(3, "user", "more")])
    no_asst = _session([T(0, "user", "x"), T(1, "user", "y")], sid="s2")
    reqs = mw.build_requests([s, no_asst], granularity="session", model="m")
    assert len(reqs) == 1
    msg = reqs[0]["body"]["messages"][1]["content"]
    assert "[turn 0] [user]\nhi" in msg
    assert "[turn 1] [assistant]\nyo" in msg
    assert "[turn 3] [user]\nmore" in msg
    assert "[turn 2]" not in msg and "Date: 2024-05-01" in msg


# 3 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("g", ["turn", "session"])
def test_request_body_shape(g):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a")])
    (req,) = mw.build_requests([s], granularity=g, model="m")
    body = req["body"]
    assert body["messages"][0] == {"role": "system", "content": mw.prompt_for(g)}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == mw.MAX_COMPLETION_TOKENS
    assert "model" not in body
    assert "{item_shape}" not in body["messages"][0]["content"]


# 4 ---------------------------------------------------------------------------------------------

def test_parse_items_errors_and_empty():
    kw = dict(granularity="turn", assistant_indices={1}, default_turn=1)
    with pytest.raises(mw.WriterError):
        mw.parse_items("not json", **kw)
    with pytest.raises(mw.WriterError):
        mw.parse_items(None, **kw)
    with pytest.raises(mw.WriterError):
        mw.parse_items('{"foo": 1}', **kw)
    with pytest.raises(mw.WriterError):
        mw.parse_items('{"items": "x"}', **kw)
    assert mw.parse_items('{"items": []}', **kw) == ([], 0)


def test_parse_items_turn_mode_drops_bad_text_collapses_ws_ignores_turn_field():
    content = json.dumps({"items": [
        {"text": "  a   b\n c "}, {"text": ""}, {"text": "   "}, {"text": 5}, "str", {"turn": 1},
        {"text": "kept", "turn": 99},
    ]})
    items, dropped = mw.parse_items(content, granularity="turn", assistant_indices={7}, default_turn=7)
    assert [(i.turn_index, i.text) for i in items] == [(7, "a b c"), (7, "kept")]
    assert dropped == 5


def test_parse_items_session_mode_citations():
    content = json.dumps({"items": [
        {"turn": 1, "text": "ok"},
        {"turn": 0, "text": "user turn"},
        {"turn": 2, "text": "missing index"},
        {"turn": True, "text": "bool"},
        {"text": "absent"},
        {"turn": "3", "text": "string three"},
        {"turn": "x", "text": "garbage"},
    ]})
    items, dropped = mw.parse_items(content, granularity="session", assistant_indices={1, 3})
    assert [(i.turn_index, i.text) for i in items] == [(1, "ok"), (3, "string three")]
    assert dropped == 5


# 5 ---------------------------------------------------------------------------------------------

def test_write_memories_keys_order_and_report(tmp_path):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a1"), T(2, "user", "q2"), T(4, "assistant", "a2")])
    replies = {"|t1": {"items": [{"text": "one"}, {"text": "two"}, {"text": " "}]},
               "|t4": {"items": [{"text": "three"}]}}

    def reply(cid):
        key = "|t1" if cid.endswith("|t1") else "|t4"
        return _ok(cid, json.dumps(replies[key]), pt=100, ct=7, cost=0.5)

    items, report = _run([s], FakeFlex(reply), tmp_path)
    assert {k: [i.text for i in v] for k, v in items.items()} == {
        ("ns1", "s1", 1): ["one", "two"], ("ns1", "s1", 4): ["three"]}
    assert (report.requests, report.items, report.dropped) == (2, 3, 1)
    assert (report.prompt_tokens, report.completion_tokens) == (200, 14)
    assert report.cost_usd == pytest.approx(1.0)


def test_max_items_per_turn_cap(tmp_path):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a")])
    body = json.dumps({"items": [{"text": f"i{n}"} for n in range(25)]})
    items, report = _run([s], FakeFlex(lambda cid: _ok(cid, body)), tmp_path)
    kept = items[("ns1", "s1", 1)]
    assert len(kept) == mw.MAX_ITEMS_PER_TURN == 20
    assert [i.text for i in kept] == [f"i{n}" for n in range(20)]
    assert (report.items, report.dropped) == (20, 5)


# 6 / 7 -----------------------------------------------------------------------------------------

def test_failed_request_raises_drops_row_and_rerun_retries_only_it(tmp_path):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a1"), T(2, "user", "q"), T(3, "assistant", "a3")])
    good = json.dumps({"items": [{"text": "x"}]})

    def broken(cid):
        if cid.endswith("|t3"):
            return FlexResult(cid, None, 0, 0, 0.0, error="HTTP 500")
        return _ok(cid, good)

    fake = FakeFlex(broken)
    with pytest.raises(mw.WriterError, match="1 of 2"):
        _run([s], fake, tmp_path)
    ckpt = tmp_path / "w.jsonl"
    ids = _ckpt_ids(ckpt)
    assert len(ids) == 1 and ids[0].endswith("|t1")

    fixed = FakeFlex(lambda cid: _ok(cid, good))
    items, report = _run([s], fixed, tmp_path)
    assert len(fixed.calls) == 1 and len(fixed.calls[0]) == 1 and fixed.calls[0][0].endswith("|t3")
    assert set(items) == {("ns1", "s1", 1), ("ns1", "s1", 3)}
    assert report.items == 2


def test_cid_missing_from_results_raises(tmp_path):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a")])
    with pytest.raises(mw.WriterError):
        _run([s], FakeFlex(lambda cid: None), tmp_path)


# 8 ---------------------------------------------------------------------------------------------

def test_unparseable_reply_names_cid_and_is_not_replayed(tmp_path):
    s = _session([T(0, "user", "q"), T(1, "assistant", "a1"), T(2, "user", "q"), T(3, "assistant", "a3")])
    good = json.dumps({"items": [{"text": "x"}]})

    def reply(cid):
        return _ok(cid, "sorry, no JSON" if cid.endswith("|t3") else good)

    with pytest.raises(mw.WriterError) as exc:
        _run([s], FakeFlex(reply), tmp_path)
    bad_cid = mw.custom_id(mw.request_key(s, "turn", 3), granularity="turn", model="m1")
    assert bad_cid in str(exc.value)
    ids = _ckpt_ids(tmp_path / "w.jsonl")
    assert bad_cid not in ids and len(ids) == 1

    fixed = FakeFlex(lambda cid: _ok(cid, good))
    items, _ = _run([s], fixed, tmp_path)
    assert fixed.calls == [[bad_cid]]
    assert ("ns1", "s1", 3) in items


# 9 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("g", ["turn", "session"])
def test_no_assistant_turns_never_calls_run_flex(tmp_path, g):
    s = _session([T(0, "user", "q"), T(1, "user", "q2")])

    def boom(*a, **k):
        raise AssertionError("run_flex must not be called")

    items, report = _run([s], boom, tmp_path, granularity=g)
    assert items == {} and report.requests == 0
