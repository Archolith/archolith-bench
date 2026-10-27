"""LME_ASSISTANT_STORE_ONLY: raw assistant turns as store-only source memories (#168).

On, every assistant turn is sent verbatim, with role="assistant", as one episode grounded on its
own turn evidence, and no store_only: Menhir's assistant-turn policy (LME_ASSISTANT_TURN_POLICY ->
MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED) keeps it store-only. Off (the default), the adaptive
segmenter routes it as before.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LME = ROOT / "scripts" / "longmemeval"


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_assistant_store_only_test", "scripts/longmemeval/lib/ingest.py")

# Short and fact-free: the segmenter routes it CONTEXT_ONLY (evidence only).
PLAIN = "Sure, happy to help with that."
# Long, multi-sentence advice with the details SSA questions ask about.
LONG = (
    "Here are three hikes near Portland. Multnomah Falls is 2.4 miles round trip. "
    "Angel's Rest is 4.8 miles with 1,500 feet of gain. Dog Mountain is 6.9 miles and best in May. "
    "Bring layers, since the Gorge is windy, and start early on weekends to find parking."
)
USER = "I moved to Portland last month and want to go hiking."


class _Adapter:
    def sessions(self, item):
        return item["sessions"]


def _item() -> dict:
    return {"sessions": [[
        {"role": "user", "content": USER},
        {"role": "assistant", "content": PLAIN},
        {"role": "user", "content": "Which one is the longest?"},
        {"role": "assistant", "content": LONG},
    ]]}


def _plan(monkeypatch: pytest.MonkeyPatch, enabled: bool, segmentation: str = "adaptive"):
    monkeypatch.setattr(ingest, "ASSISTANT_STORE_ONLY", enabled)
    return list(ingest._iter_item_turns(_Adapter(), _item(), "ns-a", segmentation))


def test_on_every_assistant_turn_is_one_verbatim_raw_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = [t for t in _plan(monkeypatch, True) if t.role == "assistant"]
    assert [t.content for t in assistant] == [PLAIN, LONG]
    # Menhir decides store-only from the role; the bench must not ask for it.
    assert all(t.extract and t.raw_turn and not t.store_only and t.evidence_text == t.content
               for t in assistant)
    assert all(t.segment_index == 0 and t.writer_index == -1 for t in assistant)


def test_flag_leaves_user_turns_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    def user_rows(enabled: bool):
        return [(t.content, t.extract, t.store_only, t.raw_turn)
                for t in _plan(monkeypatch, enabled) if t.role == "user"]

    assert user_rows(True) == user_rows(False)


def test_off_assistant_turns_route_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    assistant = [t for t in _plan(monkeypatch, False) if t.role == "assistant"]
    assert assistant and not any(t.store_only or t.raw_turn for t in assistant)
    assert [t for t in assistant if t.content == PLAIN][0].extract is False


def test_flag_does_not_touch_writer_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def rows(enabled: bool):
        return [(t.content, t.extract, t.store_only, t.raw_turn)
                for t in _plan(monkeypatch, enabled, "adaptive-writer")]

    assert rows(True) == rows(False)


def test_flag_changes_the_plan_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    on = _plan(monkeypatch, True)
    off = _plan(monkeypatch, False)
    assert ingest._turn_plan_digest(on, "adaptive") != ingest._turn_plan_digest(off, "adaptive")


class _Client:
    def __init__(self) -> None:
        self.evidence: list[dict] = []
        self.episodes: list[dict] = []

    def record_turn_evidence(self, namespace, text, **kwargs):
        self.evidence.append({"text": text, **kwargs})
        return {"turn_id": f"turn-{len(self.evidence)}"}

    def ingest(self, namespace, role, content, **kwargs):
        self.episodes.append({"role": role, "content": content, **kwargs})
        return {"episode_id": f"episode-{len(self.episodes)}"}


def test_raw_assistant_turn_is_grounded_names_its_role_and_sends_no_store_only() -> None:
    client = _Client()
    uuid = ingest._ingest_turn(client, "ns-a", "assistant", LONG, session_id="s0", evidence_text=LONG,
                               turn_key="ns-a|s0|t3", ingest_key="ns-a|s0|t3#s0", raw_turn=True)
    assert uuid == "episode-1"
    assert [(e["text"], e["role"], e["declarant"]) for e in client.evidence] == [(LONG, "assistant", "assistant")]
    [episode] = client.episodes
    assert episode["content"] == LONG
    assert "store_only" not in episode
    assert episode["memory_role"] == "assistant"
    assert episode["source"] is None
    assert episode["turn_evidence_uuid"] == "turn-1"


def test_raw_turn_request_differs_from_the_old_store_only_request_only_by_store_only() -> None:
    old, new = _Client(), _Client()
    kw = dict(session_id="s0", evidence_text=LONG, turn_key="ns-a|s0|t3", ingest_key="ns-a|s0|t3#s0")
    ingest._ingest_turn(old, "ns-a", "assistant", LONG, store_only=True, **kw)
    ingest._ingest_turn(new, "ns-a", "assistant", LONG, raw_turn=True, **kw)
    assert old.evidence == new.evidence
    [old_episode], [new_episode] = old.episodes, new.episodes
    assert old_episode.pop("store_only") is True
    assert old_episode == new_episode


def test_store_only_assistant_turn_is_grounded_and_names_its_role() -> None:
    client = _Client()
    uuid = ingest._ingest_turn(client, "ns-a", "assistant", LONG, session_id="s0", evidence_text=LONG,
                               turn_key="ns-a|s0|t3", ingest_key="ns-a|s0|t3#s0", store_only=True)
    assert uuid == "episode-1"
    assert [(e["text"], e["role"], e["declarant"]) for e in client.evidence] == [(LONG, "assistant", "assistant")]
    [episode] = client.episodes
    assert episode["content"] == LONG
    assert episode["store_only"] is True
    assert episode["memory_role"] == "assistant"
    assert episode["source"] is None
    assert episode["turn_evidence_uuid"] == "turn-1"


def test_store_only_user_turn_request_is_unchanged() -> None:
    client = _Client()
    ingest._ingest_turn(client, "ns-a", "user", "What should I cook in my Instant Pot?", session_id="s0",
                        turn_key="ns-a|s0|t0", ingest_key="ns-a|s0|t0#s0", store_only=True)
    [episode] = client.episodes
    assert "memory_role" not in episode
    assert episode["store_only"] is True and episode["source"] == "user"


def test_config_defaults_off_and_build_records_and_checks_it() -> None:
    config = (LME / "config.sh").read_text(encoding="utf-8")
    assert re.search(r'^LME_ASSISTANT_STORE_ONLY="\$\{LME_ASSISTANT_STORE_ONLY:-0\}"', config, re.MULTILINE)
    assert re.search(r'^LME_ASSISTANT_TURN_POLICY="\$\{LME_ASSISTANT_TURN_POLICY:-1\}"', config, re.MULTILINE)
    build = (LME / "build_graph.sh").read_text(encoding="utf-8")
    assert 'export LME_ASSISTANT_STORE_ONLY="${LME_ASSISTANT_STORE_ONLY}"' in build
    assert '--setting "assistant_store_only=${LME_ASSISTANT_STORE_ONLY}"' in build
    assert '--setting "assistant_turn_policy=${LME_ASSISTANT_TURN_POLICY}"' in build
    assert 'export MENHIR_ASSISTANT_TURNS_STORE_ONLY_ENABLED="${LME_ASSISTANT_TURN_POLICY}"' in build
    serve = build.index('"${MENHIR_FRONTIER_BIN}" serve')
    assert build.index("LME_ASSISTANT_STORE_ONLY=1 needs LME_SEGMENTATION=adaptive") < serve
    # An older Menhir ignores the policy env and would extract every raw assistant turn.
    assert build.index("has no assistant-turn policy") < serve
