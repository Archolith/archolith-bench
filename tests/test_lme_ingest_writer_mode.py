"""Tests for the ``adaptive-writer`` segmentation mode of the LongMemEval ingest (#168 P3)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


ingest = _load_script("lme_ingest_writer_mode_test", "scripts/longmemeval/lib/ingest.py")
memory_writer = ingest.memory_writer

NS = "lme-w"
SID_A = f"{NS}-sA"
SID_B = f"{NS}-sB"
ASSISTANT_A = "You could try Powell's Books and Forest Park."
ASSISTANT_B = "Try the Cartopia pod on Hawthorne."
ASSISTANT_C = "Common GPS issues include signal loss."
GPS_USER = (
    "I've been doing some research and found a local detailer with great reviews. "
    "I was thinking of taking my car there, but I'm also considering other options. "
    "By the way, I recently had an issue with my car's GPS system on 3/22, "
    "and I had to take it back to the dealership to get it fixed. "
    "They replaced the entire system, and now it's working flawlessly. "
    "Have you heard of any common issues with GPS systems in newer cars?"
)


def _item() -> dict:
    return {
        "question_id": "w",
        "question": "Q?",
        "answer": "A",
        "question_type": "knowledge-update",
        "haystack_session_ids": ["sA", "sB"],
        "haystack_dates": ["2023/04/10 (Mon) 14:47", "2023/04/11 (Tue) 09:05"],
        "sessions": [
            [
                {"role": "user", "content": "I recently moved to Portland and I love it here."},
                {"role": "assistant", "content": ASSISTANT_A},
                {"role": "user", "content": ""},  # skipped; raw indices are non-contiguous
                {"role": "user", "content": "Thanks! What about food carts?"},
                {"role": "assistant", "content": ASSISTANT_B},
            ],
            [
                {"role": "user", "content": GPS_USER},
                {"role": "assistant", "content": ASSISTANT_C},
            ],
        ],
    }


class _Adapter:
    def sessions(self, item):
        return item["sessions"]

    def question(self, item):
        return item["question"]

    def load_items(self, **kwargs):
        return [_item()]


def _wi(turn_index: int, text: str):
    return memory_writer.WriterItem(turn_index=turn_index, text=text)


def _writer_items() -> dict:
    return {
        (NS, SID_A, 1): [_wi(1, "On 2023/04/10, the assistant recommended Powell's Books."),
                         _wi(1, "On 2023/04/10, the assistant recommended Forest Park.")],
        # session A turn 4 (after the skipped empty turn 2) has no items
        (NS, SID_B, 1): [_wi(1, "On 2023/04/11, the assistant said GPS signal can be lost.")],
    }


def _turns(segmentation: str, writer_items: dict | None = None) -> list:
    return list(ingest._iter_item_turns(_Adapter(), _item(), NS, segmentation, writer_items))


# 1 ---------------------------------------------------------------------------------------------


def _legacy_digest(plan: list, segmentation: str) -> str:
    def h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    payload = {
        "v": 1,
        "segmentation": segmentation,
        "split_min_length": ingest._SPLIT_MIN_LENGTH,
        "turns": [
            [t.turn_key, t.segment_index, t.extract, t.role, t.occurred_at, t.session_id,
             h(t.content), h(t.evidence_text)]
            for t in plan
        ],
    }
    return h(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


@pytest.mark.parametrize("segmentation", ["adaptive", "none"])
def test_non_writer_digest_is_byte_identical_to_pre_change_digest(segmentation: str) -> None:
    plan = ingest._turn_plan(_Adapter(), _item(), NS, segmentation)
    roles = {t.role for t in plan}
    assert {"user", "assistant"} <= roles
    # A skipped empty turn leaves a gap in the raw indices.
    assert f"{NS}|{SID_A}|t3" in {t.turn_key for t in plan}
    assert f"{NS}|{SID_A}|t2" not in {t.turn_key for t in plan}

    assert ingest._turn_plan_digest(plan, segmentation) == _legacy_digest(plan, segmentation)


# 2 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("segmentation", ["adaptive", "none", "sentence", "user-sentence"])
def test_non_writer_modes_ignore_writer_items(segmentation: str) -> None:
    without = _turns(segmentation)
    with_items = _turns(segmentation, _writer_items())

    assert with_items == without
    assert all(t.writer_index < 0 for t in with_items)
    assert all(t.source == "" for t in with_items)


# 3 ---------------------------------------------------------------------------------------------


def test_writer_plan_shape() -> None:
    items = _writer_items()
    adaptive = _turns("adaptive")
    writer = _turns("adaptive-writer", items)

    # User turns are exactly the adaptive plan's, all fields.
    assert [t for t in writer if t.role == "user"] == [t for t in adaptive if t.role == "user"]

    assistant_keys = [
        f"{NS}|{SID_A}|t1", f"{NS}|{SID_A}|t4", f"{NS}|{SID_B}|t1",
    ]
    content_by_key = {
        assistant_keys[0]: ASSISTANT_A, assistant_keys[1]: ASSISTANT_B, assistant_keys[2]: ASSISTANT_C,
    }
    evidence = [t for t in writer if t.role == "assistant"]
    assert [t.turn_key for t in evidence] == assistant_keys  # exactly one per assistant turn
    for t in evidence:
        assert t.extract is False
        assert t.content == t.evidence_text == content_by_key[t.turn_key]
        assert t.writer_index == -1 and t.source == ""

    agents = [t for t in writer if t.role == "agent"]
    assert len(agents) == 3
    expected = {
        assistant_keys[0]: [i.text for i in items[(NS, SID_A, 1)]],
        assistant_keys[1]: [],
        assistant_keys[2]: [i.text for i in items[(NS, SID_B, 1)]],
    }
    for key in assistant_keys:
        pos = next(i for i, t in enumerate(writer) if t.role == "assistant" and t.turn_key == key)
        following = []
        for t in writer[pos + 1:]:
            if t.role != "agent":
                break
            following.append(t)
        assert [t.content for t in following] == expected[key]
        for k, t in enumerate(following):
            assert t.extract is True
            assert t.evidence_text == content_by_key[key]
            assert t.turn_key == key
            assert t.source == "agent-writer"
            assert t.writer_index == k
            assert t.occurred_at == writer[pos].occurred_at
            assert t.session_id == writer[pos].session_id


def test_writer_items_for_user_turn_or_wrong_scope_are_ignored() -> None:
    stray = {
        (NS, SID_A, 0): [_wi(0, "keyed to a USER turn index")],
        ("lme-other", SID_A, 1): [_wi(1, "wrong namespace")],
        (NS, f"{NS}-nope", 1): [_wi(1, "wrong session")],
    }
    writer = _turns("adaptive-writer", stray)

    assert [t for t in writer if t.role == "agent"] == []
    assert writer == _turns("adaptive-writer", {})


# 4 ---------------------------------------------------------------------------------------------


def test_writer_session_keys_match_plan_turn_keys() -> None:
    sessions = ingest._writer_sessions(_Adapter(), _item(), NS)
    plan = {t.turn_key: t for t in _turns("none")}

    derived = set()
    for session in sessions:
        for st in session.turns:
            key = f"{session.namespace}|{session.session_id}|t{st.index}"
            derived.add(key)
            assert key in plan
            assert plan[key].content == st.content
            assert plan[key].role == st.role
    assert derived == set(plan)
    # The skipped empty turn is absent from both, and the next turn kept its raw index.
    assert f"{NS}|{SID_A}|t2" not in derived
    assert f"{NS}|{SID_A}|t3" in derived

    # And the writer plan's assistant evidence turns line up with the writer's assistant turns.
    writer_plan = {t.turn_key: t for t in _turns("adaptive-writer") if t.role == "assistant"}
    writer_assistants = {
        f"{s.namespace}|{s.session_id}|t{st.index}"
        for s in sessions for st in s.turns if st.role == "assistant"
    }
    assert set(writer_plan) == writer_assistants


# 5 ---------------------------------------------------------------------------------------------


def test_ingest_key_distinguishes_writer_items_from_evidence_turn() -> None:
    items = {(NS, SID_A, 1): [_wi(1, "item one"), _wi(1, "item two"), _wi(1, "item three")]}
    plan = ingest._turn_plan(_Adapter(), _item(), NS, "adaptive-writer", writer_items=items)

    key = f"{NS}|{SID_A}|t1"
    evidence = next(t for t in plan if t.role == "assistant" and t.turn_key == key)
    writers = [t for t in plan if t.writer_index >= 0 and t.turn_key == key]

    assert ingest._ingest_key(evidence) == f"{key}#s0"
    assert [ingest._ingest_key(t) for t in writers] == [f"{key}#w0", f"{key}#w1", f"{key}#w2"]
    extract_keys = [ingest._ingest_key(t) for t in plan if t.extract]
    assert len(extract_keys) == len(set(extract_keys))


# 6 ---------------------------------------------------------------------------------------------


def _writer_digest(items: dict, *, model: str = "m1", granularity: str = "turn") -> str:
    plan = ingest._turn_plan(_Adapter(), _item(), NS, "adaptive-writer", writer_items=items)
    return ingest._turn_plan_digest(
        plan, "adaptive-writer", writer_model=model, writer_granularity=granularity
    )


def test_writer_digest_covers_item_text_model_and_granularity() -> None:
    base = _writer_digest(_writer_items())

    assert _writer_digest(_writer_items()) == base  # stable across identical builds

    changed = _writer_items()
    changed[(NS, SID_A, 1)][0] = _wi(1, "a different memory text")
    assert _writer_digest(changed) != base

    assert _writer_digest(_writer_items(), model="m2") != base
    assert _writer_digest(_writer_items(), granularity="session") != base

    adaptive_plan = ingest._turn_plan(
        _Adapter(), _item(), NS, "adaptive", writer_items=_writer_items()
    )
    assert ingest._turn_plan_digest(adaptive_plan, "adaptive") != base


# 7 ---------------------------------------------------------------------------------------------


class _Client:
    def __init__(self, turn_id: str | None = "te-1") -> None:
        self.turn_id = turn_id
        self.evidence: list[dict] = []
        self.episodes: list[dict] = []

    def record_turn_evidence(self, namespace, text, **kwargs):
        self.evidence.append({"namespace": namespace, "text": text, **kwargs})
        return {"turn_id": self.turn_id} if self.turn_id else {}

    def ingest(self, namespace, role, content, **kwargs):
        self.episodes.append({"namespace": namespace, "role": role, "content": content, **kwargs})
        return {"episode_id": f"ep-{len(self.episodes)}"}


_TK = f"{NS}|{SID_A}|t1"


def _call_writer(client, *, evidence_cache=None, evidence_text=ASSISTANT_A):
    return ingest._ingest_turn(
        client, NS, "agent", "On 2023/04/10, the assistant recommended X.",
        occurred_at="2023-04-10T14:47:00+00:00", session_id=SID_A, extract=True,
        evidence_text=evidence_text, turn_key=_TK, evidence_cache=evidence_cache,
        ingest_key=f"{_TK}#w0", source="agent-writer", writer_index=0,
    )


def test_writer_turn_records_assistant_evidence_then_grounds_on_it() -> None:
    client = _Client("te-assistant")
    cache: dict[str, str] = {}

    assert _call_writer(client, evidence_cache=cache) == "ep-1"

    assert len(client.evidence) == 1
    ev = client.evidence[0]
    assert ev["text"] == ASSISTANT_A  # the assistant's words, never the writer item
    assert ev["role"] == "assistant" and ev["declarant"] == "assistant"
    assert ev["turn_key"] == _TK
    assert cache == {_TK: "te-assistant"}
    (ep,) = client.episodes
    assert ep["role"] == "agent"
    assert ep["source"] == "agent-writer"
    assert ep["memory_role"] == "agent"
    assert ep["turn_evidence_uuid"] == "te-assistant"
    assert ep["idempotency_key"] == f"{_TK}#w0"


def test_writer_turn_reuses_cached_evidence() -> None:
    client = _Client("te-new")

    _call_writer(client, evidence_cache={_TK: "te-cached"})

    assert client.evidence == []
    assert client.episodes[0]["turn_evidence_uuid"] == "te-cached"


def test_writer_turn_without_turn_id_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "REQUIRE_TURN_EVIDENCE", True)
    client = _Client(None)

    with pytest.raises(RuntimeError, match="no turn_id"):
        _call_writer(client, evidence_cache={})
    assert client.episodes == []


@pytest.mark.parametrize("evidence_text", ["", None])
def test_writer_turn_without_evidence_text_is_rejected(evidence_text) -> None:
    client = _Client()

    with pytest.raises(ValueError):
        _call_writer(client, evidence_cache={}, evidence_text=evidence_text)
    assert client.evidence == [] and client.episodes == []


def test_non_writer_assistant_turn_has_no_memory_role_and_no_source() -> None:
    client = _Client()

    ingest._ingest_turn(
        client, NS, "assistant", ASSISTANT_A, session_id=SID_A, turn_key=_TK,
        ingest_key=f"{_TK}#s0",
    )

    (ep,) = client.episodes
    assert "memory_role" not in ep
    assert ep["source"] is None
    assert client.evidence == []


# 8 ---------------------------------------------------------------------------------------------


class _Flex:
    """Stands in for ``run_flex``; records requests and answers every one with one item."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls: list[tuple[list[dict], dict]] = []

    def __call__(self, requests, **kwargs):
        self.calls.append((requests, kwargs))
        if self.raises is not None:
            raise self.raises
        results = {
            r["custom_id"]: SimpleNamespace(
                error=None,
                content=json.dumps({"items": [{"text": "On 2023/04/10, the assistant recommended X."}]}),
                prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
            )
            for r in requests
        }
        return results, {}


def _drive_main(monkeypatch, tmp_path, flex: _Flex, extra: list[str], *, api_key: str | None = "sk-test"):
    client = _Client("te-main")
    sequence: list[str] = []

    def record_turn_evidence(namespace, text, **kwargs):
        sequence.append(f"ev:{kwargs['role']}")
        client.evidence.append({"namespace": namespace, "text": text, **kwargs})
        return {"turn_id": f"te-{len(client.evidence)}-{kwargs['role']}"}

    client.record_turn_evidence = record_turn_evidence  # type: ignore[method-assign]
    for name in ("OPENAI_API_KEY", "LME_WRITER_MAX_USD", "LME_WRITER_RATES"):
        monkeypatch.delenv(name, raising=False)
    if api_key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)

    import archolith_bench.core.openai_flex as openai_flex

    monkeypatch.setattr(openai_flex, "run_flex", flex)
    monkeypatch.setattr(ingest, "LongMemEvalMemoryAdapter", _Adapter)
    monkeypatch.setattr(ingest, "HttpMenhirClient", lambda url: client)
    monkeypatch.setattr(ingest.httpx, "Client", lambda **kwargs: object())
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(ingest, "REQUIRE_TURN_EVIDENCE", True)
    monkeypatch.setattr(ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"})
    monkeypatch.setattr(
        ingest, "_drain_many",
        lambda namespaces, admin, url, **kw: {
            n: {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "llm_tasks": 1,
                "processing_attempts": 1, "total": 1, "timed_out": False}
            for n in namespaces
        },
    )
    monkeypatch.setattr(
        ingest, "_integrity_counts",
        lambda namespace, submitted: {"cross_namespace_links": 0, "user_turns_admitted": 1},
    )
    monkeypatch.setattr(ingest, "_check_menhir_health", lambda admin, url: True)
    monkeypatch.setattr(
        ingest, "_retry_failed_evidence_projections",
        lambda namespaces, admin, url, **kwargs: {ns: 0 for ns in namespaces},
    )
    counts: list[int] = []

    def counts_many(namespaces):
        counts.append(1)
        total = 0 if len(counts) == 1 else 1
        return {
            n: {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "llm_tasks": 1,
                "processing_attempts": 1, "total": total}
            for n in namespaces
        }

    monkeypatch.setattr(ingest, "_ns_state_counts_many", counts_many)
    monkeypatch.setattr(ingest, "_reset_namespace", lambda *a, **k: None)
    monkeypatch.setattr(ingest, "_await_stale_episode_settlement", lambda *a, **k: None)

    result = ingest.main(
        ["--limit", "1", "--segmentation", "adaptive-writer",
         "--manifest", str(tmp_path / "manifest.json"), *extra]
    )
    return result, client


_GOOD = ["--writer-max-usd", "5", "--writer-rates", "1,2"]


@pytest.mark.parametrize(
    "extra,api_key",
    [
        (_GOOD, None),  # no OPENAI_API_KEY
        (["--writer-rates", "1,2"], "sk-test"),  # no cap
        (["--writer-max-usd", "0", "--writer-rates", "1,2"], "sk-test"),  # zero cap
        (["--writer-max-usd", "5", "--writer-rates", "abc"], "sk-test"),  # malformed rates
        (["--writer-max-usd", "5", "--writer-rates", "1,2,3"], "sk-test"),  # wrong arity
        (["--writer-max-usd", "5"], "sk-test"),  # no rates
    ],
)
def test_main_writer_refuses_without_key_budget_or_rates(monkeypatch, tmp_path, extra, api_key) -> None:
    flex = _Flex()

    result, client = _drive_main(monkeypatch, tmp_path, flex, extra, api_key=api_key)

    assert result == 2
    assert flex.calls == []
    assert client.episodes == [] and client.evidence == []


def test_main_writer_flex_failure_stops_before_any_ingest(monkeypatch, tmp_path) -> None:
    from archolith_bench.core.openai_flex import FlexError

    flex = _Flex(raises=FlexError("boom"))

    result, client = _drive_main(monkeypatch, tmp_path, flex, _GOOD)

    assert result == 1
    assert len(flex.calls) == 1
    assert client.episodes == [] and client.evidence == []


def test_main_writer_happy_path_ingests_agent_items_grounded_on_assistant_evidence(
    monkeypatch, tmp_path
) -> None:
    flex = _Flex()

    result, client = _drive_main(monkeypatch, tmp_path, flex, _GOOD)

    assert result == 0
    (requests, kwargs), = flex.calls
    # One turn-granularity request per assistant turn (3 in the fixture).
    assert len(requests) == 3
    assert kwargs["max_usd"] == 5.0 and kwargs["rates"] == (1.0, 2.0)

    agent_eps = [e for e in client.episodes if e.get("memory_role") == "agent"]
    assert len(agent_eps) == 3
    assert all(e["source"] == "agent-writer" and e["role"] == "agent" for e in agent_eps)

    assistant_te = {
        e["text"]: f"te-{i}-assistant"
        for i, e in enumerate(client.evidence, start=1)
        if e["role"] == "assistant"
    }
    assert set(assistant_te) == {ASSISTANT_A, ASSISTANT_B, ASSISTANT_C}
    assert {e["turn_evidence_uuid"] for e in agent_eps} == set(assistant_te.values())

    # Assistant words never reach extraction; only the writer's agent items do.
    sent = {e["content"] for e in client.episodes}
    assert not sent & {ASSISTANT_A, ASSISTANT_B, ASSISTANT_C}
    assert not [e for e in client.episodes if e["role"] == "assistant"]
