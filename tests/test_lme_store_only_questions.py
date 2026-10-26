"""CONTEXT_ONLY user turns as store-only source memories (archolith-bench#43).

A user's question still states facts about them ("what should I cook in my Instant Pot"). With
LME_STORE_ONLY_CONTEXT on, such a turn becomes an episode Menhir embeds but never extracts; with
it off, the turn stays evidence-only exactly as before.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
LME = ROOT / "scripts" / "longmemeval"


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_store_only_test", "scripts/longmemeval/lib/ingest.py")

from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402

QUESTION = "Thanks! What do you think?"
FACT = "I moved to Portland last month."
ASSISTANT_PHATIC = "That's a great question!"


class _Adapter:
    def sessions(self, item):
        return item["sessions"]


def _item() -> dict:
    return {"sessions": [[
        {"role": "assistant", "content": ASSISTANT_PHATIC},
        {"role": "user", "content": QUESTION},
        {"role": "user", "content": FACT},
    ]]}


def _turns(monkeypatch: pytest.MonkeyPatch, enabled: bool, segmentation: str = "adaptive"):
    monkeypatch.setattr(ingest, "STORE_ONLY_CONTEXT", enabled)
    return list(ingest._iter_item_turns(_Adapter(), _item(), "ns-a", segmentation, writer_items={}))


def test_fixture_turns_are_context_only() -> None:
    # The premise: both short turns are CONTEXT_ONLY, the fact is not.
    assert ingest.decide_segmentation("user", QUESTION) == ingest.SegmentationMode.CONTEXT_ONLY
    assert ingest.decide_segmentation("assistant", ASSISTANT_PHATIC) == ingest.SegmentationMode.CONTEXT_ONLY
    assert ingest.decide_segmentation("user", FACT) != ingest.SegmentationMode.CONTEXT_ONLY


def test_on_a_user_question_is_a_store_only_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _turns(monkeypatch, True)
    assert [(t.role, t.content, t.extract, t.store_only) for t in turns] == [
        ("assistant", ASSISTANT_PHATIC, False, False),
        ("user", QUESTION, True, True),
        ("user", FACT, True, False),
    ]


def test_off_keeps_the_evidence_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _turns(monkeypatch, False)
    assert [(t.extract, t.store_only) for t in turns] == [(False, False), (False, False), (True, False)]


def test_writer_mode_assistant_turns_stay_evidence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _turns(monkeypatch, True, segmentation="adaptive-writer")
    assistant = [t for t in turns if t.role == "assistant"]
    assert assistant and all(not t.extract and not t.store_only for t in assistant)
    assert [t.store_only for t in turns if t.content == QUESTION] == [True]


def test_non_adaptive_segmentations_never_mark_store_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for segmentation in ("none", "sentence", "user-sentence"):
        assert not any(t.store_only for t in _turns(monkeypatch, True, segmentation))


def test_store_only_turns_count_as_submitted_and_keep_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _turns(monkeypatch, True)
    # Menhir admits a store-only episode on its evidence, so it counts toward the admitted check.
    assert ingest._user_turns_submitted(iter(turns)) == 2
    plan = ingest._turn_plan(_Adapter(), _item(), "ns-a", "adaptive")
    assert len({ingest._ingest_key(t) for t in plan if t.extract}) == sum(t.extract for t in plan)


def test_digest_changes_only_when_a_turn_is_store_only(monkeypatch: pytest.MonkeyPatch) -> None:
    on = _turns(monkeypatch, True)
    off = _turns(monkeypatch, False)
    # A plan with no store-only turn hashes exactly as before the field existed (v1 rows).
    h = lambda text: ingest.hashlib.sha256(text.encode("utf-8")).hexdigest()  # noqa: E731
    legacy = {"v": 1, "segmentation": "adaptive", "split_min_length": ingest._SPLIT_MIN_LENGTH,
              "turns": [[t.turn_key, t.segment_index, t.extract, t.role, t.occurred_at,
                         t.session_id, h(t.content), h(t.evidence_text)] for t in off]}
    assert ingest._turn_plan_digest(off, "adaptive") == h(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")))
    assert ingest._turn_plan_digest(on, "adaptive") != ingest._turn_plan_digest(off, "adaptive")
    flipped = [dataclasses.replace(t, store_only=False) if t.store_only else t for t in on]
    assert ingest._turn_plan_digest(on, "adaptive") != ingest._turn_plan_digest(flipped, "adaptive")


class _Client:
    def __init__(self) -> None:
        self.evidence: list[dict] = []
        self.episodes: list[dict] = []

    def record_turn_evidence(self, namespace, text, **kwargs):
        self.evidence.append({"text": text, **kwargs})
        return {"turn_id": f"turn-{len(self.evidence)}"}

    def ingest(self, namespace, role, content, **kwargs):
        self.episodes.append({"content": content, **kwargs})
        return {"episode_id": f"episode-{len(self.episodes)}"}


def test_store_only_turn_is_grounded_user_evidence_and_a_store_only_episode() -> None:
    client = _Client()
    uuid = ingest._ingest_turn(client, "ns-a", "user", QUESTION, session_id="s0",
                               turn_key="ns-a|s0|t1", ingest_key="ns-a|s0|t1#s0", store_only=True)
    assert uuid == "episode-1"
    assert [(e["text"], e["role"], e["declarant"]) for e in client.evidence] == [(QUESTION, "user", "user")]
    [episode] = client.episodes
    assert episode["store_only"] is True
    assert episode["source"] == "user"
    assert episode["turn_evidence_uuid"] == "turn-1"
    assert episode["idempotency_key"] == "ns-a|s0|t1#s0"


def test_an_ordinary_turn_sends_no_store_only_field() -> None:
    client = _Client()
    ingest._ingest_turn(client, "ns-a", "user", FACT, session_id="s0", turn_key="ns-a|s0|t2")
    assert "store_only" not in client.episodes[0]


def test_pool_sends_the_plan_store_only_flag_to_menhir(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = _turns(monkeypatch, True)
    monkeypatch.setattr(ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"})
    monkeypatch.setattr(ingest, "_ns_state_counts_many", lambda namespaces: {
        ns: {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "llm_tasks": 0,
             "processing_attempts": 0, "total": 1} for ns in namespaces})
    monkeypatch.setattr(ingest, "_retry_failed_evidence_projections",
                        lambda namespaces, admin, url, **kwargs: {ns: 0 for ns in namespaces})
    client = _Client()
    run = ingest.NamespaceRun(item={}, question_id="a", namespace="ns-a", plan=turns, digest="d")
    state = ingest.IngestStateLog(Path(tempfile.mkdtemp()) / "state.jsonl")
    ingest._ingest_pool([run], client, object(), "http://menhir", state, pool_size=1,
                        drain_timeout=60, finalize=lambda run, counts: None, poll_s=0)

    # Assistant phatic turn: evidence only. Question: store-only episode. Fact: ordinary episode.
    assert [(e["content"], e.get("store_only")) for e in client.episodes] == [
        (QUESTION, True), (FACT, None)]
    assert [e["text"] for e in client.evidence] == [ASSISTANT_PHATIC, QUESTION, FACT]


def _http_client(captured: list[dict]) -> HttpMenhirClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"episode_id": "e1"})

    client = HttpMenhirClient("http://menhir.test")
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_http_client_sends_store_only_only_when_set() -> None:
    captured: list[dict] = []
    client = _http_client(captured)
    client.ingest("ns", "user", QUESTION, store_only=True, wait=False)
    client.ingest("ns", "user", FACT, wait=False)
    assert captured[0]["store_only"] is True
    assert "store_only" not in captured[1]


def _queries(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    queries: list[str] = []

    def fake_counts(query: str, width: int) -> list[int]:
        queries.append(query)
        return [0] * width

    monkeypatch.setattr(ingest, "_cypher_counts", fake_counts)
    ingest._integrity_counts("lme-x", [])
    return queries


def test_integrity_exempts_store_only_from_projection_and_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    provenance, _unfounded, _valid_at, time_unpaired, _cross = _queries(monkeypatch)
    # Admission still counts a store-only admitter; only the projection check skips it.
    admitted = re.search(r"EXISTS \{[^}]*\} AS admitted", provenance).group(0)
    assert "store_only" not in admitted
    assert "coalesce(q.extraction_mode, '') <> 'store_only' } AS extracted" in provenance
    assert "WHEN extracted AND NOT projected" in provenance
    assert "coalesce(q.extraction_mode, '') <> 'store_only'" in time_unpaired


def _config_value(name: str) -> str:
    config = (LME / "config.sh").read_text(encoding="utf-8")
    match = re.search(rf'^{name}="\$\{{{name}:-([^}}]*)\}}"', config, re.MULTILINE)
    assert match, f"config.sh does not default {name}"
    return match.group(1)


def test_build_defaults_turn_both_on() -> None:
    assert _config_value("LME_STORE_ONLY_CONTEXT") == "1"
    assert _config_value("LME_EMBED_BACKFILL") == "1"


def test_build_exports_the_toggle_and_gates_after_the_ingest() -> None:
    build = (LME / "build_graph.sh").read_text(encoding="utf-8")
    assert 'export LME_STORE_ONLY_CONTEXT="${LME_STORE_ONLY_CONTEXT}"' in build
    ingest_at = build.index('/lib/ingest.py" "${INGEST_ARGS[@]}"')
    backfill_at = build.index('scripts/backfill_episode_embeddings.py" --require-complete')
    promote_at = build.index("promote_persistent.sh\"\n")
    assert ingest_at < backfill_at < promote_at
    # The serving tree's code embeds, against the NEO4J_* exported for this build.
    assert '"${MENHIR_FRONTIER_PY}" "${MENHIR_FRONTIER}/scripts/backfill_episode_embeddings.py"' in build
    assert build.index('export NEO4J_URI="bolt://localhost:${LME_BOLT}"') < backfill_at


def test_build_refuses_a_serving_tree_without_the_menhir_support_before_paid_work() -> None:
    build = (LME / "build_graph.sh").read_text(encoding="utf-8")
    start_at = build.index('"${MENHIR_FRONTIER_BIN}" serve')
    for needle in ('has no store_only ingest', 'has no backfill --require-complete'):
        assert build.index(needle) < start_at
    assert '--setting "store_only_context=${LME_STORE_ONLY_CONTEXT}"' in build
    assert '--setting "embed_backfill=${LME_EMBED_BACKFILL}"' in build
