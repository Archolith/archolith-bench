"""Tests for the persistent LongMemEval ingestion regression fixture."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "longmemeval" / "menhir_suburbs_extraction_regression.json"


def _load_script(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_script("lme_ingest_fixture_test", "scripts/longmemeval/lib/ingest.py")
verify = _load_script(
    "lme_verify_suburbs_fixture_test",
    "scripts/longmemeval/lib/verify_suburbs_fixture.py",
)


def test_ingest_defaults_resolve_from_repository_root() -> None:
    assert ingest.BENCH_ROOT == ROOT
    assert Path(ingest.DEFAULT_MANIFEST) == ROOT / "results" / "lme-ingest" / "manifest.json"


def test_adaptive_ingest_keeps_gps_update_without_unbound_duplicate() -> None:
    """A pronoun-only claim fragment loses the full turn's dealership/GPS context."""
    source = (
        "I had an issue with my car's GPS system and took it to the dealership. "
        "They replaced the entire system, and now it's working flawlessly. "
        "Have you heard of common GPS issues in newer cars?"
    )
    item = {
        "haystack_session_ids": ["answer_gps_1"],
        "haystack_dates": ["2023/04/10 (Mon) 14:47"],
        "haystack_sessions": [[{"role": "user", "content": source}]],
    }
    adapter = SimpleNamespace(sessions=lambda record: record["haystack_sessions"])

    turns = list(ingest._iter_item_turns(adapter, item, "lme-gps", "adaptive"))

    assert [turn.content for turn in turns] == [source]
    assert turns[0].evidence_text == source
    assert turns[0].role == "user"


def test_ingest_parser_accepts_fixture_and_namespace_prefix() -> None:
    args = ingest._parse_args(
        [
            "--fixture",
            str(FIXTURE),
            "--namespace-prefix",
            "fixture-",
            "--limit",
            "1",
        ]
    )
    assert args.fixture == str(FIXTURE)
    assert args.namespace_prefix == "fixture-"
    assert args.limit == 1


def test_ingest_parser_accepts_namespace_window() -> None:
    args = ingest._parse_args(
        ["--namespace-window", "2", "--manifest-item-limit", "2"]
    )
    assert args.namespace_window == 2
    assert args.manifest_item_limit == 2

    with pytest.raises(SystemExit):
        ingest._parse_args(["--namespace-window", "0"])
    with pytest.raises(SystemExit):
        ingest._parse_args(["--manifest-item-limit", "0"])


def test_ingest_parser_accepts_scalar_consolidation_controls() -> None:
    args = ingest._parse_args(
        [
            "--consolidate-scalar",
            "--consolidation-k",
            "3",
            "--consolidation-call-budget",
            "40",
        ]
    )
    assert args.consolidate_scalar is True
    assert args.consolidation_k == 3
    assert args.consolidation_call_budget == 40


def test_current_scalar_buildout_wrapper_is_fail_closed() -> None:
    script = (
        ROOT / "scripts" / "longmemeval" / "run_knowledge_update_buildout.sh"
    ).read_text(encoding="utf-8")

    assert 'LME_KU_RUN_ID is required' in script
    assert "LME_KU_ARM must be exactly 'baseline' or 'candidate'" in script
    assert 'export LME_REQUIRE_FRESH="1"' in script
    assert 'export LME_SCALAR_STATE_ENABLED="1"' in script
    assert 'export LME_SCALAR_HISTORY_ENABLED="1"' in script
    assert '[ "${LME_SCALAR_HISTORY_ENABLED}" = "1" ]' in script
    assert 'export LME_REQUIRE_TURN_EVIDENCE="1"' in script
    assert 'LME_KU_ALLOW_DIRTY' in script
    assert 'status --porcelain --untracked-files=no' in script
    assert '"fixture_sha256": "${FIXTURE_SHA256}"' in script
    assert 'FIXTURE_SHA256="${FIXTURE_SHA256//$\'\\r\'/}"' in script
    assert '"consolidation_audit_enabled": ${LME_CONSOLIDATION_AUDIT_ENABLED}' in script
    assert '"recall_audit_enabled": ${LME_RECALL_AUDIT_ENABLED}' in script
    assert 'LME_KU_INGEST_CONCURRENCY:-2' in script
    assert 'LME_KU_CHECKPOINT_ITEMS:-0' in script
    assert 'while [ ! -f "${CHECKPOINT_CONTINUE}" ]' in script
    assert '"ingest_concurrency": ${LME_INGEST_CONCURRENCY}' in script
    assert '"checkpoint_items": ${LME_KU_CHECKPOINT_ITEMS}' in script
    assert (
        'MENHIR_PERSONAL_MEMORY_SCALAR_VIEW_AUTHORITY_ENABLED="${LME_SCALAR_VIEW_AUTHORITY_ENABLED}"'
        in script
    )
    assert (
        'MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED}"'
        in script
    )
    assert 'scalar_history=${LME_SCALAR_HISTORY_ENABLED}' in script
    assert '--preflight-only' in script
    assert '[ "${LME_NONCANONICAL}" = "1" ] && PROVENANCE_BEGIN_ARGS+=("--noncanonical")' in script
    assert '"${PROVENANCE_BEGIN_ARGS[@]}"' in script


def test_build_graph_wires_same_namespace_window_into_menhir_workers() -> None:
    script = (ROOT / "scripts" / "longmemeval" / "build_graph.sh").read_text(encoding="utf-8")

    assert 'export MENHIR_INGEST_CONCURRENCY="${LME_INGEST_CONCURRENCY}"' in script
    assert '--namespace-window "${LME_INGEST_CONCURRENCY}"' in script
    assert '--manifest-item-limit "${LME_INGEST_STOP_AFTER_ITEMS}"' in script
    assert '"ingest_target_items": ${INGEST_TARGET}' in script


def test_build_graph_supports_explicit_local_extraction_provider() -> None:
    config = (ROOT / "scripts" / "longmemeval" / "config.sh").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "longmemeval" / "build_graph.sh").read_text(encoding="utf-8")

    assert 'LME_EXTRACT_PROVIDER="${LME_EXTRACT_PROVIDER:-openai}"' in config
    assert 'LME_EXTRACT_BASE_URL="${LME_EXTRACT_BASE_URL:-}"' in config
    assert 'case "${LME_EXTRACT_PROVIDER}" in' in script
    assert 'export LOCAL_LLM_BASE_URL="${LME_EXTRACT_BASE_URL}"' in script
    assert 'export LOCAL_LLM_CHAT_MODEL="${LME_EXTRACT_MODEL}"' in script
    assert 'export GRAPHITI_EMBED_PROVIDER="openai"' in script
    assert '"ingest_concurrency": ${LME_INGEST_CONCURRENCY}' in script
    assert (
        'export MENHIR_PERSONAL_MEMORY_SCALAR_HISTORY_ENABLED="${LME_SCALAR_HISTORY_ENABLED}"'
        in script
    )
    assert '"scalar_history_enabled": ${LME_SCALAR_HISTORY_ENABLED}' in script
    assert '"scalar_history_views"' in script


def test_longmemeval_container_guards_do_not_use_pipefail_sensitive_grep() -> None:
    config = (ROOT / "scripts" / "longmemeval" / "config.sh").read_text(encoding="utf-8")
    assert "lme_container_exists()" in config
    assert "lme_container_running()" in config
    assert "docker container inspect" in config

    for relative_path in (
        "build_graph.sh",
        "promote_persistent.sh",
        "recall_ab.sh",
        "retry.sh",
        "run_knowledge_update_buildout.sh",
        "run_ku_adaptive.sh",
        "run_ku_adaptive_full.sh",
        "run_ku_nosplit.sh",
        "run_ku_nosplit_full.sh",
        "run_ku_split15.sh",
        "run_suburbs_fixture.sh",
    ):
        script = (ROOT / "scripts" / "longmemeval" / relative_path).read_text(
            encoding="utf-8"
        )
        assert "docker ps --format '{{.Names}}' | grep -" not in script
        assert "docker ps -a --format '{{.Names}}' | grep -" not in script


def test_buildout_wrapper_records_every_phase_with_effective_settings() -> None:
    """Provenance must describe the phases that ran, not the intent captured at preflight."""
    script = (
        ROOT / "scripts" / "longmemeval" / "run_knowledge_update_buildout.sh"
    ).read_text(encoding="utf-8")

    for phase in ("build-graph", "build-graph-checkpoint-continuation", "recall-qa"):
        assert f"record_phase_start {phase}" in script
        assert f"record_phase_end {phase}" in script

    # The continuation flips both of these, so a single preflight snapshot describes neither
    # build phase. They are read again per phase.
    assert "require_fresh=%s" in script
    assert "ingest_stop_after_items=%s" in script
    # An arm is not defined by its threshold alone.
    for setting in (
        "segmentation=%s",
        "strict_failure_policy=%s",
        "scalar_consolidation_k=%s",
        "scalar_consolidation_call_budget=%s",
        "scalar_reconcile_attribute=%s",
        "scalar_history_enabled=%s",
        "event_history_enabled=%s",
        "event_history_authority_enabled=%s",
        "scalar_deterministic_router=%s",
        "scalar_deterministic_shadow=%s",
        "consolidation_audit_enabled=%s",
        "turn_evidence_required=%s",
    ):
        assert setting in script
    # Each phase records the manifest it inherited.
    assert '--manifest "${LME_MANIFEST_PATH}"' in script


def test_provenance_is_appended_never_overwritten() -> None:
    """Neither wrapper may `cat >` its provenance file: a resume would erase the real history."""
    buildout = (
        ROOT / "scripts" / "longmemeval" / "run_knowledge_update_buildout.sh"
    ).read_text(encoding="utf-8")
    build_graph = (ROOT / "scripts" / "longmemeval" / "build_graph.sh").read_text(encoding="utf-8")

    assert 'cat > "${LME_RESULTS_DIR}/run_provenance.json"' not in buildout
    assert 'cat > "${GRAPH_PROVENANCE_PATH}"' not in build_graph
    for script in (buildout, build_graph):
        assert "run_provenance.py" in script
        assert "begin" in script
    assert 'phase-start "${GRAPH_PROVENANCE_PATH}"' in build_graph
    assert 'phase-end "${GRAPH_PROVENANCE_PATH}"' in build_graph


def test_fixture_path_is_forwarded_to_adapter() -> None:
    calls: list[dict] = []

    class FakeAdapter:
        def load_items(self, **kwargs):
            calls.append(kwargs)
            return [{"question_id": "one"}]

    items = ingest._load_items(FakeAdapter(), limit=1, fixture_path=str(FIXTURE))
    assert items == [{"question_id": "one"}]
    assert calls == [{"limit": 1, "fixture_path": str(FIXTURE)}]


def test_empty_fixture_is_rejected() -> None:
    class EmptyAdapter:
        def load_items(self, **kwargs):
            return []

    with pytest.raises(ValueError, match="fixture is empty"):
        ingest._load_items(EmptyAdapter(), limit=1, fixture_path=str(FIXTURE))


def test_namespace_prefix_is_explicit_and_nonempty() -> None:
    assert ingest._namespace("830ce83f", "fresh-") == "fresh-830ce83f"
    with pytest.raises(ValueError, match="must not be empty"):
        ingest._namespace("830ce83f", "  ")


def test_item_turn_generation_preserves_namespace_session_order() -> None:
    class FakeAdapter:
        def sessions(self, item):
            return [item["turns"]]

    adapter = FakeAdapter()
    first = {
        "turns": [
            {"role": "user", "content": "a1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "a3"},
        ],
    }
    scheduled = [
        (turn.content, turn.session_id)
        for turn in ingest._iter_item_turns(adapter, first, "ns-a", "none")
    ]

    assert scheduled == [
        ("a1", "ns-a-s0"),
        ("a2", "ns-a-s0"),
        ("a3", "ns-a-s0"),
    ]


def test_drain_many_waits_for_global_queue_and_every_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_depths = iter([1, 0, 0])
    states = {
        "ns-a": iter(
            [
                {"pending": 1, "ready": 0, "enriching": 0, "failed": 0, "total": 1},
                {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "total": 1},
                {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "total": 1},
            ]
        ),
        "ns-b": iter(
            [
                {"pending": 0, "ready": 0, "enriching": 1, "failed": 0, "total": 1},
                {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "total": 1},
                {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "total": 1},
            ]
        ),
    }

    monkeypatch.setattr(ingest, "_queue_depth", lambda admin, url: next(queue_depths))
    monkeypatch.setattr(ingest, "_ns_state_counts", lambda namespace: next(states[namespace]))
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    drained = ingest._drain_many(
        ["ns-a", "ns-b"],
        object(),
        "http://menhir",
        poll_s=0,
        timeout_s=1,
    )

    assert drained["ns-a"]["ready"] == 1
    assert drained["ns-b"]["ready"] == 1
    assert drained["ns-a"]["timed_out"] is False
    assert drained["ns-b"]["timed_out"] is False


def test_namespace_state_counts_include_cost_and_retry_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest,
        "_cypher",
        lambda query: [["0", "20", "0", "0", "270", "24", "20"]],
    )

    assert ingest._ns_state_counts("ns-a") == {
        "pending": 0,
        "ready": 20,
        "enriching": 0,
        "failed": 0,
        "llm_tasks": 270,
        "processing_attempts": 24,
        "total": 20,
    }


def test_scalar_counts_include_state_history_and_turn_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []

    def fake_count(query: str) -> int:
        queries.append(query)
        return len(queries)

    monkeypatch.setattr(ingest, "_cypher_count", fake_count)

    counts = ingest._scalar_counts("ns-a")

    assert counts == {
        "turn_evidence": 1,
        "typed_assertions": 2,
        "scalar_views": 3,
        "scalar_history_views": 4,
        "user_founded_scalar_views": 5,
    }
    assert "view_kind:'scalar_state'" in queries[2]
    assert "view_kind:'scalar_history'" in queries[3]


def test_reset_namespace_force_deletes_graph_and_purges_turn_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_calls: list[tuple[str, dict]] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeAdmin:
        def __init__(self):
            self.posts: list[dict] = []

        def post(self, url, **kwargs):
            self.posts.append({"url": url, **kwargs})
            return FakeResponse()

    def fake_backend(admin, url, operation, body):
        backend_calls.append((operation, body))
        return {"namespace": body["namespace"], "deleted": 250}

    monkeypatch.setattr(ingest, "_backend", fake_backend)
    admin = FakeAdmin()

    ingest._reset_namespace(admin, "http://menhir/", "ns-a")

    assert backend_calls == [
        ("delete_namespace", {"namespace": "ns-a", "force": True}),
    ]
    assert admin.posts == [
        {
            "url": "http://menhir/api/phase3/reset",
            "params": {"namespace": "ns-a"},
            "timeout": 300.0,
        }
    ]


def test_ingest_window_refills_only_after_that_namespace_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = [
        ingest.WindowItem(item={}, question_id="a", namespace="ns-a"),
        ingest.WindowItem(item={}, question_id="b", namespace="ns-b"),
    ]
    turns = [
        iter([
            ingest.IngestTurn("user", "a1", None, "ns-a-s0"),
            ingest.IngestTurn("user", "a2", None, "ns-a-s0"),
        ]),
        iter([
            ingest.IngestTurn("user", "b1", None, "ns-b-s0"),
            ingest.IngestTurn("user", "b2", None, "ns-b-s0"),
        ]),
    ]
    submitted: list[str] = []
    lifecycle = {
        "episode-1": iter(["PENDING", "READY"]),
        "episode-2": iter(["READY"]),
        "episode-3": iter(["READY"]),
        "episode-4": iter(["READY"]),
    }

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        submitted.append(content)
        return f"episode-{len(submitted)}"

    def fake_processing(admin, url, episode_uuid):
        return {"processing_state": next(lifecycle[episode_uuid])}

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(ingest, "_episode_processing", fake_processing)

    requeued = ingest._ingest_window(
        window,
        turns,
        object(),
        object(),
        "http://menhir",
        timeout_s=1,
        poll_s=0,
    )

    # b2 may refill while a1 is still pending, but a2 cannot be submitted until a1 is READY.
    assert submitted == ["a1", "b1", "b2", "a2"]
    assert requeued == {"ns-a": 0, "ns-b": 0}


def test_ingest_window_retries_failure_before_advancing_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = [ingest.WindowItem(item={}, question_id="a", namespace="ns-a")]
    turns = [iter([
        ingest.IngestTurn("user", "a1", None, "ns-a-s0"),
        ingest.IngestTurn("user", "a2", None, "ns-a-s0"),
    ])]
    submitted: list[str] = []
    operations: list[str] = []
    lifecycle = {
        "episode-1": iter(["FAILED", "READY"]),
        "episode-2": iter(["READY"]),
    }

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        submitted.append(content)
        return f"episode-{len(submitted)}"

    def fake_processing(admin, url, episode_uuid):
        return {"processing_state": next(lifecycle[episode_uuid])}

    def fake_backend(admin, url, operation, body):
        operations.append(operation)
        return True

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(ingest, "_episode_processing", fake_processing)
    monkeypatch.setattr(ingest, "_backend", fake_backend)

    requeued = ingest._ingest_window(
        window,
        turns,
        object(),
        object(),
        "http://menhir",
        timeout_s=1,
        poll_s=0,
    )

    assert submitted == ["a1", "a2"]
    assert operations == ["force_reset_failed_episode", "enqueue_pending_episode"]
    assert requeued == {"ns-a": 1}


def test_retry_failed_evidence_projections_enumerates_generated_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries: list[str] = []
    operations: list[tuple[str, dict]] = []

    def fake_cypher(query: str):
        queries.append(query)
        if "namespace:'ns-a'" in query:
            return [["projection-a"], ["projection-b"]]
        return []

    def fake_backend(admin, url, operation, body):
        operations.append((operation, body))
        return True

    monkeypatch.setattr(ingest, "_cypher", fake_cypher)
    monkeypatch.setattr(ingest, "_backend", fake_backend)

    requeued = ingest._retry_failed_evidence_projections(
        ["ns-a", "ns-b", "ns-a"],
        object(),
        "http://menhir",
    )

    assert requeued == {"ns-a": 2, "ns-b": 0}
    assert all("processing_state='FAILED'" in query for query in queries)
    assert all("coalesce(e.is_evidence_projection, false)" in query for query in queries)
    assert operations == [
        ("force_reset_failed_episode", {"episode_uuid": "projection-a"}),
        ("enqueue_pending_episode", {"episode_uuid": "projection-a"}),
        ("force_reset_failed_episode", {"episode_uuid": "projection-b"}),
        ("enqueue_pending_episode", {"episode_uuid": "projection-b"}),
    ]


def test_retry_failed_evidence_projection_fails_closed_when_enqueue_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest,
        "_failed_evidence_projection_uuids",
        lambda namespace: ["projection-a"],
    )
    responses = iter([True, False])
    monkeypatch.setattr(
        ingest,
        "_backend",
        lambda admin, url, operation, body: next(responses),
    )

    with pytest.raises(RuntimeError, match="projection-a.*ns-a"):
        ingest._retry_failed_evidence_projections(
            ["ns-a"],
            object(),
            "http://menhir",
        )


def test_failed_evidence_projection_retry_redrains_and_updates_manifest_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = {
        "ns-a": {"failed": 1, "timed_out": False},
        "ns-b": {"failed": 0, "timed_out": False},
    }
    settled = {
        "ns-a": {"failed": 0, "timed_out": False},
        "ns-b": {"failed": 0, "timed_out": False},
    }
    requeued = {"ns-a": 0, "ns-b": 0}
    drain_calls: list[list[str]] = []

    monkeypatch.setattr(
        ingest,
        "_retry_failed_evidence_projections",
        lambda namespaces, admin, url: {"ns-a": 1, "ns-b": 0},
    )

    def fake_drain(namespaces, admin, url, **kwargs):
        drain_calls.append(namespaces)
        return settled

    monkeypatch.setattr(ingest, "_drain_many", fake_drain)

    result = ingest._retry_failed_evidence_projections_after_drain(
        ["ns-a", "ns-b"],
        object(),
        "http://menhir",
        initial,
        requeued,
        timeout_s=60.0,
    )

    assert result is settled
    assert requeued == {"ns-a": 1, "ns-b": 0}
    assert drain_calls == [["ns-a", "ns-b"]]


def test_manifest_checkpoint_is_atomic(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    rows = [{"question_id": "one"}]

    ingest._write_manifest(manifest_path, rows)

    assert manifest_path.read_text(encoding="utf-8") == '[\n  {\n    "question_id": "one"\n  }\n]'
    assert not manifest_path.with_suffix(".json.tmp").exists()


def test_scalar_window_refuses_residual_failed_episodes() -> None:
    drained = {
        "ns-a": {"failed": 0},
        "ns-b": {"failed": 1},
    }

    with pytest.raises(RuntimeError, match="ns-b=1"):
        ingest._require_no_failed_episodes(drained)


def test_failed_gate_has_no_tolerance_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAILED episode means the namespace does not match the transcript. No env var buys past it."""
    monkeypatch.setenv("LME_KU_MAX_FAILURES_PER_NS", "5")

    with pytest.raises(RuntimeError, match="ns-b=1"):
        ingest._require_no_failed_episodes({"ns-a": {"failed": 0}, "ns-b": {"failed": 1}})


# ---------------------------------------------------------------------------
# Evidence-only routing for CONTEXT_ONLY turns
# ---------------------------------------------------------------------------

class _FakeSessionAdapter:
    def sessions(self, item):
        return item["sessions"]


def test_context_only_turns_keep_their_own_boundaries_and_role() -> None:
    """CONTEXT_ONLY turns are never merged into a neighbour.

    Merging concatenated two source turns into one record, which destroys the per-turn
    TurnEvidence boundaries the scalar audit counts and -- across a role change -- filed
    assistant words under the user's evidence.
    """
    item = {
        "sessions": [[
            {"role": "assistant", "content": "That's a great question!"},
            {"role": "user", "content": "Thanks! What do you think?"},
            {"role": "user", "content": "I moved to Portland last month."},
        ]],
    }

    turns = list(
        ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "adaptive")
    )

    # One emitted turn per source turn, content byte-identical, role preserved.
    assert [(turn.role, turn.content, turn.extract) for turn in turns] == [
        ("assistant", "That's a great question!", False),
        ("user", "Thanks! What do you think?", False),
        ("user", "I moved to Portland last month.", True),
    ]
    # No assistant text anywhere in a user turn.
    user_text = " ".join(turn.content for turn in turns if turn.role == "user")
    assert "That's a great question!" not in user_text


def test_lme_session_date_reaches_turn_evidence_as_world_time() -> None:
    item = {
        "sessions": [[{"role": "user", "content": "I have added 25 postcards."}]],
        "haystack_dates": ["2023/11/30 (Thu) 20:25"],
        "haystack_session_ids": ["answer-a-2"],
    }
    turn = next(iter(ingest._iter_item_turns(_FakeSessionAdapter(), item, "lme-a", "none")))
    evidence: list[dict] = []

    class FakeClient:
        def record_turn_evidence(self, namespace, text, **kwargs):
            evidence.append({"namespace": namespace, "text": text, **kwargs})
            return {"turn_id": "turn-25"}

        def ingest(self, *args, **kwargs):
            return {"episode_id": "episode-25"}

    ingest._ingest_turn(
        FakeClient(),
        "lme-a",
        turn.role,
        turn.content,
        occurred_at=turn.occurred_at,
        session_id=turn.session_id,
        turn_key=turn.turn_key,
    )

    assert turn.occurred_at == "2023-11-30T20:25:00+00:00"
    assert evidence[0]["occurred_at"] == turn.occurred_at


def test_evidence_only_turn_records_one_record_and_creates_no_episode() -> None:
    evidence: list[dict] = []
    episodes: list[dict] = []

    class FakeClient:
        def record_turn_evidence(self, namespace, text, **kwargs):
            evidence.append({"namespace": namespace, "text": text, **kwargs})
            return {"turn_id": f"turn-{len(evidence)}"}

        def ingest(self, *args, **kwargs):
            episodes.append({"args": args, "kwargs": kwargs})
            return {"episode_id": "episode-1"}

    episode_uuid = ingest._ingest_turn(
        FakeClient(),
        "ns-a",
        "assistant",
        "That's a great question!",
        session_id="ns-a-s0",
        extract=False,
    )

    assert episode_uuid == ""
    assert episodes == []
    assert len(evidence) == 1
    # An assistant turn is never captured as user evidence: menhir's evidence projection is
    # fail-closed on role='user' AND declarant='user', so this record stays inert.
    assert evidence[0]["role"] == "assistant"
    assert evidence[0]["declarant"] == "assistant"
    assert evidence[0]["text"] == "That's a great question!"


def test_required_evidence_without_turn_id_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Client:
        def record_turn_evidence(self, namespace, text, **kwargs):
            return {}

        def ingest(self, *args, **kwargs):
            raise AssertionError("an ungrounded turn must never be ingested")

    monkeypatch.setattr(ingest, "REQUIRE_TURN_EVIDENCE", True)

    with pytest.raises(RuntimeError, match="returned no turn_id"):
        ingest._ingest_turn(
            Client(),
            "ns",
            "user",
            "I moved to Seattle.",
            extract=False,
            turn_key="ns|session|t0",
        )


def test_extracting_user_turn_still_grounds_on_user_evidence() -> None:
    evidence: list[dict] = []
    episodes: list[dict] = []

    class FakeClient:
        def record_turn_evidence(self, namespace, text, **kwargs):
            evidence.append(kwargs)
            return {"turn_id": "turn-1"}

        def ingest(self, *args, **kwargs):
            episodes.append(kwargs)
            assert kwargs["turn_evidence_uuid"] == "turn-1"
            assert kwargs["source"] == "user"
            return {"episode_id": "episode-1"}

    assert ingest._ingest_turn(
        FakeClient(),
        "ns-a",
        "user",
        "I moved to Portland.",
        session_id="ns-a-s0",
        occurred_at="2023-11-30T20:25:00Z",
    ) == "episode-1"
    assert evidence[0]["declarant"] == "user"
    assert evidence[0]["occurred_at"] == "2023-11-30T20:25:00Z"
    assert episodes[0]["occurred_at"] == "2023-11-30T20:25:00Z"


class _RecordingClient:
    """Captures the evidence and episode calls a window makes."""

    def __init__(self) -> None:
        self.evidence: list[dict] = []
        self.episodes: list[dict] = []

    def record_turn_evidence(self, namespace, text, **kwargs):
        # Mirror menhir: MERGE on turn_key, so one key is one node.
        key = kwargs.get("turn_key")
        for existing in self.evidence:
            if key is not None and existing["turn_key"] == key:
                return {"turn_id": existing["turn_id"]}
        record = {
            "turn_id": f"evidence-{len(self.evidence) + 1}",
            "namespace": namespace,
            "text": text,
            **kwargs,
        }
        self.evidence.append(record)
        return {"turn_id": record["turn_id"]}

    def ingest(self, namespace, role, content, **kwargs):
        self.episodes.append({"content": content, **kwargs})
        return {"episode_id": f"episode-{len(self.episodes)}"}


def _run_window(client, turns: list) -> None:
    ingest._ingest_window(
        [ingest.WindowItem(item={}, question_id="a", namespace="ns-a")],
        [iter(turns)],
        client,
        object(),
        "http://menhir",
        timeout_s=1,
        poll_s=0,
    )


def test_repeated_identical_turns_stay_distinct_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Menhir's fallback turn_key hashes the TEXT, merging two genuine repeats onto one node.

    A user who states the same fact twice made two observations; collapsing them loses one.
    """
    monkeypatch.setattr(
        ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"}
    )
    client = _RecordingClient()
    item = {"sessions": [[
        {"role": "user", "content": "I have 20 coins."},
        {"role": "user", "content": "I have 20 coins."},
    ]]}
    turns = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "none"))

    assert turns[0].turn_key != turns[1].turn_key
    _run_window(client, turns)

    assert len(client.evidence) == 2
    assert len({record["turn_id"] for record in client.evidence}) == 2


def test_one_evidence_record_per_source_turn_across_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split turn is still ONE source turn: N episodes, one evidence node, whole-turn text."""
    monkeypatch.setattr(
        ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"}
    )
    client = _RecordingClient()
    original = (
        "I moved to Portland in March. My rent is now $2,400. "
        "I switched to the night shift at the hospital downtown."
    )
    item = {"sessions": [[{"role": "user", "content": original}]]}
    turns = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "sentence"))

    assert len(turns) > 1, "this turn must actually split for the test to mean anything"
    assert len({turn.turn_key for turn in turns}) == 1
    _run_window(client, turns)

    # One evidence node, holding the user's ACTUAL words -- not a fragment.
    assert len(client.evidence) == 1
    assert client.evidence[0]["text"] == original
    # Every episode cites that same evidence node.
    assert len(client.episodes) == len(turns)
    assert {episode["turn_evidence_uuid"] for episode in client.episodes} == {"evidence-1"}


def test_context_only_turn_is_one_evidence_and_no_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"}
    )
    client = _RecordingClient()
    item = {"sessions": [[{"role": "assistant", "content": "That's a great question!"}]]}
    turns = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "adaptive"))

    _run_window(client, turns)

    assert len(client.evidence) == 1
    assert client.evidence[0]["declarant"] == "assistant"
    assert client.episodes == []


def test_turn_key_is_stable_and_namespace_qualified() -> None:
    item = {"sessions": [[
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]]}
    first = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "none"))
    again = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-a", "none"))
    other = list(ingest._iter_item_turns(_FakeSessionAdapter(), item, "ns-b", "none"))

    # Stable across re-iteration (a resume must reuse, not duplicate, the same turn's evidence).
    assert [turn.turn_key for turn in first] == [turn.turn_key for turn in again]
    # Distinct per turn, and never shared between namespaces.
    assert len({turn.turn_key for turn in first}) == 2
    assert not {turn.turn_key for turn in first} & {turn.turn_key for turn in other}


def test_window_scheduler_does_not_wait_on_evidence_only_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evidence-only turn has no episode, so it must not occupy the active slot.

    If it did, the scheduler would poll forever for a lifecycle row that does not exist.
    """
    window = [ingest.WindowItem(item={}, question_id="a", namespace="ns-a")]
    turns = [iter([
        ingest.IngestTurn("user", "a1", None, "ns-a-s0", extract=False),
        ingest.IngestTurn("assistant", "a2", None, "ns-a-s0", extract=False),
        ingest.IngestTurn("user", "a3", None, "ns-a-s0", extract=True),
    ])]
    submitted: list[tuple[str, bool]] = []

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        extract = kwargs.get("extract", True)
        submitted.append((content, extract))
        return f"episode-{len(submitted)}" if extract else ""

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(
        ingest, "_episode_processing", lambda admin, url, uuid: {"processing_state": "READY"}
    )

    requeued = ingest._ingest_window(
        window, turns, object(), object(), "http://menhir", timeout_s=1, poll_s=0
    )

    assert submitted == [("a1", False), ("a2", False), ("a3", True)]
    assert requeued == {"ns-a": 0}
    # Every source turn is counted, extracted or not.
    assert window[0].turns == 3


# ---------------------------------------------------------------------------
# Stale-worker / reset ordering
# ---------------------------------------------------------------------------

def test_settlement_waits_for_stale_lifecycle_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([
        {"pending": 0, "enriching": 1},
        {"pending": 1, "enriching": 0},
        {"pending": 0, "enriching": 0},
    ])
    monkeypatch.setattr(ingest, "_ns_state_counts", lambda namespace: next(states))
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    ingest._await_stale_episode_settlement(["ns-a"], poll_s=0, timeout_s=30)

    assert next(states, "exhausted") == "exhausted"


def test_settlement_fails_closed_when_work_never_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timing out must raise, not reset anyway -- resetting under a live worker is the race."""
    monkeypatch.setattr(
        ingest, "_ns_state_counts", lambda namespace: {"pending": 0, "enriching": 2}
    )
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="did not settle"):
        ingest._await_stale_episode_settlement(["ns-a"], poll_s=0, timeout_s=0)


def test_settlement_fails_closed_when_state_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """-1 means cypher could not answer. Unreadable is not settled."""
    monkeypatch.setattr(
        ingest, "_ns_state_counts", lambda namespace: {"pending": -1, "enriching": -1}
    )
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="could not be read"):
        ingest._await_stale_episode_settlement(["ns-a"], poll_s=0, timeout_s=0)


def test_main_submits_a_namespace_window_round_robin_and_manifests_after_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        {
            "question_id": "a",
            "question": "Question A?",
            "answer": "A",
            "question_type": "knowledge-update",
            "sessions": [[
                {"role": "user", "content": "a1"},
                {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "a3"},
            ]],
        },
        {
            "question_id": "b",
            "question": "Question B?",
            "answer": "B",
            "question_type": "knowledge-update",
            "sessions": [[
                {"role": "user", "content": "b1"},
                {"role": "assistant", "content": "b2"},
            ]],
        },
        {
            "question_id": "c",
            "question": "Question C?",
            "answer": "C",
            "question_type": "knowledge-update",
            "sessions": [[
                {"role": "user", "content": "c1"},
            ]],
        },
    ]
    submitted: list[tuple[str, str]] = []
    reset: list[str] = []
    drain_calls: list[list[str]] = []
    ordering: list[str] = []

    class FakeAdapter:
        def load_items(self, **kwargs):
            return items

        def sessions(self, item):
            return item["sessions"]

        def question(self, item):
            return item["question"]

    class FakeClient:
        pass

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        submitted.append((namespace, content))
        return f"episode-{len(submitted)}"

    def fake_backend(admin, url, operation, body):
        assert operation == "fetch_episode_processing"
        return {"processing_state": "READY", "episode_uuid": body["episode_uuid"]}

    def fake_drain_many(namespaces, admin, url, **kwargs):
        drain_calls.append(namespaces)
        return {
            namespace: {
                "pending": 0,
                "ready": 2,
                "enriching": 0,
                "failed": 0,
                "llm_tasks": 10,
                "processing_attempts": 2,
                "total": 2,
                "timed_out": False,
            }
            for namespace in namespaces
        }

    monkeypatch.setattr(ingest, "LongMemEvalMemoryAdapter", FakeAdapter)
    monkeypatch.setattr(ingest, "HttpMenhirClient", lambda url: FakeClient())
    monkeypatch.setattr(ingest.httpx, "Client", lambda **kwargs: object())
    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(ingest, "_backend", fake_backend)
    monkeypatch.setattr(ingest, "_drain_many", fake_drain_many)
    def fake_reset(admin, url, namespace):
        ordering.append(f"reset:{namespace}")
        reset.append(namespace)

    def fake_settle(namespaces, **kwargs):
        ordering.append(f"settle:{','.join(namespaces)}")

    monkeypatch.setattr(ingest, "_reset_namespace", fake_reset)
    monkeypatch.setattr(ingest, "_await_stale_episode_settlement", fake_settle)

    manifest_path = tmp_path / "manifest.json"
    result = ingest.main(
        [
            "--limit",
            "3",
            "--manifest-item-limit",
            "2",
            "--namespace-window",
            "2",
            "--segmentation",
            "none",
            "--manifest",
            str(manifest_path),
        ]
    )

    assert result == 0
    assert reset == ["lme-a", "lme-b"]
    # Settlement is confirmed BEFORE the reset. Resetting first would delete the
    # PENDING/ENRICHING rows that prove a stale worker is alive, so the wait would read zero
    # against a worker still writing into the namespace.
    assert ordering == ["settle:lme-a,lme-b", "reset:lme-a", "reset:lme-b"]
    assert submitted == [
        ("lme-a", "a1"),
        ("lme-b", "b1"),
        ("lme-a", "a2"),
        ("lme-b", "b2"),
        ("lme-a", "a3"),
    ]
    assert drain_calls == [["lme-a", "lme-b"]]
    manifest = ingest.json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in manifest] == ["a", "b"]
    assert all(row["namespace_window"] == 2 for row in manifest)
    assert all(row["enrichment_llm_tasks"] == 10 for row in manifest)


def test_suburbs_fixture_contract_preserves_failure_shape() -> None:
    contract = verify._load_contract(FIXTURE)
    assert contract["question_id"] == "830ce83f-suburbs-fix-v1"
    assert contract["namespace"] == "lme-830ce83f-suburbs-fix-v1"
    assert contract["turns"] == 14
    assert contract["current_object_contains"] == "suburb"
    assert contract["stale_object"] == "Chicago"
    assert contract["required_menhir_commit"].startswith("c949dfa")


def test_graph_verification_checks_current_and_stale_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = verify._load_contract(FIXTURE)
    observed_queries: list[str] = []
    values = iter([1, 1, 1, 0, 1])

    def fake_count(container: str, password: str, query: str) -> int:
        assert container == "fixture-neo4j"
        assert password == "not-printed"
        observed_queries.append(query)
        return next(values)

    monkeypatch.setattr(verify, "_cypher_count", fake_count)
    counts = verify._graph_counts(contract, "fixture-neo4j", "not-printed")

    assert counts == {
        "suburb_entities": 1,
        "current_suburb_edges": 1,
        "expired_stale_edges": 1,
        "current_stale_edges": 0,
        "target_episodes": 1,
    }
    joined = "\n".join(observed_queries)
    assert "edge.invalid_at IS NULL" in joined
    assert "edge.invalid_at IS NOT NULL OR edge.expired_at IS NOT NULL" in joined
