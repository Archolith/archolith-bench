"""AMA-Bench State Updating adapter and ingest planning (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from archolith_bench.harness import StubMenhirClient, get_adapter, run_memory_ab
from archolith_bench.harness.ama_bench import (
    AmaBenchStateAdapter,
    namespace_for,
    question_items,
    render_steps,
    select_episodes,
    step_time,
)


def _episode(eid, domain, tokens, types, steps=2):
    return {
        "episode_id": eid,
        "task": f"task {eid}",
        "task_type": "t",
        "domain": domain,
        "success": True,
        "num_turns": steps,
        "total_tokens": tokens,
        "trajectory": [
            {"turn_idx": i, "action": f"act{i}", "observation": f"obs{i} of ep{eid}"} for i in range(steps)
        ],
        "qa_pairs": [
            {"question": f"q{eid}{t}", "answer": f"gold{eid}{t}", "type": t, "question_uuid": f"u{eid}{t}{n}"}
            for n, t in enumerate(types)
        ],
    }


EPISODES = [
    _episode(0, "WEB", 1000, "AC"),
    _episode(1, "WEB", 1000, "C"),
    _episode(2, "WEB", 1000, "A"),        # no type C question
    _episode(3, "GAME", 999_999, "C"),    # too long
    _episode(4, "GAME", 500, "CC"),
    _episode(5, "OPENWORLD_QA", 100, "C"),  # excluded domain
]


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "ama.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in EPISODES) + "\n", encoding="utf-8")
    return path


def test_selection_is_deterministic_and_filtered():
    picked = select_episodes(EPISODES, per_domain=1, max_tokens=60_000)
    assert [e["episode_id"] for e in picked] == [4, 0]  # sorted by domain, lowest ids first
    assert [e["episode_id"] for e in select_episodes(EPISODES, per_domain=5)] == [4, 0, 1]


def test_explicit_episode_ids_override_filters():
    assert [e["episode_id"] for e in select_episodes(EPISODES, episode_ids=(5, 3))] == [3, 5]


def test_steps_are_rendered_in_order_with_increasing_times():
    turns = render_steps(EPISODES[0])
    assert turns[0]["content"] == "Task: task 0"
    assert turns[1]["content"] == "Step 0\nAction: act0\nObservation: obs0 of ep0"
    assert [t["occurred_at"] for t in turns] == sorted(t["occurred_at"] for t in turns)
    assert len({t["occurred_at"] for t in turns}) == len(turns)
    assert turns[0]["occurred_at"] == step_time(0)


def test_questions_share_their_episode_namespace():
    items = question_items([EPISODES[4]])
    assert [i["question_id"] for i in items] == ["u4C0", "u4C1"]
    assert {i["namespace"] for i in items} == {namespace_for(4)}
    assert all(i["question_type"] == "ama-C" for i in items)


def test_adapter_is_registered_and_loads_type_c(dataset: Path, monkeypatch):
    for var in ("AMA_EPISODE_IDS", "AMA_QA_TYPES", "AMA_PER_DOMAIN", "AMA_MAX_TOKENS", "AMA_EXCLUDE_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    adapter = get_adapter("ama-bench-state")
    assert isinstance(adapter, AmaBenchStateAdapter)
    items = adapter.load_items(fixture_path=dataset)
    assert [i["question_id"] for i in items] == ["u4C0", "u4C1", "u0C1", "u1C0"]
    monkeypatch.setenv("AMA_EPISODE_IDS", "1")
    assert [i["question_id"] for i in adapter.load_items(fixture_path=dataset)] == ["u1C0"]


def test_adapter_requires_a_dataset(monkeypatch):
    monkeypatch.delenv("AMA_BENCH_DATASET", raising=False)
    with pytest.raises(ValueError):
        AmaBenchStateAdapter().load_items()


def test_recall_only_reads_each_items_own_namespace(dataset: Path, monkeypatch):
    monkeypatch.setenv("AMA_EPISODE_IDS", "0,1")
    client = StubMenhirClient()
    prebuilt = {namespace_for(0): ["assistant: gold0C"], namespace_for(1): ["assistant: gold1C"]}
    client._groups.update({k: list(v) for k, v in prebuilt.items()})

    def send_fn(client_, base_url, api_key, messages, model, **kwargs):
        user = messages[-1]["content"]
        text = "gold0C" if "gold0C" in user else ("gold1C" if "gold1C" in user else "I don't know")
        return text, 1.0, {"prompt_tokens": 1, "completion_tokens": 1}

    ab = run_memory_ab(
        AmaBenchStateAdapter(),
        arms=("menhir_recall",),
        fixture_path=dataset,
        client=client,
        send_fn=send_fn,
        recall_only=True,
    )
    assert [r.correct for r in ab.arms["menhir_recall"].results] == [True, True]
    assert client._groups == prebuilt  # nothing ingested, reset or created


def test_ingest_dry_run_plans_without_a_server(dataset: Path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import ingest_ama

    rc = ingest_ama.main([
        "--menhir-url", "http://127.0.0.1:1", "--dataset", str(dataset),
        "--out", "unused", "--dry-run",
    ])
    assert rc == 0
    assert "plan: 3 episodes, 9 steps, 2,500 tokens, 4 questions" in capsys.readouterr().out


def test_progress_counts_ready_and_failed_as_done_and_renders(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import progress_ama

    from archolith_bench.dashboard import (
        EPISODE_PROGRESS_FILE,
        render_html,
        scan_episode_progress,
    )

    counts = {namespace_for(0): {"READY": 2, "FAILED": 1}, namespace_for(1): {"READY": 1, "ENRICHING": 1}}
    payload = progress_ama.build_progress([EPISODES[0], EPISODES[1]], counts, "ama-test")
    assert [(r["steps_total"], r["ready"], r["failed"], r["in_flight"]) for r in payload["episodes"]] == [
        (3, 2, 1, 0), (3, 1, 0, 1),
    ]

    run_dir = tmp_path / "ama-test"
    run_dir.mkdir()
    progress_ama.write_atomic(run_dir / EPISODE_PROGRESS_FILE, payload)
    loaded = scan_episode_progress(tmp_path)
    assert loaded == payload

    page = render_html([], None, total_items=None, episode_progress=loaded)
    assert "Episode ingest" in page
    assert "4/6 steps (66.7%)" in page and "1/2 episodes finished" in page and "1 failed steps" in page
    assert render_html([], None, total_items=None).count("Episode ingest") == 0


def test_parse_memories_accepts_json_and_saves_nothing_on_bad_output():
    from archolith_bench.harness.ama_bench import parse_memories

    assert parse_memories('{"memories": ["Step 3: tests fail", "  "]}') == ["Step 3: tests fail"]
    assert parse_memories('```json\n{"memories": ["Step 1: x"]}\n```') == ["Step 1: x"]
    assert parse_memories('{"memories": []}') == []
    assert parse_memories("not json") == []
    assert parse_memories('{"memories": "Step 1: not a list"}') == []
    assert len(parse_memories(json.dumps({"memories": [f"m{i}" for i in range(9)]}))) == 5


def test_memory_agent_sees_task_saved_memories_and_the_step():
    from archolith_bench.harness.ama_bench import memory_agent_messages

    messages = memory_agent_messages(EPISODES[0], EPISODES[0]["trajectory"][1], ["Step 0: began"])
    user = messages[1]["content"]
    assert "Task:\ntask 0" in user and "- Step 0: began" in user
    assert "Step 1\nAction: act1\nObservation: obs1 of ep0" in user
    assert "(none yet)" in memory_agent_messages(EPISODES[0], EPISODES[0]["trajectory"][0], [])[1]["content"]


def test_crafted_turns_follow_step_order_and_step_times():
    from archolith_bench.harness.ama_bench import crafted_turns

    turns = crafted_turns([
        {"step": 2, "memories": ["Step 2: b"]},
        {"step": 0, "memories": ["Step 0: a", "Step 0: a2"]},
        {"step": 1, "memories": []},
    ])
    assert [t["content"] for t in turns] == ["Step 0: a", "Step 0: a2", "Step 2: b"]
    assert [t["occurred_at"] for t in turns] == [step_time(1), step_time(1), step_time(3)]


def test_craft_episode_resumes_without_re_asking_finished_steps(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import craft_memories

    asked: list[str] = []

    def fake_ask(client, base_url, api_key, model, messages):  # noqa: ANN001
        step_line = next(line for line in messages[1]["content"].splitlines() if line.startswith("Step "))
        asked.append(step_line)
        return json.dumps({"memories": [f"{step_line}: noted"]})

    monkeypatch.setattr(craft_memories, "ask", fake_ask)
    (tmp_path / "ep0.json").write_text(json.dumps([{"step": 0, "memories": ["Step 0: cached"], "raw": None}]))
    result = craft_memories.craft_episode(EPISODES[0], tmp_path, base_url="u", api_key="k", model="m")
    assert asked == ["Step 1"]
    assert result == {"episode_id": 0, "domain": "WEB", "steps": 2, "steps_done": 2, "memories": 2, "complete": True}


def test_ingest_refuses_incomplete_crafted_memories(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import ingest_ama

    (tmp_path / "ep0.json").write_text(json.dumps([{"step": 0, "memories": ["Step 0: a"]}]))
    with pytest.raises(ValueError):
        ingest_ama.load_crafted(tmp_path, EPISODES[0])
    with pytest.raises(FileNotFoundError):
        ingest_ama.load_crafted(tmp_path, EPISODES[1])


def test_memory_agent_retries_a_timed_out_call():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import httpx

    import craft_memories

    calls = {"n": 0}

    def handler(request):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"memories": []}'}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert craft_memories.ask(client, "http://x/v1", "k", "m", []) == '{"memories": []}'
    assert calls["n"] == 3

    calls["n"] = -10  # never recovers
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(httpx.TimeoutException):
        craft_memories.ask(client, "http://x/v1", "k", "m", [])


def _tl_episode():
    return {
        "episode_id": 7, "domain": "SOFTWARE", "task": "fix bug",
        "trajectory": [
            {"turn_idx": 0, "action": "run tests", "observation": "3 failed, 10 passed in 2.1s"},
            {"turn_idx": 1, "action": "edit card.py", "observation": "File edited successfully."},
            {"turn_idx": 2, "action": "run tests", "observation": "13 passed in 2.0s"},
            {"turn_idx": 3, "action": "run tests", "observation": "13 passed in 1.9s"},
        ],
    }


def test_verify_timelines_keeps_only_quoted_ordered_changes():
    from archolith_bench.harness.ama_bench import verify_timelines

    raw = json.dumps({"timelines": [
        {"subject": "test suite", "attribute": "result", "states": [
            {"step": 0, "value": "3 failing", "evidence": "3 failed, 10 passed"},
            {"step": 2, "value": "all passing", "evidence": "13 passed in 2.0s"},
            {"step": 3, "value": "all passing", "evidence": "13 passed in 1.9s"},    # repeat, not a change
            {"step": 1, "value": "invented", "evidence": "tests are green now"},      # not in step 1
        ]},
        {"subject": "x", "attribute": "y", "states": [                              # one real state only
            {"step": 1, "value": "edited", "evidence": "File edited successfully"},
            {"step": 2, "value": "made up", "evidence": "never appears anywhere"},
        ]},
    ]})
    kept, stats = verify_timelines(_tl_episode(), raw)
    assert [(s["step"], s["value"]) for s in kept[0]["states"]] == [(0, "3 failing"), (2, "all passing")]
    assert len(kept) == 1 and stats["timelines_kept"] == 1 and stats["states_proposed"] == 6
    assert verify_timelines(_tl_episode(), "not json") == ([], {"timelines_proposed": 0, "states_proposed": 0,
                                                                "states_verified": 0, "timelines_kept": 0})


def test_gold_questions_templates_current_previous_timeline():
    from archolith_bench.harness.ama_bench import gold_questions

    states = [{"step": 0, "value": "3 failing", "evidence": "e"}, {"step": 2, "value": "flaky", "evidence": "e"},
              {"step": 5, "value": "all passing", "evidence": "e"}]
    qs = {q["question_type"]: q for q in gold_questions(_tl_episode(), [{"subject": "test suite", "attribute": "result", "states": states}])}
    assert qs["current"]["answer"] == "all passing" and qs["current"]["stale_answers"] == ["3 failing", "flaky"]
    assert qs["previous"]["answer"] == "flaky" and qs["previous"]["stale_answers"] == ["3 failing"]
    assert qs["timeline"]["answer"] == "3 failing -> flaky -> all passing"
    assert "namespace" not in qs["current"]


def test_wait_until_settled_needs_two_quiet_polls_and_times_out():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import ingest_ama

    seq = iter([3, 0, 1, 0, 0])
    assert ingest_ama.wait_until_settled(lambda: next(seq), sleep=lambda s: None) is True
    now = {"t": 0.0}

    def tick(s):  # noqa: ANN001
        now["t"] += s

    assert ingest_ama.wait_until_settled(lambda: 1, poll_s=10, timeout_s=30, sleep=tick, clock=lambda: now["t"]) is False


def test_stale_answers_never_equal_the_gold_answer():
    from archolith_bench.harness.ama_bench import gold_questions

    states = [{"step": i, "value": v, "evidence": "e"} for i, v in enumerate(["True", "False", "True"])]
    qs = {q["question_type"]: q for q in gold_questions(_tl_episode(), [{"subject": "s", "attribute": "a", "states": states}])}
    assert qs["current"]["answer"] == "True" and qs["current"]["stale_answers"] == ["False"]
    assert qs["previous"]["answer"] == "False" and qs["previous"]["stale_answers"] == []


def test_apply_check_drops_off_subject_and_refinements_and_fails_closed():
    from archolith_bench.harness.ama_bench import apply_check

    tl = {"subject": "s", "attribute": "a", "states": [
        {"step": 1, "value": "CMU", "evidence": "e"},
        {"step": 2, "value": "CMU, Tech Street", "evidence": "e"},   # refinement
        {"step": 3, "value": "other row", "evidence": "e"},          # off subject
        {"step": 4, "value": "Pitt", "evidence": "e"},
    ]}
    raw = json.dumps({"states": [
        {"step": 1, "about_subject": True, "real_change": True},
        {"step": 2, "about_subject": True, "real_change": False},
        {"step": 3, "about_subject": False, "real_change": True},
        {"step": 4, "about_subject": True, "real_change": True},
    ]})
    assert [s["value"] for s in apply_check(tl, raw)["states"]] == ["CMU", "Pitt"]
    assert apply_check(tl, "garbage") is None
    assert apply_check(tl, json.dumps({"states": [{"step": 1, "about_subject": True, "real_change": True}]})) is None


def test_long_flip_flop_timelines_get_no_timeline_question():
    from archolith_bench.harness.ama_bench import MAX_TIMELINE_STATES, gold_questions

    states = [{"step": i, "value": "open" if i % 2 else "closed", "evidence": "e"} for i in range(MAX_TIMELINE_STATES + 1)]
    kinds = [q["question_type"] for q in gold_questions(_tl_episode(), [{"subject": "menu", "attribute": "state", "states": states}])]
    assert kinds == ["current", "previous"]


def _gold_rows():
    return [
        {"question_id": "ep7-t0-current", "episode_id": 7, "domain": "SOFTWARE", "question_type": "current",
         "question": "latest result?", "answer": "all passing", "stale_answers": ["3 failing"]},
        {"question_id": "ep7-t0-timeline", "episode_id": 7, "domain": "SOFTWARE", "question_type": "timeline",
         "question": "list results", "answer": "3 failing -> all passing", "stale_answers": []},
        {"question_id": "ep8-t0-current", "episode_id": 8, "domain": "WEB", "question_type": "current",
         "question": "latest page?", "answer": "checkout", "stale_answers": ["cart"]},
    ]


def test_gold_items_use_the_current_namespace_prefix_and_episode_filter(tmp_path: Path, monkeypatch):
    from archolith_bench.harness.ama_bench import AmaBenchStateAdapter

    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _gold_rows()), encoding="utf-8")
    monkeypatch.setenv("AMA_GOLD_PATH", str(path))
    monkeypatch.setenv("AMA_NAMESPACE_PREFIX", "ama-crafted-ep")
    monkeypatch.delenv("AMA_EPISODE_IDS", raising=False)
    items = AmaBenchStateAdapter().load_items()
    assert [(i["question_id"], i["namespace"]) for i in items] == [
        ("ep7-t0-current", "ama-crafted-ep7"), ("ep7-t0-timeline", "ama-crafted-ep7"), ("ep8-t0-current", "ama-crafted-ep8")]
    monkeypatch.setenv("AMA_EPISODE_IDS", "8")
    assert [i["question_id"] for i in AmaBenchStateAdapter().load_items()] == ["ep8-t0-current"]


def test_scorer_labels_fail_closed_and_summarize(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ama"))
    import score_gold

    assert score_gold.parse_label('{"label": "stale"}', "current") == "stale"
    assert score_gold.parse_label('{"label": "partial"}', "current") == "unscored"   # not a state label
    assert score_gold.parse_label('{"label": "partial"}', "timeline") == "partial"
    assert score_gold.parse_label("nonsense", "current") == "unscored"

    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(r) for r in _gold_rows()), encoding="utf-8")
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"arms": {"menhir_recall": {"results": [
        {"task_id": "ep7-t0-current", "response_text": "3 tests fail"},
        {"task_id": "ep7-t0-timeline", "response_text": "failing then passing"},
    ]}}}), encoding="utf-8")
    verdicts = {"latest result?": "stale", "list results": "correct"}

    def fake_ask(client, base_url, api_key, model, messages):  # noqa: ANN001
        q = next(line for line in messages[1]["content"].splitlines() if line.startswith("Question: "))
        return json.dumps({"label": verdicts[q.removeprefix("Question: ")]})

    monkeypatch.setattr(score_gold, "ask", fake_ask)
    out = tmp_path / "scored"
    rc = score_gold.main(["--answers", str(answers), "--gold", str(gold), "--out", str(out),
                          "--base-url", "u", "--model", "m", "--workers", "2"])
    report = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    assert rc == 1 and report["missing_answers"] == ["ep8-t0-current"]   # an unanswered question fails the run
    assert report["overall"]["n"] == 2 and report["by_type"]["current"]["stale"] == 1
    assert report["by_type"]["timeline"]["correct_rate"] == 1.0
    assert report["by_domain_type"]["SOFTWARE/current"]["stale_rate"] == 1.0
