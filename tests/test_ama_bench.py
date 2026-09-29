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
