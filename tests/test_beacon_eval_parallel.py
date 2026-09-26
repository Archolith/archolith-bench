"""The parallel matrix (--workers): overlap, budget reserves held by in-flight runs, the first
stop admitting no new run, shared state prepared once, and results in matrix order."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from archolith_bench.beacon_eval import isolation as isolation_mod
from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.isolation import isolated_config_home, isolated_env
from archolith_bench.beacon_eval.models import Gold, RepoPin, RunResult, Task
from archolith_bench.beacon_eval.runner import RateLimited, RunnerConfig, run_matrix

GOLD = Gold(docs=("README.md",))
PIN = RepoPin(name="demo", url="", commit="0" * 40, local_path=".")


def _tasks(n: int) -> list[Task]:
    return [Task(repo="demo", task_id=f"t{i}", kind="why", prompt="Why?", gold=GOLD, reviewed=True)
            for i in range(n)]


class FakeRuns:
    """Stands in for run_one: sleeps, tracks concurrency, and can raise on chosen tasks."""

    def __init__(self, cost: float = 0.01, sleep: float = 0.05, fail: dict[str, Exception] | None = None):
        self.cost, self.sleep, self.fail = cost, sleep, fail or {}
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.started: list[str] = []

    def __call__(self, config, pin, task, condition, repeat) -> RunResult:  # noqa: ANN001
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append(f"{task.task_id}-{condition}-{repeat}")
        try:
            time.sleep(self.sleep)
            if task.task_id in self.fail:
                run_dir = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}"
                run_dir.mkdir(parents=True, exist_ok=True)
                failed = RunResult(task.repo, task.task_id, condition, repeat, None, "", error="429",
                                   cost_usd=self.cost)
                (run_dir / "result.json").write_text(json.dumps(failed.__dict__), encoding="utf-8")
                raise self.fail[task.task_id]
            return RunResult(task.repo, task.task_id, condition, repeat, {"findings": []}, "",
                             input_tokens=10, output_tokens=5, cost_usd=self.cost)
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prepared: list[str] = []
    monkeypatch.setattr(runner_mod, "_ensure_source", lambda pin, cache: prepared.append(f"src:{pin.name}"))
    monkeypatch.setattr(runner_mod, "build_beacon", lambda config, pin: prepared.append(f"beacon:{pin.name}"))

    def make(fake: FakeRuns, **budget) -> RunnerConfig:  # noqa: ANN003
        monkeypatch.setattr(runner_mod, "run_one", fake)
        return RunnerConfig(workdir=tmp_path / "work", beacon_python="py", opencode_cmd=["x"],
                            budget_tokens=None, run_reserve_tokens=None, **budget)

    return make, prepared


def test_runs_overlap_and_come_back_in_matrix_order(setup) -> None:
    make, _ = setup
    fake = FakeRuns(sleep=0.1)
    config = make(fake, budget_usd=5.0, run_reserve_usd=0.03)
    results, stopped = run_matrix(config, {"demo": PIN}, _tasks(3), ("A", "B"), repeats=2, workers=3)
    assert stopped == "" and fake.peak == 3
    assert [(r.task_id, r.condition, r.repeat) for r in results] == [
        (f"t{t}", c, r) for r in (1, 2) for t in range(3) for c in ("A", "B")
    ]
    assert len((config.workdir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_in_flight_runs_hold_their_reserve(setup) -> None:
    make, _ = setup
    fake = FakeRuns(cost=0.01, sleep=0.05)
    # Two reserves fit under the cap, a third does not: never more than two in flight.
    config = make(fake, budget_usd=0.25, run_reserve_usd=0.1)
    results, stopped = run_matrix(config, {"demo": PIN}, _tasks(8), ("A",), workers=4)
    assert fake.peak <= 2
    assert "reserve would exceed" in stopped or stopped == ""
    assert sum(r.cost_usd for r in results) <= 0.25


def test_the_first_rate_limit_admits_no_new_run(setup) -> None:
    make, _ = setup
    fake = FakeRuns(sleep=0.05, fail={"t1": RateLimited("429 Too Many Requests")})
    config = make(fake, budget_usd=5.0, run_reserve_usd=0.01)
    results, stopped = run_matrix(config, {"demo": PIN}, _tasks(10), ("A",), workers=2)
    assert "429" in stopped
    # t0 and t1 start together; t2 may start when t0 finishes first; nothing after the stop.
    assert len(fake.started) <= 3
    assert any(r.task_id == "t1" and r.error for r in results)  # the stopping run is recorded


def test_shared_state_is_prepared_once_before_runs(setup) -> None:
    make, prepared = setup
    fake = FakeRuns(sleep=0.01)
    config = make(fake, budget_usd=5.0, run_reserve_usd=0.01)
    run_matrix(config, {"demo": PIN}, _tasks(4), ("A", "B"), workers=3)
    assert prepared == ["src:demo", "beacon:demo"]


def test_one_worker_keeps_the_serial_path(setup) -> None:
    make, prepared = setup
    fake = FakeRuns(sleep=0.0)
    config = make(fake, budget_usd=5.0, run_reserve_usd=0.01)
    results, stopped = run_matrix(config, {"demo": PIN}, _tasks(2), ("A",))
    assert stopped == "" and len(results) == 2 and fake.peak == 1 and prepared == []


def test_each_run_gets_its_own_opencode_data_and_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_data = tmp_path / "user-data"
    (user_data / "opencode" / "bin").mkdir(parents=True)
    (user_data / "opencode" / "bin" / "rg.exe").write_bytes(b"rg")
    (user_data / "opencode" / "opencode.db").write_bytes(b"sessions")
    monkeypatch.setattr(isolation_mod, "default_data_source", lambda: user_data)
    source_config = tmp_path / "opencode.json"
    source_config.write_text(json.dumps({"provider": {"x": {}}}), encoding="utf-8")
    with isolated_config_home(source_config, "x/model") as home:
        env = isolated_env({}, home)
        data = Path(env["XDG_DATA_HOME"])
        assert data.parent == home and Path(env["XDG_STATE_HOME"]).is_dir()
        assert (data / "opencode" / "bin" / "rg.exe").read_bytes() == b"rg"
        assert not (data / "opencode" / "opencode.db").exists()  # never the user's sessions
    assert not home.exists()
