"""Beacon-eval run lifecycle (PR #3 review findings 5, 8, 9, 10, 11): process-tree
termination, per-commit beacon caches, concurrent first writers, cancellation, and
retry artifact archiving.

Offline and deterministic: real processes appear only as sleeping stand-ins for
OpenCode and the Beacon servers, and ``beacon`` itself is a fake package that counts
its invocations. No model, judge or paid call is made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.models import Gold, RepoPin, RunResult, Task
from archolith_bench.beacon_eval.runner import (
    RateLimited,
    RunnerConfig,
    _kill_live_children,
    _kill_tree,
    _register_child,
    _unregister_child,
    build_beacon,
    export_beacon_snapshot,
    run_matrix,
    stream_opencode,
)

GOLD = Gold()
TASK = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD)
PIN = RepoPin(name="demo", url="", commit="0" * 40, local_path=".")


def _tasks(n: int) -> list[Task]:
    return [Task(repo="demo", task_id=f"t{i}", kind="why", prompt="Why?", gold=GOLD, reviewed=True)
            for i in range(n)]


# ---------------------------------------------------------------------------
# Small process and repository helpers
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    """Whether the process *pid* is still running (best effort, for assertions)."""
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _waits_for_death(pid: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)
    return True


def _repo(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "."],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


def _commit(root: Path, name: str) -> None:
    (root / name).write_text(name + "\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", name],
        check=True, capture_output=True,
    )


def _pin(root: Path, name: str = "demo") -> RepoPin:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return RepoPin(name=name, url="", commit=commit, local_path=str(root))


# ---------------------------------------------------------------------------
# Findings 8 and 9: per-commit caches, concurrent first writers
# ---------------------------------------------------------------------------

#: A stand-in ``beacon`` package: counts every invocation in a file (so a reuse versus
#: rebuild is observable), takes its time (so racing first writers really do overlap),
#: and writes the artifact the real one would.
FAKE_BEACON_MAIN = r"""
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
counter = Path(os.environ["BEACON_FAKE_COUNTER"])
counter.parent.mkdir(parents=True, exist_ok=True)
count = int(counter.read_text(encoding="utf-8") or "0") + 1
counter.write_text(str(count), encoding="utf-8")
time.sleep(0.2)
if args and args[0] == "build":
    repo = Path(args[args.index("--repo") + 1])
    (repo / "beacon.generated.yaml").write_text("project: fake\n", encoding="utf-8")
elif args and args[0] == "export":
    out = Path(args[args.index("--output") + 1])
    out.write_text('{"snapshot": true}\n', encoding="utf-8")
"""


def _fake_beacon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = tmp_path / "fakebeacon" / "beacon"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(FAKE_BEACON_MAIN, encoding="utf-8")
    counter = tmp_path / "beacon-invocations.txt"
    counter.write_text("0", encoding="utf-8")
    monkeypatch.setenv("BEACON_FAKE_COUNTER", str(counter))
    return counter


def _beacon_config(tmp_path: Path) -> RunnerConfig:
    return RunnerConfig(
        workdir=tmp_path / "work",
        beacon_python=sys.executable,
        beacon_src=str(tmp_path / "fakebeacon"),
        opencode_cmd=["x"],
        budget_tokens=None,
        run_reserve_tokens=None,
    )


def test_build_beacon_reuses_the_same_commit_and_rebuilds_on_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _fake_beacon(tmp_path, monkeypatch)
    config = _beacon_config(tmp_path)
    root = _repo(tmp_path / "repo")
    pin_a = _pin(root)

    first = build_beacon(config, pin_a)
    assert first == config.workdir / "beacons" / f"demo-{pin_a.commit[:12]}" / "beacon.generated.yaml"
    assert first.is_file() and counter.read_text(encoding="utf-8") == "1"

    assert build_beacon(config, pin_a) == first  # same commit: reused, not rebuilt
    assert counter.read_text(encoding="utf-8") == "1"

    _commit(root, "second.txt")
    second = build_beacon(config, _pin(root))
    assert second.parent.name == f"demo-{_pin(root).commit[:12]}" and second.is_file()
    assert counter.read_text(encoding="utf-8") == "2"  # a new commit for the same name rebuilds
    assert first.is_file()  # the old build is left untouched


def test_snapshot_export_reuses_the_same_commit_and_rebuilds_on_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = _fake_beacon(tmp_path, monkeypatch)
    config = _beacon_config(tmp_path)
    root = _repo(tmp_path / "repo")
    pin_a = _pin(root)

    snapshot = export_beacon_snapshot(config, pin_a)  # one build plus one export
    assert snapshot.name == "beacon.snapshot.json"
    assert snapshot.parent.name == f"demo-{pin_a.commit[:12]}"
    assert counter.read_text(encoding="utf-8") == "2"

    assert export_beacon_snapshot(config, pin_a) == snapshot  # same commit: no beacon run
    assert counter.read_text(encoding="utf-8") == "2"

    _commit(root, "second.txt")
    rebuilt = export_beacon_snapshot(config, _pin(root))
    assert rebuilt.parent != snapshot.parent and rebuilt.is_file()
    assert counter.read_text(encoding="utf-8") == "4"


def _race_two_callers(call: Any, config: RunnerConfig, pin: RepoPin) -> tuple[list[Any], list[BaseException]]:
    barrier = threading.Barrier(2)
    results: list[Any] = []
    errors: list[BaseException] = []

    def worker() -> None:
        barrier.wait()
        try:
            results.append(call(config, pin))
        except BaseException as exc:  # noqa: BLE001 - asserted below, never lost
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()
    return results, errors


def test_concurrent_first_writers_export_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _fake_beacon(tmp_path, monkeypatch)
    config = _beacon_config(tmp_path)
    pin = _pin(_repo(tmp_path / "repo"))

    results, errors = _race_two_callers(export_beacon_snapshot, config, pin)
    assert not errors
    assert len(results) == 2 and results[0] == results[1] and results[0].is_file()
    # One build plus one export: the second caller reused the first one's result.
    assert counter.read_text(encoding="utf-8") == "2"


def test_concurrent_first_builders_build_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = _fake_beacon(tmp_path, monkeypatch)
    config = _beacon_config(tmp_path)
    pin = _pin(_repo(tmp_path / "repo"))

    results, errors = _race_two_callers(build_beacon, config, pin)
    assert not errors
    assert len(results) == 2 and results[0] == results[1] and results[0].is_file()
    assert counter.read_text(encoding="utf-8") == "1"


# ---------------------------------------------------------------------------
# Finding 5: _kill_tree ends the whole tree, parent dead or alive
# ---------------------------------------------------------------------------

#: A stand-in OpenCode/Beacon: prints the pid of a sleeping child it spawned, holds a
#: pipe's write end (and passes it to the child, which holds it too), then sleeps. The
#: pipe's read end therefore EOFs exactly when every member of the tree is gone. On
#: Windows the pipe travels as a raw handle (an inherited handle has no fd number), so
#: the stand-in turns it back into an fd before handing it to its own child.
TREE_STAND_IN = r"""
import subprocess
import sys
import time

write_fd = int(sys.argv[1])
if sys.platform == "win32":
    import msvcrt
    write_fd = msvcrt.open_osfhandle(write_fd, 0)
kwargs = {} if sys.platform == "win32" else {"pass_fds": (write_fd,)}
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(120)"],
    stdout=write_fd, stderr=subprocess.DEVNULL, **kwargs,
)
print(child.pid, flush=True)
time.sleep(120)
"""


def _spawn_tree(tmp_path: Path) -> tuple[subprocess.Popen[str], int]:
    """Start the stand-in the way the runner starts children (own group on POSIX)."""
    script = tmp_path / "tree_stand_in.py"
    script.write_text(TREE_STAND_IN, encoding="utf-8")
    read_fd, write_fd = os.pipe()
    kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL, "text": True,
        **runner_mod._child_kwargs(),
    }
    if sys.platform == "win32":
        import msvcrt

        os.set_inheritable(write_fd, True)
        # close_fds=False: without it Windows inherits only the std handles, not ours.
        kwargs["close_fds"] = False
        pipe_arg: Any = msvcrt.get_osfhandle(write_fd)
    else:
        kwargs["pass_fds"] = (write_fd,)
        pipe_arg = write_fd
    parent = subprocess.Popen([sys.executable, str(script), str(pipe_arg)], **kwargs)
    os.close(write_fd)
    # The runner tracks every child it starts (the Windows kill-on-close job is what
    # reaches a tree whose root already exited); the stand-in starts the same way.
    runner_mod._track(parent)
    return parent, read_fd


def _read_to_eof(read_fd: int, timeout_s: float = 30.0) -> bool:
    """True once the pipe hits EOF (every writer died); False if it stalls to the timeout."""
    outcome: list[bytes] = []
    reader = threading.Thread(target=lambda: outcome.append(os.read(read_fd, 1)), daemon=True)
    reader.start()
    reader.join(timeout_s)
    return outcome == [b""]


def test_kill_tree_ends_a_grandchild(tmp_path: Path) -> None:
    parent, read_fd = _spawn_tree(tmp_path)
    try:
        assert parent.stdout is not None
        grandchild = int(parent.stdout.readline())
        assert grandchild > 0
        _kill_tree(parent)
        assert parent.wait(timeout=30) is not None
        assert _read_to_eof(read_fd)  # the spawned child died with the tree
    finally:
        _kill_tree(parent)
        os.close(read_fd)


def test_kill_tree_after_the_parent_exited_still_ends_the_tree(tmp_path: Path) -> None:
    parent, read_fd = _spawn_tree(tmp_path)
    try:
        assert parent.stdout is not None
        grandchild = int(parent.stdout.readline())
        parent.kill()  # the parent exits on its own; the child it started does not
        assert parent.wait(timeout=30) is not None
        assert _alive(grandchild)
        _kill_tree(parent)  # the pid is still ours (creation time matches): the tree must end
        assert _read_to_eof(read_fd)
    finally:
        _kill_tree(parent)
        os.close(read_fd)


def test_kill_live_children_ends_every_registered_tree(tmp_path: Path) -> None:
    procs: list[subprocess.Popen[str]] = []
    try:
        for _ in range(2):
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                **runner_mod._child_kwargs(),
            )
            _register_child(proc)
            procs.append(proc)
        assert runner_mod._live_children
        _kill_live_children()
        for proc in procs:
            assert proc.wait(timeout=30) is not None
        assert not runner_mod._live_children  # _kill_tree unregistered them
    finally:
        for proc in procs:
            _kill_tree(proc)


# ---------------------------------------------------------------------------
# Finding 10: cancellation
# ---------------------------------------------------------------------------


def test_a_keyboard_interrupt_in_stream_opencode_kills_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "sleepy_opencode.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def boom(self: runner_mod.EventLog, line: str) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner_mod.EventLog, "feed", boom)
    with pytest.raises(KeyboardInterrupt):
        stream_opencode(
            [sys.executable, str(script)], "prompt", tmp_path, dict(os.environ), run_dir,
            None, 60,
        )
    grandchild = int((run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert _waits_for_death(grandchild)  # the finally killed the tree mid-stream


def test_a_parallel_stop_kills_running_children_promptly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pids: dict[str, int] = {}

    def fake_run_one(config: RunnerConfig, pin: RepoPin, task: Task, condition: str,
                     repeat: int) -> RunResult:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"], **runner_mod._child_kwargs()
        )
        _register_child(proc)
        pids[task.task_id] = proc.pid
        try:
            if task.task_id == "t0":
                time.sleep(0.5)  # both runs in flight, both children registered
                proc.kill()
                proc.wait(timeout=10)
                run_dir = config.workdir / "runs" / f"{task.repo}-{task.task_id}-{condition}-{repeat}"
                run_dir.mkdir(parents=True, exist_ok=True)
                stopped = RunResult(task.repo, task.task_id, condition, repeat, None, "",
                                    error="429 simulated")
                (run_dir / "result.json").write_text(json.dumps(asdict(stopped)), encoding="utf-8")
                raise RateLimited("429 simulated")
            proc.wait(timeout=60)  # ends when the stop kills the child, not after 120s
            return RunResult(task.repo, task.task_id, condition, repeat, {"findings": []}, "",
                             error=f"exit code {proc.returncode}")
        finally:
            _unregister_child(proc)

    monkeypatch.setattr(runner_mod, "run_one", fake_run_one)
    config = RunnerConfig(workdir=tmp_path / "work", beacon_python="py", opencode_cmd=["x"],
                          budget_tokens=None, run_reserve_tokens=None)
    started = time.monotonic()
    results, stopped = run_matrix(config, {"demo": PIN}, _tasks(2), ("A",), workers=2)
    elapsed = time.monotonic() - started

    assert "429" in stopped
    assert elapsed < 60  # prompt: the 120s children were killed, not waited on
    assert {result.task_id for result in results} == {"t0", "t1"}  # both runs recorded
    interrupted = next(result for result in results if result.task_id == "t1")
    assert interrupted.error  # recorded with an error, like other stopped runs
    for task_id, pid in pids.items():
        assert not _alive(pid), f"{task_id}'s child survived the stop"


# ---------------------------------------------------------------------------
# Finding 11: a rerun archives the previous attempt
# ---------------------------------------------------------------------------

#: A stand-in OpenCode that answers cleanly; STUB_FAIL makes it exit nonzero after
#: answering (a failed run --resume redoes), STUB_COST varies the spend per attempt.
STUB = r"""
import json
import os
import sys

sys.stdin.read()
fail = os.environ.get("STUB_FAIL", "") == "1"
cost = float(os.environ.get("STUB_COST", "0.001"))
answer = {"docs": [], "files": [], "commands": [], "guardrails": [], "verdict": "",
          "plan": [], "citations": []}
print(json.dumps({"type": "text", "part": {"type": "text",
                  "text": "```json\n" + json.dumps(answer) + "\n```"}}))
print(json.dumps({"type": "step_finish",
                  "part": {"tokens": {"input": 100, "output": 10}, "cost": cost}}))
if fail:
    sys.exit(3)
"""


def _user_config(tmp_path: Path) -> Path:
    path = tmp_path / "user-opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "other/x",
        "provider": {"other": {"options": {}}},
    }), encoding="utf-8")
    return path


def _stub_config(tmp_path: Path) -> RunnerConfig:
    stub = tmp_path / "stub_opencode.py"
    stub.write_text(STUB, encoding="utf-8")
    return RunnerConfig(
        workdir=tmp_path / "work",
        beacon_python="py",
        opencode_cmd=[sys.executable, str(stub)],
        model="other/x",
        config_source=_user_config(tmp_path),
        budget_tokens=None,
        run_reserve_tokens=None,
        timeout_s=60,
    )


def test_a_rerun_archives_the_previous_attempt_s_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stub_config(tmp_path)
    pin = _pin(_repo(tmp_path / "repo"))
    results, stopped = run_matrix(config, {"demo": pin}, [TASK], ("A",))
    assert stopped == "" and results[0].cost_usd == 0.001
    run_dir = config.workdir / "runs" / "demo-t1-A-1"
    assert not (run_dir / "attempts").exists()

    monkeypatch.setenv("STUB_COST", "0.042")
    results, stopped = run_matrix(config, {"demo": pin}, [TASK], ("A",))
    assert stopped == "" and results[0].cost_usd == 0.042

    attempt = run_dir / "attempts" / "1"
    for name in ("prompt.txt", "events.jsonl", "stderr.log", "result.json"):
        assert (attempt / name).is_file(), name
    saved = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    assert saved["cost_usd"] == 0.001  # the old attempt's cost stays visible
    current = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert current["cost_usd"] == 0.042  # the new attempt stands alone at the top level
    assert not (run_dir / "attempts" / "2").exists()


def test_resume_redoes_a_failed_run_and_keeps_the_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _stub_config(tmp_path)
    pin = _pin(_repo(tmp_path / "repo"))
    monkeypatch.setenv("STUB_FAIL", "1")
    results, stopped = run_matrix(config, {"demo": pin}, [TASK], ("A",), resume=True)
    assert stopped == "" and results[0].error == "exit code 3" and results[0].answer is not None

    monkeypatch.setenv("STUB_FAIL", "")
    results, stopped = run_matrix(config, {"demo": pin}, [TASK], ("A",), resume=True)
    assert stopped == "" and results[0].error == ""  # redone, because the saved run had an error
    run_dir = config.workdir / "runs" / "demo-t1-A-1"
    saved = json.loads((run_dir / "attempts" / "1" / "result.json").read_text(encoding="utf-8"))
    assert saved["error"] == "exit code 3"  # the failed attempt is archived, not lost
    current = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert current["answer"] is not None and current["error"] == ""
