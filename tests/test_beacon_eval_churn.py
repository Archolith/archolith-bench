"""File-churn reductions in the beacon-eval runner (offline; no model or network).

Seal via git alternates, delete checkouts after scoring (rescore rebuilds them), share
OpenCode's config-dir dependency install, and refuse new runs below a disk floor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.isolation import isolated_config_home
from archolith_bench.beacon_eval.models import Gold, RepoPin, Task
from archolith_bench.beacon_eval.runner import (
    RunnerConfig,
    export_commit,
    rescore,
    run_matrix,
    seal_checkout,
    seal_repo,
)

STUB = r"""
import json, os, sys
prompt = sys.stdin.read()
if "WRITE" in prompt:
    with open("NEW.md", "w", encoding="utf-8") as handle:
        handle.write("one\ntwo\nthree\n")
answer = {"docs": ["AGENTS.md"], "files": [], "commands": [], "guardrails": [], "verdict": "",
          "plan": [], "citations": [{"path": "NEW.md", "line_start": 1, "line_end": 3},
                                    {"path": "AGENTS.md", "line_start": 1, "line_end": 2}]}
print(json.dumps({"type": "text", "part": {"type": "text", "text": "```json\n" + json.dumps(answer) + "\n```"}}))
print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 100, "output": 10}}}))
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _gitb(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _repo(tmp_path: Path, attributes: str = "", ignored_tracked: bool = False) -> RepoPin:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    # Bytes, not text mode: Windows text mode would write CRLF into the blobs.
    (root / "AGENTS.md").write_bytes(b"# Agents\nNever commit secrets.\n")
    (root / "src" / "app.py").write_bytes(b"print('hi')\n")
    if attributes:
        (root / ".gitattributes").write_text(attributes, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "core.autocrlf", "false")
    if ignored_tracked:
        (root / ".gitignore").write_bytes(b"*.log\n")
        (root / "tracked.log").write_bytes(b"kept\n")
        _git(root, "add", "-f", "tracked.log")
    commit = _commit(root, "init")
    return RepoPin(name="demo", url="", commit=commit, local_path=str(root))


def _old_seal(pin: RepoPin, tmp_path: Path, name: str) -> Path:
    """The seal every earlier run used: export, then add --all + commit in the checkout."""
    checkout = export_commit(pin, tmp_path / name / "checkout", tmp_path / "cache")
    assert seal_checkout(checkout) == "copied"
    return checkout


def _new_seal(pin: RepoPin, tmp_path: Path, name: str) -> Path:
    checkout = export_commit(pin, tmp_path / name / "checkout", tmp_path / "cache")
    assert seal_checkout(checkout, seal_repo(pin, tmp_path / "work")) == "alternates"
    return checkout


def test_alternates_seal_is_a_clean_single_commit_root_with_few_files(tmp_path: Path) -> None:
    pin = _repo(tmp_path)
    # Under an unrelated repository, as checkouts sit under the results tree.
    outer = tmp_path / "outer"
    outer.mkdir()
    _git(outer, "init", "-q")
    checkout = export_commit(pin, outer / "run" / "checkout", tmp_path / "cache")
    assert seal_checkout(checkout, seal_repo(pin, tmp_path / "work")) == "alternates"
    assert Path(_git(checkout, "rev-parse", "--show-toplevel").strip()).resolve() == checkout.resolve()
    assert _git(checkout, "status", "--porcelain") == ""
    assert len(_git(checkout, "log", "--oneline").splitlines()) == 1
    assert _git(checkout, "cat-file", "-p", "HEAD:AGENTS.md").startswith("# Agents")
    # At most the run's own commit (none when it matches the seal repo's to the second).
    assert int(_git(checkout, "count-objects").split()[0]) <= 1
    assert sum(len(files) for _, _, files in os.walk(checkout / ".git")) < 40


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_the_agent_sees_exactly_what_the_old_seal_gave_it(tmp_path: Path, autocrlf: str) -> None:
    # Tracked-but-ignored files and CRLF exports are where a pin-tree seal would differ.
    pin = _repo(tmp_path, ignored_tracked=True)
    _git(Path(pin.local_path), "config", "core.autocrlf", autocrlf)
    old = _old_seal(pin, tmp_path, "old")
    new = _new_seal(pin, tmp_path, "new")
    assert (b"\r\n" in (new / "AGENTS.md").read_bytes()) == (autocrlf == "true")
    for args in (
        ("rev-parse", "HEAD^{tree}"),
        ("ls-files", "-s"),
        ("ls-files", "--others", "--ignored", "--exclude-standard"),
        ("status", "--porcelain", "--ignored"),
        ("cat-file", "-p", "HEAD:AGENTS.md"),
        ("config", "--local", "--list"),
        ("log", "--all", "--format=%an %ae %s"),
    ):
        assert _gitb(new, *args) == _gitb(old, *args), args
    assert b"tracked.log" not in _gitb(new, "ls-files")


def test_later_history_of_the_repository_is_not_reachable(tmp_path: Path) -> None:
    pin = _repo(tmp_path)
    root = Path(pin.local_path)
    (root / "LATER.md").write_text("the answer\n", encoding="utf-8")
    later = _commit(root, "later")
    later_blob = _git(root, "rev-parse", "HEAD:LATER.md").strip()
    checkout = _new_seal(pin, tmp_path, "run")
    listed = _git(checkout, "cat-file", "--batch-all-objects", "--batch-check")
    assert later not in listed and later_blob not in listed
    probe = subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", later_blob], capture_output=True)
    assert probe.returncode != 0


def test_a_checkout_that_does_not_match_the_seal_falls_back_to_a_packed_copy(tmp_path: Path) -> None:
    pin = _repo(tmp_path)
    checkout = export_commit(pin, tmp_path / "run" / "checkout", tmp_path / "cache")
    (checkout / "AGENTS.md").write_text("drifted\n", encoding="utf-8")
    assert seal_checkout(checkout, seal_repo(pin, tmp_path / "work")) == "copied"
    assert _git(checkout, "status", "--porcelain") == ""
    assert len(_git(checkout, "log", "--oneline").splitlines()) == 1
    assert _git(checkout, "count-objects").startswith("0 objects")  # all packed
    assert not (checkout / ".git" / "objects" / "info" / "alternates").exists()


def test_the_seal_repo_is_built_once_and_holds_no_working_files(tmp_path: Path) -> None:
    pin = _repo(tmp_path)
    first = seal_repo(pin, tmp_path / "work")
    stamp = (first / "tree").stat().st_mtime_ns
    assert seal_repo(pin, tmp_path / "work") == first and (first / "tree").stat().st_mtime_ns == stamp
    assert sorted(p.name for p in first.iterdir()) == [".git", "tree"]
    assert [p.name for p in first.parent.iterdir() if ".staging-" in p.name] == []


def _sealed(pin: RepoPin, tmp_path: Path) -> tuple[Path, Path, dict[str, int]]:
    run_dir = tmp_path / "run"
    checkout = export_commit(pin, run_dir / "checkout", tmp_path / "cache")
    baseline = runner_mod.file_times(checkout)
    seal_checkout(checkout, seal_repo(pin, tmp_path / "work"))
    return run_dir, checkout, baseline


def _write_pin(run_dir: Path, pin: RepoPin) -> None:
    (run_dir / "pin.json").write_text(json.dumps({"name": pin.name, "url": pin.url, "commit": pin.commit,
                                                  "local_path": pin.local_path, "seal": "alternates"}), encoding="utf-8")


def test_saved_changes_rebuild_new_changed_and_deleted_files(tmp_path: Path) -> None:
    pin = _repo(tmp_path)
    run_dir, checkout, baseline = _sealed(pin, tmp_path)
    (checkout / "NEW.md").write_bytes(b"new\r\nfile\n")
    (checkout / "AGENTS.md").write_bytes(b"changed\n")
    (checkout / "src" / "app.py").unlink()
    runner_mod.save_changes(checkout, run_dir, baseline)
    _write_pin(run_dir, pin)
    rebuilt = runner_mod.rebuild_checkout(run_dir, tmp_path / "rebuilt", tmp_path / "cache")
    assert (rebuilt / "NEW.md").read_bytes() == b"new\r\nfile\n"
    assert (rebuilt / "AGENTS.md").read_bytes() == b"changed\n"
    assert not (rebuilt / "src" / "app.py").exists()
    assert json.loads((run_dir / "changes.deleted.json").read_text(encoding="utf-8")) == ["src/app.py"]


def test_committed_and_ignored_agent_files_are_saved(tmp_path: Path) -> None:
    pin = _repo(tmp_path, ignored_tracked=True)
    run_dir, checkout, baseline = _sealed(pin, tmp_path)
    (checkout / "AGENTS.md").write_text("committed by the agent\n", encoding="utf-8")
    _git(checkout, "-c", "user.email=a@a", "-c", "user.name=a", "commit", "-qam", "agent")
    (checkout / "notes.log").write_text("ignored but cited\n", encoding="utf-8")
    assert _git(checkout, "status", "--porcelain") == ""  # git alone would miss both
    runner_mod.save_changes(checkout, run_dir, baseline)
    _write_pin(run_dir, pin)
    rebuilt = runner_mod.rebuild_checkout(run_dir, tmp_path / "rebuilt", tmp_path / "cache")
    assert (rebuilt / "AGENTS.md").read_text(encoding="utf-8") == "committed by the agent\n"
    assert (rebuilt / "notes.log").read_text(encoding="utf-8") == "ignored but cited\n"


@pytest.fixture
def stub_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stub = tmp_path / "stub_opencode.py"
    stub.write_text(STUB, encoding="utf-8")
    source = tmp_path / "user" / "opencode.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"provider": {"deepseek": {"options": {}}}}), encoding="utf-8")
    monkeypatch.setenv("PWD", str(tmp_path))
    config = RunnerConfig(
        workdir=tmp_path / "work", beacon_python="py", opencode_cmd=[sys.executable, str(stub)],
        model="deepseek/m", config_source=source, budget_tokens=None, run_reserve_tokens=None,
        timeout_s=60,
    )
    return config, _repo(tmp_path)


GOLD = Gold(docs=("AGENTS.md",))


def test_a_run_keeps_its_pin_and_changes_but_not_its_checkout(stub_config) -> None:
    config, pin = stub_config
    task = Task(repo="demo", task_id="t", kind="k", prompt="read only", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",))
    assert stopped == "" and results[0].scores["citation_location_validity"] == 0.5
    run_dir = config.workdir / "runs" / "demo-t-A-1"
    assert not (run_dir / "checkout").exists()
    saved = json.loads((run_dir / "pin.json").read_text(encoding="utf-8"))
    assert saved["commit"] == pin.commit and saved["seal"] == "alternates"
    assert (run_dir / "changes.status").read_bytes() == b""
    assert not (run_dir / "changes.tar").exists() and not (run_dir / "changes.deleted.json").exists()


def test_keep_checkouts_keeps_the_sealed_checkout(stub_config) -> None:
    config, pin = stub_config
    config.keep_checkouts = True
    task = Task(repo="demo", task_id="t", kind="k", prompt="read only", gold=GOLD)
    run_matrix(config, {"demo": pin}, [task], ("A",))
    checkout = config.workdir / "runs" / "demo-t-A-1" / "checkout"
    assert (checkout / "AGENTS.md").is_file() and _git(checkout, "status", "--porcelain") == ""


def test_rescore_of_a_deleted_checkout_matches_the_kept_one_including_agent_changes(stub_config) -> None:
    config, pin = stub_config
    config.keep_checkouts = True
    tasks = [
        Task(repo="demo", task_id="ro", kind="k", prompt="read only", gold=GOLD),
        Task(repo="demo", task_id="rw", kind="k", prompt="WRITE a file", gold=GOLD),
    ]
    results, _ = run_matrix(config, {"demo": pin}, tasks, ("A",), repeats=2)
    by_id = {(r.task_id, r.condition, r.repeat): r.scores for r in results}
    assert len(by_id) == 4
    # The agent's new file is what makes NEW.md's citation valid.
    assert by_id[("rw", "A", 1)]["citation_location_validity"] == 1.0
    assert by_id[("ro", "A", 1)]["citation_location_validity"] == 0.5
    rw = config.workdir / "runs" / "demo-rw-A-1"
    assert b"NEW.md" in (rw / "changes.status").read_bytes() and (rw / "changes.tar").is_file()
    kept = {(r.task_id, r.condition, r.repeat): r.scores for r in rescore(config.workdir, tasks)}
    for run_dir in (config.workdir / "runs").iterdir():
        runner_mod._rmtree(run_dir / "checkout")
    rebuilt = {(r.task_id, r.condition, r.repeat): r.scores for r in rescore(config.workdir, tasks)}
    assert rebuilt == kept == by_id
    assert not (config.workdir / "rescore-tmp").exists()


def test_rescore_refuses_a_run_with_neither_checkout_nor_pin(stub_config) -> None:
    config, pin = stub_config
    task = Task(repo="demo", task_id="t", kind="k", prompt="read only", gold=GOLD)
    run_matrix(config, {"demo": pin}, [task], ("A",))
    (config.workdir / "runs" / "demo-t-A-1" / "pin.json").unlink()
    with pytest.raises(FileNotFoundError, match="no checkout and no pin.json"):
        rescore(config.workdir, [task])


def test_a_rerun_does_not_inherit_the_earlier_attempts_changes(stub_config) -> None:
    config, pin = stub_config
    writer = Task(repo="demo", task_id="r", kind="k", prompt="WRITE a file", gold=GOLD)
    reader = Task(repo="demo", task_id="r", kind="k", prompt="read only", gold=GOLD)
    runner_mod.run_one(config, pin, writer, "A", 1)
    run_dir = config.workdir / "runs" / "demo-r-A-1"
    assert (run_dir / "changes.tar").is_file()
    second = runner_mod.run_one(config, pin, reader, "A", 1)  # same folder, as --resume reruns it
    assert not (run_dir / "changes.tar").exists() and not (run_dir / "changes.deleted.json").exists()
    assert second.scores["citation_location_validity"] == 0.5
    assert rescore(config.workdir, [reader])[0].scores == second.scores


def test_a_checkout_that_cannot_be_deleted_still_records_the_run_and_is_rebuilt(
    stub_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, pin = stub_config
    real = runner_mod._rmtree

    def held(path: Path) -> None:
        if path.name == "checkout" and path.exists():
            (path / "AGENTS.md").unlink()  # partly deleted, then a held file stops it
            raise PermissionError("file in use")
        real(path)

    monkeypatch.setattr(runner_mod, "_rmtree", held)
    task = Task(repo="demo", task_id="h", kind="k", prompt="WRITE a file", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",))
    assert stopped == "" and len(results) == 1
    run_dir = config.workdir / "runs" / "demo-h-A-1"
    assert (run_dir / "checkout.partial").is_file()
    assert len((config.workdir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    monkeypatch.setattr(runner_mod, "_rmtree", real)
    # Scored against a rebuilt checkout, not the half-deleted one (which lacks AGENTS.md).
    assert rescore(config.workdir, [task])[0].scores == results[0].scores


Usage = namedtuple("Usage", "total used free")


@pytest.mark.parametrize("workers", [1, 2])
def test_no_run_starts_below_the_disk_floor(stub_config, monkeypatch: pytest.MonkeyPatch, workers: int) -> None:
    config, pin = stub_config
    config.min_free_gb = 15.0
    monkeypatch.setattr(runner_mod.shutil, "disk_usage", lambda path: Usage(100e9, 90e9, 10e9))
    task = Task(repo="demo", task_id="t", kind="k", prompt="read only", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",), workers=workers)
    assert results == [] and "10.0 GB free" in stopped and "floor 15 GB" in stopped
    assert not (config.workdir / "runs").exists()


def test_the_disk_floor_can_be_switched_off(stub_config, monkeypatch: pytest.MonkeyPatch) -> None:
    config, pin = stub_config
    config.min_free_gb = None
    monkeypatch.setattr(runner_mod.shutil, "disk_usage", lambda path: Usage(100e9, 99e9, 1e9))
    task = Task(repo="demo", task_id="t", kind="k", prompt="read only", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",))
    assert stopped == "" and len(results) == 1


def _install(opencode_dir: Path, complete: bool = True) -> None:
    """What OpenCode's npm install leaves in its config dir."""
    plugin = opencode_dir / "node_modules" / "@opencode-ai" / "plugin"
    plugin.mkdir(parents=True)
    (plugin / "package.json").write_text('{"name": "@opencode-ai/plugin"}', encoding="utf-8")
    (opencode_dir / "package.json").write_text('{"dependencies": {}}', encoding="utf-8")
    if complete:
        (opencode_dir / "package-lock.json").write_text("{}", encoding="utf-8")


def _user_config(tmp_path: Path) -> Path:
    source = tmp_path / "user" / "opencode.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps({"provider": {"x": {}}}), encoding="utf-8")
    return source


def test_the_first_finished_install_becomes_the_shared_template(tmp_path: Path) -> None:
    template = tmp_path / "deps"
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        assert not (home / "opencode" / "node_modules").exists()
        _install(home / "opencode")
    assert (template / "node_modules" / "@opencode-ai" / "plugin" / "package.json").is_file()
    assert (template / "package-lock.json").is_file() and not home.exists()

    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        linked = home / "opencode" / "node_modules"
        assert (linked / "@opencode-ai" / "plugin" / "package.json").is_file()
        assert (home / "opencode" / "package-lock.json").is_file()
        assert os.path.realpath(linked) == os.path.realpath(template / "node_modules")
    # Removing the run's home leaves the shared install alone.
    assert not home.exists()
    assert (template / "node_modules" / "@opencode-ai" / "plugin" / "package.json").is_file()


def test_an_unfinished_install_is_not_shared(tmp_path: Path) -> None:
    template = tmp_path / "deps"
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        _install(home / "opencode", complete=False)
    assert not template.exists() and not home.exists()


def test_an_existing_template_is_not_replaced(tmp_path: Path) -> None:
    template = tmp_path / "deps"
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        _install(home / "opencode")
    marker = template / "node_modules" / "marker"
    marker.write_text("first", encoding="utf-8")
    # A run that started before the template existed installs its own; the first stays.
    (template / "node_modules").rename(tmp_path / "held")  # so this run does not link
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        (tmp_path / "held").rename(template / "node_modules")
        _install(home / "opencode")
    assert marker.read_text(encoding="utf-8") == "first"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("deps.staging-")] == []


def test_a_damaged_template_is_not_linked(tmp_path: Path) -> None:
    template = tmp_path / "deps"
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        _install(home / "opencode")
    (template / "node_modules" / "@opencode-ai" / "plugin" / "package.json").unlink()
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        assert not (home / "opencode" / "node_modules").exists()


def test_a_full_temp_dir_while_keeping_the_install_still_removes_the_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from archolith_bench.beacon_eval import isolation

    real = isolation.tempfile.mkdtemp

    def full(*args, **kwargs):  # noqa: ANN002, ANN003
        if ".staging-" in str(kwargs.get("prefix", "")):
            raise OSError(28, "No space left on device")
        return real(*args, **kwargs)

    monkeypatch.setattr(isolation.tempfile, "mkdtemp", full)
    template = tmp_path / "deps"
    with isolated_config_home(_user_config(tmp_path), "x/m", deps_template=template) as home:
        _install(home / "opencode")
    assert not home.exists() and not template.exists()


def test_the_run_home_is_removed_even_with_read_only_git_objects(tmp_path: Path) -> None:
    # OpenCode's snapshot store under the data home is read-only git objects.
    with isolated_config_home(_user_config(tmp_path), "x/m") as home:
        obj = home / ".data" / "opencode" / "snapshot" / "objects" / "ab" / "cdef"
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b"blob")
        os.chmod(obj, 0o444)
    assert not home.exists()


def test_ripgrep_is_hard_linked_not_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_data = tmp_path / "data"
    (user_data / "opencode" / "bin").mkdir(parents=True)
    rg = user_data / "opencode" / "bin" / "rg.exe"
    rg.write_bytes(b"rg")
    monkeypatch.setenv("XDG_DATA_HOME", str(user_data))
    with isolated_config_home(_user_config(tmp_path), "x/m") as home:
        seeded = home / ".data" / "opencode" / "bin" / "rg.exe"
        assert seeded.read_bytes() == b"rg" and os.path.samefile(seeded, rg)
    assert rg.read_bytes() == b"rg"
