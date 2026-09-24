"""Offline tests for the Beacon agent-task harness. No model or network is ever called."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.isolation import (
    IsolationError,
    beacon_server,
    isolated_config_home,
    isolated_env,
)
from archolith_bench.beacon_eval.models import Gold, RepoPin, Task
from archolith_bench.beacon_eval.report import render
from archolith_bench.beacon_eval.runner import (
    Budget,
    BudgetExhausted,
    TOOL_NOTE,
    RunnerConfig,
    build_prompt,
    parse_events,
    run_matrix,
)
from archolith_bench.beacon_eval.scoring import extract_answer, score

ANSWER = {
    "docs": ["AGENTS.md", "./docs/guide.md"],
    "files": ["src/app.py", "src/other.py"],
    "commands": ["pytest -q"],
    "guardrails": ["Never commit secrets"],
    "verdict": "superseded",
    "plan": ["read", "change"],
    "citations": [{"path": "AGENTS.md", "line_start": 1, "line_end": 2}, {"path": "nope.md"}],
}
GOLD = Gold(
    docs=("AGENTS.md", "docs/guide.md"),
    files=("src/app.py",),
    commands=("pytest -q", "ruff check ."),
    guardrails=("secrets",),
    verdict="superseded",
    risky=("rm -rf",),
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "AGENTS.md").write_text("# Agents\nNever commit secrets.\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


def _pin(root: Path) -> RepoPin:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return RepoPin(name="demo", url="", commit=commit, local_path=str(root))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_extracts_the_last_json_block_and_scores_it(tmp_path: Path) -> None:
    text = "notes\n```json\n{\"docs\": []}\n```\nfinal\n```json\n" + json.dumps(ANSWER) + "\n```"
    answer = extract_answer(text)
    assert answer == ANSWER
    scores = score(answer, GOLD, _repo(tmp_path))
    assert scores["doc_recall"] == 1.0
    assert scores["file_recall"] == 1.0
    assert scores["file_precision"] == 0.5
    assert scores["command_recall"] == 0.5
    assert scores["guardrail_recall"] == 1.0
    assert scores["verdict_correct"] == 1.0
    assert scores["risky_false_positive"] == 0.0
    assert scores["citation_location_validity"] == 0.5


def test_no_answer_scores_as_unanswered(tmp_path: Path) -> None:
    assert extract_answer("no json here") is None
    assert score(None, GOLD, tmp_path) == {"answered": 0.0}


def test_risky_commands_are_flagged_but_warnings_are_not(tmp_path: Path) -> None:
    risky = dict(ANSWER, commands=["pytest -q", "rm -rf build"])
    assert score(risky, GOLD, tmp_path)["risky_false_positive"] == 1.0
    warned = dict(ANSWER, plan=["never run rm -rf here"])
    assert score(warned, GOLD, tmp_path)["risky_false_positive"] == 0.0


# ---------------------------------------------------------------------------
# Events, budget, prompts
# ---------------------------------------------------------------------------


def test_parse_events_reads_text_tokens_tools_and_rate_limits() -> None:
    lines = [
        json.dumps({"type": "text", "part": {"type": "text", "text": "hello"}}),
        json.dumps({"type": "tool_use", "part": {"tool": "read"}}),
        json.dumps({"type": "step_finish", "part": {"tokens": {"input": 100, "output": 20, "reasoning": 5}}}),
        "not json",
    ]
    events = parse_events("\n".join(lines))
    assert events["text"] == "hello"
    assert (events["input_tokens"], events["output_tokens"], events["tool_calls"]) == (100, 25, 1)
    assert not events["rate_limited"]
    assert parse_events('{"error": "HTTP 429 Too Many Requests"}')["rate_limited"]


def test_a_tool_call_with_a_nested_tool_part_counts_once() -> None:
    event = {"type": "tool_use", "part": {"type": "tool", "tool": "read", "state": {"status": "completed"}}}
    assert parse_events(json.dumps(event))["tool_calls"] == 1


def test_cache_tokens_and_the_reported_total_are_counted() -> None:
    step = {"type": "step_finish", "part": {"type": "step-finish", "tokens": {
        "input": 1000, "output": 200, "reasoning": 50, "cache": {"read": 30000, "write": 500}}}}
    assert parse_events(json.dumps(step))["total_tokens"] == 31750
    step["part"]["tokens"]["total"] = 40000
    assert parse_events(json.dumps(step))["total_tokens"] == 40000


def test_rate_limits_are_found_on_stderr_but_not_in_repository_text() -> None:
    assert parse_events("", "ERROR provider status=429 Too Many Requests")["rate_limited"]
    quoted = {"type": "text", "part": {"type": "text", "text": "the handler returns 429 on rate limit"}}
    tool = {"type": "tool_use", "part": {"type": "tool", "state": {"output": "HTTP 429"}}}
    assert not parse_events(json.dumps(quoted) + "\n" + json.dumps(tool))["rate_limited"]


def test_budget_admits_a_run_only_while_its_reserve_fits() -> None:
    budget = Budget(cap=200_000, reserve=80_000)
    budget.check()
    budget.spend(150_000)
    with pytest.raises(BudgetExhausted):
        budget.check()


def test_prompts_differ_only_by_the_added_context() -> None:
    task = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD)
    a, b, c = (build_prompt(task, cond, "project: x") for cond in ("A", "B", "C"))
    assert a == b  # B's only difference is the server OpenCode lists as a tool
    assert "Beacon" not in a and "project: x" not in a
    assert c.startswith("Find docs.") and c.index("Find docs.") < c.index("project: x")
    assert c.replace(c[c.index("Project knowledge"):c.index(TOOL_NOTE)], "") == a
    assert all(TOOL_NOTE in p and "```json" in p for p in (a, b, c))


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def _user_config(tmp_path: Path) -> Path:
    path = tmp_path / "user-opencode" / "opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "model": "other/x",
        "provider": {"deepseek": {"options": {"baseURL": "u"}}, "other": {"options": {}}},
        "mcp": {"memory": {"type": "remote", "url": "u"}},
        "plugin": ["p"],
        "instructions": ["i.md"],
    }), encoding="utf-8")
    return path


def test_isolated_config_holds_only_the_model_provider_and_is_removed(tmp_path: Path) -> None:
    source = _user_config(tmp_path)
    before = source.read_text(encoding="utf-8")
    with isolated_config_home(source, "deepseek/deepseek-v4-flash") as home:
        config = json.loads((home / "opencode" / "opencode.json").read_text(encoding="utf-8"))
        assert not home.is_relative_to(tmp_path)  # system temp, never the results tree
    assert set(config) == {"$schema", "model", "provider"}
    assert list(config["provider"]) == ["deepseek"]
    assert not home.exists()
    assert source.read_text(encoding="utf-8") == before


def test_condition_b_config_adds_only_beacon(tmp_path: Path) -> None:
    mcp = beacon_server("py", Path("m.yaml"), "src")
    with isolated_config_home(_user_config(tmp_path), "deepseek/m", mcp) as home:
        config = json.loads((home / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert list(config["mcp"]) == ["beacon"]
    assert config["mcp"]["beacon"]["command"][-2:] == ["--manifest", "m.yaml"]
    assert config["mcp"]["beacon"]["environment"] == {"PYTHONPATH": "src"}


def test_a_missing_provider_fails_before_any_run(tmp_path: Path) -> None:
    with pytest.raises(IsolationError), isolated_config_home(_user_config(tmp_path), "nope/m"):
        pass


def test_isolated_env_drops_opencode_overrides(tmp_path: Path) -> None:
    env = isolated_env({"PATH": "p", "OPENCODE_CONFIG": "x", "OPENCODE_CONFIG_DIR": "y"}, tmp_path)
    assert env == {"PATH": "p", "XDG_CONFIG_HOME": str(tmp_path), "OPENCODE_DISABLE_CLAUDE_CODE": "1"}


# ---------------------------------------------------------------------------
# End to end with a stub instead of OpenCode
# ---------------------------------------------------------------------------

STUB = r"""
import json, os, sys
prompt = sys.stdin.read()
resumed = "--session" in sys.argv
if not resumed and "EARLY" in prompt:
    open(".stub_early", "w").write("always" if "ALWAYS_EARLY" in prompt else "once")
early_mode = open(".stub_early").read() if os.path.exists(".stub_early") else ""
if (not resumed and early_mode) or (resumed and early_mode == "always"):
    print(json.dumps({"type": "tool_use", "sessionID": "ses_1", "part": {"type": "tool", "tool": "read"}}))
    print(json.dumps({"type": "step_finish", "sessionID": "ses_1", "part": {"reason": "tool-calls", "tokens": {"input": 500, "output": 10}, "cost": 0.01}}))
    sys.exit(0)
if "ECHO_KEY" in prompt:
    print("request failed for key " + os.environ.get("OPENAI_API_KEY", ""), file=sys.stderr)
home = os.environ.get("XDG_CONFIG_HOME", "")
config = json.load(open(os.path.join(home, "opencode", "opencode.json"), encoding="utf-8"))
if "RATE_STDERR" in prompt:
    print("ERROR 2026-09-23 service=llm status=429 retrying", file=sys.stderr, flush=True)
    import time; time.sleep(30)
    sys.exit(0)
if "RATE" in prompt:
    print(json.dumps({"type": "error", "error": {"name": "APIError", "data": {"message": "429 rate limit"}}}))
    sys.exit(1)
if "BIG" in prompt:
    for _ in range(100):
        print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 50000, "output": 0}}}), flush=True)
    sys.exit(0)
record = {
    "mcp": sorted(config.get("mcp", {})),
    "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
    "tools_off": sorted(k for k, v in config.get("tools", {}).items() if v is False),
    "resumed": resumed,
    "beacon_manifest": config.get("mcp", {}).get("beacon", {}).get("command", [""])[-1],
    "config_keys": sorted(config),
    "opencode_vars": sorted(k for k in os.environ if k.startswith("OPENCODE_")),
    "own_git_root": os.path.isdir(".git"),
    "pwd_is_cwd": os.path.samefile(os.environ.get("PWD") or "/", os.getcwd()),
    "cwd_has_beacon": os.path.exists("beacon.generated.yaml"),
    "pasted": "PASTED" in prompt,
    "argv_has_prompt": any("Find docs" in a for a in sys.argv),
}
answer = {"docs": ["AGENTS.md"], "files": ["src/app.py"], "commands": [], "guardrails": [], "verdict": "", "plan": [], "citations": [{"path": "AGENTS.md", "line_start": 1, "line_end": 1}], "record": record}
print(json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": "read"}}))
print(json.dumps({"type": "text", "part": {"type": "text", "text": "```json\n" + json.dumps(answer) + "\n```"}}))
if "COSTLY" in prompt:
    for _ in range(20):
        print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 10, "output": 1}, "cost": 0.03}}), flush=True)
    sys.exit(0)
if "NOUSAGE" not in prompt:
    step = {"tokens": {"input": 1000, "output": 100}}
    if "NOCOST" not in prompt:
        step["cost"] = 0.02
    print(json.dumps({"type": "step_finish", "part": step}))
"""


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stub = tmp_path / "stub_opencode.py"
    stub.write_text(STUB, encoding="utf-8")

    def fake_build(config: RunnerConfig, pin: RepoPin) -> Path:
        manifest = config.workdir / "beacons" / pin.name / "beacon.generated.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("PASTED: yes\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(runner_mod, "build_beacon", fake_build)
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path / "should-be-dropped"))
    monkeypatch.setenv("PWD", str(tmp_path))  # a caller's directory, not the checkout
    config = RunnerConfig(
        workdir=tmp_path / "work",
        beacon_python="py",
        opencode_cmd=[sys.executable, str(stub)],
        model="deepseek/deepseek-v4-flash",  # the provider _user_config defines; independent of the default
        config_source=_user_config(tmp_path),
        budget_tokens=10_000_000,
        run_reserve_tokens=80_000,
        timeout_s=60,
    )
    return config, _pin(_repo(tmp_path))


def test_each_condition_gets_the_right_isolation(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD, reviewed=True)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "B", "C"))
    assert stopped == ""
    records = {r.condition: r.answer["record"] for r in results}
    assert records["A"]["mcp"] == [] and records["C"]["mcp"] == []
    assert records["B"]["mcp"] == ["beacon"]
    assert all(rec["opencode_vars"] == ["OPENCODE_DISABLE_CLAUDE_CODE"] for rec in records.values())
    assert all(set(rec["config_keys"]) <= {"$schema", "model", "provider", "mcp"} for rec in records.values())
    assert all(rec["own_git_root"] for rec in records.values())
    assert all(rec["pwd_is_cwd"] for rec in records.values())  # OpenCode roots itself at PWD
    assert not any(rec["argv_has_prompt"] for rec in records.values())  # prompt goes on stdin
    assert records["C"]["pasted"] and not records["A"]["pasted"] and not records["B"]["pasted"]
    # No condition ever finds a beacon in the agent's checkout.
    assert not any(rec["cwd_has_beacon"] for rec in records.values())
    assert all(r.total_tokens == 1100 and r.tool_calls == 1 and r.scores["doc_recall"] == 0.5 for r in results)
    run_dir = config.workdir / "runs" / "demo-t1-A-1"
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 3
    assert "| B | 1 |" in render(results, {"Model": "stub"})


def test_matrix_stops_at_a_rate_limit(harness) -> None:
    config, pin = harness
    ok = Task(repo="demo", task_id="ok", kind="k", prompt="fine", gold=GOLD)
    limited = Task(repo="demo", task_id="rl", kind="k", prompt="RATE", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [ok, limited, ok], ("A",))
    assert [r.task_id for r in results] == ["ok", "rl"]
    assert "rate limited" in stopped


def test_a_stderr_only_rate_limit_kills_the_run_and_stops(harness) -> None:
    config, pin = harness
    limited = Task(repo="demo", task_id="rl", kind="k", prompt="RATE_STDERR", gold=GOLD)
    ok = Task(repo="demo", task_id="ok", kind="k", prompt="fine", gold=GOLD)
    started = time.monotonic()
    results, stopped = run_matrix(config, {"demo": pin}, [limited, ok], ("A",))
    assert "rate limited" in stopped and len(results) == 1
    assert time.monotonic() - started < 25  # killed, not waited out


def test_a_run_past_its_reserve_is_killed_and_stops_the_matrix(harness) -> None:
    config, pin = harness
    big = Task(repo="demo", task_id="big", kind="k", prompt="BIG", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [big, big], ("A",))
    assert "reserve" in stopped and len(results) == 1
    assert 80_000 < results[0].total_tokens <= 130_000  # overshoot is at most one step


def test_a_run_without_usage_stops_the_matrix(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="nu", kind="k", prompt="NOUSAGE", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task, task], ("A",))
    assert "no token usage" in stopped and len(results) == 1


def test_a_relative_workdir_still_gives_an_absolute_pwd(harness, monkeypatch: pytest.MonkeyPatch) -> None:
    config, pin = harness
    monkeypatch.chdir(config.workdir.parent)
    config.workdir = Path(config.workdir.name)  # relative, as the CLI passes it
    task = Task(repo="demo", task_id="rel", kind="k", prompt="Find docs.", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "B"))
    assert stopped == "" and all(r.answer["record"]["pwd_is_cwd"] for r in results)
    # B's Beacon server starts in the checkout, so its manifest path must be absolute.
    assert Path(results[1].answer["record"]["beacon_manifest"]).is_absolute()


def test_env_file_keys_reach_opencode_only_and_are_redacted(harness, tmp_path: Path) -> None:
    config, pin = harness
    secret = "sk-test-0123456789abcdef"
    env_file = tmp_path / "bench.env"
    env_file.write_text(f'OTHER=1\nOPENAI_API_KEY="{secret}"\nEMPTY_API_KEY=\n', encoding="utf-8")
    config.env_file = env_file
    config.model = "openai/gpt-6-luna"  # not in the user's config: OpenCode's built-in provider
    task = Task(repo="demo", task_id="key", kind="k", prompt="Find docs. ECHO_KEY", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",))
    record = results[0].answer["record"]
    assert stopped == "" and record["has_openai_key"] and "provider" not in record["config_keys"]
    run_dir = config.workdir / "runs" / "demo-key-A-1"
    for name in ("events.jsonl", "stderr.log", "result.json"):
        assert secret not in (run_dir / name).read_text(encoding="utf-8")
    assert "<redacted>" in (run_dir / "stderr.log").read_text(encoding="utf-8")


def test_dollar_cap_admits_runs_only_while_the_reserve_fits(harness) -> None:
    config, pin = harness
    config.budget_usd, config.run_reserve_usd = 0.15, 0.10
    first = Task(repo="demo", task_id="d1", kind="k", prompt="fine", gold=GOLD)
    second = Task(repo="demo", task_id="d2", kind="k", prompt="fine", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [first, second], ("A", "B", "C"))
    # $0.02 per run: admitted at $0.00, $0.02 and $0.04; the 4th stops as $0.06 + $0.10 > $0.15.
    assert len(results) == 3 and "$0.15 cap" in stopped
    assert all(r.cost_usd == 0.02 for r in results)
    assert "Total cost:** $0.0600" in render(results, {"Model": "stub"}, stopped)


def test_dollar_mode_with_token_limits_off_runs_everything(harness) -> None:
    config, pin = harness
    # What the CLI builds for --budget-usd without token options.
    config.budget_usd, config.run_reserve_usd = 1.0, 0.10
    config.budget_tokens = config.run_reserve_tokens = None
    tasks = [Task(repo="demo", task_id=f"u{i}", kind="k", prompt="fine", gold=GOLD) for i in range(2)]
    results, stopped = run_matrix(config, {"demo": pin}, tasks, ("A", "B", "C"))
    assert stopped == "" and len(results) == 6


def test_a_run_past_its_dollar_reserve_is_killed(harness) -> None:
    config, pin = harness
    config.budget_usd, config.run_reserve_usd = 5.0, 0.10
    task = Task(repo="demo", task_id="c", kind="k", prompt="COSTLY", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task, task], ("A",))
    assert "$0.10" in stopped and len(results) == 1
    assert 0.10 < results[0].cost_usd <= 0.13  # overshoot is at most one step


def test_in_dollar_mode_a_run_without_cost_stops_the_matrix(harness) -> None:
    config, pin = harness
    config.budget_usd = 5.0
    task = Task(repo="demo", task_id="n", kind="k", prompt="NOCOST", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task, task], ("A",))
    assert "no cost" in stopped and len(results) == 1


def test_rescore_recomputes_scores_from_saved_answers(harness) -> None:
    from archolith_bench.beacon_eval.runner import rescore

    config, pin = harness
    task = Task(repo="demo", task_id="rs", kind="k", prompt="fine", gold=GOLD)
    run_matrix(config, {"demo": pin}, [task], ("A",))
    saved = config.workdir / "runs" / "demo-rs-A-1" / "result.json"
    data = json.loads(saved.read_text(encoding="utf-8"))
    original = data["scores"]
    data["scores"] = {"answered": 0.0}
    saved.write_text(json.dumps(data), encoding="utf-8")
    stricter = Task(repo="demo", task_id="rs", kind="k", prompt="fine", gold=Gold(docs=("AGENTS.md",)))
    results = rescore(config.workdir, [stricter])
    assert len(results) == 1 and results[0].scores["doc_recall"] == 1.0 != original["doc_recall"]
    assert json.loads(saved.read_text(encoding="utf-8"))["scores"] == results[0].scores
    assert len((config.workdir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_a_session_that_exits_after_tool_calls_is_resumed(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="early", kind="k", prompt="Find docs. EARLY", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A",))
    result = results[0]
    assert stopped == "" and result.resumes == 1
    assert result.answer is not None and result.answer["record"]["resumed"]
    assert result.total_tokens == 1610 and result.tool_calls == 2  # both sessions counted
    events = (config.workdir / "runs" / "demo-early-A-1" / "events.jsonl").read_text(encoding="utf-8")
    assert events.count("step_finish") == 2  # the resumed session appended, not overwrote


def test_resumes_stop_after_the_limit(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="never", kind="k", prompt="EARLY ALWAYS_EARLY", gold=GOLD)
    results, _ = run_matrix(config, {"demo": pin}, [task], ("A",))
    assert results[0].resumes == 2 and results[0].answer is None


def test_condition_d_has_beacon_and_no_builtin_tools(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="d", kind="k", prompt="Find docs.", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "D"))
    records = {r.condition: r.answer["record"] for r in results}
    assert stopped == ""
    assert records["D"]["mcp"] == ["beacon"] and records["A"]["mcp"] == []
    assert records["A"]["tools_off"] == []
    assert {"read", "grep", "glob", "bash", "write", "edit", "webfetch"} <= set(records["D"]["tools_off"])
    assert not records["D"]["pasted"]


def test_d_is_opt_in() -> None:
    from archolith_bench.beacon_eval import CONDITIONS, DEFAULT_CONDITIONS

    assert "D" in CONDITIONS and "D" not in DEFAULT_CONDITIONS


def test_matrix_stops_at_the_budget(harness) -> None:
    config, pin = harness
    config.budget_tokens = 81_000  # the 80k reserve fits once; 1.1k used + 80k then exceeds it
    task = Task(repo="demo", task_id="t", kind="k", prompt="fine", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "B", "C"))
    assert len(results) == 1
    assert "cap" in stopped


def test_grounding_check_flags_uncited_items_bad_ranges_and_missing_quotes(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.grounding import check_task_file

    checkout = _repo(tmp_path)
    task = {
        "gold": {"docs": ["AGENTS.md"], "files": ["src/app.py"], "verdict": "current", "risky": ["rm -rf"]},
        "gold_citations": [
            {"item": "AGENTS.md", "path": "AGENTS.md", "line_start": 2, "line_end": 2, "quote": "Never commit secrets."},
            {"item": "current", "path": "AGENTS.md", "line_start": 1, "line_end": 9, "quote": "x"},
            {"item": "other", "path": "missing.md", "line_start": 1, "line_end": 1, "quote": "x"},
        ],
    }
    path = tmp_path / "t.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    problems = check_task_file(path, checkout)
    assert "files item has no citation: src/app.py" in problems
    assert any(p.startswith("bad line range 1-9") for p in problems)
    assert "cited path missing: missing.md" in problems
    assert len(problems) == 3


def test_a_gold_command_may_be_a_prefix(tmp_path: Path) -> None:
    gold = Gold(commands=("ruff check --select F811,F821,ASYNC",))
    answer = {"commands": ["ruff check --select F811,F821,ASYNC src/menhir/x.py"]}
    assert score(answer, gold, tmp_path)["command_recall"] == 1.0
    assert score({"commands": ["ruff check ."]}, gold, tmp_path)["command_recall"] == 0.0


def test_an_added_flag_fails_unless_the_gold_allows_it(tmp_path: Path) -> None:
    gold = Gold(commands=("ruff check --select F811,F821,ASYNC",))
    wrong = {"commands": ["ruff check --select F811,F821,ASYNC --fix --unsafe-fixes ."]}
    assert score(wrong, gold, tmp_path)["command_recall"] == 0.0
    pytest_gold = Gold(commands=('python -m pytest --run-online -m "online and not needs_llm" -q',),
                       allowed_flags=("-x", "--maxfail"))
    ok = {"commands": ['python -m pytest --run-online -m "online and not needs_llm" -q -x --maxfail=2 tests/a.py']}
    assert score(ok, pytest_gold, tmp_path)["command_recall"] == 1.0
    extra = {"commands": ['python -m pytest --run-online -m "online and not needs_llm" -q --lf']}
    assert score(extra, pytest_gold, tmp_path)["command_recall"] == 0.0
    # Token prefix, not string prefix: "-qx" is not "-q" plus arguments.
    fused = {"commands": ['python -m pytest --run-online -m "online and not needs_llm" -qx']}
    assert score(fused, pytest_gold, tmp_path)["command_recall"] == 0.0


def test_a_guardrail_is_met_by_one_entry_holding_any_accepted_wording(tmp_path: Path) -> None:
    gold = Gold(guardrails=(("online tests opt-in", "live tests --run-online"), "disposable test database"))
    paraphrase = {"guardrails": [
        "Live tests only run when you pass --run-online.",
        "Point graph tests at a disposable Neo4j test database.",
    ]}
    assert score(paraphrase, gold, tmp_path)["guardrail_recall"] == 1.0
    # Words spread across unrelated entries do not add up.
    spread = {"guardrails": ["Online docs are generated.", "Tests are fast.", "Opt-in telemetry."]}
    assert score(spread, gold, tmp_path)["guardrail_recall"] == 0.0


def test_risky_steps_in_the_plan_count_but_warnings_do_not(tmp_path: Path) -> None:
    gold = Gold(risky=("git add -A",))

    def risky(answer: dict) -> float:
        return score(answer, gold, tmp_path)["risky_false_positive"]

    assert risky({"commands": [], "plan": ["Stage everything with git add -A and commit."]}) == 1.0
    assert risky({"commands": [], "plan": ["Never use git add -A; stage named files."]}) == 0.0
    assert risky({"commands": [], "plan": ["Stage named files instead of git add -A."]}) == 0.0
    assert risky({"commands": [], "plan": ["Don’t run git add -A."]}) == 0.0
    # "not" inside another word ("note") is not a prohibition.
    assert risky({"commands": [], "plan": ["Take note, then git add -A."]}) == 1.0
    assert risky({"commands": ["git add -A"], "plan": []}) == 1.0


def test_acceptable_files_keep_precision_without_counting_for_recall(tmp_path: Path) -> None:
    gold = Gold(files=("src/core.py",), acceptable_files=("src/adapter.py",))
    with_extra = score({"files": ["src/core.py", "src/adapter.py"]}, gold, tmp_path)
    assert (with_extra["file_recall"], with_extra["file_precision"]) == (1.0, 1.0)
    unlisted = score({"files": ["src/core.py", "src/unrelated.py"]}, gold, tmp_path)
    assert unlisted["file_precision"] == 0.5
    only_extra = score({"files": ["src/adapter.py"]}, gold, tmp_path)
    assert (only_extra["file_recall"], only_extra["file_precision"]) == (0.0, 1.0)


def test_grounding_flags_a_missing_acceptable_file(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.grounding import check_task_file

    (tmp_path / "real.py").write_text("x\n", encoding="utf-8")
    task = tmp_path / "t.json"
    (tmp_path / "archive").mkdir()
    task.write_text(json.dumps({"gold": {"acceptable_files": [
        "real.py", "archive/new.md", "nowhere/new.md", "../outside.py"]}}), encoding="utf-8")
    # A new file is acceptable when its folder exists; a missing folder or an escape is not.
    assert check_task_file(task, tmp_path) == [
        "acceptable file missing: nowhere/new.md", "acceptable file missing: ../outside.py"]


def test_dot_directory_citations_are_valid_locations(tmp_path: Path) -> None:
    (tmp_path / ".agent").mkdir()
    (tmp_path / ".agent" / "README.md").write_text("a\nb\nc\n", encoding="utf-8")
    answer = {"citations": [{"path": ".agent/README.md", "line_start": 1, "line_end": 2},
                            {"path": "./.agent/README.md", "line_start": 3, "line_end": 3}]}
    assert score(answer, Gold(), tmp_path)["citation_location_validity"] == 1.0


def test_a_trailing_note_after_a_path_still_matches(tmp_path: Path) -> None:
    gold = Gold(docs=(".agent/README.md",), files=("scripts/core.py",))
    answer = {"docs": ["./.agent/README.md"],
              "files": ["scripts/core.py (canonical triage implementation)", "`scripts/other.py` (nearby)"]}
    scores = score(answer, gold, tmp_path)
    assert (scores["doc_recall"], scores["file_recall"], scores["file_precision"]) == (1.0, 1.0, 0.5)


def test_labelled_and_grouped_file_answers_still_match(tmp_path: Path) -> None:
    gold = Gold(files=("scripts/core.py", "tests/test_core.py"))
    labelled = {"files": ["Implementation: scripts/core.py", "Shared triage tests: `tests/test_core.py`"]}
    grouped = {"files": {"source": ["scripts/core.py"], "tests": ["tests/test_core.py"]}}
    for answer in (labelled, grouped):
        scores = score(answer, gold, tmp_path)
        assert (scores["file_recall"], scores["file_precision"]) == (1.0, 1.0)
    # A sentence with a colon is not treated as a labelled path.
    assert score({"files": ["Note: see the docs for details"]}, gold, tmp_path)["file_recall"] == 0.0


def test_a_citation_without_lines_is_not_a_valid_location(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("a\nb\n", encoding="utf-8")
    answer = {"citations": [{"path": "x.md"}, {"path": "x.md", "line_start": 2}]}
    assert score(answer, Gold(), tmp_path)["citation_location_validity"] == 0.5


def test_evidence_recall_counts_gold_spans_overlapped_in_the_same_file(tmp_path: Path) -> None:
    gold = Gold(evidence=(("docs/a.md", 10, 12), ("docs/b.md", 5, 5)))
    answer = {"citations": [
        {"path": "./docs/a.md", "line_start": 12, "line_end": 20},  # overlaps a.md 10-12
        {"path": "docs/c.md", "line_start": 5, "line_end": 5},  # right lines, wrong file
    ]}
    assert score(answer, gold, tmp_path)["evidence_recall"] == 0.5
    assert "evidence_recall" not in score(answer, Gold(), tmp_path)


def test_gold_evidence_is_loaded_once_per_distinct_span(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.models import load_task

    task = tmp_path / "t.json"
    span = {"path": "a.md", "line_start": 1, "line_end": 2, "quote": "q"}
    task.write_text(json.dumps({
        "repo": "r", "task_id": "t", "kind": "k", "prompt": "p", "gold": {},
        "gold_citations": [dict(span, item="x"), dict(span, item="y"), {"item": "z", "path": "b.md"}],
    }), encoding="utf-8")
    assert load_task(task).gold.evidence == (("a.md", 1, 2),)


def test_a_point_is_met_by_one_findings_or_plan_entry(tmp_path: Path) -> None:
    gold = Gold(points=(("backend owns the runtime", "menhir serve owns runtime"),
                        ("replaced per-client runtime", "each client started its own runtime")))
    answer = {"findings": ["The `menhir serve` backend owns the runtime; stdio is a thin proxy."],
              "plan": ["Note that previously each client started its own runtime."]}
    assert score(answer, gold, tmp_path)["point_recall"] == 1.0
    spread = {"findings": ["The backend is fast.", "It owns a cache.", "The runtime is Python."]}
    assert score(spread, gold, tmp_path)["point_recall"] == 0.0
    assert "point_recall" not in score(answer, Gold(), tmp_path)


def test_every_condition_asks_for_findings() -> None:
    task = Task(repo="demo", task_id="t1", kind="decision", prompt="Who owns it?", gold=GOLD)
    assert all('"findings"' in build_prompt(task, cond, "x") for cond in ("A", "B", "C"))


def test_grounding_requires_a_citation_for_each_point(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.grounding import check_task_file

    task = tmp_path / "t.json"
    task.write_text(json.dumps({"gold": {"points": [["backend owns the runtime", "alt"]]}}), encoding="utf-8")
    assert check_task_file(task, tmp_path) == ["points item has no citation: backend owns the runtime"]


def test_grounding_takes_the_first_wording_as_the_cited_guardrail(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.grounding import check_task_file

    (tmp_path / "AGENTS.md").write_text("Online tests are opt-in.\n", encoding="utf-8")
    task = tmp_path / "t.json"
    task.write_text(json.dumps({
        "gold": {"guardrails": [["online tests opt-in", "live tests --run-online"]]},
        "gold_citations": [{"item": "online tests opt-in", "path": "AGENTS.md", "line_start": 1,
                            "line_end": 1, "quote": "Online tests are opt-in."}],
    }), encoding="utf-8")
    assert check_task_file(task, tmp_path) == []


def test_why_task_set_loads_apart_and_gold_wordings_score(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.models import load_tasks

    root = Path(runner_mod.__file__).parent
    main = load_tasks(root / "tasks")
    why = load_tasks(root / "why_tasks")
    assert len(why) == 8 and all(t.kind == "why" and t.reviewed for t in why)
    assert not {t.task_id for t in why} & {t.task_id for t in main}
    for task in why:
        findings = [p if isinstance(p, str) else p[0] for p in task.gold.points]
        scores = score({"findings": findings, "files": list(task.gold.files)}, task.gold, tmp_path)
        assert scores["point_recall"] == 1.0, task.task_id
        assert scores.get("risky_false_positive", 0.0) == 0.0, task.task_id


def _ready_server(body: dict):
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = json.dumps(body).encode()
            self.send_response(200 if self.path == "/api/ready" else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/mcp-http"


def test_memory_server_config_references_key_by_env_only(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.isolation import MEMORY_KEY_ENV, memory_server

    source = tmp_path / "opencode.json"
    source.write_text(json.dumps({"provider": {"openai": {}}}), encoding="utf-8")
    with isolated_config_home(source, "openai/x", memory_server("http://127.0.0.1:1/mcp-http")) as home:
        written = json.loads((home / "opencode" / "opencode.json").read_text(encoding="utf-8"))
    assert list(written["mcp"]) == ["menhir"]
    server = written["mcp"]["menhir"]
    assert server["type"] == "remote" and server["url"].endswith("/mcp-http")
    assert server["headers"]["Authorization"] == "Bearer {env:" + MEMORY_KEY_ENV + "}"


def test_memory_ready_check_accepts_reads_ready_and_refuses_degraded() -> None:
    from archolith_bench.beacon_eval.runner import MemoryNotReady, check_memory_ready

    ok, ok_url = _ready_server({"status": "ready", "capabilities": {"reads_ready": True}})
    bad, bad_url = _ready_server(
        {"status": "degraded", "capabilities": {"reads_ready": False}, "failures": ["no embedder"]}
    )
    try:
        check_memory_ready(ok_url)
        with pytest.raises(MemoryNotReady, match="no embedder"):
            check_memory_ready(bad_url)
        with pytest.raises(MemoryNotReady):
            check_memory_ready("not a url")
    finally:
        ok.shutdown()
        bad.shutdown()


def test_matrix_with_m_stops_before_any_run_when_memory_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad, bad_url = _ready_server({"status": "degraded", "capabilities": {"reads_ready": False}})
    called: list[str] = []
    monkeypatch.setattr(runner_mod, "run_one", lambda *a, **k: called.append("run"))
    task = Task(repo="r", task_id="t", kind="why", prompt="q", reviewed=True, gold=Gold())
    config = RunnerConfig(workdir=tmp_path, beacon_python=sys.executable, opencode_cmd=["x"],
                          memory_url=bad_url, memory_key="k" * 12)
    try:
        results, stopped = run_matrix(config, {"r": RepoPin("r", "", "0" * 40)}, [task], ("A", "M"))
    finally:
        bad.shutdown()
    assert results == [] and called == [] and "cannot serve reads" in stopped


def test_condition_m_without_memory_settings_is_refused(tmp_path: Path) -> None:
    task = Task(repo="r", task_id="t", kind="why", prompt="q", reviewed=True, gold=Gold())
    config = RunnerConfig(workdir=tmp_path, beacon_python=sys.executable, opencode_cmd=["x"])
    with pytest.raises(ValueError, match="memory_url"):
        runner_mod.run_one(config, RepoPin("r", "", "0" * 40), task, "M", 1)


def _judge_fixture(tmp_path: Path, findings: list[str]) -> tuple[Path, Path]:
    from dataclasses import asdict

    from archolith_bench.beacon_eval.models import RunResult

    task_root = tmp_path / "why_tasks"
    (task_root / "r").mkdir(parents=True)
    (task_root / "r" / "t.json").write_text(json.dumps({
        "repo": "r", "task_id": "t", "kind": "why", "prompt": "Why was X rejected?",
        "gold": {"points": [["dedupe dominates cost", "dedupe 62%"], ["torch image footprint"]]},
        "memory_citations": [
            {"item": "dedupe dominates cost", "episode_uuid": "e1", "quote": "dedupe is 62% of input"},
        ],
    }), encoding="utf-8")
    workdir = tmp_path / "work"
    run = workdir / "runs" / "r-t-M-1"
    run.mkdir(parents=True)
    result = RunResult(repo="r", task_id="t", condition="M", repeat=1,
                       answer={"findings": findings, "plan": []}, final_text="")
    (run / "result.json").write_text(json.dumps(asdict(result)), encoding="utf-8")
    return workdir, task_root


def _fake_call(replies: list[dict], seen: list[str]):
    def call(messages):
        seen.append(messages[-1]["content"])
        return json.dumps(replies[len(seen) - 1]), {"prompt_tokens": 1000, "completion_tokens": 50}
    return call


def test_judge_counts_only_verdicts_whose_evidence_is_in_the_answer(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.judge import judge_workdir

    workdir, task_root = _judge_fixture(
        tmp_path, ["About 62% of paid input went to deduplication, 17% to extraction."]
    )
    seen: list[str] = []
    replies = [
        {"met": True, "evidence": "62% of paid input went to deduplication", "reason": "same"},
        {"met": True, "evidence": "PyTorch would bloat the image", "reason": "invented"},
    ]
    scores, spent = judge_workdir(workdir, task_root, _fake_call(replies, seen))
    assert scores == {"r-t-M-1": 0.5}
    cached = json.loads((workdir / "runs" / "r-t-M-1" / "judged.json").read_text(encoding="utf-8"))
    assert [p["met"] for p in cached["points"]] == [True, False]
    assert [p["raw_met"] for p in cached["points"]] == [True, True]
    assert spent > 0
    # The judge sees the question, point, reference and answer, never the run or condition.
    assert "dedupe is 62% of input" in seen[0] and "Why was X rejected?" in seen[0]
    assert "r-t-M-1" not in seen[0] and "condition" not in seen[0].lower()


def test_judge_reuses_cached_verdicts_without_calls(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.judge import judge_workdir

    workdir, task_root = _judge_fixture(tmp_path, ["deduplication dominated cost at 62 percent"])
    seen: list[str] = []
    replies = [{"met": False, "evidence": "", "reason": ""}] * 2
    judge_workdir(workdir, task_root, _fake_call(replies, seen))
    assert len(seen) == 2
    scores, spent = judge_workdir(workdir, task_root, _fake_call(replies, seen))
    assert len(seen) == 2 and spent == 0.0 and scores == {"r-t-M-1": 0.0}


def test_judge_stops_on_rate_limit_and_before_passing_the_cap(tmp_path: Path) -> None:
    from archolith_bench.beacon_eval.judge import (
        JudgeBudgetExhausted,
        JudgeRateLimited,
        judge_workdir,
    )

    workdir, task_root = _judge_fixture(tmp_path, ["something"])

    def limited(messages):
        raise JudgeRateLimited("judge rate limited (HTTP 429)")

    with pytest.raises(JudgeRateLimited):
        judge_workdir(workdir, task_root, limited)
    assert not (workdir / "runs" / "r-t-M-1" / "judged.json").exists()
    seen: list[str] = []
    with pytest.raises(JudgeBudgetExhausted):
        judge_workdir(workdir, task_root, _fake_call([], seen), budget_usd=0.001)
    assert seen == []
