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
if "NOUSAGE" not in prompt:
    print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 1000, "output": 100}}}))
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
