"""Offline tests for the Beacon agent-task harness. No model or network is ever called."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from archolith_bench.beacon_eval import runner as runner_mod
from archolith_bench.beacon_eval.isolation import beacon_overlay, stripped_config
from archolith_bench.beacon_eval.models import Gold, RepoPin, Task
from archolith_bench.beacon_eval.report import render
from archolith_bench.beacon_eval.runner import (
    Budget,
    BudgetExhausted,
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
    assert scores["citation_validity"] == 0.5


def test_no_answer_scores_as_unanswered(tmp_path: Path) -> None:
    assert extract_answer("no json here") is None
    assert score(None, GOLD, tmp_path) == {"answered": 0.0}


def test_risky_content_is_flagged(tmp_path: Path) -> None:
    risky = dict(ANSWER, plan=["rm -rf build"])
    assert score(risky, GOLD, tmp_path)["risky_false_positive"] == 1.0


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


def test_budget_stops_before_the_cap() -> None:
    budget = Budget(cap=200_000)
    budget.check()
    budget.spend(150_000)
    with pytest.raises(BudgetExhausted):
        budget.check()


def test_prompts_differ_only_by_condition() -> None:
    task = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD)
    a, b, c = (build_prompt(task, cond, "project: x") for cond in ("A", "B", "C"))
    assert "Beacon" not in a and "beacon.generated.yaml" not in a
    assert "Beacon MCP server" in b and "project: x" not in b
    assert "project: x" in c
    assert all("```json" in p for p in (a, b, c))


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_stripped_config_disables_every_mcp_server_and_leaves_the_source(tmp_path: Path) -> None:
    source = tmp_path / "opencode"
    (source / "node_modules" / "pkg").mkdir(parents=True)
    (source / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    original = {"model": "m", "mcp": {"memory": {"type": "remote", "url": "u"}, "vps": {"type": "local"}}}
    (source / "opencode.json").write_text(json.dumps(original), encoding="utf-8")
    run_dir = tmp_path / "run"
    with stripped_config(source, run_dir) as config_dir:
        copied = json.loads((config_dir / "opencode.json").read_text(encoding="utf-8"))
        assert all(server["enabled"] is False for server in copied["mcp"].values())
        assert (config_dir / "node_modules" / "pkg" / "index.js").is_file()
    assert not (run_dir / "opencode-config").exists()
    assert json.loads((source / "opencode.json").read_text(encoding="utf-8")) == original
    assert (source / "node_modules" / "pkg" / "index.js").is_file()  # link removed, target kept


def test_beacon_overlay_adds_only_beacon() -> None:
    overlay = json.loads(beacon_overlay("py", Path("m.yaml"), "src"))
    assert list(overlay["mcp"]) == ["beacon"]
    assert overlay["mcp"]["beacon"]["command"][-2:] == ["--manifest", "m.yaml"]
    assert overlay["mcp"]["beacon"]["environment"] == {"PYTHONPATH": "src"}


# ---------------------------------------------------------------------------
# End to end with a stub instead of OpenCode
# ---------------------------------------------------------------------------

STUB = r"""
import json, os, sys
config_dir = os.environ.get("OPENCODE_CONFIG_DIR", "")
overlay = os.environ.get("OPENCODE_CONFIG_CONTENT", "")
prompt = sys.argv[-1]
if "RATE" in prompt:
    print(json.dumps({"error": "429 rate limit"}))
    sys.exit(1)
record = {"config_dir_set": bool(config_dir), "overlay": overlay, "cwd_has_beacon": os.path.exists("beacon.generated.yaml"), "pasted": "PASTED" in prompt}
answer = {"docs": ["AGENTS.md"], "files": ["src/app.py"], "commands": [], "guardrails": [], "verdict": "", "plan": [], "citations": [{"path": "AGENTS.md", "line_start": 1, "line_end": 1}], "record": record}
print(json.dumps({"type": "text", "part": {"type": "text", "text": "```json\n" + json.dumps(answer) + "\n```"}}))
print(json.dumps({"type": "step_finish", "part": {"tokens": {"input": 1000, "output": 100}}}))
"""


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stub = tmp_path / "stub_opencode.py"
    stub.write_text(STUB, encoding="utf-8")
    source = tmp_path / "opencode"
    source.mkdir()
    (source / "opencode.json").write_text(json.dumps({"mcp": {"memory": {}}}), encoding="utf-8")

    def fake_build(config: RunnerConfig, pin: RepoPin) -> Path:
        manifest = config.workdir / "beacons" / pin.name / "beacon.generated.yaml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("PASTED: yes\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(runner_mod, "build_beacon", fake_build)
    config = RunnerConfig(
        workdir=tmp_path / "work",
        beacon_python="py",
        opencode_cmd=[sys.executable, str(stub)],
        config_source=source,
        budget_tokens=10_000_000,
    )
    return config, _pin(_repo(tmp_path))


def test_each_condition_gets_the_right_isolation(harness) -> None:
    config, pin = harness
    task = Task(repo="demo", task_id="t1", kind="docs_and_files", prompt="Find docs.", gold=GOLD, reviewed=True)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "B", "C"))
    assert stopped == ""
    records = {r.condition: r.answer["record"] for r in results}
    assert all(rec["config_dir_set"] for rec in records.values())
    assert records["A"]["overlay"] == "" and not records["A"]["pasted"]
    assert json.loads(records["B"]["overlay"])["mcp"].keys() == {"beacon"}
    assert records["C"]["pasted"] and records["C"]["overlay"] == ""
    # No condition ever finds a beacon in the agent's checkout.
    assert not any(rec["cwd_has_beacon"] for rec in records.values())
    assert all(r.total_tokens == 1100 and r.scores["doc_recall"] == 0.5 for r in results)
    assert "| B | 1 |" in render(results, {"Model": "stub"})


def test_matrix_stops_at_a_rate_limit(harness) -> None:
    config, pin = harness
    ok = Task(repo="demo", task_id="ok", kind="k", prompt="fine", gold=GOLD)
    limited = Task(repo="demo", task_id="rl", kind="k", prompt="RATE", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [ok, limited, ok], ("A",))
    assert len(results) == 1
    assert "rate limited" in stopped


def test_matrix_stops_at_the_budget(harness) -> None:
    config, pin = harness
    config.budget_tokens = 81_000  # 80k estimate fits once; 1.1k used + 80k then exceeds it
    task = Task(repo="demo", task_id="t", kind="k", prompt="fine", gold=GOLD)
    results, stopped = run_matrix(config, {"demo": pin}, [task], ("A", "B", "C"))
    assert len(results) == 1
    assert "cap" in stopped
