"""HarnessSprintExecutor: archolith-harness in place of OpenCode, offline (scripted Responses replies)."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest
from cth_flex import Deferred, ErrorInfo, Kind, Stopped, TierMismatch

from archolith_bench.beacon_eval.runner import RateLimited
from archolith_bench.engineering_evidence.harness_executor import HarnessSprintExecutor
from archolith_bench.engineering_evidence.live_executor import LiveSprintConfig
from archolith_bench.engineering_evidence.sprint import NOTES_FILE, SprintFixture
from archolith_harness.tools import ToolResult

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "sprint_simonw_llm_764dc386.json"
SOURCE = "llm/default_plugins/openai_models.py"


def _step(*, items, input_tokens=1000, cached=0, output=50, reasoning=10, tier="flex"):
    return {
        "output": items,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": output,
            "output_tokens_details": {"reasoning_tokens": reasoning},
        },
        "service_tier": tier,
        "status": "completed",
    }


def _call(name, args, call_id="c1"):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args), "call_id": call_id}


def _message(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


class _Send:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _FakeMcp:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def schemas(self):
        return [{"type": "function", "name": "add_memory", "description": "save",
                 "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}]

    def call(self, name, args):
        self.calls.append((name, args))
        return ToolResult(output="saved", ok=True)

    def close(self):
        self.closed = True


def _executor(tmp_path: Path, monkeypatch, send, *, mcp=None, connects=None, flex_mode="fallback", **cfg):
    monkeypatch.setattr(
        "archolith_bench.beacon_eval.isolation.default_data_source", lambda: tmp_path / "empty_opencode_data"
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    src_cfg = tmp_path / "opencode.json"
    src_cfg.write_text(json.dumps({"provider": {"openai": {"options": {}}}}), encoding="utf-8")

    def fake_export(pin, dest: Path, cache: Path) -> Path:
        f = dest / SOURCE
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# routes through the Responses API by default\n", encoding="utf-8")
        return dest

    def connect(url, allow, headers=None):
        if connects is not None:
            connects.append((url, allow, headers))
        return mcp

    return HarnessSprintExecutor(
        LiveSprintConfig(work_dir=tmp_path / "work", cache_dir=tmp_path / "cache", source_config=src_cfg, **cfg),
        flex_mode=flex_mode,
        send_fn=send,
        mcp_connect_fn=connect,
        export_fn=fake_export,
        resolve_cmd_fn=lambda: pytest.fail("the harness must not resolve an OpenCode binary"),
        check_memory_fn=lambda url: None,
        notify_file_dirty_fn=lambda *a, **k: {},
        post_fn=lambda *a, **k: [],
        sleep_fn=lambda s: None,
    )


def _run(ex, arm, *, index=0):
    fixture = SprintFixture.from_file(FIXTURE_PATH)
    return ex.run_session(fixture=fixture, session=fixture.sessions[index], arm=arm, session_index=index,
                          namespace="ns", repeat=1)


def test_session_usage_maps_onto_the_sprint_record(tmp_path: Path, monkeypatch) -> None:
    send = _Send([
        _step(items=[_call("read_file", {"path": SOURCE})], input_tokens=2000, cached=1500, output=40, reasoning=30),
        _step(items=[_message("the answer")], input_tokens=2200, cached=2000, output=100, reasoning=20),
    ])
    ex = _executor(tmp_path, monkeypatch, send, gzip_events=True)
    res = _run(ex, "no_menhir")

    assert res.response_text == "the answer" and res.error == "" and not res.censored
    # Uncached input is the primary metric: input_tokens excludes cache reads, as OpenCode's does.
    assert (res.input_tokens, res.cache_read_tokens, res.cache_write_tokens) == (700, 3500, 0)
    assert (res.output_tokens, res.reasoning_tokens) == (90, 50)  # visible output excludes reasoning
    assert res.cost_usd == pytest.approx((700 * 0.05 + 3500 * 0.005 + 140 * 0.25) / 1e6)
    assert (res.turns, res.tool_calls, res.files_read) == (2, 1, [SOURCE])
    # The request carries the prompt the shared path wrote, the bare model id, and Flex.
    first = send.requests[0]
    assert first["model"] == "gpt-6-luna" and first["service_tier"] == "flex"
    assert first["input"][0]["content"] == (res.run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert not (res.run_dir / "events.jsonl").exists()
    with gzip.open(res.run_dir / "events.jsonl.gz", "rt", encoding="utf-8") as fh:
        kinds = [json.loads(line)["kind"] for line in fh]
    assert kinds == ["step", "tool", "step"]
    agent = json.loads((res.run_dir / "agent.json").read_text(encoding="utf-8"))
    assert agent["agent"] == "archolith-harness" and agent["served_tiers"] == ["flex"]


def test_notes_arm_may_write_only_the_notes_file(tmp_path: Path, monkeypatch) -> None:
    send = _Send([
        _step(items=[_call("write_file", {"path": NOTES_FILE, "content": "note\n"}, "a"),
                     _call("write_file", {"path": SOURCE, "content": "x"}, "b")]),
        _step(items=[_message("done")]),
    ])
    ex = _executor(tmp_path, monkeypatch, send)
    res = _run(ex, "notes_file")
    checkout = res.repo_root
    assert (checkout / NOTES_FILE).read_text(encoding="utf-8") == "note\n"
    assert "Responses API" in (checkout / SOURCE).read_text(encoding="utf-8")  # untouched
    assert res.notes_violations == [] and res.error == ""


def test_no_memory_arm_gets_no_write_tool_and_no_mcp(tmp_path: Path, monkeypatch) -> None:
    send = _Send([_step(items=[_message("done")])])
    connects: list = []
    ex = _executor(tmp_path, monkeypatch, send, mcp=_FakeMcp(), connects=connects)
    _run(ex, "no_menhir")
    names = {tool["name"] for tool in send.requests[0]["tools"]}
    assert "write_file" not in names and "add_memory" not in names and connects == []


def test_menhir_arm_connects_counts_saves_and_closes(tmp_path: Path, monkeypatch) -> None:
    mcp, connects = _FakeMcp(), []
    send = _Send([
        _step(items=[_call("add_memory", {"content": "decision"})]),
        _step(items=[_message("done")]),
    ])
    ex = _executor(tmp_path, monkeypatch, send, mcp=mcp, connects=connects,
                   menhir_mcp_url="http://127.0.0.1:1/mcp-http", menhir_api_key="k")
    res = _run(ex, "menhir_longitudinal")
    assert connects == [("http://127.0.0.1:1/mcp-http", None, {"Authorization": "Bearer k"})]
    assert mcp.calls == [("add_memory", {"content": "decision"})] and mcp.closed
    assert (res.memories_saved, res.todos_saved) == (1, 0)


@pytest.mark.parametrize("error", [
    Stopped("stopped: quota", ErrorInfo(Kind.QUOTA, status=429, code="insufficient_quota")),
    Stopped("stopped: auth", ErrorInfo(Kind.AUTH, status=401)),
    TierMismatch("served default", {"service_tier": "default"}),
])
def test_quota_auth_and_tier_mismatch_halt_the_campaign(tmp_path, monkeypatch, error) -> None:
    ex = _executor(tmp_path, monkeypatch, _Send([error]))
    with pytest.raises(RateLimited):
        _run(ex, "no_menhir")


def test_a_rate_limit_timer_expiry_is_a_timed_out_session_not_a_halt(tmp_path: Path, monkeypatch) -> None:
    from archolith_harness.flex_send import TimerExpired

    error = TimerExpired("timer ran out", ErrorInfo(Kind.RATE_LIMIT, status=429, code="rate_limit_exceeded"))
    ex = _executor(tmp_path, monkeypatch, _Send([error]))
    res = _run(ex, "no_menhir")
    assert res.error == "timeout" and res.censored


def test_flex_capacity_wait_fails_only_the_session(tmp_path: Path, monkeypatch) -> None:
    error = Deferred("deferred: capacity", retry_at=0.0, info=ErrorInfo(Kind.CAPACITY, status=429))
    ex = _executor(tmp_path, monkeypatch, _Send([error]))
    res = _run(ex, "no_menhir")
    assert res.error.startswith("Deferred") and res.response_text == ""


def test_unpriced_model_is_refused_before_any_session(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="no pricing for .not-a-model-flex"):
        _executor(tmp_path, monkeypatch, _Send([]), model="openai/not-a-model")


def _cli():
    spec = importlib.util.spec_from_file_location("sprint_cli", ROOT / "scripts" / "run_longitudinal_sprint_bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_resume_never_mixes_agents_in_one_run_dir(tmp_path: Path, capsys) -> None:
    cli = _cli()
    base = ["--fixture", str(FIXTURE_PATH), "--repeats", "1", "--max-sessions", "1", "--arms", "no_menhir",
            "--run-id", "mix", "--out-root", str(tmp_path)]
    assert cli.main(base) == 0  # dry run, OpenCode (the default)
    assert cli.main([*base, "--resume"]) == 0
    assert cli.main([*base, "--resume", "--agent", "archolith-harness"]) == 2
    assert "agent" in capsys.readouterr().err


def test_cli_agent_version_is_the_harness_git_state() -> None:
    cli = _cli()
    assert cli._agent_version("opencode") is None
    assert cli._agent_version("archolith-harness")  # git state of the editable install


class _TimedSend(_Send):
    takes_timeout = True

    def __init__(self, replies):
        super().__init__(replies)
        self.timeouts: list = []

    def __call__(self, request, *, timeout_s=None):
        self.timeouts.append(timeout_s)
        return super().__call__(request)


def test_fallback_steps_are_priced_at_standard_and_recorded(tmp_path: Path, monkeypatch) -> None:
    send = _TimedSend([
        _step(items=[_call("list_directory", {"path": "."})], input_tokens=1000, cached=0, output=100, reasoning=0),
        _step(items=[_message("done")], input_tokens=1000, cached=0, output=100, reasoning=0, tier="default"),
    ])
    ex = _executor(tmp_path, monkeypatch, send, timeout_s=300.0)
    res = _run(ex, "no_menhir")
    flex = (1000 * 0.05 + 100 * 0.25) / 1e6
    standard = (1000 * 0.10 + 100 * 0.50) / 1e6
    assert res.cost_usd == pytest.approx(flex + standard)
    agent = json.loads((res.run_dir / "agent.json").read_text(encoding="utf-8"))
    assert agent["flex_mode"] == "fallback" and agent["standard_steps"] == 1
    assert send.timeouts[0] == pytest.approx(300.0, abs=5)  # the session's time left reaches the sender


def test_open_mode_has_no_session_clock(tmp_path: Path, monkeypatch) -> None:
    send = _TimedSend([_step(items=[_message("done")])])
    ex = _executor(tmp_path, monkeypatch, send, flex_mode="open", timeout_s=300.0)
    _run(ex, "no_menhir")
    assert send.timeouts == [None]


def test_unknown_flex_mode_is_refused(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="flex_mode"):
        HarnessSprintExecutor(LiveSprintConfig(work_dir=tmp_path, cache_dir=tmp_path), flex_mode="sometimes")


def test_resume_never_mixes_harness_flex_modes(tmp_path: Path) -> None:
    cli = _cli()
    base = ["--fixture", str(FIXTURE_PATH), "--repeats", "1", "--max-sessions", "1", "--arms", "no_menhir",
            "--run-id", "mode", "--out-root", str(tmp_path), "--agent", "archolith-harness"]
    assert cli.main([*base, "--harness-flex", "open"]) == 0
    assert cli.main([*base, "--harness-flex", "open", "--resume"]) == 0
    assert cli.main([*base, "--harness-flex", "fallback", "--resume"]) == 2
