"""OpenRouter batch client against a fake OpenRouter (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from archolith_bench.core import openrouter_batch as ob

REQS = [
    {"custom_id": "a", "body": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}},
    {"custom_id": "b", "body": {"messages": [{"role": "user", "content": "yo"}], "max_tokens": 10}},
]


def _doc(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


class FakeRouter:
    def __init__(self, statuses=("in_progress", "completed"), results=None):
        self.statuses = list(statuses)
        self.results = results if results is not None else [
            {"custom_id": "a", "response": {"body": _doc("A!")}},
            {"custom_id": "b", "result": {"response": {"body": _doc("B!")}}},
        ]
        self.submits, self.polls, self.submitted = 0, 0, None

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.submits += 1
            self.submitted = json.loads(request.content)
            return httpx.Response(200, json={"id": "batch_1", "status": "validating"})
        self.polls += 1
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        body = {"id": "batch_1", "status": status, "usage": {"cost": 0.0012}}
        if status == "completed":
            body["results"] = self.results
        return httpx.Response(200, json=body)


def _run(router, tmp_path, **kw):
    with httpx.Client(transport=httpx.MockTransport(router.handler)) as client:
        return ob.run_batch(REQS, model="openai/gpt-6-luna", checkpoint=tmp_path / "cp.json", api_key="k",
                            max_usd=kw.pop("max_usd", 1.0), client=client, sleep=lambda s: None,
                            rates=(1e-7, 5e-7), log=lambda m: None, **kw)


def test_submits_polls_and_maps_nested_results(tmp_path):
    router = FakeRouter()
    results, summary = _run(router, tmp_path)
    assert router.submitted["endpoint"] == "/v1/chat/completions"
    assert router.submitted["model"] == "openai/gpt-6-luna"
    assert router.submitted["requests"][0]["body"]["model"] == "openai/gpt-6-luna"
    assert {k: v.content for k, v in results.items()} == {"a": "A!", "b": "B!"}
    assert summary["status"] == "completed" and summary["actual_cost_usd"] == 0.0012 and summary["errors"] == 0
    assert router.polls == 2


def test_resume_never_resubmits(tmp_path):
    first = FakeRouter(statuses=("in_progress",))
    with httpx.Client(transport=httpx.MockTransport(first.handler)) as client, pytest.raises(ob.BatchError):
        ob.run_batch(REQS, model="m", checkpoint=tmp_path / "cp.json", api_key="k", max_usd=1.0, client=client,
                     sleep=lambda s: None, rates=(1e-7, 5e-7), timeout_s=0, log=lambda m: None)
    assert first.submits == 1 and json.loads((tmp_path / "cp.json").read_text())["batch_id"] == "batch_1"
    second = FakeRouter()
    results, _ = _run(second, tmp_path)
    assert second.submits == 0 and results["a"].content == "A!"


def test_resume_refuses_a_different_request_set(tmp_path):
    (tmp_path / "cp.json").write_text(json.dumps({"batch_id": "x", "custom_ids": ["zzz"]}))
    with pytest.raises(ob.BatchError, match="different request set"):
        _run(FakeRouter(), tmp_path)


def test_cost_cap_refuses_before_submitting(tmp_path):
    router = FakeRouter()
    with pytest.raises(ob.BatchError, match="exceeds cap"):
        _run(router, tmp_path, max_usd=1e-9)
    assert router.submits == 0 and not (tmp_path / "cp.json").exists()


@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
def test_non_completed_terminal_status_fails_closed(tmp_path, status):
    with pytest.raises(ob.BatchError, match=status):
        _run(FakeRouter(statuses=(status,)), tmp_path)


def test_duplicate_or_unexpected_ids_fail_and_missing_are_errors(tmp_path):
    dup = FakeRouter(results=[{"custom_id": "a", "body": _doc("1")}, {"custom_id": "a", "body": _doc("2")}])
    with pytest.raises(ob.BatchError, match="duplicate"):
        _run(dup, tmp_path / "d")
    odd = FakeRouter(results=[{"custom_id": "zzz", "body": _doc("1")}])
    with pytest.raises(ob.BatchError, match="unexpected"):
        _run(odd, tmp_path / "u")
    partial = FakeRouter(results=[{"custom_id": "a", "body": _doc("1")}, {"custom_id": "b", "error": "overloaded"}])
    results, summary = _run(partial, tmp_path / "p")
    assert results["a"].content == "1" and results["b"].content is None
    assert "overloaded" in results["b"].error and summary["errors"] == 1


def test_duplicate_custom_ids_in_the_request_are_rejected(tmp_path):
    with pytest.raises(ob.BatchError, match="unique"):
        ob.run_batch(REQS + [REQS[0]], model="m", checkpoint=tmp_path / "cp.json", api_key="k", max_usd=1.0,
                     client=httpx.Client(transport=httpx.MockTransport(FakeRouter().handler)), rates=(0, 0))


def test_estimate_scales_with_prompt_and_output_budget():
    small = ob.estimate_max_usd(REQS, 1e-7, 5e-7, reasoning_allowance=0)
    big = ob.estimate_max_usd(REQS, 1e-7, 5e-7, reasoning_allowance=4000)
    assert 0 < small < big


def test_a_just_created_batch_that_404s_briefly_is_retried_then_fails_after_grace(tmp_path):
    router = FakeRouter()
    seen = {"n": 0}
    real = router.handler

    def flaky(request):
        if request.method == "GET" and seen["n"] < 2:
            seen["n"] += 1
            return httpx.Response(404, json={"error": "not found"})
        return real(request)

    router.handler = flaky
    results, _ = _run(router, tmp_path)
    assert results["a"].content == "A!" and router.submits == 1

    gone = FakeRouter()
    gone.handler = lambda request: (httpx.Response(200, json={"id": "batch_1", "status": "validating"})
                                    if request.method == "POST" else httpx.Response(404, json={}))
    with pytest.raises(httpx.HTTPStatusError):
        _run(gone, tmp_path / "g", not_found_grace_s=0)
