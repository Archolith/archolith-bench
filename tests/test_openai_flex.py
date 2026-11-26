"""Offline tests for the OpenAI flex client (httpx.MockTransport, sleep stubbed)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from archolith_bench.core.openai_flex import FlexError, run_flex


def _ok_response(content: str = "hi") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


def _status_response(status: int, body: str | dict = "error") -> httpx.Response:
    return httpx.Response(
        status,
        content=body.encode() if isinstance(body, str) else json.dumps(body).encode(),
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


def _make_client(handler) -> tuple[httpx.Client, list]:
    calls: list = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.read().decode()))
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(counting_handler)), calls


def _run(handler, tmp_path: Path, requests=None, *, max_usd: float = 10.0, api_key: str = "sk-secret", **kwargs):
    client, calls = _make_client(handler)
    no_sleep = lambda _seconds: None  # noqa: E731
    logged: list[str] = []
    reqs = requests or [{"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "hello"}]}}]
    results, summary = run_flex(
        reqs,
        model="gpt-6-luna",
        checkpoint=tmp_path / "cp.jsonl",
        api_key=api_key,
        max_usd=max_usd,
        rates=(0.40, 1.60),
        client=client,
        sleep=no_sleep,
        log=logged.append,
        **kwargs,
    )
    return results, summary, calls, logged


def test_success_path(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    results, summary, calls, _ = _run(handler, tmp_path)
    assert "r1" in results
    res = results["r1"]
    assert res.content == "hi"
    assert res.error is None
    # cost = 1000*0.40/1e6 + 500*1.60/1e6 = 0.0004 + 0.0008
    assert res.cost_usd == pytest.approx(0.0012)
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["service_tier"] == "flex"
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["resumed"] == 0
    assert summary["model"] == "gpt-6-luna"
    assert summary["cost_usd"] == pytest.approx(0.0012)


def test_caller_body_fields_preserved(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    reqs = [
        {
            "custom_id": "r1",
            "body": {
                "messages": [{"role": "user", "content": "x"}],
                "reasoning_effort": "low",
                "max_completion_tokens": 64,
                "response_format": {"type": "text"},
            },
        }
    ]
    _, _, calls, _ = _run(handler, tmp_path, requests=reqs)
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_completion_tokens"] == 64
    assert calls[0]["response_format"] == {"type": "text"}


def test_duplicate_custom_id_raises_before_any_call(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected")

    reqs = [
        {"custom_id": "r1", "body": {"messages": []}},
        {"custom_id": "r1", "body": {"messages": []}},
    ]
    with pytest.raises(ValueError):
        _run(handler, tmp_path, requests=reqs)


def test_checkpoint_resume_skips_and_counts(tmp_path: Path) -> None:
    cp = tmp_path / "cp.jsonl"
    cp.write_text(
        json.dumps(
            {
                "custom_id": "r1",
                "content": "cached",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_usd": 0.0001,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    client, calls = _make_client(handler)
    reqs = [
        {"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "a"}]}},
        {"custom_id": "r2", "body": {"messages": [{"role": "user", "content": "b"}]}},
    ]
    results, summary = run_flex(
        reqs,
        model="gpt-6-luna",
        checkpoint=cp,
        api_key="sk-secret",
        max_usd=10.0,
        rates=(0.40, 1.60),
        client=client,
        sleep=lambda _s: None,
    )
    assert len(calls) == 1
    assert results["r1"].content == "cached"
    assert summary["resumed"] == 1
    assert summary["requests"] == 2


def test_capacity_429_retried_then_succeeds(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _status_response(429, '{"error": {"code": "resource_unavailable"}}')
        return _ok_response()

    _, summary, calls, _ = _run(handler, tmp_path)
    assert attempts["n"] == 3
    assert summary["completed"] == 1


def test_capacity_429_beyond_max_raises_flex_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _status_response(429, '{"error": {"code": "resource_unavailable"}}')

    with pytest.raises(FlexError):
        _run(handler, tmp_path, max_capacity_retries=2)


def test_non_capacity_429_immediate_flex_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _status_response(429, '{"error": {"code": "rate_limit_exceeded"}}')

    attempts = {"n": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(counting_handler))
    with pytest.raises(FlexError):
        run_flex(
            [{"custom_id": "r1", "body": {"messages": []}}],
            model="gpt-6-luna",
            checkpoint=tmp_path / "cp.jsonl",
            api_key="sk-secret",
            max_usd=10.0,
            rates=(0.40, 1.60),
            client=client,
            sleep=lambda _s: None,
        )
    assert attempts["n"] == 1


def test_402_immediate_flex_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _status_response(402, "payment required")

    with pytest.raises(FlexError):
        _run(handler, tmp_path)


def test_503_retried_then_succeeds(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _status_response(503)
        return _ok_response()

    _, summary, _, _ = _run(handler, tmp_path)
    assert attempts["n"] == 3
    assert summary["completed"] == 1


def test_connect_error_retried_then_succeeds(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("boom", request=_request)
        return _ok_response()

    _, summary, _, _ = _run(handler, tmp_path)
    assert attempts["n"] == 3
    assert summary["completed"] == 1


def test_persistent_503_is_per_request_failure_and_run_continues(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.read().decode())["messages"][0]["content"] == "fail":
            return _status_response(503)
        return _ok_response(content="ok")

    reqs = [
        {"custom_id": "bad", "body": {"messages": [{"role": "user", "content": "fail"}]}},
        {"custom_id": "good", "body": {"messages": [{"role": "user", "content": "fine"}]}},
    ]
    results, summary, _, _ = _run(handler, tmp_path, requests=reqs, workers=1)
    assert results["bad"].content is None
    assert "503" in (results["bad"].error or "")
    assert results["good"].content == "ok"
    assert summary["completed"] == 1
    assert summary["failed"] == 1


def test_400_per_request_failure_no_retry(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return _status_response(400, "bad request")

    results, summary, _, _ = _run(handler, tmp_path)
    assert attempts["n"] == 1
    assert results["r1"].content is None
    assert summary["failed"] == 1


def test_cost_cap_stops_and_checkpoints_finished_rows(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    reqs = [
        {"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content": "x"}]}}
        for i in range(4)
    ]
    # Each success costs 0.0012; cap 0.0024 allows exactly 2 requests, then blocks.
    with pytest.raises(FlexError):
        _run(handler, tmp_path, requests=reqs, workers=1, max_usd=0.0024)

    rows = [
        json.loads(line)
        for line in (tmp_path / "cp.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2
    assert {row["custom_id"] for row in rows} == {"r0", "r1"}


def test_client_overrides_caller_model_and_service_tier(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    reqs = [
        {
            "custom_id": "r1",
            "body": {
                "messages": [{"role": "user", "content": "x"}],
                "model": "evil-model",
                "service_tier": "default",
            },
        }
    ]
    _, _, calls, _ = _run(handler, tmp_path, requests=reqs)
    assert calls[0]["model"] == "gpt-6-luna"
    assert calls[0]["service_tier"] == "flex"


def test_unparseable_200_returns_per_request_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not json",
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    results, summary, _, _ = _run(handler, tmp_path)
    res = results["r1"]
    assert res.content is None
    assert res.error == "bad response: JSONDecodeError"
    assert res.prompt_tokens == 0
    assert res.completion_tokens == 0
    assert res.cost_usd == 0.0
    assert summary["failed"] == 1


def test_malformed_200_shape_returns_per_request_error(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[1, 2, 3],
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

    results, summary, _, _ = _run(handler, tmp_path)
    res = results["r1"]
    assert res.content is None
    assert res.error is not None and res.error.startswith("bad response: ")
    assert summary["failed"] == 1


def test_stray_checkpoint_row_filtered_from_results(tmp_path: Path) -> None:
    cp = tmp_path / "cp.jsonl"
    rows = [
        {"custom_id": "stray", "content": "old", "prompt_tokens": 1, "completion_tokens": 1, "cost_usd": 9.99, "error": None},
        {"custom_id": "r1", "content": "cached", "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0001, "error": None},
    ]
    cp.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    reqs = [{"custom_id": "r1", "body": {"messages": [{"role": "user", "content": "a"}]}}]
    results, summary, calls, _ = _run(handler, tmp_path, requests=reqs)
    assert set(results) == {"r1"}
    assert results["r1"].content == "cached"
    assert summary["resumed"] == 1
    assert summary["requests"] == 1
    assert summary["cost_usd"] == pytest.approx(0.0001)
    assert calls == []


def test_api_key_never_logged(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _ok_response()

    reqs = [{"custom_id": f"r{i}", "body": {"messages": [{"role": "user", "content": "x"}]}} for i in range(300)]
    _, _, _, logged = _run(handler, tmp_path, requests=reqs, workers=8)
    assert any("done" in line for line in logged)
    assert all("sk-secret" not in line for line in logged)


def _tiered_response(tier: str | None, model: str | None) -> httpx.Response:
    body: dict = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    if tier is not None:
        body["service_tier"] = tier
    if model is not None:
        body["model"] = model
    return httpx.Response(200, json=body,
                          request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


def test_response_tier_and_model_recorded(tmp_path: Path) -> None:
    results, _, _, _ = _run(lambda _r: _tiered_response("flex", "gpt-6-luna-2026-10-01"), tmp_path)
    assert results["r1"].service_tier == "flex"
    assert results["r1"].response_model == "gpt-6-luna-2026-10-01"


def test_response_non_flex_tier_stops_run(tmp_path: Path) -> None:
    with pytest.raises(FlexError, match="service_tier"):
        _run(lambda _r: _tiered_response("default", "gpt-6-luna"), tmp_path)


def test_response_other_model_stops_run(tmp_path: Path) -> None:
    with pytest.raises(FlexError, match="response model"):
        _run(lambda _r: _tiered_response("flex", "gpt-4o"), tmp_path)


_CAP = '{"error": {"code": "resource_unavailable"}}'


def _fb_kwargs(**extra):
    return {"fallback_tier": "default", "fallback_rates": (1.0, 4.0), "workers": 1, **extra}


def _fb_handler(tier: str = "default"):
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.read().decode())["service_tier"] == "flex":
            return _status_response(429, _CAP)
        return _tiered_response(tier, "gpt-6-luna")

    return handler


def test_fallback_after_capacity_retries_exhausted(tmp_path: Path) -> None:
    results, summary, calls, _ = _run(_fb_handler(), tmp_path, max_capacity_retries=2, **_fb_kwargs())
    assert [c["service_tier"] for c in calls] == ["flex", "flex", "flex", "default"]
    assert results["r1"].fell_back is True
    assert results["r1"].service_tier == "default"
    assert summary["fallbacks"] == 1


def test_fallback_after_elapsed_seconds(tmp_path: Path) -> None:
    ticks = iter(range(0, 1000, 10))
    results, _, calls, _ = _run(
        _fb_handler(), tmp_path, max_capacity_retries=50,
        **_fb_kwargs(fallback_after_s=15.0, clock=lambda: float(next(ticks))),
    )
    # first_attempt=0; 429 at t=10 (<15, retry); 429 at t=20 (>=15, fall back)
    assert [c["service_tier"] for c in calls] == ["flex", "flex", "default"]
    assert results["r1"].fell_back is True


def test_fallback_response_costed_at_fallback_rates(tmp_path: Path) -> None:
    results, summary, _, _ = _run(_fb_handler(), tmp_path, max_capacity_retries=0, **_fb_kwargs())
    # 10 prompt * 1.0/1e6 + 5 completion * 4.0/1e6
    assert results["r1"].cost_usd == pytest.approx(0.00003)
    assert summary["cost_usd"] == pytest.approx(0.00003)


def test_no_fallback_when_flex_succeeds(tmp_path: Path) -> None:
    results, summary, calls, _ = _run(
        lambda _r: _tiered_response("flex", "gpt-6-luna"), tmp_path, **_fb_kwargs()
    )
    assert len(calls) == 1
    assert results["r1"].fell_back is False
    assert results["r1"].cost_usd == pytest.approx(0.000012)
    assert summary["fallbacks"] == 0


def test_fallback_only_once(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _status_response(429, _CAP)

    with pytest.raises(FlexError, match="capacity retries exhausted"):
        _run(handler, tmp_path, max_capacity_retries=1, **_fb_kwargs())


def test_non_flex_tier_without_fallback_still_raises(tmp_path: Path) -> None:
    with pytest.raises(FlexError, match="service_tier"):
        _run(lambda _r: _tiered_response("default", "gpt-6-luna"), tmp_path)


def test_fallback_unexpected_tier_after_fallback_raises(tmp_path: Path) -> None:
    with pytest.raises(FlexError, match="service_tier"):
        _run(_fb_handler(tier="priority"), tmp_path, max_capacity_retries=0, **_fb_kwargs())


def test_fallback_tier_without_rates_raises_before_call(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected")

    with pytest.raises(ValueError):
        _run(handler, tmp_path, fallback_tier="default")


def test_non_capacity_429_never_falls_back(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _status_response(429, '{"error": {"code": "rate_limit_exceeded"}}')

    with pytest.raises(FlexError):
        _run(handler, tmp_path, **_fb_kwargs(fallback_after_s=0.0))


def test_non_capacity_429_sends_single_flex_call(tmp_path: Path) -> None:
    client, calls = _make_client(lambda _r: _status_response(429, '{"error": {"code": "rate_limit_exceeded"}}'))
    with pytest.raises(FlexError):
        run_flex(
            [{"custom_id": "r1", "body": {"messages": []}}],
            model="gpt-6-luna", checkpoint=tmp_path / "cp.jsonl", api_key="k", max_usd=10.0,
            rates=(0.4, 1.6), client=client, sleep=lambda _s: None,
            fallback_tier="default", fallback_after_s=0.0, fallback_rates=(1.0, 4.0),
        )
    assert [c["service_tier"] for c in calls] == ["flex"]


def test_transport_errors_never_fall_back(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    results, _, calls, _ = _run(handler, tmp_path, **_fb_kwargs(fallback_after_s=0.0))
    assert {c["service_tier"] for c in calls} == {"flex"}
    assert results["r1"].error is not None
    assert results["r1"].fell_back is False


def test_fell_back_checkpoint_round_trip_and_old_rows(tmp_path: Path) -> None:
    _run(_fb_handler(), tmp_path, max_capacity_retries=0, **_fb_kwargs())
    row = json.loads((tmp_path / "cp.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["fell_back"] is True
    results, summary, calls, _ = _run(lambda _r: _ok_response(), tmp_path)
    assert calls == []
    assert results["r1"].fell_back is True
    assert summary["fallbacks"] == 1
    old = tmp_path / "old"
    old.mkdir()
    (old / "cp.jsonl").write_text(
        json.dumps({"custom_id": "r1", "content": "c", "prompt_tokens": 1, "completion_tokens": 1,
                    "cost_usd": 0.0, "error": None}) + "\n",
        encoding="utf-8",
    )
    results, summary, _, _ = _run(lambda _r: _ok_response(), old)
    assert results["r1"].fell_back is False
    assert summary["fallbacks"] == 0
