"""The ingest stops when Menhir reports a provider state that waiting cannot fix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_ingest():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "ingest.py"
    spec = importlib.util.spec_from_file_location("lme_ingest_provider_stop_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_ingest_provider_stop_test"] = module
    spec.loader.exec_module(module)
    return module


ingest = _load_ingest()


class _Response:
    def __init__(self, body, *, status: int = 200, bad_json: bool = False) -> None:
        self._body = body
        self._status = status
        self._bad_json = bad_json

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._body


class _Admin:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.urls: list[str] = []

    def get(self, url: str) -> _Response:
        self.urls.append(url)
        return self.response


def _healthy() -> _Admin:
    return _Admin(_Response({"status": "ok", "provider_quota_exhausted": None}))


def _quota_exhausted() -> _Admin:
    return _Admin(
        _Response({"status": "degraded", "provider_quota_exhausted": "insufficient_quota"})
    )


def test_quota_exhausted_health_stops_the_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "_queue_depth", lambda admin, url: pytest.fail("polled on"))
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)
    admin = _quota_exhausted()

    with pytest.raises(ingest.ProviderStoppedError, match="provider_quota_exhausted"):
        ingest._drain_many(["ns-a"], admin, "http://menhir/", poll_s=0, timeout_s=5)

    assert admin.urls == ["http://menhir/api/health"]


def test_auth_failure_health_also_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    admin = _Admin(_Response({"provider_auth_failure": "invalid_api_key"}))
    with pytest.raises(ingest.ProviderStoppedError, match="provider_auth_failure"):
        ingest._raise_if_provider_stopped(admin, "http://menhir")


@pytest.mark.parametrize(
    "admin",
    [
        _healthy(),
        _Admin(_Response({}, status=503)),
        _Admin(_Response(None, bad_json=True)),
        _Admin(_Response(["not", "a", "dict"])),
        object(),  # existing tests pass a bare object as the admin client
    ],
)
def test_unreadable_or_clean_health_does_not_stop(admin) -> None:
    assert ingest._provider_stop_reason(admin, "http://menhir") is None
    ingest._raise_if_provider_stopped(admin, "http://menhir")


def test_clean_health_lets_the_drain_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "_queue_depth", lambda admin, url: 0)
    monkeypatch.setattr(
        ingest,
        "_ns_state_counts",
        lambda namespace: {"pending": 0, "ready": 1, "enriching": 0, "failed": 0, "total": 1},
    )
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)

    drained = ingest._drain_many(["ns-a"], _healthy(), "http://menhir", poll_s=0, timeout_s=5)

    assert drained["ns-a"]["timed_out"] is False


def test_quota_exhausted_stops_the_window_before_retrying_or_submitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[str] = []

    def fake_ingest_turn(client, namespace, role, content, **kwargs):
        submitted.append(content)
        return f"ep-{len(submitted)}"

    monkeypatch.setattr(ingest, "_ingest_turn", fake_ingest_turn)
    monkeypatch.setattr(
        ingest,
        "_episode_processing",
        lambda admin, url, episode_uuid: pytest.fail("episode polled after the provider stop"),
    )
    window = [ingest.WindowItem(item={}, question_id="q1", namespace="ns-a")]
    turns = iter(
        [
            SimpleNamespace(
                role="user", content=text, occurred_at=None, session_id="s0",
                extract=True, evidence_text="", turn_key="",
            )
            for text in ("first", "second")
        ]
    )

    with pytest.raises(ingest.ProviderStoppedError):
        ingest._ingest_window(
            window, [turns], object(), _quota_exhausted(), "http://menhir",
            timeout_s=60, poll_s=0,
        )

    # Only the first turn went in; the stop came before any later submission.
    assert submitted == ["first"]
