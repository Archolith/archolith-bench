"""The bench sends Menhir the structured speaker role (Archolith/menhir#271).

Menhir takes the speaker from POST /api/memory `role`, or from the linked turn evidence for a
grounded turn, and never from a "user:" text prefix. The prefix stays on ungrounded episode text
only as extraction input, so existing arms stay comparable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from archolith_bench.harness import HttpMenhirClient
from archolith_bench.harness.menhir_client import MENHIR_TURN_ROLES, menhir_turn_role

ROOT = Path(__file__).resolve().parents[1]
TEXT = "I moved to Lisbon two weeks ago."


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"turn_id": "turn-1", "episode_id": "ep-1"}


class _Http:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append((url, kwargs.get("json") or {}))
        return _Response()


def _client() -> tuple[HttpMenhirClient, _Http]:
    client = HttpMenhirClient("http://throwaway-menhir.local")
    http = _Http()
    client._client = http  # type: ignore[assignment]
    return client, http


# ---------------------------------------------------------------- role mapping


def test_role_vocabulary_matches_menhir() -> None:
    assert MENHIR_TURN_ROLES == {"user", "assistant", "tool", "agent", "system"}


@pytest.mark.parametrize(
    ("role", "expected"),
    [("user", "user"), ("assistant", "assistant"), (" Assistant ", "assistant"), ("USER", "user"),
     ("", "user"), ("   ", "user"), ("system", "system"), ("human", None), ("bot", None)],
)
def test_menhir_turn_role(role: str, expected: str | None) -> None:
    assert menhir_turn_role(role) == expected


# ---------------------------------------------------------------- client payload


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_ungrounded_turn_sends_role_and_keeps_the_text_prefix(role: str) -> None:
    client, http = _client()
    client.ingest("ns", role, TEXT)
    payload = http.posts[0][1]
    assert payload["role"] == role
    assert payload["episode"] == f"{role}: {TEXT}"


def test_grounded_user_turn_sends_role_with_raw_text_equal_to_its_evidence() -> None:
    client, http = _client()
    turn_id = client.record_turn_evidence("ns", TEXT, role="user")["turn_id"]
    client.ingest("ns", "user", TEXT, source="user", turn_evidence_uuid=turn_id)
    evidence, episode = http.posts[0][1], http.posts[1][1]
    assert evidence["text"] == episode["episode"] == TEXT
    assert episode["role"] == "user"


def test_role_menhir_does_not_know_is_omitted() -> None:
    client, http = _client()
    client.ingest("ns", "narrator", TEXT)
    payload = http.posts[0][1]
    assert "role" not in payload
    assert payload["episode"] == f"narrator: {TEXT}"


def test_label_workaround_is_gone() -> None:
    import archolith_bench.harness.menhir_client as module

    assert not hasattr(module, "turn_evidence_text")
    with pytest.raises(TypeError):
        HttpMenhirClient("http://x", label_grounded_user_turns=True)  # type: ignore[call-arg]
    for path in ("scripts/longmemeval/config.sh", "scripts/longmemeval/build_graph.sh",
                 "scripts/longmemeval/lib/ingest.py", "scripts/longmemeval/lib/run_provenance.py"):
        assert "label_grounded" not in (ROOT / path).read_text(encoding="utf-8").lower(), path


# ---------------------------------------------------------------- run identity


provenance = _load("lme_run_provenance_turn_role_test", "scripts/longmemeval/lib/run_provenance.py")


def _attempt(**overrides) -> dict:
    record = {
        "run_id": "r1",
        "started_at": "2026-10-12T00:00:00Z",
        "menhir_commit": "aaaa111",
        "bench_commit": "bbbb222",
        "container": "c",
        "volume": "v",
        "canonical_self_binding_mode": "off",
    }
    record.update(overrides)
    return record


def _begin(path: Path, attempt: dict) -> None:
    provenance._write_atomic(path, provenance.begin(path, attempt))


@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_a_run_ingested_without_turn_roles(tmp_path: Path, noncanonical: bool) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt())
    with pytest.raises(provenance.ProvenanceMismatch, match="without structured turn roles"):
        provenance.begin(path, _attempt(sends_turn_role=1), noncanonical=noncanonical)


def test_resume_refuses_a_run_recorded_with_the_old_label_workaround(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt(label_grounded_user_turns=1))
    with pytest.raises(provenance.ProvenanceMismatch, match="without structured turn roles"):
        provenance.begin(path, _attempt(sends_turn_role=1), noncanonical=True)


def test_resume_of_a_turn_role_run_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt(sends_turn_role=1))
    provenance.begin(path, _attempt(sends_turn_role=1))


def test_build_records_turn_role_identity_before_ingest() -> None:
    script = (ROOT / "scripts/longmemeval/build_graph.sh").read_text(encoding="utf-8")
    assert '  "sends_turn_role": 1,\n' in script
    assert '--setting "sends_turn_role=1"' in script
    assert script.index('"sends_turn_role": 1') < script.index("lib/ingest.py")
