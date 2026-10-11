"""LME_LABEL_GROUNDED_USER_TURNS: "user: " on grounded user turns AND their evidence.

Menhir currently decides the speaker from a "user:" text prefix (Archolith/menhir#271), while
grounding needs the episode text to equal its :TurnEvidence text exactly. The workaround only
holds if both texts carry the same label, and a resume can never mix labelled and raw evidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from archolith_bench.harness import HttpMenhirClient
from archolith_bench.harness.menhir_client import episode_body, turn_evidence_text

ROOT = Path(__file__).resolve().parents[1]
TEXT = "I moved to Lisbon two weeks ago."


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _menhir_sees_user(text: str) -> bool:
    """Mirror of Menhir's _episode_role check (graphiti_extraction_policy.py:490)."""
    return str(text or "").lstrip().lower().startswith("user:")


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


def _client(label: bool) -> tuple[HttpMenhirClient, _Http]:
    client = HttpMenhirClient("http://throwaway-menhir.local", label_grounded_user_turns=label)
    http = _Http()
    client._client = http  # type: ignore[assignment]
    return client, http


# ---------------------------------------------------------------- client


def test_label_on_grounded_user_episode_equals_its_evidence_and_reads_as_user() -> None:
    client, http = _client(True)
    turn_id = client.record_turn_evidence("ns", TEXT, role="user")["turn_id"]
    client.ingest("ns", "user", TEXT, source="user", turn_evidence_uuid=turn_id)

    evidence = http.posts[0][1]["text"]
    episode = http.posts[1][1]["episode"]
    assert evidence == episode == f"user: {TEXT}"
    assert _menhir_sees_user(episode)


def test_label_off_is_unchanged_raw_evidence_and_raw_grounded_episode() -> None:
    client, http = _client(False)
    client.record_turn_evidence("ns", TEXT, role="user")
    client.ingest("ns", "user", TEXT, turn_evidence_uuid="turn-1")

    assert http.posts[0][1]["text"] == TEXT
    assert http.posts[1][1]["episode"] == TEXT
    assert HttpMenhirClient("http://x").label_grounded_user_turns is False


def test_label_on_leaves_assistant_evidence_raw() -> None:
    # Evidence-only assistant turns keep their own words; the label is for user speech only.
    client, http = _client(True)
    client.record_turn_evidence("ns", "Sure, here is a plan.", role="assistant", declarant="assistant")
    assert http.posts[0][1]["text"] == "Sure, here is a plan."


def test_label_on_does_not_double_label_ungrounded_turns() -> None:
    client, http = _client(True)
    client.ingest("ns", "user", TEXT)  # no evidence: the existing "{role}: " path
    assert http.posts[0][1]["episode"] == f"user: {TEXT}"


@pytest.mark.parametrize("role", ["user", "User", " USER ", ""])
def test_user_role_spellings_are_labelled(role: str) -> None:
    assert turn_evidence_text(role, TEXT, label_grounded_user=True) == f"user: {TEXT}"
    assert episode_body(role, TEXT, grounded=True, label_grounded_user=True) == f"user: {TEXT}"


@pytest.mark.parametrize("role", ["assistant", "system"])
def test_non_user_grounded_text_is_never_labelled(role: str) -> None:
    assert episode_body(role, TEXT, grounded=True, label_grounded_user=True) == TEXT


# ---------------------------------------------------------------- partial resume


ingest = _load("lme_ingest_label_test", "scripts/longmemeval/lib/ingest.py")


class _Adapter:
    def sessions(self, item):
        return item["sessions"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (True, ["user: u1", "assistant: a1", "user: u2"]),
        (False, ["u1", "assistant: a1", "u2"]),
    ],
)
def test_resume_compares_against_the_bodies_this_client_would_send(
    monkeypatch: pytest.MonkeyPatch, label: bool, expected: list[str]
) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(
        ingest, "_stored_submitted_episodes", lambda ns, bodies: seen.append(list(bodies)) or []
    )
    monkeypatch.setattr(ingest, "_reset_namespace", lambda admin, url, ns: None)
    item = {
        "question_id": "q1",
        "sessions": [
            [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}],
            [{"role": "user", "content": "u2"}],
        ],
    }
    state = ingest.WindowItem(item=item, question_id="q1", namespace="ns1")
    ingest._resume_or_reset(
        SimpleNamespace(label_grounded_user_turns=label), object(), "http://x", state,
        adapter=_Adapter(), segmentation="none", evidence_by_turn={},
    )
    assert seen == [expected]


@pytest.mark.parametrize(("value", "on"), [(None, False), ("0", False), ("1", True), ("yes", False)])
def test_ingest_reads_the_setting_from_env(
    monkeypatch: pytest.MonkeyPatch, value: str | None, on: bool
) -> None:
    if value is None:
        monkeypatch.delenv("LME_LABEL_GROUNDED_USER_TURNS", raising=False)
    else:
        monkeypatch.setenv("LME_LABEL_GROUNDED_USER_TURNS", value)
    module = _load("lme_ingest_label_env_test", "scripts/longmemeval/lib/ingest.py")
    assert module.LABEL_GROUNDED_USER_TURNS is on


def test_ingest_builds_its_client_with_the_setting() -> None:
    source = (ROOT / "scripts/longmemeval/lib/ingest.py").read_text(encoding="utf-8")
    assert "HttpMenhirClient(args.menhir_url, label_grounded_user_turns=True)\n" in source
    assert "        if LABEL_GROUNDED_USER_TURNS\n" in source


# ---------------------------------------------------------------- run identity


provenance = _load("lme_run_provenance_label_test", "scripts/longmemeval/lib/run_provenance.py")


def _attempt(**overrides) -> dict:
    record = {
        "run_id": "r1",
        "started_at": "2026-10-11T00:00:00Z",
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
def test_resume_refuses_turning_labels_on_for_a_run_recorded_without_them(
    tmp_path: Path, noncanonical: bool
) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt())  # recorded before the key existed: raw evidence
    with pytest.raises(provenance.ProvenanceMismatch, match="did not label"):
        provenance.begin(path, _attempt(label_grounded_user_turns=1), noncanonical=noncanonical)


def test_resume_with_labels_off_matches_a_run_recorded_without_the_key(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt())
    provenance.begin(path, _attempt(label_grounded_user_turns=0))


@pytest.mark.parametrize(("recorded", "attempted"), [(1, 0), (0, 1)])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_resume_refuses_switching_the_label(
    tmp_path: Path, recorded: int, attempted: int, noncanonical: bool
) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt(label_grounded_user_turns=recorded))
    with pytest.raises(provenance.ProvenanceMismatch, match="label_grounded_user_turns"):
        provenance.begin(
            path, _attempt(label_grounded_user_turns=attempted), noncanonical=noncanonical
        )


def test_resume_with_the_same_label_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "run_provenance.json"
    _begin(path, _attempt(label_grounded_user_turns=1))
    provenance.begin(path, _attempt(label_grounded_user_turns=1))


# ---------------------------------------------------------------- shell wiring


def test_build_wiring_defaults_off_validates_records_and_exports() -> None:
    config = (ROOT / "scripts/longmemeval/config.sh").read_text(encoding="utf-8")
    script = (ROOT / "scripts/longmemeval/build_graph.sh").read_text(encoding="utf-8")
    assert 'LME_LABEL_GROUNDED_USER_TURNS="${LME_LABEL_GROUNDED_USER_TURNS:-0}"' in config
    assert 'die "LME_LABEL_GROUNDED_USER_TURNS must be 0 or 1"' in script
    assert '"label_grounded_user_turns": ${LME_LABEL_GROUNDED_USER_TURNS},' in script
    assert '--setting "label_grounded_user_turns=${LME_LABEL_GROUNDED_USER_TURNS}"' in script
    assert 'export LME_LABEL_GROUNDED_USER_TURNS="${LME_LABEL_GROUNDED_USER_TURNS}"' in script
    # The identity record is written before ingest starts, so a refused resume never ingests.
    assert script.index('"label_grounded_user_turns": ${') < script.index("lib/ingest.py")


@pytest.mark.parametrize(("value", "ok"), [("0", True), ("1", True), ("2", False), ("", False), ("yes", False)])
def test_build_rejects_values_other_than_0_or_1(value: str, ok: bool) -> None:
    import os
    import shutil
    import subprocess

    git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    script = (ROOT / "scripts/longmemeval/build_graph.sh").read_text(encoding="utf-8")
    start = script.index('case "${LME_LABEL_GROUNDED_USER_TURNS}" in')
    block = script[start:script.index("esac", start) + len("esac")]
    command = "set -eu\ndie(){ printf '%s\\n' \"$*\" >&2; exit 1; }\n" + block
    env = dict(os.environ, LME_LABEL_GROUNDED_USER_TURNS=value)
    result = subprocess.run([bash, "-c", command], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is ok, result.stderr
    if not ok:
        assert "must be 0 or 1" in result.stderr
