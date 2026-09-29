"""ingest.py's evidence-required gate defaults ON when the env var is unset."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INGEST_PATH = ROOT / "scripts" / "longmemeval" / "lib" / "ingest.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, INGEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_require_turn_evidence_defaults_on_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LME_REQUIRE_TURN_EVIDENCE", raising=False)
    module = _load_module("lme_ingest_require_evidence_unset_test")
    assert module.REQUIRE_TURN_EVIDENCE is True


def test_require_turn_evidence_off_when_env_zero(monkeypatch) -> None:
    monkeypatch.setenv("LME_REQUIRE_TURN_EVIDENCE", "0")
    module = _load_module("lme_ingest_require_evidence_zero_test")
    assert module.REQUIRE_TURN_EVIDENCE is False


def test_require_turn_evidence_on_when_env_one(monkeypatch) -> None:
    monkeypatch.setenv("LME_REQUIRE_TURN_EVIDENCE", "1")
    module = _load_module("lme_ingest_require_evidence_one_test")
    assert module.REQUIRE_TURN_EVIDENCE is True
