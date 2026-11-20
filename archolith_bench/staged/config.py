"""Arm configuration, resolved-config record and run id."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

_BUILTIN_PROMPT = "builtin:longmemeval-memory"
_CONTEXTS = ("none", "legacy", "menhir")

_TOP_KEYS = {"name", "context", "recall", "render", "build_context", "answer"}
_SECTION_KEYS = {
    "recall": {"limit"},
    "render": {"source_memories", "counted_authority_only", "drill_down", "counted_verdict_limit"},
    "build_context": {"max_tokens", "preset"},
    "answer": {"prompt", "model"},
}


@dataclass(frozen=True)
class ArmConfig:
    """One benchmark arm: how context is produced and how the answer is prompted."""

    name: str
    context: Literal["none", "legacy", "menhir"]
    recall_limit: int = 20
    drill_down: bool = False
    counted_authority_only: bool = False
    source_memories: bool = True
    counted_verdict_limit: int | None = None
    context_max_tokens: int = 4000
    context_preset: str = "knowledge"
    answer_prompt: str = _BUILTIN_PROMPT
    answer_model: str | None = None

    def __post_init__(self) -> None:
        if self.context not in _CONTEXTS:
            raise ValueError(f"context must be one of {_CONTEXTS}, got {self.context!r}")
        if self.drill_down:
            raise ValueError("drill-down is not supported by the staged pipeline yet")
        if self.answer_prompt != _BUILTIN_PROMPT:
            raise ValueError(f"answer prompt must be {_BUILTIN_PROMPT!r} in this version")


def arm_from_dict(d: dict[str, Any]) -> ArmConfig:
    """Build an ArmConfig from the YAML layout; unknown keys raise ValueError."""
    if not isinstance(d, dict):
        raise ValueError("arm config must be a mapping")
    for key in d:
        if key not in _TOP_KEYS:
            raise ValueError(f"unknown key in arm config: {key}")
    sections: dict[str, dict] = {}
    for sec, allowed in _SECTION_KEYS.items():
        val = d.get(sec) or {}
        if not isinstance(val, dict):
            raise ValueError(f"{sec} must be a mapping")
        for key in val:
            if key not in allowed:
                raise ValueError(f"unknown key in {sec}: {key}")
        sections[sec] = val
    if "name" not in d or "context" not in d:
        raise ValueError("arm config requires 'name' and 'context'")
    kwargs: dict[str, Any] = {"name": d["name"], "context": d["context"]}
    mapping = {
        ("recall", "limit"): "recall_limit",
        ("render", "source_memories"): "source_memories",
        ("render", "counted_authority_only"): "counted_authority_only",
        ("render", "drill_down"): "drill_down",
        ("render", "counted_verdict_limit"): "counted_verdict_limit",
        ("build_context", "max_tokens"): "context_max_tokens",
        ("build_context", "preset"): "context_preset",
        ("answer", "prompt"): "answer_prompt",
        ("answer", "model"): "answer_model",
    }
    for (sec, key), field in mapping.items():
        if key in sections[sec]:
            kwargs[field] = sections[sec][key]
    return ArmConfig(**kwargs)


def load_arm(path: str | Path) -> ArmConfig:
    """Load an arm YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return arm_from_dict(data)


def renderer_sha256() -> str:
    """Hash of the legacy renderer's module (CRLF normalised)."""
    from ..harness import menhir_client

    raw = Path(menhir_client.__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def resolve(
    arm: ArmConfig,
    *,
    benchmark_id: str,
    fixture_path: str,
    fixture_sha256: str,
    prompt_fingerprint: str,
    renderer_sha256: str,
    menhir_commit: str | None,
    namespace_template: str,
    model: str | None,
    judge: dict,
) -> dict:
    """Everything that determines a run's outputs, as a JSON-able dict."""
    return {
        "arm": asdict(arm),
        "benchmark": benchmark_id,
        "fixture": {"path": fixture_path, "sha256": fixture_sha256},
        "answer_prompt_fingerprint": prompt_fingerprint,
        "renderer": {
            "name": "legacy" if arm.context == "legacy" else None,
            "sha256": renderer_sha256,
        },
        "menhir_commit": menhir_commit,
        "namespace_template": namespace_template,
        "model": model,
        "judge": judge,
    }


def run_id(resolved: dict) -> str:
    """Stable id: arm name plus a hash of the resolved config."""
    blob = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return f"{resolved['arm']['name']}-{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]}"
