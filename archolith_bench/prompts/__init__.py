"""Prompt files: answer and judge prompts live here as text, loaded and hashed by relative name."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent


@cache
def load(name: str) -> str:
    """Text of prompts/<name>, CRLF normalised so a Windows checkout reads the same bytes."""
    return (PROMPT_DIR / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


def sha256(name: str) -> str:
    return hashlib.sha256(load(name).encode("utf-8")).hexdigest()


def hashes(names: list[str]) -> dict[str, str]:
    """{name: sha256} for each prompt file, for the resolved run config."""
    return {n: sha256(n) for n in sorted(names)}


ANSWER_LONGMEMEVAL = "answer/longmemeval-memory.system.txt"
JUDGE_GENERIC = "judge/generic.system.txt"
JUDGE_GENERIC_ABSTENTION = "judge/generic-abstention.system.txt"
JUDGE_UNOFFICIAL = "judge/unofficial.system.txt"
JUDGE_LME_ABSTENTION = "judge/longmemeval/abstention.txt"
LME_QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "single-session-preference",
)


def lme_judge(question_type: str) -> str:
    return f"judge/longmemeval/{question_type}.txt"


# Every prompt file a LongMemEval staged run can send (answer + official judge + unofficial judge).
LONGMEMEVAL_RUN_PROMPTS = [
    ANSWER_LONGMEMEVAL,
    JUDGE_GENERIC,
    JUDGE_GENERIC_ABSTENTION,
    JUDGE_UNOFFICIAL,
    JUDGE_LME_ABSTENTION,
    *(lme_judge(t) for t in LME_QUESTION_TYPES),
]
