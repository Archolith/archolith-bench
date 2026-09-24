"""LLM judge for "why" tasks: does a saved answer state each gold point?

The deterministic scorer needs every word of one accepted wording in one answer entry. That
suits short rules but misses explanations worded differently ("deduplication" for "dedupe",
"PyTorch" for "torch", numbers). For ``why`` tasks this module adds ``point_recall_judged``
beside the deterministic ``point_recall``; it replaces nothing.

For each gold point the judge model sees the question, the point with its accepted wordings, the
cited memory quotes as reference, and the answer's ``findings`` and ``plan``. It never sees the
condition, run name or other runs. A "met" verdict counts only when its evidence is copied from
the answer (whitespace and case aside); otherwise the point is not met and ``raw_met`` records
what the model said. Verdicts are cached per run (model + answer hash), so re-judging is free.

Stops: an HTTP 429 raises :class:`JudgeRateLimited` (never retried); a call that could pass
the dollar cap raises :class:`JudgeBudgetExhausted` before it is made. The API key is read by
the caller from an env file and never logged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_JUDGE_MODEL = "gpt-6-luna"  # owner decision 2026-09-24
#: Dollars per million (input, output) tokens (OpenCode's models.dev catalog, 2026-09-24);
#: a model without a price cannot keep the cap. Luna's reasoning tokens bill as output.
PRICES_PER_M = {"gpt-6-luna": (0.10, 0.50), "gpt-4o-mini": (0.15, 0.60)}
#: Reasoning models reject a temperature; only these get temperature 0.
TEMPERATURE_MODELS = frozenset({"gpt-4o-mini"})
#: Dollars set aside for the next call when checking the cap.
CALL_RESERVE_USD = 0.002
_MIN_EVIDENCE_CHARS = 8

SYSTEM_PROMPT = (
    "You grade one point of an answer to a question about a software project. Decide whether "
    "the ANSWER states the GOLD POINT: the same reason or fact, in any wording. Paraphrases, "
    "synonyms and equivalent numbers count. A vaguer, partial or different reason does not. "
    "The REFERENCE is the recorded source of the gold point, for context only; the answer "
    "need not quote it. Reply with a JSON object: {\"met\": true or false, \"evidence\": "
    "\"the shortest passage copied exactly from the ANSWER that states the point, or empty\", "
    "\"reason\": \"one short sentence\"}."
)

Call = Callable[[list[dict[str, str]]], tuple[str, dict[str, int]]]


class JudgeRateLimited(RuntimeError):
    """The judge's provider refused with a rate limit; judging stops, nothing is retried."""


class JudgeBudgetExhausted(RuntimeError):
    """The next judge call could pass the dollar cap."""


@dataclass
class PointVerdict:
    point: str
    met: bool
    raw_met: bool
    evidence: str
    reason: str


def _norm(text: str) -> str:
    return " ".join(str(text).split()).lower()


def answer_lines(answer: dict[str, Any] | None) -> list[str]:
    """The entries the deterministic point scorer reads: ``findings`` then ``plan``."""
    if not isinstance(answer, dict):
        return []
    lines: list[str] = []
    for key in ("findings", "plan"):
        value = answer.get(key)
        items = value if isinstance(value, list) else [value] if value else []
        lines += [json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                  for item in items]
    return [line for line in lines if line.strip()]


def load_references(task_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Per task: the question and each point's wordings with its cited memory quotes."""
    refs: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(task_root.glob("*/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        quotes: dict[str, list[str]] = {}
        for citation in data.get("memory_citations") or []:
            quotes.setdefault(str(citation.get("item", "")), []).append(str(citation.get("quote", "")))
        points = []
        for point in (data.get("gold") or {}).get("points") or []:
            wordings = [point] if isinstance(point, str) else list(point)
            points.append({"wordings": wordings, "quotes": quotes.get(wordings[0], [])})
        refs[(data["repo"], data["task_id"])] = {"prompt": data["prompt"], "points": points}
    return refs


def build_messages(question: str, point: dict[str, Any], lines: list[str]) -> list[dict[str, str]]:
    wordings = point["wordings"]
    user = "\n\n".join(
        [
            "QUESTION:\n" + question.strip(),
            "GOLD POINT: " + wordings[0]
            + ("\nAccepted wordings: " + "; ".join(wordings[1:]) if len(wordings) > 1 else ""),
            "REFERENCE:\n" + ("\n".join("- " + q for q in point["quotes"]) or "(none)"),
            "ANSWER:\n" + ("\n".join("- " + line for line in lines) or "(empty)"),
        ]
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def judge_point(call: Call, question: str, point: dict[str, Any], lines: list[str]) -> tuple[PointVerdict, dict[str, int]]:
    text, usage = call(build_messages(question, point, lines))
    try:
        data = json.loads(text)
    except ValueError:
        data = {}
    raw_met = data.get("met") is True
    evidence = str(data.get("evidence") or "")
    grounded = len(evidence.strip()) >= _MIN_EVIDENCE_CHARS and _norm(evidence) in _norm("\n".join(lines))
    verdict = PointVerdict(
        point=point["wordings"][0],
        met=raw_met and grounded,
        raw_met=raw_met,
        evidence=evidence[:400],
        reason=str(data.get("reason") or "")[:300],
    )
    return verdict, usage


def call_cost(model: str, usage: dict[str, int]) -> float:
    price_in, price_out = PRICES_PER_M[model]
    return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6


def openai_call(api_key: str, model: str, timeout_s: float = 60.0) -> Call:
    """A chat-completions caller (JSON output; temperature 0 where supported). 429 raises, never retries."""

    def call(messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "response_format": {"type": "json_object"},
        }
        if model in TEMPERATURE_MODELS:
            payload["temperature"] = 0
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - fixed https URL
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                raise JudgeRateLimited(f"judge rate limited (HTTP 429): {exc.reason}") from None
            raise RuntimeError(f"judge call failed: HTTP {exc.code} {exc.reason}") from None
        usage = data.get("usage") or {}
        return str(data["choices"][0]["message"]["content"]), {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }

    return call


def judge_workdir(
    workdir: Path,
    task_root: Path,
    call: Call,
    model: str = DEFAULT_JUDGE_MODEL,
    budget_usd: float = 0.10,
) -> tuple[dict[str, float], float]:
    """Judge every saved run whose task has references. Returns (run name -> score, dollars spent).

    Each run's verdicts go to ``judged.json`` beside its ``result.json``; a cached verdict for the
    same model and answer is reused without a call.
    """
    if model not in PRICES_PER_M:
        raise ValueError(f"no price for judge model {model!r}; the cap cannot be kept")
    refs = load_references(task_root)
    spent = 0.0
    scores: dict[str, float] = {}
    for path in sorted((workdir / "runs").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        ref = refs.get((result["repo"], result["task_id"]))
        if ref is None or not ref["points"]:
            continue
        lines = answer_lines(result.get("answer"))
        digest = hashlib.sha256(json.dumps(lines, ensure_ascii=False).encode("utf-8")).hexdigest()
        cache = path.parent / "judged.json"
        if cache.is_file():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("model") == model and cached.get("answer_sha256") == digest:
                scores[path.parent.name] = cached["point_recall_judged"]
                continue
        verdicts: list[PointVerdict] = []
        for point in ref["points"]:
            if not lines:
                verdicts.append(PointVerdict(point["wordings"][0], False, False, "", "no answer"))
                continue
            if spent + CALL_RESERVE_USD > budget_usd:
                stop = JudgeBudgetExhausted(
                    f"judge stopped at ${spent:.4f}: the next call could pass the ${budget_usd:.2f} cap"
                )
                stop.spent = spent  # type: ignore[attr-defined]
                raise stop
            try:
                verdict, usage = judge_point(call, ref["prompt"], point, lines)
            except JudgeRateLimited as exc:
                exc.spent = spent  # type: ignore[attr-defined]
                raise
            spent += call_cost(model, usage)
            verdicts.append(verdict)
        value = sum(v.met for v in verdicts) / len(verdicts)
        cache.write_text(
            json.dumps(
                {"model": model, "answer_sha256": digest, "point_recall_judged": value,
                 "points": [v.__dict__ for v in verdicts]},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        scores[path.parent.name] = value
    return scores, spent
