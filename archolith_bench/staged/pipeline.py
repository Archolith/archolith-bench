"""Search, answer and judge stages over a RunDir."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..harness.menhir_client import render_recall_snippets
from .config import ArmConfig
from .rundir import RunDir


def render_context(arm: ArmConfig, response: Any) -> str:
    """Turn a stored search response into the answer prompt's memory context."""
    if arm.context == "none":
        return ""
    if arm.context == "legacy":
        return "\n".join(
            render_recall_snippets(
                response,
                arm.recall_limit,
                drill_down=arm.drill_down,
                counted_authority_only=arm.counted_authority_only,
                source_memories=arm.source_memories,
                counted_verdict_limit=arm.counted_verdict_limit,
            )
        )
    return str((response or {}).get("context") or "")


def _item_id(item: dict, i: int) -> str:
    return str(item.get("question_id") or f"item-{i}")


async def run_search(
    rundir: RunDir,
    arm: ArmConfig,
    items: list[dict],
    adapter: Any,
    system: Any,
    *,
    namespace_template: str = "lme-{question_id}",
) -> None:
    """Recall-only over a pre-built graph; never ingests or resets."""
    stage = rundir.stage("search")
    done = stage.done_ids()
    date_fn = getattr(adapter, "question_date", None)
    for i, item in enumerate(items):
        item_id = _item_id(item, i)
        if item_id in done:
            continue
        question = adapter.question(item)
        question_date = date_fn(item) if callable(date_fn) else None
        ns = namespace_template.format(question_id=item.get("question_id") or item_id)
        response: Any = None
        if arm.context == "legacy":
            response = await system.search(
                ns,
                question,
                limit=arm.recall_limit,
                counted_authority_only=arm.counted_authority_only,
                counted_verdict_limit=arm.counted_verdict_limit,
            )
        elif arm.context == "menhir":
            response = await system.context(
                ns, question, max_tokens=arm.context_max_tokens, preset=arm.context_preset
            )
        stage.append(
            {
                "item_id": item_id,
                "namespace": ns,
                "question": question,
                "question_date": question_date,
                "question_type": item.get("question_type"),
                "response": response,
                "context": render_context(arm, response),
            }
        )


def build_answer_messages(adapter: Any, record: dict) -> list[dict]:
    """Same call rule as memory_ab: pass the date only when present."""
    context = record["context"]
    question = record["question"]
    question_date = record.get("question_date")
    if question_date:
        return adapter.build_messages(context, question, question_date=question_date)
    return adapter.build_messages(context, question)


def run_answer(
    rundir: RunDir,
    arm: ArmConfig,
    adapter: Any,
    *,
    send_fn,  # noqa: ANN001
    chat_client: Any,
    chat_base_url: str,
    api_key: str,
    model: str,
) -> None:
    """Answer every searched item not yet answered."""
    out = rundir.stage("answers")
    done = out.done_ids()
    for rec in rundir.stage("search").records():
        if rec["item_id"] in done:
            continue
        messages = build_answer_messages(adapter, rec)
        text, latency_ms, usage = send_fn(chat_client, chat_base_url, api_key, messages, model, strict=True)
        blob = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        out.append(
            {
                "item_id": rec["item_id"],
                "messages": messages,
                "messages_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                "response_text": text,
                "latency_ms": latency_ms,
                "usage": usage,
                "model": model,
            }
        )


def run_judge(
    rundir: RunDir,
    items_by_id: dict[str, dict],
    *,
    score_fn,  # noqa: ANN001
    unofficial_fn=None,  # noqa: ANN001
) -> None:
    """Judge every answered item not yet judged.

    With ``unofficial_fn``, official-wrong items also get the unofficial substance judge, as in
    memory_ab: it is reported separately, and its failure leaves the item unjudged rather than
    blocking the official label.
    """
    out = rundir.stage("judged")
    done = out.done_ids()
    types = {r["item_id"]: r.get("question_type") for r in rundir.stage("search").records()}
    for rec in rundir.stage("answers").records():
        item_id = rec["item_id"]
        if item_id in done:
            continue
        item = items_by_id[item_id]
        correct = bool(score_fn(item, rec["response_text"]))
        row = {
            "item_id": item_id,
            "correct": correct,
            "question_type": types.get(item_id),
            "judge_usage": dict(getattr(score_fn, "last_usage", {}) or {}),
        }
        if unofficial_fn is not None:
            unofficial: bool | None = True if correct else None
            usage: dict = {}
            if not correct:
                try:
                    unofficial = bool(unofficial_fn(item, rec["response_text"]))
                    usage = dict(getattr(unofficial_fn, "last_usage", {}) or {})
                except Exception as exc:  # noqa: BLE001
                    print(f"  [unofficial judge] {item_id}: failed ({exc}); left unjudged")
            row["unofficial_correct"] = unofficial
            row["unofficial_usage"] = usage
        out.append(row)


def accuracy_summary(rows: list[dict]) -> dict:
    """Accuracy overall and per question type, plus the unofficial judge where it ran."""
    per: dict[str, list[bool]] = {}
    for r in rows:
        per.setdefault(str(r.get("question_type")), []).append(bool(r["correct"]))
    n = len(rows)
    out: dict[str, Any] = {
        "n": n,
        "accuracy": (sum(bool(r["correct"]) for r in rows) / n) if n else 0.0,
        "per_type": {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in per.items()},
    }
    judged = [r for r in rows if r.get("unofficial_correct") is not None]
    if judged:
        out["unofficial"] = {
            "n": len(judged),
            "accuracy": sum(bool(r["unofficial_correct"]) for r in judged) / len(judged),
        }
    return out


def build_report(rundir: RunDir) -> dict:
    """Accuracy overall and per question type; written to report.json/.md."""
    report = {"run_id": rundir.run_id, **accuracy_summary(rundir.stage("judged").records())}
    rundir.write_report(report)
    return report
