"""Held-out, recall-only comparison of compact typed memory packet shapes.

The panel is pre-registered below and excludes the five development cases used while
designing the packet shapes.  Both arms use the same production retrieval results and
the same generic query-to-scalar identity matcher.  This script never ingests or
mutates graph state.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import dotenv_values

BENCH_DIR = Path(__file__).resolve().parents[3]
MENHIR_DIR = BENCH_DIR.parent / "menhir"
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(MENHIR_DIR / "src"))

from archolith_bench.harness import (  # noqa: E402
    LLMJudgeScorer,
    MemoryCheckpoint,
    ab_result_to_dict,
    checkpoint_path_for,
    get_adapter,
    run_memory_ab,
)
from archolith_bench.harness.memory_ab import SINGLE_RECALL  # noqa: E402
from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402
from menhir.explorer.recall_packet_prototype import (  # noqa: E402
    build_query_filtered_recall_packet,
)

SOURCE_RUN_ID = "scalar-canonical-ku78-v1-20260806"
FIXTURE_PATH = BENCH_DIR / "fixtures" / "longmemeval" / "knowledge_update_subset.json"
RUN_ID = f"{SOURCE_RUN_ID}-heldout-packet-shape-panel-v2-20260808"

# Fixed before scoring. Development cases intentionally excluded:
# 6a1eabeb, 6aeb4375, 830ce83f, 852ce960, 945e3d21.
PANEL = (
    ("89941a93", "changing_scalar/count/current"),
    ("2698e78f", "changing_scalar/frequency/current"),
    ("50635ada", "changing_scalar/status/history"),
    ("e66b632c", "changing_scalar/duration/history"),
    ("9ea5eabc", "scalar_adjacent/money_irrelevant"),
    ("618f13b2", "scalar_adjacent/duration_irrelevant"),
    ("59524333", "scalar_adjacent/clock_time_direct"),
    ("71315a70", "content_only/duration"),
    ("e493bb7c", "content_only/location"),
    ("b01defab", "content_only/completion"),
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_HISTORY_INTENT = re.compile(
    r"\b(?:previous(?:ly)?|earlier|formerly|before|used\s+to|history|historical|"
    r"original(?:ly)?|prior|past|then|at\s+first)\b",
    re.IGNORECASE,
)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "the", "to", "was", "what", "when", "where",
    "which", "who", "with", "you", "your", "current", "currently", "now", "user",
}


def _tokens(value: Any) -> set[str]:
    result: set[str] = set()
    for token in _TOKEN_RE.findall(str(value or "").lower()):
        if token in _STOP or len(token) < 2:
            continue
        result.add(token)
        if len(token) > 4 and token.endswith("s"):
            result.add(token[:-1])
        if len(token) > 4 and token.endswith("ed"):
            stem = token[:-2]
            result.add(stem)
            if stem.endswith("i"):
                result.add(stem[:-1] + "y")
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            result.add(stem)
            if len(stem) > 2 and stem[-1] == stem[-2]:
                result.add(stem[:-1])
    return result


def _identity_relevant(view: dict[str, Any], query_tokens: set[str]) -> bool:
    attribute = _tokens(view.get("attribute"))
    scope = _tokens(view.get("scope"))
    if scope:
        return bool(query_tokens & scope) or len(query_tokens & attribute) >= 2
    if not attribute:
        return False
    required = 1 if len(attribute) == 1 else 2
    return len(query_tokens & attribute) >= required


def _display(view: dict[str, Any]) -> str:
    value = str(view.get("value") or "").strip()
    if view.get("value_kind") == "duration" and str(view.get("unit") or "").lower() == "seconds":
        try:
            seconds = float(value)
        except ValueError:
            pass
        else:
            sign = "-" if seconds < 0 else ""
            total = abs(seconds)
            whole = int(total)
            fraction = total - whole
            hours, remainder = divmod(whole, 3600)
            minutes, secs = divmod(remainder, 60)
            suffix = f"{fraction:.9f}".split(".", 1)[1].rstrip("0") if fraction else ""
            sec_text = f"{secs:02d}" + (f".{suffix}" if suffix else "")
            return f"{sign}{hours}:{minutes:02d}:{sec_text}" if hours else f"{sign}{minutes}:{sec_text}"
    return str(view.get("display") or value).strip()


def _view_entry(view: dict[str, Any], assertions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sources = [assertions[item] for item in view.get("contributor_ids") or [] if item in assertions]
    source = sources[-1] if sources else {}
    return {
        "subject": view.get("subject"),
        "attribute": view.get("attribute"),
        "scope": view.get("scope") or None,
        "display_value": _display(view),
        "canonical_value": view.get("value"),
        "unit": view.get("unit") or None,
        "value_kind": view.get("value_kind"),
        "derivation": view.get("derivation"),
        "valid_at": view.get("valid_at"),
        "current": bool(view.get("current")),
        "source_quote": source.get("source_quote") or source.get("stated_span"),
    }


def _context_entries(packet: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in packet.get("sections") or []:
        if section.get("id") in {"current_state", "change_history", "completed_events"}:
            continue
        for entry in section.get("entries") or []:
            identifier = str(entry.get("id") or "")
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            entries.append(
                {
                    "rank": entry.get("retrieval_rank"),
                    "kind": entry.get("kind") or "content",
                    "name": entry.get("subject"),
                    "content": entry.get("content") or entry.get("value"),
                    "valid_at": entry.get("valid_at"),
                }
            )
    return sorted(entries, key=lambda item: int(item.get("rank") or 10_000))[:4]


def _select(task: dict[str, Any], query_packet: dict[str, Any], query: str) -> dict[str, Any]:
    graph = task["live_graph"]
    assertions = {str(row.get("id")): row for row in graph.get("assertions") or []}
    views = list(graph.get("views") or [])
    query_tokens = _tokens(query)
    relevant = [view for view in views if _identity_relevant(view, query_tokens)]
    current = [_view_entry(view, assertions) for view in relevant if view.get("current")]
    current_keys = {
        (str(view.get("subject") or "").lower(), str(view.get("attribute") or "").lower(), str(view.get("scope") or "").lower())
        for view in relevant
        if view.get("current")
    }
    history_requested = bool(_HISTORY_INTENT.search(query))
    history_views = [
        view
        for view in views
        if (str(view.get("subject") or "").lower(), str(view.get("attribute") or "").lower(), str(view.get("scope") or "").lower()) in current_keys
        and (not view.get("current") or history_requested)
    ]
    if history_requested and not history_views:
        history_views = relevant
    history = [_view_entry(view, assertions) for view in sorted(history_views, key=lambda item: str(item.get("valid_at") or ""))]
    return {
        "query": query,
        "intent": "historical" if history_requested else "current_or_general",
        "current_state": current[:3],
        "scalar_history": history[:5],
        "retrieved_context": _context_entries(query_packet),
    }


def _labeled_text(selected: dict[str, Any]) -> str:
    lines = [
        "MEMORY PACKET compact-labeled/v1",
        "RULES:",
        "- For current questions, authoritative current state outranks retrieved context.",
        "- Scalar history is audit evidence and must not replace current state unless the question asks about the past.",
        "- Retrieved context is supporting evidence and may contain stale, tentative, or irrelevant statements.",
        f"- Query intent: {selected['intent']}",
        "",
        "[AUTHORITATIVE CURRENT STATE]",
    ]
    for entry in selected["current_state"]:
        target = f"{entry['subject']}.{entry['attribute']}"
        lines.append(f"- {target} = {entry['display_value']} | valid_at={entry['valid_at']} | derivation={entry['derivation']}")
        if entry.get("source_quote"):
            lines.append(f"  evidence: {entry['source_quote']}")
    if not selected["current_state"]:
        lines.append("- none")
    lines.extend(("", "[SCALAR HISTORY — AUDIT ONLY UNLESS THE QUERY ASKS ABOUT THE PAST]"))
    for entry in selected["scalar_history"]:
        target = f"{entry['subject']}.{entry['attribute']}"
        status = "CURRENT" if entry["current"] else "SUPERSEDED"
        lines.append(f"- {status}: {target} = {entry['display_value']} | valid_at={entry['valid_at']}")
    if not selected["scalar_history"]:
        lines.append("- none")
    lines.extend(("", "[RETRIEVED CONTEXT — SUPPORTING ONLY]"))
    for entry in selected["retrieved_context"]:
        lines.append(f"- rank={entry['rank']} | valid_at={entry['valid_at']} | {entry['content']}")
    if not selected["retrieved_context"]:
        lines.append("- none")
    return "\n".join(lines)


def _typed_json(selected: dict[str, Any]) -> str:
    payload = {
        "packet_type": "typed-authority/v1",
        "query_intent": selected["intent"],
        "authority_contract": {
            "current_state": "authoritative for current questions",
            "scalar_history": "audit-only unless the query asks about the past",
            "retrieved_context": "supporting; may be stale, tentative, or irrelevant",
        },
        "current_state": selected["current_state"],
        "scalar_history": selected["scalar_history"],
        "retrieved_context": selected["retrieved_context"],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


_RETRIEVAL_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


class PanelPacketClient(HttpMenhirClient):
    def __init__(self, base_url: str, *, shape: str) -> None:
        super().__init__(base_url)
        self._shape = shape

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        run = quote(SOURCE_RUN_ID, safe="")
        namespace = quote(group_id, safe="")
        task_url = self._base_url.rstrip("/") + f"/explorer/api/recall-lab/bench-runs/{run}/tasks/{namespace}"
        task_response = self._client.get(task_url, headers=self._headers)
        task_response.raise_for_status()
        task = task_response.json()
        cache_key = (group_id, query)
        results = _RETRIEVAL_CACHE.get(cache_key)
        if results is None:
            recall_response = self._client.post(
                self._base_url.rstrip("/") + "/explorer/api/recall-lab/run",
                headers=self._headers,
                json={
                    "query": query,
                    "namespace": group_id,
                    "limit": limit,
                    "candidate_k": max(50, limit),
                    "include_session": True,
                    "include_superseded": False,
                    "include_invalidated": True,
                    "judge": False,
                    "arms": [
                        {
                            "id": "production",
                            "label": "Production",
                            "enabled": True,
                            "tuning": {},
                        }
                    ],
                },
            )
            recall_response.raise_for_status()
            recall_payload = recall_response.json()
            arm = (recall_payload.get("arms") or [{}])[0]
            if not arm.get("ok") or arm.get("degraded"):
                raise RuntimeError(f"production retrieval failed for {group_id}")
            results = list(arm.get("results") or [])
            _RETRIEVAL_CACHE[cache_key] = results
        full_packet = (task.get("live_graph") or {}).get("recall_packet") or {}
        packet = build_query_filtered_recall_packet(
            full_packet,
            results,
            query,
            authority_layer=[],
            event_authority_layer=[],
            max_chars=6_000,
            max_general=4,
        )
        if self._shape == "compact_labeled":
            text = packet["text"]
        else:
            text = json.dumps(
                {
                    "packet_type": "typed-authority/query-filtered-v2",
                    "query": query,
                    "usage_policy": packet["usage_policy"],
                    "sections": [
                        section for section in packet["sections"] if section.get("entries")
                    ],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        return [text]


def _key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    return str(dotenv_values(BENCH_DIR / ".env").get("OPENAI_API_KEY") or "")


def _result_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    arm = data["arms"][SINGLE_RECALL]
    return list(arm["results"])


def main() -> None:
    menhir_url = os.environ.get("MENHIR_URL", "http://127.0.0.1:8150").rstrip("/")
    from archolith_bench.harness.memory_ab import assert_not_production

    assert_not_production(menhir_url)
    key = _key()
    if not key:
        raise SystemExit("OPENAI_API_KEY is unavailable")
    all_items = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    by_id = {item["question_id"]: item for item in all_items}
    panel_items = [by_id[qid] for qid, _ in PANEL]
    results_dir = BENCH_DIR / "results" / "lme-ku-buildout" / RUN_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    panel_fixture = results_dir / "panel_fixture.json"
    panel_fixture.write_text(json.dumps(panel_items, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "panel_registration.json").write_text(
        json.dumps({"registered_before_scoring": True, "development_ids_excluded": True, "panel": PANEL}, indent=2),
        encoding="utf-8",
    )

    adapter = get_adapter("longmemeval-menhir")
    answer_model = os.environ.get("LME_ANSWER_MODEL", "gpt-4o-mini")
    judge_model = os.environ.get("LME_JUDGE_MODEL", "gpt-4o-mini")
    outputs: dict[str, Any] = {}
    started_at = datetime.now(UTC).isoformat()
    for shape in ("compact_labeled", "typed_json"):
        shape_dir = results_dir / shape
        shape_dir.mkdir(exist_ok=True)
        checkpoint = MemoryCheckpoint(checkpoint_path_for(shape_dir, adapter.benchmark_id, answer_model, variant="oracle"))
        print(f"[panel] shape={shape} checkpointed={checkpoint.done_count(SINGLE_RECALL)}")
        scorer = LLMJudgeScorer(
            base_url="https://api.openai.com/v1", api_key=key, model=judge_model
        )
        ab = run_memory_ab(
            adapter,
            arms=[SINGLE_RECALL],
            subset="knowledge-update",
            limit=len(panel_items),
            fixture_path=str(panel_fixture),
            client=PanelPacketClient(menhir_url, shape=shape),
            model=answer_model,
            chat_base_url="https://api.openai.com/v1",
            api_key=key,
            recall_limit=8,
            recall_only=True,
            namespace_template="lme-{question_id}",
            checkpoint=checkpoint,
            score_fn=scorer,
        )
        data = ab_result_to_dict(ab)
        (shape_dir / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs[shape] = data

    report: dict[str, Any] = {
        "run_id": RUN_ID,
        "canonical": False,
        "noncanonical": True,
        "recall_only": True,
        "graph_mutated": False,
        "reingested": False,
        "registered_before_scoring": True,
        "source_run_id": SOURCE_RUN_ID,
        "implementation": "corrected query-filtered builder over raw production retrieval",
        "answer_model": answer_model,
        "judge_model": judge_model,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "panel": [{"question_id": qid, "stratum": stratum} for qid, stratum in PANEL],
        "shapes": {},
    }
    for shape, data in outputs.items():
        rows = _result_rows(data)
        token_total = sum(
            int(row.get(key) or 0)
            for row in rows
            for key in ("input_tokens", "output_tokens", "scorer_input_tokens", "scorer_output_tokens")
        )
        report["shapes"][shape] = {
            "correct": sum(bool(row.get("correct")) for row in rows),
            "n": len(rows),
            "tokens": token_total,
            "tasks": [
                {
                    "question_id": row.get("task_id"),
                    "correct": bool(row.get("correct")),
                    "response": row.get("response_text"),
                    "tokens": sum(int(row.get(key) or 0) for key in ("input_tokens", "output_tokens", "scorer_input_tokens", "scorer_output_tokens")),
                }
                for row in rows
            ],
        }
    (results_dir / "run_provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["shapes"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
