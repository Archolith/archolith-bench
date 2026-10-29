"""Rescore the most recent 78-item knowledge-update LME slice through Menhir's Recall Lab
"E" tuning arm (production base + frontier, soft evidence-anchor) against the EXISTING
graph from scalar-canonical-ku78-v1-20260806 -- no reingest.

Arm E tuning (menhir src/menhir/explorer/recall_lab.py DEFAULT_ARMS):
    enable_facet_candidates=True, facet_weight=0.5, enable_oracle_ranking=True,
    enable_intent_lens=True, enable_warden_gate=True, enable_evidence_anchor=False

This reuses the SAME harness (run_memory_ab), adapter (LongMemEvalMemoryAdapter), fixture,
scorer (LLMJudgeScorer, LongMemEval-comparable), and namespace convention
(lme-{question_id}) as run_knowledge_update_buildout.sh's Phase 2 -- only the recall
transport differs: instead of client.recall() -> POST /api/recall (production tuning),
RecallLabArmEClient.recall() -> POST /explorer/api/recall-lab/run with arm E's tuning dict.

Snippet formatting (authority-layer labeling, [AUTHORITATIVE CURRENT MEMORY] etc.) is
inherited byte-for-byte from HttpMenhirClient.recall() by subclassing and only replacing
the HTTP call + payload; the canonical run's own results.md used the same formatting path,
so this is an apples-to-apples comparison now that recall_lab.py forwards authority_layer
(see menhir commit 4e8cf47 -- previously Recall Lab silently dropped it).

Usage:
    MENHIR_URL=http://localhost:8151 OPENAI_API_KEY=... \
      python scripts/longmemeval/analysis/run_arm_e_rescore.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx

BENCH_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCH_DIR))

from archolith_bench.harness import (  # noqa: E402
    LLMJudgeScorer,
    MemoryCheckpoint,
    ab_result_to_dict,
    checkpoint_path_for,
    get_adapter,
    run_memory_ab,
    write_harness_evidence,
)
from archolith_bench.harness.memory_ab import NO_MEMORY, SINGLE_RECALL  # noqa: E402
from archolith_bench.harness.menhir_client import (  # noqa: E402
    HttpMenhirClient,
    _format_authority_record,
    _recall_item_text,
    is_verb_facet,
    place_verb_facets,
)

ARM_E_TUNING: dict[str, Any] = {
    "enable_facet_candidates": True,
    "facet_weight": 0.5,
    "enable_oracle_ranking": True,
    "enable_intent_lens": True,
    "enable_warden_gate": True,
    "enable_evidence_anchor": False,
}

FIXTURE_PATH = BENCH_DIR / "fixtures" / "longmemeval" / "knowledge_update_subset.json"
RUN_ID = os.getenv("LME_ARM_E_RUN_ID", "scalar-canonical-ku78-v1-20260806-arm-e-rescore")
RESULTS_DIR = BENCH_DIR / "results" / "lme-ku-buildout" / RUN_ID


class RecallLabArmEClient(HttpMenhirClient):
    """Same client as the canonical run, except recall() goes through Recall Lab's arm E
    instead of production /api/recall. ingest()/new_group()/reset() are unused in
    recall_only mode and inherited unchanged (never called)."""

    def __init__(self, base_url: str, *, api_key: str = "") -> None:
        super().__init__(base_url, api_key=api_key)
        self._recall_lab_path = "/explorer/api/recall-lab/run"

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        url = self._base_url.rstrip("/") + self._recall_lab_path
        payload = {
            "query": query,
            "namespace": group_id,
            "limit": limit,
            "candidate_k": max(limit, 50),
            "include_session": True,
            "include_invalidated": True,
            "judge": False,
            "arms": [{"id": "e", "label": "E", "tuning": ARM_E_TUNING}],
        }
        response = self._client.post(url, json=payload, headers=self._headers)
        response.raise_for_status()
        data = response.json()
        arms = data.get("arms") or []
        if not arms or not arms[0].get("ok"):
            reason = arms[0].get("error") if arms else "no arms in response"
            raise RuntimeError(f"Recall Lab arm E failed for query={query!r}: {reason}")
        arm = arms[0]

        items: list[dict[str, Any]] = list(arm.get("results") or [])
        authority_records = [r for r in (arm.get("authority_layer") or []) if isinstance(r, dict)]
        authority_view_ids = {
            str(record["view_uuid"]) for record in authority_records if record.get("view_uuid")
        }

        snippets: list[str] = []
        for record in authority_records:
            rendered = _format_authority_record(record)
            if rendered:
                snippets.append(rendered)

        n_head = len(snippets)
        facet_snippets: list[str] = []
        for item in items:
            if str(item.get("uuid") or "") in authority_view_ids:
                continue
            text = _recall_item_text(item)
            if not text:
                continue
            if authority_records:
                memory_type = str(item.get("memory_type") or "memory").strip().lower()
                if item.get("is_superseded_view") is True:
                    prefix = f"[SUPERSEDED {memory_type} MEMORY | historical only]"
                elif item.get("is_scalar_authority") is True:
                    prefix = f"[CURRENT {memory_type} MEMORY | authoritative]"
                else:
                    prefix = f"[RELATED {memory_type} MEMORY | non-authoritative]"
                text = f"{prefix} {text}"
            if is_verb_facet(item):
                facet_snippets.append(text)
            else:
                snippets.append(text)

        return place_verb_facets(snippets, n_head, facet_snippets, limit)


def main() -> None:
    menhir_url = os.environ.get("MENHIR_URL", "").rstrip("/")
    if not menhir_url:
        print("ERROR: MENHIR_URL is required (a recall-only menhir instance on the "
              "scalar-canonical-ku78-v1-20260806 graph, NOT production)", file=sys.stderr)
        sys.exit(1)
    judge_key = os.environ.get("OPENAI_API_KEY", "")
    if not judge_key:
        print("ERROR: OPENAI_API_KEY is required (answer model + judge model)", file=sys.stderr)
        sys.exit(1)
    if not FIXTURE_PATH.exists():
        print(f"ERROR: fixture not found: {FIXTURE_PATH}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adapter = get_adapter("longmemeval-menhir")
    client = RecallLabArmEClient(menhir_url, api_key=os.environ.get("MENHIR_API_KEY", ""))
    score_fn = LLMJudgeScorer(
        base_url=os.environ.get("UPSTREAM_BASE_URL") or "https://api.openai.com/v1",
        api_key=judge_key,
        model=os.environ.get("LME_JUDGE_MODEL", "gpt-6-luna"),
    )
    checkpoint = MemoryCheckpoint(
        checkpoint_path_for(RESULTS_DIR, adapter.benchmark_id, os.environ.get("LME_ANSWER_MODEL", "gpt-6-luna"))
    )
    print(f"[arm-e-rescore] {checkpoint.done_count()} item-results already checkpointed")

    print(f"[arm-e-rescore] scoring 78 knowledge-update items via Recall Lab arm E "
          f"against {menhir_url} (namespace lme-{{question_id}}, no reingest)")

    ab = run_memory_ab(
        adapter,
        arms=[NO_MEMORY, SINGLE_RECALL],
        subset="knowledge-update",
        limit=78,
        fixture_path=str(FIXTURE_PATH),
        client=client,
        model=os.environ.get("LME_ANSWER_MODEL", "gpt-6-luna"),
        chat_base_url=os.environ.get("UPSTREAM_BASE_URL") or "https://api.openai.com/v1",
        api_key=judge_key,
        recall_limit=int(os.environ.get("LME_RECALL_LIMIT", "10")),
        recall_only=True,
        namespace_template="lme-{question_id}",
        checkpoint=checkpoint,
        score_fn=score_fn,
    )

    for arm, r in ab.arms.items():
        print(f"  {arm}: score={r.score:.3f} n={r.n} in={r.input_tokens:,} "
              f"out={r.output_tokens:,} cost=${r.cost_usd:.6f}")
    for arm, d in ab.deltas.items():
        print(f"  delta {arm}: score {d['score_delta']:+.3f}")

    out_path = RESULTS_DIR / "results.md"
    write_harness_evidence(ab, out_path, output_format="markdown")
    print(f"[arm-e-rescore] evidence written to {out_path}")
    (RESULTS_DIR / "results.json").write_text(
        __import__("json").dumps(ab_result_to_dict(ab), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
