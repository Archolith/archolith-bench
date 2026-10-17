"""Rescore the most recent 78-item knowledge-update LME slice through one of Menhir's Recall
Lab tuning arms (D/E/F/G/H) against the EXISTING graph from scalar-canonical-ku78-v1-20260806
-- no reingest. Generalizes run_arm_e_rescore.py (which stays as the E-specific original) so
the same D vs E vs F vs G vs H decomposition can run without duplicating the harness plumbing.

Arm tunings (menhir src/menhir/explorer/recall_lab.py DEFAULT_ARMS; RecallLabTuning defaults
enable_assertion_shadow=True, enable_evidence_anchor=True, everything else in ARMS below off):
    d: facet_candidates=True, facet_weight=0.5, oracle_ranking=True, intent_lens=True,
       warden_gate=True                                   (E's frontier, HARD evidence anchor)
    e: same as d + evidence_anchor=False                   (already run separately: 63/78)
    f: assertion_shadow=False, facet_candidates=True, facet_weight=0.5
                                                             (facet candidates ALONE)
    g: assertion_shadow=False, oracle_ranking=True, intent_lens=True
                                                             (oracle ranking + intent lens ALONE)
    h: facet_candidates=True, facet_weight=0.5, oracle_ranking=True, intent_lens=True,
       warden_gate=False, evidence_anchor=False             (E without the warden gate)

Skips the no_memory control (tuning-independent, already established: canonical run 6/78,
this session's E run 5/78 -- ~1-question judge-noise floor at N=78) to save cost/time; only
scores the menhir_recall arm for whichever tuning arm is selected.

Depends on menhir commit 4e8cf47 (Recall Lab forwarding authority_layer/is_scalar_authority/
is_superseded_view) -- without it this comparison is invalid because the KU scoring prompt
depends on that labeling.

Usage:
    LME_ARM_ID=f MENHIR_URL=http://localhost:8161 OPENAI_API_KEY=... \
      python scripts/longmemeval/analysis/run_arm_rescore.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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
from archolith_bench.harness.memory_ab import SINGLE_RECALL  # noqa: E402
from archolith_bench.harness.menhir_client import (  # noqa: E402
    HttpMenhirClient,
    _format_authority_record,
    _recall_item_text,
    is_verb_facet,
    place_verb_facets,
)

ARMS: dict[str, dict[str, Any]] = {
    "d": {
        "enable_facet_candidates": True, "facet_weight": 0.5,
        "enable_oracle_ranking": True, "enable_intent_lens": True,
        "enable_warden_gate": True,
    },
    "e": {
        "enable_facet_candidates": True, "facet_weight": 0.5,
        "enable_oracle_ranking": True, "enable_intent_lens": True,
        "enable_warden_gate": True, "enable_evidence_anchor": False,
    },
    "f": {
        "enable_assertion_shadow": False,
        "enable_facet_candidates": True, "facet_weight": 0.5,
    },
    "g": {
        "enable_assertion_shadow": False,
        "enable_oracle_ranking": True, "enable_intent_lens": True,
    },
    "h": {
        "enable_facet_candidates": True, "facet_weight": 0.5,
        "enable_oracle_ranking": True, "enable_intent_lens": True,
        "enable_warden_gate": False, "enable_evidence_anchor": False,
    },
}

FIXTURE_PATH = BENCH_DIR / "fixtures" / "longmemeval" / "knowledge_update_subset.json"


class RecallLabArmClient(HttpMenhirClient):
    """Same client shape as run_arm_e_rescore.RecallLabArmEClient, parameterized by arm_id."""

    def __init__(self, base_url: str, *, arm_id: str, tuning: dict[str, Any], api_key: str = "") -> None:
        super().__init__(base_url, api_key=api_key)
        self._recall_lab_path = "/explorer/api/recall-lab/run"
        self._arm_id = arm_id
        self._tuning = tuning

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
            "arms": [{"id": self._arm_id, "label": self._arm_id.upper(), "tuning": self._tuning}],
        }
        response = self._client.post(url, json=payload, headers=self._headers)
        response.raise_for_status()
        data = response.json()
        arms = data.get("arms") or []
        if not arms or not arms[0].get("ok"):
            reason = arms[0].get("error") if arms else "no arms in response"
            raise RuntimeError(f"Recall Lab arm {self._arm_id} failed for query={query!r}: {reason}")
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
    arm_id = os.environ.get("LME_ARM_ID", "").strip().lower()
    if arm_id not in ARMS:
        print(f"ERROR: LME_ARM_ID must be one of {sorted(ARMS)}, got {arm_id!r}", file=sys.stderr)
        sys.exit(1)
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

    suffix = os.environ.get("LME_RUN_SUFFIX", "")
    run_id = f"scalar-canonical-ku78-v1-20260806-arm-{arm_id}-rescore{suffix}"
    results_dir = BENCH_DIR / "results" / "lme-ku-buildout" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter("longmemeval-menhir")
    client = RecallLabArmClient(
        menhir_url, arm_id=arm_id, tuning=ARMS[arm_id], api_key=os.environ.get("MENHIR_API_KEY", "")
    )
    score_fn = LLMJudgeScorer(
        base_url=os.environ.get("UPSTREAM_BASE_URL") or "https://api.openai.com/v1",
        api_key=judge_key,
        model=os.environ.get("LME_JUDGE_MODEL", "gpt-6-luna"),
    )
    checkpoint = MemoryCheckpoint(
        checkpoint_path_for(results_dir, adapter.benchmark_id, os.environ.get("LME_ANSWER_MODEL", "gpt-4o"))
    )
    print(f"[arm-{arm_id}-rescore] {checkpoint.done_count()} item-results already checkpointed")
    print(f"[arm-{arm_id}-rescore] tuning={ARMS[arm_id]}")
    print(f"[arm-{arm_id}-rescore] scoring 78 knowledge-update items via Recall Lab arm {arm_id} "
          f"against {menhir_url} (namespace lme-{{question_id}}, no reingest)")

    ab = run_memory_ab(
        adapter,
        arms=[SINGLE_RECALL],
        subset="knowledge-update",
        limit=78,
        fixture_path=str(FIXTURE_PATH),
        client=client,
        model=os.environ.get("LME_ANSWER_MODEL", "gpt-4o"),
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

    out_path = results_dir / "results.md"
    write_harness_evidence(ab, out_path, output_format="markdown")
    print(f"[arm-{arm_id}-rescore] evidence written to {out_path}")
    (results_dir / "results.json").write_text(
        __import__("json").dumps(ab_result_to_dict(ab), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
