"""Retrieval metrics for the facet ladder.

Everything here is computed from a query's ranked id list plus the corpus, so the
same functions score every condition (BM25, embedding, facet, ...) identically.
The R2 plan requires these be reported **together** — a condition that cuts
wrong-scope injection while gutting recall has not won.

Metrics needing a generation model (`answer_grounding_accuracy`) are out of scope
for this offline harness and are left to the home run; everything below is
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

from archolith_bench.core.ir_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: F401  (runner uses M.*)

from .models import SCOPE_FACETS, Memory, Query


def stale_hit_rate(
    ranked: Sequence[str], memories_by_id: dict[str, Memory], query: Query, k: int
) -> float:
    """Fraction of the top-k that are stale/superseded under a current-intent query.

    Always 0.0 for non-current queries: returning historical memories is correct
    when the query is asking about history.
    """
    if query.intent != "current":
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    stale = sum(1 for mid in top if mid in memories_by_id and memories_by_id[mid].is_stale)
    return stale / len(top)


def wrong_scope_injection_rate(
    ranked: Sequence[str], memories_by_id: dict[str, Memory], query: Query, k: int
) -> float:
    """Fraction of the top-k whose scope facets conflict with the query's scope."""
    top = ranked[:k]
    if not top:
        return 0.0
    wrong = sum(1 for mid in top if mid in memories_by_id and _scope_conflict(query, memories_by_id[mid]))
    return wrong / len(top)


def support_sufficiency(ranked: Sequence[str], support_ids: Sequence[str], k: int) -> float:
    """1.0 if the top-k covers *all* gold support ids, else 0.0.

    Distinct from recall@k (fractional): this asks the binary "did we surface
    enough to actually answer?" question. Averaged over queries it is the share
    fully supported.
    """
    if not support_ids:
        return 0.0
    top = set(ranked[:k])
    return 1.0 if all(sid in top for sid in support_ids) else 0.0


def false_neighbor_rate(
    ranked: Sequence[str], memories_by_id: dict[str, Memory], query: Query, k: int
) -> float:
    """Top-k non-support items that *look* relevant: share a topic facet but miss.

    A "false neighbor" shares object/symbol/operation with the query yet is not
    gold support — the classic stale-semantic-neighbor / wrong-repo-same-topic
    failure. This is stricter than (1 − precision): an unrelated doc that slipped
    in does not count, only convincing-looking wrong answers do.
    """
    top = ranked[:k]
    if not top:
        return 0.0
    support = set(query.support_ids)
    qf = query.facets
    topic = qf.values("object") | qf.values("symbol") | qf.values("operation")
    if not topic:
        return 0.0
    false_neighbors = 0
    for mid in top:
        if mid in support or mid not in memories_by_id:
            continue
        mf = memories_by_id[mid].facets
        m_topic = mf.values("object") | mf.values("symbol") | mf.values("operation")
        if topic & m_topic:
            false_neighbors += 1
    return false_neighbors / len(top)


def paraphrase_stability(top_k_by_query: dict[str, list[str]], queries: Sequence[Query], k: int) -> float:
    """Mean Jaccard overlap of top-k sets across queries sharing a paraphrase group.

    A facet/meet-point ranker should be *more* stable across paraphrases than a
    lexical one. Returns 0.0 when the fixture defines no paraphrase groups.
    """
    groups: dict[str, list[str]] = {}
    for query in queries:
        if query.paraphrase_group:
            groups.setdefault(query.paraphrase_group, []).append(query.id)

    pair_scores: list[float] = []
    for member_ids in groups.values():
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                a = set(top_k_by_query.get(member_ids[i], [])[:k])
                b = set(top_k_by_query.get(member_ids[j], [])[:k])
                union = a | b
                pair_scores.append(len(a & b) / len(union) if union else 1.0)
    return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0


def _scope_conflict(query: Query, memory: Memory) -> bool:
    """True if any scope facet is set on both sides and disagrees."""
    qf = query.facets
    mf = memory.facets
    for facet in SCOPE_FACETS:
        q_val = getattr(qf, facet)
        m_val = getattr(mf, facet)
        if q_val and m_val and q_val != m_val:
            return True
    return False
