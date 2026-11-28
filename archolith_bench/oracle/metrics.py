"""Metrics for the oracle ladder (menhir R4-R7).

Standard retrieval metrics plus the temporal/scope metrics the oracle pipeline is
*supposed* to move — reported together, because a combiner that suppresses stale
hits while gutting recall has not won (same discipline as the facet bench).

All deterministic; nothing here needs a generation model.
"""

from __future__ import annotations

from collections.abc import Sequence

from archolith_bench.core.ir_metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k  # noqa: F401  (runner uses M.*)

from .models import SCOPE_KEYS, OracleMemory, OracleQuery


def stale_hit_rate(ranked: Sequence[str], by_id: dict[str, OracleMemory], query: OracleQuery, k: int) -> float:
    """Top-k fraction that is stale under a current-intent query (0 for historical)."""
    if query.intent != "current":
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for mid in top if mid in by_id and by_id[mid].is_stale) / len(top)


def wrong_scope_injection_rate(
    ranked: Sequence[str], by_id: dict[str, OracleMemory], query: OracleQuery, k: int
) -> float:
    """Top-k fraction whose scope keys conflict with the query's scope."""
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for mid in top if mid in by_id and _scope_conflict(query, by_id[mid])) / len(top)


def current_truth_suppression_accuracy(
    ranked: Sequence[str], by_id: dict[str, OracleMemory], query: OracleQuery, k: int
) -> float:
    """Of the stale memories in the corpus, the fraction correctly kept OUT of top-k.

    Only meaningful for current-intent queries; returns 1.0 when there is nothing
    stale to suppress (no false credit, no false penalty).
    """
    if query.intent != "current":
        return 1.0
    stale_ids = [mid for mid, m in by_id.items() if m.is_stale]
    if not stale_ids:
        return 1.0
    top = set(ranked[:k])
    suppressed = sum(1 for mid in stale_ids if mid not in top)
    return suppressed / len(stale_ids)


def historical_context_preservation(
    ranked: Sequence[str], support_ids: Sequence[str], query: OracleQuery, k: int
) -> float:
    """For historical-intent queries, recall of the (historical) gold support.

    Guards against a combiner that learns to suppress everything stale and thereby
    fails historical queries. 1.0 (n/a) for non-historical queries.
    """
    if query.intent != "historical":
        return 1.0
    return recall_at_k(ranked, support_ids, k)


def _scope_conflict(query: OracleQuery, memory: OracleMemory) -> bool:
    for key in SCOPE_KEYS:
        q_val = getattr(query, key)
        m_val = getattr(memory, key)
        if q_val and m_val and q_val != m_val:
            return True
    return False
