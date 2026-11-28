"""Binary-relevance IR metrics over a ranked id list and a set of gold support ids.

Shared by the facet, oracle, r1 and intent benchmarks. Known-item (any-cluster-member)
variants stay in r1/metrics.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"]


def recall_at_k(ranked: Sequence[str], support_ids: Sequence[str], k: int) -> float:
    """Fraction of gold support ids present in the top-k (all of them are wanted)."""
    if not support_ids:
        return 0.0
    top = set(ranked[:k])
    hit = sum(1 for sid in support_ids if sid in top)
    return hit / len(support_ids)


def precision_at_k(ranked: Sequence[str], support_ids: Sequence[str], k: int) -> float:
    """Fraction of the top-k that are gold support ids."""
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    support = set(support_ids)
    hit = sum(1 for mid in top if mid in support)
    return hit / len(top)


def mrr(ranked: Sequence[str], support_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first gold support id (0 if none retrieved)."""
    support = set(support_ids)
    for position, mid in enumerate(ranked, start=1):
        if mid in support:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], support_ids: Sequence[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    support = set(support_ids)
    if not support:
        return 0.0
    dcg = 0.0
    for position, mid in enumerate(ranked[:k], start=1):
        if mid in support:
            dcg += 1.0 / math.log2(position + 1)
    ideal_hits = min(len(support), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
