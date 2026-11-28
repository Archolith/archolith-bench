"""R1 retrieval metrics, computed from a ranked id list + the gold corpus.

The R1 win condition (deferred-verification.md) requires reporting these
together: a condition that lifts exact-string / symbol recall while inflating
stale-hit or wrong-scope injection has NOT won. ``exact_string_recall`` and
``symbol_recall`` are averaged only over the queries that actually target an
exact string / symbol (their family denominator), so an all-families run still
reports each family-specific metric honestly.
"""

from __future__ import annotations

from collections.abc import Sequence

from archolith_bench.core.ir_metrics import recall_at_k  # noqa: F401  (runner uses M.*)

from .models import R1Memory, R1Query


# ---------------------------------------------------------------------------
# Known-item retrieval (auto-generated eval, plan v6 §2 / §6.2)
#
# One query has ONE intended answer memory, plus its DUPLICATE CLUSTER: the clone
# is a duplicate of prod, and prod holds near-duplicate memories, so retrieving any
# semantically-identical sibling is a correct answer, not a miss. So the gold is a
# SET (the cluster) and "any member counts", which is the opposite of recall_at_k's
# "all members" semantics -- hence a separate function so the two are never confused.
# ---------------------------------------------------------------------------

# Sentinel used when the caller has no explicit limit; callers should pass the real
# recall() limit so an absent gold gets rank = limit + 1 (see known_item_rank).
_DEFAULT_ABSENT_RANK_BASE = 1_000_000


def known_item_rank(
    ranked: Sequence[str], gold_ids: Sequence[str], *, limit: int | None = None
) -> int:
    """1-based rank of the FIRST gold-cluster member in ``ranked``.

    ``gold_ids`` is the duplicate cluster (one or more ids that all count as the
    answer). If no cluster member appears, the gold is ABSENT and its rank is
    ``limit + 1`` -- a finite, total, monotone value so rank comparisons and MRR are
    always defined (plan v6 §6.2). ``limit`` should be the recall() ``limit`` used to
    produce ``ranked``; if omitted, a large sentinel base is used so absence still
    sorts strictly worse than any present rank.
    """
    gold = set(gold_ids)
    for i, mid in enumerate(ranked):
        if mid in gold:
            return i + 1
    base = limit if limit is not None else max(len(ranked), _DEFAULT_ABSENT_RANK_BASE)
    return base + 1


def known_item_recall_at_k(ranked: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """1.0 if ANY gold-cluster member is in the top-k, else 0.0.

    Binary per query (a query has one answer, present or not). Average across queries
    upstream to get recall@k. Note nested-ness: recall@5 == 1.0 implies recall@10 == 1.0,
    which is exactly why the win gate must use improvement_mode="any" (plan v6 §6.3).
    """
    if not gold_ids:
        return 0.0
    top = set(ranked[:k])
    return 1.0 if any(g in top for g in gold_ids) else 0.0


def reciprocal_rank(rank: int) -> float:
    """1/rank. Pairs with known_item_rank's limit+1 convention (never divides by zero)."""
    if rank <= 0:
        raise ValueError(f"rank must be >= 1 (got {rank}); known_item_rank never returns < 1")
    return 1.0 / rank


def mrr(ranks: Sequence[int]) -> float:
    """Mean reciprocal rank over per-query ranks (each from known_item_rank)."""
    if not ranks:
        return 0.0
    return sum(reciprocal_rank(r) for r in ranks) / len(ranks)


def rank_regressed(rank_challenger: int, rank_baseline: int, rank_tolerance: int = 0) -> bool:
    """True if the challenger's gold rank is worse than the baseline's, beyond tolerance.

    ``rank_tolerance`` is an INTEGER (ranks are integers) and is deliberately distinct
    from the float ``regress_tolerance`` used for metric deltas in the win gate
    (plan v6 §6.2). Worse == larger rank number.
    """
    return rank_challenger > rank_baseline + rank_tolerance


def exact_string_hit(
    ranked: Sequence[str], memories_by_id: dict[str, R1Memory], query: R1Query, k: int
) -> float | None:
    """1.0 if a top-k memory contains the query's target exact string, else 0.0.

    Returns None when the query has no exact-string target (excluded from the
    average). A memory "contains" the string if it is in its gold exact_strings
    set or appears verbatim in its text.
    """
    target = query.target_exact_string
    if not target:
        return None
    for mid in ranked[:k]:
        mem = memories_by_id.get(mid)
        if mem and (target in mem.exact_strings or target in mem.text):
            return 1.0
    return 0.0


def symbol_hit(
    ranked: Sequence[str], memories_by_id: dict[str, R1Memory], query: R1Query, k: int
) -> float | None:
    """1.0 if a top-k memory carries the query's target symbol, else 0.0.

    Returns None when the query has no symbol target (excluded from the average).
    """
    target = query.target_symbol
    if not target:
        return None
    for mid in ranked[:k]:
        mem = memories_by_id.get(mid)
        if mem and target in mem.symbols:
            return 1.0
    return 0.0


def stale_hit_rate(
    ranked: Sequence[str], memories_by_id: dict[str, R1Memory], query: R1Query, k: int
) -> float:
    """Fraction of top-k that are stale under a current-intent query.

    Always 0.0 for historical-intent queries: returning historical memories is
    correct when the query asks about history.
    """
    if query.intent != "current":
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    stale = sum(1 for mid in top if mid in memories_by_id and memories_by_id[mid].stale)
    return stale / len(top)


def wrong_scope_injection_rate(
    ranked: Sequence[str], memories_by_id: dict[str, R1Memory], query: R1Query, k: int
) -> float:
    """Fraction of top-k whose repo/project conflicts with the query's scope."""
    top = ranked[:k]
    if not top:
        return 0.0
    wrong = sum(
        1 for mid in top if mid in memories_by_id and _scope_conflict(query, memories_by_id[mid])
    )
    return wrong / len(top)


def _scope_conflict(query: R1Query, memory: R1Memory) -> bool:
    """True if repo or project is set on both sides and disagrees."""
    for attr in ("repo", "project"):
        q_val = getattr(query, attr)
        m_val = getattr(memory, attr)
        if q_val and m_val and q_val != m_val:
            return True
    return False
