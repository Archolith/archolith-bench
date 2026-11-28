"""Metrics for the intent benchmark.

Headline metric is role-based intent-correct@1: did the top-1 ranked candidate carry a
role that is PREFERRED for the query's classified intent? This is robust to ties among
several preferred-role candidates (the design's point is that intent surfaces a
*task-appropriate* artifact first, not one exact id). nDCG (for the no-harm arm) is
reused from core.ir_metrics.
"""

from __future__ import annotations

from typing import Mapping

from archolith_bench.core.ir_metrics import ndcg_at_k  # re-exported for the no-harm arm

from .classifier import TaskIntent
from .matrix import Affinity, affinity
from .roles import derive_content_role

__all__ = ["ndcg_at_k", "role_is_preferred", "intent_correct_at_1"]


def role_is_preferred(metadata: Mapping[str, object], intent: TaskIntent) -> bool:
    """True if any role the candidate carries is PREFERRED for the intent."""
    return any(affinity(intent, role) is Affinity.PREFER for role in derive_content_role(metadata))


def intent_correct_at_1(top_metadata: Mapping[str, object] | None, intent: TaskIntent) -> float:
    """1.0 if the top-1 candidate's role is preferred for the (true) intent, else 0.0."""
    if top_metadata is None:
        return 0.0
    return 1.0 if role_is_preferred(top_metadata, intent) else 0.0
