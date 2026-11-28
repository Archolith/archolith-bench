"""One classifier for rate-limit and API error responses.

Classification only: retry/stop policy belongs to each caller. Flex capacity 429s are
retryable; quota, billing and auth failures are stops (workspace rule).
"""

from __future__ import annotations

from enum import Enum


class Kind(str, Enum):
    CAPACITY = "capacity"
    QUOTA = "quota"
    BILLING = "billing"
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    OTHER = "other"


CAPACITY_MARKERS = (
    "flex_unavailable",
    "resource_unavailable",
    "resource unavailable",
    "flex does not have sufficient resources",
)
QUOTA_MARKERS = ("insufficient_quota",)
BILLING_MARKERS = ("credit_balance", "billing_hard_limit", "billing_not_active")


def mentions_capacity(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in CAPACITY_MARKERS)


def classify(status: int | None, text: str) -> Kind:
    lowered = (text or "").lower()
    if status == 429 and mentions_capacity(lowered):
        return Kind.CAPACITY
    if any(marker in lowered for marker in QUOTA_MARKERS):
        return Kind.QUOTA
    if status == 402 or any(marker in lowered for marker in BILLING_MARKERS):
        return Kind.BILLING
    if status in (401, 403):
        return Kind.AUTH
    if status == 429:
        return Kind.RATE_LIMIT
    return Kind.OTHER
