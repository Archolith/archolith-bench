"""Pre-registered v2 verdict rules (rubric version ``sprint-v2``) as pure functions.

Everything here works on plain per-chain numbers so it can be unit-tested without a run.
A *chain* is one independent run of all of a fixture's sessions under one arm.

Rules, per repo, Menhir (M) against a baseline (B), with one number per chain:

* Accuracy WIN: mean(M) - mean(B) >= +0.02 AND min(M) > max(B).
* Accuracy LOSS: mean(M) - mean(B) <= -0.02 AND max(M) < min(B).
* Otherwise DIRECTIONAL (the mean passes the threshold but the ranges overlap) or
  NO_DIFFERENCE (the mean does not pass the threshold).
* Efficiency (primary = client uncached + server uncached tokens, lower is better):
  WIN: mean delta <= -10% AND max(M) < min(B). DIRECTIONAL / NO_DIFFERENCE as above;
  an increase of at least +10% is COST.
* The combined verdict states both parts. There is no OR rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Sequence

from archolith_bench.engineering_evidence.sprint import SessionOutcome

RUBRIC_VERSION = "sprint-v2"
ACCURACY_DELTA_THRESHOLD = 0.02
EFFICIENCY_DELTA_PCT_THRESHOLD = 10.0
_EPS = 1e-9

WIN = "WIN"
LOSS = "LOSS"
COST = "COST"
DIRECTIONAL = "DIRECTIONAL"
NO_DIFFERENCE = "NO_DIFFERENCE"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def ranges_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """True unless one set lies strictly above or strictly below the other."""
    if not a or not b:
        return False
    return not (min(a) > max(b) or max(a) < min(b))


def accuracy_verdict(menhir: Sequence[float], baseline: Sequence[float]) -> str:
    if not menhir or not baseline:
        return INSUFFICIENT_DATA
    delta = fmean(menhir) - fmean(baseline)
    if delta >= ACCURACY_DELTA_THRESHOLD - _EPS:
        return WIN if min(menhir) > max(baseline) else DIRECTIONAL
    if delta <= -ACCURACY_DELTA_THRESHOLD + _EPS:
        return LOSS if max(menhir) < min(baseline) else DIRECTIONAL
    return NO_DIFFERENCE


def efficiency_delta_pct(menhir: Sequence[float], baseline: Sequence[float]) -> float | None:
    if not menhir or not baseline:
        return None
    base = fmean(baseline)
    if base <= 0:
        return None
    return (fmean(menhir) - base) / base * 100.0


def efficiency_verdict(menhir: Sequence[float], baseline: Sequence[float]) -> str:
    pct = efficiency_delta_pct(menhir, baseline)
    if pct is None:
        return INSUFFICIENT_DATA
    if pct <= -EFFICIENCY_DELTA_PCT_THRESHOLD + _EPS:
        return WIN if max(menhir) < min(baseline) else DIRECTIONAL
    if pct >= EFFICIENCY_DELTA_PCT_THRESHOLD - _EPS:
        return COST
    return NO_DIFFERENCE


def combined_verdict(accuracy: str, efficiency: str) -> str:
    return f"ACCURACY_{accuracy}__EFFICIENCY_{efficiency}"


def compare_chains(menhir_chains: Sequence[dict[str, Any]], baseline_chains: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Verdicts from :func:`summarize_chain` rows. Only complete chains count.

    A chain with a session that was not run (campaign stopped early) is excluded from the
    verdict inputs rather than scored as 0.
    """
    m = [c for c in menhir_chains if c["complete"]]
    b = [c for c in baseline_chains if c["complete"]]
    m_acc = [c["accuracy"] for c in m]
    b_acc = [c["accuracy"] for c in b]
    m_eff = [c["primary_efficiency_tokens"] for c in m]
    b_eff = [c["primary_efficiency_tokens"] for c in b]
    acc = accuracy_verdict(m_acc, b_acc)
    eff = efficiency_verdict(m_eff, b_eff)
    return {
        "rubric_version": RUBRIC_VERSION,
        "chains_used": {"menhir": len(m), "baseline": len(b)},
        "chains_excluded_incomplete": {
            "menhir": len(menhir_chains) - len(m),
            "baseline": len(baseline_chains) - len(b),
        },
        "accuracy": {
            "menhir": m_acc,
            "baseline": b_acc,
            "mean_delta": None if not (m_acc and b_acc) else round(fmean(m_acc) - fmean(b_acc), 4),
            "ranges_overlap": ranges_overlap(m_acc, b_acc),
            "verdict": acc,
        },
        "efficiency": {
            "metric": "client_uncached_tokens + server_uncached_tokens",
            "menhir": m_eff,
            "baseline": b_eff,
            "mean_delta_pct": (
                None
                if efficiency_delta_pct(m_eff, b_eff) is None
                else round(efficiency_delta_pct(m_eff, b_eff) or 0.0, 2)
            ),
            "ranges_overlap": ranges_overlap(m_eff, b_eff),
            "verdict": eff,
        },
        "combined_verdict": combined_verdict(acc, eff),
    }


def summarize_chain(
    outcomes: Iterable[SessionOutcome],
    *,
    server: dict[str, Any] | None = None,
    exclude_sessions: frozenset[str] = frozenset(),
    expected_sessions: int | None = None,
) -> dict[str, Any]:
    """Per-chain numbers. *server* is that chain's attributed server usage (Menhir only).

    ``exclude_sessions`` drops those session ids from every sum (sensitivity report). Sessions
    that were not run are never scored as 0: they are left out of the accuracy mean and make
    the chain incomplete. Server totals are per chain, so they are not reduced by
    ``exclude_sessions`` (they cannot be split by session).
    """
    rows = [o for o in outcomes if o.session_id not in exclude_sessions]
    ran = [o for o in rows if not o.not_run]
    server = server or {}
    client_uncached = sum(o.uncached_tokens for o in ran)
    server_uncached = int(server.get("uncached_tokens", 0) or 0)
    expected = expected_sessions if expected_sessions is not None else len(rows)
    return {
        "accuracy": round(fmean(o.overall_score for o in ran), 4) if ran else 0.0,
        "sessions_run": len(ran),
        "sessions_not_run": len(rows) - len(ran),
        "complete": len(ran) == expected and len(ran) > 0,
        "censored_sessions": [o.session_id for o in ran if o.censored],
        "client_uncached_tokens": client_uncached,
        "server_uncached_tokens": server_uncached,
        "primary_efficiency_tokens": client_uncached + server_uncached,
        "client_total_tokens": sum(o.total_tokens for o in ran),
        "server_total_tokens": int(server.get("total_tokens", 0) or 0),
        "input_tokens": sum(o.input_tokens for o in ran),
        "output_tokens": sum(o.output_tokens for o in ran),
        "reasoning_tokens": sum(o.reasoning_tokens for o in ran),
        "cache_read_tokens": sum(o.cache_read_tokens for o in ran),
        "cache_write_tokens": sum(o.cache_write_tokens for o in ran),
        "client_cost_usd": round(sum(o.cost_usd for o in ran), 6),
        "server_cost_usd": round(float(server.get("cost_usd", 0.0) or 0.0), 6),
        "server_calls": int(server.get("calls", 0) or 0),
        "turns": sum(o.turns for o in ran),
        "tool_calls": sum(o.tool_calls for o in ran),
        "files_read": sum(len(o.files_read) for o in ran),
        "redundant_file_reads": sum(o.redundant_file_reads for o in ran),
        "notes_violations": sum(len(o.notes_violations) for o in ran),
    }


def sessions_censored_anywhere(chains: Iterable[Iterable[SessionOutcome]]) -> frozenset[str]:
    """Session ids where at least one chain (any arm) was censored."""
    return frozenset(
        o.session_id for outcomes in chains for o in outcomes if o.censored and not o.not_run
    )


@dataclass(frozen=True)
class ChainView:
    arm: str
    repeat: int
    outcomes: tuple[SessionOutcome, ...]
    server: dict[str, Any]


def build_verdict_report(
    chains: Sequence[ChainView],
    *,
    expected_sessions: int,
    primary_baseline: str = "notes_file",
    secondary_baseline: str = "no_menhir",
    menhir_arm: str = "menhir_longitudinal",
) -> dict[str, Any]:
    """Main verdicts plus the censored-session sensitivity report, from per-chain data."""

    def _summaries(exclude: frozenset[str]) -> dict[str, list[dict[str, Any]]]:
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for chain in sorted(chains, key=lambda c: (c.arm, c.repeat)):
            row = summarize_chain(
                chain.outcomes,
                server=chain.server,
                exclude_sessions=exclude,
                expected_sessions=expected_sessions - len(exclude & {o.session_id for o in chain.outcomes}),
            )
            row["repeat"] = chain.repeat
            by_arm.setdefault(chain.arm, []).append(row)
        return by_arm

    def _compare(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        m = by_arm.get(menhir_arm, [])
        for label, base in (("primary_vs_" + primary_baseline, primary_baseline),
                            ("secondary_vs_" + secondary_baseline, secondary_baseline)):
            if base in by_arm and m:
                out[label] = compare_chains(m, by_arm[base])
        return out

    main = _summaries(frozenset())
    censored = sessions_censored_anywhere(c.outcomes for c in chains)
    sensitivity_arms = _summaries(censored)
    return {
        "rubric_version": RUBRIC_VERSION,
        "per_chain": main,
        "verdicts": _compare(main),
        "sensitivity_excluding_censored_sessions": {
            "excluded_session_ids": sorted(censored),
            "note": (
                "Server usage is per chain and cannot be split by session, so it is included "
                "unchanged here."
            ),
            "per_chain": sensitivity_arms,
            "verdicts": _compare(sensitivity_arms),
        },
    }
