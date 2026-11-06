#!/usr/bin/env python
"""Test whether a real cosine beats rank scores: H1-H4 from labelled recall batches. No LLM.

Plan: menhir-recall-real-similarity-test-plan-2026-11-05. Inputs come from Menhir's
``scripts/recall_lab_batch.py``, ``candidate_labels.py`` and, optionally,
``recall_summary_cosine.py``:

    python recall_signal_eval.py --batch out.jsonl --labels labels.jsonl --queries queries.jsonl \
        --dataset o500.json --arms arms.json --summary-cosines summary.jsonl \
        --output report.json --markdown report.md

Every constant below is pre-registered. Changing one after opening Phase 3 data voids the
hypothesis it feeds. The unit of analysis is the question, and CIs are paired percentile
bootstraps over questions, so every signal sees the same resamples.

Answer accuracy is not read here. A floor that clears H3 here stays ``pending_accuracy`` until
the paid end-to-end check, and H4 stays ``pending_r1`` until the R1 transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_labels import (  # noqa: E402
    Gold,
    load_batch,
    load_gold,
    load_items,
    query_kind,
    question_of,
)

SEED = 20261105
N_BOOT = 2000
BOOT_CHUNK = 100
BASELINE_ARM = "base"
KS = (5, 10)
PRIMARY_K = 10

H1_SUPPORT_RATIO = 0.50
H1_KILL_RATIO = 0.10
H2_MIN_GAIN = 0.05
H3_MAX_LOSS = 0.01
H3_MIN_NOISE_CUT = 0.25
H3_MIN_TOKEN_CUT = 0.15
H4_MAX_ECE = 0.05
H4_KILL_ECE = 0.10
ECE_BINS = 10
FOLDS = 5
LOGISTIC_L2 = 1.0

COSINE = "name_cosine"
RRF = "rrf"
PRIMARY_CELL = "strict_no_hubs"
CELLS = {
    "strict_no_hubs": ("label_strict", False),
    "strict_with_hubs": ("label_strict", True),
    "lenient_no_hubs": ("label_lenient", False),
    "lenient_with_hubs": ("label_lenient", True),
}
TYPE_CODES = {
    "knowledge-update": "KU",
    "multi-session": "MS",
    "single-session-user": "SSU",
    "temporal-reasoning": "TR",
    "single-session-preference": "SSP",
    "single-session-assistant": "SSA",
}
FIT_TYPES = frozenset({"KU", "MS", "SSU"})
TEST_TYPES = frozenset({"TR", "SSP"})
OVERLAP_STRATA = (("none", 0.0, 0.0), ("partial", 0.0, 0.5), ("high", 0.5, 1.0))

Row = dict[str, Any]


# --- Ranking metrics --------------------------------------------------------------------------


def _weighted_metrics(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """AUC and average precision per row of ``weights``, one weight per group.

    Tied scores form one block: a tied pair counts half for AUC, and average precision reads
    precision at the end of the block, so neither depends on input order.
    """
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    y = labels[order].astype(float)
    item_weights = weights[:, groups[order]].astype(float)
    pos = np.add.reduceat(item_weights * y, starts, axis=1)
    neg = np.add.reduceat(item_weights * (1 - y), starts, axis=1)
    tp, fp = pos.cumsum(axis=1), neg.cumsum(axis=1)
    total_pos, total_neg = tp[:, -1], fp[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        wins = (pos * (total_neg[:, None] - fp)).sum(axis=1) + 0.5 * (pos * neg).sum(axis=1)
        auc = wins / (total_pos * total_neg)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        average_precision = (pos * precision).sum(axis=1) / total_pos
    return auc, average_precision


def auc_ap(scores: Iterable[float], labels: Iterable[bool]) -> tuple[float, float]:
    """Unweighted AUC and average precision. NaN when one class is absent."""
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=bool)
    auc, ap = _weighted_metrics(s, y, np.zeros(len(s), dtype=int), np.ones((1, 1)))
    return float(auc[0]), float(ap[0])


def bootstrap_weights(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    """How often each group appears in each resample. The same seed pairs every signal."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(n_boot, n_groups))
    weights = np.zeros((n_boot, n_groups), dtype=np.int32)
    np.add.at(weights, (np.arange(n_boot)[:, None], picks), 1)
    return weights


def interval(estimate: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {"estimate": float(estimate), "low": float(low), "high": float(high)}


def signal_metrics(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, weights: np.ndarray
) -> dict[str, tuple[float, np.ndarray]]:
    """Pooled and mean within-question AUC and PR-AUC, each as (estimate, resamples)."""
    n_groups = weights.shape[1]
    pooled_auc, pooled_ap = _weighted_metrics(scores, labels, groups, np.ones((1, n_groups)))
    boot_auc, boot_ap = [], []
    for start in range(0, len(weights), BOOT_CHUNK):
        chunk_auc, chunk_ap = _weighted_metrics(
            scores, labels, groups, weights[start:start + BOOT_CHUNK]
        )
        boot_auc.append(chunk_auc)
        boot_ap.append(chunk_ap)

    per_auc = np.full(n_groups, np.nan)
    per_ap = np.full(n_groups, np.nan)
    for group in range(n_groups):
        mask = groups == group
        per_auc[group], per_ap[group] = auc_ap(scores[mask], labels[mask])
    valid = ~np.isnan(per_auc)

    def within(values: np.ndarray) -> tuple[float, np.ndarray]:
        filled = np.where(valid, values, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = (weights @ filled) / (weights @ valid.astype(float))
        return float(np.nanmean(values)), samples

    return {
        "pooled_auc": (float(pooled_auc[0]), np.concatenate(boot_auc)),
        "pooled_pr_auc": (float(pooled_ap[0]), np.concatenate(boot_ap)),
        "within_auc": within(per_auc),
        "within_pr_auc": within(per_ap),
    }


# --- Candidate table --------------------------------------------------------------------------


def tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def token_overlap(query: str | None, name: str | None) -> float:
    """Share of the name's tokens that appear in the query."""
    name_tokens = tokens(name)
    return len(name_tokens & tokens(query)) / len(name_tokens) if name_tokens else 0.0


def _missing_low(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def candidate_table(
    rows: Iterable[Row],
    labels: dict[tuple[str, str], Row],
    queries: dict[str, str],
    summary_cosines: dict[tuple[str, str], float | None],
    arm: str = BASELINE_ARM,
) -> list[Row]:
    """Every plain-entity candidate of real queries in one arm, floor-dropped ones included.

    Views, source memories and unlabelled candidates are left out of H2; ``label_report`` counts
    them. A missing signal ranks last, so it cannot help the signal that lacks it.
    """
    table: list[Row] = []
    for row in rows:
        if row["arm"] != arm:
            continue
        pool = row.get("candidates") or ()
        similarities = sorted(float(c["similarity"]) for c in pool)
        low, high = (similarities[0], similarities[-1]) if similarities else (0.0, 0.0)
        for candidate in pool:
            meta = labels.get((row["id"], candidate["uuid"]))
            if not meta or meta["kind"] != "entity" or meta["query_kind"] != "real":
                continue
            similarity = float(candidate["similarity"])
            name_cosine = candidate.get("name_cosine")
            summary = summary_cosines.get((row["id"], candidate["uuid"]))
            present = [v for v in (name_cosine, summary) if v is not None]
            bm25 = candidate.get("bm25_rank")
            below = sum(s < similarity for s in similarities)
            table.append({
                "id": row["id"],
                "uuid": candidate["uuid"],
                "type": TYPE_CODES.get(meta["question_type"], str(meta["question_type"])),
                "source": candidate["source"],
                "is_hub": bool(meta["is_hub"]),
                "label_strict": meta["label_strict"],
                "label_lenient": meta["label_lenient"],
                "label_route": meta["label_route"],
                "lane_agreement": bool(
                    candidate.get("cosine_rank") is not None and bm25 is not None
                ),
                "overlap": token_overlap(queries.get(row["id"]), meta.get("name")),
                "signals": {
                    RRF: similarity,
                    "final_score": _missing_low(candidate.get("final_score")),
                    COSINE: _missing_low(name_cosine),
                    "summary_cosine": _missing_low(summary),
                    "max_name_summary": max(present) if present else float("-inf"),
                    "rrf_minmax": (similarity - low) / (high - low) if high > low else 0.0,
                    "rrf_percentile": below / max(1, len(similarities) - 1),
                    "bm25_rank": float("-inf") if bm25 is None else -float(bm25),
                    "token_overlap": token_overlap(queries.get(row["id"]), meta.get("name")),
                },
            })
    return table


def cell_rows(table: list[Row], cell: str) -> tuple[list[Row], str]:
    """The rows one H2 cell scores. SSA is reported alone, and untraceable rows are never scored."""
    label_key, keep_hubs = CELLS[cell]
    rows = [
        r for r in table
        if r["type"] != "SSA" and r[label_key] is not None and (keep_hubs or not r["is_hub"])
    ]
    return rows, label_key


def _arrays(rows: list[Row], label_key: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = sorted({r["id"] for r in rows})
    index = {qid: i for i, qid in enumerate(ids)}
    groups = np.array([index[r["id"]] for r in rows], dtype=int)
    labels = np.array([bool(r[label_key]) for r in rows])
    return groups, labels, ids


def score_cell(table: list[Row], cell: str, n_boot: int, seed: int) -> dict[str, Any]:
    rows, label_key = cell_rows(table, cell)
    if not rows:
        return {"questions": 0, "candidates": 0, "signals": {}, "cosine_minus": {}}
    groups, labels, ids = _arrays(rows, label_key)
    weights = bootstrap_weights(len(ids), n_boot, seed)
    metrics = {
        name: signal_metrics(np.array([r["signals"][name] for r in rows]), labels, groups, weights)
        for name in rows[0]["signals"]
    }
    return {
        "questions": len(ids),
        "candidates": len(rows),
        "positive_share": float(labels.mean()),
        "signals": {
            name: {m: interval(est, samples) for m, (est, samples) in values.items()}
            for name, values in metrics.items()
        },
        "cosine_minus": {
            name: {
                m: interval(metrics[COSINE][m][0] - est, metrics[COSINE][m][1] - samples)
                for m, (est, samples) in values.items()
            }
            for name, values in metrics.items() if name != COSINE
        },
    }


def h2_verdict(primary: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered reading of the primary cell, against RRF.

    Supported: gain of at least +0.05 with CI above 0 on both pooled and within-question AUC.
    Killed: neither CI is above 0, or RRF wins (CI below 0) on either. Otherwise inconclusive,
    which covers a gain on one scale only.
    """
    versus = primary.get("cosine_minus", {}).get(RRF)
    if not versus:
        return {"status": "no_data"}
    pooled, within = versus["pooled_auc"], versus["within_auc"]
    gains = [d["estimate"] >= H2_MIN_GAIN and d["low"] > 0 for d in (pooled, within)]
    above = [d["low"] > 0 for d in (pooled, within)]
    rrf_wins = any(d["high"] < 0 for d in (pooled, within))
    if all(gains):
        status = "supported"
    elif rrf_wins or not any(above):
        status = "killed"
    else:
        status = "inconclusive"
    return {"status": status, "pooled_auc_gain": pooled, "within_auc_gain": within}


def overlap_strata(table: list[Row]) -> dict[str, Any]:
    """Pooled AUC inside token-overlap strata: does cosine add anything beyond shared words?"""
    rows, label_key = cell_rows(table, PRIMARY_CELL)
    report: dict[str, Any] = {}
    for name, low, high in OVERLAP_STRATA:
        chosen = [
            r for r in rows
            if (r["overlap"] == 0 if high == 0 else low < r["overlap"] <= high)
        ]
        labels = [bool(r[label_key]) for r in chosen]
        entry: dict[str, Any] = {"candidates": len(chosen), "positives": sum(labels)}
        if chosen:
            for signal in (COSINE, RRF):
                entry[signal] = auc_ap([r["signals"][signal] for r in chosen], labels)[0]
        report[name] = entry
    return report


def per_type(table: list[Row]) -> dict[str, Any]:
    """Mean within-question AUC per question type, SSA included. Primary label, no hubs."""
    label_key, _ = CELLS[PRIMARY_CELL]
    by_type: dict[str, dict[str, list[Row]]] = defaultdict(lambda: defaultdict(list))
    for row in table:
        if row[label_key] is not None and not row["is_hub"]:
            by_type[row["type"]][row["id"]].append(row)
    report = {}
    for code, questions in sorted(by_type.items()):
        scores: dict[str, list[float]] = {COSINE: [], RRF: []}
        for rows in questions.values():
            labels = [bool(r[label_key]) for r in rows]
            for signal, values in scores.items():
                values.append(auc_ap([r["signals"][signal] for r in rows], labels)[0])
        report[code] = {"questions": len(questions)} | {
            s: float(np.nanmean(v)) if not np.all(np.isnan(v)) else None for s, v in scores.items()
        }
    return report


def coverage_ceiling(
    table: list[Row], kinds: dict[str, str], types: dict[str, str]
) -> dict[str, Any]:
    """Share of gold questions with any answer-bearing candidate in the pre-floor pool.

    A question that returned no candidates stays in the denominator as a miss.
    """
    gold_ids = {i for i, kind in kinds.items() if kind == "real"}
    outside_ssa = {i for i in gold_ids if TYPE_CODES.get(types.get(i)) != "SSA"}
    report: dict[str, Any] = {"questions": len(gold_ids), "questions_outside_ssa": len(outside_ssa)}
    for label_key in ("label_strict", "label_lenient"):
        hit = {r["id"] for r in table if r[label_key]}
        report[label_key] = len(hit & gold_ids) / len(gold_ids) if gold_ids else None
        report[f"{label_key}_outside_ssa"] = (
            len(hit & outside_ssa) / len(outside_ssa) if outside_ssa else None
        )
    return report


# --- H1 ---------------------------------------------------------------------------------------


def score_h1(rows: list[Row], kinds: dict[str, str], n_boot: int, seed: int) -> dict[str, Any]:
    """Share of queries with a ``high`` result, nulls against real, with a bootstrap ratio CI."""
    baseline = [r for r in rows if r["arm"] == BASELINE_ARM]

    def flags(wanted: frozenset[str]) -> np.ndarray:
        return np.array([
            any(item.get("relevance") == "high" for item in r.get("results") or ())
            for r in baseline if kinds.get(r["id"]) in wanted
        ], dtype=float)

    real = flags(frozenset({"real"}))
    report: dict[str, Any] = {"real": {"queries": len(real)}}
    if real.size:
        report["real"]["high_share"] = float(real.mean())
    nulls = flags(frozenset({"swapped", "abstention"}))
    for name, values in (("swapped", flags(frozenset({"swapped"}))),
                         ("abstention", flags(frozenset({"abstention"}))),
                         ("all_nulls", nulls)):
        report[name] = {"queries": len(values)}
        if values.size:
            report[name]["high_share"] = float(values.mean())
    if not real.size or not nulls.size or not real.mean():
        report["verdict"] = {"status": "no_data"}
        return report

    rng = np.random.default_rng(seed)
    real_boot = rng.choice(real, size=(n_boot, real.size)).mean(axis=1)
    null_boot = rng.choice(nulls, size=(n_boot, nulls.size)).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = interval(float(nulls.mean() / real.mean()), null_boot / real_boot)
    if ratio["estimate"] >= H1_SUPPORT_RATIO:
        status = "supported"
    elif ratio["estimate"] < H1_KILL_RATIO:
        status = "killed"
    else:
        status = "inconclusive"
    report["verdict"] = {"status": status, "null_to_real_ratio": ratio}
    return report


# --- H3 ---------------------------------------------------------------------------------------


def _top(row: Row, k: int) -> list[Row]:
    return sorted(row.get("results") or (), key=lambda r: r["rank"])[:k]


def _answers(results: list[Row], labels: dict[tuple[str, str], Row], row_id: str) -> set[str]:
    return {
        r["uuid"] for r in results
        if (labels.get((row_id, r["uuid"])) or {}).get("label_lenient")
    }


def _noise(results: list[Row], labels: dict[tuple[str, str], Row], row_id: str) -> int:
    """Results known not to bear the answer. Untraceable results are not noise."""
    return sum(
        (labels.get((row_id, r["uuid"])) or {}).get("label_lenient") is False for r in results
    )


def _floor_dropped(row: Row) -> list[str]:
    return sorted(c["uuid"] for c in row.get("candidates") or () if not c.get("survived_floor"))


def score_arm(
    baseline: dict[str, Row],
    arm: dict[str, Row],
    labels: dict[tuple[str, str], Row],
    kinds: dict[str, str],
    types: dict[str, str],
    k: int,
) -> dict[str, Any]:
    """One floor arm against the baseline at one k, on questions present in both.

    Loss is over real questions outside SSA. Noise and tokens cover real and null queries, and
    are also split out because null queries are where a floor should do its work.
    """
    gold_ids = [
        i for i in baseline
        if i in arm and kinds.get(i) == "real" and TYPE_CODES.get(types.get(i)) != "SSA"
    ]
    flips: list[Row] = []
    lost = 0
    for row_id in gold_ids:
        before = _answers(_top(baseline[row_id], k), labels, row_id)
        after = _answers(_top(arm[row_id], k), labels, row_id)
        lost += bool(before) and not after
        if before == after:
            continue
        base_dropped = set(_floor_dropped(baseline[row_id]))
        fired = sorted(set(_floor_dropped(arm[row_id])) - base_dropped)
        flips.append({
            "id": row_id,
            "type": TYPE_CODES.get(types.get(row_id), types.get(row_id)),
            "direction": "lost_last" if before and not after else (
                "gained_first" if after and not before else "moved"
            ),
            "left_top_k": sorted(before - after),
            "entered_top_k": sorted(after - before),
            "floor_fired": bool(fired),
            "floor_dropped_answers": sorted(before & set(fired)),
        })

    def totals(ids: list[str]) -> dict[str, Any]:
        noise_before = sum(_noise(_top(baseline[i], k), labels, i) for i in ids)
        noise_after = sum(_noise(_top(arm[i], k), labels, i) for i in ids)
        tokens_before = sum(r.get("tokens", 0) for i in ids for r in _top(baseline[i], k))
        tokens_after = sum(r.get("tokens", 0) for i in ids for r in _top(arm[i], k))
        return {
            "queries": len(ids),
            "noise_before": noise_before,
            "noise_after": noise_after,
            "noise_cut": 1 - noise_after / noise_before if noise_before else None,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "token_cut": 1 - tokens_after / tokens_before if tokens_before else None,
        }

    shared = [i for i in baseline if i in arm]
    return {
        "k": k,
        "gold_questions": len(gold_ids),
        "lost_questions": lost,
        "loss_rate": lost / len(gold_ids) if gold_ids else None,
        "all": totals(shared),
        "real": totals([i for i in shared if kinds.get(i) == "real"]),
        "null": totals([i for i in shared if kinds.get(i) in ("swapped", "abstention")]),
        "flips": flips,
    }


def h3_verdict(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered reading at ``PRIMARY_K``, over all queries. Accuracy comes later."""
    bounded, candidates = [], []
    for arm_id, report in arms.items():
        cut = next((r for r in report["by_k"] if r["k"] == PRIMARY_K), None)
        if not cut or cut["loss_rate"] is None or cut["loss_rate"] > H3_MAX_LOSS:
            continue
        noise, token = cut["all"]["noise_cut"] or 0.0, cut["all"]["token_cut"] or 0.0
        if noise > 0:
            bounded.append(arm_id)
        if noise >= H3_MIN_NOISE_CUT or token >= H3_MIN_TOKEN_CUT:
            candidates.append(arm_id)
    if not arms:
        status = "no_data"
    elif candidates:
        status = "pending_accuracy"
    elif bounded:
        status = "inconclusive"
    else:
        status = "killed"
    return {"status": status, "candidates": candidates, "within_loss_bound": bounded}


def score_h3(
    rows: list[Row],
    labels: dict[tuple[str, str], Row],
    kinds: dict[str, str],
    types: dict[str, str],
    arm_tuning: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["id"]] = row
    baseline = by_arm.get(BASELINE_ARM, {})
    arms = {
        arm_id: {
            "tau": arm_tuning.get(arm_id, {}).get("name_cosine_floor"),
            "all_sources": arm_tuning.get(arm_id, {}).get("name_cosine_floor_all_sources", False),
            "by_k": [score_arm(baseline, arm, labels, kinds, types, k) for k in KS],
        }
        for arm_id, arm in sorted(by_arm.items()) if arm_id != BASELINE_ARM
    }
    return {"arms": arms, "verdict": h3_verdict(arms)}


# --- H4 ---------------------------------------------------------------------------------------


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = LOGISTIC_L2) -> np.ndarray:
    """Newton-fit L2 logistic regression. Column 0 of ``x`` is the unpenalized intercept."""
    weights = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1]) * l2
    penalty[0, 0] = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))
        gradient = x.T @ (p - y) + penalty @ weights
        hessian = (x * (p * (1 - p))[:, None]).T @ x + penalty + np.eye(x.shape[1]) * 1e-9
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.abs(step).max() < 1e-8:
            break
    return weights


def predict_logistic(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool adjacent violators. Returns block upper bounds and block means, both ascending."""
    finite = np.isfinite(x)
    unique, inverse = np.unique(x[finite], return_inverse=True)
    sums = np.bincount(inverse, weights=y[finite])
    counts = np.bincount(inverse).astype(float)
    blocks: list[list[float]] = []
    for upper, total, count in zip(unique, sums, counts):
        blocks.append([float(upper), float(total), float(count)])
        while len(blocks) > 1 and (
            blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]
        ):
            last = blocks.pop()
            blocks[-1] = [last[0], blocks[-1][1] + last[1], blocks[-1][2] + last[2]]
    return np.array([b[0] for b in blocks]), np.array([b[1] / b[2] for b in blocks])


def predict_isotonic(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    uppers, means = model
    return means[np.minimum(np.searchsorted(uppers, x, side="left"), len(means) - 1)]


def reliability(p: np.ndarray, y: np.ndarray) -> list[Row]:
    bins = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    table = []
    for b in range(ECE_BINS):
        mask = bins == b
        if mask.any():
            table.append({
                "bin": b, "count": int(mask.sum()),
                "predicted": float(p[mask].mean()), "observed": float(y[mask].mean()),
            })
    return table


def expected_calibration_error(p: np.ndarray, y: np.ndarray) -> float:
    table = reliability(p, y)
    return sum(r["count"] * abs(r["predicted"] - r["observed"]) for r in table) / len(p)


def _features(rows: list[Row], sources: list[str], use_summary: bool) -> np.ndarray:
    """Cosines, lane agreement, rank percentile and source. A missing cosine gets a flag."""
    cosines = (COSINE, "summary_cosine") if use_summary else (COSINE,)
    columns = []
    for row in rows:
        signals = row["signals"]
        values: list[float] = []
        for name in cosines:
            present = np.isfinite(signals[name])
            values += [signals[name] if present else 0.0, 0.0 if present else 1.0]
        values += [float(row["lane_agreement"]), signals["rrf_percentile"]]
        values += [float(row["source"] == s) for s in sources]
        columns.append(values)
    # An empty split still needs its width.
    return np.array(columns, dtype=float).reshape(len(rows), 2 * len(cosines) + 2 + len(sources))


def _standardize(fit: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, spread = fit.mean(axis=0), fit.std(axis=0)
    spread[spread == 0] = 1.0
    return [
        np.c_[np.ones(len(m)), (m - mean) / spread] for m in (fit, *others)
    ]


def _evaluate(
    fit: list[Row], tests: dict[str, list[Row]], label_key: str
) -> dict[str, Any]:
    """Fit on ``fit`` and score each test split: logistic, logistic+isotonic, isotonic RRF."""
    sources = sorted({r["source"] for r in fit})
    use_summary = any(np.isfinite(r["signals"]["summary_cosine"]) for r in fit)
    fit_y = np.array([bool(r[label_key]) for r in fit], dtype=float)
    fit_rrf = np.array([r["signals"][RRF] for r in fit])
    names = list(tests)
    fit_x, *test_x = _standardize(
        _features(fit, sources, use_summary),
        *[_features(tests[n], sources, use_summary) for n in names],
    )
    weights = fit_logistic(fit_x, fit_y)
    stacked = fit_isotonic(predict_logistic(fit_x, weights), fit_y)
    rrf_model = fit_isotonic(fit_rrf, fit_y)

    report: dict[str, Any] = {"coefficients": weights.tolist(), "sources": sources,
                              "uses_summary_cosine": use_summary, "splits": {}}
    for name, x in zip(names, test_x):
        rows = tests[name]
        y = np.array([bool(r[label_key]) for r in rows], dtype=float)
        logistic = predict_logistic(x, weights)
        predictions = {
            "logistic": logistic,
            "logistic_isotonic": predict_isotonic(logistic, stacked),
            "rrf_isotonic": predict_isotonic(np.array([r["signals"][RRF] for r in rows]), rrf_model),
        }
        report["splits"][name] = {
            "candidates": len(rows),
            "positive_share": float(y.mean()) if len(y) else None,
            "models": {
                model: {
                    "ece": expected_calibration_error(p, y),
                    "brier": float(np.mean((p - y) ** 2)),
                    "reliability": reliability(p, y),
                }
                for model, p in predictions.items()
            } if len(y) else {},
        }
    return report


def _fold(question_id: str) -> int:
    return int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % FOLDS


def score_h4(table: list[Row]) -> dict[str, Any]:
    """Fit on KU, MS and SSU; test on TR and SSP; SSA alone; then a 5-fold check by question."""
    label_key, _ = CELLS[PRIMARY_CELL]
    usable = [r for r in table if r[label_key] is not None and not r["is_hub"]]
    fit = [r for r in usable if r["type"] in FIT_TYPES]
    held_out = [r for r in usable if r["type"] in TEST_TYPES]
    ssa = [r for r in usable if r["type"] == "SSA"]
    if not fit or not held_out or len({bool(r[label_key]) for r in fit}) < 2:
        return {"verdict": {"status": "no_data"}}

    primary = _evaluate(fit, {"held_out": held_out, "ssa": ssa}, label_key)
    folds = []
    outside_ssa = [r for r in usable if r["type"] != "SSA"]
    for fold in range(FOLDS):
        train = [r for r in outside_ssa if _fold(r["id"]) != fold]
        test = [r for r in outside_ssa if _fold(r["id"]) == fold]
        if test and len({bool(r[label_key]) for r in train}) == 2:
            split = _evaluate(train, {"fold": test}, label_key)["splits"]["fold"]
            folds.append({"fold": fold, **{
                m: {"ece": v["ece"], "brier": v["brier"]} for m, v in split["models"].items()
            }})

    models = primary["splits"]["held_out"]["models"]
    ece = models["logistic"]["ece"]
    gain = models["logistic"]["brier"] < models["rrf_isotonic"]["brier"]
    if ece > H4_KILL_ECE or not gain:
        status = "killed"
    elif ece <= H4_MAX_ECE:
        status = "pending_r1"
    else:
        status = "inconclusive"
    return {
        **primary,
        "folds": folds,
        "verdict": {"status": status, "held_out_ece": ece, "beats_calibrated_rrf": gain},
    }


# --- Report -----------------------------------------------------------------------------------


def label_report(labels: dict[tuple[str, str], Row]) -> dict[str, Any]:
    """Route and kind counts, so unlabelled and prefix rows stay visible."""
    report: dict[str, Counter[str]] = defaultdict(Counter)
    for meta in labels.values():
        report["route"][meta["label_route"]] += 1
        report["kind"][meta["kind"]] += 1
        report["query_kind"][meta["query_kind"]] += 1
        if meta["kind"] != "entity":
            report[f"{meta['kind']}_lenient"][str(meta["label_lenient"])] += 1
    return {name: dict(counts) for name, counts in report.items()}


def question_meta(
    rows: Iterable[Row], gold: dict[str, Gold]
) -> tuple[dict[str, str], dict[str, str]]:
    """Kind and type per batch id. Taken from the batch, so an empty recall still counts."""
    kinds: dict[str, str] = {}
    types: dict[str, str] = {}
    for row in rows:
        question = gold.get(question_of(row["id"]))
        kinds[row["id"]] = query_kind(row["id"], question)
        types[row["id"]] = question.question_type if question else None
    return kinds, types


def evaluate(
    rows: list[Row],
    labels: dict[tuple[str, str], Row],
    gold: dict[str, Gold],
    queries: dict[str, str],
    summary_cosines: dict[tuple[str, str], float | None],
    arm_tuning: dict[str, dict[str, Any]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    kinds, types = question_meta(rows, gold)
    table = candidate_table(rows, labels, queries, summary_cosines)
    cells = {cell: score_cell(table, cell, n_boot, seed) for cell in CELLS}
    h2 = {
        "cells": cells,
        "verdict": h2_verdict(cells[PRIMARY_CELL]),
        "per_type": per_type(table),
        "overlap_strata": overlap_strata(table),
        "coverage_ceiling": coverage_ceiling(table, kinds, types),
    }
    report = {
        "preregistration": {"seed": seed, "n_boot": n_boot, "primary_cell": PRIMARY_CELL,
                            "primary_k": PRIMARY_K},
        "labels": label_report(labels),
        "h1": score_h1(rows, kinds, n_boot, seed),
        "h2": h2,
        "h3": score_h3(rows, labels, kinds, types, arm_tuning),
        "h4": score_h4(table),
    }
    if h2["verdict"]["status"] == "killed":
        report["h3"]["exploratory_only"] = report["h4"]["exploratory_only"] = True
    return report


def _ci(cell: dict[str, float] | None) -> str:
    if not cell:
        return "n/a"
    return f"{cell['estimate']:.3f} [{cell['low']:.3f}, {cell['high']:.3f}]"


def render_markdown(report: dict[str, Any]) -> str:
    h1, h2, h3, h4 = (report[h] for h in ("h1", "h2", "h3", "h4"))
    lines = [
        "# Recall signal evaluation", "",
        f"Seed {report['preregistration']['seed']}, {report['preregistration']['n_boot']} "
        f"resamples over questions. Primary cell `{PRIMARY_CELL}`, primary k {PRIMARY_K}.", "",
        "| Hypothesis | Status |", "|---|---|",
        f"| H1 labels mislead on nulls | {h1['verdict']['status']} |",
        f"| H2 cosine beats RRF | {h2['verdict']['status']} |",
        f"| H3 cosine floor | {h3['verdict']['status']} |",
        f"| H4 calibration | {h4['verdict']['status']} |", "",
    ]
    if report["h3"].get("exploratory_only"):
        lines += ["H2 was killed, so H3 and H4 are exploratory only.", ""]

    lines += ["## H1", "", "| Queries | n | Share with `high` |", "|---|---|---|"]
    for name in ("real", "swapped", "abstention", "all_nulls"):
        share = h1[name].get("high_share")
        lines.append(f"| {name} | {h1[name]['queries']} | "
                     f"{'n/a' if share is None else f'{share:.3f}'} |")
    lines += ["", f"Null to real ratio: {_ci(h1['verdict'].get('null_to_real_ratio'))}", ""]

    lines += ["## H2", ""]
    for cell, data in h2["cells"].items():
        lines += [f"### {cell}: {data['questions']} questions, {data['candidates']} candidates", "",
                  "| Signal | Pooled AUC | Within AUC | Pooled PR-AUC | Within PR-AUC | "
                  "Cosine minus signal (within AUC) |",
                  "|---|---|---|---|---|---|"]
        for name, metrics in data["signals"].items():
            versus = data["cosine_minus"].get(name, {}).get("within_auc")
            lines.append(
                f"| {name} | {_ci(metrics['pooled_auc'])} | {_ci(metrics['within_auc'])} | "
                f"{_ci(metrics['pooled_pr_auc'])} | {_ci(metrics['within_pr_auc'])} | {_ci(versus)} |"
            )
        lines.append("")
    lines += ["### Per type", "", "| Type | Questions | Cosine | RRF |", "|---|---|---|---|"]
    for code, row in h2["per_type"].items():
        lines.append(f"| {code} | {row['questions']} | {row[COSINE]} | {row[RRF]} |")
    lines += ["", f"Coverage ceiling: `{json.dumps(h2['coverage_ceiling'])}`",
              f"Overlap strata: `{json.dumps(h2['overlap_strata'])}`", ""]

    lines += ["## H3", "", "| Arm | tau | All sources | k | Loss | Noise cut | Token cut | Flips |",
              "|---|---|---|---|---|---|---|---|"]
    for arm_id, arm in h3["arms"].items():
        for cut in arm["by_k"]:
            lines.append(
                f"| {arm_id} | {arm['tau']} | {arm['all_sources']} | {cut['k']} | "
                f"{cut['lost_questions']}/{cut['gold_questions']} | {cut['all']['noise_cut']} | "
                f"{cut['all']['token_cut']} | {len(cut['flips'])} |"
            )
    lines += ["", "Token cuts are estimates from result content, not `build_context` counts.", ""]

    lines += ["## H4", ""]
    for split, data in h4.get("splits", {}).items():
        lines += [f"### {split}: {data['candidates']} candidates", "",
                  "| Model | ECE | Brier |", "|---|---|---|"]
        for model, metrics in data["models"].items():
            lines.append(f"| {model} | {metrics['ece']:.4f} | {metrics['brier']:.4f} |")
        lines.append("")
    lines += ["## Labels", "", f"`{json.dumps(report['labels'])}`", ""]
    return "\n".join(lines)


def _jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite(value: Any) -> Any:
    """Strict JSON: NaN and infinities become null."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--summary-cosines", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows, failed = load_batch(args.batch)
    if failed:
        raise SystemExit(f"{failed} (id, arm) pairs failed; rerun the batch first")
    labels = {(m["id"], m["uuid"]): m for m in _jsonl(args.labels)}
    queries = {q["id"]: q["query"] for q in _jsonl(args.queries)}
    summaries = {}
    if args.summary_cosines:
        summaries = {(s["id"], s["uuid"]): s["summary_cosine"]
                     for s in _jsonl(args.summary_cosines)}
    arm_tuning = {
        arm["id"]: arm.get("tuning", {})
        for arm in json.loads(args.arms.read_text(encoding="utf-8"))
    }

    gold = load_gold(load_items(args.dataset))
    report = _finite(evaluate(rows, labels, gold, queries, summaries, arm_tuning))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    for name in ("h1", "h2", "h3", "h4"):
        print(name, report[name]["verdict"]["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
