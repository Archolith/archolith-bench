"""Compare sprint v2 runs side by side, saying first what differs between their builds.

A difference across runs is only attributable to what was meant to change. Each run's report
records the Menhir it served, the prompt set, rubric, agent and model; this module names every
identity key that differs so a cross-version comparison is never read as a like-for-like one.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

#: Identity keys from ``provenance`` (plus the top-level ``rubric_version``). A mismatch in any of
#: these is a confound; the Menhir keys are the ones a release bump changes.
MENHIR_KEYS = ("menhir_release", "menhir_commit", "loopback_profile", "server_chat_model",
               "server_service_tier")
OTHER_KEYS = ("rubric_version", "prompt_set", "fixture_sha256", "agent", "agent_version", "harness_flex",
              "model", "client_service_tier", "cache_shaping", "timeout_s")


def load_report(run_dir: str | Path) -> dict:
    path = Path(run_dir)
    return json.loads((path / "report.json" if path.is_dir() else path).read_text(encoding="utf-8"))


def identity(report: dict) -> dict:
    prov = report.get("provenance") or {}
    ident = {k: prov.get(k) for k in (*MENHIR_KEYS, *OTHER_KEYS) if k != "rubric_version"}
    ident["rubric_version"] = report.get("rubric_version")
    # Runs older than menhir_release recorded only the commit; say so rather than call it a match.
    if "menhir_release" not in prov:
        ident["menhir_release"] = "(not recorded)"
    return ident


def mismatches(reports: list[dict]) -> dict[str, list]:
    """Identity keys whose values differ across ``reports`` -> the per-run values, in order."""
    idents = [identity(r) for r in reports]
    keys = [*MENHIR_KEYS, *OTHER_KEYS]
    return {k: [i.get(k) for i in idents] for k in keys if len({json.dumps(i.get(k)) for i in idents}) > 1}


def arm_summary(report: dict) -> dict[str, dict]:
    out = {}
    for arm, rows in report["analysis"]["per_chain"].items():
        if not rows:
            continue
        out[arm] = {
            "chains": len(rows),
            "accuracy": round(mean(r["accuracy"] for r in rows), 4),
            "uncached_tokens": round(mean(r["primary_efficiency_tokens"] for r in rows)),
            "cost_usd": round(mean(r["client_cost_usd"] + r["server_cost_usd"] for r in rows), 4),
        }
    return out


def render(run_dirs: list[str | Path]) -> tuple[str, dict[str, list]]:
    reports = [load_report(d) for d in run_dirs]
    diff = mismatches(reports)
    lines = []
    labels = [f"{r.get('fixture', '?')}/{r.get('run_id', '?')}" for r in reports]
    menhir_diff = [k for k in diff if k in MENHIR_KEYS]
    if menhir_diff:
        lines.append("WARNING: these runs served different Menhir builds; differences between them are "
                     "not attributable to the prompt or agent alone.")
    for key, values in diff.items():
        lines.append(f"  differs: {key}: " + " | ".join(f"{lab}={val}" for lab, val in zip(labels, values)))
    if not diff:
        lines.append("Builds and settings match on every identity key.")
    lines.append("")
    summaries = [arm_summary(r) for r in reports]
    arms = sorted({a for s in summaries for a in s})
    lines.append(f"{'arm':22s} {'run':40s} {'chains':>6s} {'acc':>7s} {'uncached':>10s} {'cost$':>8s}")
    for arm in arms:
        for lab, summ in zip(labels, summaries):
            if arm in summ:
                s = summ[arm]
                lines.append(f"{arm:22s} {lab:40s} {s['chains']:6d} {s['accuracy']:7.4f} "
                             f"{s['uncached_tokens']:10d} {s['cost_usd']:8.4f}")
    return "\n".join(lines), diff
