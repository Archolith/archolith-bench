"""Rerun a saved run from a later stage, and compare two runs. Neither touches Menhir."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from .config import run_id
from .pipeline import accuracy_summary
from .rundir import RunDir

RERUN_STAGES = ("answer", "judge")


def load_resolved(run_path: Path) -> dict:
    return yaml.safe_load((Path(run_path) / "config.resolved.yaml").read_text(encoding="utf-8"))


def derive_run(
    src: Path,
    runs_root: Path,
    from_stage: str,
    *,
    prompt_files: dict[str, str],
    judge: dict,
    model: str | None = None,
    answer_prompt_fingerprint: str | None = None,
    tag: str | None = None,
) -> RunDir:
    """New run dir from ``src``: copies the stages before ``from_stage`` and records the lineage.

    From ``judge`` the answers are reused, so the answer prompt hash stays the source's; only the
    judge prompt hashes and judge settings are refreshed. From ``answer`` everything after search
    is redone with the current prompts and ``model``.
    """
    if from_stage not in RERUN_STAGES:
        raise ValueError(f"--from must be one of {RERUN_STAGES}, got {from_stage!r}")
    src = Path(src)
    copy = ["search"] + (["answers"] if from_stage == "judge" else [])
    for name in copy:
        if not (src / f"{name}.jsonl").exists():
            raise FileNotFoundError(f"{src / f'{name}.jsonl'} missing: cannot rerun from {from_stage}")
    resolved = load_resolved(src)
    old_prompts: dict[str, str] = dict(resolved.get("prompt_files") or {})
    if from_stage == "judge":
        new_prompts = {k: v for k, v in old_prompts.items() if not k.startswith("judge/")}
        new_prompts.update({k: v for k, v in prompt_files.items() if k.startswith("judge/")})
    else:
        new_prompts = dict(prompt_files)
        if model is not None:
            resolved["model"] = model
        if answer_prompt_fingerprint is not None:
            resolved["answer_prompt_fingerprint"] = answer_prompt_fingerprint
    resolved["prompt_files"] = new_prompts
    resolved["judge"] = judge
    resolved["derived_from"] = {"run_id": src.name, "from": from_stage}
    if tag:
        # Same settings rerun twice would otherwise resume the first rerun's dir.
        resolved["derived_from"]["tag"] = tag

    rd = RunDir(Path(runs_root), run_id(resolved))
    rd.write_config(resolved)
    for name in copy:
        target = rd.stage(name).path
        if not target.exists():
            shutil.copyfile(src / f"{name}.jsonl", target)
    return rd


def compare_runs(a: Path, b: Path) -> dict[str, Any]:
    """Accuracy of both runs on their common items, plus the items whose label flipped."""
    rows_a = {r["item_id"]: r for r in RunDir(Path(a).parent, Path(a).name).stage("judged").records()}
    rows_b = {r["item_id"]: r for r in RunDir(Path(b).parent, Path(b).name).stage("judged").records()}
    common = sorted(set(rows_a) & set(rows_b))
    return {
        "a": Path(a).name,
        "b": Path(b).name,
        "n_common": len(common),
        "only_in_a": sorted(set(rows_a) - set(rows_b)),
        "only_in_b": sorted(set(rows_b) - set(rows_a)),
        "summary_a": accuracy_summary([rows_a[i] for i in common]),
        "summary_b": accuracy_summary([rows_b[i] for i in common]),
        "a_correct_b_wrong": [i for i in common if rows_a[i]["correct"] and not rows_b[i]["correct"]],
        "b_correct_a_wrong": [i for i in common if rows_b[i]["correct"] and not rows_a[i]["correct"]],
    }


def format_compare(c: dict[str, Any]) -> str:
    sa, sb = c["summary_a"], c["summary_b"]
    lines = [
        f"A: {c['a']}",
        f"B: {c['b']}",
        f"common items: {c['n_common']} (only in A: {len(c['only_in_a'])}, only in B: {len(c['only_in_b'])})",
        f"accuracy: A {sa['accuracy']:.3f}  B {sb['accuracy']:.3f}",
    ]
    for t in sorted(set(sa["per_type"]) | set(sb["per_type"])):
        ra, rb = sa["per_type"].get(t), sb["per_type"].get(t)
        fa = f"{ra['accuracy']:.3f} (n={ra['n']})" if ra else "-"
        fb = f"{rb['accuracy']:.3f} (n={rb['n']})" if rb else "-"
        lines.append(f"  {t}: A {fa}  B {fb}")
    if "unofficial" in sa or "unofficial" in sb:
        ua, ub = sa.get("unofficial"), sb.get("unofficial")
        lines.append(
            "unofficial: A " + (f"{ua['accuracy']:.3f} (n={ua['n']})" if ua else "-")
            + "  B " + (f"{ub['accuracy']:.3f} (n={ub['n']})" if ub else "-")
        )
    lines.append(f"A right, B wrong ({len(c['a_correct_b_wrong'])}): {', '.join(c['a_correct_b_wrong'])}")
    lines.append(f"B right, A wrong ({len(c['b_correct_a_wrong'])}): {', '.join(c['b_correct_a_wrong'])}")
    return "\n".join(lines)
