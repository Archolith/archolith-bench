"""Report persistence for archolith-bench.

Handles file I/O: saving benchmark results, writing transcripts,
and generating BENCHMARKS.md aggregations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def save_results(data: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_str = f"_{data['budget']}b" if data['budget'] else ""
    arm_label = data.get("arm", "proxy")
    filename = f"benchmark_{data['scenario']}_{arm_label}{budget_str}.json"
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")

    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for label in ("direct", "arm"):
        transcript_path = transcripts_dir / f"{data['scenario']}_{arm_label}{budget_str}_{label}.md"
        with open(transcript_path, "w", encoding="utf-8") as tf:
            tf.write(f"# {data['scenario']} ({arm_label}) -- {label.upper()} transcript\n")
            tf.write(f"Model: {data['model']} | Budget: {data['budget']} | Turns: {data['turns_run']}\n\n")
            for t in data["turns"]:
                tf.write(f"---\n## Turn {t['turn']}\n\n")
                tf.write(f"**User:** {t.get('user_msg', t.get('user_msg_preview', ''))}\n\n")
                resp = t[label].get("response", t[label].get("response_preview", ""))
                tf.write(f"**{label.title()}** ({t[label]['output_tokens']} tokens, "
                         f"{t[label]['input_tokens']} in, {t[label]['latency_ms']:.0f}ms):\n\n")
                tf.write(f"{resp}\n\n")
    print(f"Transcripts saved to {transcripts_dir}/")
    return path




def write_benchmarks_md(results_dir: Path, out_path: Path) -> None:
    """Generate a BENCHMARKS.md from results/ JSON files.

    Reads filter_results.json, audit_comparison.json, and any
    benchmark_*.json files. Writes a markdown report with per-product
    sections and a cross-product roll-up.
    """
    results_dir = Path(results_dir)
    out_path = Path(out_path)
    benchmark_results = {
        path: _load_json(path)
        for path in sorted(results_dir.glob("benchmark_*.json"))
    }

    lines: list[str] = []
    lines.append("# archolith-bench — Benchmark Results\n")
    lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M UTC', __import__('time').gmtime())}\n")
    lines.append(
        "\n> **Launch-claim status:** this generated report may include historical, fixture, or local "
        "methodology runs. It is not a source of headline numbers by itself. Public claims must come "
        "from `HEADLINE-NUMBERS.md`, which currently has no active values until refreshed launch "
        "evidence exists.\n\n"
    )

    # ---- Filter section ----
    filter_path = results_dir / "filter_results.json"
    if filter_path.exists():
        with open(filter_path, encoding="utf-8") as f:
            filter_data = json.load(f)
        lines.append("## Filter Suite (archolith-filter)\n")
        lines.append("Token-savings compression ratio measured on real tool-output corpora.\n")
        lines.append("| Category | Samples | Raw Tokens | Filtered | Savings |\n")
        lines.append("|----------|---------|------------|----------|--------|\n")
        for cat in filter_data.get("per_category", []):
            lines.append(
                f"| {cat['category']} | {cat['samples']} | {cat['total_raw_tokens']:,} "
                f"| {cat['total_filtered_tokens']:,} | {cat['savings_ratio']:.1%} |\n"
            )
        lines.append(
            f"| **Total** | {filter_data['total_samples']} | {filter_data['total_raw_tokens']:,} "
            f"| {filter_data['total_filtered_tokens']:,} | **{filter_data['aggregate_savings_ratio']:.1%}** |\n"
        )
        lines.append("\n")
    else:
        lines.append("## Filter Suite (archolith-filter)\n")
        lines.append("*No filter results found. Run `archolith-bench filter --corpora corpora/` to generate.*\n\n")

    # ---- Proxy section ----
    # Glob ALL benchmark arm files (incl. *_direct.json baselines) -- the cost
    # verdict needs the passthrough/direct arm, not just *proxy* files.
    if benchmark_results:
        lines.append("## Proxy Suite\n")
        lines.append("Multi-turn token savings, effective cost, and continuity metrics across proxy experiment arms.\n")
        lines.append(
            "\n> **Stale evidence caveat:** existing proxy result files include dated, single-scenario or "
            "single-model runs. Do not use these as refreshed launch headline numbers until P4 evidence "
            "refresh lands after the remediation sessions.\n\n"
        )

        # Check if any results have cost data
        has_cost = any(
            "total_effective_cost_usd" in d.get("summary", {})
            for d in benchmark_results.values()
        )

        if has_cost:
            lines.append(
                "| Scenario | Arm | Budget | Direct In | Arm In | Upstream Input Reduction | "
                "Internal Curation Savings | Eff Cost | Cache | Recall Pres. |\n"
            )
            lines.append(
                "|----------|-----|--------|-----------|--------|--------------------------|"
                "---------------------------|----------|-------|-------------|\n"
            )
        else:
            lines.append(
                "| Scenario | Arm | Budget | Direct In | Arm In | Upstream Input Reduction | "
                "Internal Curation Savings | Recall Pres. |\n"
            )
            lines.append(
                "|----------|-----|--------|-----------|--------|--------------------------|"
                "---------------------------|-------------|\n"
            )

        cost_passthrough: dict[str, dict] = {}  # scenario -> {"cost", "cache_ok"}

        for d in benchmark_results.values():
            s = d.get("summary", {})
            q = d.get("quality", {})
            recall = f"{q.get('recall_preservation', 0):.0%}" if q else "N/A"
            upstream_reduction = s.get("upstream_input_reduction_ratio")
            if upstream_reduction is None:
                direct_in = s.get("total_direct_input_tokens", 0)
                arm_in = s.get("total_proxy_input_tokens", 0)
                upstream_reduction = ((direct_in - arm_in) / direct_in) if direct_in else 0.0
            line = (
                f"| {d.get('scenario', '?')} | {d.get('arm', '?')} | "
                f"{d.get('budget') or 'default'} | {s.get('total_direct_input_tokens', 0):,} | "
                f"{s.get('total_proxy_input_tokens', 0):,} | {upstream_reduction:.1%} | "
                f"{s.get('overall_savings_ratio', 0):.1%} |"
            )
            if has_cost and "total_effective_cost_usd" in s:
                cost_str = f" ${s['total_effective_cost_usd']:.4f}"
                cache_str = "yes" if s.get("cache_data_available") else "no"
                line += f" {cost_str} | {cache_str} |"
            if has_cost and not s.get("cache_data_available"):
                line += " ⚠ no cache data |"
            line += f" {recall} |\n"
            lines.append(line)

            # Track passthrough cost + cache availability for verdict
            arm = d.get("arm", "")
            scenario = d.get("scenario", "")
            if has_cost and arm in ("direct",) and "total_effective_cost_usd" in s:
                cost_passthrough[scenario] = {
                    "cost": s["total_effective_cost_usd"],
                    "cache_ok": s.get("cache_data_available", False),
                }

        if has_cost and cost_passthrough:
            lines.append("\n### Cost Verdict\n\n")
            for d in benchmark_results.values():
                s = d.get("summary", {})
                arm = d.get("arm", "")
                scenario = d.get("scenario", "")
                if arm in ("direct",) or "total_effective_cost_usd" not in s:
                    continue
                proxy_cost = s["total_effective_cost_usd"]
                pt = cost_passthrough.get(scenario)
                if pt is not None:
                    passthrough_cost = pt["cost"]
                    delta = proxy_cost - passthrough_cost
                    # A valid comparison needs cache data on BOTH sides; if the
                    # passthrough cost is a full-rate estimate, stay INCONCLUSIVE.
                    cache_ok = s.get("cache_data_available", False) and pt["cache_ok"]
                    if not cache_ok:
                        verdict = "INCONCLUSIVE (no cache data)"
                    elif delta < 0:
                        verdict = f"**PROXY CHEAPER** by ${abs(delta):.4f}"
                    else:
                        verdict = f"**PROXY MORE EXPENSIVE** by ${delta:.4f}"
                    lines.append(
                        f"- **{scenario}** [{arm}]: ${proxy_cost:.4f} vs passthrough "
                        f"${passthrough_cost:.4f} — {verdict}\n"
                    )
        lines.append("\n")
    else:
        lines.append("## Proxy Suite\n")
        lines.append("*Pending live-proxy run. Run `archolith-bench proxy --all --arms direct,proxy_plus_filter` to generate.*\n\n")

    # ---- Audit section ----
    audit_path = results_dir / "audit_comparison.json"
    if audit_path.exists():
        with open(audit_path, encoding="utf-8") as f:
            audit_data = json.load(f)
        lines.append("## Audit Suite (Archolith Skree)\n")
        before_p = str(audit_data.get("before_path", ""))
        after_p = str(audit_data.get("after_path", ""))
        is_sample = "fixture" in before_p.lower() or "fixture" in after_p.lower()
        if is_sample:
            lines.append(
                "> **Note:** sample/fixture data, not a live audit run. The numbers "
                "below reflect the bundled `fixtures/` inputs and demonstrate the "
                "report format only. Run `archolith-bench audit` against real "
                "before/after session logs to produce measured results.\n\n"
            )
        lines.append("MCP token-waste reduction before vs after.\n")
        lines.append(f"Source: `{before_p}` -> `{after_p}`\n\n")
        lines.append("| Server | Before | After | Change | Pct | Status |\n")
        lines.append("|--------|--------|-------|--------|-----|--------|\n")
        for s in audit_data.get("per_server", []):
            lines.append(
                f"| {s['server']} | {s['before_tokens']:,} | {s['after_tokens']:,} "
                f"| {s['token_change']:+,} | {s['token_change_pct']:+.1f}% | {s['status']} |\n"
            )
        # Total row uses the same change convention as the per-server rows
        # (after - before; negative = reduction) for sign consistency.
        agg_change = audit_data["after_total_tokens"] - audit_data["before_total_tokens"]
        agg_change_pct = -audit_data["token_reduction_pct"]
        lines.append(
            f"| **Total** | {audit_data['before_total_tokens']:,} | {audit_data['after_total_tokens']:,} "
            f"| {agg_change:+,} | {agg_change_pct:+.1f}% | - |\n"
        )
        lines.append(
            f"\n**Token reduction:** {audit_data['token_reduction']:,} ({audit_data['token_reduction_pct']:.1f}%). "
            f"**Waste reduction:** {audit_data['waste_reduction']:,} ({audit_data['waste_reduction_pct']:.1f}%).\n\n"
        )
    else:
        lines.append("## Audit Suite (Archolith Skree)\n")
        lines.append("*Pending before/after audit logs. Run `archolith-bench audit --before <log> --after <log>` to generate.*\n\n")

    # ---- Industry benchmark coverage ----
    industry_path = results_dir / "industry_benchmarks.json"
    if industry_path.exists():
        with open(industry_path, encoding="utf-8") as f:
            industry_data = json.load(f)
        lines.append("## Industry-Trusted Benchmark Coverage\n")
        lines.append(
            "Coverage matrix for external benchmark families that are relevant enough "
            "to anchor launch claims. Candidate benchmarks are not completed results.\n\n"
        )
        lines.append("| Product | Suite | Benchmark | Status | Evidence |\n")
        lines.append("|---------|-------|-----------|--------|----------|\n")
        for entry in industry_data.get("benchmarks", []):
            lines.append(
                f"| {entry['product']} | {entry['suite']} | {entry['name']} | "
                f"{entry['status']} | `{entry['evidence_path']}` |\n"
            )
        lines.append("\n")
    else:
        lines.append("## Industry-Trusted Benchmark Coverage\n")
        lines.append(
            "*Pending matrix. Run `archolith-bench industry --launch-only` to generate "
            "the benchmark coverage registry before launch.*\n\n"
        )

    # ---- Stack section ----
    lines.append("## Stack Suite (Four-Way Comparison)\n")
    lines.append(
        "*Experimental and pending refreshed live-proxy run. Run `archolith-bench "
        "stack --all` to generate; do not use stack results as launch headlines "
        "until tracked evidence is added under `benchmarks/`.*\n"
    )

    # ---- Evidence status ----
    lines.extend(_evidence_status_section(out_path.parent))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _group_by_artifact(issues: list) -> dict[str, list[str]]:
    """Collapse issues to one line per artifact.

    A file missing every schema field otherwise emits a bullet per field, which
    buries the artifacts that have a single real problem.
    """
    grouped: dict[str, list[str]] = {}
    missing_fields: dict[str, list[str]] = {}

    for issue in issues:
        name = Path(issue.path).name
        if issue.code == "missing_field":
            # Take the quoted name. Positional parsing on "field " broke for
            # Markdown artifacts, whose message reads "...missing required key
            # 'x'" -- every one of them rendered as the literal
            # "archolith-evidence".
            match = re.search(r"['\"]([^'\"]+)['\"]", issue.message)
            field_name = match.group(1) if match else issue.message
            missing_fields.setdefault(name, []).append(field_name)
        else:
            grouped.setdefault(name, []).append(issue.message)

    for name, fields in missing_fields.items():
        grouped.setdefault(name, []).insert(
            0, f"missing {len(fields)} required field(s): {', '.join(fields)}"
        )
    return dict(sorted(grouped.items()))


def _evidence_status_section(repo_root: Path) -> list[str]:
    """Render the evidence policy's findings.

    Deliberately renders rather than recomputes: staleness and eligibility are
    decided by ``core.evidence_policy`` so this report and
    ``scripts/check_evidence_policy.py`` can never disagree about which
    artifacts are quotable.
    """
    from archolith_bench.core.evidence_policy import (
        is_markdown_evidence,
        validate_policy,
    )

    headline = repo_root / "HEADLINE-NUMBERS.md"
    bench_dir = repo_root / "benchmarks"
    lines = ["\n## Evidence Status\n"]

    if not headline.exists() or not bench_dir.is_dir():
        lines.append(
            "*Evidence policy not evaluated: `HEADLINE-NUMBERS.md` or `benchmarks/` "
            "not found relative to this report.*\n"
        )
        return lines

    paths = [
        p for p in sorted(bench_dir.glob("*"))
        if p.suffix.lower() == ".json"
        or (p.suffix.lower() == ".md" and is_markdown_evidence(p))
    ]
    result = validate_policy(headline, paths)
    summary = result.summary

    verdict = "PASS" if result.ok else "FAIL"
    lines.append(
        f"\nEvidence policy: **{verdict}** - "
        f"{summary.get('evidence_files_checked', 0)} artifact(s) checked, "
        f"{summary.get('public_copy_allowed', 0)} cleared for public copy, "
        f"{summary.get('headline_active_claims', 0)} active headline claim(s).\n"
    )

    if result.errors:
        lines.append("\n**Errors - these artifacts fail the evidence policy:**\n\n")
        for name, msgs in _group_by_artifact(result.errors).items():
            lines.append(f"- `{name}`: {'; '.join(msgs)}\n")

    stale = [w for w in result.warnings if w.code in
             ("incomplete_provenance", "untracked_raw_source", "missing_field")]
    if stale:
        lines.append(
            "\n**Stale or incomplete evidence** - valid, but not promotable to a "
            "headline claim as written:\n\n"
        )
        for name, msgs in _group_by_artifact(stale).items():
            lines.append(f"- `{name}`: {'; '.join(msgs)}\n")

    if not result.errors and not stale:
        lines.append("\nAll tracked evidence carries complete provenance.\n")

    lines.append(
        "\n*Regenerate the artifact index with "
        "`python scripts/check_evidence_policy.py --evidence-dir benchmarks/ "
        "--manifest benchmarks/evidence-manifest.md`.*\n"
    )
    return lines


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
