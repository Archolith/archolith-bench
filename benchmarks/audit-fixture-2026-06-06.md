# Audit Fixture Evidence - 2026-06-06

<!-- archolith-evidence
product: archolith-mcp-audit
command: archolith-bench audit
commit: unknown
run_date: 2026-06-06
source: results/audit_comparison.json
source_tracked: false
public_copy_allowed: false
note: Fixture-only report-format evidence. Not launch-headline evidence.
-->

This is fixture-only report-format evidence for `archolith-bench audit`.
It is not launch-headline evidence.

Source artifact was generated under local `results/` and is not tracked raw:

- `results/audit_comparison.json`

| Metric | Value |
|--------|-------|
| Before total tokens | 342,800 |
| After total tokens | 192,300 |
| Token reduction | 150,500 |
| Token reduction pct | 43.9% |
| Before total waste | 116,700 |
| After total waste | 33,200 |
| Waste reduction | 83,500 |
| Waste reduction pct | 71.55% |

Do not use the 71.5% fixture waste-reduction number in README, archolith.dev, or
launch copy until a live before/after audit run confirms it.
