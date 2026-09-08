# Proxy Code Review Evidence - 2026-05-30

<!-- archolith-evidence
product: archolith-context
command: archolith-bench proxy
commit: unknown
run_date: 2026-05-30
source: results/benchmark_code_review_*.json
source_tracked: false
public_copy_allowed: false
note: Retired 2026-07-07: historical, single-scenario, DeepSeek upstream.
-->

Historical `archolith-bench proxy` run for the `code_review` scenario using
DeepSeek upstream.

Source artifacts were generated under local `results/` and are not tracked raw:

- `results/benchmark_code_review_proxy_only_15000b.json`
- `results/benchmark_code_review_proxy_only_4000b.json`
- `results/benchmark_code_review_proxy_plus_filter_15000b.json`
- `results/benchmark_code_review_proxy_plus_filter_4000b.json`

## Public Metric

For launch-facing copy, use actual upstream input reduction:

```text
(direct_input_tokens - arm_input_tokens) / direct_input_tokens
```

| Scenario | Arm | Budget | Direct In | Arm In | Upstream Input Reduction | Internal Curation Savings | Recall Pres. |
|----------|-----|--------|-----------|--------|--------------------------|---------------------------|--------------|
| code_review | proxy_only | 15000 | 108,516 | 80,520 | 25.8% | 58.6% | 57% |
| code_review | proxy_only | 4000 | 108,516 | 91,701 | 15.5% | 58.6% | 91% |
| code_review | proxy_plus_filter | 15000 | 108,376 | 111,021 | -2.4% | 58.7% | 60% |
| code_review | proxy_plus_filter | 4000 | 108,516 | 105,139 | 3.1% | 58.6% | 54% |

## Notes

- The historical `58.6%` value is an internal context-curation savings metric,
  not direct upstream input reduction.
- `proxy_plus_filter` at 15K used more upstream input than direct in this run,
  so it is not a positive launch claim.
- This run is useful evidence for methodology, but it should be refreshed before
  launch against the current proxy configuration.
