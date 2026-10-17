## 2026-10-16 - Run the drill-down tool loop over the Responses API for OpenAI reasoning models

- gpt-6-luna rejects function tools on `/v1/chat/completions` unless reasoning is turned off, which made the first `--drill-down` run fail with a 400. `send_with_tools` now sends OpenAI reasoning models through `/v1/responses` (`send_responses_with_tools`), keeping their default reasoning so the tool arm stays comparable with the other arms. Other models keep the chat-completions loop.
- Same contract as the chat loop: shared retry policy, last round forbids tools, an incomplete or empty reply raises instead of being graded. Reasoning items are passed back with tool results, as the API requires.

## 2026-10-16 - Let the answer model drill into collapsed Menhir counts

- `harness --recall-only --drill-down`: when Menhir collapses a scalar View to its total, the recall shows Menhir's own `built from N items; drill down: get_scalar_view_items(view_uuid=...)` line, and the answer model gets that tool plus a prompt note to use it as needed. Calls are served from `GET /api/scalar-views/{view_uuid}/items`, only for views shown in that item's recall and only in its namespace.
- Items whose recall shows no collapsed View are answered exactly as before. Without `--drill-down` nothing changes.
- New `send_chat_with_tools` shares `send_chat`'s retry policy (Flex waits, no retry on `insufficient_quota`); its last round forbids tools so the model has to answer. Tool calls are recorded in each result's `raw_usage`.

## 2026-10-05 - Check provenance, source time and cross-namespace links at acceptance

- After each item, ingest records read-only graph integrity counts in the manifest: submitted vs admitted user turns, admitted turns that were never projected, unfounded assertions, Graphiti episodes whose `valid_at` is not a submitted session date, READY queue episodes not paired to their Graphiti time, and cross-namespace links.
- `validate_run.py` adds `provenance_chain` and `source_time_integrity` checks, and `namespace_isolation` also fails on cross-namespace links. A missing or unreadable count fails as unknown.

## 2026-10-05 - Require an expected item count for canonical acceptance

- `lme.sh validate` refuses to run without `--expected N` (positive integer), and `validate_run.py --require-fresh-clean` refuses without `--expected-items`. Before this, `manifest_cardinality` passed any non-empty manifest, so a short canonical run could be accepted.
- `validate(require_fresh_clean=True)` without an expected count reports `manifest_cardinality` as FAIL for library callers.

## 2026-10-04 - Validate requested LongMemEval processing lanes

- Explicitly configure, validate, forward, and freeze Graphiti canonical-self binding mode across build resumes.
- Pass opt-in counter and event selections through the build server and Phase-3 request.
- Refuse disabled, failed, unfinished, or unmeasured requested lanes before manifesting and at final acceptance.
- Record lane receipts and separate total, scalar, and event call counts; refuse incompatible resumes.
- Add offline API-response and shell-wiring regressions, and document the companion Menhir response fields.
