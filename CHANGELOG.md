## 2026-10-22 - Resume a killed LME graph ingest where it stopped

- Each extractable segment is posted with a stable `idempotency_key` (`<turn_key>#s<segment>`), so a replay after a kill reuses the anchor Menhir already has (needs Menhir with #372). A 409 (same key, different request) stops the run without a retry.
- `ingest.py` keeps an append-only, fsynced state log beside the manifest (`ingest-state-<manifest>.jsonl`): each namespace's turn-plan digest, retries made, explicit resets and finished namespaces. A resume refuses a namespace whose plan changed, never retries an episode or projection twice, and never resets a namespace unless `--reset-namespace` / `--reset-file` names it.
- Namespaces run through a sliding pool; each one settles and writes its manifest row as soon as it finishes. Orphaned ENRICHING rows from a killed run are waited on until Menhir's lease frees them.
- Graph state is read in one batched query over the Neo4j HTTP API (`LME_NEO4J_HTTP_URL`), falling back to `docker exec`. A Menhir outage ends the run after `--outage-timeout` (300 s) instead of waiting out the drain timeout.
- An episode or projection that FAILED on the provider account (quota/auth) is not a content failure: it spends no retry and is never terminal. The run stops while Menhir still reports the condition and requeues the row once it clears (at most 3 times per run). `failed_requeued` counts retries that happened; the manifest adds `provider_requeued`.
- One ingest client per manifest (OS lock), one segmentation mode per build, and no duplicate ingest keys in a plan. A failed graph read during the projection sweep holds the namespace back instead of reading as "no failures". The final drain re-checks every manifested namespace on each run.
- `build_graph.sh` takes `LME_RESUME=1` (container, volume, provenance and manifest dir must exist and match) and refuses to start when something already serves the Menhir port. `validate_run.py` accepts an interrupted ingest phase followed by a completed one and reports `attempt_count`. `retry.py` only counts and retries episodes under `LME_NS_PREFIX`.

## 2026-10-17 - Keep Menhir verb-facet facts outside the limit in every memory mode

- The value-augmented arms and agentic recall re-capped the context to `recall_limit`, so verb-facet facts displaced ranked memories there. `recall()` now returns the facet snippets as `extras`, and those merges keep them in place without counting them. Agentic recall keeps at most 3 in total.
- The client keeps at most 3 facet facts per recall, the same cap Menhir uses.
- The Recall Lab rescore scripts render facet facts the same way as the client.
- Responses without facet facts give the same context as before.

## 2026-10-16 - Run the drill-down tool loop over the Responses API for OpenAI reasoning models

- gpt-6-luna rejects function tools on `/v1/chat/completions` unless reasoning is turned off, which made the first `--drill-down` run fail with a 400. `send_with_tools` now sends OpenAI reasoning models through `/v1/responses` (`send_responses_with_tools`), keeping their default reasoning so the tool arm stays comparable with the other arms. Other models keep the chat-completions loop.
- Same contract as the chat loop: shared retry policy, last round forbids tools. Any response whose status is not `completed`, or an empty reply, raises instead of being graded or having its tool calls run; the error carries the API's status, incomplete details and error. Reasoning items are passed back with tool results, as the API requires.
- The tool is sent with `strict: false`, matching chat-completions semantics for its optional arguments.
- `--resume` with `--drill-down` uses its own checkpoint file, so drill-down and plain answers for the same arm never mix.

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
