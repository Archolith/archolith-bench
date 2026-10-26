## 2026-10-25 - LME answer prompt states computed numbers plainly, not as a lower bound

- The memory answer prompt asks for the number the retrieved figures give (a count, total, sum or difference) as the answer, not "at least N", "over N" or "more than N", with approximate or missing figures noted after it. "About N" is unaffected. The answer `prompt_fingerprint` changes for every arm.
- Why: in the src-sm10 o500 arm, 9 answers led with "at least" or "over". Only 2 passed under both the old judge and the official LongMemEval judge (#41), and 3 got different grades from the two, even where the number matched the gold ($185, $720, 10, 3). The 34 answers that led with "about", "around" or "approximately" passed both judges 32 times.
- `tests/test_longmemeval_answer_prompt.py`: `test_computed_numbers_are_stated_plainly_not_as_a_lower_bound`.

## 2026-10-25 - Render Menhir source memories by default

- `--source-memories` is now on by default; `--no-source-memories` restores the old context. Every harness run, including `run_stratified_slice`, gets the source block, and the checkpoint tag gains `sm`: scores from here on are not like-for-like with earlier baselines.
- o500 recall-only A/B (#39, Menhir 37e309a3, #37 prompt, limit 20, #38 said time, one run per arm, paired): 0.738 without, 0.804 with Menhir's default 10 source memories (+6.6 points, 95% CI +3.6 to +9.6, 46 wins, 13 losses), 0.808 with 20 (within noise of 10). Multi-session gained most (+16); 31 items wrong in at least 6 of the 7 earlier runs now pass. Answer input tokens +24%.
- `tests/test_longmemeval_answer_prompt.py`: `test_harness_source_memories_default_on_with_opt_out`.

## 2026-10-25 - Optionally render Menhir source memories (`--source-memories`)

- `--source-memories` renders the recall response's `source_memories` (raw episodes, oldest first) as one block after the memories and counted verdicts, outside `--recall-limit`, in Menhir's own `build_context` format without the pool tag. Off by default; off leaves the context and `prompt_fingerprint` unchanged, on adds `sm` to the checkpoint tag. A blocking event advisory still suppresses it; agentic recall keeps one block.
- Why: Menhir sends these by default and its agent context shows them, but the bench dropped them. On o500 stable multi-session failures the rendered context held 68% of the evidence numbers (92% with source memories); temporal 57% (93%).
- `tests/test_menhir_client_source_memories.py`.

## 2026-10-25 - Menhir recall facts render their statement time

- `archolith_bench/harness/menhir_client.py`: a fact line becomes `- <happened> | said <time> | <fact>` when Menhir sends `said_at` (Menhir `MENHIR_FRONTIER_FACT_SAID_TIME`, off by default); unchanged otherwise, so the answer prompt is the same for arms without it.
- `tests/test_menhir_client_contract.py`: `test_recall_renders_said_time_only_when_menhir_sends_it`.

## 2026-10-25 - LME answer prompt picks the best-supported candidate; recall limit 20

- The memory answer prompt no longer stops at "I don't know" when the recall holds candidate answers: it gives the best-supported one, names the other, and abstains only when no memory addresses the question. With unknown source times it prefers a value stated as a change ("now", "switched to"). Conflicting authoritative records lose their authority instead of forcing "I don't know". The answer `prompt_fingerprint` changes for every arm.
- `--recall-limit` defaults to 20 (was 10), as do `run_memory_ab` (`LME_RECALL_LIMIT`) and `run_stratified_slice`.
- o500 recall-only A/B (Menhir 953ca6e + #383, bench 06ab93b, one run per arm): 0.694 baseline, 0.686 with limit 20 alone, 0.710 with the prompt alone, 0.720 with both (+2.6 points, 95% CI +0.2 to +5.0; "don't know" answers 67 to 36; abstention items did not drop). Not measured together with the #36 writer prompt text.

## 2026-10-24 - Agent memory writer for assistant turns (`LME_SEGMENTATION=adaptive-writer`, #168 P3)

- New segmentation mode `adaptive-writer`: user turns go through `adaptive` unchanged; assistant turns are recorded as TurnEvidence only, and an LLM writer (`scripts/longmemeval/lib/memory_writer.py`, OpenAI Flex) turns each one into 0..20 standalone, dated, third-person items about what the assistant said. Each item is ingested as `role=agent`, `source=agent-writer`, grounded on its assistant turn's evidence, so it stays agent-tier and cannot supersede a user statement in Menhir.
- `LME_WRITER_GRANULARITY=turn|session` (one request per assistant turn, or per session with cited turns), `LME_WRITER_MODEL` (default `gpt-6-luna`). `LME_WRITER_MAX_USD` and `LME_WRITER_RATES` ("IN,OUT" USD per 1M) are required; ingest refuses to start without them or `OPENAI_API_KEY`.
- The writer runs as a prepass with a JSONL checkpoint (`writer-<manifest stem>-<granularity>.jsonl` beside the manifest, under the manifest's run lock); request ids carry model, prompt sha and a digest of the session text, so a resume replays the same items and a changed input is asked again. A failed or unparseable reply, or one whose items are all malformed or mis-cited, stops the build before any ingest and is retried on the next run, never read as "no items". Writer items enter the turn-plan digest; other modes' digests are unchanged.
- Recall renders Menhir's `provenance.said_by` as `(agent-recorded)`, `(user + agent)` or `(provenance unknown)`, and the answer prompt explains the labels. This applies to every arm, so the answer `prompt_fingerprint` changes for all of them: scores from here on are not like-for-like with earlier baselines. `archolith_bench/harness/{longmemeval,menhir_client}.py` and `core/openai_flex.py` are now in `bench-surface.yaml`.
- The graph attempt record and run provenance add `writer_model`, `writer_granularity` and `writer_prompt_sha` in writer mode. The ingest state log records writer mode as `adaptive-writer:<granularity>:<model>:<prompt sha>`, so continuing a build under another writer configuration is refused; in writer mode the startup refusals that do not need the plan run before the writer spends.

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
