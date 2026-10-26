## 2026-10-26 - LME answer prompt gives tailored suggestions for advice questions

- `_MEMORY_SYSTEM_PROMPT`: when a question asks for suggestions, tips, advice or recommendations, the answer model gives concrete suggestions tailored to what the memory shows about the user, and may use general knowledge for it, instead of saying the memory holds no suggestions. The prompt fingerprint changes, so existing checkpoints are not reused.
- Measured by replaying the answer step 3 times over the o500 src-spread-par recorded contexts (same recall, same judge): 0.803 to 0.813 mean; preference questions 0.722 to 0.844, other types within noise.
- `tests/test_longmemeval_answer_prompt.py`.

## 2026-10-26 - Opt-in per-recall trace capture (`--recall-trace`)

- `--recall-trace` asks Menhir for the retrieval trace and stores a compact, content-free copy per item in the checkpoint `recall_trace`: result ranks, per-candidate source/similarity/floor/score/ranks, source-memory ids and cosines. Off by default; it changes no answer and not the checkpoint name.
- `tests/test_recall_trace.py`.

## 2026-10-25 - Accept verbatim-skipped user episodes in the pairing check

- `_integrity_counts` (`ready_episodes_time_unpaired`): when Menhir's evidence projection already extracted the same text, it marks the client episode READY with substage `verbatim_skipped` and `extraction_skipped_for` set to that projection. Such an episode has no Graphiti episode of its own, so every one counted as unpaired: 512 of 515 READY user episodes in a stopped o500 build. It now passes when the named projection is READY, in the same namespace, with the same `reference_time`. The projection is itself checked for its own Graphiti pairing. On the stopped build the count drops from 512 to 18; those 18 are skips whose projection was still ENRICHING when the build stopped.
- `tests/test_lme_verbatim_skip_pairing.py`.

## 2026-10-25 - Keep user questions as store-only source memories; gate builds on complete embeddings

- CONTEXT_ONLY user turns (questions, requests) still state facts about the user, but were captured as evidence only, so recall never saw them (#43). With `LME_STORE_ONLY_CONTEXT=1` (config.sh default; the standalone `ingest.py` default stays 0) each becomes an episode sent with `store_only`: Menhir records and embeds it as a source memory but never extracts or projects it (Menhir #388). Assistant CONTEXT_ONLY turns and writer-mode assistant turns stay evidence only. `HttpMenhirClient.ingest` gains `store_only`, sent only when true. The plan digest changes only for plans that contain a store-only turn.
- `_integrity_counts`: a store-only admitter no longer counts as `admitted_turns_unprojected`, and store-only queue nodes are not expected to pair with a Graphiti episode (`ready_episodes_time_unpaired`). They still count as admitted user turns.
- `build_graph.sh` runs Menhir's `backfill_episode_embeddings.py --require-complete` after the ingest (`LME_EMBED_BACKFILL=1` default; Menhir #389). Benchmark mode runs no scheduler, so a READY episode whose embedding the worker gave up on stayed invisible to the source-memory lane; the build now embeds the gaps and stops if any remain. Both settings are recorded in the build provenance, and the build refuses to start, before any paid call, when the serving Menhir tree lacks either feature. The backfill script stays out of `bench-surface.yaml` until CI's pinned Menhir (f145640f) is bumped past it.
- `tests/test_lme_store_only_questions.py`.

## 2026-10-25 - Grade LongMemEval with its official judge prompts; `_abs` ids are abstentions

- `--scorer llm-judge` now sends LongMemEval items LongMemEval's own judge prompts, verbatim from `get_anscheck_prompt()` in `src/evaluation/evaluate_qa.py` (commit d6dc8b50a2d9): one per question type (temporal forgives off-by-one counts, knowledge-update accepts the old value next to the updated one, preference grades against the rubric), sent as a single user message. Items without a LongMemEval question type keep the generic prompts. The judge model is unchanged.
- Unanswerable items are found by their `_abs` question id, as LongMemEval does. Before, `_is_abstention()` only matched `question_type == "abstention"` or an empty gold, so all 30 `_abs` items in o500 were graded as "does the answer match this explanation?". The containment scorer gets the same fix.
- Scores from here on are not like-for-like with earlier baselines. Judge runs get their own checkpoint tag (`oj`), so a resume never mixes grades. `archolith_bench/harness/scoring.py` is added to the `score` stage of `bench-surface.yaml`.
- `tests/test_lme_official_judge.py`.

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
