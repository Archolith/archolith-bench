## 2026-10-24 - Agent memory writer for assistant turns (`LME_SEGMENTATION=adaptive-writer`, #168 P3)

- New segmentation mode `adaptive-writer`: user turns go through `adaptive` unchanged; assistant turns are recorded as TurnEvidence only, and an LLM writer (`scripts/longmemeval/lib/memory_writer.py`, OpenAI Flex) turns each one into 0..20 standalone, dated, third-person items about what the assistant said. Each item is ingested as `role=agent`, `source=agent-writer`, grounded on its assistant turn's evidence, so it stays agent-tier and cannot supersede a user statement in Menhir.
- `LME_WRITER_GRANULARITY=turn|session` (one request per assistant turn, or per session with cited turns), `LME_WRITER_MODEL` (default `gpt-6-luna`). `LME_WRITER_MAX_USD` and `LME_WRITER_RATES` ("IN,OUT" USD per 1M) are required; ingest refuses to start without them or `OPENAI_API_KEY`.
- The writer runs as a prepass with a JSONL checkpoint (`writer-<manifest stem>-<granularity>.jsonl` beside the manifest, under the manifest's run lock); request ids carry model, prompt sha and a digest of the session text, so a resume replays the same items and a changed input is asked again. A failed or unparseable reply, or one whose items are all malformed or mis-cited, stops the build before any ingest and is retried on the next run, never read as "no items". Writer items enter the turn-plan digest; other modes' digests are unchanged.
- Recall renders Menhir's `provenance.said_by` as `(agent-recorded)`, `(user + agent)` or `(provenance unknown)`, and the answer prompt explains the labels. This applies to every arm, so the answer `prompt_fingerprint` changes for all of them: scores from here on are not like-for-like with earlier baselines. `archolith_bench/harness/{longmemeval,menhir_client}.py` and `core/openai_flex.py` are now in `bench-surface.yaml`.
- The graph attempt record and run provenance add `writer_model`, `writer_granularity` and `writer_prompt_sha` in writer mode. The ingest state log records writer mode as `adaptive-writer:<granularity>:<model>:<prompt sha>`, so continuing a build under another writer configuration is refused; in writer mode the startup refusals that do not need the plan run before the writer spends.

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
