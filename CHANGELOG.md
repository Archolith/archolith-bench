## 2026-10-12 - Send Menhir the structured speaker role

- `HttpMenhirClient.ingest` sends `role` (`user`, `assistant`, `tool`, `agent`, `system`; an empty role is `user`; anything else is omitted) on every episode. Menhir takes the speaker from it, or from the linked turn evidence for a grounded turn, and no longer reads a `user:` text prefix (Archolith/menhir#271, #272, #273). Older Menhir builds ignore the field.
- Episode text is unchanged: grounded user turns stay raw and equal to their evidence, and ungrounded turns keep `{role}: ` because Menhir does not show the role to the extraction model, so extraction input on existing arms stays comparable.
- `LME_LABEL_GROUNDED_USER_TURNS` is removed (reverts 6103f3d); the structured role replaces it.
- Run identity records `sends_turn_role`. A resume of a run ingested without it, including one built with the label workaround, is refused before ingest in canonical and noncanonical mode; start a fresh run.

## 2026-10-11 - Opt-in speaker label on grounded user turns

- `LME_LABEL_GROUNDED_USER_TURNS=1` writes `user: ` into each user turn's evidence and into its grounded episode, so the two stay byte-identical and Menhir's text-prefix speaker check (anchored-time resolver) sees the turn as the user's. Workaround until Menhir reads speaker role from structured data (Archolith/menhir#271). Default off; off sends exactly what it did before.
- The setting is part of the run identity: a resume that changes it, or turns it on for a run recorded without it, is refused before ingest. With it on, every user `:TurnEvidence` text carries the label, which Menhir lanes that read evidence text will also see.

## 2026-10-11 - Opt-in partial resume of unmanifested LongMemEval items

- `LME_RESUME_PARTIAL=1` (`--resume-partial`) keeps an unmanifested item's stored episodes when they are byte-for-byte the first k episodes the item submits, in order, and submits only the rest. The comparison runs inside Neo4j. A mismatch resets the namespace as before, and an unreadable state fails closed. Default off; the reset path is unchanged.
- With the flag on, the stale-settlement wait outlasts the stored episode leases plus one drain, instead of a flat 60s. A FAILED last episode is adopted with one retry. Resumed manifest rows carry `resumed`.
- Before `menhir serve` starts on a partial resume, `lib/liveness_gate.py` refuses if any non-ssh process holds a bolt connection to this Neo4j, or a `serve --port` or `ingest.py` for this port is alive. Unknown state refuses.

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
