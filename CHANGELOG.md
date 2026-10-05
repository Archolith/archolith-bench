## 2026-10-05 - Require an expected item count for canonical acceptance

- `lme.sh validate` refuses to run without `--expected N` (positive integer), and `validate_run.py --require-fresh-clean` refuses without `--expected-items`. Before this, `manifest_cardinality` passed any non-empty manifest, so a short canonical run could be accepted.
- `validate(require_fresh_clean=True)` without an expected count reports `manifest_cardinality` as FAIL for library callers.

## 2026-10-04 - Validate requested LongMemEval processing lanes

- Explicitly configure, validate, forward, and freeze Graphiti canonical-self binding mode across build resumes.
- Pass opt-in counter and event selections through the build server and Phase-3 request.
- Refuse disabled, failed, unfinished, or unmeasured requested lanes before manifesting and at final acceptance.
- Record lane receipts and separate total, scalar, and event call counts; refuse incompatible resumes.
- Add offline API-response and shell-wiring regressions, and document the companion Menhir response fields.
