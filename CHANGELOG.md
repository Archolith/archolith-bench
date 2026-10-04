## 2026-10-04 - Validate requested LongMemEval processing lanes

- Pass opt-in counter and event selections through the build server and Phase-3 request.
- Refuse disabled, failed, unfinished, or unmeasured requested lanes before manifesting and at final acceptance.
- Record lane receipts and separate total, scalar, and event call counts; refuse incompatible resumes.
- Add offline API-response and shell-wiring regressions, and document the companion Menhir response fields.
