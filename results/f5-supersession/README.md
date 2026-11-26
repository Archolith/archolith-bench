# F5 supersession runs: full3, full4-91, full5-91, full5b-91

F5 asks whether Menhir retires an old architecture decision when a later ADR replaces it, without
retiring decisions that nothing replaced. Each run ingests the same 121 ADRs from 5 open-source repos
(arm M-U, unlabelled text, `--streams 5`), consolidates SESSION -> PERSISTENT, then answers 48
questions from Menhir's own context (`--packer menhir --context-tokens 9000`, `reranker_only`), judged
by gpt-6-luna. Every chat call is gpt-6-luna on OpenAI Flex.

## Runs

| run | date (UTC) | Graphiti | Menhir |
|---|---|---|---|
| full3 | 2026-11-24 | fork 0.31.2.post3 | main before #984 |
| full4-91 | 2026-11-25 | Archolith/graphiti #33 + #34 before review fixes (exact heads not recorded) | #984 |
| full5-91 | 2026-11-26 | #34 at `5df9a04` (stacked on #33 `853179f`, base 0.32.1.post1 `3f900b9`) | #984 at `59b56503` |
| full5b-91 | 2026-11-26 | same as full5-91 | same as full5-91 |

full5-91 differs from full4-91 by the #34 review fixes and by F3: the batched edge-dedupe prompt now
carries all of `resolve_edge`'s rules and its eight examples (1,648-token system prompt). full5b-91 is a
second full5-91 build with the same code, for noise. Its ingest was stopped once with the `STOP` file
and resumed; resume skips finished ADRs.

## Results

| | full3 | full4-91 | full5-91 | full5b-91 |
|---|---|---|---|---|
| Overall correct | 43/48 | 42/48 | 41/48 | 43/48 |
| Current: correct / stale / wrong | 14 / 1 / 1 | 15 / 0 / 1 | 15 / 1 / 0 | 15 / 0 / 1 |
| Before: correct / wrong | 10 / 0 | 9 / 1 | 9 / 1 | 10 / 0 |
| Controls: correct / wrong | 19 / 3 | 18 / 4 | 17 / 5 | 18 / 4 |
| Old decisions retired by the new ADR | 11/16 | 14/16 | 13/16 | 12/16 |
| Controls wrongly retired | 5/22 | 3/22 | 2/22 | 1/22 |
| Graphiti chat calls | 3,261 | 966 | 994 | 1,014 |
| All chat calls (incl. identity_judgment) | 4,017 | 1,713 | 1,726 | 1,809 |
| Ingest + consolidation chat cost, cache-discounted | $0.578 | $0.433 | $0.457 | $0.470 |
| Eval cost (answer + judge) | | $0.021 | $0.022 | $0.022 |

Cost uses Flex prices of $0.05 per 1M uncached input, $0.005 cached, $0.25 output, from `llm_usage.csv`.

full5-91 vs full4-91: 3 of 48 answers flipped (`dito-P3-current` correct -> stale, `ris-C0004`
correct -> wrong, `ris-P8-current` wrong -> correct). `dito-P3` was not retired in either run, so
its flip comes from retrieval or answering, not dedupe. `ris-P8` lost its only retiring edge yet its
answer became correct. Edge counts per decision swing widely between runs (supa-P11: 43 vs 26), so
one run per configuration cannot separate differences of this size from noise. F3's direct cost is
about +$0.004 (279 batch dedupe calls vs 292, each about 340 tokens longer); most of the +$0.024 is
extraction (180 `extract_message` calls vs 145), which F3 does not change.

## Noise

Two kinds of repeat measure how much of that is noise:

- **Graph rebuild (full5b-91):** the same code as full5-91, ingested again. 4 answers flipped
  (`dito-P3-current` stale -> correct, `dito-P7-before` wrong -> correct, `ris-C0004` wrong -> correct,
  `ris-P8-current` correct -> wrong) and 2 pair verdicts changed (`dito-P1` not retired -> retired by
  another ADR, `dito-P4` retired -> not retired). That is more change than F3 showed (3 answers, 1
  pair), and all three of F3's answer flips reversed. full5b-91 matches full4-91 on every answer
  but `dito-P7-before`.
- **Eval only (`-r2`):** retrieve, answer and judge again on the kept full5-91 and full4-91 graphs.
  full5-91 scored 41/48 again with 1 flip (`dito-P3-current` stale -> wrong); 34 of 48 context
  packets were identical. full4-91 scored 42/48 again with 2 flips (`dito-P3-current` correct ->
  wrong, `ris-P8-current` wrong -> correct); 37 of 48 packets were identical.

So one graph and one eval give about +/-2 correct answers and +/-1 retired pair between runs of the
same code, and `dito-P3-current` and `ris-P8-current` flip even on an unchanged graph. F3 is not
distinguishable from noise on this set; it costs about +$0.004 per ingest.

## Files

- `dataset/items.json`: the 48 questions (current, before, control) with gold answers and the ADR
  pairs they test.
- `dataset/corpus.jsonl`: the 121 ADRs. `text_U` is the ingested unlabelled text, `text_L` the
  labelled text, `removed_spans` what was stripped, plus repo, commit `sha`, path and date.
  `dataset/corpus-audit.md` lists every stripped span for review.
- `runs/<run>/`: `ingest.jsonl` (per-ADR episode, edges created, edges expired), `consolidate.json`,
  `mechanism.json` / `mechanism.md` (retirement per pair and per control), `llm_usage.csv` (one row
  per model call: operation, tokens, cache, tier, status, error code).
- `runs/<run>-ctx-rr/` and `runs/<run>-ctx-rr-r2/` (eval-only repeat): `packets.jsonl` (the context each question got), `answers.jsonl`,
  `judgments.jsonl`, `report.md`, `summary.json`.

`llm_usage.csv` is exported from Menhir's telemetry DB with only the usage columns; the other tables
(host and session details, payload previews) are not published. For full5-91 it comes from a snapshot
taken before the eval serve, so it holds ingest and consolidation only, like the other two runs.

## Sources and licenses

Each document in the corpus keeps its source repository's license:

| repo | ADRs | license |
|---|---:|---|
| supabase/cli | 34 | MIT |
| digitalservicebund/digitalcheck-dito | 28 | MIT |
| digitalservicebund/ris-search | 28 | GPL-3.0 |
| phel-lang/phel-lang | 23 | MIT |
| FIWARE/data-space-connector | 8 | Apache-2.0 |

## Reproducing

The harness is `archolith-fixtures/scripts/f5/` (`f5_serve.sh up|consolidate|stop`, `f5_run.py`
stages, `f5_flex_audit.py`). Set `PYTHONPATH` to the Graphiti tree, then Menhir's `src`, before
`f5_serve.sh up`; set `F5_PY` to a Python with python-dotenv. For the eval serve, export
`MENHIR_RECALL_RANKING_STRATEGY=reranker_only` and `MENHIR_RECALL_RERANK_MODEL_DIR`, and snapshot the
run's telemetry DB first, since that serve writes to it. `f5_run.py retrieve --serve <menhir tree>`
(after `f5_serve.sh neo4j`) serves a Menhir tree through `archolith_bench.staged`, which needs an
archolith-bench checkout at or after `a19e5c5`.
