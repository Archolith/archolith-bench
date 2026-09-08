# Corpus Cards

Provenance for every sample in `corpora/`. These files feed the filter suite's per-category
compression table, so a mislabeled sample becomes a wrong number downstream.

Compiled 2026-09-08 from git history and file content. Token counts are from
`core.metrics.estimate_tokens` and match the category totals in `BENCHMARKS.md`.

**Category is inferred from the filename stem**, not from content — `core/corpus.py:_infer_category`
regex-matches the stem against `git_diff|git_log|git_status|git_show|search|logs|json|read_file|lint|build|test|generic`,
and `CATEGORY_COMMANDS` then attributes a representative command to it (e.g. `git_diff` ->
`git diff --staged`). Nothing validates that the content matches the name. Two files below
are misfiled as a result.

## Origin

All `.txt` samples were captured from real working sessions in this workspace:

| Commit | Date | Files |
|--------|------|-------|
| `35cf047` | 2026-05-29 | `git_diff.txt` |
| `5112334` | 2026-05-30 | the other 11 `.txt` samples |
| `6d1e3f9` | 2026-07-14 | `menhir_recall_anchors.json`, `menhir_recall_negatives.json` |

<!-- archolith-claim-scan: ignore-next-line -->
`5112334` is titled "M2 representative filter corpus — 50% savings from real sessions", which
is the basis for calling these real rather than synthetic. Content corroborates it: real
`yawn.market` Java paths, real Spring Boot startup logs, real commit subjects.

Some sanitization happened. `search_grep.txt` carries rewritten paths of the form
`/home/user/projects\projects\yawn\...` — a POSIX prefix spliced onto Windows separators,
which is a substitution artifact rather than anything a tool emitted. Treat absolute paths in
these files as rewritten, not original.

## Cards

| File | Category | Tokens | Kind | Notes |
|------|----------|--------|------|-------|
| `git_diff.txt` | git_diff | 1,424 | real | **Mixed capture.** Opens with `git status` output ("On branch master / Changes not staged"), then 16 diff-marker lines. Shaped like `git status -v`, not `git diff`. |
| `git_diff_large.txt` | git_diff | 3,142 | real | **Misfiled — contains no diff.** See below. |
| `git_log.txt` | git_log | 580 | real | `git log --oneline`-shaped; real commit subjects from a yawn.market ingest-budget branch. |
| `git_status.txt` | git_status | 774 | real | Porcelain-shaped ` M` / ` D` lines over `.agent/` paths. Matches its name. |
| `json_mcp.txt` | json | 2,078 | real | Single JSON object whose `result` is an escaped multi-line Docker log (Spring Boot boot, Flyway, Hikari). Real MCP tool-return shape. Contains infrastructure detail — db name, ports, service names. No credentials seen. |
| `logs_real.txt` | logs | 619 | real | Shell session output about tracked-file counts under `projects/archolith`. Thin for a "logs" exemplar. |
| `logs_server_boot.txt` | logs | 358 | real | Server boot log. Smallest sample in the corpus. |
| `read_file_source.txt` | read_file | 1,296 | real | Source-file read. Matches its name. |
| `search_grep.txt` | search | 1,739 | real | grep output with context lines. Paths rewritten (see Origin). |
<!-- archolith-claim-scan: ignore-next-line -->
| `search_large.txt` | search | 3,849 | real | Largest single sample. Dominates the `search` category (69% of 5,588). |
| `test_pytest.txt` | test | 760 | real | pytest output. |
| `test_verbose.txt` | test | 1,091 | real | Verbose pytest output. |
| `menhir_recall_anchors.json` | n/a | — | curated | Recall anchors for the content-vector experiment (`6d1e3f9`). Not part of the filter corpus; `list_corpora` only picks up `.txt`. |
| `menhir_recall_negatives.json` | n/a | — | curated | Negative examples for the same experiment. Same exclusion. |

## Known defect: `git_diff_large.txt` is not a diff

`git_diff_large.txt` contains **zero** `diff --git` lines and zero `+`/`-` change lines. Its
content is a failed `cd` followed by a wall of repeated git CRLF warnings:

```
/usr/bin/bash: line 20: cd: projects/archolith: No such file or directory
warning: in the working copy of '.agent/CONVENTIONS.md', LF will be replaced by CRLF ...
warning: in the working copy of '.agent/README.md', LF will be replaced by CRLF ...
```

Because the stem matches `git_diff`, it is filed under that category and attributed to the
<!-- archolith-claim-scan: ignore-next-line -->
command `git diff --staged`. It is **3,142 of the 4,566 tokens in the git_diff category — 69%**.

This matters beyond bookkeeping. The file is near-identical repeated lines, which is the most
compressible text a filter can be handed. Whatever compression ratio the `git_diff` row
reports is therefore dominated by a sample that is not a diff and not representative of one,
in the direction that flatters the filter.

The 2026-05-30 filter evidence is already retired in `HEADLINE-NUMBERS.md` and must not be
quoted. But this defect survives a re-run: refreshing the evidence without fixing the corpus
reproduces the same distortion with a newer date on it.

Suggested fix, not applied here because it changes every filter number:

- Rename to reflect content (`logs_git_warnings.txt` -> category `logs`), or drop it.
- Rename `git_diff.txt` to match its `git status -v` shape, or trim it to the diff portion.
- Add a content check to `list_corpora` so a stem-inferred category that disagrees with the
  content shape fails loudly instead of silently mis-attributing.

## Gaps

- **No capture dates.** Commit date is a ceiling, not the capture time. The `json_mcp.txt`
  logs carry internal timestamps of 2026-05-26/27, which is the only real evidence of when
  any of this was recorded.
- **No tool/version provenance.** Which git, which pytest, which grep produced these is not
  recorded and is no longer recoverable from the files.
- **Thin categories.** `git_log`, `git_status`, `json`, and `read_file` have exactly one
  sample each. A per-category percentage from n=1 is a single observation, not a rate.
- **No token-count pinning.** These counts depend on `estimate_tokens`, which delegates to
  `archolith_maintenance.token_accounting`. A tokenizer change moves every number here.
