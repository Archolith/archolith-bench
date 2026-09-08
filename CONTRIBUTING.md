# Contributing to archolith-bench

## Getting started

```bash
git clone <repo-url> && cd archolith-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Adding corpus samples

Place `.txt` files in `corpora/` with a category prefix:

| Prefix | Filter |
|--------|--------|
| `git_diff_*` | Git diff hunk compression |
| `git_log_*` | Git log commit windowing |
| `git_status_*` | Git status truncation |
| `search_*` | Search result match capping |
| `read_file_*` | Source code import/comment collapse |
| `test_*` | Test output summary |
| `build_*` | Build log compression |
| `json_*` | JSON depth/key truncation |
| `logs_*` | Log line deduplication |

Prefer real tool outputs extracted from agent sessions over synthetic data.

## Adding scenarios

Scenarios are JSON files in `scenarios/` defining multi-turn conversations.
See `scenarios/code_review.json` for the schema.

## Commit conventions

- `feat:` new suite, scenario, or corpus category
- `fix:` bug fix in measurement or reporting
- `chore:` dependency updates, CI, cleanup
- `docs:` README, BENCHMARKS.md, comments

## Code style

- Python 3.11+, type hints, `from __future__ import annotations`
- No external formatting tool enforced; keep consistent with existing code

## Contributor License Agreement

By submitting a pull request, you agree to the [CLA](CLA.md). In short:
you keep your copyright, but you grant the project owner a broad license
to use your contribution — including under commercial licenses. The project
itself is distributed under the [Apache License 2.0](LICENSE).

Every PR template includes a CLA checkbox. Please check it before submitting.
