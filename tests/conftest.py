"""Pytest collection guard for menhir-coupled ladder benches.

The R3 (belief/temporal/scope) and R5 (cost-aware scheduler) ladder benches
import menhir's pure-domain modules -- e.g. ``menhir.domain.belief`` and
``menhir.domain.git_staleness`` -- to exercise the ported signal layer. ``menhir``
is a separate repository, not a dependency of archolith-bench, and it is not
installable in CI (its private ``archolith-mcp-framework`` + graphiti chain cannot be
resolved there).

So: when ``menhir`` is not importable we skip those ladder modules and run the
rest of the offline bench suite. When a menhir checkout is on ``PYTHONPATH``
(local ladder runs), the modules collect and run normally -- no behavior change.
"""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("menhir") is None:
    # r3/ and r5/ bench subpackages import menhir at module import time.
    collect_ignore_glob += ["test_r3_*.py", "test_r5_*.py"]
    # engineering_evidence (S4+S5 suite and the longitudinal sprint) does too.
    collect_ignore_glob += [
        "test_engineering_evidence_bench.py",
        "test_longitudinal_sprint_bench.py",
        "test_sprint_v2.py",
        "test_sprint_prompts_v21.py",
        "test_harness_executor.py",
    ]

if importlib.util.find_spec("cth_flex") is None:
    # The archolith-harness agent runs model calls through the private cth.flex package.
    collect_ignore_glob += ["test_harness_executor.py"]

# The suite smoke test exercises the optional audit/filter integrations, which
# import the private sibling packages `archolith_mcp_audit` / `archolith_filter`
# (pyproject `audit` / `filter` extras). The public base bench runs without them,
# so skip that smoke test when either optional dependency is absent.
if importlib.util.find_spec("archolith_mcp_audit") is None or importlib.util.find_spec("archolith_filter") is None:
    collect_ignore_glob += ["test_suite_smoke.py"]
