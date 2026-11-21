"""The build pipeline's settings, resolved in Python instead of by sourcing config.sh (P5).

``python -m archolith_bench.build`` used to run build_graph.sh, which sourced config.sh and piped
every LME_*/MENHIR_*/BENCH_* variable to pipeline.py. The CLI now resolves the same settings here
and starts pipeline.py itself. DEFAULTS restates the config.sh lines whose names pipeline.py reads,
with the same expressions and in the same order; tests/test_build_settings.py sources config.sh and
checks the two agree. config.sh and build_graph.sh stay for the KU buildout, the smokes and the
research tools.
"""

from __future__ import annotations

import re
from pathlib import Path

from .pipeline import expand

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONTAINER = "menhir-lme-neo4j"

# (name, config.sh expression), in config.sh order. The YAML sets most of these already; the
# defaults apply to the rest and to fields the YAML leaves null.
DEFAULTS: tuple[tuple[str, str], ...] = (
    ("MENHIR_MAIN", "${MENHIR_MAIN:-${ARCH_DIR}/menhir}"),
    ("MENHIR_FRONTIER", "${MENHIR_FRONTIER:-${MENHIR_MAIN}}"),
    ("LME_NEO4J_NAME", "${LME_NEO4J_NAME:-menhir-lme-neo4j}"),
    ("LME_BOLT", "${LME_BOLT:-7689}"),
    ("LME_HTTP", "${LME_HTTP:-7476}"),
    ("LME_NEO4J_PW", "${LME_NEO4J_PW:-lmedata123}"),
    ("LME_NEO4J_VOL", "${LME_NEO4J_VOL:-menhir-lme-data}"),
    ("LME_NEO4J_IMAGE", "${LME_NEO4J_IMAGE:-neo4j:5.26-community}"),
    ("LME_NEO4J_HEAP", "${LME_NEO4J_HEAP:-2G}"),
    ("LME_PORT_BUILD", "${LME_PORT_BUILD:-8102}"),
    ("LME_EXTRACT_MODEL", "${LME_EXTRACT_MODEL:-gpt-6-luna}"),
    ("LME_EXTRACT_PROVIDER", "${LME_EXTRACT_PROVIDER:-openai}"),
    ("LME_EXTRACT_BASE_URL", "${LME_EXTRACT_BASE_URL:-}"),
    ("LME_EXTRACT_API_KEY", "${LME_EXTRACT_API_KEY:-not-needed}"),
    ("LME_EMBED_MODEL", "${LME_EMBED_MODEL:-text-embedding-3-small}"),
    ("LME_OPENAI_SERVICE_TIER", "${LME_OPENAI_SERVICE_TIER:-flex}"),
    ("BENCHMARK_SERVICE_TIER", "${BENCHMARK_SERVICE_TIER:-${LME_OPENAI_SERVICE_TIER}}"),
    ("MENHIR_OPENAI_SERVICE_TIER", "${MENHIR_OPENAI_SERVICE_TIER:-${LME_OPENAI_SERVICE_TIER}}"),
    ("LME_ANCHORED_TIME_RESOLVER", "${LME_ANCHORED_TIME_RESOLVER:-1}"),
    ("LME_ANCHORED_TIME_MODEL", "${LME_ANCHORED_TIME_MODEL:-gpt-6-luna}"),
    ("LME_ANCHORED_TIME_ATTACH_MODEL", "${LME_ANCHORED_TIME_ATTACH_MODEL:-${LME_ANCHORED_TIME_MODEL}}"),
    ("LME_MAX_LLM_CALLS_PER_JOB", "${LME_MAX_LLM_CALLS_PER_JOB:-100}"),
    ("LME_WRITER_MODEL", "${LME_WRITER_MODEL:-gpt-6-luna}"),
    ("LME_WRITER_GRANULARITY", "${LME_WRITER_GRANULARITY:-turn}"),
    ("LME_WRITER_MAX_USD", "${LME_WRITER_MAX_USD:-}"),
    ("LME_WRITER_RATES", "${LME_WRITER_RATES:-}"),
    ("LME_SEGMENTATION", "${LME_SEGMENTATION:-adaptive}"),
    ("LME_REQUIRE_FRESH", "${LME_REQUIRE_FRESH:-0}"),
    ("LME_RESUME", "${LME_RESUME:-0}"),
    ("LME_RESET_FILE", "${LME_RESET_FILE:-}"),
    ("LME_NONCANONICAL", "${LME_NONCANONICAL:-0}"),
    ("LME_REQUIRE_TURN_EVIDENCE", "${LME_REQUIRE_TURN_EVIDENCE:-1}"),
    ("LME_STORE_ONLY_CONTEXT", "${LME_STORE_ONLY_CONTEXT:-1}"),
    ("LME_ASSISTANT_STORE_ONLY", "${LME_ASSISTANT_STORE_ONLY:-0}"),
    ("LME_EMBED_BACKFILL", "${LME_EMBED_BACKFILL:-1}"),
    ("LME_CONSOLIDATION_AUDIT_ENABLED", "${LME_CONSOLIDATION_AUDIT_ENABLED:-1}"),
    ("LME_RECALL_AUDIT_ENABLED", "${LME_RECALL_AUDIT_ENABLED:-1}"),
    ("LME_MENHIR_LOG_LEVEL", "${LME_MENHIR_LOG_LEVEL:-INFO}"),
    ("LME_COUNTER_STATE_ENABLED", "${LME_COUNTER_STATE_ENABLED:-0}"),
    ("LME_EVENT_HISTORY_ENABLED", "${LME_EVENT_HISTORY_ENABLED:-0}"),
    ("LME_SCALAR_STATE_ENABLED", "${LME_SCALAR_STATE_ENABLED:-0}"),
    ("LME_SCALAR_HISTORY_ENABLED", "${LME_SCALAR_HISTORY_ENABLED:-0}"),
    ("LME_SCALAR_CONSOLIDATION_K", "${LME_SCALAR_CONSOLIDATION_K:-3}"),
    ("LME_SCALAR_CALL_BUDGET", "${LME_SCALAR_CALL_BUDGET:-50}"),
    ("LME_INGEST_CONCURRENCY", "${LME_INGEST_CONCURRENCY:-30}"),
    ("LME_SCALAR_THRESHOLD", "${LME_SCALAR_THRESHOLD:-2/3}"),
    ("LME_CANONICAL_SELF_BINDING_MODE",
     "${LME_CANONICAL_SELF_BINDING_MODE:-${MENHIR_CANONICAL_SELF_BINDING_MODE:-off}}"),
    ("LME_REQUIRE_SCALAR_OUTPUT", "${LME_REQUIRE_SCALAR_OUTPUT:-1}"),
    ("LME_BACKFILL_DATES", "${LME_BACKFILL_DATES:-0}"),
    ("LME_NS_PREFIX", "${LME_NS_PREFIX:-lme-}"),
    ("LME_DATASET", "${LME_DATASET:-xiaowu0162/longmemeval}"),
    ("LONGMEMEVAL_VARIANT", "${LONGMEMEVAL_VARIANT:-oracle}"),
    ("LME_RESULTS_DIR", "${LME_RESULTS_DIR:-${BENCH_DIR}/results/lme-ingest}"),
)
# config.sh's two exports: the only names it adds to the environment children inherit.
EXPORTED = ("BENCHMARK_SERVICE_TIER", "MENHIR_OPENAI_SERVICE_TIER")
# build_graph.sh hands pipeline.py these shell variables on stdin.
_PASSED = re.compile(r"^(LME_|MENHIR_|BENCH_)|^LONGMEMEVAL_VARIANT$")


def _bin(path: str) -> str:
    return f"{path}.exe" if Path(f"{path}.exe").is_file() else path


def resolve(env: dict[str, str], bench_dir: str | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """(settings for pipeline.py, environment for its process), as build_graph.sh produced them.

    ``env`` is the environment build_graph.sh would have started with: the CLI's child env with
    the YAML's assignments and run-control flags applied.
    """
    s = dict(env)
    s["BENCH_DIR"] = env.get("BENCH_DIR") or bench_dir or REPO_ROOT.as_posix()
    s["ARCH_DIR"] = env.get("ARCH_DIR") or s["BENCH_DIR"].rsplit("/", 1)[0]
    for name, expr in DEFAULTS:
        s[name] = expand(expr, s)
    results, name = s["LME_RESULTS_DIR"], s["LME_NEO4J_NAME"]
    scoped = "" if name == CANONICAL_CONTAINER else f"-{name}"
    s["LME_MANIFEST_PATH"] = s.get("LME_MANIFEST_PATH") or f"{results}/manifest{scoped}.json"
    s["LME_REVERT_SNAPSHOT_PATH"] = (s.get("LME_REVERT_SNAPSHOT_PATH")
                                     or f"{results}/date-backfill-revert{scoped}.json")
    for tree in ("MENHIR_MAIN", "MENHIR_FRONTIER"):
        s[f"{tree}_PY"] = s.get(f"{tree}_PY") or f"{s[tree]}/.venv/Scripts/python.exe"
        s[f"{tree}_BIN"] = s.get(f"{tree}_BIN") or _bin(f"{s[tree]}/.venv/Scripts/menhir")
    s["BENCH_PY"] = s.get("BENCH_PY") or f"{s['BENCH_DIR']}/.venv/Scripts/python.exe"
    if "ARCH_DIR" not in env:  # a shell variable build_graph.sh never passed on
        del s["ARCH_DIR"]
    exported = dict(env)
    for var in EXPORTED:
        exported[var] = s[var]
    return s, exported


def stdin_pairs(s: dict[str, str]) -> bytes:
    """build_graph.sh's NUL-separated name/value pairs for the variables it passes on."""
    out = bytearray()
    for name in sorted(n for n in s if _PASSED.search(n)):
        out += name.encode("utf-8") + b"\0" + s[name].encode("utf-8", "surrogateescape") + b"\0"
    return bytes(out)
