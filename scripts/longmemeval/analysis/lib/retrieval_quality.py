"""Deterministic retrieval-quality harness (no answer-model spend).

For each LongMemEval question it measures, per retrieval arm:
  - gold_rank : smallest k such that the gold answer's tokens are all present in the
                ACCUMULATED text of the top-k candidates (i.e. "answer becomes present
                in context at rank k"). None if never present within the cap.
  - present@k : gold_rank <= k.
  - support_rank : smallest k such that the SUPPORT tokens (evidence-turn coverage) are
                   present at >= 50% coverage threshold. This separates retrieval from
                   reasoning: support_rank measures if the _evidence_ is retrieved,
                   gold_rank measures if the _answer_ can be synthesized.

Arms (one /api/recall per question feeds both menhir arms):
  - menhir          : the context the answer model sees in the S4 harness -- /api/recall rendered by
                      the bench's own HttpMenhirClient (authority verdicts, memories, counted verdicts
                      and the source-memory block, in that client's order). This is the gated arm.
  - menhir_results  : only results[].name/content of the same response (the pre-2026-11 measure).
  - graphiti_vector : graphiti-core search_() over the same graph, edge cosine similarity only.
                      This is the gated vector-only baseline.
  - graphiti_hybrid : graphiti-core search() (BM25 + cosine, RRF), the pre-2026-11 baseline.
                      Reported, not gated.

This isolates RETRIEVAL from generation/judge noise: if the gold is not in top-k, it's a
candidate-generation problem; if it is and the answer is still wrong, it's packing/generation.

Every arm call is retried; a call that still fails is recorded on the row and makes the run
INCOMPLETE (overall FAIL, exit 1). Rows are appended to rows.jsonl as they finish and a rerun
resumes from it.
"""
import os, re, json, asyncio, statistics, sys, datetime, subprocess, time, hashlib
from collections import Counter
from pathlib import Path

import glob
import httpx
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    SearchConfig,
)

# Read tolerantly so this module is importable by offline tests; main() enforces them.
MENHIR_URL = os.getenv("MENHIR_URL", "").rstrip("/")
KEY = os.getenv("OPENAI_API_KEY", "")
BOLT = os.getenv("LME_BOLT", "bolt://localhost:7689")
# config.sh's LME_BOLT convention is a bare port (e.g. "7699"), not a full bolt:// URI --
# normalize the same way lib/backfill_dates.py already does. Without this, a caller
# pointing the harness at a non-default graph (LME_BOLT=7699) silently falls through
# unnormalized to a bare "7699" URI, which the neo4j driver rejects/mishandles.
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
PW = os.getenv("LME_NEO4J_PW", "lmedata123")

_BENCH_ROOT = Path(__file__).resolve().parents[4]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from archolith_bench.harness.scalar_bolt import assert_not_prod  # noqa: E402
from archolith_bench.harness.memory_ab import assert_not_production  # noqa: E402
from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402

N = int(os.getenv("LME_N", "30"))
CAP = int(os.getenv("LME_TOPK", "20"))
KS = (5, 10, 20)
EMBEDDING_MODEL = "text-embedding-3-small"
ATTEMPTS = int(os.getenv("LME_IR_ATTEMPTS", "3"))
RETRY_DELAY_S = float(os.getenv("LME_IR_RETRY_DELAY_S", "5"))

# Arm key prefixes on each row: <prefix>_rank (gold) and <prefix>_supp (support).
ARMS = (
    ("menhir", "m"),
    ("menhir_results", "mr"),
    ("graphiti_vector", "gv"),
    ("graphiti_hybrid", "gh"),
)
GATE_ARM, BASELINE_ARM = "m", "gv"

_PUNCT = str.maketrans({c: " " for c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""})
_STOP = set("a an the of to and or in on at for with is was are were be been "
            "i you he she it we they my your his her its our their me him them "
            "do did does what when where which who how why that this these those".split())


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower().translate(_PUNCT)).strip()


def _content_tokens(t: str) -> set:
    return {w for w in _norm(t).split() if w not in _STOP}


def _gold_tokens(t: str) -> set:
    # Distinctive gold tokens: drop stopwords so common words don't fake "presence".
    return {w for w in _norm(t).split() if w not in _STOP and len(w) > 1}


def gold_rank(gold: str, ranked_texts: list[str]) -> int | None:
    """Smallest 1-based k s.t. all distinctive gold tokens appear in top-k accumulated text."""
    gt = _gold_tokens(gold)
    if not gt:
        return None
    acc: set = set()
    for i, txt in enumerate(ranked_texts, 1):
        acc |= _content_tokens(txt)
        if gt <= acc:
            return i
    return None


def evidence_tokens(item: dict) -> set:
    """Distinctive tokens of the SUPPORT: the has_answer evidence turns. Unlike the gold
    answer (often a computed value stored nowhere), the support IS in memory — so this
    separates retrieval failure from reasoning failure."""
    toks: set = set()
    for sess in (item.get("haystack_sessions") or []):
        for turn in (sess if isinstance(sess, list) else []):
            if isinstance(turn, dict) and turn.get("has_answer"):
                toks |= _content_tokens(turn.get("content", ""))
    return toks


def support_rank(support: set, ranked_texts: list[str], thresh: float = 0.5) -> int | None:
    """Smallest 1-based k s.t. the top-k accumulated text COVERS >= thresh of the support
    tokens (coverage, not all-or-nothing, because enrichment rewords raw turns)."""
    if not support:
        return None
    acc: set = set()
    need = max(1, int(len(support) * thresh))
    for i, txt in enumerate(ranked_texts, 1):
        acc |= _content_tokens(txt)
        if len(acc & support) >= need:
            return i
    return None


def make_menhir_client(url: str, api_key: str = "", transport=None):
    """The S4 harness client (source memories on, as the shipped `ship` arm ran) plus a holder
    for the raw /api/recall body, so one call yields both menhir arms."""
    mc = HttpMenhirClient(url, api_key=api_key, source_memories=True)
    captured: dict = {}

    def _keep(resp: httpx.Response) -> None:
        if resp.request.url.path.endswith("/api/recall"):
            resp.read()
            try:
                captured["data"] = resp.json()
            except ValueError:
                captured["data"] = None

    # Private attribute: reuse the S4 renderer without a second recall per question.
    # test_retrieval_quality_arms.py exercises this hook against a mock transport.
    if transport is not None:
        mc._client = httpx.Client(transport=transport, timeout=180.0)
    mc._client.event_hooks = {"request": [], "response": [_keep]}
    return mc, captured


def snippet_kind(packet, rank: int | None) -> str | None:
    """Which part of the rendered context a rank lands in."""
    if rank is None or rank > len(packet):
        return None
    s = packet[rank - 1]
    if s in getattr(packet, "sources", ()):
        return "source_memories"
    if s in getattr(packet, "counted", ()):
        return "counted_verdict"
    return "recall"


def menhir_recall(mc, captured: dict, q: str, ns: str):
    """Returns (packet, results_texts, raw_results). packet is the S4-rendered context."""
    captured.clear()
    packet = mc.recall(ns, q, limit=CAP)
    data = captured.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("recall response body was not captured or not a JSON object")
    raw = [it for it in (data.get("results") or []) if isinstance(it, dict)]
    texts = [f"{it.get('name') or ''} {it.get('content') or ''}".strip() for it in raw]
    return packet, texts, raw


VECTOR_ONLY_CONFIG = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.cosine_similarity], reranker=EdgeReranker.rrf
    ),
    limit=CAP,
)


async def graphiti_vector(g, q: str, ns: str) -> list[str]:
    res = await g.search_(q, config=VECTOR_ONLY_CONFIG, group_ids=[ns])
    return [(getattr(e, "fact", None) or "") for e in res.edges]


async def graphiti_hybrid(g, q: str, ns: str) -> list[str]:
    edges = await g.search(q, group_ids=[ns], num_results=CAP)
    return [(getattr(e, "fact", None) or "") for e in edges]


def call_with_retry(fn, *args, attempts: int | None = None, delay: float | None = None):
    """(value, None) on success, (None, last_error) after `attempts` failures."""
    attempts = ATTEMPTS if attempts is None else attempts
    delay = RETRY_DELAY_S if delay is None else delay
    last = None
    for i in range(attempts):
        try:
            return fn(*args), None
        except Exception as e:  # recorded on the row, never swallowed into an empty arm
            last = f"{type(e).__name__}: {e}"[:300]
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    return None, last


async def acall_with_retry(fn, *args, attempts: int | None = None, delay: float | None = None):
    attempts = ATTEMPTS if attempts is None else attempts
    delay = RETRY_DELAY_S if delay is None else delay
    last = None
    for i in range(attempts):
        try:
            return await fn(*args), None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:300]
            if i < attempts - 1:
                await asyncio.sleep(delay * (i + 1))
    return None, last


def _median_str(s: dict) -> str:
    """median_rank is numeric or None; never format it with `:s` directly."""
    return str(s["median_rank"]) if s["median_rank"] is not None else "None"


def format_wide_line(label: str, s: dict) -> str:
    """Summary row incl. present@20 (menhir-vs-graphiti table). Kept module-level so the
    offline tests exercise the real format path rather than a copy of it."""
    return (f"{label:16s} {s['present@3']:9d}  {s['present@5']:9d}  {s['present@10']:10d}  "
            f"{s['present@20']:10d}  {s['found']:5d}/{s['total']:<5d}  "
            f"{_median_str(s):>11s}  {s['mrr@10']:.4f}")


def format_narrow_line(label: str, s: dict) -> str:
    """Summary row without present@20 (gold-vs-support table)."""
    return (f"{label:16s} {s['present@3']:9d}  {s['present@5']:9d}  {s['present@10']:10d}  "
            f"{s['found']:5d}/{s['total']:<5d}  {_median_str(s):>11s}  {s['mrr@10']:.4f}")


# The explainability contract, verified against a live /api/recall response on 2026-07-15.
# Kept for reference; it is NOT the gate any more: menhir's response model fills
# relevance_basis with a constant default and falls back for retrieval_score_kind and
# final_score, so field presence holds on every result regardless of the backend.
EXPLAINABILITY_FIELDS = ("final_score", "retrieval_score_kind", "relevance_basis")


def has_explainability(result: dict) -> bool:
    """Field presence only (the pre-2026-11 check). See has_score_attribution for the gate."""
    if not isinstance(result, dict):
        return False
    return all(f in result for f in EXPLAINABILITY_FIELDS)


def has_score_attribution(result: dict) -> bool:
    """Gate 3, per result: the backend supplied a numeric retrieval_score. The /api/recall route
    passes retrieval_score through as None when the backend gives none (no default), unlike
    final_score / retrieval_score_kind / relevance_basis, which it fills in."""
    if not isinstance(result, dict):
        return False
    v = result.get("retrieval_score")
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def mrr_at_k(ranks: list[int | None], k: int = 10) -> float:
    """Mean Reciprocal Rank at k: mean of (1/rank if rank <= k else 0) for the subset.
    Handles None (not found) by treating as 0 contribution.
    """
    if not ranks:
        return 0.0
    rr = [1.0 / r if r is not None and r <= k else 0.0 for r in ranks]
    return sum(rr) / len(rr)


def compute_gates(rows: list[dict]) -> dict:
    """Gate verdicts over clean rows. Gate rules fixed before the 2026-11 run:
    1. Hit@3(support): menhir (S4 context) > graphiti vector-only, strictly.
    2. MRR@10(support): menhir >= graphiti vector-only AND menhir > 0 (a 0 = 0 tie fails).
    3. Score attribution: every menhir result carries a backend-supplied retrieval_score.
    Any row with an arm error makes the run incomplete and the overall verdict FAIL."""
    clean = [r for r in rows if not r.get("errors")]
    n = len(clean)
    m = summarize(clean, f"{GATE_ARM}_supp")
    b = summarize(clean, f"{BASELINE_ARM}_supp")
    m_hit3 = m["present@3"] / n if n else 0.0
    b_hit3 = b["present@3"] / n if n else 0.0
    gate1 = (m_hit3 > b_hit3) if n else False
    gate2 = (m["mrr@10"] >= b["mrr@10"] and m["mrr@10"] > 0) if n else False
    total = sum(r.get("m_results_n", 0) for r in clean)
    attributed = sum(r.get("m_attributed_n", 0) for r in clean)
    gate3 = (attributed == total) if total else None
    incomplete = len(rows) - n
    measured = [g for g in (gate1, gate2, gate3) if g is not None]
    overall = bool(measured) and all(measured) and incomplete == 0
    return {
        "n_clean": n, "n_incomplete": incomplete,
        "gate1": gate1, "m_hit3": m_hit3, "b_hit3": b_hit3,
        "gate2": gate2, "m_mrr": m["mrr@10"], "b_mrr": b["mrr@10"],
        "gate3": gate3, "results_total": total, "results_attributed": attributed,
        "overall": overall,
    }


# NOTE: session-scope leakage is deliberately NOT a gate here. It is a boolean invariant,
# not a retrieval-quality metric, and menhir already pins it in its own suite:
#   menhir/tests/test_recall_service.py::test_recall_filters_session_nodes_by_default
#   menhir/tests/test_recall_service.py::test_recall_includes_session_nodes_when_requested
# Cite those as the launch evidence; do not rebuild them behind Docker + API spend.
# Supersession is read off the per-question-type breakdown for `knowledge-update`, which is
# what that LongMemEval type measures on real data.


def _git(args: list[str], cwd: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def load_done(rows_path: Path) -> dict[str, dict]:
    """Rows already finished without errors (resume). Errored rows are redone."""
    done: dict[str, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("errors"):
                    done[r["qid"]] = r
    return done


def select_items(all_items: list[dict], per_type: int) -> list[dict]:
    # STRATIFIED sample: first PER_TYPE of each question_type (the file is grouped by type, so
    # plain --limit N only ever saw temporal-reasoning). 133 covers the full 500.
    by_type: dict[str, list] = {}
    for it in all_items:
        by_type.setdefault(it.get("question_type"), []).append(it)
    items = []
    for _t, lst in by_type.items():
        items.extend(lst[:per_type])
    return items


async def score_item(it: dict, mc, captured: dict, g) -> dict | None:
    qid = str(it.get("question_id") or it.get("id"))
    ns = f"lme-{qid}"
    q = it.get("question", "")
    gold = str(it.get("answer", ""))
    if _norm(gold) in {"", "no answer", "unknown"}:
        return None
    errors: dict[str, str] = {}
    m_out, err = call_with_retry(menhir_recall, mc, captured, q, ns)
    if err:
        errors["menhir"] = err
    gv, err = await acall_with_retry(graphiti_vector, g, q, ns)
    if err:
        errors["graphiti_vector"] = err
    gh, err = await acall_with_retry(graphiti_hybrid, g, q, ns)
    if err:
        errors["graphiti_hybrid"] = err

    packet, m_texts, m_raw = m_out if m_out else ([], [], [])
    packet_texts = list(packet)
    supp = evidence_tokens(it)
    row = {
        "qid": qid, "qtype": it.get("question_type"), "gold": gold[:60],
        "supp_n": len(supp), "errors": errors,
        "m_n": len(packet_texts), "mr_n": len(m_texts),
        "gv_n": len(gv or []), "gh_n": len(gh or []),
        "m_results_n": len(m_raw),
        "m_attributed_n": sum(1 for r in m_raw if has_score_attribution(r)),
        "m_relevance_basis": dict(Counter(str(r.get("relevance_basis")) for r in m_raw)),
        "m_score_kind": dict(Counter(str(r.get("retrieval_score_kind")) for r in m_raw)),
    }
    for _label, p, texts in (("menhir", "m", packet_texts), ("menhir_results", "mr", m_texts),
                             ("graphiti_vector", "gv", gv or []),
                             ("graphiti_hybrid", "gh", gh or [])):
        row[f"{p}_rank"] = gold_rank(gold, texts)
        row[f"{p}_supp"] = support_rank(supp, texts)
    row["m_supp_kind"] = snippet_kind(packet, row["m_supp"])
    return row


def _require_env() -> dict:
    missing = [k for k in ("MENHIR_URL", "OPENAI_API_KEY", "MENHIR_SERVED_SRC", "LME_IR_OUT")
               if not os.getenv(k)]
    if missing:
        raise SystemExit(f"missing env: {', '.join(missing)}")
    served = os.environ["MENHIR_SERVED_SRC"]
    commit = _git(["rev-parse", "HEAD"], served)
    if not commit:
        raise SystemExit(f"MENHIR_SERVED_SRC is not a git checkout: {served}")
    return {
        "menhir_src": served,
        "menhir_commit": commit,
        "menhir_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"], served)),
        "bench_commit": _git(["rev-parse", "HEAD"], str(_BENCH_ROOT)) or "unknown",
        "bench_dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"],
                                 str(_BENCH_ROOT))),
    }


async def main():
    prov = _require_env()
    assert_not_production(MENHIR_URL)
    assert_not_prod(BOLT)
    out_dir = Path(os.environ["LME_IR_OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "ir-gate.json").exists():
        raise SystemExit(f"{out_dir / 'ir-gate.json'} exists; use a new LME_IR_OUT")
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not cached:
        raise SystemExit("longmemeval_oracle not found in HF cache; run a bench arm first to populate it.")
    dataset_bytes = Path(cached[0]).read_bytes()
    all_items = json.loads(dataset_bytes.decode("utf-8"))
    per_type = int(os.getenv("LME_PER_TYPE", "15"))
    items = select_items(all_items, per_type)

    rows_path = out_dir / "rows.jsonl"
    done = load_done(rows_path)
    mc, captured = make_menhir_client(
        MENHIR_URL, api_key=os.getenv("MENHIR_AGENT_KEY") or os.getenv("MENHIR_API_KEY") or "")
    driver = Neo4jDriver(uri=BOLT, user="neo4j", password=PW, database="neo4j")
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=KEY, embedding_model=EMBEDDING_MODEL, embedding_dim=1536))
    g = Graphiti(uri=BOLT, user="neo4j", password=PW, graph_driver=driver, embedder=embedder,
                 llm_client=None, cross_encoder=None)

    rows: list[dict] = []
    skipped = 0
    try:
        with open(rows_path, "a", encoding="utf-8") as rf:
            for i, it in enumerate(items, 1):
                qid = str(it.get("question_id") or it.get("id"))
                if qid in done:
                    rows.append(done[qid])
                    continue
                row = await score_item(it, mc, captured, g)
                if row is None:
                    skipped += 1
                    continue
                rf.write(json.dumps(row) + "\n")
                rf.flush()
                rows.append(row)
                if i % 25 == 0:
                    print(f"  {i}/{len(items)} done", flush=True)
    finally:
        await g.close()
        mc.close()

    print_report(rows)
    gates = compute_gates(rows)
    print_gates(gates)
    meta = {
        **prov,
        "run_id": os.getenv("LME_RUN_ID", f"run-{datetime.datetime.now().isoformat()}"),
        "timestamp": datetime.datetime.now().isoformat(),
        "menhir_url": MENHIR_URL,
        "snapshot": os.getenv("LME_SNAPSHOT", "unrecorded"),
        "menhir_profile": os.getenv("LME_ARM_PROFILE", "unrecorded"),
        "neo4j_image": os.getenv("LME_NEO4J_IMAGE", "unrecorded"),
        "dataset": "longmemeval_oracle",
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "embedding_model": EMBEDDING_MODEL,
        "python": sys.executable,
        "graphiti_core_version": _pkg_version("archolith-graphiti-core"),
        "topk": CAP, "per_type": per_type, "items_selected": len(items),
        "items_skipped_no_gold": skipped,
        "recall_payload": {"limit": CAP, "include_session": True, "include_invalidated": True,
                           "client": "HttpMenhirClient(source_memories=True)"},
    }
    emit_artifacts(out_dir, rows, gates, meta)
    if not gates["overall"]:
        sys.exit(1)


def print_report(rows: list[dict]) -> None:
    clean = [r for r in rows if not r.get("errors")]
    print(f"\n===== RETRIEVAL QUALITY (N={len(clean)} clean of {len(rows)}, cap top-{CAP}) =====")
    print(f"{'':16s} present@3  present@5  present@10  present@20  found/total  median_rank  MRR@10")
    for label, p in ARMS:
        print(format_wide_line(f"gold {label}"[:16], summarize(clean, f"{p}_rank")))
    print(f"\n{'':16s} present@3  present@5  present@10  found/total  median_rank  MRR@10")
    for label, p in ARMS:
        print(format_narrow_line(f"supp {label}"[:16], summarize(clean, f"{p}_supp")))
    print("\n===== BY QUESTION TYPE (support present@10) =====")
    print(f"{'type':26s} n    " + "  ".join(f"{p:>8s}" for _l, p in ARMS))
    for t in sorted({str(r["qtype"]) for r in clean}):
        sub = [r for r in clean if str(r["qtype"]) == t]
        cells = "  ".join(f"{summarize(sub, f'{p}_supp')['present@10']:>4d}/{len(sub):<3d}"
                          for _l, p in ARMS)
        print(f"{t[:26]:26s} {len(sub):<4d} {cells}")


def print_gates(gates: dict) -> None:
    print("\n===== GATE VERDICT =====")
    print(f"Gate 1 (Hit@3 support, menhir > graphiti vector-only): menhir={gates['m_hit3']:.2%} "
          f"vector={gates['b_hit3']:.2%} -> {gate_text(gates['gate1'])}")
    print(f"Gate 2 (MRR@10 support, menhir >= vector-only and > 0): {gates['m_mrr']:.4f} vs "
          f"{gates['b_mrr']:.4f} -> {gate_text(gates['gate2'])}")
    print(f"Gate 3 (score attribution on every menhir result): {gates['results_attributed']}/"
          f"{gates['results_total']} -> {gate_text(gates['gate3'])}")
    print(f"Incomplete rows (an arm failed after retries): {gates['n_incomplete']}")
    print(f"OVERALL VERDICT: {gate_text(gates['overall'])}")


def gate_text(v) -> str:
    return "PASS" if v is True else "FAIL" if v is False else "NOT RUN"


def emit_artifacts(out_dir: Path, rows: list[dict], gates: dict, meta: dict) -> None:
    """ir-gate.json (all rows, gates, provenance) and ir-gate.md. Failure raises: a run whose
    evidence was not written must not exit 0."""
    clean = [r for r in rows if not r.get("errors")]
    per_type = {}
    for t in sorted({str(r["qtype"]) for r in clean}):
        sub = [r for r in clean if str(r["qtype"]) == t]
        per_type[t] = {"n": len(sub)}
        for _label, p in ARMS:
            s = summarize(sub, f"{p}_supp")
            per_type[t][f"{p}_support_present@3"] = s["present@3"]
            per_type[t][f"{p}_support_present@10"] = s["present@10"]
            per_type[t][f"{p}_support_mrr@10"] = s["mrr@10"]
            per_type[t][f"{p}_gold_present@10"] = summarize(sub, f"{p}_rank")["present@10"]
    arms = {}
    for label, p in ARMS:
        arms[label] = {"gold": summarize(clean, f"{p}_rank"), "support": summarize(clean, f"{p}_supp")}
    basis = Counter()
    kinds = Counter()
    supp_kind = Counter(str(r.get("m_supp_kind")) for r in clean)
    for r in clean:
        basis.update(r.get("m_relevance_basis") or {})
        kinds.update(r.get("m_score_kind") or {})
    caveats = [
        "Oracle variant: each namespace holds only that question's evidence sessions.",
        "Support presence is token-overlap coverage (>= 50% of the has_answer turns' content "
        "tokens); gold presence needs every distinctive gold token. Paraphrased answers miss.",
        "Rows with no has_answer turns have no support set and count as misses for every arm.",
        "The menhir arm is the S4-rendered context: the source-memory block and counted verdicts "
        "sit after the top-k memories, so they reach Hit@3/MRR@10 only when few memories return.",
        "Gate 1 was made relative (beat the baseline) on 2026-07-15; the baseline became "
        "vector-only and the menhir arm the S4 context before this run, not after it.",
        f"relevance_basis values seen: {dict(basis)}; it is a response-model default, not an "
        "explanation, and Gate 3 does not use it.",
        "Session-scope leakage is pinned by menhir's unit tests, not measured here.",
    ]
    data = {**meta, "gates": gates, "arms": arms, "per_type": per_type,
            "menhir_support_hit_kind": dict(supp_kind), "menhir_score_kinds": dict(kinds),
            "menhir_relevance_basis": dict(basis), "caveats": caveats, "rows": rows}
    (out_dir / "ir-gate.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    lines = [
        "# LongMemEval Oracle IR gate", "",
        f"Run `{meta['run_id']}`, {meta['timestamp']}. Menhir `{meta['menhir_commit']}` "
        f"(dirty={meta['menhir_dirty']}), bench `{meta['bench_commit']}` "
        f"(dirty={meta['bench_dirty']}), snapshot `{meta['snapshot']}`, profile "
        f"`{meta['menhir_profile']}`, top-{meta['topk']}, {len(clean)} clean rows of {len(rows)}.", "",
        "| Gate | Rule | Result | Status |", "|---|---|---|---|",
        f"| Hit@3 (support) | menhir > graphiti vector-only | {gates['m_hit3']:.2%} vs "
        f"{gates['b_hit3']:.2%} | {gate_text(gates['gate1'])} |",
        f"| MRR@10 (support) | menhir >= vector-only, menhir > 0 | {gates['m_mrr']:.4f} vs "
        f"{gates['b_mrr']:.4f} | {gate_text(gates['gate2'])} |",
        f"| Score attribution | every menhir result has a backend retrieval_score | "
        f"{gates['results_attributed']}/{gates['results_total']} | {gate_text(gates['gate3'])} |",
        "", f"**Overall:** {gate_text(gates['overall'])} (incomplete rows: {gates['n_incomplete']})",
        "", "## Arms (support)", "",
        "| Arm | present@3 | present@10 | present@20 | MRR@10 |", "|---|---|---|---|---|",
    ]
    for label, _p in ARMS:
        s = arms[label]["support"]
        lines.append(f"| {label} | {s['present@3']}/{s['total']} | {s['present@10']}/{s['total']} "
                     f"| {s['present@20']}/{s['total']} | {s['mrr@10']:.4f} |")
    lines += ["", "## Per type (support present@10)", "",
              "| Type | n | " + " | ".join(label for label, _p in ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    for t, pt in per_type.items():
        lines.append(f"| {t} | {pt['n']} | " + " | ".join(
            f"{pt[f'{p}_support_present@10']}" for _l, p in ARMS) + " |")
    lines += ["", f"Menhir support hits by context part: {dict(supp_kind)}", "",
              "## Caveats", ""] + [f"- {c}" for c in caveats]
    (out_dir / "ir-gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nArtifacts: {out_dir / 'ir-gate.json'}, {out_dir / 'ir-gate.md'}")


def summarize(subset, key):
    ranks = [r[key] for r in subset]
    found = [x for x in ranks if x is not None]
    line = {f"present@{k}": sum(1 for x in found if x <= k) for k in KS}
    line["present@3"] = sum(1 for x in found if x <= 3)
    line["found"] = len(found)
    line["total"] = len(ranks)
    line["median_rank"] = statistics.median(found) if found else None
    line["mrr@10"] = mrr_at_k(ranks, k=10)
    return line


if __name__ == "__main__":
    asyncio.run(main())
