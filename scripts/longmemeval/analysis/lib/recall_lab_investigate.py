"""Investigate the M1 gate's low Hit@3: is a specific question a genuine retrieval miss, or a
token-overlap matching-methodology artifact?

`retrieval_quality.py`'s gold_rank/support_rank are cheap, deterministic, GPT-free token-overlap
checks -- exactly why they scale to n=500 without spend. But they can't tell "the right memory was
retrieved but the gold answer is a paraphrase" apart from "the right memory was never retrieved at
all". This script complements that harness (does not replace it) with a small, gold-aware LLM judge
over a HAND-PICKED sample of questions that scored a complete miss (gold_rank=None on BOTH menhir
and graphiti) in the 2026-07-15 full n=500 run -- see SAMPLE_QIDS below for provenance.

Per question, three independent things are gathered, blinded, and judged in ONE call:
  - menhir  : the same /api/recall call retrieval_quality.py makes (production config).
  - graphiti: the same graphiti-core direct search() retrieval_quality.py makes (vector-only).
  - arm_e   : BEST-EFFORT. menhir's Recall Lab "E" tuning arm (production base + frontier, soft
              evidence-anchor) via the Explorer's /explorer/api/recall-lab/run endpoint. If that
              endpoint isn't reachable in benchmark mode (auth/runtime wiring not verified for this
              script), the arm is skipped and the question still gets judged on the other two.

Judge design differs from menhir's own recall_lab.judge_recall_lab: that judge is COMPARATIVE
(which anonymous arm is relatively better) and has no ground truth. This judge is ABSOLUTE and
gold-aware: given the question + the known-correct answer, does THIS arm's retrieved content
contain what's needed to derive that answer, independent of how any other arm did. That directly
answers "is this a real retrieval gap" rather than "which arm looks nicer".

Usage: MENHIR_URL=http://localhost:8109 OPENAI_API_KEY=... python recall_lab_investigate.py
Env: MENHIR_URL, OPENAI_API_KEY, LME_BOLT, LME_NEO4J_PW, LME_JUDGE_MODEL (default gpt-6-luna),
     BENCHMARK_SERVICE_TIER (default flex),
     RLI_EXPLORER_URL (default = MENHIR_URL, since Explorer is mounted on the same app/port).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_BENCH_ROOT = str(Path(__file__).resolve().parents[4])
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)
from archolith_bench.core.api import call_with_flex_retry, openai_chat_kwargs  # noqa: E402

import httpx
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from openai import OpenAI

MENHIR_URL = os.getenv("MENHIR_URL", "").rstrip("/")
EXPLORER_URL = os.getenv("RLI_EXPLORER_URL", MENHIR_URL).rstrip("/")
KEY = os.getenv("OPENAI_API_KEY", "")
BOLT = os.getenv("LME_BOLT", "bolt://localhost:7689")
if not BOLT.startswith("bolt://"):
    BOLT = f"bolt://localhost:{BOLT}"
PW = os.getenv("LME_NEO4J_PW", "lmedata123")
JUDGE_MODEL = os.getenv("LME_JUDGE_MODEL", "gpt-6-luna")
CAP = int(os.getenv("LME_TOPK", "10"))  # fewer than retrieval_quality.py's 20: judge context is bounded

# Hand-picked from the 2026-07-15 full n=500 run's per-question table (full_gate_recal.log):
# 3 complete-miss questions (gold_rank=None on BOTH menhir and graphiti) per question type, chosen
# to spread across all 6 types rather than clustering in the weakest one. single-session-preference
# is deliberately included in full even though ALL 30 of its items missed gold_rank (0/30) -- that
# type is the specific hypothesis under test (abstractive/paraphrased gold answers vs token-overlap
# matching), so its 3 sample items double as a targeted check of that hypothesis.
SAMPLE_QIDS = [
    "gpt4_2655b836", "0bb5a684", "08f4fc43",          # temporal-reasoning
    "0a995998", "6d550036", "gpt4_59c863d7",          # multi-session
    "6a1eabeb", "6aeb4375", "830ce83f",               # knowledge-update
    "8a2466db", "06878be2", "75832dbd",               # single-session-preference (hypothesis test)
    "7161e7e2", "89527b6b", "e9327a54",               # single-session-assistant
    "118b2229", "58ef2f1c", "5d3d2817",               # single-session-user
    # Widening pass (2026-07-15b): knowledge-update over-weighted (92% miss rate in the full run,
    # highest of any type, and the one case above -- 830ce83f -- showed a stale-fact/supersession
    # pattern, not an extraction or ranking gap) to check whether that pattern is systematic.
    "852ce960", "945e3d21", "d7c942c3", "89941a93", "07741c44", "a1eacc2a", "184da446",
    "031748ae", "4d6b87c8", "08e075c7", "f9e8c073", "2698e78f", "b6019101", "45dc21b6", "6071bd76",
    # Widening pass: 4 more per remaining type, for a larger evidence base per failure category.
    "2c63a862", "2a1811e2", "gpt4_0b2f1d21", "f0853d11",       # temporal-reasoning
    "b5ef892d", "3a704032", "gpt4_d84a3211", "aae3761f",       # multi-session
    "6ae235be", "1903aded", "ceb54acb", "0e5e2d1a",            # single-session-assistant
    "3b6f954b", "726462e0", "94f70d80", "66f24dbb",            # single-session-user
]


def _load_items() -> dict[str, dict]:
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not cached:
        raise SystemExit("longmemeval_oracle not found in HF cache")
    items = json.load(open(cached[0], encoding="utf-8"))
    return {str(it.get("question_id") or it.get("id")): it for it in items}


def menhir_recall(q: str, ns: str) -> list[dict]:
    r = httpx.post(f"{MENHIR_URL}/api/recall",
                   json={"query": q, "limit": CAP, "namespace": ns, "include_session": True},
                   timeout=90)
    r.raise_for_status()
    return [it for it in (r.json().get("results") or []) if isinstance(it, dict)]


async def graphiti_search(g: Graphiti, q: str, ns: str) -> list[dict]:
    try:
        edges = await g.search(q, group_ids=[ns], num_results=CAP)
    except Exception:
        return []
    return [{"name": "", "content": getattr(e, "fact", None) or ""} for e in edges]


def recall_lab_arm_e(q: str, ns: str) -> list[dict] | None:
    """Best-effort: menhir's Recall Lab tuning arm E via the Explorer HTTP API. Returns None
    (not an empty list -- distinguishes "arm unreachable" from "arm found nothing") if the
    endpoint isn't reachable, since this script runs outside menhir's process and the endpoint's
    auth/runtime wiring under MENHIR_BENCHMARK_MODE was not verified before writing this."""
    body = {
        "query": q, "namespace": ns, "limit": CAP, "candidate_k": max(CAP, 50),
        "include_session": True, "judge": False,
        "arms": [{
            "id": "e", "label": "E",
            "tuning": {
                "enable_facet_candidates": True, "facet_weight": 0.5,
                "enable_oracle_ranking": True, "enable_intent_lens": True,
                "enable_warden_gate": True, "enable_evidence_anchor": False,
            },
        }],
    }
    try:
        r = httpx.post(f"{EXPLORER_URL}/explorer/api/recall-lab/run", json=body, timeout=30)
        r.raise_for_status()
        arms = r.json().get("arms") or []
        if not arms or not arms[0].get("ok"):
            return None
        return [
            {"name": m.get("name", ""), "content": m.get("content", "")}
            for m in (arms[0].get("results") or [])
        ]
    except Exception:
        return None


_JUDGE_SYSTEM = """You are an impartial retrieval-quality auditor for a long-term memory system.
You will be shown a QUESTION, the CORRECT ANSWER (ground truth, already known -- do not
second-guess it), and several anonymous, independently-retrieved sets of memory snippets. Each set
came from a DIFFERENT retrieval method attempting to answer the same question from the same
underlying conversation history.

For EACH set independently (do not compare sets to each other; a set's verdict must not depend on
how any other set did), decide: does this set of snippets, on its own, contain enough information
for a careful reader to derive the correct answer? Judge only the visible snippet text.

Use these verdicts:
- "yes": the answer (or the exact facts needed to compute/state it) is directly present.
- "partial": relevant/on-topic snippets are present but the specific fact needed is missing,
  incomplete, or requires an inference the snippets don't support.
- "no": the snippets are irrelevant, empty, or provide no path to the answer.

Also give a one-sentence reason per set, and a top-level note: is the correct answer likely a
LITERAL fact that should appear verbatim in a supporting snippet, or an ABSTRACTIVE/paraphrased
summary that a real supporting snippet would only imply rather than state outright (e.g. a stated
preference summarized in different words)? This distinguishes a genuine retrieval miss from a
case where token-overlap matching would fail even on a correct retrieval.

Return JSON only, matching the exact template supplied. No markdown, no commentary outside JSON."""


def _judge_prompt(question: str, gold: str, anonymous_arms: list[tuple[str, list[dict]]]) -> str:
    sections = [f"QUESTION:\n{question}", f"CORRECT ANSWER (ground truth):\n{gold}",
                "ANONYMOUS RETRIEVED SETS:"]
    for alias, results in anonymous_arms:
        lines = [f"\n{alias}:"]
        if not results:
            lines.append("[NO RESULTS]")
        for i, row in enumerate(results, start=1):
            name = str(row.get("name") or "").strip()
            content = str(row.get("content") or "").strip()
            text = f"{name}: {content}" if content and content != name else (content or name)
            lines.append(f"{i}. {text[:600]}")
        sections.append("\n".join(lines))
    aliases = [alias for alias, _ in anonymous_arms]
    template = {
        "answer_type": "literal | abstractive",
        "verdicts": {alias: {"verdict": "yes|partial|no", "reason": "one sentence"} for alias in aliases},
    }
    sections.extend([
        "REQUIRED IDENTIFIERS:\n" + ", ".join(aliases),
        "EXACT OUTPUT TEMPLATE (replace values, preserve every key):\n" + json.dumps(template, indent=2),
    ])
    return "\n\n".join(sections)


def _parse_judge_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge did not return a JSON object")
    return json.loads(cleaned[start:end + 1])


def judge_question(client: OpenAI, question: str, gold: str, arms: dict[str, list[dict] | None]) -> dict:
    """arms: {label -> results or None (unreachable, excluded from judging)}."""
    judgeable = [(label, results) for label, results in arms.items() if results is not None]
    if not judgeable:
        return {"ok": False, "error": "no arms reachable for this question"}
    shuffled = list(judgeable)
    random.SystemRandom().shuffle(shuffled)
    anonymous = [(f"R{i}", results) for i, (_label, results) in enumerate(shuffled, start=1)]
    alias_to_label = {alias: label for alias, (label, _r) in zip((a for a, _ in anonymous), shuffled)}

    resp = call_with_flex_retry(lambda: client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _judge_prompt(question, gold, anonymous)},
        ],
        **openai_chat_kwargs(JUDGE_MODEL, max_tokens=900, temperature=0.0),
    ))
    raw = resp.choices[0].message.content or ""
    try:
        parsed = _parse_judge_json(raw)
        verdicts = parsed.get("verdicts") or {}
        by_label = {alias_to_label[alias]: v for alias, v in verdicts.items() if alias in alias_to_label}
        return {"ok": True, "answer_type": parsed.get("answer_type"), "verdicts": by_label}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "raw": raw[:500]}


async def main() -> None:
    if not MENHIR_URL or not KEY:
        raise SystemExit("MENHIR_URL and OPENAI_API_KEY must be set")
    items = _load_items()
    client = OpenAI(api_key=KEY)
    driver = Neo4jDriver(uri=BOLT, user="neo4j", password=PW, database="neo4j")
    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(api_key=KEY, embedding_model="text-embedding-3-small", embedding_dim=1536))
    g = Graphiti(uri=BOLT, user="neo4j", password=PW, graph_driver=driver, embedder=embedder, llm_client=None, cross_encoder=None)

    rows = []
    arm_e_reachable = None  # tri-state: None=untested, True/False once we know
    try:
        for qid in SAMPLE_QIDS:
            it = items.get(qid)
            if it is None:
                print(f"SKIP {qid}: not found in dataset", flush=True)
                continue
            ns = f"lme-{qid}"
            question = it.get("question", "")
            gold = str(it.get("answer", ""))
            qtype = it.get("question_type")

            m_results = menhir_recall(question, ns)
            g_results = await graphiti_search(g, question, ns)
            e_results = recall_lab_arm_e(question, ns)
            if arm_e_reachable is None:
                arm_e_reachable = e_results is not None
                print(f"Recall Lab arm E: {'reachable' if arm_e_reachable else 'NOT reachable (skipping for all questions)'}", flush=True)

            verdict = judge_question(
                client, question, gold,
                {"menhir": m_results, "graphiti": g_results, "recall_lab_e": e_results},
            )
            rows.append({
                "qid": qid, "qtype": qtype, "question": question, "gold": gold,
                "menhir_n": len(m_results), "graphiti_n": len(g_results),
                "arm_e_n": (len(e_results) if e_results is not None else None),
                "judge": verdict,
            })
            v = verdict.get("verdicts", {}) if verdict.get("ok") else {}
            print(f"{qid:16s} {str(qtype)[:20]:20s} "
                  f"menhir={v.get('menhir', {}).get('verdict', '?'):8s} "
                  f"graphiti={v.get('graphiti', {}).get('verdict', '?'):8s} "
                  f"answer_type={verdict.get('answer_type', '?')}", flush=True)
    finally:
        await g.close()

    _report(rows)


def _report(rows: list[dict]) -> None:
    n = len(rows)
    ok_rows = [r for r in rows if r["judge"].get("ok")]
    print(f"\n===== SUMMARY (n={n}, {len(ok_rows)} judged successfully) =====")
    for arm in ("menhir", "graphiti", "recall_lab_e"):
        counts = {"yes": 0, "partial": 0, "no": 0, "missing": 0}
        for r in ok_rows:
            v = r["judge"]["verdicts"].get(arm)
            counts[v["verdict"] if v else "missing"] += 1
        print(f"  {arm:14s} yes={counts['yes']:2d} partial={counts['partial']:2d} "
              f"no={counts['no']:2d} not-judged={counts['missing']:2d}")
    literal = sum(1 for r in ok_rows if r["judge"].get("answer_type") == "literal")
    abstractive = sum(1 for r in ok_rows if r["judge"].get("answer_type") == "abstractive")
    print(f"  answer_type: literal={literal} abstractive={abstractive}")

    out_dir = Path("results/lme-recall-lab-investigate")
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    json_path = out_dir / f"investigate-{date}.json"
    json.dump({"n": n, "rows": rows}, open(json_path, "w", encoding="utf-8"), indent=2)

    md_path = out_dir / f"investigate-{date}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# LME Recall Miss Investigation (gold-aware LLM judge)\n\n")
        f.write(
            "Complements `retrieval_quality.py`'s token-overlap gold_rank/support_rank on a "
            f"hand-picked sample of {n} questions that scored a complete miss "
            "(gold_rank=None on both menhir and graphiti) in the 2026-07-15 full n=500 run. "
            "Judged per-arm, absolute (not comparative), gold-aware.\n\n"
        )
        f.write("## Summary\n\n| Arm | yes | partial | no | not judged |\n|---|---|---|---|---|\n")
        for arm in ("menhir", "graphiti", "recall_lab_e"):
            counts = {"yes": 0, "partial": 0, "no": 0, "missing": 0}
            for r in ok_rows:
                v = r["judge"]["verdicts"].get(arm)
                counts[v["verdict"] if v else "missing"] += 1
            f.write(f"| {arm} | {counts['yes']} | {counts['partial']} | {counts['no']} | {counts['missing']} |\n")
        f.write(f"\nanswer_type: literal={literal}, abstractive={abstractive} "
                "(abstractive answers are structurally hard for token-overlap matching even on a correct retrieval)\n")
        f.write("\n## Per-Question Detail\n\n")
        for r in rows:
            f.write(f"### {r['qid']} ({r['qtype']})\n\n")
            f.write(f"**Question:** {r['question']}\n\n**Gold answer:** {r['gold']}\n\n")
            j = r["judge"]
            if not j.get("ok"):
                f.write(f"Judge failed: {j.get('error')}\n\n---\n\n")
                continue
            f.write(f"**Answer type:** {j.get('answer_type')}\n\n")
            for arm, n_key in (("menhir", "menhir_n"), ("graphiti", "graphiti_n"), ("recall_lab_e", "arm_e_n")):
                v = j["verdicts"].get(arm)
                n_results = r[n_key]
                if v is None:
                    f.write(f"- **{arm}** ({n_results if n_results is not None else 'unreachable'} results): not judged\n")
                else:
                    f.write(f"- **{arm}** ({n_results} results): **{v['verdict']}** -- {v['reason']}\n")
            f.write("\n---\n\n")

    print(f"\nJSON: {json_path}\nMarkdown: {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
