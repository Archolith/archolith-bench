"""Perception boundary — live threshold tuning (handoff step 5).

Design of record: `menhir-frontier/.agent/for-review/HANDOFF-2026-07-02-perception-boundary.md`
(step 5) + this repo's `.agent/plans/d0-entropy-delta-counting-slice.md` (Arm B: the demand + the
false-positive risk). Steps 1-4 (the gate) are BUILT in `menhir.services.perception`; this is the
one live-only piece: set the single knob (`threshold`) to a precision target. It also measures
**Lever B** (broadened triangulation): with `PT_CROSS=1` the sweep injects a holistic second
derivation (`extract_stated_total`) as gate veto-4, so a confidently-wrong itemized SUM
(`gpt4_d84a3211`: bike_spend=225 vs gold 185, unanimous) is turned WRONG -> ABSTAIN. `PT_CROSS=0`
reproduces the prior baseline. Target: that qid flips to abstain, held-out FP unchanged.

WHAT IT MEASURES (dataset + LLM only — NO graph, NO writes). For each namespace we run the real
k-sample extractor (PT_MODEL; temp>0 only on a non-reasoning model) ONCE, then replay the deterministic `gate` at several
thresholds over the SAME samples (so the sweep is free after extraction):

  * counting slice (the 14 numeric-answer qids): a commit is CORRECT if its value == gold, WRONG if
    it commits a value that isn't gold (a dangerous current-state View), else ABSTAIN. The gate's job
    is to turn would-be WRONG commits into ABSTAIN as the threshold rises.
  * held-out non-counting slice: ANY commit is over-extraction (a View materialized where the query
    isn't a count). The handoff's precision target is ZERO wrong current-state Views here.

Operating point = the highest-recall threshold with WRONG=0 on counting AND 0 over-extraction on
held-out. Recall (correct commits) is the free variable; precision is the constraint.

API-RATE PROTOCOL: on a 429 we STOP immediately and report — never keep hammering the key.

Env: OPENAI_API_KEY (else read from archolith-bench/.env), PT_MODEL=gpt-6-luna, PT_TEMP=0.7 (not
     sent to reasoning models), PT_K=5, BENCHMARK_SERVICE_TIER (default flex),
     PT_THRESHOLDS=0.6,0.8,1.0, PT_COUNT_LIMIT=14, PT_HELDOUT_LIMIT=12, PT_EMBED=1 (dedup on),
     PT_CROSS=1 (Lever B cross-check on; 0 = prior baseline), PT_OUT (~/perception-tune.json).
     Reuses entropy._items/_evidence_prefixes for the same slice.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict

# menhir-frontier is a sibling repo; the gate lives there. Windows path (bench venv Python).
_FRONTIER_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "menhir-frontier", "src")
sys.path.insert(0, os.path.abspath(_FRONTIER_SRC))

from menhir.services.perception import (  # noqa: E402
    Episode, extract_once, extract_stated_total, gate,
)

# reuse the D0 instrument's dataset loader + gold-evidence logic (same slice, same _norm).
sys.path.insert(0, os.path.dirname(__file__))
import entropy  # noqa: E402

_BENCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)
from archolith_bench.core.api import call_with_flex_retry, openai_chat_kwargs  # noqa: E402

# gpt-4o-mini is deprecated. A reasoning model takes no temperature, so on Luna PT_TEMP is not
# sent and the k samples come from the model's default sampling (thresholds were tuned on mini).
MODEL = os.getenv("PT_MODEL", "gpt-6-luna")
TEMP = float(os.getenv("PT_TEMP", "0.7"))
K = int(os.getenv("PT_K", "5"))
THRESHOLDS = [float(x) for x in os.getenv("PT_THRESHOLDS", "0.6,0.8,1.0").split(",")]
COUNT_LIMIT = int(os.getenv("PT_COUNT_LIMIT", "14"))
HELDOUT_LIMIT = int(os.getenv("PT_HELDOUT_LIMIT", "12"))
EMBED_ON = os.getenv("PT_EMBED", "1") == "1"
CROSS_ON = os.getenv("PT_CROSS", "1") == "1"  # Lever B holistic cross-check (veto-4); 0 = prior baseline
EMBED_MODEL = os.getenv("PT_EMBED_MODEL", "text-embedding-3-small")
OUT = os.getenv("PT_OUT", os.path.expanduser("~/perception-tune.json"))
MAX_TURN_CHARS = 2000


# ----------------------------------------------------------------------------- OpenAI seam


def _load_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
    try:
        with open(env, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    raise SystemExit("OPENAI_API_KEY not set and not found in archolith-bench/.env")


def _rate_limit_stop(exc: Exception) -> None:
    print(f"\n!!! 429 RATE LIMIT — STOPPING per protocol.\n    {type(exc).__name__}: {exc}")
    print("    Do not re-run until the reset window passes. Partial results not written.")
    raise SystemExit(2)


def make_llm(client):
    def llm_complete(system: str, user: str) -> str:
        try:
            # Flex capacity 429s are waited out; any other 429 still stops per protocol.
            r = call_with_flex_retry(lambda: client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                **openai_chat_kwargs(MODEL, temperature=TEMP),
            ))
            return r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "rate limit" in str(e).lower():
                _rate_limit_stop(e)
            raise
    return llm_complete


def make_embed(client):
    if not EMBED_ON:
        return None
    cache: dict[str, list[float]] = {}

    def embed(text: str):
        if text in cache:
            return cache[text]
        try:
            v = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "rate limit" in str(e).lower():
                _rate_limit_stop(e)
            return None
        cache[text] = list(v)
        return cache[text]
    return embed


# ----------------------------------------------------------------------------- slice + episodes


def _is_count_answer(answer: str) -> bool:
    """The D0 counting slice: answer is a PURE count/amount, not merely digit-bearing prose. Strip
    $ , whitespace and a trailing k/m multiplier, then require a bare number. Reproduces exactly the
    plan's 14 counting qids (verified), excluding 'Samsung Galaxy S21' / '7 days.' style answers."""
    s = str(answer).strip().lower().rstrip(".").replace("$", "").replace(",", "").replace(" ", "")
    s = re.sub(r"[km]$", "", s)
    return bool(s) and (s.isdigit() or s.replace(".", "", 1).isdigit())


def _gold_values(answer: str) -> list[float]:
    """Canonical gold candidates from a messy answer string ('$185', '$2,500', '$400k', '220').
    Returns every plausible numeric reading so scoring isn't brittle to the k/thousands convention."""
    s = str(answer).lower().replace(",", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([km])?", s)
    if not m:
        return []
    base = float(m.group(1))
    cands = {base}
    if m.group(2) == "k":
        cands.add(base * 1_000)
    elif m.group(2) == "m":
        cands.add(base * 1_000_000)
    return sorted(cands)


def _episodes(item) -> list[Episode]:
    """User turns across all sessions, each grounded with its session date so the extractor can emit
    a concrete `when`. User turns only (perception reads the user's own statements — matches Arm B)."""
    dates = item.get("haystack_dates") or []
    out: list[Episode] = []
    for si, sess in enumerate(item.get("haystack_sessions") or []):
        if not isinstance(sess, list):
            continue
        date = str(dates[si])[:10] if si < len(dates) else ""
        for ti, turn in enumerate(sess):
            if not isinstance(turn, dict) or turn.get("role") != "user":
                continue
            content = (turn.get("content") or "")[:MAX_TURN_CHARS]
            if len(content) < 8:
                continue
            out.append(Episode(uuid=f"{item['question_id']}-{si}-{ti}",
                               content=f"[{date}] {content}"))
    return out


# ----------------------------------------------------------------------------- scoring


def _matches_gold(value: float, golds: list[float]) -> bool:
    for g in golds:
        tol = max(0.5, abs(g) * 0.001)  # cent-level for money, exact for counts
        if abs(value - g) <= tol:
            return True
    return False


def _score_counting(decisions, golds) -> str:
    committed = [d for d in decisions if d.committed]
    if not committed:
        return "abstain"
    if any(_matches_gold(d.value, golds) for d in committed):
        return "correct"
    return "wrong"  # committed a value that isn't gold — the dangerous current-state View


def main():
    from openai import OpenAI
    client = OpenAI(api_key=_load_key())
    llm = make_llm(client)
    embed = make_embed(client)

    items = entropy._items()  # same TYPES/PER sample as the D0 instrument
    counting = [it for it in items if _is_count_answer(it["answer"])][:COUNT_LIMIT]
    heldout = [it for it in items if not _is_count_answer(it["answer"])][:HELDOUT_LIMIT]
    print(f"slice: {len(counting)} counting (gold numeric), {len(heldout)} held-out non-counting; "
          f"k={K}, temp={TEMP}, model={MODEL}, dedup={'on' if embed else 'off'}, "
          f"cross_check={'on' if CROSS_ON else 'off'}")

    records = []
    t0 = time.time()
    for tag, group in (("counting", counting), ("heldout", heldout)):
        for it in group:
            qid = it["question_id"]
            eps = _episodes(it)
            samples = [extract_once(eps, llm) for _ in range(K)]  # k temp>0 extractions
            golds = _gold_values(it["answer"]) if tag == "counting" else []

            # Lever B: a holistic second derivation of each measure's total, memoized per (qid, measure)
            # so the threshold sweep replays it for free — one extra LLM call per distinct measure, not
            # per threshold (the plan's k=1 cost guard). None (no basis) => no veto, precision unchanged.
            cross_cache: dict[str, float | None] = {}

            def cross_check(measure: str, _eps=eps) -> float | None:
                if measure not in cross_cache:
                    cross_cache[measure] = extract_stated_total(_eps, measure, llm)
                return cross_cache[measure]
            rec = {"qid": qid, "tag": tag, "qtype": it["question_type"],
                   "answer": it["answer"], "golds": golds, "n_eps": len(eps), "by_threshold": {}}
            for th in THRESHOLDS:
                decisions = gate(samples, threshold=th, embed=embed,
                                 cross_check=cross_check if CROSS_ON else None)
                committed = [{"subject": d.subject, "measure": d.measure, "reducer": d.reducer,
                              "value": d.value, "agreement": round(d.agreement, 3),
                              "triangulated": d.triangulated, "cross_total": d.cross_total}
                             for d in decisions if d.committed]
                rec["by_threshold"][str(th)] = {
                    "committed": committed,
                    "verdict": _score_counting(decisions, golds) if tag == "counting"
                    else ("over_extracted" if committed else "clean"),
                }
            records.append(rec)
            v10 = rec["by_threshold"][str(THRESHOLDS[-1])]["verdict"]
            print(f"  [{tag:8s}] {qid:14s} gold={str(it['answer'])[:8]:8s} "
                  f"@{THRESHOLDS[-1]}={v10}")

    json.dump({"config": {"model": MODEL, "temp": TEMP, "k": K, "thresholds": THRESHOLDS,
                          "embed": bool(embed), "cross_check": CROSS_ON}, "records": records},
              open(OUT, "w"), indent=2)

    # ---- sweep report ----
    print(f"\n===== PERCEPTION THRESHOLD SWEEP (elapsed {time.time()-t0:.0f}s) =====")
    print("{:>6s} | {:>7s} {:>5s} {:>7s} | {:>13s}".format(
        "thresh", "correct", "wrong", "abstain", "heldout_FP"))
    n_count = sum(1 for r in records if r["tag"] == "counting")
    n_held = sum(1 for r in records if r["tag"] == "heldout")
    for th in THRESHOLDS:
        c = defaultdict(int)
        for r in records:
            v = r["by_threshold"][str(th)]["verdict"]
            c[v] += 1
        print("{:>6.2f} | {:>4d}/{:<2d} {:>5d} {:>7d} | {:>8d}/{:<3d}".format(
            th, c["correct"], n_count, c["wrong"], c["abstain"], c["over_extracted"], n_held))
    print(f"\ntarget operating point: highest thresh with wrong=0 AND heldout_FP=0 (rows -> {OUT})")


if __name__ == "__main__":
    main()
