"""Brief A/B — does the BriefBuilder brief beat the flat brief on answer accuracy?

The Mode-B harness feeds /api/recall (a flat list) to the answer model and never calls
/api/context, so it cannot measure the BriefBuilder (which lives in build_context). This
standalone eval closes that gap: for N stratified questions it pulls the packed brief from
/api/context, answers with the answer model, and judges with the judge model — comparing two
brief sets (flag OFF vs ON) that the runner (brief_ab.sh) gathers by serving menhir twice.

Two phases (so we serve each flag value once):
  MODE=collect BRIEF_TAG=off MENHIR_URL=... python brief_ab.py   -> $BRIEF_OUT_DIR/briefs_off.json
  MODE=collect BRIEF_TAG=on  MENHIR_URL=... python brief_ab.py   -> $BRIEF_OUT_DIR/briefs_on.json
  MODE=score  python brief_ab.py                                  -> answer+judge both, report

Env: MODE, BRIEF_TAG, MENHIR_URL, BRIEF_OUT_DIR, BRIEF_PER_TYPE, BRIEF_TYPES,
     OPENAI_API_KEY, ANSWER_MODEL, JUDGE_MODEL.
"""
import glob, json, os, sys, collections
import httpx

_HERE = os.path.dirname(__file__)

MODE = os.getenv("MODE", "collect")
TAG = os.getenv("BRIEF_TAG", "off")
MENHIR_URL = os.getenv("MENHIR_URL", "http://localhost:8118").rstrip("/")
PER = int(os.getenv("BRIEF_PER_TYPE", "10"))
TYPES = os.getenv("BRIEF_TYPES", "temporal-reasoning,knowledge-update,multi-session").split(",")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")  # only required for MODE=score (answer+judge)
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
OUT = os.getenv("BRIEF_OUT_DIR", os.path.expanduser("~/lme-brief-ab"))
os.makedirs(OUT, exist_ok=True)


def _assert_target_allowed(url: str) -> None:
    _bench = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from archolith_bench.harness.memory_ab import assert_not_production

    assert_not_production(url)


def _items():
    cached = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--xiaowu0162--longmemeval/snapshots/*/longmemeval_oracle"))
    if not cached:
        raise SystemExit("LongMemEval oracle dataset not found in HF cache")
    allit = json.load(open(cached[0], encoding="utf-8"))
    by = collections.defaultdict(list)
    for it in allit:
        by[it.get("question_type")].append(it)
    out = []
    for t in TYPES:
        out.extend(by.get(t, [])[:PER])
    return out


def collect():
    # Guarded in collect(), the only path that contacts a live instance;
    # MODE=score reads committed files and needs no target.
    _assert_target_allowed(MENHIR_URL)

    rows = []
    with httpx.Client(timeout=120) as c:
        for it in _items():
            qid = str(it["question_id"])
            r = c.post(f"{MENHIR_URL}/api/context",
                       json={"query": it["question"], "limit": 10, "namespace": f"lme-{qid}",
                             "max_tokens": 4000})
            r.raise_for_status()
            rows.append({"qid": qid, "qtype": it["question_type"], "question": it["question"],
                         "answer": str(it.get("answer", "")), "brief": r.json().get("context") or ""})
    path = f"{OUT}/briefs_{TAG}.json"
    json.dump(rows, open(path, "w"), indent=2)
    print(f"[collect:{TAG}] wrote {len(rows)} briefs -> {path}  "
          f"(empty: {sum(1 for x in rows if not x['brief'])})")


_ANSWER_SYS = ("You are answering a question using ONLY the provided memory context about a "
               "user's past conversations. Answer concisely. If the context does not contain "
               "the answer, say you don't know.")
_JUDGE_SYS = ("You grade whether a predicted answer matches the gold answer for a question. "
              "Reply with exactly 'yes' or 'no'. Accept paraphrases and semantically equivalent "
              "answers; require the key fact to match.")


def _chat(c, model, system, user):
    if not OPENAI_KEY:
        raise SystemExit("MODE=score requires OPENAI_API_KEY")
    r = c.post("https://api.openai.com/v1/chat/completions",
               headers={"Authorization": f"Bearer {OPENAI_KEY}"},
               json={"model": model, "temperature": 0,
                     "messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def score():
    off = {r["qid"]: r for r in json.load(open(f"{OUT}/briefs_off.json"))}
    on = {r["qid"]: r for r in json.load(open(f"{OUT}/briefs_on.json"))}
    qids = [q for q in off if q in on]
    by_type = collections.defaultdict(lambda: {"off": [0, 0], "on": [0, 0]})
    detail = []
    with httpx.Client(timeout=120) as c:
        for qid in qids:
            base = off[qid]
            qt, q, gold = base["qtype"], base["question"], base["answer"]
            row = {"qid": qid, "qtype": qt}
            for tag, src in (("off", off), ("on", on)):
                brief = src[qid]["brief"]
                pred = _chat(c, ANSWER_MODEL, _ANSWER_SYS,
                             f"Memory context:\n{brief}\n\nQuestion: {q}\nAnswer:")
                verdict = _chat(c, JUDGE_MODEL, _JUDGE_SYS,
                                f"Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\nCorrect?")
                ok = verdict.lower().startswith("y")
                by_type[qt][tag][0] += int(ok)
                by_type[qt][tag][1] += 1
                row[tag] = {"pred": pred[:120], "ok": ok}
            detail.append(row)
    json.dump(detail, open(f"{OUT}/brief_ab_detail.json", "w"), indent=2)

    print(f"\n===== BRIEF A/B ({len(qids)} questions, answer={ANSWER_MODEL}, judge={JUDGE_MODEL}) =====")
    print(f"{'question_type':24s} {'brief OFF':>12s} {'brief ON':>12s}   delta")
    tot = {"off": [0, 0], "on": [0, 0]}
    for qt in TYPES:
        if qt not in by_type:
            continue
        o, n = by_type[qt]["off"], by_type[qt]["on"]
        tot["off"][0] += o[0]; tot["off"][1] += o[1]
        tot["on"][0] += n[0]; tot["on"][1] += n[1]
        oa = o[0] / o[1] if o[1] else 0
        na = n[0] / n[1] if n[1] else 0
        print(f"{qt:24s} {o[0]:>3d}/{o[1]:<3d} {oa:>5.2f} {n[0]:>3d}/{n[1]:<3d} {na:>5.2f}   {na-oa:+.2f}")
    oa = tot["off"][0] / max(tot["off"][1], 1)
    na = tot["on"][0] / max(tot["on"][1], 1)
    print(f"{'-'*24}")
    print(f"{'OVERALL':24s} {tot['off'][0]:>3d}/{tot['off'][1]:<3d} {oa:>5.2f} "
          f"{tot['on'][0]:>3d}/{tot['on'][1]:<3d} {na:>5.2f}   {na-oa:+.2f}")
    print(f"\n(detail -> {OUT}/brief_ab_detail.json; N per cell is small — treat ±1 question as noise)")


if __name__ == "__main__":
    (collect if MODE == "collect" else score)()
