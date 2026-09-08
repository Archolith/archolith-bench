"""Answer A/B — do the committed perception Views improve END-TO-END answer accuracy?

The ROI question the rest of the perception work leaves open: we proved precision (no wrong Views)
and the D0 retrieval collapse (a correct View goes censored -> rank-1), but never that the Views
change actual answers. This measures exactly that, scoped to the COUNTING SLICE (the questions a
count/total View can plausibly answer), via the answer-facing path:

    /api/context (BriefBuilder — where committed Views surface) -> gpt-4o answer -> gpt-4o-mini judge

Two arms, gathered by the runner (`answer_ab.sh`) serving menhir over the SAME graph and deleting the
perception Views between collect passes:
    MODE=collect AB_TAG=withviews  -> briefs_withviews.json   (Views present)
    MODE=collect AB_TAG=noviews    -> briefs_noviews.json     (Views deleted)
    MODE=score                     -> answer+judge both, report per-qid + aggregate delta

Isolates the Views' effect: identical questions, identical retriever, the only difference is whether
the committed aggregate is in the graph. Env: MODE, AB_TAG, MENHIR_URL, AB_OUT, ANSWER_MODEL,
JUDGE_MODEL, OPENAI_API_KEY (score only). STOP on 429 per protocol.
"""
import json, os, re, sys, collections
import httpx

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
import entropy  # noqa: E402  (dataset loader — same stratified sample as every other tool)

MODE = os.getenv("MODE", "collect")
TAG = os.getenv("AB_TAG", "withviews")
MENHIR_URL = os.getenv("MENHIR_URL", "http://localhost:8120").rstrip("/")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4o")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
OUT = os.getenv("AB_OUT", os.path.expanduser("~/lme-answer-ab"))
os.makedirs(OUT, exist_ok=True)


def _assert_target_allowed(url: str) -> None:
    _bench = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
    if _bench not in sys.path:
        sys.path.insert(0, _bench)
    from archolith_bench.harness.memory_ab import assert_not_production

    assert_not_production(url)


def _is_count_answer(answer: str) -> bool:
    s = str(answer).strip().lower().rstrip(".").replace("$", "").replace(",", "").replace(" ", "")
    s = re.sub(r"[km]$", "", s)
    return bool(s) and (s.isdigit() or s.replace(".", "", 1).isdigit())


def _counting_slice():
    return [it for it in entropy._items() if _is_count_answer(it["answer"])]


def collect():
    # Guarded in collect(), the only path that contacts a live instance;
    # MODE=score reads committed files and needs no target.
    _assert_target_allowed(MENHIR_URL)

    rows = []
    with httpx.Client(timeout=120) as c:
        for it in _counting_slice():
            qid = str(it["question_id"])
            r = c.post(f"{MENHIR_URL}/api/context",
                       json={"query": it["question"], "limit": 10, "namespace": f"lme-{qid}",
                             "max_tokens": 4000})
            r.raise_for_status()
            rows.append({"qid": qid, "qtype": it["question_type"], "question": it["question"],
                         "answer": str(it.get("answer", "")), "brief": r.json().get("context") or ""})
    path = f"{OUT}/briefs_{TAG}.json"
    json.dump(rows, open(path, "w"), indent=2)
    print(f"[collect:{TAG}] {len(rows)} counting briefs -> {path}  "
          f"(empty briefs: {sum(1 for x in rows if not x['brief'])})")


_ANSWER_SYS = ("You are answering a question using ONLY the provided memory context about a user's "
               "past conversations. Answer concisely. If the context does not contain the answer, "
               "say you don't know.")
_JUDGE_SYS = ("You grade whether a predicted answer matches the gold answer for a question. Reply "
              "with exactly 'yes' or 'no'. Accept paraphrases and semantically equivalent answers; "
              "require the key fact (the number/amount) to match.")


def _chat(c, model, system, user):
    if not OPENAI_KEY:
        raise SystemExit("MODE=score requires OPENAI_API_KEY")
    try:
        r = c.post("https://api.openai.com/v1/chat/completions",
                   headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                   json={"model": model, "temperature": 0,
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]})
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print("\n!!! 429 RATE LIMIT — STOPPING per protocol.", e.response.text[:200])
            raise SystemExit(2)
        raise
    return r.json()["choices"][0]["message"]["content"].strip()


def score():
    wv = {r["qid"]: r for r in json.load(open(f"{OUT}/briefs_withviews.json"))}
    nv = {r["qid"]: r for r in json.load(open(f"{OUT}/briefs_noviews.json"))}
    qids = [q for q in wv if q in nv]
    tally = {"withviews": [0, 0], "noviews": [0, 0]}
    detail = []
    with httpx.Client(timeout=120) as c:
        for qid in qids:
            base = wv[qid]
            q, gold = base["question"], base["answer"]
            row = {"qid": qid, "gold": gold, "brief_differs": wv[qid]["brief"] != nv[qid]["brief"]}
            for tag, src in (("noviews", nv), ("withviews", wv)):
                pred = _chat(c, ANSWER_MODEL, _ANSWER_SYS,
                             f"Memory context:\n{src[qid]['brief']}\n\nQuestion: {q}\nAnswer:")
                verdict = _chat(c, JUDGE_MODEL, _JUDGE_SYS,
                                f"Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\nCorrect?")
                ok = verdict.lower().startswith("y")
                tally[tag][0] += int(ok); tally[tag][1] += 1
                row[tag] = {"pred": pred[:100], "ok": ok}
            flip = "  <== FLIP" if row["withviews"]["ok"] != row["noviews"]["ok"] else ""
            print(f"  {qid:14s} gold={gold[:8]:8s} noviews={'Y' if row['noviews']['ok'] else 'n'} "
                  f"withviews={'Y' if row['withviews']['ok'] else 'n'} "
                  f"brief_differs={row['brief_differs']}{flip}")
            detail.append(row)
    json.dump(detail, open(f"{OUT}/answer_ab_detail.json", "w"), indent=2)

    nvv, wvv = tally["noviews"], tally["withviews"]
    na = nvv[0] / max(nvv[1], 1); wa = wvv[0] / max(wvv[1], 1)
    print(f"\n===== ANSWER A/B — counting slice ({len(qids)} q, answer={ANSWER_MODEL}, judge={JUDGE_MODEL}) =====")
    print(f"  NO views  : {nvv[0]:>2d}/{nvv[1]:<2d}  {na:.2f}")
    print(f"  WITH views: {wvv[0]:>2d}/{wvv[1]:<2d}  {wa:.2f}   delta {wa-na:+.2f}")
    diff = sum(1 for r in detail if r["brief_differs"])
    print(f"  ({diff}/{len(qids)} briefs actually differ between arms — the rest are no-op namespaces)")
    print(f"  detail -> {OUT}/answer_ab_detail.json; N is small — read per-qid FLIPs, not just the aggregate")


if __name__ == "__main__":
    (collect if MODE == "collect" else score)()
