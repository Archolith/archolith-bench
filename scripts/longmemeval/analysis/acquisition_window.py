"""Lever A (σ WINDOW) live measurement — windowed acquisition counts over the D0 counting slice.

The A5 phase of `menhir-frontier/.agent/plans/perception-window-and-triangulation.md`. For each
target qid: extract acquisition events (real LLM via perception_tune, k samples), fold each sample to a lossless
timeline, then answer the question's WINDOW ("how many X in the last month?") as a read-time δ — the
relative phrase resolved against the question date, `count_in_window` over the timeline. Reports the
k-sample distribution + mode vs gold. Dataset+LLM only, no graph writes. STOP on 429 per protocol.

Target: `3a704032` (gold 3 plants acquired in the last month) — the case the count-floor safely
abstained on, now answerable because acquisitions live on a timeline and the window is a read-time δ.

Env: AW_QIDS (default 3a704032), AW_K=5, AW_MODEL/TEMP via perception_tune, AW_OUT.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "menhir-frontier", "src")))
sys.path.insert(0, _HERE)

from menhir.domain.fold_algebra import Event, timeline  # noqa: E402
from menhir.services.perception import extract_once  # noqa: E402
from menhir.services.windowed_fold import count_in_window, resolve_window  # noqa: E402

import entropy  # noqa: E402
import perception_tune as pt  # noqa: E402

QIDS = os.getenv("AW_QIDS", "3a704032").split(",")
K = int(os.getenv("AW_K", "5"))
OUT = os.getenv("AW_OUT", os.path.expanduser("~/acquisition-window.json"))

_PHRASE_RE = re.compile(r"(?:last|past|previous)\s+(?:\d+\s+)?(?:day|week|month|year)s?|this (?:month|year)")


def _window_phrase(question: str) -> str:
    m = _PHRASE_RE.search(question.lower())
    return m.group(0) if m else "last month"


def _acquisition_entries(groups) -> list[dict]:
    """All acquire-kind events across a sample's groups (aggregate-keyed by the extractor), folded to
    lossless timeline entries (dedup by (when, item))."""
    evs = [Event(when=e.when, kind=e.kind, value=e.value, identity=e.identity,
                 what=(e.identity or e.what), episode_uuid=e.episode_uuid)
           for g in groups for e in g.events if e.kind == "acquire"]
    return timeline(evs)


def main():
    from openai import OpenAI
    client = OpenAI(api_key=pt._load_key())
    llm = pt.make_llm(client)
    items = {str(it["question_id"]): it for it in entropy._items()}

    records = []
    for qid in QIDS:
        it = items.get(qid)
        if it is None:
            print(f"  {qid}: not in sample"); continue
        eps = pt._episodes(it)
        ref = str(it["question_date"])[:10].replace("/", "-")
        phrase = _window_phrase(it["question"])
        r_since, r_until = resolve_window(phrase, ref)                  # rolling (strict)
        c_since, c_until = resolve_window(phrase, ref, calendar=True)   # calendar (natural NL)
        rolling, calendar, samples = [], [], []
        for _ in range(K):
            entries = _acquisition_entries(extract_once(eps, llm))
            rolling.append(count_in_window(entries, since=r_since, until=r_until, distinct=True))
            calendar.append(count_in_window(entries, since=c_since, until=c_until, distinct=True))
            samples.append(entries)
        r_mode = collections.Counter(rolling).most_common(1)[0][0]
        c_mode = collections.Counter(calendar).most_common(1)[0][0]
        gold = pt._gold_values(it["answer"])
        hit = "CORRECT" if gold and abs(c_mode - gold[0]) < 0.5 else "miss"
        print(f"  {qid:14s} gold={str(it['answer']):>6s}  window='{phrase}'  "
              f"rolling[{r_since}..{r_until}]={rolling} mode {r_mode}  |  "
              f"calendar[{c_since}..{c_until}]={calendar} mode {c_mode}  -> {hit} (calendar)")
        modal = next(s for s, c in zip(samples, calendar) if c == c_mode)
        print(f"                 acquisitions: {[(e['when'], e['what']) for e in modal]}")
        records.append({"qid": qid, "gold": it["answer"], "phrase": phrase,
                        "rolling": {"since": r_since, "until": r_until, "counts": rolling, "mode": r_mode},
                        "calendar": {"since": c_since, "until": c_until, "counts": calendar, "mode": c_mode},
                        "hit": hit, "acquisitions": [(e["when"], e["what"]) for e in modal]})

    json.dump(records, open(OUT, "w"), indent=2)
    print(f"\nrows -> {OUT}")


if __name__ == "__main__":
    main()
