"""Replay captured LLM requests as A/B arms and compare each arm's answers to the captured original.

Needs a bodies.jsonl written by the budget proxy with --log-bodies (rows carry the original
"messages" and the rest of the "request", e.g. response_format). Send the replays through a plain
budget proxy (no transforms) so every call is capped and priced:

    python scripts/ama/replay_ab.py --bodies run/bodies.jsonl --schema NodeResolutions --n 40 \
        --arm A: --arm B:node_slim --arm C:+effort=none --base-url http://127.0.0.1:8765/v1 --out ab.json

Arm spec: NAME:transform1,transform2[,+effort=LEVEL]. Arm "A" with no transforms is the control;
its disagreement with the captured original is the noise floor for every other arm.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from archolith_bench.ci import prompt_transforms as pt


MAX_TOKENS: dict[str, int] = {}  # arm name -> max_tokens override (+max_tokens=N)


def parse_arm(spec: str) -> tuple[str, tuple[str, ...], str | None]:
    name, _, rest = spec.partition(":")
    names, effort = [], None
    for part in (p.strip() for p in rest.split(",") if p.strip()):
        if part.startswith("+effort="):
            effort = part.split("=", 1)[1]
        elif part.startswith("+max_tokens="):
            MAX_TOKENS[name] = int(part.split("=", 1)[1])
        elif part in pt.TRANSFORMS:
            names.append(part)
        else:
            raise SystemExit(f"unknown transform {part!r} in arm {spec!r}")
    return name, tuple(names), effort


def _json(text):
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def decisions(schema: str, content: str | None):
    """Normalize one response into comparable decision units: dict key -> value."""
    if schema == "text":
        m = re.match(r"\s*([A-Za-z]+)", content or "")
        return {"verdict": m.group(1).upper() if m else None}
    d = _json(content)
    if not isinstance(d, dict):
        return None
    if schema == "EdgeDuplicate":
        return {"dup": tuple(sorted(d.get("duplicate_facts") or [])),
                "contra": tuple(sorted(d.get("contradicted_facts") or []))}
    if schema == "NodeResolutions":
        return {f"e{r.get('id')}": r.get("duplicate_candidate_id") for r in d.get("entity_resolutions") or []}
    if schema == "BatchEdgeTimestamps":
        return {f"f{i}": (t.get("valid_at"), t.get("invalid_at")) for i, t in enumerate(d.get("timestamps") or [])}
    if schema == "EdgeTimestamps":
        return {"f0": (d.get("valid_at"), d.get("invalid_at"))}
    if schema == "CombinedExtraction":
        return {"entities": frozenset((e.get("name") or "").lower() for e in d.get("extracted_entities") or []),
                "n_edges": len(d.get("edges") or [])}
    return {"raw": json.dumps(d, sort_keys=True)}


def agree(schema: str, a, b) -> tuple[int, int]:
    """(matching units, total units) between two normalized decisions."""
    if a is None or b is None:
        return 0, max(len(a or {}), len(b or {}), 1)
    if schema == "CombinedExtraction":
        ea, eb = a["entities"], b["entities"]
        return len(ea & eb), max(len(ea | eb), 1)
    keys = set(a) | set(b)
    return sum(a.get(k) == b.get(k) for k in keys), max(len(keys), 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=Path, required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-empty", action="store_true",
                    help="also replay calls whose captured response was empty (e.g. reasoning ate max_tokens)")
    args = ap.parse_args(argv)
    arms = [parse_arm(a) for a in args.arm]

    rows = [json.loads(line) for line in args.bodies.open(encoding="utf-8")]
    pool = [r for r in rows if r.get("schema") == args.schema and r.get("status") == 200
            and (r.get("response") or args.include_empty) and r.get("request") is not None]
    if not pool:
        raise SystemExit(f"no replayable {args.schema} rows (need a capture with the full 'request' logged)")
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    client = httpx.Client(timeout=180.0)

    def run(item):
        row, (arm, names, effort) = item
        req = {**copy.deepcopy(row["request"]), "messages": copy.deepcopy(row["messages"])}
        pt.apply(req, row["schema"], names, {row["schema"]: effort} if effort else {})
        if arm in MAX_TOKENS:
            req["max_tokens"] = MAX_TOKENS[arm]
        r = client.post(f"{args.base_url}/chat/completions", json=req,
                        headers={"Authorization": "Bearer via-budget-proxy"})
        if r.status_code in (402, 429):
            raise SystemExit(f"STOP: HTTP {r.status_code} {r.text[:300]}")
        body = r.json() if r.status_code == 200 else {}
        usage = body.get("usage") or {}
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        return {"arm": arm, "status": r.status_code, "content": content,
                "in_tok": usage.get("prompt_tokens", 0), "out_tok": usage.get("completion_tokens", 0),
                "reasoning_tok": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)}

    jobs = [(row, arm) for row in sample for arm in arms]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(run, jobs))

    report = {"schema": args.schema, "n": len(sample), "arms": {}}
    for arm, names, effort in arms:
        match = total = in_tok = out_tok = reas = 0
        for (row, (a, _, _)), res in zip(jobs, results):
            if a != arm:
                continue
            m, t = agree(args.schema, decisions(args.schema, row["response"]),
                         decisions(args.schema, res["content"]))
            match, total = match + m, total + t
            in_tok, out_tok, reas = in_tok + res["in_tok"], out_tok + res["out_tok"], reas + res["reasoning_tok"]
        report["arms"][arm] = {"transforms": list(names), "effort": effort,
                               "agreement_with_captured": round(match / max(total, 1), 4),
                               "units": total, "in_tok": in_tok, "out_tok": out_tok, "reasoning_tok": reas}
    report["detail"] = [{"arm": j[1][0], **{k: v for k, v in r.items() if k != "arm"}}
                        for j, r in zip(jobs, results)]
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for arm, s in report["arms"].items():
        print(f"{arm:4s} {'+'.join(s['transforms']) or '-':34s} effort={s['effort'] or '-':5s} "
              f"agree={100 * s['agreement_with_captured']:5.1f}% over {s['units']} units  "
              f"in={s['in_tok']} out={s['out_tok']} reasoning={s['reasoning_tok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
