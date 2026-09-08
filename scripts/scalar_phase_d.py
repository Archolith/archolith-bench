#!/usr/bin/env python3
"""Phase D counterfactual recall runner -- connect to a LIVE throwaway menhir (scalar scheduler ON),
seed the Phase D episodes (stale predecessors + current values), wait for Views to materialize, then
score BASELINE recall vs a counterfactual VIEW-AWARE answer per current-state question.

Nothing in production recall changes: the View-aware answer is composed offline in the harness. Run
against a stack brought up by `SS_PHASE_D=1 bash scripts/run_scalar_state_e2e.sh --keep`, or point it at
any throwaway with `--menhir-url` + `--neo4j-uri`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from archolith_bench.harness.memory_ab import assert_not_production
from archolith_bench.harness.menhir_client import HttpMenhirClient
from archolith_bench.harness.scalar_bolt import assert_not_prod
from archolith_bench.harness.scalar_phase_d import (
    PhaseDBoltReader,
    phase_d_cases,
    phase_d_result_to_dict,
    run_phase_d,
    _seed_prompts,
)


def _fmt(b) -> str:
    return "yes" if b else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--menhir-url", required=True)
    ap.add_argument("--neo4j-uri", required=True)
    ap.add_argument("--neo4j-user", default="neo4j")
    ap.add_argument("--neo4j-password", default="scalarthrowaway")
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--max-wait-s", type=float, default=150.0)
    ap.add_argument("--target-views", type=int, default=3,
                    help="expected current-state View slots; wait for this many before scoring")
    ap.add_argument("--settle-s", type=float, default=12.0,
                    help="once target is reached OR growth stalls, wait this long with NO new View "
                         "before scoring, so a slot that materializes late is never read as absent")
    ap.add_argument("--out", default=None, help="write the full JSON result here")
    args = ap.parse_args()

    # Both targets are written to (reset + seed), so both are guarded at
    # the point the arguments resolve, before any client is constructed.
    assert_not_production(args.menhir_url)
    assert_not_prod(args.neo4j_uri)

    ns = args.namespace or f"pd-{int(time.time())}"
    cases = phase_d_cases()
    client = HttpMenhirClient(args.menhir_url)
    bolt = PhaseDBoltReader(args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)

    print(f"== Phase D counterfactual recall ==\n  menhir={args.menhir_url}  bolt={args.neo4j_uri}  ns={ns}\n")

    # 0. clean slate + 1. seed stale->current episodes -----------------------------------------
    client.reset(ns)
    for body in _seed_prompts(cases):
        ev = client.record_turn_evidence(ns, body, turn_key=f"{ns}:{body[:24]}")
        turn_uuid = ev.get("turn_id") or ev.get("turn_evidence_uuid")
        client.ingest(ns, "user", body, source="user", turn_evidence_uuid=turn_uuid, wait=True)
    print(f"  seeded {len(_seed_prompts(cases))} episodes (stale predecessors + current values)")

    # 2. wait for the async scheduler to materialize Views, THEN settle -------------------------
    # Scoring mid-materialization would read a not-yet-folded slot as absent OR catch a transient
    # value, so wait until the View count reaches the target OR stops growing, then hold for `settle-s`
    # with NO new View before scoring. This makes the counterfactual read a STABLE graph.
    start = time.monotonic()
    deadline = start + args.max_wait_s
    views: list = []
    last_count = -1
    stable_since = None
    while time.monotonic() < deadline:
        views = bolt.read_scalar_views(ns)
        now = time.monotonic()
        if len(views) != last_count:
            last_count = len(views)
            stable_since = now  # growth -> reset the settle window
        reached = len(views) >= args.target_views
        settled = stable_since is not None and (now - stable_since) >= args.settle_s
        if reached and settled:
            break
        if settled and len(views) > 0:
            break  # growth stalled below target (stochastic yield) but stable -> score what we have
        time.sleep(3.0)
    waited = round(time.monotonic() - start, 1)
    print(f"  {len(views)} Views materialized + settled after {waited}s\n")

    # 3. score (episodes already seeded -> ingest=False; the driver reads Views + runs recall) --
    result = run_phase_d(client, bolt, cases=cases, namespace=ns, ingest=False)

    # ---- report ----
    print(f"{'case_id':22} {'kind':11} {'detect':7} {'slot':5} {'view':8} {'stale_b':8} "
          f"{'improv':7} {'wrongf':7} answer")
    print("-" * 100)
    for q in result.questions:
        print(f"{q.case_id:22} {q.kind:11} {_fmt(q.current_state_detected):7} {_fmt(q.slot_overlap):5} "
              f"{q.view_status:8} {_fmt(q.stale_in_baseline):8} {_fmt(q.answer_improved):7} "
              f"{_fmt(q.wrongful_suppression):7} {str(q.view_aware_answer)[:32]!r}")

    m = result.metrics
    p = m["primary_over_correct_views"]
    c = m["coverage_all_questions"]
    print("\n-- PRIMARY (over view_status==correct rows only) --")
    print(f"   correct-View rows:      {p['correct_view_rows']}")
    print(f"   answer improved:        {p['answer_improved']}/{p['correct_view_rows']}  "
          f"(rate {p['answer_improved_rate']})")
    print(f"   wrongful suppression:   {p['wrongful_suppression']}")
    print("\n-- END-TO-END COVERAGE (all questions) --")
    print(f"   total questions:        {c['total_questions']}")
    for k in ("current_state_detected", "subject_resolved", "slot_overlap", "view_correct",
              "stale_in_baseline", "answer_improved"):
        print(f"   {k:24} {c[k]}/{c['total_questions']}")
    print(f"\n-- CONTROLS -- clean={m['controls_clean']} violations={m['control_violations']}")
    if result.warnings:
        print(f"\n-- warnings --\n   " + "\n   ".join(result.warnings))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(phase_d_result_to_dict(result), f, indent=2, ensure_ascii=False)
        print(f"\nfull result -> {args.out}")

    bolt.close()
    # exit non-zero if a control was violated (a real Phase D failure), else 0
    sys.exit(1 if not m["controls_clean"] else 0)


if __name__ == "__main__":
    main()
