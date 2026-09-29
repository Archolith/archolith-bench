"""Run Bench's budget-capped proxy in the foreground for a paid AMA-Bench run.

Point every chat client (Menhir's LLM, the answer model, the judge) at
``http://127.0.0.1:<port>/v1``. The proxy holds the real key, prices each response from its
reported usage, and answers 429 to everything once ``--max-usd`` is reached.

    OPENROUTER_API_KEY=... python scripts/ama/serve_budget_proxy.py \
        --max-usd 1 --price-input 0.10 --price-output 0.50 --out results/ama-smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--price-input", type=float, required=True, help="USD per 1M input tokens")
    ap.add_argument("--price-output", type=float, required=True, help="USD per 1M output tokens")
    ap.add_argument("--upstream", default="https://openrouter.ai/api")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-calls", type=int, default=20_000)
    ap.add_argument("--max-seconds", type=float, default=6 * 3600)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    key = os.getenv(args.key_env, "")
    if not key:
        print(f"ERROR: {args.key_env} is not set", file=sys.stderr)
        return 2
    # BudgetState reads its prices from these at import time.
    os.environ["BENCH_PRICE_INPUT_PER_1M"] = str(args.price_input)
    os.environ["BENCH_PRICE_OUTPUT_PER_1M"] = str(args.price_output)
    from archolith_bench.ci.budget_proxy import BudgetProxy

    args.out.mkdir(parents=True, exist_ok=True)
    proxy = BudgetProxy(
        api_key=key,
        upstream=args.upstream,
        port=args.port,
        trace_file=args.out / "budget_trace.jsonl",
        budget_file=args.out / "budget.json",
        max_calls=args.max_calls,
        max_usd=args.max_usd,
        max_seconds=args.max_seconds,
    )
    proxy.start()
    print(f"budget proxy on {proxy.base_url}/v1 -> {args.upstream} cap=${args.max_usd}", flush=True)
    reported = False
    try:
        while True:
            time.sleep(5)
            reason = proxy.enforce_caps() or proxy.kill_reason()
            if reason and not reported:
                print(f"CAP REACHED: {reason} (now refusing every call)", flush=True)
                reported = True
    except KeyboardInterrupt:
        pass
    finally:
        proxy.state.write_budget()
        proxy.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
