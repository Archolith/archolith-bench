"""Memory agent: read each AMA trajectory step and write add_memory-style memories where useful.

For every selected episode, a model plays the agent: step by step it sees the task, the memories
it saved so far, and the current action and observation, and returns zero or more memories.
Results are cached per episode in ``<out>/crafted/ep<id>.json`` (resumable: finished steps are
never re-asked). ``ingest_ama.py --crafted-dir`` then ingests only these memories.

Episodes run in parallel; steps within an episode run in order (each step sees earlier memories).
Any 429 (budget cap or upstream rate limit) stops the whole run.

    python scripts/ama/craft_memories.py --dataset .../open_end_qa_set.jsonl \
        --base-url http://127.0.0.1:8765/v1 --model openai/gpt-6-luna --out results/ama-crafted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from archolith_bench.harness.ama_bench import (
    DATASET_ENV,
    DEFAULT_EXCLUDE_DOMAINS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PER_DOMAIN,
    load_episodes,
    memory_agent_messages,
    parse_memories,
    select_episodes,
)

STOP = threading.Event()


class RateLimited(RuntimeError):
    pass


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def ask(client: httpx.Client, base_url: str, api_key: str, model: str, messages: list[dict]) -> str:
    resp = client.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "response_format": {"type": "json_object"}},
        timeout=180.0,
    )
    if resp.status_code == 429:
        raise RateLimited(resp.text[:300])
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"].get("content") or ""


def craft_episode(episode: dict, out_dir: Path, *, base_url: str, api_key: str, model: str) -> dict:
    path = out_dir / f"ep{episode['episode_id']}.json"
    crafted: list[dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    done = {int(e["step"]) for e in crafted}
    memories = [m for e in sorted(crafted, key=lambda e: int(e["step"])) for m in e["memories"]]
    with httpx.Client() as client:
        for step in episode.get("trajectory", []):
            if STOP.is_set():
                break
            idx = int(step.get("turn_idx", 0))
            if idx in done:
                continue
            text = ask(client, base_url, api_key, model, memory_agent_messages(episode, step, memories))
            saved = parse_memories(text)
            # Keep the reply when nothing was saved, so "chose nothing" and "malformed" can be told apart.
            crafted.append({"step": idx, "memories": saved, "raw": None if saved else text[:2000]})
            memories.extend(saved)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(crafted, indent=1), encoding="utf-8")
            os.replace(tmp, path)
    steps = len(episode.get("trajectory", []))
    return {
        "episode_id": episode["episode_id"],
        "domain": episode["domain"],
        "steps": steps,
        "steps_done": len(crafted),
        "memories": sum(len(e["memories"]) for e in crafted),
        "complete": len(crafted) >= steps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.getenv(DATASET_ENV))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=os.getenv("UPSTREAM_API_KEY", "unused"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--episode-ids", type=lambda v: tuple(int(i) for i in _csv(v)), default=None)
    ap.add_argument("--qa-types", type=_csv, default=("C",))
    ap.add_argument("--per-domain", type=int, default=DEFAULT_PER_DOMAIN)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--exclude-domains", type=_csv, default=DEFAULT_EXCLUDE_DOMAINS)
    ap.add_argument("--workers", type=int, default=25)
    args = ap.parse_args(argv)

    if not args.dataset:
        print(f"ERROR: pass --dataset or set {DATASET_ENV}", file=sys.stderr)
        return 2
    episodes = select_episodes(
        load_episodes(args.dataset),
        qa_types=args.qa_types,
        per_domain=args.per_domain,
        max_tokens=args.max_tokens,
        exclude_domains=args.exclude_domains,
        episode_ids=args.episode_ids,
    )
    out_dir = args.out / "crafted"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"crafting memories for {len(episodes)} episodes with {args.model}", flush=True)
    results, failed = [], 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(craft_episode, e, out_dir, base_url=args.base_url, api_key=args.api_key, model=args.model): e
            for e in episodes
        }
        for fut in as_completed(futures):
            try:
                result = fut.result()
                results.append(result)
                print(f"  {result}", flush=True)
            except RateLimited as exc:
                STOP.set()
                failed += 1
                print(f"STOPPING: 429 from {args.base_url}: {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report every episode, then fail the run
                failed += 1
                print(f"  episode {futures[fut]['episode_id']} failed: {exc.__class__.__name__}: {exc}", flush=True)
    complete = sum(r["complete"] for r in results)
    print(f"complete episodes: {complete}/{len(episodes)}; memories: {sum(r['memories'] for r in results)}", flush=True)
    return 0 if complete == len(episodes) and not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
