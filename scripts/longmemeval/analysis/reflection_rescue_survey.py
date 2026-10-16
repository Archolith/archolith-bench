"""Phase 1 measurement harness: survey all assistant turns in the KU fixture for
recoverable durable claims via reflection rescue.

Runs the teacher model (default gpt-6-luna on Flex; was 4o-mini) on every assistant turn
with its preceding user context.
Outputs a structured JSON with per-turn results so downstream analysis can:
  - Count assistant turns producing ≥1 candidate reflection
  - Measure precision against known gold answers
  - Identify false positives and unsupported claims
  - Inform nano prompt design

Usage:
    # Pilot run (first 30 turns)
    python reflection_rescue_survey.py --pilot 30

    # Full survey (all 915 assistant turns)
    python reflection_rescue_survey.py

    # Resume after interruption
    python reflection_rescue_survey.py --resume

Requires OPENAI_API_KEY in the environment or archolith-bench/.env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Resolve paths relative to this script.
SCRIPT_DIR = Path(__file__).resolve().parent          # analysis/
LME_DIR = SCRIPT_DIR.parent                            # scripts/longmemeval/
BENCH_ROOT = LME_DIR.parents[1]                        # archolith-bench/

# Load OPENAI_API_KEY from Bench's own .env if not already set.
if not os.environ.get("OPENAI_API_KEY"):
    env_path = BENCH_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                break

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment or archolith-bench/.env", file=sys.stderr)
    sys.exit(1)

from openai import AsyncOpenAI  # noqa: E402

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
from archolith_bench.core.api import acall_with_flex_retry, openai_chat_kwargs  # noqa: E402

DEFAULT_MODEL = "gpt-6-luna"

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

FIXTURE_PATH = BENCH_ROOT / "fixtures" / "longmemeval" / "knowledge_update_subset.json"

# Default output directory (alongside other KU buildout results).
OUTPUT_DIR = BENCH_ROOT / "results" / "lme-ku-buildout" / "reflection-rescue-survey"


@dataclass
class AssistantTurn:
    """An assistant turn with its surrounding context."""
    question_id: str
    session_idx: int
    turn_idx: int           # index within the session
    content: str
    has_answer: bool        # from fixture gold labels
    preceding_user_turns: list[dict] = field(default_factory=list)
    # Each dict: {"role": "user", "content": "...", "has_answer": bool}


@dataclass
class ReflectionResult:
    """Result of running the reflection rescue prompt on one assistant turn."""
    question_id: str
    session_idx: int
    turn_idx: int
    has_answer: bool                     # fixture gold label
    reflections: list[dict] = field(default_factory=list)
    # Each dict from the LLM structured output:
    # {
    #   "claim": str,
    #   "supporting_turn_index": int,    # index into preceding_user_turns
    #   "supporting_quote": str,
    #   "support_type": str,             # direct_paraphrase | close_paraphrase | inference | computation | unsupported
    #   "origin": str,                   # user_reflection | tool_reflection | deterministic_computation | assistant_inference | unsupported
    #   "confidence": float,
    #   "needs_review": bool
    # }
    raw_response: str = ""
    error: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def load_assistant_turns(fixture_path: Path) -> list[AssistantTurn]:
    """Extract all assistant turns with their preceding user context."""
    items = json.loads(fixture_path.read_text(encoding="utf-8"))
    turns: list[AssistantTurn] = []

    for item in items:
        qid = str(item["question_id"])
        for s_idx, session in enumerate(item.get("haystack_sessions", [])):
            # Collect preceding user turns as we go.
            preceding_users: list[dict] = []
            for t_idx, turn in enumerate(session):
                role = turn.get("role", "")
                content = turn.get("content", "")
                has_answer = turn.get("has_answer", False)

                if role == "user":
                    preceding_users.append({
                        "role": "user",
                        "content": content,
                        "has_answer": has_answer,
                        "turn_idx": t_idx,
                    })
                elif role == "assistant":
                    # Take last 4 preceding user turns as context window.
                    context = list(preceding_users[-4:])
                    turns.append(AssistantTurn(
                        question_id=qid,
                        session_idx=s_idx,
                        turn_idx=t_idx,
                        content=content,
                        has_answer=has_answer,
                        preceding_user_turns=context,
                    ))

    return turns


# ---------------------------------------------------------------------------
# Reflection rescue prompt
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """\
You are analyzing an assistant's response to determine if it reflects any concrete, \
durable facts that the user stated in their preceding messages.

A "reflection" is when the assistant restates, paraphrases, summarizes, or computes \
from a fact the user provided. We want to identify these so the underlying user fact \
can be stored as a long-term memory.

## Rules

1. Only extract claims that are **concrete, durable facts** — things worth remembering \
long-term (personal details, life events, preferences, states, changes, achievements, \
relationships, locations, jobs, etc.)
2. Each claim must be **directly supported** by something the user said in the preceding \
turns. Cite the exact supporting quote.
3. Do NOT extract:
   - Generic advice the assistant is giving
   - Questions the assistant asks
   - Hypotheticals or suggestions
   - Claims where the assistant is making a new inference not stated by the user
   - Transient conversational filler
4. Classify each reflection's origin:
   - `user_reflection`: assistant restates something the user said
   - `tool_reflection`: assistant restates tool/API output
   - `deterministic_computation`: assistant computed from user-provided inputs (e.g., date math)
   - `assistant_inference`: assistant's own conclusion beyond what user said
   - `unsupported`: no supporting evidence in preceding turns
5. Classify support type:
   - `direct_paraphrase`: nearly verbatim restatement
   - `close_paraphrase`: same fact, different words
   - `inference`: requires a reasoning step (e.g., date subtraction)
   - `computation`: arithmetic or date calculation
   - `unsupported`: no match found
6. Set `needs_review: true` when:
   - The claim combines multiple user turns
   - Pronoun resolution is uncertain
   - Arithmetic or date reasoning is required
   - The assistant generalizes beyond the user's exact words
   - Multiple candidate source quotes exist
7. Set `confidence` (0.0-1.0) based on how directly the user evidence supports the claim.
8. If no durable reflections exist, return an empty `reflections` array.

## Preceding user turns

{user_context}

## Assistant response to analyze

{assistant_content}

## Output format

Respond with ONLY a valid JSON object:
{{"reflections": [{{"claim": "...", "supporting_turn_index": 0, "supporting_quote": "...", "support_type": "direct_paraphrase|close_paraphrase|inference|computation|unsupported", "origin": "user_reflection|tool_reflection|deterministic_computation|assistant_inference|unsupported", "confidence": 0.95, "needs_review": false}}]}}"""


def _format_user_context(preceding: list[dict]) -> str:
    """Format preceding user turns for the prompt."""
    if not preceding:
        return "(No preceding user turns available)"
    parts = []
    for i, turn in enumerate(preceding):
        parts.append(f"[Turn {i}] User:\n{turn['content']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

async def run_reflection_rescue(
    client: AsyncOpenAI,
    turn: AssistantTurn,
    model: str = DEFAULT_MODEL,
) -> ReflectionResult:
    """Call the reflection rescue prompt on one assistant turn."""
    user_context = _format_user_context(turn.preceding_user_turns)
    prompt = REFLECTION_PROMPT.format(
        user_context=user_context,
        assistant_content=turn.content,
    )

    result = ReflectionResult(
        question_id=turn.question_id,
        session_idx=turn.session_idx,
        turn_idx=turn.turn_idx,
        has_answer=turn.has_answer,
    )

    t0 = time.time()
    try:
        resp = await acall_with_flex_retry(lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **openai_chat_kwargs(model, max_tokens=2048, temperature=0),
        ))
        result.latency_ms = int((time.time() - t0) * 1000)
        result.prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        result.completion_tokens = resp.usage.completion_tokens if resp.usage else 0

        raw = resp.choices[0].message.content.strip()
        result.raw_response = raw
        data = json.loads(raw)
        result.reflections = data.get("reflections", [])

    except Exception as exc:
        result.latency_ms = int((time.time() - t0) * 1000)
        result.error = f"{exc.__class__.__name__}: {exc}"

    return result


# ---------------------------------------------------------------------------
# Batch runner with concurrency control
# ---------------------------------------------------------------------------

async def run_batch(
    turns: list[AssistantTurn],
    *,
    model: str = DEFAULT_MODEL,
    concurrency: int = 10,
    output_path: Path | None = None,
    resume_from: dict | None = None,
) -> list[ReflectionResult]:
    """Run reflection rescue on a batch of assistant turns with rate limiting."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[ReflectionResult] = []

    # Build set of already-done turn keys for resume.
    done_keys: set[str] = set()
    if resume_from:
        for r in resume_from.get("results", []):
            key = f"{r['question_id']}:{r['session_idx']}:{r['turn_idx']}"
            done_keys.add(key)
            results.append(ReflectionResult(**{
                k: v for k, v in r.items()
                if k in ReflectionResult.__dataclass_fields__
            }))
        print(f"RESUME: {len(done_keys)} turns already processed, skipping", flush=True)

    remaining = [
        t for t in turns
        if f"{t.question_id}:{t.session_idx}:{t.turn_idx}" not in done_keys
    ]

    if not remaining:
        print("All turns already processed.", flush=True)
        return results

    completed = len(done_keys)
    total = len(turns)
    t_start = time.time()

    async def process_one(turn: AssistantTurn) -> ReflectionResult:
        nonlocal completed
        async with semaphore:
            result = await run_reflection_rescue(client, turn, model=model)
            completed += 1
            n_reflections = len(result.reflections)
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed else 0
            eta = (total - completed) / rate if rate else 0
            status = "ERR" if result.error else f"{n_reflections}r"
            ha = " [HAS_ANSWER]" if turn.has_answer else ""
            print(
                f"[{completed}/{total}] {turn.question_id} s{turn.session_idx}t{turn.turn_idx} "
                f"{status} {result.latency_ms}ms{ha} eta={eta:.0f}s",
                flush=True,
            )
            return result

    # Process in order but with concurrency.
    tasks = [process_one(t) for t in remaining]
    new_results = await asyncio.gather(*tasks)
    results.extend(new_results)

    # Sort by fixture order for consistency.
    results.sort(key=lambda r: (r.question_id, r.session_idx, r.turn_idx))

    # Write incremental output.
    if output_path:
        _write_output(results, turns, model, output_path)

    return results


def _write_output(
    results: list[ReflectionResult],
    turns: list[AssistantTurn],
    model: str,
    output_path: Path,
) -> None:
    """Write survey results with summary statistics."""
    total = len(results)
    with_reflections = sum(1 for r in results if r.reflections)
    total_reflections = sum(len(r.reflections) for r in results)
    errors = sum(1 for r in results if r.error)
    has_answer_turns = [r for r in results if r.has_answer]
    has_answer_with_reflections = sum(1 for r in has_answer_turns if r.reflections)

    # Cost estimation.
    total_prompt_tokens = sum(r.prompt_tokens for r in results)
    total_completion_tokens = sum(r.completion_tokens for r in results)
    # Only gpt-4o-mini has a known rate here ($0.15/M input, $0.60/M output); any other
    # model is reported unpriced rather than costed at mini's rate.
    if model == "gpt-4o-mini":
        total_cost = (total_prompt_tokens * 0.15 + total_completion_tokens * 0.60) / 1_000_000
    else:
        total_cost = None

    # Confidence distribution.
    all_confidences = [
        ref.get("confidence", 0)
        for r in results
        for ref in r.reflections
    ]

    # Origin distribution.
    origin_counts: dict[str, int] = {}
    support_type_counts: dict[str, int] = {}
    needs_review_count = 0
    for r in results:
        for ref in r.reflections:
            origin = ref.get("origin", "unknown")
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            st = ref.get("support_type", "unknown")
            support_type_counts[st] = support_type_counts.get(st, 0) + 1
            if ref.get("needs_review"):
                needs_review_count += 1

    avg_latency = sum(r.latency_ms for r in results) / total if total else 0

    output = {
        "survey_metadata": {
            "model": model,
            "fixture": str(FIXTURE_PATH.name),
            "total_assistant_turns": total,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": {
            "turns_with_reflections": with_reflections,
            "turns_without_reflections": total - with_reflections - errors,
            "total_reflections": total_reflections,
            "avg_reflections_per_positive_turn": (
                total_reflections / with_reflections if with_reflections else 0
            ),
            "reflection_rate": with_reflections / total if total else 0,
            "errors": errors,
            "has_answer_turns": len(has_answer_turns),
            "has_answer_recovered": has_answer_with_reflections,
        },
        "cost": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "estimated_cost_usd": round(total_cost, 4) if total_cost is not None else None,
        },
        "latency": {
            "avg_ms": round(avg_latency),
            "min_ms": min((r.latency_ms for r in results), default=0),
            "max_ms": max((r.latency_ms for r in results), default=0),
        },
        "distributions": {
            "origin_counts": origin_counts,
            "support_type_counts": support_type_counts,
            "needs_review_count": needs_review_count,
            "confidence_buckets": {
                "0.0-0.5": sum(1 for c in all_confidences if c < 0.5),
                "0.5-0.7": sum(1 for c in all_confidences if 0.5 <= c < 0.7),
                "0.7-0.9": sum(1 for c in all_confidences if 0.7 <= c < 0.9),
                "0.9-1.0": sum(1 for c in all_confidences if c >= 0.9),
            },
        },
        "results": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    print(f"  Turns with reflections: {with_reflections}/{total} ({with_reflections/total*100:.1f}%)", flush=True)
    print(f"  Total reflections: {total_reflections}", flush=True)
    print(f"  Avg reflections/positive turn: {total_reflections/with_reflections:.1f}" if with_reflections else "", flush=True)
    print(f"  has_answer recovered: {has_answer_with_reflections}/{len(has_answer_turns)}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    cost_text = f"${total_cost:.4f}" if total_cost is not None else f"unpriced for {model}"
    print(f"  Cost: {cost_text} ({total_prompt_tokens} prompt + {total_completion_tokens} completion tokens)", flush=True)
    print(f"  Avg latency: {avg_latency:.0f}ms", flush=True)
    print(f"  Origins: {origin_counts}", flush=True)
    print(f"  Support types: {support_type_counts}", flush=True)
    print(f"  Needs review: {needs_review_count}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reflection rescue survey — Phase 1 measurement")
    ap.add_argument(
        "--pilot", type=int, default=0,
        help="Run only the first N assistant turns (0 = all)",
    )
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model to use for reflection extraction (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API calls (default: 10)",
    )
    ap.add_argument(
        "--output", default=None,
        help="Output JSON path (default: results/lme-ku-buildout/reflection-rescue-survey/survey-<model>.json)",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
    )
    ap.add_argument(
        "--fixture", default=str(FIXTURE_PATH),
        help=f"Fixture path (default: {FIXTURE_PATH.name})",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        print(f"ERROR: Fixture not found: {fixture_path}", file=sys.stderr)
        return 1

    print(f"Loading assistant turns from {fixture_path.name}...", flush=True)
    all_turns = load_assistant_turns(fixture_path)
    print(f"Loaded {len(all_turns)} assistant turns", flush=True)

    if args.pilot > 0:
        turns = all_turns[:args.pilot]
        print(f"PILOT MODE: using first {len(turns)} turns", flush=True)
    else:
        turns = all_turns

    # Output path.
    model_slug = args.model.replace("/", "-").replace(":", "-")
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"-pilot{args.pilot}" if args.pilot else ""
        output_path = OUTPUT_DIR / f"survey-{model_slug}{suffix}.json"

    # Resume support.
    resume_data = None
    if args.resume and output_path.exists():
        print(f"Loading resume data from {output_path}...", flush=True)
        resume_data = json.loads(output_path.read_text(encoding="utf-8"))

    print(f"Model: {args.model}", flush=True)
    print(f"Concurrency: {args.concurrency}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(flush=True)

    results = asyncio.run(run_batch(
        turns,
        model=args.model,
        concurrency=args.concurrency,
        output_path=output_path,
        resume_from=resume_data,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
