"""Recall-only LongMemEval evaluation of Menhir's typed recall-packet formats.

The source graph is the completed canonical 78-item knowledge-update ingest.  This
runner never ingests, resets, deletes, or mutates graph state. In ``full`` mode it asks
Recall Lab for the exact inspection packet displayed by the ``Raw LLM packet`` view. In
the default ``query_filtered`` mode it asks production recall to select evidence first,
then consumes the typed, budgeted packet returned by the task packet endpoint.

Results are intentionally NONCANONICAL; production recall and graph state remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

BENCH_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCH_DIR))

from archolith_bench.harness import (  # noqa: E402
    LLMJudgeScorer,
    MemoryCheckpoint,
    ab_result_to_dict,
    checkpoint_path_for,
    get_adapter,
    run_memory_ab,
    write_harness_evidence,
)
from archolith_bench.harness.memory_ab import SINGLE_RECALL  # noqa: E402
from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402

SOURCE_RUN_ID = "scalar-canonical-ku78-v1-20260806"
DEFAULT_RUN_ID = f"{SOURCE_RUN_ID}-typed-packet-v1-rescore"
PACKET_VERSION = "typed-recall-packet/prototype-v1"
QUERY_FILTERED_PACKET_VERSION = "typed-recall-packet/query-filtered-v1"
QUERY_FILTERED_DEFAULT_RUN_ID = f"{SOURCE_RUN_ID}-query-filtered-packet-v1-rescore"
DEFAULT_PROVIDER_BASE_URL = "https://api.openai.com/v1"
FIXTURE_PATH = BENCH_DIR / "fixtures" / "longmemeval" / "knowledge_update_subset.json"
SOURCE_RESULTS_DIR = BENCH_DIR / "results" / "lme-ku-buildout" / SOURCE_RUN_ID
BASELINE_CHECKPOINT = (
    SOURCE_RESULTS_DIR
    / "harness_recall"
    / ".checkpoint_longmemeval-menhir_oracle_gpt-4o.jsonl"
)


class TypedRecallPacketClient(HttpMenhirClient):
    """Read the exact prototype packet from Recall Lab without running retrieval."""

    def __init__(
        self,
        base_url: str,
        *,
        source_run_id: str = SOURCE_RUN_ID,
        api_key: str = "",
    ) -> None:
        super().__init__(base_url, api_key=api_key)
        self._source_run_id = source_run_id

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        del limit  # The packet is one versioned context document, not a ranked result list.
        run_id = quote(self._source_run_id, safe="")
        namespace = quote(group_id, safe="")
        url = (
            self._base_url.rstrip("/")
            + f"/explorer/api/recall-lab/bench-runs/{run_id}/tasks/{namespace}"
        )
        response = self._client.get(url, headers=self._headers)
        response.raise_for_status()
        data = response.json()

        if data.get("contract") != "bench-inspection/v1":
            raise RuntimeError(f"unexpected Recall Lab contract for {group_id!r}")
        if data.get("namespace") != group_id:
            raise RuntimeError(f"Recall Lab returned the wrong namespace for {group_id!r}")
        if data.get("graph_available") is not True:
            raise RuntimeError(f"live graph is unavailable for {group_id!r}")
        returned_question = str(data.get("question") or "").strip()
        if returned_question != query.strip():
            raise RuntimeError(f"fixture/API question mismatch for {group_id!r}")

        packet = (data.get("live_graph") or {}).get("recall_packet") or {}
        if packet.get("version") != PACKET_VERSION:
            raise RuntimeError(f"unexpected packet version for {group_id!r}: {packet.get('version')!r}")
        if packet.get("production_recall_changed") is not False:
            raise RuntimeError(f"prototype boundary drift for {group_id!r}")
        text = str(packet.get("text") or "").strip()
        if not text.startswith(f"MEMORY PACKET {PACKET_VERSION}"):
            raise RuntimeError(f"typed packet text is missing for {group_id!r}")
        return [text]


class QueryFilteredTypedPacketClient(HttpMenhirClient):
    """Use production recall to select evidence, then request its typed packet."""

    def __init__(
        self,
        base_url: str,
        *,
        source_run_id: str = SOURCE_RUN_ID,
        api_key: str = "",
        max_chars: int = 6_000,
        max_general: int = 4,
    ) -> None:
        super().__init__(base_url, api_key=api_key)
        self._source_run_id = source_run_id
        self._max_chars = max_chars
        self._max_general = max_general

    def recall(self, group_id: str, query: str, limit: int = 10) -> list[str]:
        run_id = quote(self._source_run_id, safe="")
        namespace = quote(group_id, safe="")
        url = (
            self._base_url.rstrip("/")
            + f"/explorer/api/recall-lab/bench-runs/{run_id}/tasks/{namespace}/recall-packet"
        )
        response = self._client.post(
            url,
            headers=self._headers,
            json={
                "query": query,
                "limit": limit,
                "max_chars": self._max_chars,
                "max_general": self._max_general,
            },
        )
        response.raise_for_status()
        data = response.json()
        if data.get("contract") != QUERY_FILTERED_PACKET_VERSION:
            raise RuntimeError(f"unexpected query-filtered contract for {group_id!r}")
        if data.get("namespace") != group_id:
            raise RuntimeError(f"Recall Lab returned the wrong namespace for {group_id!r}")
        packet = data.get("packet") or {}
        if packet.get("version") != QUERY_FILTERED_PACKET_VERSION:
            raise RuntimeError(
                f"unexpected query-filtered packet version for {group_id!r}: "
                f"{packet.get('version')!r}"
            )
        if packet.get("production_recall_changed") is not False:
            raise RuntimeError(f"query-filtered boundary drift for {group_id!r}")
        text = str(packet.get("text") or "").strip()
        if not text.startswith(f"MEMORY PACKET {QUERY_FILTERED_PACKET_VERSION}"):
            raise RuntimeError(f"query-filtered packet text is missing for {group_id!r}")
        return [text]


def _load_checkpoint_rows(path: Path, arm: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("arm") != arm or not isinstance(row.get("result"), dict):
                continue
            task_id = str(row.get("task_id") or row["result"].get("task_id") or "")
            if task_id:
                rows[task_id] = row["result"]
    return rows


def _token_totals(rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    answer_input = sum(int(row.get("input_tokens") or 0) for row in rows.values())
    answer_output = sum(int(row.get("output_tokens") or 0) for row in rows.values())
    judge_input = sum(int(row.get("scorer_input_tokens") or 0) for row in rows.values())
    judge_output = sum(int(row.get("scorer_output_tokens") or 0) for row in rows.values())
    return {
        "answer_input": answer_input,
        "answer_output": answer_output,
        "judge_input": judge_input,
        "judge_output": judge_output,
        "total": answer_input + answer_output + judge_input + judge_output,
    }


def build_comparison(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build task-level transitions and token totals for two checkpoint maps."""
    shared = sorted(set(baseline) & set(candidate))
    recovered = [task_id for task_id in shared if not baseline[task_id].get("correct") and candidate[task_id].get("correct")]
    regressed = [task_id for task_id in shared if baseline[task_id].get("correct") and not candidate[task_id].get("correct")]
    stayed_pass = [task_id for task_id in shared if baseline[task_id].get("correct") and candidate[task_id].get("correct")]
    stayed_fail = [task_id for task_id in shared if not baseline[task_id].get("correct") and not candidate[task_id].get("correct")]
    baseline_correct = sum(bool(baseline[task_id].get("correct")) for task_id in shared)
    candidate_correct = sum(bool(candidate[task_id].get("correct")) for task_id in shared)
    n = len(shared)
    return {
        "n": n,
        "baseline": {
            "correct": baseline_correct,
            "score": baseline_correct / n if n else 0.0,
            "tokens": _token_totals({task_id: baseline[task_id] for task_id in shared}),
        },
        "candidate": {
            "correct": candidate_correct,
            "score": candidate_correct / n if n else 0.0,
            "tokens": _token_totals({task_id: candidate[task_id] for task_id in shared}),
        },
        "score_delta": (candidate_correct - baseline_correct) / n if n else 0.0,
        "transitions": {
            "recovered": recovered,
            "regressed": regressed,
            "stayed_pass": stayed_pass,
            "stayed_fail": stayed_fail,
        },
        "missing_from_candidate": sorted(set(baseline) - set(candidate)),
        "missing_from_baseline": sorted(set(candidate) - set(baseline)),
    }


def require_complete_model_evidence(ab: Any, *, expected_items: int) -> None:
    """Reject transport/auth failures before they become a misleading score report."""
    problems: list[str] = []
    for arm_name, arm in ab.arms.items():
        if arm.n != expected_items:
            problems.append(f"{arm_name}: expected {expected_items} rows, got {arm.n}")
        for result in arm.results:
            response = str(result.response_text or "").lstrip()
            if response.startswith(("[ERROR", "[HTTP ERROR]")):
                problems.append(f"{result.task_id}: answer call failed")
            if result.input_tokens <= 0 or result.output_tokens <= 0:
                problems.append(f"{result.task_id}: answer usage is missing")
            if result.scorer_input_tokens <= 0 or result.scorer_output_tokens <= 0:
                problems.append(f"{result.task_id}: judge usage is missing")
    if problems:
        preview = "; ".join(problems[:8])
        suffix = f"; plus {len(problems) - 8} more" if len(problems) > 8 else ""
        raise RuntimeError(
            "typed-packet evaluation has incomplete model evidence; refusing to write a "
            f"quality comparison: {preview}{suffix}"
        )


def _write_comparison_markdown(comparison: dict[str, Any], path: Path) -> None:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    transitions = comparison["transitions"]
    lines = [
        "# Typed recall-packet prototype vs canonical production recall\n\n",
        "This is a **NONCANONICAL recall-only experiment**. It reuses the canonical graph; "
        "no ingestion or graph mutation occurred.\n\n",
        "| Context arm | N | Correct | Score | Answer input | Answer output | Judge input | Judge output | Total tokens |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for label, values in (("Canonical production recall", baseline), ("Typed packet prototype", candidate)):
        tokens = values["tokens"]
        lines.append(
            f"| {label} | {comparison['n']} | {values['correct']} | {values['score']:.3f} | "
            f"{tokens['answer_input']:,} | {tokens['answer_output']:,} | "
            f"{tokens['judge_input']:,} | {tokens['judge_output']:,} | {tokens['total']:,} |\n"
        )
    lines.extend(
        (
            f"\nScore delta: **{comparison['score_delta']:+.3f}**\n\n",
            f"- Recovered ({len(transitions['recovered'])}): "
            f"{', '.join(transitions['recovered']) or 'none'}\n",
            f"- Regressed ({len(transitions['regressed'])}): "
            f"{', '.join(transitions['regressed']) or 'none'}\n",
            f"- Still failed ({len(transitions['stayed_fail'])}): "
            f"{', '.join(transitions['stayed_fail']) or 'none'}\n",
        )
    )
    path.write_text("".join(lines), encoding="utf-8")


def _git_identity(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    menhir_url = os.environ.get("MENHIR_URL", "").rstrip("/")
    if not menhir_url:
        raise SystemExit("MENHIR_URL is required (the read-only Recall Lab instance)")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY is required for the answer and judge calls")
    if not FIXTURE_PATH.exists() or not BASELINE_CHECKPOINT.exists():
        raise SystemExit("canonical fixture or baseline checkpoint is missing")

    answer_model = os.environ.get("LME_ANSWER_MODEL", "gpt-4o")
    judge_model = os.environ.get("LME_JUDGE_MODEL", "gpt-6-luna")
    provider_base_url = os.environ.get(
        "LME_TYPED_PACKET_BASE_URL", DEFAULT_PROVIDER_BASE_URL
    ).rstrip("/")
    limit = int(os.environ.get("LME_TYPED_PACKET_LIMIT", "78"))
    packet_mode = os.environ.get("LME_TYPED_PACKET_MODE", "query_filtered").strip().lower()
    if packet_mode not in {"full", "query_filtered"}:
        raise SystemExit("LME_TYPED_PACKET_MODE must be full or query_filtered")
    default_run_id = QUERY_FILTERED_DEFAULT_RUN_ID if packet_mode == "query_filtered" else DEFAULT_RUN_ID
    run_id = os.environ.get("LME_TYPED_PACKET_RUN_ID", default_run_id)
    results_dir = BENCH_DIR / "results" / "lme-ku-buildout" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()

    adapter = get_adapter("longmemeval-menhir")
    client: HttpMenhirClient
    if packet_mode == "query_filtered":
        client = QueryFilteredTypedPacketClient(
            menhir_url,
            source_run_id=SOURCE_RUN_ID,
            api_key=os.environ.get("MENHIR_API_KEY", ""),
            max_chars=int(os.environ.get("LME_TYPED_PACKET_MAX_CHARS", "6000")),
            max_general=int(os.environ.get("LME_TYPED_PACKET_MAX_GENERAL", "4")),
        )
    else:
        client = TypedRecallPacketClient(
            menhir_url,
            source_run_id=SOURCE_RUN_ID,
            api_key=os.environ.get("MENHIR_API_KEY", ""),
        )
    scorer = LLMJudgeScorer(
        base_url=provider_base_url,
        api_key=openai_key,
        model=judge_model,
    )
    checkpoint = MemoryCheckpoint(
        checkpoint_path_for(results_dir, adapter.benchmark_id, answer_model, variant="oracle")
    )
    print(f"[typed-packet] {checkpoint.done_count(SINGLE_RECALL)} candidate items checkpointed")
    print(
        f"[typed-packet] recall-only mode={packet_mode} limit={limit} "
        f"source={SOURCE_RUN_ID} endpoint={menhir_url}"
    )

    ab = run_memory_ab(
        adapter,
        arms=[SINGLE_RECALL],
        subset="knowledge-update",
        limit=limit,
        fixture_path=str(FIXTURE_PATH),
        client=client,
        model=answer_model,
        chat_base_url=provider_base_url,
        api_key=openai_key,
        recall_limit=1,
        recall_only=True,
        namespace_template="lme-{question_id}",
        checkpoint=checkpoint,
        score_fn=scorer,
    )

    require_complete_model_evidence(ab, expected_items=limit)

    write_harness_evidence(ab, results_dir / "results.md", output_format="markdown")
    (results_dir / "results.json").write_text(
        json.dumps(ab_result_to_dict(ab), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    baseline_rows = _load_checkpoint_rows(BASELINE_CHECKPOINT, SINGLE_RECALL)
    candidate_rows = _load_checkpoint_rows(checkpoint.path, SINGLE_RECALL)
    comparison = build_comparison(baseline_rows, candidate_rows)
    (results_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    _write_comparison_markdown(comparison, results_dir / "comparison.md")

    source_provenance = json.loads((SOURCE_RESULTS_DIR / "run_provenance.json").read_text(encoding="utf-8"))
    provenance = {
        "run_id": run_id,
        "canonical": False,
        "noncanonical": True,
        "recall_only": True,
        "reingested": False,
        "graph_mutated": False,
        "source_run_id": SOURCE_RUN_ID,
        "packet_version": (
            QUERY_FILTERED_PACKET_VERSION if packet_mode == "query_filtered" else PACKET_VERSION
        ),
        "packet_mode": packet_mode,
        "packet_budget": (
            {
                "max_chars": int(os.environ.get("LME_TYPED_PACKET_MAX_CHARS", "6000")),
                "max_general": int(os.environ.get("LME_TYPED_PACKET_MAX_GENERAL", "4")),
            }
            if packet_mode == "query_filtered"
            else None
        ),
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "limit": limit,
        "answer_model": answer_model,
        "judge_model": judge_model,
        "provider_base_url": provider_base_url,
        "fixture": str(FIXTURE_PATH),
        "fixture_sha256": _sha256(FIXTURE_PATH),
        "source_graph_identity": source_provenance.get("identity"),
        "code": {
            "bench": _git_identity(BENCH_DIR),
            "menhir": _git_identity(BENCH_DIR.parent / "menhir"),
        },
        "comparison": comparison,
    }
    (results_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    candidate = comparison["candidate"]
    baseline = comparison["baseline"]
    print(
        f"[typed-packet] baseline={baseline['correct']}/{comparison['n']} "
        f"candidate={candidate['correct']}/{comparison['n']} "
        f"delta={comparison['score_delta']:+.3f}"
    )
    print(f"[typed-packet] evidence written to {results_dir}")


if __name__ == "__main__":
    main()
