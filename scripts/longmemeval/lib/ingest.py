"""Throwaway: RESUMABLE ingest of LongMemEval haystacks into a PERSISTENT menhir/Neo4j under
STABLE namespaces (lme-<question_id>), no per-item reset on success -- so recall-only A/B
(main vs frontier) can run against the pre-ingested graph without re-ingesting.

Resumability:
- An incremental manifest (results/lme-ingest/manifest.json) records each FULLY-ingested AND
  FULLY-ENRICHED item.
- On restart, items already in the manifest are SKIPPED.
- An item NOT in the manifest may have partial data from a prior crash, so its namespace is
  force-reset before re-ingesting. The reset removes the graph partition, scalar state, and
  namespace-keyed TurnEvidence -- guaranteeing no duplicate evidence or episodes.
- Per-turn ingest has bounded retry/backoff; if it still fails the script exits non-zero and a
  re-run resumes from the manifest.

Enrichment completeness (why a namespace-window DRAIN exists):
- The HTTP ingest `wait=true` is best-effort with a 60s/episode cap, so slow episodes (which need
  up to MENHIR_MAX_LLM_CALLS_PER_JOB extraction calls) leave a PENDING/ENRICHING backlog behind a
  single background worker. Marking an item "done" the instant its last turn POST returns would
  record a HALF-ENRICHED namespace -- recall would then run against an incomplete graph.
- A bounded window keeps exactly one active episode per namespace. When one reaches a terminal
  state, the driver submits that namespace's next chronological episode. Menhir can therefore
  enrich distinct namespaces concurrently without ever claiming later work from the same namespace.
- After submitting a window we DRAIN: poll until menhir's queue_depth==0 (authoritative
  unprocessed-work count, from /api/stats) AND no episode in any window namespace is PENDING or
  ENRICHING (the two non-terminal lifecycle states, read from Neo4j) -- stable across two polls --
  before writing the manifest. This makes "manifest done" mean "fully enriched", and keeps resume
  correct. (Gating on ENRICHING alone let the drain settle while a PENDING backlog the worker had
  not yet claimed still existed, which then slipped an unenriched namespace into scalar
  consolidation and aborted the run.)
- Benchmark mode disables the scheduler, so FAILED episodes are never auto-retried. The window
  scheduler immediately retries submitted episodes once before advancing that namespace. After
  drain, a bounded sweep retries generated evidence-projection episodes once; any residual failure
  is recorded in the final manifest and blocks scalar consolidation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Derive real Windows paths from <repo>/scripts/longmemeval/lib/ingest.py.
# parents[3] is the repository root; the old parents[1] value silently placed the
# standalone default manifest under scripts/longmemeval instead of results/lme-ingest.
# Environment-provided manifest and fixture paths remain explicit overrides.
BENCH_ROOT = Path(__file__).resolve().parents[3]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
LIB_ROOT = Path(__file__).resolve().parent
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

DEFAULT_MANIFEST = os.getenv(
    "LME_MANIFEST_PATH",
    str(BENCH_ROOT / "results" / "lme-ingest" / "manifest.json"),
)


import httpx  # noqa: E402
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter  # noqa: E402
from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402

from claim_segmenter import (  # noqa: E402
    ClaimSegment,
    SegmentationMode,
    _heuristic_extract_claims,
    segmentation_mode as decide_segmentation,
)


def _heuristic_extract_claims_sync(content: str, role: str) -> list[ClaimSegment]:
    """Synchronous wrapper for Stage B heuristic claim extraction."""
    return _heuristic_extract_claims(content, role)

# Throwaway lme Neo4j (must match _lme_build_db.sh).
NEO4J_CONTAINER = os.getenv("LME_NEO4J_CONTAINER", "menhir-lme-neo4j")
NEO4J_PW = os.getenv("LME_NEO4J_PW", "lmedata123")
REQUIRE_TURN_EVIDENCE = os.getenv("LME_REQUIRE_TURN_EVIDENCE", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

# Sentence splitting — long multi-topic messages bury factual claims in geographic
# noise, causing Graphiti's node resolution to collapse distinct entities (e.g.
# "the suburbs" into "Chicago").  Splitting isolates each claim so extraction
# sees one topic per episode.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_SPLIT_MIN_LENGTH = int(os.getenv("LME_SPLIT_MIN_LENGTH", "100"))


@dataclass(frozen=True)
class IngestTurn:
    role: str
    content: str
    occurred_at: str | None
    session_id: str
    # False for CONTEXT_ONLY turns: capture the turn as evidence, create no episode.
    # Never merge such a turn into a neighbour -- see _iter_item_turns.
    extract: bool = True
    # The ORIGINAL source turn's full text. ``content`` may be one segment of it (sentence
    # splitting, SEGMENT_CLAIMS); evidence always records the whole turn, once.
    evidence_text: str = ""
    # Stable identity of the source turn this came from: namespace + session + turn index.
    # Every segment of one source turn carries the same key, and it is what menhir's
    # :TurnEvidence MERGE keys on. Without it menhir falls back to hashing the TEXT, which
    # collapses two genuinely repeated identical turns onto one evidence node.
    turn_key: str = ""


@dataclass
class WindowItem:
    item: dict
    question_id: str
    namespace: str
    turns: int = 0


def _split_sentences(text: str, min_length: int = _SPLIT_MIN_LENGTH) -> list[str]:
    """Split *text* on sentence boundaries when it exceeds *min_length* chars.

    Returns a list of non-empty stripped strings.  Short texts return as a
    single-element list so the caller can always iterate.
    """
    if len(text) <= min_length:
        return [text]
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _parse_lme_date(date_str: str | None) -> str | None:
    """Parse a LongMemEval haystack_date string to an ISO-8601 UTC string.

    LME format: "2023/07/14 (Fri) 08:30".  Returns None on any parse failure
    so callers fall back to menhir's default (ingestion time).
    """
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
        return dt.replace(tzinfo=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def _ingest_turn(
    client: HttpMenhirClient,
    ns: str,
    role: str,
    content: str,
    *,
    tries: int = 4,
    occurred_at: str | None = None,
    session_id: str | None = None,
    extract: bool = True,
    evidence_text: str | None = None,
    turn_key: str | None = None,
    evidence_cache: dict[str, str] | None = None,
) -> str:
    """Ingest one turn with bounded exponential backoff on transient errors.

    For user turns, records :TurnEvidence first so the ``source="user"`` claim
    passes Menhir's admission gate (requires a grounding UUID).  Without it the
    gate downgrades the claim to ``agent_inference`` and writes a noisy
    admission-denial entity.

    EXACTLY ONE :TurnEvidence PER SOURCE TURN. Two things would otherwise break that:

    * ``turn_key`` is the identity menhir MERGEs evidence on. Supplied with none, it
      derives one by hashing the TEXT, so a user who says the same thing twice in a
      session ("I have 20 coins") collapses onto a single evidence node and the second
      observation disappears. The caller therefore supplies a key built from the
      namespace, session and source-turn index, which is distinct per genuine turn and
      stable across retries of the same turn.
    * ``content`` may be one segment of a turn (sentence splitting, SEGMENT_CLAIMS).
      Evidence records ``evidence_text`` -- the whole original turn -- and
      ``evidence_cache`` makes the first segment record it while every later segment of
      the same turn reuses that UUID. So N episodes from one turn all cite ONE evidence
      node holding the user's actual words, rather than N evidence nodes holding
      fragments of them.

    ``extract=False`` is the evidence-only path for CONTEXT_ONLY turns: the turn is still
    captured as exactly one :TurnEvidence record carrying its own text and its own
    ``role``/``declarant``, but no :Episodic node is created, so nothing is sent to
    extraction and the returned episode uuid is empty. Turns are never combined --
    evidence boundaries and per-turn counts stay identical to the source transcript, and
    an assistant turn can never be captured under ``declarant="user"`` (menhir's evidence
    projection is fail-closed on ``role='user' AND declarant='user'``, so assistant
    evidence stays inert rather than founding a user-declared claim).

    Uses wait=False: the namespace-window scheduler is the completeness
    guarantee, so we do not block on an HTTP request per turn. It still keeps
    at most one episode active per namespace while allowing different
    namespaces to use multiple enrichment workers.
    """
    normalized_role = role.strip().lower() or "user"
    is_user = normalized_role == "user"
    turn_evidence_uuid: str | None = None
    # The evidence body is the ORIGINAL turn, never the segment being ingested.
    evidence_body = content if evidence_text is None else evidence_text

    # A sibling segment of the same source turn already recorded the evidence.
    if evidence_cache is not None and turn_key:
        turn_evidence_uuid = evidence_cache.get(turn_key)

    # Ground the user-tier claim by recording turn evidence first. Evidence-only
    # turns are captured for every role, since they have no episode to ground and
    # the record is the only trace the source turn leaves.
    if turn_evidence_uuid is None and (is_user or not extract):
        for attempt in range(tries):
            try:
                ev = client.record_turn_evidence(
                    ns,
                    evidence_body,
                    role=normalized_role,
                    declarant=normalized_role,
                    session_id=session_id,
                    occurred_at=occurred_at,
                    turn_key=turn_key,
                )
                turn_evidence_uuid = ev.get("turn_id")
                if turn_evidence_uuid and evidence_cache is not None and turn_key:
                    evidence_cache[turn_key] = turn_evidence_uuid
                break
            except (httpx.HTTPError,) as exc:
                if attempt == tries - 1:
                    if REQUIRE_TURN_EVIDENCE:
                        raise RuntimeError(
                            f"turn-evidence capture failed after {tries} attempts"
                        ) from exc
                    # Legacy compatibility for non-scalar diagnostic builds.
                    print(f"    turn-evidence failed after {tries} attempts: {exc.__class__.__name__}; "
                          "ingesting without grounding", flush=True)
                wait = 2 ** (attempt + 1)
                print(f"    turn-evidence retry {attempt+1}/{tries} after {exc.__class__.__name__} -> sleep {wait}s", flush=True)
                time.sleep(wait)

    if REQUIRE_TURN_EVIDENCE and (is_user or not extract) and not turn_evidence_uuid:
        raise RuntimeError(
            "turn-evidence capture returned no turn_id; refusing to continue without "
            f"grounding for namespace {ns}"
        )

    # Evidence-only: the turn is recorded, but there is no episode to enrich.
    if not extract:
        return ""

    for attempt in range(tries):
        try:
            result = client.ingest(
                ns,
                role,
                content,
                occurred_at=occurred_at,
                session_id=session_id,
                # A user's own utterance is external testimony -> "user" is a Guard-5 anchor kind,
                # so facts the user stated survive the EvidenceAnchorWarden. Assistant/system turns
                # stay unanchored (default "remote-api" -> agent_inference).
                source=("user" if is_user else None),
                turn_evidence_uuid=turn_evidence_uuid,
                wait=False,
            )
            episode_uuid = str((result or {}).get("episode_id") or "")
            if not episode_uuid:
                raise RuntimeError(f"menhir did not return an episode id for namespace {ns}")
            return episode_uuid
        except (httpx.HTTPError,) as exc:
            if attempt == tries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    ingest retry {attempt+1}/{tries} after {exc.__class__.__name__} -> sleep {wait}s", flush=True)
            time.sleep(wait)


def _cypher(query: str) -> list[list[str]]:
    """Run a read-only cypher query against the lme Neo4j via docker exec; return data rows
    (header dropped). Returns [] on any failure (docker missing / neo4j unreachable)."""
    try:
        proc = subprocess.run(
            ["docker", "exec", NEO4J_CONTAINER, "cypher-shell", "-u", "neo4j", "-p", NEO4J_PW,
             "--format", "plain", query],
            capture_output=True, text=True, timeout=300,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    rows = []
    for ln in lines[1:]:  # drop header
        rows.append([c.strip().strip('"') for c in ln.split(", ")])
    return rows


def _ns_state_counts(ns: str) -> dict[str, int]:
    """Per-namespace processing-state counts from Neo4j. PENDING and ENRICHING are the two
    non-terminal lifecycle states (in-flight); READY and FAILED are terminal (see menhir
    episode_lifecycle.mark_episode_ready/_failed). Episodic nodes with a NULL processing_state are
    graphiti-derived resolved nodes, not ingest episodes, and are deliberately excluded from the
    in-flight signal so the drain never waits on them forever."""
    rows = _cypher(
        f"MATCH (e:Episodic {{namespace:'{ns}'}}) RETURN "
        "sum(CASE WHEN e.processing_state='PENDING' THEN 1 ELSE 0 END) AS pending, "
        "sum(CASE WHEN e.processing_state='READY' THEN 1 ELSE 0 END) AS ready, "
        "sum(CASE WHEN e.processing_state='ENRICHING' THEN 1 ELSE 0 END) AS enriching, "
        "sum(CASE WHEN e.processing_state='FAILED' THEN 1 ELSE 0 END) AS failed, "
        "sum(coalesce(toInteger(e.processing_llm_tasks_total), 0)) AS llm_tasks, "
        "sum(coalesce(toInteger(e.processing_attempts), 0)) AS processing_attempts, "
        "count(*) AS total;"
    )
    if not rows or len(rows[0]) < 7:
        return {
            "pending": -1,
            "ready": -1,
            "enriching": -1,
            "failed": -1,
            "llm_tasks": -1,
            "processing_attempts": -1,
            "total": -1,
        }
    r = rows[0]
    return {
        "pending": int(r[0] or 0),
        "ready": int(r[1] or 0),
        "enriching": int(r[2] or 0),
        "failed": int(r[3] or 0),
        "llm_tasks": int(r[4] or 0),
        "processing_attempts": int(r[5] or 0),
        "total": int(r[6] or 0),
    }


def _cypher_count(query: str) -> int:
    rows = _cypher(query)
    if not rows or not rows[0]:
        return -1
    return int(rows[0][0] or 0)


def _scalar_counts(ns: str) -> dict[str, int]:
    """Return the materialization/provenance counts needed to trust a scalar snapshot."""
    return {
        "turn_evidence": _cypher_count(
            f"MATCH (t:TurnEvidence {{namespace:'{ns}'}}) RETURN count(t);"
        ),
        "typed_assertions": _cypher_count(
            f"MATCH (a:TypedAssertion {{namespace:'{ns}'}}) RETURN count(a);"
        ),
        "scalar_views": _cypher_count(
            f"MATCH (v:Entity {{group_id:'{ns}', view_kind:'scalar_state'}}) "
            "WHERE coalesce(v.view_current, true) RETURN count(v);"
        ),
        "scalar_history_views": _cypher_count(
            f"MATCH (v:Entity {{group_id:'{ns}', view_kind:'scalar_history'}}) "
            "WHERE coalesce(v.view_current, true) RETURN count(v);"
        ),
        "user_founded_scalar_views": _cypher_count(
            f"MATCH (t:TurnEvidence {{namespace:'{ns}', declarant:'user'}})-[:FOUNDS]->"
            f"(a:TypedAssertion {{namespace:'{ns}'}})<-[:CURRENT_ANCHOR]-"
            f"(v:Entity {{group_id:'{ns}', view_kind:'scalar_state'}}) "
            "WHERE coalesce(v.view_current, true) RETURN count(DISTINCT v);"
        ),
    }


def _consolidate_scalar(
    admin: httpx.Client,
    menhir_url: str,
    namespace: str,
    *,
    k: int,
    call_budget: int,
) -> dict:
    response = admin.post(
        menhir_url.rstrip("/") + "/api/phase3/run",
        json={
            "namespace": namespace,
            "k": k,
            "source": "longmemeval-scalar-build",
            "call_budget": call_budget,
            "counter_state": False,
        },
        timeout=1800.0,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("scalar_enabled"):
        raise RuntimeError("Menhir did not enable scalar consolidation")
    if int(result.get("scalar_namespaces_processed", 0)) != 1:
        raise RuntimeError(f"scalar consolidation did not finish namespace {namespace}: {result}")
    # A scalar namespace can legitimately abstain from materializing an assertion, but it cannot
    # claim consolidation without exercising the k-sample perception boundary. This also catches
    # stale servers whose scheduler used to snapshot llm_calls before the scalar pass.
    if int(result.get("llm_calls", 0)) < k:
        raise RuntimeError(
            f"scalar consolidation did not exercise {k} LLM samples for {namespace}: {result}"
        )
    return result


def _queue_depth(admin: httpx.Client, menhir_url: str) -> int:
    """Menhir's authoritative global count of unprocessed enrichment work. -1 if unavailable."""
    try:
        resp = admin.get(menhir_url.rstrip("/") + "/api/stats")
        resp.raise_for_status()
        return int(resp.json().get("queue_depth", -1))
    except Exception:
        return -1


def _backend(admin: httpx.Client, menhir_url: str, op: str, body: dict | None = None):
    """Invoke an internal backend operation; raises on HTTP error."""
    url = menhir_url.rstrip("/") + "/api/internal/backend/" + op
    resp = admin.post(url, json=(body or {}))
    resp.raise_for_status()
    return resp.json()


def _reset_namespace(admin: httpx.Client, menhir_url: str, namespace: str) -> None:
    """Fully clear one throwaway namespace, including namespace-keyed TurnEvidence.

    The ordinary DELETE endpoint has a 200-node safety ceiling and does not purge TurnEvidence.
    This benchmark owns its isolated namespace, so force-delete the graph partition explicitly,
    then call the Phase-3 reset to clear evidence that is keyed by namespace rather than group_id.
    """
    _backend(
        admin,
        menhir_url,
        "delete_namespace",
        {"namespace": namespace, "force": True},
    )
    response = admin.post(
        menhir_url.rstrip("/") + "/api/phase3/reset",
        params={"namespace": namespace},
        timeout=300.0,
    )
    response.raise_for_status()


def _await_stale_episode_settlement(
    namespaces: list[str],
    *,
    poll_s: float = 2.0,
    timeout_s: float = 60.0,
) -> None:
    """Block until no stale episode is PENDING or ENRICHING in *namespaces*. Fail closed.

    When Menhir starts with ``ALLOW_RESUME`` it auto-recovers stale enrichment leases from a
    previous killed run. Those workers are already running when this script starts, and they
    write entities, edges, TurnEvidence and lifecycle rows into namespaces this window is about
    to own.

    ORDER IS THE WHOLE POINT: this must run BEFORE any reset. A reset deletes the very
    PENDING/ENRICHING rows that show a worker is alive, so settlement checked afterwards reads
    zero instantly while the worker is still mid-flight in memory -- and its next write lands in
    the namespace the reset just cleaned. Wait first on the rows that exist, then reset once.

    An unreadable count (-1, cypher unreachable) is never settlement: it is indistinguishable
    from a busy worker, so it keeps polling and, if it persists, times out. Timing out RAISES
    rather than resetting anyway -- resetting on an unconfirmed worker is the same race this
    function exists to close, and continuing would ingest into a partition a straggler is still
    writing to. Stopping costs a re-run; proceeding costs a silently corrupted namespace whose
    numbers still look plausible.
    """
    deadline = time.time() + timeout_s
    last: dict[str, dict[str, int]] = {}
    while True:
        last = {namespace: _ns_state_counts(namespace) for namespace in namespaces}
        unsettled = {
            namespace: state
            for namespace, state in last.items()
            if state.get("pending", -1) != 0 or state.get("enriching", -1) != 0
        }
        if not unsettled:
            return
        if time.time() >= deadline:
            details = ", ".join(
                f"{namespace}=pending:{state.get('pending')}/enriching:{state.get('enriching')}"
                for namespace, state in unsettled.items()
            )
            raise RuntimeError(
                "stale enrichment did not settle within "
                f"{timeout_s:.0f}s; refusing to reset a namespace a worker may still be "
                "writing to (a -1 count means the state could not be read): " + details
            )
        time.sleep(poll_s)


def _reset_window_namespaces(
    admin: httpx.Client,
    menhir_url: str,
    namespaces: list[str],
) -> None:
    """Force-reset every namespace in a window (graph partition + phase-3 evidence)."""
    for namespace in namespaces:
        _reset_namespace(admin, menhir_url, namespace)


def _drain_many(
    namespaces: list[str],
    admin: httpx.Client,
    menhir_url: str,
    *,
    idle_polls: int = 2,
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
) -> dict[str, dict[str, int | bool]]:
    """Block until the global queue and every supplied namespace are fully settled.

    PENDING is gated on, not just ENRICHING: an episode a worker has not yet claimed sits in
    PENDING with enriching==0. Unknown queue/state counts are treated as in-flight, and the
    settled condition must remain true for ``idle_polls`` consecutive checks.
    """
    ordered_namespaces = list(dict.fromkeys(namespaces))
    if not ordered_namespaces:
        raise ValueError("at least one namespace is required to drain")
    t0 = time.time()
    settled = 0
    last = {
        namespace: {
            "pending": -1,
            "ready": -1,
            "enriching": -1,
            "failed": -1,
            "llm_tasks": -1,
            "processing_attempts": -1,
            "total": -1,
        }
        for namespace in ordered_namespaces
    }
    while time.time() - t0 < timeout_s:
        qd = _queue_depth(admin, menhir_url)
        last = {namespace: _ns_state_counts(namespace) for namespace in ordered_namespaces}
        # pending/enriching==-1 means cypher timed out / Neo4j temporarily unreachable. Treat
        # unknown as in-flight so we keep polling rather than settling early; only settle when
        # cypher confirms both non-terminal counts are 0.
        in_flight = 0
        for state in last.values():
            pending = state["pending"] if state["pending"] >= 0 else 1
            enriching = state["enriching"] if state["enriching"] >= 0 else 1
            in_flight += pending + enriching
        if qd == 0 and in_flight == 0:
            settled += 1
            if settled >= idle_polls:
                return {namespace: {**state, "timed_out": False} for namespace, state in last.items()}
        else:
            settled = 0
        time.sleep(poll_s)
    return {namespace: {**state, "timed_out": True} for namespace, state in last.items()}


def _drain(
    ns: str,
    admin: httpx.Client,
    menhir_url: str,
    *,
    idle_polls: int = 2,
    poll_s: float = 2.0,
    timeout_s: float = 1800.0,
) -> dict[str, int | bool]:
    """Backward-compatible single-namespace wrapper around :func:`_drain_many`."""
    return _drain_many(
        [ns],
        admin,
        menhir_url,
        idle_polls=idle_polls,
        poll_s=poll_s,
        timeout_s=timeout_s,
    )[ns]


def _failed_evidence_projection_uuids(namespace: str) -> list[str]:
    """Return failed generated projection episodes in one namespace.

    Submitted transcript episodes are retried by ``_ingest_window`` while their chronological
    position is still active. Evidence projections are generated asynchronously by Menhir, so
    their UUIDs are invisible to that scheduler and require this post-drain enumeration.
    """
    rows = _cypher(
        f"MATCH (e:Episodic {{namespace:'{namespace}'}}) "
        "WHERE e.processing_state='FAILED' "
        "AND coalesce(e.is_evidence_projection, false) "
        "RETURN e.uuid AS uuid ORDER BY e.uuid;"
    )
    return [str(row[0]) for row in rows if row and str(row[0]).strip()]


def _retry_failed_evidence_projections(
    namespaces: list[str],
    admin: httpx.Client,
    menhir_url: str,
) -> dict[str, int]:
    """Reset and enqueue each failed generated projection exactly once."""
    requeued = {namespace: 0 for namespace in dict.fromkeys(namespaces)}
    for namespace in requeued:
        for episode_uuid in _failed_evidence_projection_uuids(namespace):
            reset = _backend(
                admin,
                menhir_url,
                "force_reset_failed_episode",
                {"episode_uuid": episode_uuid},
            )
            if not reset:
                raise RuntimeError(
                    "could not reset failed evidence projection "
                    f"{episode_uuid} in namespace {namespace}"
                )
            enqueued = _backend(
                admin,
                menhir_url,
                "enqueue_pending_episode",
                {"episode_uuid": episode_uuid},
            )
            if not enqueued:
                raise RuntimeError(
                    "could not enqueue reset evidence projection "
                    f"{episode_uuid} in namespace {namespace}"
                )
            requeued[namespace] += 1
    return requeued


def _retry_failed_evidence_projections_after_drain(
    namespaces: list[str],
    admin: httpx.Client,
    menhir_url: str,
    drained_by_namespace: dict[str, dict[str, int | bool]],
    requeued_by_namespace: dict[str, int],
    *,
    timeout_s: float,
) -> dict[str, dict[str, int | bool]]:
    """Retry generated projection failures once and return the resulting settled snapshot."""
    if not any(
        int(drained.get("failed", 0)) > 0
        for drained in drained_by_namespace.values()
    ):
        return drained_by_namespace

    projection_requeues = _retry_failed_evidence_projections(
        namespaces,
        admin,
        menhir_url,
    )
    if not any(projection_requeues.values()):
        return drained_by_namespace

    for namespace, count in projection_requeues.items():
        requeued_by_namespace[namespace] += count
    retried_snapshot = _drain_many(
        namespaces,
        admin,
        menhir_url,
        timeout_s=timeout_s,
    )
    timed_out = [
        namespace
        for namespace, drained in retried_snapshot.items()
        if drained.get("timed_out")
    ]
    if timed_out:
        raise RuntimeError(
            "namespace-window projection retry drain timed out; "
            "refusing to manifest incomplete namespaces: "
            + ", ".join(timed_out)
        )
    return retried_snapshot


def _episode_processing(
    admin: httpx.Client,
    menhir_url: str,
    episode_uuid: str,
) -> dict | None:
    """Fetch one episode's lifecycle row without touching Neo4j directly."""
    try:
        result = _backend(
            admin,
            menhir_url,
            "fetch_episode_processing",
            {"episode_uuid": episode_uuid},
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def _ingest_window(
    window: list[WindowItem],
    turn_iterators: list[Iterator[IngestTurn]],
    client: HttpMenhirClient,
    admin: httpx.Client,
    menhir_url: str,
    *,
    timeout_s: float,
    poll_s: float = 1.0,
) -> dict[str, int]:
    """Keep one episode active per namespace while distinct namespaces run concurrently.

    A later episode is not submitted until the preceding episode in that namespace reaches READY
    or FAILED. This is stronger than relying on the per-namespace extraction lock: multiple workers
    can claim queued episodes before they reach that lock, so pre-filling a namespace could reorder
    temporal updates.
    """
    if len(window) != len(turn_iterators):
        raise ValueError("window and turn iterator counts must match")

    active: dict[int, tuple[str, bool]] = {}
    requeued = {state.namespace: 0 for state in window}
    started_at = time.time()
    # turn_key -> :TurnEvidence uuid. Every segment cut from one source turn reuses the first
    # segment's record, so one source turn is one evidence node no matter how it was split.
    # Keys are namespace-qualified, so sharing one map across the window is safe.
    evidence_by_turn: dict[str, str] = {}

    def submit_next(index: int) -> bool:
        """Advance one namespace to its next *extractable* turn.

        Evidence-only turns create no episode, so there is no lifecycle row to
        wait on. They are recorded in order and skipped over here rather than
        occupying the namespace's single active slot -- which would deadlock the
        drain, since a missing episode never reaches READY.
        """
        state = window[index]
        while True:
            try:
                turn = next(turn_iterators[index])
            except StopIteration:
                return False
            episode_uuid = _ingest_turn(
                client,
                state.namespace,
                turn.role,
                turn.content,
                occurred_at=turn.occurred_at,
                session_id=turn.session_id,
                extract=turn.extract,
                evidence_text=turn.evidence_text or turn.content,
                turn_key=turn.turn_key or None,
                evidence_cache=evidence_by_turn,
            )
            state.turns += 1
            if episode_uuid:
                active[index] = (episode_uuid, False)
                return True

    for index in range(len(window)):
        submit_next(index)

    while active:
        if time.time() - started_at >= timeout_s:
            pending = ", ".join(
                f"{window[index].namespace}:{episode_uuid}"
                for index, (episode_uuid, _) in active.items()
            )
            raise RuntimeError(
                "namespace-window episode scheduler timed out; refusing to submit later turns: "
                + pending
            )

        progressed = False
        for index, (episode_uuid, already_retried) in list(active.items()):
            row = _episode_processing(admin, menhir_url, episode_uuid)
            if row is None:
                continue
            processing_state = str(row.get("processing_state") or "").upper().rsplit(".", 1)[-1]
            if processing_state == "READY":
                active.pop(index, None)
                submit_next(index)
                progressed = True
            elif processing_state == "FAILED":
                if not already_retried:
                    try:
                        reset = _backend(
                            admin,
                            menhir_url,
                            "force_reset_failed_episode",
                            {"episode_uuid": episode_uuid},
                        )
                    except Exception:
                        reset = False
                    if reset:
                        try:
                            _backend(
                                admin,
                                menhir_url,
                                "enqueue_pending_episode",
                                {"episode_uuid": episode_uuid},
                            )
                        except Exception:
                            # An idle worker's lease-recovery poll will still discover the PENDING
                            # row. Keep monitoring the same episode instead of advancing out of order.
                            pass
                        active[index] = (episode_uuid, True)
                        requeued[window[index].namespace] += 1
                        progressed = True
                        continue
                # A terminal failure has no later retry that could reorder this namespace. Record
                # it in the final state counts and continue with the next chronological episode.
                active.pop(index, None)
                submit_next(index)
                progressed = True

        if active and not progressed:
            time.sleep(poll_s)

    return requeued


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--menhir-url", default="http://localhost:8102")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument(
        "--fixture",
        default=os.getenv("LME_FIXTURE_PATH"),
        help="offline LongMemEval JSON array; defaults to LME_FIXTURE_PATH when set",
    )
    ap.add_argument(
        "--namespace-prefix",
        default=os.getenv("LME_NS_PREFIX", "lme-"),
        help="namespace prefix for ingested items (default: LME_NS_PREFIX or lme-)",
    )
    ap.add_argument("--drain-timeout", type=float, default=1800.0,
                    help="max seconds to wait for a namespace window's enrichment to settle")
    ap.add_argument(
        "--namespace-window",
        type=_positive_int,
        default=os.getenv("LME_INGEST_CONCURRENCY", "1"),
        help="number of item namespaces to enrich concurrently before draining (default: 1)",
    )
    ap.add_argument(
        "--manifest-item-limit",
        type=_positive_int,
        help=(
            "hard checkpoint: ingest only the first N loaded items while keeping --limit as "
            "the full fixture size for a later manifest-backed resume"
        ),
    )
    ap.add_argument("--consolidate-scalar", action="store_true",
                    help="run scalar-only Phase 3 consolidation before manifesting each namespace")
    ap.add_argument("--consolidation-k", type=int, default=3)
    ap.add_argument("--consolidation-call-budget", type=int, default=50)
    ap.add_argument(
        "--segmentation",
        choices=["none", "sentence", "user-sentence", "adaptive"],
        default=os.getenv("LME_SEGMENTATION", "sentence"),
        help=(
            "Turn segmentation strategy: "
            "'none' = no splitting (arm A), "
            "'sentence' = blind sentence split (arm B, default/legacy), "
            "'user-sentence' = split only user turns (arm C), "
            "'adaptive' = claim-aware segmentation (arm D)"
        ),
    )
    return ap.parse_args(argv)

def _load_items(
    adapter: LongMemEvalMemoryAdapter,
    *,
    limit: int,
    fixture_path: str | None,
) -> list[dict]:
    items = adapter.load_items(limit=limit, fixture_path=fixture_path)
    if fixture_path and not items:
        raise ValueError(f"LongMemEval fixture is empty: {fixture_path}")
    return items


def _namespace(question_id: str, prefix: str) -> str:
    clean_prefix = prefix.strip()
    if not clean_prefix:
        raise ValueError("namespace prefix must not be empty")
    return f"{clean_prefix}{question_id}"


def _iter_item_turns(
    adapter: LongMemEvalMemoryAdapter,
    item: dict,
    namespace: str,
    segmentation: str,
) -> Iterator[IngestTurn]:
    haystack_dates = item.get("haystack_dates") or []
    haystack_session_ids = item.get("haystack_session_ids") or []
    for session_index, session in enumerate(adapter.sessions(item)):
        session_date = haystack_dates[session_index] if session_index < len(haystack_dates) else None
        raw_session_id = (
            haystack_session_ids[session_index]
            if session_index < len(haystack_session_ids)
            else None
        )
        occurred_at = _parse_lme_date(session_date)
        # Namespace-qualify the session id. LongMemEval reuses raw session ids across temporal
        # variants; sharing them lets Graphiti deduplicate the second namespace against the first.
        session_id = (
            f"{namespace}-{raw_session_id}"
            if raw_session_id
            else f"{namespace}-s{session_index}"
        )
        for turn_index, turn in enumerate(session):
            content = turn.get("content", "")
            if not content:
                continue
            role = turn.get("role", "user")
            contents: list[str]
            # CONTEXT_ONLY turns take the evidence-only path below instead of
            # being merged into a neighbour. Merging would concatenate two source
            # turns into one record, destroying the per-turn TurnEvidence
            # boundaries and counts the scalar audit reads, and -- when the two
            # turns had different roles -- would file assistant words under
            # declarant="user".
            extract = True
            if segmentation == "none":
                contents = [content]
            elif segmentation == "sentence":
                contents = _split_sentences(content)
            elif segmentation == "user-sentence":
                contents = (
                    _split_sentences(content)
                    if role.strip().lower() == "user"
                    else [content]
                )
            elif segmentation == "adaptive":
                mode = decide_segmentation(role, content)
                if mode == SegmentationMode.SKIP:
                    contents = []
                elif mode == SegmentationMode.CONTEXT_ONLY:
                    # Captured as evidence, never extracted. Purely interrogative
                    # or phatic turns produce only {"name":"user"} with zero edges,
                    # so sending them to extraction fails the episode; skipping
                    # extraction keeps the turn's own evidence record intact.
                    contents = [content]
                    extract = False
                elif mode == SegmentationMode.SEGMENT_CLAIMS:
                    contents = [
                        content,
                        *[claim.text for claim in _heuristic_extract_claims_sync(content, role)],
                    ]
                else:
                    contents = [content]
            else:
                contents = [content]
            # One identity per SOURCE turn, shared by every segment cut from it. The index is
            # over the raw session, so it does not shift when a turn is skipped, and it keeps
            # two identical repeated turns distinct -- menhir's fallback key hashes the text
            # and would merge them onto one evidence node.
            turn_key = f"{namespace}|{session_id}|t{turn_index}"
            for segmented_content in contents:
                yield IngestTurn(
                    role=role,
                    content=segmented_content,
                    occurred_at=occurred_at,
                    session_id=session_id,
                    extract=extract,
                    # Evidence is the whole turn, even when the episode is one segment of it.
                    evidence_text=content,
                    turn_key=turn_key,
                )


def _write_manifest(path: Path, manifest: list[dict]) -> None:
    """Atomically checkpoint completed namespaces for safe resume after interruption."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _require_no_failed_episodes(
    drained_by_namespace: dict[str, dict[str, int | bool]],
) -> None:
    """Stop a scalar build before more paid work when a completed window is incomplete.

    The gate is strict and has no tolerance knob. A FAILED episode means some turn of
    that item never made it into the graph, so every downstream number -- scalar views,
    user-founded views, recall accuracy -- is computed over a namespace that does not
    match the transcript. Scoring a partial graph is worse than stopping: it produces a
    plausible number nobody can trust. A per-namespace failure allowance was tried and
    reverted because it converted that hard stop into a line of log output, letting a
    build ship silently-lossy namespaces. Turns that legitimately yield no edges belong
    on the evidence-only path in ``_iter_item_turns``, which creates no episode to fail.
    """
    residual_failures = {
        namespace: int(drained.get("failed", 0))
        for namespace, drained in drained_by_namespace.items()
        if int(drained.get("failed", 0)) > 0
    }
    if residual_failures:
        details = ", ".join(
            f"{namespace}={failed}" for namespace, failed in residual_failures.items()
        )
        raise RuntimeError(
            "scalar namespace window has residual FAILED episodes after bounded retry; "
            "refusing further paid work: " + details
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    adapter = LongMemEvalMemoryAdapter()
    source = (
        f"fixture={args.fixture}"
        if args.fixture
        else f"variant={os.getenv('LONGMEMEVAL_VARIANT', 's')}"
    )
    print(f"loading LongMemEval items (limit={args.limit}, {source})...", flush=True)
    items = _load_items(adapter, limit=args.limit, fixture_path=args.fixture)
    print(f"loaded {len(items)} items", flush=True)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    done_ids: set[str] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        done_ids = {m["question_id"] for m in manifest}
        print(f"RESUME: {len(done_ids)} items already ingested (skipping them)", flush=True)

    # This path resets namespaces and ingests. Guarded at construction so a
    # mistyped --menhir-url cannot write into a real instance.
    from archolith_bench.harness.memory_ab import assert_not_production

    assert_not_production(args.menhir_url)
    client = HttpMenhirClient(args.menhir_url)
    admin = httpx.Client(timeout=120.0)
    t_all = time.time()
    target_items = (
        items[:args.manifest_item_limit]
        if args.manifest_item_limit is not None
        else items
    )
    remaining = [
        it
        for it in target_items
        if str(it.get("question_id") or "") not in done_ids
    ]
    if len(target_items) < len(items):
        print(
            f"HARD CHECKPOINT: targeting {len(target_items)}/{len(items)} loaded items",
            flush=True,
        )
    print(f"to ingest this run: {len(remaining)} items", flush=True)

    completed_this_run = 0
    for window_start in range(0, len(remaining), args.namespace_window):
        raw_window = remaining[window_start:window_start + args.namespace_window]
        window = [
            WindowItem(
                item=item,
                question_id=str(item.get("question_id") or f"lme-{window_start + index}"),
                namespace=_namespace(
                    str(item.get("question_id") or f"lme-{window_start + index}"),
                    args.namespace_prefix,
                ),
            )
            for index, item in enumerate(raw_window)
        ]
        window_started = time.time()

        window_namespaces = [state.namespace for state in window]

        # Menhir's stale-lease recovery may already be enriching a previous run's episodes into
        # these namespaces. Wait for that to finish FIRST -- while the PENDING/ENRICHING rows
        # that prove a worker is alive still exist. Resetting first would delete those rows and
        # make settlement read zero against a worker that is still running, which is the race
        # itself. This raises rather than proceeding if settlement cannot be confirmed.
        _await_stale_episode_settlement(window_namespaces)

        # Only now, with no writer left, reset. An interrupted window has no manifest rows, so
        # every namespace is fully cleared before resubmitting -- including TurnEvidence, which
        # is not group_id-keyed. One reset after confirmed settlement is enough; a second pass
        # would only re-open the gap it is meant to close.
        _reset_window_namespaces(admin, args.menhir_url, window_namespaces)

        turn_iterators = [
            _iter_item_turns(adapter, state.item, state.namespace, args.segmentation)
            for state in window
        ]
        requeued_by_namespace = _ingest_window(
            window,
            turn_iterators,
            client,
            admin,
            args.menhir_url,
            timeout_s=args.drain_timeout,
        )

        drained_by_namespace = _drain_many(
            window_namespaces,
            admin,
            args.menhir_url,
            timeout_s=args.drain_timeout,
        )
        timed_out = [
            namespace
            for namespace, drained in drained_by_namespace.items()
            if drained.get("timed_out")
        ]
        if timed_out:
            raise RuntimeError(
                "namespace-window drain timed out; refusing to manifest incomplete namespaces: "
                + ", ".join(timed_out)
            )

        # Menhir creates evidence-projection episodes asynchronously. Their UUIDs never pass
        # through _ingest_window's submitted-episode scheduler, so enumerate and retry only
        # those generated children here. Submitted parent episodes keep their existing one-retry
        # limit and cannot accidentally receive a second attempt from this sweep.
        drained_by_namespace = _retry_failed_evidence_projections_after_drain(
            window_namespaces,
            admin,
            args.menhir_url,
            drained_by_namespace,
            requeued_by_namespace,
            timeout_s=args.drain_timeout,
        )
        if args.consolidate_scalar:
            _require_no_failed_episodes(drained_by_namespace)

        # Scalar consolidation remains sequential: it is a small fraction of build time and
        # parallel calls would add provider-rate and graph-write risk without material speedup.
        for state in window:
            drained = drained_by_namespace[state.namespace]
            requeued = requeued_by_namespace[state.namespace]
            scalar_result: dict = {}
            scalar_counts: dict[str, int] = {}
            if args.consolidate_scalar:
                scalar_result = _consolidate_scalar(
                    admin,
                    args.menhir_url,
                    state.namespace,
                    k=args.consolidation_k,
                    call_budget=args.consolidation_call_budget,
                )
                scalar_counts = _scalar_counts(state.namespace)
                if scalar_counts["turn_evidence"] <= 0:
                    raise RuntimeError(
                        f"no TurnEvidence captured for scalar namespace {state.namespace}"
                    )

            manifest.append({
                "question_id": state.question_id,
                "namespace": state.namespace,
                "fixture": str(Path(args.fixture).resolve()) if args.fixture else None,
                "question": adapter.question(state.item),
                "answer": str(state.item.get("answer", "")),
                "question_type": state.item.get("question_type"),
                "turns": state.turns,
                "episodes": drained.get("total"),
                "ready": drained.get("ready"),
                "failed_remaining": drained.get("failed"),
                "failed_requeued": requeued,
                "enrichment_llm_tasks": drained.get("llm_tasks"),
                "processing_attempts": drained.get("processing_attempts"),
                "drain_timed_out": drained.get("timed_out", False),
                "namespace_window": args.namespace_window,
                "scalar_consolidated": bool(
                    args.consolidate_scalar
                    and int(scalar_result.get("scalar_namespaces_processed", 0)) == 1
                    and int(scalar_result.get("llm_calls", 0)) >= args.consolidation_k
                ),
                "scalar_llm_calls": int(scalar_result.get("llm_calls", 0)),
                "scalar_states_written": int(scalar_result.get("scalar_states_written", 0)),
                **scalar_counts,
            })
            _write_manifest(manifest_path, manifest)
            completed_this_run += 1
            elapsed = time.time() - t_all
            rate = completed_this_run / elapsed if elapsed else 0
            eta = (len(remaining) - completed_this_run) / rate if rate else 0
            print(
                f"[{len(done_ids)+completed_this_run}/{len(items)}] {state.question_id} "
                f"ns={state.namespace} turns={state.turns} episodes={drained.get('total')} "
                f"ready={drained.get('ready')} failed={drained.get('failed')} "
                f"requeued={requeued} llm_tasks={drained.get('llm_tasks')} "
                f"window={time.time()-window_started:.0f}s "
                f"total={elapsed:.0f}s eta={eta/3600:.1f}h",
                flush=True,
            )

    outcome = "CHECKPOINT" if len(target_items) < len(items) else "DONE"
    print(
        f"\n{outcome}: manifest has {len(manifest)}/{len(items)} items -> {args.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
