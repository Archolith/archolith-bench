"""Throwaway: RESUMABLE ingest of LongMemEval haystacks into a PERSISTENT menhir/Neo4j under
STABLE namespaces (lme-<question_id>), no per-item reset on success -- so recall-only A/B
(main vs frontier) can run against the pre-ingested graph without re-ingesting.

Resume WITHOUT reset:
- An incremental manifest (results/lme-ingest/manifest.json) records each FULLY-ingested AND
  FULLY-ENRICHED item; on restart those items are SKIPPED.
- Every extractable segment is submitted with an idempotency key (``<turn_key>#s<segment>``),
  so a re-run replays the same episode instead of creating a duplicate (P1). Evidence MERGEs on
  turn_key. A resumed namespace is therefore continued, never wiped.
- A state log next to the manifest (ingest-state-<manifest>.jsonl, see ingest_state.py) records
  each namespace's turn-plan digest, every retry spent (durably, BEFORE the reset), resets and
  completions. A resume refuses if a recorded digest no longer matches (P5), and refuses a
  namespace that already holds Episodic rows but has no recorded plan.
- Namespaces are reset ONLY when listed (--reset-namespace / --reset-file); each is first waited
  on until stale enrichment settles, then reset (P4).
- Per-turn ingest has bounded retry/backoff; HTTP 409 (key reused with different content) is
  never retried. Menhir/Neo4j down beyond --outage-timeout exits non-zero; re-run to resume.

Sliding namespace pool (--namespace-window = pool size):
- Up to N namespaces run at once, each with exactly ONE active episode. The next segment of a
  namespace is submitted only when the previous one is READY or terminally FAILED (P2), so
  workers never reorder temporal updates. A finished namespace frees its slot for the next item.
- Enrichment completeness: a namespace is settled when no episode is PENDING or ENRICHING for 2
  consecutive polls (Neo4j). Only then is it swept once for failed evidence-projection episodes
  and manifested, so "manifest done" means "fully enriched". A global drain runs once at the end.
- Benchmark mode disables the scheduler, so FAILED episodes are never auto-retried: the pool
  retries each submitted episode and each generated projection at most once; any residual
  failure is recorded in the manifest and blocks scalar consolidation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
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
from ingest_state import IngestStateLog  # noqa: E402
from neo4j_http import Neo4jHttpReader, Neo4jUnavailable  # noqa: E402
from validate_run import consolidation_errors  # noqa: E402
from archolith_bench.harness.longmemeval import LongMemEvalMemoryAdapter  # noqa: E402
from archolith_bench.harness.menhir_client import HttpMenhirClient  # noqa: E402

import memory_writer  # noqa: E402
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
REQUIRE_TURN_EVIDENCE = os.getenv("LME_REQUIRE_TURN_EVIDENCE", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
# CONTEXT_ONLY user turns (questions, requests) still state facts ("what should I cook in my
# Instant Pot"). On: keep them as store-only source memories Menhir embeds but never extracts.
# Off: evidence only, no episode (the pre-#43 behaviour). Needs a Menhir with store_only ingest,
# which an older Menhir silently ignores (it would extract the turn), so it is off unless set;
# config.sh turns it on for builds, and build_graph.sh checks the serving tree first.
STORE_ONLY_CONTEXT = os.getenv("LME_STORE_ONLY_CONTEXT", "0").strip().lower() in {
    "1", "true", "yes", "on",
}
# Menhir's :Episodic.extraction_mode value for store-only episodes.
_STORE_ONLY = "store_only"

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
    # Position among the segments cut from one source turn; with turn_key it is the
    # idempotency identity of the episode (P1).
    segment_index: int = 0
    # Non-empty overrides the ingest ``source`` (memory-writer items are "agent-writer").
    source: str = ""
    # >= 0 marks a memory-writer item: its position among the items written for one assistant
    # turn. ``turn_key`` is then the assistant turn's, and the evidence is that turn's.
    writer_index: int = -1
    # True for a CONTEXT_ONLY user turn kept as a source memory: an embedded :Episodic node
    # Menhir never extracts (archolith-bench#43). Only meaningful with extract=True.
    store_only: bool = False


@dataclass
class NamespaceRun:
    """Per-namespace pool state: the fixed turn plan and where submission has got to."""

    item: dict
    question_id: str
    namespace: str
    plan: list[IngestTurn] = field(default_factory=list)
    digest: str = ""
    segmentation: str | None = None
    cursor: int = 0
    # (episode_uuid, already_retried) for the one in-flight episode, if any.
    active: tuple[str, bool] | None = None
    phase: str = "submitting"
    settle_polls: int = 0
    settle_first_at: float = 0.0
    last_progress_at: float = 0.0
    # turn_key -> :TurnEvidence uuid, so every segment of one source turn reuses one record.
    evidence_cache: dict[str, str] = field(default_factory=dict)


class ServerUnavailableError(RuntimeError):
    """Menhir or Neo4j stayed unreachable past --outage-timeout."""


class IdempotencyConflictError(RuntimeError):
    """Menhir returned 409: the same ingest key was sent with different content."""


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
    ingest_key: str | None = None,
    source: str = "",
    writer_index: int = -1,
    store_only: bool = False,
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

    ``store_only=True`` (CONTEXT_ONLY user turns) records evidence and creates the episode as a
    source memory that Menhir embeds but never extracts or projects.

    Uses wait=False: the namespace-window scheduler is the completeness
    guarantee, so we do not block on an HTTP request per turn. It still keeps
    at most one episode active per namespace while allowing different
    namespaces to use multiple enrichment workers.
    """
    normalized_role = role.strip().lower() or "user"
    is_user = normalized_role == "user"
    turn_evidence_uuid: str | None = None
    # A memory-writer item (agent-tier) is grounded on the ASSISTANT turn it was written from.
    is_writer = writer_index >= 0
    if is_writer and not evidence_text:
        raise ValueError("a memory-writer turn needs the assistant turn's evidence_text")
    # The evidence body is the ORIGINAL turn, never the segment being ingested.
    evidence_body = content if evidence_text is None else evidence_text
    evidence_role = "assistant" if is_writer else normalized_role

    # A sibling segment of the same source turn already recorded the evidence.
    if evidence_cache is not None and turn_key:
        turn_evidence_uuid = evidence_cache.get(turn_key)

    # Ground the user-tier claim by recording turn evidence first. Evidence-only
    # turns are captured for every role, since they have no episode to ground and
    # the record is the only trace the source turn leaves. A writer item re-records its
    # assistant turn's evidence when the cache is empty (a resume); Menhir MERGEs on turn_key.
    if turn_evidence_uuid is None and (is_user or not extract or is_writer):
        for attempt in range(tries):
            try:
                ev = client.record_turn_evidence(
                    ns,
                    evidence_body,
                    role=evidence_role,
                    declarant=evidence_role,
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

    if REQUIRE_TURN_EVIDENCE and (is_user or not extract or is_writer) and not turn_evidence_uuid:
        raise RuntimeError(
            "turn-evidence capture returned no turn_id; refusing to continue without "
            f"grounding for namespace {ns}"
        )

    # Evidence-only: the turn is recorded, but there is no episode to enrich.
    if not extract:
        return ""

    # Only sent when supplied, so a resubmit of the same segment returns the first episode.
    key_kwargs = {"idempotency_key": ingest_key} if ingest_key is not None else {}
    if is_writer:
        key_kwargs["memory_role"] = memory_writer.WRITER_ROLE
    if store_only:
        key_kwargs["store_only"] = True
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
                source=(source or ("user" if is_user else None)),
                turn_evidence_uuid=turn_evidence_uuid,
                wait=False,
                **key_kwargs,
            )
            episode_uuid = str((result or {}).get("episode_id") or "")
            if not episode_uuid:
                raise RuntimeError(f"menhir did not return an episode id for namespace {ns}")
            return episode_uuid
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise IdempotencyConflictError(
                    f"menhir returned 409 for namespace {ns} ingest key {ingest_key!r}: this key "
                    "was first submitted with a different request (changed segment text, or a "
                    "different evidence link); reset the namespace (--reset-namespace) to "
                    "re-ingest it"
                ) from exc
            if attempt == tries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    ingest retry {attempt+1}/{tries} after {exc.__class__.__name__} -> sleep {wait}s", flush=True)
            time.sleep(wait)
        except (httpx.HTTPError,) as exc:
            if attempt == tries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    ingest retry {attempt+1}/{tries} after {exc.__class__.__name__} -> sleep {wait}s", flush=True)
            time.sleep(wait)


_NEO4J_HTTP = Neo4jHttpReader.from_env()


def _plain_cell(cell: object) -> str:
    """Render a JSON cell the way cypher-shell's plain format does."""
    if cell is None:
        return "NULL"
    if isinstance(cell, bool):
        return "true" if cell else "false"
    return str(cell)


def _ns_state_counts_many(namespaces: list[str]) -> dict[str, dict[str, int]]:
    """Batched ``_ns_state_counts``; -1 values mean unreadable."""
    names = list(dict.fromkeys(namespaces))
    if _NEO4J_HTTP is not None:
        try:
            return _NEO4J_HTTP.namespace_state_counts(names)
        except Neo4jUnavailable:
            pass
    return {ns: _ns_state_counts(ns) for ns in names}


def _cypher(query: str) -> list[list[str]]:
    """Run a read-only cypher query against the lme Neo4j (HTTP when LME_NEO4J_HTTP_URL is set,
    else docker exec); return data rows (header dropped). Returns [] on any failure."""
    if _NEO4J_HTTP is not None:
        try:
            return [
                [_plain_cell(cell) for cell in row]
                for row in _NEO4J_HTTP.rows(query.strip().rstrip(";"))
            ]
        except Neo4jUnavailable:
            pass
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


def _cypher_counts(query: str, width: int) -> list[int]:
    """Read one aggregate row of ``width`` integers; every value is -1 when it cannot be read."""
    rows = _cypher(query)
    if not rows or len(rows[0]) < width:
        return [-1] * width
    try:
        return [int(value or 0) for value in rows[0][:width]]
    except ValueError:
        return [-1] * width


def _submitted_epoch_millis(item: dict) -> list[int]:
    """The session dates this item submits as ``occurred_at`` (and so as reference_time)."""
    submitted = set()
    for raw in item.get("haystack_dates") or []:
        iso = _parse_lme_date(raw)
        if iso:
            submitted.add(int(datetime.fromisoformat(iso).timestamp() * 1000))
    return sorted(submitted)


def _user_turns_submitted(turns: Iterator[IngestTurn]) -> int:
    """Distinct user source turns ingested as episodes (store-only included, since Menhir admits
    those too), keyed as ``_ingest_turn`` keys evidence.

    Counted per source turn, not per segment: Menhir's admission gate may legitimately downgrade
    a heuristic claim segment that is not verbatim in the turn, but the whole turn must be admitted.
    """
    return len({
        turn.turn_key
        for turn in turns
        if turn.extract and (turn.role.strip().lower() or "user") == "user"
    })


def _integrity_counts(ns: str, submitted_epoch_millis: list[int]) -> dict[str, int]:
    """Read-only per-namespace graph checks for final acceptance (validate_run).

    ``user_turns_admitted`` is compared there with ``user_turns_submitted``; every other value is
    a violation count that must be zero. -1 means the count could not be read.
    """
    nl = ns.lower()
    admitted, unprojected = _cypher_counts(
        # Provenance: a user turn's evidence is admitted by a user-tier memory and projected.
        # A store-only episode is admitted but never projected (Menhir mints no projection for
        # it), so only an extracted admitter makes a missing projection a violation.
        f"MATCH (t:TurnEvidence {{namespace:'{ns}'}}) "
        "WITH t, EXISTS { MATCH (q:Episodic)-[:ADMITTED_ON]->(t) WHERE q.valid_at IS NULL "
        "AND q.source = 'user' AND NOT coalesce(q.is_evidence_projection, false) } AS admitted, "
        "EXISTS { MATCH (q:Episodic)-[:ADMITTED_ON]->(t) WHERE q.valid_at IS NULL "
        "AND q.source = 'user' AND NOT coalesce(q.is_evidence_projection, false) "
        f"AND coalesce(q.extraction_mode, '') <> '{_STORE_ONLY}' }} AS extracted, "
        "EXISTS { MATCH (p:Episodic)-[:ADMITTED_ON]->(t) "
        "WHERE coalesce(p.is_evidence_projection, false) } AS projected "
        "RETURN sum(CASE WHEN admitted THEN 1 ELSE 0 END), "
        "sum(CASE WHEN extracted AND NOT projected THEN 1 ELSE 0 END);",
        2,
    )
    (unfounded,) = _cypher_counts(
        # Every assertion traces to this namespace's evidence: FOUNDS directly, or GROUNDS from an
        # episode admitted on it (view_write_repository resolves Episodic anchors the same way).
        "MATCH (a) WHERE (a:TypedAssertion OR a:TypedEventAssertion) "
        f"AND a.namespace = '{ns}' "
        f"AND NOT EXISTS {{ MATCH (:TurnEvidence {{namespace:'{ns}'}})-[:FOUNDS]->(a) }} "
        f"AND NOT EXISTS {{ MATCH (:TurnEvidence {{namespace:'{ns}'}})<-[:ADMITTED_ON]-"
        "(:Episodic)-[:GROUNDS]->(a) } "
        "RETURN count(a);",
        1,
    )
    (valid_at_unsubmitted,) = _cypher_counts(
        # Graphiti episodes (group_id; Menhir's queue node carries namespace instead).
        f"MATCH (g:Episodic {{group_id:'{ns}'}}) "
        "RETURN sum(CASE WHEN g.valid_at IS NULL THEN 1 "
        f"WHEN datetime(toString(g.valid_at)).epochMillis IN {submitted_epoch_millis} THEN 0 "
        "ELSE 1 END);",
        1,
    )
    (time_unpaired,) = _cypher_counts(
        # Each READY queue node resolves to exactly one Graphiti episode dated at its reference_time.
        # A store-only node is never extracted, so it has no Graphiti episode to pair with.
        # A verbatim-skipped client (Menhir marks it READY with substage 'verbatim_skipped' and
        # extraction_skipped_for = the projection that extracted the same text) has none either:
        # it passes only when that projection is READY in this namespace with the same
        # reference_time. The projection is itself a queue node here, so its own pairing is checked.
        f"MATCH (q:Episodic {{namespace:'{ns}'}}) "
        "WHERE q.valid_at IS NULL AND q.processing_state = 'READY' "
        f"AND coalesce(q.extraction_mode, '') <> '{_STORE_ONLY}' "
        f"OPTIONAL MATCH (g:Episodic {{group_id:'{ns}'}}) "
        "WHERE g.uuid = q.resolved_episode_uuid AND g.valid_at IS NOT NULL "
        "WITH q, collect(g) AS gs, "
        "q.processing_substage = 'verbatim_skipped' AND q.extraction_skipped_for IS NOT NULL "
        "AS skipped "
        f"OPTIONAL MATCH (p:Episodic {{namespace:'{ns}'}}) "
        "WHERE skipped AND p.uuid = q.extraction_skipped_for "
        "AND coalesce(p.is_evidence_projection, false) AND p.processing_state = 'READY' "
        "WITH q, gs, skipped, collect(p) AS ps "
        "RETURN sum(CASE WHEN skipped THEN "
        "CASE WHEN size(ps) = 1 AND q.reference_time IS NOT NULL "
        "AND ps[0].reference_time IS NOT NULL "
        "AND datetime(toString(ps[0].reference_time)).epochMillis "
        "= datetime(toString(q.reference_time)).epochMillis THEN 0 ELSE 1 END "
        "WHEN size(gs) <> 1 OR q.reference_time IS NULL THEN 1 "
        "WHEN datetime(toString(gs[0].valid_at)).epochMillis "
        "<> datetime(toString(q.reference_time)).epochMillis THEN 1 ELSE 0 END);",
        1,
    )
    (cross_namespace,) = _cypher_counts(
        # Nodes stamped with two silos, plus any edge from this namespace to another silo's node,
        # or carrying another silo's namespace/group_id itself.
        f"MATCH (a) WHERE toLower(a.namespace) = '{nl}' OR toLower(a.group_id) = '{nl}' "
        "WITH a, CASE WHEN a.namespace IS NOT NULL AND a.group_id IS NOT NULL "
        "AND toLower(a.namespace) <> toLower(a.group_id) THEN 1 ELSE 0 END AS split "
        "OPTIONAL MATCH (a)-[r]-(b) "
        f"WHERE toLower(coalesce(b.namespace, '{nl}')) <> '{nl}' "
        f"OR toLower(coalesce(b.group_id, '{nl}')) <> '{nl}' "
        f"OR toLower(coalesce(r.namespace, '{nl}')) <> '{nl}' "
        f"OR toLower(coalesce(r.group_id, '{nl}')) <> '{nl}' "
        "WITH a, split, count(r) AS crossing "
        "RETURN sum(split) + sum(crossing);",
        1,
    )
    return {
        "user_turns_admitted": admitted,
        "admitted_turns_unprojected": unprojected,
        "assertions_unfounded": unfounded,
        "episodes_valid_at_unsubmitted": valid_at_unsubmitted,
        "ready_episodes_time_unpaired": time_unpaired,
        "cross_namespace_links": cross_namespace,
    }


def _consolidate_lanes(
    admin: httpx.Client,
    menhir_url: str,
    namespace: str,
    *,
    k: int,
    call_budget: int,
    scalar_state: bool = True,
    counter_state: bool = False,
    event_history: bool = False,
) -> dict:
    response = admin.post(
        menhir_url.rstrip("/") + "/api/phase3/run",
        json={
            "namespace": namespace,
            "k": k,
            "source": "longmemeval-scalar-build",
            "call_budget": call_budget,
            "counter_state": counter_state,
        },
        timeout=1800.0,
    )
    response.raise_for_status()
    result = response.json()
    errors = consolidation_errors(result, {
        "scalar": scalar_state, "counter": counter_state, "event": event_history,
    }, namespace=namespace, k=k)
    if errors:
        raise RuntimeError("; ".join(errors))
    return result


def _queue_depth(admin: httpx.Client, menhir_url: str) -> int:
    """Menhir's authoritative global count of unprocessed enrichment work. -1 if unavailable."""
    try:
        resp = admin.get(menhir_url.rstrip("/") + "/api/stats")
        resp.raise_for_status()
        return int(resp.json().get("queue_depth", -1))
    except Exception:
        return -1


class ProviderStoppedError(RuntimeError):
    """Menhir reports a provider state that waiting cannot fix (quota exhausted, auth failure)."""


def _provider_stop_reason(admin: httpx.Client, menhir_url: str) -> str | None:
    """Menhir's standing provider quota/auth failure from /api/health, or None.

    An unreadable health response is not a stop: the drain's own counts still decide.
    """
    try:
        resp = admin.get(menhir_url.rstrip("/") + "/api/health")
        resp.raise_for_status()
        body = resp.json()
    except Exception:
        return None
    return _stop_reason_from_body(body)


def _stop_reason_from_body(body: object) -> str | None:
    for key in ("provider_quota_exhausted", "provider_auth_failure"):
        value = body.get(key) if isinstance(body, dict) else None
        if value:
            return f"{key}: {value}"
    return None


def _provider_stopped_error(stop_reason: str) -> ProviderStoppedError:
    return ProviderStoppedError(
        f"menhir reports {stop_reason}; stopping the ingest (fix the provider account, "
        "then resume)"
    )


def _raise_if_provider_stopped(admin: httpx.Client, menhir_url: str) -> None:
    stop_reason = _provider_stop_reason(admin, menhir_url)
    if stop_reason:
        raise _provider_stopped_error(stop_reason)


def _check_menhir_health(admin: httpx.Client, menhir_url: str) -> bool:
    """One GET /api/health. Raises ProviderStoppedError on a standing quota/auth failure;
    returns True only for a 2xx JSON body (False feeds the outage clock)."""
    try:
        resp = admin.get(menhir_url.rstrip("/") + "/api/health")
        resp.raise_for_status()
        body = resp.json()
    except Exception:
        return False
    stop_reason = _stop_reason_from_body(body)
    if stop_reason:
        raise _provider_stopped_error(stop_reason)
    return isinstance(body, dict)


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
        # Quota exhaustion fails every later episode fast and retries cannot fix it, so stop the
        # build here instead of settling into a window of FAILED rows and moving on.
        _raise_if_provider_stopped(admin, menhir_url)
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


class GraphUnreadable(RuntimeError):
    """A graph read failed; the answer is unknown, not empty."""


# Menhir records these in processing_error when the provider account, not the content, failed.
_PROVIDER_FAILURE_MARKERS = (
    "insufficient_quota", "provider_quota_exhausted", "provider_auth_failure", "invalid_api_key",
)


def _provider_caused(error: object) -> bool:
    text = str(error or "").lower()
    return any(marker in text for marker in _PROVIDER_FAILURE_MARKERS)


def _failed_evidence_projections(namespace: str) -> list[tuple[str, bool]]:
    """(uuid, provider_caused) of failed generated projection episodes in one namespace.

    Submitted transcript episodes are retried by ``_ingest_pool`` while their chronological
    position is still active. Evidence projections are generated asynchronously by Menhir, so
    their UUIDs are invisible to that scheduler and require this post-drain enumeration.
    Raises GraphUnreadable when the read fails (``_cypher`` returns [] for a failure and for
    "none"), so a count that always returns one row is read first and must match the list.
    """
    where = (
        f"MATCH (e:Episodic {{namespace:'{namespace}'}}) "
        "WHERE e.processing_state='FAILED' AND coalesce(e.is_evidence_projection, false) "
    )
    count_rows = _cypher(where + "RETURN count(e) AS n;")
    try:
        expected = int(count_rows[0][0])
    except (IndexError, TypeError, ValueError):
        raise GraphUnreadable(f"could not count failed evidence projections in {namespace}")
    if expected == 0:
        return []
    markers = ", ".join(f"'{marker}'" for marker in _PROVIDER_FAILURE_MARKERS)
    rows = _cypher(
        where
        + "WITH e, toLower(coalesce(toString(e.processing_error), '')) AS err "
        + f"RETURN e.uuid AS uuid, any(m IN [{markers}] WHERE err CONTAINS m) AS provider "
        + "ORDER BY uuid;"
    )
    found = [(str(row[0]), str(row[1]).lower() == "true") for row in rows
             if len(row) >= 2 and str(row[0]).strip()]
    if len(found) != expected:
        raise GraphUnreadable(f"could not list failed evidence projections in {namespace}")
    return found


def _failed_evidence_projection_uuids(namespace: str) -> list[str]:
    return [uuid for uuid, _ in _failed_evidence_projections(namespace)]


def _requeue_provider_failure(
    admin: httpx.Client, menhir_url: str, episode_uuid: str,
) -> bool:
    """Requeue a FAILED row whose cause was the provider account (quota/auth), not its content.

    Such a failure never spends the one-retry budget and never makes the row terminal: while
    Menhir still reports the condition, ``_check_menhir_health`` raises ProviderStoppedError and
    the run stops; once it has cleared (normally on a resume), the row is reset and requeued.
    Returns False when Menhir is unreachable or the reset failed (try again next cycle).
    """
    if not _check_menhir_health(admin, menhir_url):
        return False
    try:
        if not _backend(admin, menhir_url, "force_reset_failed_episode",
                        {"episode_uuid": episode_uuid}):
            return False
    except Exception:
        return False
    try:
        _backend(admin, menhir_url, "enqueue_pending_episode", {"episode_uuid": episode_uuid})
    except Exception:
        pass  # lease recovery still finds the PENDING row
    return True


def _retry_failed_evidence_projections(
    namespaces: list[str],
    admin: httpx.Client,
    menhir_url: str,
    *,
    skip: set[str] | None = None,
    on_retry: Callable[[str, str], None] | None = None,
    on_requeued: Callable[[str, str, str], None] | None = None,
) -> dict[str, int]:
    """Reset and enqueue each failed generated projection at most once.

    ``skip`` holds uuids already retried (never retried again); ``on_retry(namespace, uuid)``
    runs BEFORE that uuid's reset so the retry budget is durable; ``on_requeued(namespace,
    uuid, kind)`` runs after a successful reset. Provider-caused failures (quota/auth) use
    ``_requeue_provider_failure`` instead and spend no budget. Raises GraphUnreadable when the
    failed set cannot be read, or when an enqueue is refused and the reset row cannot be read.
    """
    requeued = {namespace: 0 for namespace in dict.fromkeys(namespaces)}
    for namespace in requeued:
        for episode_uuid, provider_caused in _failed_evidence_projections(namespace):
            if provider_caused:
                if _requeue_provider_failure(admin, menhir_url, episode_uuid):
                    if on_requeued is not None:
                        on_requeued(namespace, episode_uuid, "provider")
                    requeued[namespace] += 1
                    continue
                raise GraphUnreadable(
                    f"could not requeue provider-failed projection {episode_uuid} in {namespace}"
                )
            if skip and episode_uuid in skip:
                continue
            if on_retry is not None:
                on_retry(namespace, episode_uuid)
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
            # False also means "already queued or claimed": Menhir's lease sweep or a worker can
            # take the reset row first. A readable row is in flight (PENDING rows are swept);
            # only an unconfirmed one fails closed, and the pool retries GraphUnreadable.
            if not enqueued and _episode_processing(admin, menhir_url, episode_uuid) is None:
                raise GraphUnreadable(
                    "could not enqueue or read reset evidence projection "
                    f"{episode_uuid} in namespace {namespace}"
                )
            if on_requeued is not None:
                on_requeued(namespace, episode_uuid, "projection")
            requeued[namespace] += 1
    return requeued


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


def _read_reset_file(path: str) -> list[str]:
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            names.append(text)
    return names


def _startup_checks(
    runs: list[NamespaceRun],
    state: IngestStateLog,
    args: argparse.Namespace,
    admin: httpx.Client,
    *,
    mode: str | None = None,
) -> None:
    """Refuse unsafe resumes, then reset ONLY the explicitly listed namespaces (P4/P5/P1).

    Runs before any submit. Refusals raise RuntimeError and write nothing. ``mode`` is the
    recorded run mode (``_run_mode``); None means ``args.segmentation``.
    """
    remaining = [run.namespace for run in runs]
    listed = _checked_reset_list(remaining, args)
    listed_set = set(listed)
    _refuse_other_mode(state, listed_set, getattr(args, "segmentation", None) if mode is None else mode)

    mismatched = []
    for run in runs:
        recorded = state.progress(run.namespace).plan_digest
        if run.namespace not in listed_set and recorded is not None and recorded != run.digest:
            mismatched.append(run.namespace)
    if mismatched:
        raise RuntimeError(
            "resume refused: turn plan changed since the recorded plan (list with "
            "--reset-namespace to start over): " + ", ".join(mismatched)
        )

    _refuse_unrecorded_dirty(remaining, state, listed_set)

    for namespace in listed:
        # Settle first: a reset deletes the PENDING/ENRICHING rows that prove a worker is alive.
        _await_stale_episode_settlement([namespace], timeout_s=args.reset_settle_timeout)
        _reset_namespace(admin, args.menhir_url, namespace)
        state.record_reset(namespace)


def _checked_reset_list(remaining: list[str], args: argparse.Namespace) -> list[str]:
    listed = list(args.reset_namespace or [])
    if args.reset_file:
        listed.extend(_read_reset_file(args.reset_file))
    listed = list(dict.fromkeys(listed))
    not_remaining = [ns for ns in listed if ns not in remaining]
    if not_remaining:
        raise RuntimeError(
            "reset refused: listed namespaces are not among the remaining (non-manifested) "
            "namespaces: " + ", ".join(not_remaining)
        )
    return listed


def _refuse_other_mode(state: IngestStateLog, listed_set: set[str], mode: str | None) -> None:
    # One build, one segmentation mode: a resume under another mode would ingest the namespaces
    # it has not started differently from those it has (including finished ones).
    other_mode = sorted(
        ns for ns in state.namespaces()
        if ns not in listed_set
        and state.progress(ns).segmentation not in (None, mode)
    )
    if mode is not None and other_mode:
        modes = sorted({str(state.progress(ns).segmentation) for ns in other_mode})
        raise RuntimeError(
            f"resume refused: namespaces were ingested with segmentation {modes}, this run uses "
            f"{mode!r}: " + ", ".join(other_mode[:10])
        )


def _refuse_before_planning(
    remaining: list[str], state: IngestStateLog, args: argparse.Namespace, mode: str | None
) -> None:
    """The startup refusals that do not need the turn plan, run before a paid writer prepass."""
    listed_set = set(_checked_reset_list(remaining, args))
    _refuse_other_mode(state, listed_set, mode)
    _refuse_unrecorded_dirty(remaining, state, listed_set)


def _refuse_unrecorded_dirty(remaining: list[str], state: IngestStateLog, listed_set: set[str]) -> None:
    unrecorded = [
        ns for ns in remaining
        if ns not in listed_set and state.progress(ns).plan_digest is None
    ]
    if unrecorded:
        counts = _ns_state_counts_many(unrecorded)
        unreadable = [ns for ns in unrecorded if int(counts.get(ns, {}).get("total", -1)) < 0]
        if unreadable:
            raise RuntimeError(
                "resume refused: could not read the Episodic row count (graph unreachable) for: "
                + ", ".join(unreadable)
            )
        dirty = [ns for ns in unrecorded if int(counts[ns]["total"]) > 0]
        if dirty:
            raise RuntimeError(
                "resume refused: namespace has graph data but no recorded plan; list it with "
                "--reset-namespace: " + ", ".join(dirty)
            )


def _ingest_pool(
    runs: list[NamespaceRun],
    client: HttpMenhirClient,
    admin: httpx.Client,
    menhir_url: str,
    state: IngestStateLog,
    *,
    pool_size: int,
    drain_timeout: float,
    finalize,
    outage_timeout: float = 300.0,
    poll_s: float = 1.0,
    settle_gap_s: float = 2.0,
    max_provider_requeues: int = 3,
) -> list[str]:
    """Sliding pool: up to ``pool_size`` namespaces in flight, one active episode in each.

    P2: segment k+1 of a namespace is only submitted once segment k is READY or terminally
    FAILED, so Menhir workers can never reorder temporal updates inside a namespace. Each
    namespace then settles (PENDING and ENRICHING both zero on two readings at least
    ``settle_gap_s`` apart), gets one bounded sweep of failed evidence projections, and
    ``finalize(run, counts)`` writes its manifest row. Content-failure retries are recorded in
    the state log BEFORE the reset, so the one-retry budget survives a crash; provider-caused
    failures (quota/auth) spend no budget. Returns the namespaces finalized, in completion order.
    """
    queue = list(runs)
    running: list[NamespaceRun] = []
    completed: list[str] = []
    bad_since: float | None = None

    def submit_next(run: NamespaceRun) -> None:
        """Advance to the next *extractable* turn; evidence-only turns hold no slot."""
        retried = state.progress(run.namespace).retried_episodes
        while run.cursor < len(run.plan):
            turn = run.plan[run.cursor]
            run.cursor += 1
            turn_kwargs = (
                {"source": turn.source, "writer_index": turn.writer_index}
                if turn.writer_index >= 0
                else {}
            )
            if turn.store_only:
                turn_kwargs["store_only"] = True
            episode_uuid = _ingest_turn(
                client,
                run.namespace,
                turn.role,
                turn.content,
                occurred_at=turn.occurred_at,
                session_id=turn.session_id,
                extract=turn.extract,
                evidence_text=turn.evidence_text or turn.content,
                turn_key=turn.turn_key or None,
                evidence_cache=run.evidence_cache,
                ingest_key=_ingest_key(turn),
                **turn_kwargs,
            )
            run.last_progress_at = time.time()
            if episode_uuid:
                run.active = (episode_uuid, episode_uuid in retried)
                return
        run.active = None
        run.phase = "settling"
        run.settle_polls = 0

    def note_bad() -> None:
        """Start (or continue) the outage clock and enforce it now, not at the end of a cycle:
        with hanging requests one cycle can outlast the whole limit."""
        nonlocal bad_since
        now = time.time()
        if bad_since is None:
            bad_since = now
        if now - bad_since >= outage_timeout:
            raise ServerUnavailableError(
                f"menhir or neo4j unreachable for {outage_timeout:.0f}s; stopping "
                "(re-run with the same arguments to resume)"
            )

    provider_requeues: dict[str, int] = {}

    def on_requeued(namespace: str, episode_uuid: str, kind: str) -> None:
        if kind == "provider":
            provider_requeues[episode_uuid] = provider_requeues.get(episode_uuid, 0) + 1
            if provider_requeues[episode_uuid] > max_provider_requeues:
                raise ProviderStoppedError(
                    f"projection {episode_uuid} ({namespace}) failed on the provider account "
                    f"{max_provider_requeues} times this run; stopping (fix the provider "
                    "account, then resume)"
                )
        state.record_requeued(namespace, episode_uuid, kind)

    while queue or running:
        # Before any poll, retry or submit: exhausted quota fails everything fast otherwise.
        healthy = _check_menhir_health(admin, menhir_url)
        if not healthy:
            note_bad()
        progressed = False

        while queue and len(running) < pool_size:
            run = queue.pop(0)
            if state.progress(run.namespace).plan_digest != run.digest:
                state.record_plan(run.namespace, run.digest, len(run.plan), run.segmentation)
            run.last_progress_at = time.time()
            running.append(run)
            submit_next(run)
            progressed = True

        reads = 0
        unreadable = 0

        def unreadable_read() -> None:
            nonlocal unreadable
            unreadable += 1
            if not healthy or unreadable == reads:
                note_bad()

        for run in list(running):
            if run.phase != "submitting" or run.active is None:
                continue
            episode_uuid, already_retried = run.active
            row = _episode_processing(admin, menhir_url, episode_uuid)
            reads += 1
            if row is None:
                unreadable_read()
                continue
            processing_state = str(row.get("processing_state") or "").upper().rsplit(".", 1)[-1]
            if processing_state == "READY":
                run.active = None
                submit_next(run)
                progressed = True
            elif processing_state == "FAILED":
                if _provider_caused(row.get("processing_error")):
                    # Not a content failure: spends no budget and is never terminal. Raises
                    # ProviderStoppedError while Menhir still reports the condition.
                    provider_requeues[episode_uuid] = provider_requeues.get(episode_uuid, 0) + 1
                    if provider_requeues[episode_uuid] > max_provider_requeues:
                        raise ProviderStoppedError(
                            f"episode {episode_uuid} ({run.namespace}) failed on the provider "
                            f"account {max_provider_requeues} times this run; stopping (fix the "
                            "provider account, then resume)"
                        )
                    if _requeue_provider_failure(admin, menhir_url, episode_uuid):
                        state.record_requeued(run.namespace, episode_uuid, "provider")
                        run.last_progress_at = time.time()
                        progressed = True
                    else:
                        unreadable_read()
                    continue
                if not already_retried:
                    # Budget is spent durably BEFORE the reset, so a crash cannot grant a second.
                    state.record_retry(run.namespace, episode_uuid, "episode")
                    try:
                        reset = _backend(
                            admin, menhir_url, "force_reset_failed_episode",
                            {"episode_uuid": episode_uuid},
                        )
                    except Exception:
                        reset = False
                    if reset:
                        try:
                            _backend(
                                admin, menhir_url, "enqueue_pending_episode",
                                {"episode_uuid": episode_uuid},
                            )
                        except Exception:
                            # Lease recovery will still find the PENDING row; keep waiting on it.
                            pass
                        state.record_requeued(run.namespace, episode_uuid, "episode")
                        run.active = (episode_uuid, True)
                        run.last_progress_at = time.time()
                        progressed = True
                        continue
                    print(f"    reset of FAILED {episode_uuid} ({run.namespace}) failed; its one "
                          "retry is spent and it stays FAILED", flush=True)
                # Terminal: no further retry could reorder this namespace; counts record it.
                run.active = None
                submit_next(run)
                progressed = True

        settling = [run for run in running if run.phase == "settling"]
        settled: list[tuple[NamespaceRun, dict]] = []
        if settling:
            counts = _ns_state_counts_many([run.namespace for run in settling])
            now = time.time()
            for run in settling:
                snapshot = counts.get(run.namespace) or {}
                reads += 1
                pending = int(snapshot.get("pending", -1))
                enriching = int(snapshot.get("enriching", -1))
                if pending < 0 or enriching < 0:
                    run.settle_polls = 0
                    unreadable_read()
                    continue
                if pending or enriching:
                    run.settle_polls = 0
                    continue
                # Two zero readings at least settle_gap_s apart, not two back-to-back polls.
                if run.settle_polls == 0:
                    run.settle_first_at = now
                run.settle_polls += 1
                run.last_progress_at = now
                if run.settle_polls >= 2 and now - run.settle_first_at >= settle_gap_s:
                    settled.append((run, snapshot))

        # Watchdog on this cycle's fresh readings, BEFORE any finalize: a long consolidation of
        # one namespace must not count as "no progress" for the others.
        now = time.time()
        for run in running:
            if now - run.last_progress_at >= drain_timeout:
                where = run.active[0] if run.active else run.phase
                raise RuntimeError(
                    f"namespace {run.namespace} made no progress for {drain_timeout:.0f}s "
                    f"({where}); refusing to continue"
                )

        for run, snapshot in settled:
            try:
                requeued = _retry_failed_evidence_projections(
                    [run.namespace],
                    admin,
                    menhir_url,
                    skip=set(state.progress(run.namespace).retried_projections),
                    on_retry=lambda ns, uuid: state.record_retry(ns, uuid, "projection"),
                    on_requeued=on_requeued,
                )
            except GraphUnreadable:
                run.settle_polls = 0
                reads += 1
                unreadable_read()
                continue
            if requeued.get(run.namespace):
                run.settle_polls = 0
                run.last_progress_at = time.time()
                progressed = True
                continue
            finalize(run, snapshot)
            running.remove(run)
            completed.append(run.namespace)
            progressed = True

        if healthy and not (reads and unreadable == reads):
            bad_since = None

        if (queue or running) and not progressed:
            time.sleep(poll_s)

    return completed


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
                    help="max seconds one namespace may make no progress (and the final "
                         "drain may run) before the run stops")
    ap.add_argument(
        "--namespace-window",
        type=_positive_int,
        default=os.getenv("LME_INGEST_CONCURRENCY", "1"),
        help="sliding pool size: item namespaces ingested/enriched concurrently (default: 1)",
    )
    ap.add_argument(
        "--reset-namespace",
        action="append",
        default=[],
        metavar="NS",
        help="namespace to force-reset before ingesting (repeatable); nothing else is reset",
    )
    ap.add_argument(
        "--reset-file",
        default=None,
        help="file with one namespace to reset per line (blank lines and # comments ignored)",
    )
    ap.add_argument("--reset-settle-timeout", type=float, default=1200.0,
                    help="max seconds to wait for stale enrichment to settle before each reset")
    ap.add_argument("--outage-timeout", type=float, default=300.0,
                    help="stop with a non-zero exit after menhir/neo4j fail this many seconds")
    ap.add_argument(
        "--manifest-item-limit",
        type=_positive_int,
        help=(
            "hard checkpoint: ingest only the first N loaded items while keeping --limit as "
            "the full fixture size for a later manifest-backed resume"
        ),
    )
    ap.add_argument("--consolidate-scalar", action="store_true",
                    help="require scalar Phase 3 consolidation before manifesting each namespace")
    ap.add_argument("--consolidate-counter", action="store_true")
    ap.add_argument("--consolidate-events", action="store_true")
    ap.add_argument("--consolidation-k", type=int, default=3)
    ap.add_argument("--consolidation-call-budget", type=int, default=50)
    ap.add_argument(
        "--segmentation",
        choices=["none", "sentence", "user-sentence", "adaptive", "adaptive-writer"],
        default=os.getenv("LME_SEGMENTATION", "sentence"),
        help=(
            "Turn segmentation strategy: "
            "'none' = no splitting (arm A), "
            "'sentence' = blind sentence split (arm B, default/legacy), "
            "'user-sentence' = split only user turns (arm C), "
            "'adaptive' = claim-aware segmentation (arm D), "
            "'adaptive-writer' = adaptive for user turns; assistant turns are evidence only and an "
            "LLM memory writer produces agent-tier memories from them (#168)"
        ),
    )
    ap.add_argument("--writer-granularity", choices=list(memory_writer.GRANULARITIES),
                    default=os.getenv("LME_WRITER_GRANULARITY") or "turn",
                    help="adaptive-writer: one writer request per assistant turn or per session")
    ap.add_argument("--writer-model", default=os.getenv("LME_WRITER_MODEL") or "gpt-6-luna",
                    help="adaptive-writer: OpenAI Flex model for the memory writer")
    ap.add_argument("--writer-checkpoint", default=None,
                    help="adaptive-writer: writer JSONL checkpoint (default: "
                         "writer-<manifest stem>-<granularity>.jsonl next to the manifest)")
    # Strings, validated only in writer mode: config.sh exports these empty under `set -a`, and a
    # typed default would make argparse reject every build in every mode.
    ap.add_argument("--writer-max-usd", default=os.getenv("LME_WRITER_MAX_USD") or None,
                    help="adaptive-writer: required spend cap in USD (or LME_WRITER_MAX_USD)")
    ap.add_argument("--writer-rates", default=os.getenv("LME_WRITER_RATES") or None,
                    help="adaptive-writer: required 'IN,OUT' USD per 1M tokens (or LME_WRITER_RATES)")
    ap.add_argument("--writer-workers", type=_positive_int, default=16,
                    help="adaptive-writer: concurrent writer requests")
    args = ap.parse_args(argv)
    # argparse does not check an env-supplied default against `choices`.
    if args.segmentation == "adaptive-writer" and args.writer_granularity not in memory_writer.GRANULARITIES:
        ap.error(f"--writer-granularity must be one of {memory_writer.GRANULARITIES}")
    return args


def _finite_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _run_mode(args: argparse.Namespace) -> str | None:
    """The mode a namespace's plan is recorded under; a resume must match it for every namespace.

    Writer mode includes granularity, model and prompt sha, so a build continued under another
    writer configuration is refused, not mixed into one graph.
    """
    segmentation = getattr(args, "segmentation", None)
    if segmentation != "adaptive-writer":
        return segmentation
    granularity = args.writer_granularity
    return f"adaptive-writer:{granularity}:{args.writer_model}:{memory_writer.prompt_sha(granularity)[:16]}"


def _run_writer_prepass(
    args: argparse.Namespace,
    adapter: LongMemEvalMemoryAdapter,
    remaining: list[dict],
    manifest_path: Path,
) -> tuple[dict | None, int]:
    """Run (or replay) the memory writer for the not-yet-ingested items.

    Returns ``(items, 0)``, or ``(None, exit_code)`` after printing the error: 2 for a missing
    key/budget/rates (nothing was sent), 1 for a writer or Flex failure (nothing was ingested).
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    max_usd = _finite_float(args.writer_max_usd) if args.writer_max_usd is not None else None
    rates: tuple[float, float] | None = None
    parts = str(args.writer_rates or "").split(",")
    if len(parts) == 2:
        rate_in, rate_out = (_finite_float(part) for part in parts)
        # Zero, negative or non-finite rates would keep the computed cost under any cap.
        if rate_in is not None and rate_out is not None and rate_in > 0 and rate_out > 0:
            rates = (rate_in, rate_out)
    problems = []
    if not api_key:
        problems.append("OPENAI_API_KEY is not set")
    if max_usd is None or max_usd <= 0:
        problems.append("--writer-max-usd (or LME_WRITER_MAX_USD) must be a finite number > 0")
    if rates is None:
        problems.append("--writer-rates (or LME_WRITER_RATES) must be 'IN,OUT' USD per 1M tokens, both > 0")
    if problems:
        print("adaptive-writer: " + "; ".join(problems), file=sys.stderr)
        return None, 2

    sessions: list[memory_writer.WriterSession] = []
    for index, item in enumerate(remaining):
        namespace = _namespace(str(item.get("question_id") or f"lme-{index}"), args.namespace_prefix)
        sessions.extend(_writer_sessions(adapter, item, namespace))
    # Items are keyed by (namespace, session id, turn): a repeated session id would merge two
    # sessions' items and only fail the plan's duplicate-key check after the writer has spent.
    repeated = sorted(f"{ns}|{sid}" for (ns, sid), n in
                      Counter((s.namespace, s.session_id) for s in sessions).items() if n > 1)
    if repeated:
        print("adaptive-writer: repeated session ids: " + ", ".join(repeated[:10]), file=sys.stderr)
        return None, 2

    from archolith_bench.core.openai_flex import FlexError, run_flex

    # One checkpoint per manifest, so the manifest's run lock also serialises the checkpoint;
    # builds sharing a results directory never rewrite each other's paid rows.
    checkpoint = (
        Path(args.writer_checkpoint)
        if args.writer_checkpoint
        else manifest_path.parent / f"writer-{manifest_path.stem}-{args.writer_granularity}.jsonl"
    )
    try:
        items, report = memory_writer.write_memories(
            sessions,
            granularity=args.writer_granularity,
            model=args.writer_model,
            checkpoint=checkpoint,
            run_flex=run_flex,
            flex_kwargs={
                "api_key": api_key,
                "max_usd": max_usd,
                "rates": rates,
                "workers": args.writer_workers,
            },
        )
    except (memory_writer.WriterError, FlexError) as exc:
        print(f"adaptive-writer failed: {exc}", file=sys.stderr)
        return None, 1
    print(
        f"adaptive-writer: requests={report.requests} items={report.items} dropped={report.dropped} "
        f"prompt_tokens={report.prompt_tokens} completion_tokens={report.completion_tokens} "
        f"cost_usd={report.cost_usd:.4f}",
        file=sys.stderr,
    )
    return items, 0

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


def _iter_sessions(
    adapter: LongMemEvalMemoryAdapter,
    item: dict,
    namespace: str,
) -> Iterator[tuple[str, str | None, str | None, list[tuple[int, str, str]]]]:
    """``(session_id, session_date, occurred_at, [(raw turn index, role, content)])`` per session.

    Empty-content turns are skipped; indices are over the raw session. Shared by the turn plan and
    the memory-writer prepass so both key a turn identically.
    """
    haystack_dates = item.get("haystack_dates") or []
    haystack_session_ids = item.get("haystack_session_ids") or []
    for session_index, session in enumerate(adapter.sessions(item)):
        session_date = haystack_dates[session_index] if session_index < len(haystack_dates) else None
        raw_session_id = (
            haystack_session_ids[session_index]
            if session_index < len(haystack_session_ids)
            else None
        )
        # Namespace-qualify the session id. LongMemEval reuses raw session ids across temporal
        # variants; sharing them lets Graphiti deduplicate the second namespace against the first.
        session_id = (
            f"{namespace}-{raw_session_id}"
            if raw_session_id
            else f"{namespace}-s{session_index}"
        )
        turns = [
            (turn_index, turn.get("role", "user"), turn.get("content", ""))
            for turn_index, turn in enumerate(session)
            if turn.get("content", "")
        ]
        yield session_id, session_date, _parse_lme_date(session_date), turns


def _writer_sessions(
    adapter: LongMemEvalMemoryAdapter,
    item: dict,
    namespace: str,
) -> list[memory_writer.WriterSession]:
    return [
        memory_writer.WriterSession(
            namespace=namespace,
            session_id=session_id,
            date=session_date or "",
            turns=tuple(memory_writer.SessionTurn(index=i, role=r, content=c) for i, r, c in turns),
        )
        for session_id, session_date, _occurred_at, turns in _iter_sessions(adapter, item, namespace)
    ]


def _iter_item_turns(
    adapter: LongMemEvalMemoryAdapter,
    item: dict,
    namespace: str,
    segmentation: str,
    writer_items: dict | None = None,
) -> Iterator[IngestTurn]:
    writer_mode = segmentation == "adaptive-writer"
    for session_id, _session_date, occurred_at, session_turns in _iter_sessions(adapter, item, namespace):
        for turn_index, role, content in session_turns:
            contents: list[str]
            # CONTEXT_ONLY turns take the evidence-only path below instead of
            # being merged into a neighbour. Merging would concatenate two source
            # turns into one record, destroying the per-turn TurnEvidence
            # boundaries and counts the scalar audit reads, and -- when the two
            # turns had different roles -- would file assistant words under
            # declarant="user".
            extract = True
            store_only = False
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
            elif segmentation in ("adaptive", "adaptive-writer"):
                mode = decide_segmentation(role, content)
                if writer_mode and role.strip().lower() == "assistant":
                    # The writer replaces assistant extraction: the turn is evidence only, and the
                    # writer's items (below) are the agent-tier memories grounded on it.
                    contents = [content]
                    extract = False
                elif mode == SegmentationMode.SKIP:
                    contents = []
                elif mode == SegmentationMode.CONTEXT_ONLY:
                    # Captured as evidence, never extracted. Purely interrogative
                    # or phatic turns produce only {"name":"user"} with zero edges,
                    # so sending them to extraction fails the episode; skipping
                    # extraction keeps the turn's own evidence record intact.
                    contents = [content]
                    extract = False
                    # A user's question still states facts about them (archolith-bench#43): keep
                    # it as a store-only source memory -- embedded, never extracted. Assistant
                    # CONTEXT_ONLY turns stay evidence only.
                    if STORE_ONLY_CONTEXT and (role.strip().lower() or "user") == "user":
                        extract = True
                        store_only = True
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
            for segment_index, segmented_content in enumerate(contents):
                yield IngestTurn(
                    role=role,
                    content=segmented_content,
                    occurred_at=occurred_at,
                    session_id=session_id,
                    extract=extract,
                    # Evidence is the whole turn, even when the episode is one segment of it.
                    evidence_text=content,
                    turn_key=turn_key,
                    segment_index=segment_index,
                    store_only=store_only,
                )
            if writer_mode and role.strip().lower() == "assistant":
                for writer_index, writer_item in enumerate(
                    (writer_items or {}).get((namespace, session_id, turn_index), [])
                ):
                    yield IngestTurn(
                        role="agent",
                        content=writer_item.text,
                        occurred_at=occurred_at,
                        session_id=session_id,
                        extract=True,
                        evidence_text=content,
                        turn_key=turn_key,
                        segment_index=0,
                        source=memory_writer.WRITER_SOURCE,
                        writer_index=writer_index,
                    )


def _ingest_key(turn: IngestTurn) -> str:
    """Idempotency key for one extractable segment: stable across attempts (P1), <= 256 chars."""
    if turn.writer_index >= 0:
        key = f"{turn.turn_key}#w{turn.writer_index}"
    else:
        key = f"{turn.turn_key}#s{turn.segment_index}"
    if len(key) > 256:
        return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key


def _turn_plan(
    adapter: LongMemEvalMemoryAdapter,
    item: dict,
    namespace: str,
    segmentation: str,
    writer_items: dict | None = None,
) -> list[IngestTurn]:
    if writer_items is None:
        plan = list(_iter_item_turns(adapter, item, namespace, segmentation))
    else:
        plan = list(_iter_item_turns(adapter, item, namespace, segmentation, writer_items))
    # P1 needs one key per segment: a duplicate would replay one segment in place of another.
    keys = Counter(_ingest_key(turn) for turn in plan if turn.extract)
    duplicates = sorted(key for key, n in keys.items() if n > 1)
    if duplicates:
        raise RuntimeError(
            f"turn plan for {namespace} repeats ingest keys (duplicate turns in the source?): "
            + ", ".join(duplicates[:5])
        )
    return plan


def _hold_run_lock(path: Path):
    """Exclusive OS lock on *path* for this process's lifetime (released on exit or crash)."""
    handle = open(path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            f"another ingest run holds {path}; refusing to run two clients on one manifest"
        ) from exc
    return handle


def _turn_plan_digest(
    plan: list[IngestTurn],
    segmentation: str,
    *,
    writer_model: str | None = None,
    writer_granularity: str | None = None,
) -> str:
    """Digest of everything that decides what gets submitted; a resume must match it (P5)."""
    def _h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _turn_payload(t: IngestTurn) -> list:
        row = [t.turn_key, t.segment_index, t.extract, t.role, t.occurred_at, t.session_id,
               _h(t.content), _h(t.evidence_text)]
        if t.writer_index >= 0:
            row += [t.source, t.writer_index]
        if t.store_only:
            # Appended only when set, so a plan without store-only turns keeps its old digest.
            row.append("store_only")
        return row

    payload = {
        "v": 1,
        "segmentation": segmentation,
        "split_min_length": _SPLIT_MIN_LENGTH,
        "turns": [_turn_payload(t) for t in plan],
    }
    if segmentation == "adaptive-writer":
        payload["writer"] = {
            "model": writer_model,
            "granularity": writer_granularity,
            "prompt_sha": memory_writer.prompt_sha(writer_granularity or "turn"),
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _h(canonical)


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
    requested_lanes = {
        "scalar": args.consolidate_scalar,
        "counter": args.consolidate_counter,
        "event": args.consolidate_events,
    }
    consolidate_requested = any(requested_lanes.values())

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
    # Held for the whole run, before the manifest is read: two clients on one manifest would
    # each spend the same retry budget and overwrite each other's rows.
    run_lock = _hold_run_lock(IngestStateLog.path_for(manifest_path).with_suffix(".lock"))
    manifest: list[dict] = []
    done_ids: set[str] = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest:
            if consolidate_requested:
                if row.get("consolidation_requested") != requested_lanes:
                    raise RuntimeError("resume refused: manifest processing lanes differ or are missing")
                errors = consolidation_errors(row.get("consolidation_result", {}), requested_lanes,
                                              namespace=row["namespace"], k=args.consolidation_k)
                if errors:
                    raise RuntimeError("resume refused: " + "; ".join(errors))
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

    state = IngestStateLog(IngestStateLog.path_for(manifest_path))
    run_mode = _run_mode(args)
    writer_items: dict | None = None
    writer_digest_kwargs: dict = {}
    if args.segmentation == "adaptive-writer":
        # Refuse what does not depend on the plan before the writer spends anything.
        _refuse_before_planning(
            [_namespace(str(it.get("question_id") or f"lme-{i}"), args.namespace_prefix)
             for i, it in enumerate(remaining)],
            state, args, run_mode,
        )
        writer_items, writer_exit = _run_writer_prepass(args, adapter, remaining, manifest_path)
        if writer_items is None:
            return writer_exit
        writer_digest_kwargs = {
            "writer_model": args.writer_model,
            "writer_granularity": args.writer_granularity,
        }

    runs = []
    for index, item in enumerate(remaining):
        question_id = str(item.get("question_id") or f"lme-{index}")
        namespace = _namespace(question_id, args.namespace_prefix)
        plan = _turn_plan(adapter, item, namespace, args.segmentation, writer_items=writer_items)
        runs.append(NamespaceRun(
            item=item,
            question_id=question_id,
            namespace=namespace,
            plan=plan,
            digest=_turn_plan_digest(plan, args.segmentation, **writer_digest_kwargs),
            segmentation=run_mode,
        ))
    _startup_checks(runs, state, args, admin, mode=run_mode)

    completed_this_run: list[str] = []

    def finalize(run: NamespaceRun, drained: dict[str, int]) -> None:
        if consolidate_requested:
            _require_no_failed_episodes({run.namespace: drained})
        scalar_result: dict = {}
        scalar_counts: dict[str, int] = {}
        if consolidate_requested:
            scalar_result = _consolidate_lanes(
                admin,
                args.menhir_url,
                run.namespace,
                k=args.consolidation_k,
                call_budget=args.consolidation_call_budget,
                scalar_state=args.consolidate_scalar,
                counter_state=args.consolidate_counter,
                event_history=args.consolidate_events,
            )
        if args.consolidate_scalar:
            scalar_counts = _scalar_counts(run.namespace)
            if scalar_counts["turn_evidence"] <= 0:
                raise RuntimeError(
                    f"no TurnEvidence captured for scalar namespace {run.namespace}"
                )
        # Recorded for final acceptance, which refuses any violation or unreadable count.
        integrity = _integrity_counts(run.namespace, _submitted_epoch_millis(run.item))
        user_turns_submitted = _user_turns_submitted(run.plan)
        progress = state.progress(run.namespace)
        # Retries that happened (reset succeeded), not budget spent on a failed reset.
        requeued = progress.requeued.get("episode", 0) + progress.requeued.get("projection", 0)

        manifest.append({
            "question_id": run.question_id,
            "namespace": run.namespace,
            "fixture": str(Path(args.fixture).resolve()) if args.fixture else None,
            "question": adapter.question(run.item),
            "answer": str(run.item.get("answer", "")),
            "question_type": run.item.get("question_type"),
            "turns": len(run.plan),
            "episodes": drained.get("total"),
            "ready": drained.get("ready"),
            "failed_remaining": drained.get("failed"),
            "failed_requeued": requeued,
            "provider_requeued": progress.requeued.get("provider", 0),
            "enrichment_llm_tasks": drained.get("llm_tasks"),
            "processing_attempts": drained.get("processing_attempts"),
            "drain_timed_out": False,
            "namespace_window": args.namespace_window,
            "scalar_consolidated": bool(
                args.consolidate_scalar
                and int(scalar_result.get("scalar_namespaces_processed", 0)) == 1
                and int(scalar_result.get("scalar_llm_calls", 0)) >= args.consolidation_k
            ),
            "consolidation_requested": requested_lanes,
            "consolidation_k": args.consolidation_k,
            "consolidation_result": scalar_result,
            "consolidation_llm_calls": int(scalar_result.get("llm_calls", 0)),
            "event_llm_calls": int(scalar_result.get("event_llm_calls", 0)),
            "scalar_llm_calls": int(scalar_result.get("scalar_llm_calls", 0)),
            "scalar_states_written": int(scalar_result.get("scalar_states_written", 0)),
            **scalar_counts,
            "user_turns_submitted": user_turns_submitted,
            **integrity,
        })
        _write_manifest(manifest_path, manifest)
        state.record_done(run.namespace, run.question_id)
        completed_this_run.append(run.namespace)
        elapsed = time.time() - t_all
        rate = len(completed_this_run) / elapsed if elapsed else 0
        eta = (len(remaining) - len(completed_this_run)) / rate if rate else 0
        print(
            f"[{len(done_ids)+len(completed_this_run)}/{len(items)}] {run.question_id} "
            f"ns={run.namespace} turns={len(run.plan)} episodes={drained.get('total')} "
            f"ready={drained.get('ready')} failed={drained.get('failed')} "
            f"requeued={requeued} llm_tasks={drained.get('llm_tasks')} "
            f"total={elapsed:.0f}s eta={eta/3600:.1f}h",
            flush=True,
        )

    _ingest_pool(
        runs,
        client,
        admin,
        args.menhir_url,
        state,
        pool_size=args.namespace_window,
        drain_timeout=args.drain_timeout,
        outage_timeout=args.outage_timeout,
        finalize=finalize,
    )

    # Final global drain over EVERY manifested namespace, not just this run's: a drain that timed
    # out in an earlier run must be re-checked by a plain re-run, never skipped (per-namespace
    # settling already gated each row; this confirms the global queue is empty too).
    manifested = [str(row["namespace"]) for row in manifest]
    if manifested:
        final = _drain_many(
            manifested, admin, args.menhir_url, timeout_s=args.drain_timeout
        )
        timed_out = [ns for ns, drained in final.items() if drained.get("timed_out")]
        if timed_out:
            raise RuntimeError(
                "final drain timed out after the namespace pool finished: " + ", ".join(timed_out)
            )

    run_lock.close()
    outcome = "CHECKPOINT" if len(target_items) < len(items) else "DONE"
    print(
        f"\n{outcome}: manifest has {len(manifest)}/{len(items)} items -> {args.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
