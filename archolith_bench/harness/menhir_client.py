"""
Menhir memory client implementations for Mode-B benchmark driver.

StubMenhirClient: deterministic in-memory backend for offline/deterministic tests.
HttpMenhirClient: HTTP client scaffolded for a real throwaway menhir instance.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import uuid
from typing import Any

import httpx

# Selection-failure gates that indicate no single event object could be anchored.
# When the event authority layer reports one of these, recall emits an advisory and
# suppresses the ordinary result items (there is no authoritative event to compare them against).
_BLOCKING_EVENT_GATES = frozenset({"anchor", "ambiguity", "time", "scope", "no_candidate"})


def _format_duration_alias(value: Any, unit: str) -> str | None:
    """Return a human-readable duration and clock form for integral seconds."""
    if unit.lower() not in {"s", "sec", "second", "seconds"}:
        return None
    try:
        seconds_value = Decimal(str(value))
    except InvalidOperation:
        return None
    if not seconds_value.is_finite() or seconds_value < 0 or seconds_value != seconds_value.to_integral_value():
        return None

    total_seconds = int(seconds_value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    clock = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
    return f"{' '.join(parts)}; {clock}"


def _format_authority_value(record: dict[str, Any]) -> str:
    """Render a normalized scalar value without discarding its unit or familiar form."""
    value = record.get("value")
    unit = str(record.get("unit") or "").strip()
    rendered = str(value)
    if unit and not rendered.lower().endswith(f" {unit.lower()}"):
        rendered = f"{rendered} {unit}"
    if str(record.get("value_kind") or "").strip().lower() == "duration":
        alias = _format_duration_alias(value, unit)
        if alias:
            rendered = f"{rendered} ({alias})"
    return rendered


def _format_source_time_evidence(item: dict[str, Any]) -> str:
    """Render source/world time, explicitly distinguishing missing time from belief time."""
    if "temporal_facts" not in item:
        return ""
    temporal_facts = item.get("temporal_facts")
    if not isinstance(temporal_facts, list) or not temporal_facts:
        return "source time: unknown"

    lines = ["source-time evidence (when the fact was true):"]
    for temporal_fact in temporal_facts:
        if not isinstance(temporal_fact, dict):
            continue
        fact = str(temporal_fact.get("fact") or "").strip()
        if not fact:
            # Graphiti can leave timestamp-only bookkeeping relationships on an
            # entity. They do not identify a fact and therefore cannot provide
            # useful source-time evidence to the answer model.
            continue
        valid_at = str(temporal_fact.get("valid_at") or "").strip()
        invalid_at = str(temporal_fact.get("invalid_at") or "").strip()
        if valid_at and invalid_at:
            happened = f"{valid_at} through {invalid_at}"
        else:
            happened = valid_at or "unknown"
        role = str(temporal_fact.get("temporal_role") or "").strip().replace("_", " ")
        role_label = f" | belief: {role}" if role else ""
        lines.append(f"- {happened} | {fact}{role_label}")
    if len(lines) == 1:
        return "source time: unknown"
    return "\n".join(lines)


def _recall_item_text(item: dict[str, Any]) -> str:
    """Return the best human-readable text from one Menhir recall result."""
    name = item.get("name") if isinstance(item.get("name"), str) else ""
    body = next(
        (
            item[key]
            for key in ("content", "text", "summary", "fact")
            if isinstance(item.get(key), str) and item[key].strip()
        ),
        "",
    )
    if name and body and name.strip() != body.strip():
        text = f"{name.strip()}: {body.strip()}"
    else:
        text = body.strip() or name.strip()
    source_time = _format_source_time_evidence(item)
    if text and source_time:
        return f"{text}\n{source_time}"
    return text or source_time


def _format_authority_record(record: dict[str, Any]) -> str:
    """Render structured scalar authority as compact, provenance-rich LLM context."""
    kind = str(record.get("kind") or "").strip().lower()
    status = str(record.get("status") or "").strip().lower()
    founded = record.get("has_foundation") is not False
    if kind == "current" and status == "leads" and founded:
        lines = [
            "[AUTHORITATIVE CURRENT MEMORY]",
            "status: leads; prefer this value over conflicting related memories.",
        ]
    else:
        label = " | ".join(part for part in (kind, status) if part) or "unclassified"
        lines = ["[STRUCTURED MEMORY VERDICT]", f"status: {label}"]

    subject = str(record.get("subject") or "").strip()
    attribute = str(record.get("attribute") or "value").strip().replace("_", " ")
    scope = str(record.get("scope") or "").strip().replace("_", " ")
    value = _format_authority_value(record)
    fact_name = f"{attribute} ({scope})" if scope else attribute
    subject_prefix = f"{subject} — " if subject else ""
    lines.append(f"current fact: {subject_prefix}{fact_name} = {value}")

    valid_at = str(record.get("valid_at") or "").strip()
    if valid_at:
        lines.append(f"valid at: {valid_at}")

    contributors = record.get("contributors")
    if isinstance(contributors, list):
        support: list[str] = []
        for contributor in contributors:
            if not isinstance(contributor, dict):
                continue
            quote = str(contributor.get("stated_span") or "").strip()
            operation = str(contributor.get("operation") or "").strip().lower()
            source_time = str(contributor.get("valid_at") or "").strip()
            parts = [part for part in (operation, f'"{quote}"' if quote else "", source_time) if part]
            if parts:
                support.append(" | ".join(parts))
        if support:
            lines.append("provenance:")
            lines.extend(f"- {item}" for item in support)
    return "\n".join(lines)


def _event_authority_display(record: dict[str, Any]) -> str:
    """Pick the event label: object_display, else predicate (or subject+predicate fallback)."""
    object_display = record.get("object_display")
    if isinstance(object_display, str) and object_display.strip():
        return object_display.strip()
    subject = str(record.get("subject_uuid") or "").strip()
    predicate = str(record.get("predicate") or "").strip().replace("_", " ")
    if subject and predicate:
        return f"{subject} — {predicate}"
    return predicate


def _event_evidence_identities(record: dict[str, Any]) -> str:
    """Render the event's evidence identities from its known identity fields."""
    ids: list[str] = []
    for field in ("assertion_key", "episode_uuid", "turn_evidence_uuid"):
        value = record.get(field)
        if value is not None and str(value).strip():
            ids.append(str(value).strip())
    return ", ".join(ids)


def _format_event_authority_record(record: dict[str, Any]) -> str:
    """Render a Menhir EventAuthorityVerdict as compact, provenance-rich LLM context.

    Only ``status="leads"`` (normally ``gate="pass"``) renders an authoritative event
    lead with the selected object, its evidence identities, and a preference over
    conflicting memories. Any other status renders a non-blocking advisory that invents
    no object. A blocking selection-failure gate (anchor, ambiguity, time, scope,
    no_candidate) is handled separately by the caller, which suppresses the ordinary
    result items because there is no authoritative event to rank them against.
    """
    gate = str(record.get("gate") or "").strip().lower()
    status = str(record.get("status") or "").strip().lower()
    if status != "leads":
        lines = ["[EVENT HISTORY VERDICT | advisory]"]
        reason = record.get("reason")
        if isinstance(reason, str) and reason.strip():
            lines.append(f"note: {reason.strip()}")
        if gate:
            lines.append(f"gate: {gate}")
        return "\n".join(lines)

    lines = ["[AUTHORITATIVE EVENT HISTORY]"]
    display = _event_authority_display(record)
    if display:
        lines.append(f"event: {display}")
    predicate = str(record.get("predicate") or "").strip().replace("_", " ")
    if predicate:
        lines.append(f"predicate: {predicate}")
    object_key = record.get("object_key")
    if isinstance(object_key, str) and object_key.strip():
        lines.append(f"object key: {object_key.strip()}")
    valid_at = str(record.get("valid_at") or "").strip()
    if valid_at:
        lines.append(f"valid at: {valid_at}")
    time_basis = str(record.get("time_basis") or "").strip()
    if time_basis:
        lines.append(f"time basis: {time_basis}")
    domain = str(record.get("domain") or "").strip()
    if domain:
        lines.append(f"domain: {domain}")
    stated_span = str(record.get("stated_span") or "").strip()
    if stated_span:
        lines.append(f'quote: "{stated_span}"')
    evidence = _event_evidence_identities(record)
    if evidence:
        lines.append(f"evidence identities: {evidence}")
    if gate:
        lines.append(f"gate: {gate}")
    lines.append("preference: prefer this event over conflicting related memories")
    return "\n".join(lines)


class StubMenhirClient:
    """In-memory menhir stub for offline, deterministic testing."""

    def __init__(self) -> None:
        """Initialize empty group storage."""
        self._groups: dict[str, list[str]] = {}

    def new_group(self) -> str:
        """Return a fresh isolated namespace id."""
        return uuid.uuid4().hex

    def ingest(
        self,
        group_id: str,
        role: str,
        content: str,
        *,
        occurred_at: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        diff: str | None = None,
        wait: bool = True,
        flagged: bool = False,
        bootstrap_scope: str | None = None,
        turn_evidence_uuid: str | None = None,
    ) -> None:
        """Append a formatted snippet to the group's list."""
        if not content:
            return
        if group_id not in self._groups:
            self._groups[group_id] = []
        self._groups[group_id].append(f"{role}: {content}")

    def recall(
        self, group_id: str, query: str, limit: int = 10
    ) -> list[str]:
        """
        Return up to `limit` snippets for the group, ranked by word-token overlap.

        Snippets sharing the most lowercased word-tokens with the query come first
        (ties broken by insertion order), followed by remaining snippets in insertion order.
        If the group is unknown, return [].
        """
        if group_id not in self._groups:
            return []

        snippets = self._groups[group_id]
        query_tokens = set(query.lower().split())

        def score_snippet(indexed_snippet: tuple[int, str]) -> tuple[int, int]:
            """Return (negative overlap count, insertion index) for sorting."""
            index, snippet = indexed_snippet
            snippet_tokens = set(snippet.lower().split())
            overlap = len(query_tokens & snippet_tokens)
            return (-overlap, index)

        ranked = sorted(enumerate(snippets), key=score_snippet)
        return [snippet for _, snippet in ranked[:limit]]

    def reset(self, group_id: str) -> None:
        """Delete the group's entry if present."""
        self._groups.pop(group_id, None)


class HttpMenhirClient:
    """HTTP client for a real menhir instance (OpenAI-compatible style)."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        ingest_path: str = "/api/memory",
        recall_path: str = "/api/recall",
        timeout: float = 180.0,
    ) -> None:
        """
        Initialize HTTP client against a real menhir instance.

        A benchmark "group_id" maps to a menhir namespace (silo). Memory is ingested
        with ``wait=true`` so enrichment completes before recall; teardown uses
        ``DELETE /api/namespace/{namespace}``.

        Args:
            base_url: Root URL of the menhir instance.
            api_key: Optional Bearer token (menhir MENHIR_API_KEY) for authorization.
            ingest_path: Episode ingest endpoint (default "/api/memory").
            recall_path: Recall endpoint (default "/api/recall").
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url
        self._ingest_path = ingest_path
        self._recall_path = recall_path
        self._client = httpx.Client(timeout=timeout)
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def __enter__(self):
        """Return this client for context-manager use."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        """Close the underlying HTTP client on context-manager exit."""
        self.close()

    def new_group(self) -> str:
        """Return a fresh isolated namespace id."""
        return uuid.uuid4().hex

    def record_turn_evidence(
        self,
        namespace: str,
        text: str,
        *,
        role: str = "user",
        declarant: str = "user",
        session_id: str | None = None,
        occurred_at: str | None = None,
        source_kind: str = "archolith-bench",
        source_client: str | None = "archolith_bench",
        turn_key: str | None = None,
    ) -> dict[str, Any]:
        """Capture one user turn as :TurnEvidence so a subsequent ``source="user"`` ingest
        can cite its UUID and pass the admission gate.

        ``occurred_at`` is the source/world time for historical replay. Menhir keeps it
        separate from the server-side ``recorded_at`` processing cursor.

        Returns ``{turn_id, created, recorded_at, occurred_at}``.
        """
        url = self._base_url.rstrip("/") + "/api/turn-evidence"
        payload: dict[str, Any] = {
            "text": text,
            "role": role,
            "declarant": declarant,
            "namespace": namespace,
            "source_kind": source_kind,
        }
        if source_client is not None:
            payload["source_client"] = source_client
        if session_id is not None:
            payload["session_id"] = session_id
        if occurred_at is not None:
            payload["occurred_at"] = occurred_at
        if turn_key is not None:
            payload["turn_key"] = turn_key
        response = self._client.post(url, json=payload, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def ingest(
        self,
        group_id: str,
        role: str,
        content: str,
        *,
        occurred_at: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        diff: str | None = None,
        wait: bool = True,
        flagged: bool = False,
        bootstrap_scope: str | None = None,
        turn_evidence_uuid: str | None = None,
    ) -> dict[str, Any] | None:
        """Ingest a snippet as a menhir episode in the group's namespace silo.

        ``source`` sets the episode provenance label menhir maps to an evidence kind
        (domain/truth/kinds.py). A user's own utterance is external testimony, so pass
        ``source="user"`` for user turns — that becomes a Guard-5 external anchor, so the
        EvidenceAnchorWarden admits facts the user stated. Omit (default "remote-api" ->
        agent_inference) for assistant/system turns, which are not anchors.

        ``turn_evidence_uuid`` grounds a ``source="user"`` claim against a
        previously recorded :TurnEvidence node. Without it Menhir's admission gate
        downgrades the claim to ``agent_inference``.

        By default uses ``wait=true`` so the episode is fully enriched before
        the call returns (back-compat for Mode-B per-item driver). Pass
        ``wait=False`` for throughput-optimised bulk ingest where a separate
        drain step provides the completeness guarantee.

        ``occurred_at`` (ISO-8601) backdates the episode's Graphiti
        ``reference_time`` so temporal-KG edges reflect historical event dates
        rather than ingestion time.  ``session_id`` pins all turns of one LME
        haystack session to the same menhir session silo.

        Returns Menhir's ingest response, including ``episode_id`` when content
        was submitted, or ``None`` when ``content`` is empty.
        """
        if not content:
            return
        url = self._base_url.rstrip("/") + self._ingest_path
        # A GROUNDED claim must be byte-identical to the :TurnEvidence it cites. Menhir's admission
        # gate tests exact equality (domain/truth/assertion_spans.claim_is_grounded) -- deliberately,
        # since the older substring/token-overlap check granted apex tier to single-word
        # contradictions of a multi-token claim. record_turn_evidence() posts the RAW text, so the
        # "{role}: " decoration applied here made the claim differ from its own evidence by exactly
        # six characters and every user-tier claim was denied and silently downgraded to
        # agent_inference: 22 episodes matched 'user: ' + turn.text, 0 matched turn.text, and the
        # graph carried zero ADMITTED_ON edges across every run to date.
        #
        # Scoped to the grounded path on purpose. The prefix still labels the speaker for turns that
        # are NOT cited as evidence, so ungrounded episode text -- and therefore extraction input on
        # every existing benchmark arm -- is unchanged and stays comparable.
        episode_text = content if turn_evidence_uuid is not None else f"{role}: {content}"
        payload: dict = {"episode": episode_text, "namespace": group_id}
        if source is not None:
            payload["source"] = source
        if diff is not None:
            payload["diff"] = diff
        if occurred_at is not None:
            payload["occurred_at"] = occurred_at
        if session_id is not None:
            payload["session_id"] = session_id
        if flagged:
            payload["flagged"] = True
        if bootstrap_scope is not None:
            payload["bootstrap_scope"] = bootstrap_scope
        if turn_evidence_uuid is not None:
            payload["turn_evidence_uuid"] = turn_evidence_uuid
        response = self._client.post(
            url,
            params={"wait": "true" if wait else "false"},
            json=payload,
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def ingest_raw(
        self,
        group_id: str,
        content: str,
        *,
        source: str,
        wait: bool = True,
    ) -> None:
        """Ingest an exact episode body for deterministic compatibility probes."""

        url = self._base_url.rstrip("/") + self._ingest_path
        response = self._client.post(
            url,
            params={"wait": "true" if wait else "false"},
            json={"episode": content, "namespace": group_id, "source": source},
            headers=self._headers,
        )
        response.raise_for_status()

    def recall(
        self, group_id: str, query: str, limit: int = 10
    ) -> list[str]:
        """
        POST a query to the recall endpoint and build LLM-facing snippets.

        Tolerant of response shape: accepts top-level list or dict with
        "results"/"memories"/"snippets" key. Each element may be a string or
        dict with "text"/"content"/"summary" key. When Menhir returns a
        structured ``authority_layer``, current scalar authority is emitted first
        with its source time and supporting quote. A structured
        ``event_authority_layer`` is consumed alongside it: a non-blocking lead
        renders an AUTHORITATIVE EVENT HISTORY with its predicate, selected
        object display/key, valid_at, time_basis, domain, stated_span quote,
        evidence identities, gate, and a preference over conflicts. A blocking
        selection-failure gate (anchor, ambiguity, time, scope, no_candidate)
        returns an advisory and suppresses the ordinary result items. Other
        results are labeled as related/non-authoritative or superseded context so
        a flat text prompt does not erase Menhir's conflict-resolution semantics.
        """
        url = self._base_url.rstrip("/") + self._recall_path
        # include_session=True: benchmark memories are freshly ingested and live at
        # SESSION scope (promotion to PERSISTENT only happens via the scheduler, which
        # is off under MENHIR_BENCHMARK_MODE). Without this the HTTP recall filters them
        # all out. Matches the menhir MCP recall tools, which always pass include_session=True.
        payload = {
            "query": query,
            "limit": limit,
            "namespace": group_id,
            "include_session": True,
            # Historical source-time evidence is metadata on the already-selected memories.
            # It is required for changed/previous/later questions and does not add stale candidates.
            "include_invalidated": True,
        }
        response = self._client.post(url, json=payload, headers=self._headers)
        response.raise_for_status()
        data = response.json()

        # Extract list of items (top-level list or nested in a known key)
        items: list[Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("results", "memories", "snippets"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break

        authority_records: list[dict[str, Any]] = []
        event_authority_records: list[dict[str, Any]] = []
        if isinstance(data, dict):
            raw_authority = data.get("authority_layer")
            if isinstance(raw_authority, dict):
                authority_records = [raw_authority]
            elif isinstance(raw_authority, list):
                authority_records = [record for record in raw_authority if isinstance(record, dict)]

            raw_event_authority = data.get("event_authority_layer")
            if isinstance(raw_event_authority, dict):
                event_authority_records = [raw_event_authority]
            elif isinstance(raw_event_authority, list):
                event_authority_records = [
                    record for record in raw_event_authority if isinstance(record, dict)
                ]

        has_authority = bool(authority_records) or bool(event_authority_records)
        event_blocked = any(
            str(record.get("status") or "").strip().lower() == "advisory"
            and str(record.get("gate") or "").strip().lower() in _BLOCKING_EVENT_GATES
            for record in event_authority_records
        )

        authority_view_ids = {
            str(record["view_uuid"])
            for record in authority_records
            if record.get("view_uuid")
        }

        if event_blocked:
            # A blocking selection-failure leaves no authoritative event to rank the ordinary
            # items against, so emit only the rendered event advisories. Current-state scalar
            # authority is suppressed too: it can mislead a historical-before query whose event
            # anchor is unresolved.
            return [
                rendered
                for record in event_authority_records
                if str(record.get("status") or "").strip().lower() == "advisory"
                and str(record.get("gate") or "").strip().lower() in _BLOCKING_EVENT_GATES
                if (rendered := _format_event_authority_record(record))
            ][:limit]

        snippets: list[str] = []
        for record in authority_records:
            rendered = _format_authority_record(record)
            if rendered:
                snippets.append(rendered)

        for record in event_authority_records:
            rendered = _format_event_authority_record(record)
            if rendered:
                snippets.append(rendered)

        for item in items:
            if isinstance(item, str):
                if item.strip():
                    if has_authority:
                        snippets.append(f"[RELATED MEMORY | non-authoritative] {item.strip()}")
                    else:
                        snippets.append(item)
            elif isinstance(item, dict):
                if has_authority and str(item.get("uuid") or "") in authority_view_ids:
                    continue
                text = _recall_item_text(item)
                if not text:
                    continue
                if has_authority:
                    memory_type = str(item.get("memory_type") or "memory").strip().lower()
                    if item.get("is_superseded_view") is True:
                        prefix = f"[SUPERSEDED {memory_type} MEMORY | historical only]"
                    elif item.get("is_scalar_authority") is True:
                        prefix = f"[CURRENT {memory_type} MEMORY | authoritative]"
                    else:
                        prefix = f"[RELATED {memory_type} MEMORY | non-authoritative]"
                    snippets.append(f"{prefix} {text}")
                else:
                    snippets.append(text)

        return snippets[:limit]

    def recall_raw(self, group_id: str, query: str, limit: int = 10) -> dict[str, Any]:
        """Return the unmodified HTTP recall response for benchmark assertions."""
        url = self._base_url.rstrip("/") + self._recall_path
        response = self._client.post(
            url,
            json={
                "query": query,
                "limit": limit,
                "namespace": group_id,
                "include_session": True,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"results": data}

    def bootstrap_flagged(
        self, reader_id: str, workspace: str, limit: int = 50
    ) -> dict[str, Any]:
        """Read scoped bootstrap pins and establish the matching receipt."""
        url = self._base_url.rstrip("/") + "/api/bootstrap/flagged"
        response = self._client.get(
            url,
            params={"reader_id": reader_id, "workspace": workspace, "limit": limit},
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def bootstrap_context(
        self,
        reader_id: str,
        workspace: str,
        namespace: str,
        recent_limit: int = 50,
        query: str = "",
    ) -> dict[str, Any]:
        """Read recent context using the receipt established for this workspace."""
        url = self._base_url.rstrip("/") + "/api/bootstrap/context"
        response = self._client.post(
            url,
            json={
                "reader_id": reader_id,
                "workspace": workspace,
                "namespace": namespace,
                "recent_limit": recent_limit,
                "query": query,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def mark_file_changed(self, *, path: str, project: str) -> dict[str, Any]:
        """Mark a structural file dirty through Menhir's supported Hook Center API."""

        url = self._base_url.rstrip("/") + "/api/tool-events"
        response = self._client.post(
            url,
            json={
                "event_type": "file_changed",
                "operation": "edit",
                "path": path,
                "project": project,
                "source_client": "archolith-bench",
            },
            headers=self._headers,
        )
        response.raise_for_status()
        return response.json()

    def reset(self, group_id: str) -> None:
        """Best-effort silo teardown via DELETE /api/namespace/{namespace}.

        Swallows any exception (menhir refuses the default namespace with 400).
        """
        try:
            url = self._base_url.rstrip("/") + "/api/namespace/" + group_id
            response = self._client.delete(url, headers=self._headers)
            response.raise_for_status()
        except Exception:
            pass

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


class Phase3MenhirClient(HttpMenhirClient):
    """HTTP client for the Phase 3 (personal-memory View consolidation) black-box surface.

    Exercises the SELECTIVE-capture pipeline end to end against a THROWAWAY menhir:
    post `:TurnEvidence` (not `/api/memory` episodes), trigger consolidation, then inspect
    Views / abstention receipts / supersession — without importing menhir or issuing Cypher.
    Uses the endpoints added in menhir's routes.py:

        POST /api/turn-evidence     capture one candidate user turn
        POST /api/phase3/run        run one consolidation pass over a namespace
        GET  /api/phase3/status     dirty flag + turn-evidence count
        GET  /api/views             current counter Views (+ history) and abstention receipts
        POST /api/phase3/reset      tear down the namespace (partition + TurnEvidence)
    """

    def post_turn_evidence(
        self,
        namespace: str,
        text: str,
        *,
        role: str = "user",
        declarant: str = "user",
        session_id: str | None = None,
        source_kind: str = "archolith-bench-phase3",
        source_client: str | None = "archolith_bench",
        hook_version: str | None = "menhir-phase3-bench-v1",
        triage_reason: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        turn_key: str | None = None,
    ) -> dict[str, Any]:
        """Capture one candidate user turn as `:TurnEvidence`.

        `declarant="user"` is required for Phase 3 to consider the turn — the dirty query and
        the user-evidence loader both filter `role='user' AND declarant='user'`. Returns the
        raw response ``{turn_id, created, recorded_at}``.

        `source_client`/`hook_version` are additive provenance labels the menhir route accepts
        (older menhir builds simply ignore unknown body fields). They identify the benchmark as the
        producer of these throwaway evidence rows.
        """
        url = self._base_url.rstrip("/") + "/api/turn-evidence"
        payload: dict[str, Any] = {
            "text": text,
            "role": role,
            "declarant": declarant,
            "namespace": namespace,
            "source_kind": source_kind,
        }
        if source_client is not None:
            payload["source_client"] = source_client
        if hook_version is not None:
            payload["hook_version"] = hook_version
        if session_id is not None:
            payload["session_id"] = session_id
        if triage_reason is not None:
            payload["triage_reason"] = list(triage_reason)
        if metadata is not None:
            payload["metadata"] = metadata
        if turn_key is not None:
            payload["turn_key"] = turn_key
        response = self._client.post(url, json=payload, headers=self._headers)
        response.raise_for_status()
        return response.json()

    def run_phase3(
        self, namespace: str, *, k: int = 3, source: str = "perception"
    ) -> dict[str, Any]:
        """Run one consolidation pass over ``namespace`` and return its metrics dict."""
        url = self._base_url.rstrip("/") + "/api/phase3/run"
        response = self._client.post(
            url, json={"namespace": namespace, "k": k, "source": source}, headers=self._headers
        )
        response.raise_for_status()
        return response.json()

    def phase3_status(self, namespace: str) -> dict[str, Any]:
        """Return ``{namespace, dirty, turn_evidence}`` for the namespace."""
        url = self._base_url.rstrip("/") + "/api/phase3/status"
        response = self._client.get(
            url, params={"namespace": namespace}, headers=self._headers
        )
        response.raise_for_status()
        return response.json()

    def fetch_views(self, namespace: str, *, limit: int = 100) -> dict[str, Any]:
        """Return ``{namespace, count, views, receipts}`` — current counter Views (each with
        ``history`` and ``superseded``) plus ``subject='perception'`` abstention receipts."""
        url = self._base_url.rstrip("/") + "/api/views"
        response = self._client.get(
            url, params={"namespace": namespace, "limit": limit}, headers=self._headers
        )
        response.raise_for_status()
        return response.json()

    def reset_phase3(self, namespace: str) -> dict[str, Any]:
        """Full teardown: graphiti partition (Views + watermark) and namespace-keyed TurnEvidence.

        Returns ``{namespace, nodes_deleted, turn_evidence_deleted}``. Best-effort: a first-run
        teardown of a fresh namespace may 400/404, which is fine.
        """
        url = self._base_url.rstrip("/") + "/api/phase3/reset"
        try:
            response = self._client.post(
                url, params={"namespace": namespace}, headers=self._headers
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"namespace": namespace, "nodes_deleted": 0, "turn_evidence_deleted": 0}


class StubPhase3Client:
    """Deterministic in-memory Phase 3 backend — a NETWORK-FREE stand-in for CI smoke.

    Models the HAPPY menhir consumer (F1 semantic-family voting + F2 unique-target corrections)
    so the whole driver + scenario suite + report path runs without a live menhir or Neo4j. It
    is NOT a substitute for the live benchmark (it cannot reveal server-side extraction defects
    or the fold-SUM stochasticity) — it exists so `--offline` smoke can guard the harness itself.

    Per-namespace it tracks posted prompts + a dirty/watermark bit, then batch-re-folds every
    prompt into counter Views on each `fetch_views` (idempotent). Corrections bind to the UNIQUE
    value-matching View, abstaining when >1 matches. Recognizes the exact fixture prompt strings.
    """

    def __init__(self) -> None:
        """Initialize empty per-namespace post + dirty state."""
        self._posts: dict[str, list[str]] = {}
        self._dirty: dict[str, bool] = {}

    def new_group(self) -> str:
        """Return a fresh isolated namespace id."""
        return uuid.uuid4().hex

    def reset_phase3(self, namespace: str) -> dict[str, Any]:
        """Drop all state for the namespace."""
        n = len(self._posts.get(namespace, []))
        self._posts.pop(namespace, None)
        self._dirty.pop(namespace, None)
        return {"namespace": namespace, "nodes_deleted": n, "turn_evidence_deleted": n}

    def post_turn_evidence(self, namespace: str, text: str, *, role: str = "user",
                           declarant: str = "user", triage_reason=None, turn_key=None,
                           **_: Any) -> dict[str, Any]:
        """Record a candidate turn and mark the namespace dirty (new evidence to consolidate)."""
        self._posts.setdefault(namespace, []).append(text)
        self._dirty[namespace] = True
        return {"turn_id": uuid.uuid4().hex, "created": True, "recorded_at": "stub"}

    def phase3_status(self, namespace: str) -> dict[str, Any]:
        """Report the dirty bit + evidence count."""
        return {
            "namespace": namespace,
            "dirty": bool(self._dirty.get(namespace)),
            "turn_evidence": len(self._posts.get(namespace, [])),
        }

    def run_phase3(self, namespace: str, *, k: int = 3, source: str = "perception") -> dict[str, Any]:
        """Consolidate: clear the dirty bit (watermark debounce) and report derived counts."""
        selected = bool(self._dirty.get(namespace))
        self._dirty[namespace] = False
        views = self._build_views(namespace)
        return {
            "namespace": namespace,
            "phase3_selected": selected,
            "dirty_after": False,
            "namespaces_dirty": 1 if selected else 0,
            "namespaces_processed": 1,
            "views_written": len(views),
            "abstained": 0,
            "corrections_applied": self._corrections_applied(namespace),
            "llm_calls": 0,
        }

    def fetch_views(self, namespace: str, *, limit: int = 100) -> dict[str, Any]:
        """Return current counter Views (batch re-fold; idempotent) + empty receipts."""
        views = self._build_views(namespace)
        return {"namespace": namespace, "count": len(views), "views": views, "receipts": []}

    def recall(self, namespace: str, query: str, limit: int = 10) -> list[str]:
        """Return one snippet per current View (so recall_after > recall_before once Views exist)."""
        return [f"{v['subject']} {v['counter']} {v['value']}" for v in self._build_views(namespace)]

    def close(self) -> None:
        """No-op (in-memory)."""

    # ---- deterministic consolidation model ---------------------------------------------------

    def _measures(self, prompt: str) -> list[tuple[str, str, float]]:
        """(subject, counter, value) a prompt yields — matches the core + scenario fixture strings."""
        p = prompt.lower()
        if "movies" in p and "watch list" in p:
            return [("movies", "movies_watchlist", 25.0)]
        if "books" in p and "this year" in p:
            return [("books", "books_read", 25.0)]
        if "50 dollars and 75 dollars" in p:
            return [("bike", "bike_spend", 125.0)]
        if "one bike for $50" in p:
            return [("bike", "bike_spend", 125.0)]
        if "2 bikes for $125" in p:  # count and SUM do not merge (reducer is identity)
            return [("bike", "bikes_count", 2.0), ("bike", "bike_spend", 125.0)]
        return []

    def _correction(self, prompt: str) -> tuple[float, float] | None:
        p = prompt.lower()
        # mirrors menhir's correction_resolver connectives (the happy consumer): the "X, not Y" and
        # "not X anymore, it is Y" forms, plus the consumer-quality-pack additions (arrow "X -> Y" and
        # reverse "to Y from X"). Value-match against an existing View is still the real target guard.
        if "20, not 25" in p or "not 25 anymore, it is 20" in p:
            return (25.0, 20.0)
        if "25 -> 20" in p or "25 --> 20" in p or "to 20 from 25" in p:
            return (25.0, 20.0)
        return None

    def _fold(self, namespace: str) -> tuple[dict[str, dict], int]:
        views: dict[str, dict] = {}
        corrections: list[tuple[float, float]] = []
        for prompt in self._posts.get(namespace, []):
            for subject, counter, value in self._measures(prompt):
                views[counter] = {"subject": subject, "counter": counter, "value": value,
                                  "current": True, "history": [], "superseded": []}
            corr = self._correction(prompt)
            if corr:
                corrections.append(corr)
        applied = 0
        for old, new in corrections:
            matches = [v for v in views.values() if v["value"] == old]
            if len(matches) == 1:  # unique target -> supersede; else abstain
                v = matches[0]
                v["superseded"] = [{"value": old, "current": False,
                                    "valid_at": "t1", "expired_at": "t2"}]
                v["value"] = new
                applied += 1
        return views, applied

    def _build_views(self, namespace: str) -> list[dict[str, Any]]:
        return list(self._fold(namespace)[0].values())

    def _corrections_applied(self, namespace: str) -> int:
        return self._fold(namespace)[1]


class StubScalarStateClient:
    """Deterministic in-memory ScalarStateView backend -- a NETWORK-FREE stand-in for CI smoke.

    Implements BOTH roles the `run_scalar_state` driver needs (ingest client + bolt reader) so the
    whole harness -- driver, invariant validators, reporter -- runs without a live menhir, a real LLM,
    or Neo4j. It models the HAPPY Piece C consumer: it recognizes the default fixture prompts, and on
    read folds them into current `scalar_state` Views + a durable `:TypedAssertion` log (tier `agent`,
    one current View per slot, correct namespace). The no-entity prompt becomes a pending `unbound:`
    advisory; the non-scalar control produces nothing.

    It is NOT a substitute for the live benchmark -- it cannot reveal real LLM perception defects,
    binding races, or scheduler timing -- it exists so `--offline` smoke can guard the harness itself.
    """

    # prompt-substring -> (value_kind, value, ss_unit, subject_display, ss_attribute)
    # Models the CORRECT consumer of the fixture contract: 7 eligible positive Views only. The two
    # NEGATIVE CONTROLS are deliberately NOT recognized -- "I paid $250" is a one-off event (no standing
    # money View) and "my car is red" is a possessed object that must not bind to self (no self View) --
    # so the happy-path stub leaves both un-materialized, exactly as a correct perceiver would.
    _RECOGNIZERS: list[tuple[str, str, Any, str | None, str, str]] = [
        ("37 rare coins", "count", 37.0, None, "coins", "coin_count"),
        ("180 centimeters", "measurement", 180.0, "cm", "height", "height_measurement"),
        ("45 minutes", "duration", 45.0, "min", "commute", "commute_duration"),
        ("gym 3 times a week", "frequency", 3.0, "per_week", "gym", "gym_frequency"),
        ("7:30", "clock_time", "07:30", None, "wake up", "wake_time"),
        ("day off is wednesday", "weekday", "Wednesday", None, "day off", "day_off_weekday"),
        ("finished reading dune", "boolean", True, None, "Dune", "dune_finished"),
    ]
    _ADVISORY_MARKER = "there are 12 of them"  # no uniquely-resolvable subject
    _NONSCALAR_MARKER = "went for a walk"  # a one-off happening, not a current property

    def __init__(self) -> None:
        """Initialize empty per-namespace ingest state."""
        self._episodes: dict[str, list[str]] = {}

    # ---- ingest-client role ------------------------------------------------------------------

    def new_group(self) -> str:
        """Return a fresh isolated namespace id."""
        return uuid.uuid4().hex

    def record_turn_evidence(self, namespace: str, text: str, **kwargs: Any) -> dict[str, Any]:
        """Record a grounding turn (returns a fake turn_id the ingest can cite)."""
        return {
            "turn_id": uuid.uuid4().hex,
            "created": True,
            "recorded_at": "stub",
            "occurred_at": kwargs.get("occurred_at"),
        }

    def ingest(self, group_id: str, role: str, content: str, **_: Any) -> None:
        """Store the ingested episode body for later folding."""
        if content:
            self._episodes.setdefault(group_id, []).append(content)

    def reset(self, group_id: str) -> None:
        """Drop all state for the namespace."""
        self._episodes.pop(group_id, None)

    def close(self) -> None:
        """No-op (in-memory)."""

    # ---- bolt-reader role --------------------------------------------------------------------

    def _recognize(self, namespace: str) -> list[dict[str, Any]]:
        """Fold the ingested prompts into (kind, value, subject) recognitions for this namespace."""
        out: list[dict[str, Any]] = []
        for prompt in self._episodes.get(namespace, []):
            p = prompt.lower()
            for marker, kind, value, unit, subject, attribute in self._RECOGNIZERS:
                if marker in p:
                    out.append({
                        "subject_uuid": f"stub-{attribute}",
                        "subject_display": subject,
                        "attribute": attribute,
                        "value_kind": kind,
                        "value": value,
                        "ss_unit": unit,
                    })
                    break
        return out

    def read_typed_assertions(self, namespace: str) -> list[dict[str, Any]]:
        """Current `:TypedAssertion` rows (perception tier is always `agent`) + the advisory row."""
        rows: list[dict[str, Any]] = []
        for r in self._recognize(namespace):
            rows.append({
                "subject_uuid": r["subject_uuid"],
                "subject_display": r["subject_display"],
                "attribute": r["attribute"],
                "value_kind": r["value_kind"],
                "value": r["value"],
                "evidence_tier": "agent",
                "binding_pending": False,
                "source_key": f"stub:{r['attribute']}",
            })
        return rows

    def read_scalar_state_views(self, namespace: str) -> list[dict[str, Any]]:
        """Current materialized `scalar_state` Views -- one per recognized slot in this namespace."""
        if namespace == "default":
            return []  # the stub never leaks to the default silo
        views: list[dict[str, Any]] = []
        for r in self._recognize(namespace):
            views.append({
                "subject_uuid": r["subject_uuid"],
                "ss_attribute": r["attribute"],
                "ss_kind": r["value_kind"],
                "ss_unit": r["ss_unit"],
                "value": r["value"],
                "group_id": namespace,
                "view_key": f"{namespace}:{r['subject_uuid']}:{r['attribute']}",
            })
        return views

    def read_pending_advisories(self, namespace: str) -> list[dict[str, Any]]:
        """The no-entity prompt surfaces as a pending `unbound:` advisory."""
        advisories: list[dict[str, Any]] = []
        for prompt in self._episodes.get(namespace, []):
            if self._ADVISORY_MARKER in prompt.lower():
                advisories.append({
                    "subject_display": "unbound:stub-advisory",
                    "source_key": "stub:advisory",
                    "attribute": "unknown_count",
                    "value_kind": "count",
                    "value": 12.0,
                })
        return advisories
