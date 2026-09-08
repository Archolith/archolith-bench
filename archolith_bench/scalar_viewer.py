"""Read-only data model for the dashboard's one-task scalar-state explorer."""

from __future__ import annotations

import json
import socket
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_EVIDENCE_QUERY = """
MATCH (t:TurnEvidence {namespace: $namespace})
OPTIONAL MATCH (t)-[:FOUNDS]->(a:TypedAssertion {namespace: $namespace})
RETURN t.turn_id AS id,
       coalesce(t.role, t.declarant, "unknown") AS role,
       t.text AS text,
       t.session_id AS session_id,
       toString(t.occurred_at) AS occurred_at,
       toString(t.recorded_at) AS recorded_at,
       collect(DISTINCT a.assertion_id) AS founds
ORDER BY recorded_at, id
"""

_ASSERTIONS_QUERY = """
MATCH (a:TypedAssertion {namespace: $namespace})
RETURN a.assertion_id AS id,
       a.source_key AS source_key,
       a.episode_uuid AS evidence_id,
       a.subject_uuid AS subject_uuid,
       a.subject_display AS subject,
       a.attribute AS attribute,
       a.scope AS scope,
       a.value_kind AS value_kind,
       a.unit AS unit,
       a.operation AS operation,
       a.value AS value,
       a.stated_span AS stated_span,
       toString(a.valid_at) AS valid_at,
       toString(a.learned_at) AS learned_at,
       a.time_basis AS time_basis,
       a.evidence_tier AS evidence_tier,
       a.perceiver_version AS perceiver_version,
       coalesce(a.binding_pending, false) AS binding_pending,
       coalesce(a.superseded, false) AS superseded
ORDER BY valid_at, learned_at, id
"""

_VIEWS_QUERY = """
MATCH (v:Entity {group_id: $namespace, view_kind: "scalar_state"})
RETURN v.uuid AS id,
       v.view_key AS view_key,
       v.view_subject_uuid AS subject_uuid,
       v.view_subject AS subject,
       v.ss_attribute AS attribute,
       v.ss_scope AS scope,
       v.ss_kind AS value_kind,
       v.ss_unit AS unit,
       v.ss_value AS value,
       v.ss_display AS display,
       v.summary AS summary,
       toString(v.valid_at) AS valid_at,
       toString(v.created_at) AS created_at,
       coalesce(v.view_current, true) AS current,
       v.scalar_effective_tier AS effective_tier,
       coalesce(v.scalar_contributors, []) AS contributor_ids,
       v.supersedes AS supersedes,
       v.superseded_by AS superseded_by
ORDER BY valid_at, created_at, id
"""

_HISTORY_VIEWS_QUERY = """
MATCH (v:Entity {group_id: $namespace, view_kind: "scalar_history"})
OPTIONAL MATCH (v)-[:HISTORY_ENTRY]->(history_assertion:TypedAssertion)
WITH v, collect(DISTINCT history_assertion.assertion_id) AS contributor_ids
RETURN v.uuid AS id,
       v.view_key AS view_key,
       v.view_subject_uuid AS subject_uuid,
       v.view_subject AS subject,
       v.ss_attribute AS attribute,
       v.ss_scope AS scope,
       v.ss_kind AS value_kind,
       v.ss_unit AS unit,
       v.sh_entry_count AS entry_count,
       v.sh_signature AS signature,
       v.sh_op_counts AS op_counts,
       v.sh_first_valid_at AS first_valid_at,
       v.sh_last_valid_at AS last_valid_at,
       v.view_payload AS payload,
       contributor_ids,
       toString(v.valid_at) AS valid_at,
       toString(v.created_at) AS created_at,
       coalesce(v.view_current, true) AS current
ORDER BY valid_at, created_at, id
"""

_FACTS_QUERY = """
MATCH (s:Entity {group_id: $namespace})-[r]->(o:Entity {group_id: $namespace})
WHERE r.fact IS NOT NULL
RETURN s.name AS subject,
       type(r) AS relation,
       o.name AS object,
       r.fact AS fact,
       coalesce(r.episodes, []) AS episode_ids
ORDER BY coalesce(toString(r.valid_at), toString(r.created_at)), fact
LIMIT 40
"""

_AVAILABLE_NAMESPACES_QUERY = """
MATCH (t:TurnEvidence)
WHERE t.namespace IS NOT NULL
RETURN DISTINCT t.namespace AS namespace
ORDER BY namespace
"""


def task_catalog(ingests: list[Any]) -> list[dict[str, Any]]:
    """Return the latest manifest row for each namespace, newest manifest first."""
    by_namespace: dict[str, dict[str, Any]] = {}
    for ingest in ingests:
        for item in reversed(ingest.items):
            namespace = str(item.get("namespace") or "").strip()
            if not namespace or namespace in by_namespace:
                continue
            by_namespace[namespace] = {
                "namespace": namespace,
                "question_id": str(item.get("question_id") or ""),
                "question": str(item.get("question") or ""),
                "answer": str(item.get("answer") or ""),
                "question_type": str(item.get("question_type") or ""),
                "turns": int(item.get("turns") or 0),
                "ready": int(item.get("ready") or 0),
                "scalar_llm_calls": int(item.get("scalar_llm_calls") or 0),
                "typed_assertions": int(item.get("typed_assertions") or 0),
                "scalar_views": int(item.get("scalar_views") or 0),
            }
    return list(by_namespace.values())


def catalog_with_graph_availability(
    tasks: list[dict[str, Any]],
    available_namespaces: set[str] | None,
) -> list[dict[str, Any]]:
    """Annotate every manifest task without hiding graph-pending namespaces."""
    return [
        {
            **task,
            "graph_available": (
                None
                if available_namespaces is None
                else task["namespace"] in available_namespaces
            ),
        }
        for task in tasks
    ]


def build_memory_inventory(
    assertions: list[dict[str, Any]],
    views: list[dict[str, Any]],
    history_views: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify task recall material and explain each View's assertion operations."""
    operations_by_assertion = {
        str(assertion.get("id") or ""): str(assertion.get("operation") or "")
        for assertion in assertions
    }
    inventory: list[dict[str, Any]] = []
    for view_kind, rows in (
        ("scalar_state", views),
        ("scalar_history", history_views),
    ):
        for view in rows:
            operations = (
                {str(value) for value in (view.get("op_counts") or {})}
                if view_kind == "scalar_history" and view.get("op_counts")
                else {
                    operations_by_assertion.get(str(assertion_id), "")
                    for assertion_id in view.get("contributor_ids") or []
                }
            )
            operations.discard("")
            if operations == {"absolute"}:
                derivation = "absolute"
            elif operations == {"delta"}:
                derivation = "delta"
            elif operations:
                derivation = "mixed"
            else:
                derivation = "unknown"
            inventory.append({
                "id": str(view.get("id") or ""),
                "memory_type": "view",
                "view_kind": view_kind,
                "derivation": derivation,
                "operations": sorted(operations),
                "current": bool(view.get("current")),
                "subject": str(view.get("subject") or ""),
                "attribute": str(view.get("attribute") or ""),
                "scope": str(view.get("scope") or ""),
                "value": str(view.get("display") or view.get("value") or ""),
                "content": str(view.get("summary") or ""),
                "valid_at": view.get("valid_at"),
            })
    for index, fact in enumerate(facts):
        inventory.append({
            "id": f"content:{index}",
            "memory_type": "content",
            "view_kind": None,
            "derivation": None,
            "operations": [],
            "current": None,
            "subject": str(fact.get("subject") or ""),
            "relation": str(fact.get("relation") or ""),
            "object": str(fact.get("object") or ""),
            "content": str(fact.get("fact") or ""),
            "episode_ids": list(fact.get("episode_ids") or []),
        })
    return inventory


def _normalized_slot(values: list[Any] | tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(str(value or "").strip().lower() for value in values)


def _assertion_slot(assertion: dict[str, Any]) -> tuple[str, ...]:
    return _normalized_slot([
        assertion.get("subject_uuid"),
        assertion.get("attribute"),
        assertion.get("scope"),
        assertion.get("value_kind"),
        assertion.get("unit"),
    ])


def _audit_slot(event: dict[str, Any]) -> tuple[str, ...] | None:
    details = event.get("details") or {}
    raw_slot = details.get("slot")
    if not isinstance(raw_slot, (list, tuple)):
        return None
    if len(raw_slot) == 5:
        return _normalized_slot(raw_slot)
    if len(raw_slot) == 4 and details.get("subject_uuid"):
        return _normalized_slot([details["subject_uuid"], *raw_slot])
    return None


def annotate_assertion_fold_outcomes(
    assertions: list[dict[str, Any]],
    views: list[dict[str, Any]],
    history_views: list[dict[str, Any]],
    audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Explain each assertion's state and history projection outcomes."""
    current_state_ids: set[str] = set()
    historical_state_ids: set[str] = set()
    for view in views:
        target = current_state_ids if view.get("current") else historical_state_ids
        target.update(str(value) for value in view.get("contributor_ids") or [] if value)

    current_history_ids: set[str] = set()
    historical_history_ids: set[str] = set()
    for view in history_views:
        target = current_history_ids if view.get("current") else historical_history_ids
        target.update(str(value) for value in view.get("contributor_ids") or [] if value)
        target.update(
            str(entry["assertion_id"])
            for entry in view.get("entries") or []
            if entry.get("assertion_id")
        )

    state_events: dict[tuple[str, ...], dict[str, str]] = {}
    history_events: dict[tuple[str, ...], dict[str, str]] = {}
    for event in audit:
        slot = _audit_slot(event)
        event_name = str(event.get("event") or "")
        event_state = str(event.get("state") or "")
        details = event.get("details") or {}
        if slot and event_name == "fold" and event_state in {"abstain", "expiry"}:
            state_events[slot] = {
                "status": "abstained" if event_state == "abstain" else "expired",
                "reason": str(details.get("reason") or event_state),
            }
        elif slot and event_name == "view_write" and event_state == "stale_skipped":
            state_events[slot] = {
                "status": "write_failed",
                "reason": "stale scalar_state write was skipped",
            }
        elif slot and event_name == "history_fold" and event_state == "abstain":
            history_events[slot] = {
                "status": "abstained",
                "reason": str(details.get("reason") or event_state),
            }
        elif event_name == "history_rebuild" and event_state == "incomplete":
            for failure in details.get("failed_slots") or []:
                failed_slot = failure.get("slot_key")
                if isinstance(failed_slot, (list, tuple)) and len(failed_slot) == 5:
                    history_events[_normalized_slot(failed_slot)] = {
                        "status": "write_failed",
                        "reason": str(failure.get("error") or "scalar_history write failed"),
                    }

    annotated: list[dict[str, Any]] = []
    for assertion in assertions:
        assertion_id = str(assertion.get("id") or "")
        slot = _assertion_slot(assertion)
        pending = bool(assertion.get("binding_pending"))

        if pending:
            state = {"status": "not_folded", "reason": "subject binding is pending"}
            history = {"status": "not_folded", "reason": "subject binding is pending"}
        else:
            state = state_events.get(slot)
            if assertion_id in current_state_ids:
                state = {"status": "current", "reason": "contributes to the current scalar_state view"}
            elif state is None and assertion_id in historical_state_ids:
                state = {"status": "historical", "reason": "contributed to a superseded scalar_state view"}
            elif state is None and assertion.get("superseded"):
                state = {"status": "superseded", "reason": "assertion is superseded"}
            elif state is None:
                state = {
                    "status": "not_materialized",
                    "reason": (
                        "recorded in scalar_history but not used by the current scalar_state fold"
                        if assertion_id in current_history_ids
                        else "no scalar_state fold outcome was recorded"
                    ),
                }

            history = history_events.get(slot)
            if assertion_id in current_history_ids:
                history = {"status": "recorded", "reason": "recorded in the current scalar_history view"}
            elif history is None and assertion_id in historical_history_ids:
                history = {"status": "historical", "reason": "recorded in a superseded scalar_history view"}
            elif history is None:
                history = {
                    "status": "not_materialized",
                    "reason": "no scalar_history fold outcome was recorded",
                }

        annotated.append({
            **assertion,
            "fold_outcome": {
                "state": state,
                "history": history,
            },
        })
    return annotated


def scoring_rows(snaps: list[Any], question_id: str) -> list[dict[str, Any]]:
    """Find any recall/answer checkpoint rows already written for one task."""
    out: list[dict[str, Any]] = []
    for snap in snaps:
        for item in snap.items:
            if str(item.get("task_id") or "") != question_id:
                continue
            out.append({
                "arm": item.get("arm"),
                "correct": bool(item.get("correct")),
                "response": item.get("resp") or "",
                "recalled": item.get("recalled") or "",
                "gold": item.get("gold") or "",
            })
    return out


class ScalarTaskReader:
    """Combine Neo4j scalar provenance with the optional SQLite vote receipt."""

    def __init__(
        self,
        *,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        telemetry_db: Path | None = None,
        driver: Any = None,
    ) -> None:
        injected_driver = driver is not None
        if driver is None:
            # Guard at construction, not at the call site. This path was
            # reachable via --scalar-neo4j-uri with no check at all: the target
            # allow-list only covered the call sites someone had grepped for.
            # An injected driver is the caller's own (tests, or a driver already
            # built behind ScalarBoltReader, which guards separately).
            from archolith_bench.harness.scalar_bolt import assert_not_prod
            assert_not_prod(neo4j_uri)
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:  # pragma: no cover - depends on optional install
                raise RuntimeError(
                    "the scalar viewer needs the 'menhir-scalar' extra (neo4j driver)"
                ) from exc
            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password),
                connection_timeout=2.0,
                max_transaction_retry_time=0.0,
            )
        self._driver = driver
        self.telemetry_db = Path(telemetry_db) if telemetry_db else None
        parsed_uri = urlparse(neo4j_uri)
        self._probe_address = (
            None
            if injected_driver or not parsed_uri.hostname
            else (parsed_uri.hostname, parsed_uri.port or 7687)
        )

    def close(self) -> None:
        close = getattr(self._driver, "close", None)
        if close:
            close()

    def _query(self, query: str, namespace: str) -> list[dict[str, Any]]:
        self._require_available()
        result = self._driver.execute_query(
            query,
            namespace=namespace,
            database_="neo4j",
            routing_="r",
        )
        return [dict(record) for record in result.records]

    def _require_available(self) -> None:
        """Fail fast when an intentionally stopped inspection graph is offline."""
        if self._probe_address is None:
            return
        try:
            with socket.create_connection(self._probe_address, timeout=0.25):
                return
        except OSError as exc:
            host, port = self._probe_address
            raise RuntimeError(f"Neo4j inspection graph is offline at {host}:{port}") from exc

    def available_namespaces(self) -> set[str]:
        """Namespaces that have source evidence in the connected graph."""
        self._require_available()
        result = self._driver.execute_query(
            _AVAILABLE_NAMESPACES_QUERY,
            database_="neo4j",
            routing_="r",
        )
        return {
            str(record["namespace"])
            for record in result.records
            if record.get("namespace")
        }

    def _read_audit(
        self,
        namespace: str,
        assertions: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]], str | None]:
        if self.telemetry_db is None:
            return None, [], "No telemetry DB was configured; k-sample vote receipts are unavailable."
        if not self.telemetry_db.exists():
            return None, [], f"Telemetry DB not found: {self.telemetry_db}"

        uri = f"{self.telemetry_db.resolve().as_uri()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=2.0) as conn:
                rows = conn.execute(
                    """
                    SELECT id, recorded_at, event, status, episode_uuid, details_json
                    FROM lifecycle_events
                    WHERE phase = 'consolidation_audit'
                      AND json_extract(details_json, '$.namespace') = ?
                    ORDER BY id
                    """,
                    (namespace,),
                ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            return None, [], f"Could not read the telemetry vote receipt: {exc}"

        assertion_ids = {str(a.get("id") or "") for a in assertions}
        source_keys = {str(a.get("source_key") or "") for a in assertions}
        parsed: list[dict[str, Any]] = []
        relevant_passes: set[str] = set()
        pass_last_id: dict[str, int] = {}
        for row_id, at, event, state, pass_id, details_json in rows:
            try:
                details = json.loads(details_json or "{}")
            except (TypeError, ValueError):
                details = {"_raw": details_json}
            pid = str(pass_id or "")
            parsed.append({
                "id": int(row_id),
                "recorded_at": at,
                "event": event,
                "state": state,
                "pass_id": pid,
                "details": details,
            })
            pass_last_id[pid] = int(row_id)
            if (
                str(details.get("assertion_id") or "") in assertion_ids
                or str(details.get("source_key") or "") in source_keys
            ):
                relevant_passes.add(pid)

        if not relevant_passes:
            return None, [], (
                "Audit events exist for this namespace, but none match the assertions in this graph. "
                "The graph and telemetry may come from different benchmark attempts."
            )
        pass_id = max(relevant_passes, key=lambda pid: pass_last_id.get(pid, -1))
        events = [
            {
                "recorded_at": row["recorded_at"],
                "event": row["event"],
                "state": row["state"],
                "details": row["details"],
            }
            for row in parsed
            if row["pass_id"] == pass_id
        ]
        return pass_id, events, None

    def read(self, namespace: str) -> dict[str, Any]:
        """Read all stages for one namespace without mutating Neo4j or telemetry."""
        evidence = self._query(_EVIDENCE_QUERY, namespace)
        assertions = self._query(_ASSERTIONS_QUERY, namespace)
        views = self._query(_VIEWS_QUERY, namespace)
        history_views = self._query(_HISTORY_VIEWS_QUERY, namespace)
        # Parse the JSON payload + op_counts on history views for the dashboard.
        for hv in history_views:
            if isinstance(hv.get("payload"), str):
                try:
                    hv["entries"] = json.loads(hv["payload"])
                except (json.JSONDecodeError, TypeError):
                    hv["entries"] = []
            else:
                hv["entries"] = hv.get("payload") or []
            if isinstance(hv.get("op_counts"), str):
                try:
                    hv["op_counts"] = json.loads(hv["op_counts"])
                except (json.JSONDecodeError, TypeError):
                    hv["op_counts"] = {}
        facts = self._query(_FACTS_QUERY, namespace)
        audit_pass_id, audit, audit_warning = self._read_audit(namespace, assertions)
        assertions = annotate_assertion_fold_outcomes(assertions, views, history_views, audit)
        return {
            "namespace": namespace,
            "evidence": evidence,
            "assertions": assertions,
            "views": views,
            "history_views": history_views,
            "facts": facts,
            "memory_inventory": build_memory_inventory(assertions, views, history_views, facts),
            "audit_pass_id": audit_pass_id,
            "audit": audit,
            "audit_warning": audit_warning,
        }
