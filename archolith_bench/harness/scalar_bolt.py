"""Read-only bolt reader for the Menhir ScalarStateView (Piece C) e2e harness.

Piece C is a SHADOW build: nothing user-facing reads a `scalar_state` View yet, so Menhir's HTTP
recall/bootstrap surface will NOT show them. The only way to verify the materialized state is to query
the graph directly over bolt. This helper wraps the three verification queries from the handoff
(`menhir/.agent/for-review/HANDOFF-2026-07-20-scalar-state-end-to-end-testing.md`, section 7):

    * `read_typed_assertions`   -- the durable `:TypedAssertion` event log the LLM produced
    * `read_scalar_state_views` -- the materialized current `scalar_state` Views (one per slot)
    * `read_pending_advisories` -- assertions awaiting a unique binding (`unbound:<source_key>`)

SAFETY: this opens a driver against a THROWAWAY menhir Neo4j only. The online tests / probes run
unscoped reads; pointing at prod would read the operator's real memories. `assert_not_prod` refuses
the known prod endpoints, mirroring the `force_all_tests_onto_test_neo4j` discipline in menhir's
conftest. The reader issues READ transactions only -- it never writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .targets import TargetRefused, assert_allowed_target

# The default Neo4j bolt port. Refused even on loopback: a local instance on
# 7687 is a real Menhir, not a throwaway, which publishes on 7688.
PROD_BOLT_PORT = 7687


class ProdBoltRefused(RuntimeError):
    """Raised when a bolt URI is not an allow-listed throwaway Neo4j."""


def assert_not_prod(uri: str) -> None:
    """Refuse a bolt URI that is not an allow-listed throwaway.

    Allow-list, not deny-list. This previously named three known-prod host:port
    pairs (including a LAN IP that is no longer production) and refused the
    suffix ":7687". Both halves failed open once production moved off-LAN: a
    remote host on any other port was permitted outright.

    Now the host must be loopback, or opted in via ARCHOLITH_BENCH_ALLOW_HOSTS,
    and port 7687 is refused regardless of host. The scalar e2e harness runs
    unscoped reads, so a wrong target leaks real memories.
    """
    try:
        assert_allowed_target(uri, what="bolt URI")
    except TargetRefused as e:
        raise ProdBoltRefused(str(e)) from e

    if (uri or "").lower().rstrip("/").endswith(f":{PROD_BOLT_PORT}"):
        raise ProdBoltRefused(
            f"bolt URI {uri!r} targets port {PROD_BOLT_PORT} (the default Menhir Neo4j "
            f"port); use the throwaway (7688). Refusing."
        )


# --- query text (verbatim intent from handoff section 7; parameterized by namespace) -------------

_Q_ASSERTIONS = """
MATCH (a:TypedAssertion)
WHERE a.namespace = $ns AND NOT coalesce(a.superseded, false)
RETURN a.subject_uuid    AS subject_uuid,
       a.subject_display AS subject_display,
       a.attribute       AS attribute,
       a.value_kind      AS value_kind,
       a.value           AS value,
       a.evidence_tier   AS evidence_tier,
       coalesce(a.binding_pending, false) AS binding_pending,
       a.source_key      AS source_key
ORDER BY a.subject_uuid, a.attribute
"""

_Q_VIEWS = """
MATCH (v:Entity {view_kind: 'scalar_state'})
WHERE coalesce(v.view_current, true) AND v.group_id = $ns
RETURN v.view_subject_uuid    AS subject_uuid,
       v.ss_attribute         AS ss_attribute,
       v.ss_kind              AS ss_kind,
       v.ss_unit              AS ss_unit,
       v.view_value           AS value,
       v.group_id             AS group_id,
       v.ss_view_key_current  AS view_key
ORDER BY v.view_subject_uuid, v.ss_attribute
"""

_Q_ADVISORIES = """
MATCH (a:TypedAssertion)
WHERE a.namespace = $ns AND coalesce(a.binding_pending, false)
RETURN a.subject_display AS subject_display,
       a.source_key      AS source_key,
       a.attribute       AS attribute,
       a.value_kind      AS value_kind,
       a.value           AS value
ORDER BY a.source_key
"""


@dataclass
class ScalarBoltReader:
    """Thin read-only neo4j client for scalar-state verification against a throwaway menhir.

    Lazily imports the neo4j driver (optional bench extra `menhir-scalar`) so importing the harness
    module offline (stub-driven CI) does not require neo4j to be installed.
    """

    uri: str
    user: str = "neo4j"
    password: str = "benchthrowaway"
    database: str = "neo4j"
    _driver: Any = None

    def __post_init__(self) -> None:
        """Refuse prod up front so a misconfigured target never opens a connection."""
        assert_not_prod(self.uri)

    def connect(self) -> "ScalarBoltReader":
        """Open the underlying neo4j driver (idempotent). Returns self for chaining."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase  # lazy: optional `menhir-scalar` extra
            except ImportError as exc:  # pragma: no cover - import-guard branch
                raise ImportError(
                    "the neo4j driver is required for the scalar-state bolt reader; "
                    "install it via `pip install -e .[menhir-scalar]`"
                ) from exc
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self

    def __enter__(self) -> "ScalarBoltReader":
        """Context-manager entry: ensure connected."""
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        """Context-manager exit: close the driver."""
        self.close()

    def _read(self, query: str, ns: str) -> list[dict[str, Any]]:
        """Run one READ transaction and materialize rows as plain dicts."""
        self.connect()
        with self._driver.session(database=self.database) as session:
            result = session.run(query, ns=ns)
            return [dict(record) for record in result]

    def read_typed_assertions(self, namespace: str) -> list[dict[str, Any]]:
        """Current (non-superseded) `:TypedAssertion` rows in the namespace."""
        return self._read(_Q_ASSERTIONS, namespace)

    def read_scalar_state_views(self, namespace: str) -> list[dict[str, Any]]:
        """Current materialized `scalar_state` Views (one per slot) in the namespace."""
        return self._read(_Q_VIEWS, namespace)

    def read_pending_advisories(self, namespace: str) -> list[dict[str, Any]]:
        """Assertions awaiting a unique binding (subject is an `unbound:<source_key>` sentinel)."""
        return self._read(_Q_ADVISORIES, namespace)

    def close(self) -> None:
        """Close the underlying driver if open."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
