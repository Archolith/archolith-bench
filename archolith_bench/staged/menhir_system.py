"""Thin async wrapper over Menhir's backend client for staged runs."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..harness.menhir_client import effective_counted_verdict_limit
from ..harness.memory_ab import assert_not_production

_TERMINAL = {"READY", "FAILED"}


class MenhirSystem:
    """Reset, ingest, wait, search and build context against one Menhir backend."""

    def __init__(self, backend: Any, *, user_id: str = "archolith-bench") -> None:
        self.backend = backend
        self.user_id = user_id

    async def reset(self, ns: str) -> dict:
        """Delete the namespace (throwaway graph; P4 revisits)."""
        return await self.backend.delete_namespace(ns, force=True)

    async def record_turn_evidence(
        self,
        ns: str,
        text: str,
        *,
        role: str = "user",
        declarant: str | None = "user",
        session_id: str | None = None,
        occurred_at: str | None = None,
        turn_key: str | None = None,
    ) -> dict:
        """Capture one turn as :TurnEvidence; same request as HttpMenhirClient.record_turn_evidence."""
        return await self.backend.record_turn_evidence(
            text,
            role=role,
            declarant=declarant,
            namespace=ns,
            session_id=session_id,
            occurred_at=occurred_at,
            source_kind="archolith-bench",
            source_client="archolith_bench",
            turn_key=turn_key,
        )

    async def ingest(
        self,
        ns: str,
        role: str,
        content: str,
        *,
        occurred_at: str | None = None,
        session_id: str | None = None,
        source: str | None = None,
        turn_evidence_uuid: str | None = None,
        idempotency_key: str | None = None,
        memory_role: str | None = None,
        store_only: bool = False,
        wait: bool = False,  # noqa: ARG002 - accepted for HttpMenhirClient parity; callers poll
    ) -> dict | None:
        """Queue one episode with the request HttpMenhirClient.ingest would send (same digest)."""
        if not content:
            return None
        # A grounded claim must equal its evidence text byte for byte; ungrounded text keeps the speaker prefix.
        text = content if turn_evidence_uuid is not None else f"{role}: {content}"
        kwargs: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": session_id or uuid.uuid4().hex,
            # BackendClient defaults source to "mcp"; the REST default is "remote-api".
            "source": source or "remote-api",
            "namespace": ns,
            "occurred_at": occurred_at,
            "turn_evidence_uuid": turn_evidence_uuid,
        }
        if memory_role is not None:
            kwargs["role"] = memory_role
        if idempotency_key is not None:
            kwargs["idempotency_key"] = idempotency_key
        if store_only:
            kwargs["store_only"] = True
        return await self.backend.queue_episode(text, **kwargs)

    async def add(
        self,
        ns: str,
        *,
        text: str,
        role: str,
        session_id: str,
        occurred_at: str | None = None,
        idempotency_key: str,
        turn_key: str,
        record_evidence: bool = True,
        source: str | None = None,
    ) -> dict:
        """Ingest one turn: evidence for user turns, then the shared ingest request shape."""
        is_user = role.strip().lower() == "user"
        turn_id = None
        if record_evidence and is_user and text:
            ev = await self.record_turn_evidence(
                ns,
                text,
                role="user",
                declarant="user",
                session_id=session_id,
                occurred_at=occurred_at,
                turn_key=turn_key,
            )
            turn_id = ev.get("turn_id") or None
        return await self.ingest(
            ns,
            role,
            text,
            occurred_at=occurred_at,
            session_id=session_id,
            source=source or ("user" if is_user else None),
            turn_evidence_uuid=turn_id,
            idempotency_key=idempotency_key,
            memory_role=role,
        )

    async def wait_indexed(
        self,
        episode_ids: list[str],
        *,
        timeout_s: float = 1800.0,
        poll_s: float = 2.0,
        clock=time.monotonic,  # noqa: ANN001
        sleep=asyncio.sleep,  # noqa: ANN001
    ) -> dict[str, str]:
        """Poll until every episode is READY or FAILED; TimeoutError lists the pending ids."""
        deadline = clock() + timeout_s
        final: dict[str, str] = {}
        pending = list(episode_ids)
        while True:
            still: list[str] = []
            for eid in pending:
                row = await self.backend.fetch_episode_processing(eid)
                state = str((row or {}).get("processing_state") or "").upper()
                if state in _TERMINAL:
                    final[eid] = state
                else:
                    still.append(eid)
            pending = still
            if not pending:
                return final
            if clock() >= deadline:
                raise TimeoutError(f"episodes still pending after {timeout_s}s: {', '.join(pending)}")
            await sleep(poll_s)

    async def search(
        self,
        ns: str,
        query: str,
        *,
        limit: int,
        counted_authority_only: bool = False,
        counted_verdict_limit: int | None = None,
        trace: bool = False,
    ) -> dict:
        # Mirrors the HttpMenhirClient /api/recall request (REST sets limit_is_explicit when a limit is sent).
        return await self.backend.recall(
            query,
            limit=limit,
            limit_is_explicit=True,
            include_session=True,
            include_invalidated=True,
            namespace=ns,
            counted_verdict_limit=(
                effective_counted_verdict_limit(counted_verdict_limit) if counted_authority_only else None
            ),
            trace=trace,
        )

    async def context(self, ns: str, query: str, *, max_tokens: int, preset: str) -> dict:
        return await self.backend.build_context(query, max_tokens=max_tokens, preset=preset, namespace=ns)


def load_backend_client(menhir_root: Path, base_url: str, *, timeout_s: float = 180.0):  # noqa: ANN201
    """Construct Menhir's BackendClient from a Menhir checkout; refuses non-throwaway targets."""
    assert_not_production(base_url)
    src = str(Path(menhir_root) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from menhir.config.settings import MemorySettings
        from menhir.core.backend_impl import BackendClient
    except ImportError as exc:
        raise RuntimeError(
            "Menhir's dependencies must be importable: run with Menhir's venv "
            f"or `pip install -e {menhir_root}`"
        ) from exc
    return BackendClient(base_url, timeout_s=timeout_s, settings=MemorySettings.from_env())
