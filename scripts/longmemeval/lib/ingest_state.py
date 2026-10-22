"""Append-only ingest state log for resumable LongMemEval ingest.

One JSON object per line, folded in order into per-namespace progress. It records only what
the graph cannot tell us: which turn plan a namespace was started under, which episodes and
projections were already retried, which namespaces were reset, and which finished. A resume
reads it back so it never resets a namespace unprompted, never retries twice, and refuses a
namespace whose turn plan changed.

Appends are fsynced. A torn final line (crash mid-write) is ignored on load and cut off
before the next append; any other bad line is an error, not something to skip.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_SCHEMA = 1

_RECORD_TYPES = ("plan", "retry", "requeued", "reset", "done")
# "retry" spends the one-retry budget and is written BEFORE the reset; "requeued" is written
# after a reset succeeded, so counts report retries that happened. "provider" requeues (quota or
# auth failures, not content failures) never spend the budget.
_RETRY_KINDS = ("episode", "projection")
_REQUEUE_KINDS = ("episode", "projection", "provider")


class IngestStateError(RuntimeError):
    """The ingest state log is unreadable or contradicts itself."""


@dataclass
class NamespaceProgress:
    plan_digest: str | None = None
    segments: int = 0
    segmentation: str | None = None
    retried_episodes: set[str] = field(default_factory=set)
    retried_projections: set[str] = field(default_factory=set)
    requeued: dict[str, int] = field(default_factory=dict)
    done: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply(
    progress: NamespaceProgress, record: dict[str, Any], where: str,
) -> NamespaceProgress:
    """Fold one validated record into *progress* and return the new progress."""
    kind = record["type"]
    if kind == "plan":
        digest = record.get("digest")
        segments = record.get("segments")
        if not isinstance(digest, str) or not isinstance(segments, int):
            raise IngestStateError(f"{where}: plan record needs digest and segments")
        if progress.plan_digest is not None and progress.plan_digest != digest:
            raise IngestStateError(
                f"{where}: namespace {record['namespace']!r} has two different plan "
                "digests with no reset between them"
            )
        progress.plan_digest = digest
        progress.segments = segments
        progress.segmentation = record.get("segmentation")
    elif kind == "retry":
        uuid = record.get("episode_uuid")
        retry_kind = record.get("kind")
        if not isinstance(uuid, str) or retry_kind not in _RETRY_KINDS:
            raise IngestStateError(f"{where}: retry record needs episode_uuid and kind")
        if retry_kind == "episode":
            progress.retried_episodes.add(uuid)
        else:
            progress.retried_projections.add(uuid)
    elif kind == "requeued":
        uuid = record.get("episode_uuid")
        requeue_kind = record.get("kind")
        if not isinstance(uuid, str) or requeue_kind not in _REQUEUE_KINDS:
            raise IngestStateError(f"{where}: requeued record needs episode_uuid and kind")
        progress.requeued[requeue_kind] = progress.requeued.get(requeue_kind, 0) + 1
    elif kind == "reset":
        progress = NamespaceProgress()
    else:
        progress.done = True
    return progress


class IngestStateLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._namespaces: dict[str, NamespaceProgress] = {}
        self._good_length = 0
        self._needs_newline = False
        self._torn = False
        self._load()

    @staticmethod
    def path_for(manifest_path: Path) -> Path:
        manifest_path = Path(manifest_path)
        return manifest_path.parent / f"ingest-state-{manifest_path.stem}.jsonl"

    def progress(self, namespace: str) -> NamespaceProgress:
        return copy.deepcopy(self._namespaces.get(namespace, NamespaceProgress()))

    def namespaces(self) -> list[str]:
        return list(self._namespaces)

    def record_plan(
        self, namespace: str, digest: str, segments: int, segmentation: str | None = None,
    ) -> None:
        fields: dict[str, Any] = {"digest": digest, "segments": int(segments)}
        if segmentation is not None:
            fields["segmentation"] = segmentation
        self._append("plan", namespace, **fields)

    def record_retry(self, namespace: str, episode_uuid: str, kind: str) -> None:
        if kind not in _RETRY_KINDS:
            raise ValueError(f"retry kind must be one of {_RETRY_KINDS}, got {kind!r}")
        self._append("retry", namespace, episode_uuid=episode_uuid, kind=kind)

    def record_requeued(self, namespace: str, episode_uuid: str, kind: str) -> None:
        if kind not in _REQUEUE_KINDS:
            raise ValueError(f"requeue kind must be one of {_REQUEUE_KINDS}, got {kind!r}")
        self._append("requeued", namespace, episode_uuid=episode_uuid, kind=kind)

    def record_reset(self, namespace: str) -> None:
        self._append("reset", namespace)

    def record_done(self, namespace: str, question_id: str) -> None:
        self._append("done", namespace, question_id=question_id)

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        offset = 0
        number = 0
        while offset < len(data):
            end = data.find(b"\n", offset)
            terminated = end != -1
            line_end = end if terminated else len(data)
            raw = data[offset:line_end]
            number += 1
            where = f"{self.path}:{number}"
            try:
                record = json.loads(raw.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("not an object")
            except (ValueError, UnicodeDecodeError) as exc:
                if not terminated:
                    self._torn = True  # torn final line: ignore, cut before next append
                    break
                raise IngestStateError(f"{where}: unparsable line ({exc})") from exc
            self._fold(record, where)
            offset = line_end + 1 if terminated else line_end
            self._good_length = offset
            if not terminated:
                self._needs_newline = True

    def _fold(self, record: dict[str, Any], where: str) -> None:
        if record.get("v") != STATE_SCHEMA:
            raise IngestStateError(f"{where}: unknown schema version {record.get('v')!r}")
        if record.get("type") not in _RECORD_TYPES:
            raise IngestStateError(f"{where}: unknown record type {record.get('type')!r}")
        namespace = record.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise IngestStateError(f"{where}: record has no namespace")
        current = self._namespaces.get(namespace, NamespaceProgress())
        self._namespaces[namespace] = _apply(current, record, where)

    def _append(self, record_type: str, namespace: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "v": STATE_SCHEMA, "type": record_type, "namespace": namespace, "at": _now(),
            **fields,
        }
        current = copy.deepcopy(self._namespaces.get(namespace, NamespaceProgress()))
        updated = _apply(current, record, f"{self.path} (append)")
        payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        if self._needs_newline:
            payload = b"\n" + payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._torn and self.path.exists():
            with open(self.path, "r+b") as handle:
                handle.truncate(self._good_length)
                handle.flush()
                os.fsync(handle.fileno())
        with open(self.path, "ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._torn = False
        self._needs_newline = False
        self._namespaces[namespace] = updated
        self._good_length += len(payload)
