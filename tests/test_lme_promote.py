"""promote.py promotes exactly the manifest's namespaces through Menhir's promote_namespace_scope op."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "longmemeval" / "lib" / "promote.py"
    spec = importlib.util.spec_from_file_location("lme_promote_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["lme_promote_test"] = module
    spec.loader.exec_module(module)
    return module


promote = _load()


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_manifest_namespaces_dedupes_in_order(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [{"namespace": "lme-b"}, {"namespace": "lme-a"}, {"namespace": "lme-b"}])
    assert promote.manifest_namespaces(path) == ["lme-b", "lme-a"]


@pytest.mark.parametrize("rows", [[], [{"namespace": "lme-a"}, {"namespace": " "}], [{"turns": 3}]])
def test_manifest_without_namespaces_refuses(tmp_path: Path, rows: list[dict]) -> None:
    with pytest.raises(SystemExit):
        promote.manifest_namespaces(_manifest(tmp_path, rows))


def test_promote_calls_the_op_once_per_namespace_and_sums() -> None:
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        return httpx.Response(200, json={
            "namespace": body["namespace"], "entities_promoted": 2, "edges_promoted": 3,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert promote.promote(["lme-a", "lme-b"], "http://menhir/", client) == (4, 6)
    assert seen == [
        ("/api/internal/backend/promote_namespace_scope", {"namespace": "lme-a"}),
        ("/api/internal/backend/promote_namespace_scope", {"namespace": "lme-b"}),
    ]


def test_promote_fails_on_a_refused_namespace() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403, json={})))
    with pytest.raises(httpx.HTTPStatusError):
        promote.promote(["lme-a"], "http://menhir", client)
