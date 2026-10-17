"""Scalar View drill-down for the Menhir memory harness (``--drill-down``).

Menhir can collapse a counted View to its total and print
``built from N items; drill down: get_scalar_view_items(view_uuid=...)``. This module gives the
answer model that tool, tells it when to use it, and runs its calls against Menhir, bound to the
item's own namespace and to the view ids that item's recall actually showed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

TOOL_NAME = "get_scalar_view_items"

DRILL_DOWN_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "List the items a structured memory verdict was built from (for example each "
            "counted occurrence, with its quote and source time), one page at a time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "view_uuid": {
                    "type": "string",
                    "description": "The view_uuid from the verdict's drill-down line.",
                },
                "offset": {"type": "integer", "minimum": 0, "description": "Page start (default 0)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50,
                          "description": "Page size (default 20)."},
            },
            "required": ["view_uuid"],
            "additionalProperties": False,
        },
    },
}

DRILL_DOWN_INSTRUCTION = (
    "Drill-down tool: a structured verdict may end with 'built from N items; drill down: "
    "get_scalar_view_items(view_uuid=...)'. That verdict's supporting items (for example each "
    "counted occurrence with its quote and time) are not shown. Use the tool as needed: when the "
    "answer depends on those items -- to list or name what was counted, to check that each item "
    "really matches the question, or to recount -- call get_scalar_view_items with that view_uuid, "
    "and page with offset while next_offset is set. If the memory shown already answers the "
    "question, answer without calling it."
)

_HINT_RE = re.compile(r"drill down: get_scalar_view_items\(view_uuid=([^)\s]+)\)")
_ITEM_FIELDS = ("relation", "operation", "value", "quote", "valid_at")


def drill_down_view_ids(memory_context: str) -> list[str]:
    """The view ids named by drill-down lines in the rendered memory, in order, deduplicated."""
    return list(dict.fromkeys(_HINT_RE.findall(memory_context or "")))


def with_drill_down_instruction(messages: list[dict]) -> list[dict]:
    """Copy of ``messages`` with the drill-down instruction added to the system message."""
    out = [dict(message) for message in messages]
    for message in out:
        if message.get("role") == "system":
            message["content"] = f"{message.get('content') or ''}\n\n{DRILL_DOWN_INSTRUCTION}".strip()
            return out
    return [{"role": "system", "content": DRILL_DOWN_INSTRUCTION}, *out]


def make_drill_down_runner(client: Any, namespace: str, view_ids: list[str]) -> Callable[[str, dict], str]:
    """A ``run_tool(name, arguments) -> str`` for send_with_tools.

    Only ``get_scalar_view_items`` on a view id shown in this item's recall is served, always in
    this item's namespace; anything else gets an error message the model can read. Menhir HTTP
    errors propagate so a broken backend is never graded as a wrong answer.
    """
    allowed = set(view_ids)

    def run_tool(name: str, arguments: dict) -> str:
        if name != TOOL_NAME:
            return json.dumps({"error": f"unknown tool {name!r}"})
        view_uuid = str(arguments.get("view_uuid") or "").strip()
        if view_uuid not in allowed:
            return json.dumps({"error": "unknown view_uuid; use one from a drill-down line in the memory"})
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 20)
        offset = offset if isinstance(offset, int) and offset >= 0 else 0
        limit = min(max(limit, 1), 50) if isinstance(limit, int) else 20
        page = client.get_scalar_view_items(namespace, view_uuid, offset=offset, limit=limit)
        items = [
            {key: item.get(key) for key in _ITEM_FIELDS if item.get(key) not in (None, "")}
            for item in page.get("items") or []
            if isinstance(item, dict)
        ]
        return json.dumps(
            {
                "view_uuid": view_uuid,
                "kind": page.get("kind"),
                "total": page.get("total"),
                "offset": page.get("offset", offset),
                "next_offset": page.get("next_offset"),
                "items": items,
            },
            ensure_ascii=False,
        )

    return run_tool
