"""Opt-in request rewrites for prompt-bloat A/B tests, applied by the budget proxy or offline replay.

Each transform takes a parsed chat-completions request (dict) and its response schema name, mutates
the request in place, and returns True only when it changed something. A transform that cannot
parse its target leaves the request untouched, so an arm never silently sends a half-rewritten prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

NODE_KEEP_KEYS = ("candidate_id", "name", "entity_types", "summary")

_EXISTING_ENTITIES = re.compile(r"(<EXISTING ENTITIES>\s*)(.*?)(\s*</EXISTING ENTITIES>)", re.S)
_NEGATIVE_EXAMPLES = re.compile(r"\n?<NEGATIVE EXAMPLES>.*?</NEGATIVE EXAMPLES>\n?", re.S)
_EDGE_BLOCKS = re.compile(
    r"<EXISTING FACTS>\s*(.*?)\s*</EXISTING FACTS>.*?"
    r"<FACT INVALIDATION CANDIDATES>\s*(.*?)\s*</FACT INVALIDATION CANDIDATES>.*?"
    r"<NEW FACT>\s*(.*?)\s*</NEW FACT>",
    re.S,
)

EDGE_COMPACT_INSTRUCTIONS = """Compare the NEW FACT with the numbered facts below. idx numbering is continuous:
EXISTING FACTS first, then FACT INVALIDATION CANDIDATES.
- duplicate_facts: idx values from EXISTING FACTS only, where the NEW FACT states identical information.
  Any difference in numbers, dates or key qualifiers means it is NOT a duplicate.
- contradicted_facts: idx values from either list that the NEW FACT contradicts or supersedes
  (same relationship with an updated value or state). An EXISTING FACT can be both.
- Different events (e.g. on different days) are neither. Use empty lists when nothing applies.
Examples: "Alice joined Acme Corp in 2020" vs the same -> duplicate. "Alice works at Acme as a software
engineer" -> "... as a senior engineer" -> contradicted, not duplicate. "Bob ran 5 miles on Tuesday" vs
"Bob ran 3 miles on Wednesday" -> neither."""


def _user_message(request: dict) -> dict | None:
    for m in request.get("messages") or []:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m
    return None


def node_slim(request: dict, schema: str | None) -> bool:
    """NodeResolutions: keep only candidate_id/name/entity_types/summary per existing entity."""
    if schema != "NodeResolutions":
        return False
    m = _user_message(request)
    match = _EXISTING_ENTITIES.search(m["content"]) if m else None
    if not match:
        return False
    try:
        entities = json.loads(match.group(2))
    except json.JSONDecodeError:
        return False
    if not isinstance(entities, list):
        return False
    slim = [{k: e[k] for k in NODE_KEEP_KEYS if k in e} for e in entities if isinstance(e, dict)]
    new_block = json.dumps(slim, ensure_ascii=False)
    m["content"] = m["content"][: match.start(2)] + new_block + m["content"][match.end(2):]
    return True


NODE_AUDIT_KEYS = ("merge_audit", "merged_from", "last_merge_op_id")


def node_drop_audit(request: dict, schema: str | None) -> bool:
    """NodeResolutions: drop only merge lineage (merge_audit/merged_from/last_merge_op_id), keep all else."""
    if schema != "NodeResolutions":
        return False
    m = _user_message(request)
    match = _EXISTING_ENTITIES.search(m["content"]) if m else None
    if not match:
        return False
    try:
        entities = json.loads(match.group(2))
    except json.JSONDecodeError:
        return False
    if not isinstance(entities, list) or not any(
        isinstance(e, dict) and any(k in e for k in NODE_AUDIT_KEYS) for e in entities
    ):
        return False
    kept = [{k: v for k, v in e.items() if k not in NODE_AUDIT_KEYS} if isinstance(e, dict) else e for e in entities]
    m["content"] = m["content"][: match.start(2)] + json.dumps(kept, ensure_ascii=False) + m["content"][match.end(2):]
    return True


def edge_compact(request: dict, schema: str | None) -> bool:
    """EdgeDuplicate: same data blocks, one compact instruction block instead of the repeated rules."""
    if schema != "EdgeDuplicate":
        return False
    m = _user_message(request)
    match = _EDGE_BLOCKS.search(m["content"]) if m else None
    if not match:
        return False
    existing, candidates, new = match.groups()
    m["content"] = (
        f"{EDGE_COMPACT_INSTRUCTIONS}\n\n<EXISTING FACTS>\n{existing}\n</EXISTING FACTS>\n\n"
        f"<FACT INVALIDATION CANDIDATES>\n{candidates}\n</FACT INVALIDATION CANDIDATES>\n\n"
        f"<NEW FACT>\n{new}\n</NEW FACT>\n"
    )
    return True


def extraction_no_negative_examples(request: dict, schema: str | None) -> bool:
    """CombinedExtraction: drop the <NEGATIVE EXAMPLES> block (rules above it are kept)."""
    if schema != "CombinedExtraction":
        return False
    m = _user_message(request)
    if not m or not _NEGATIVE_EXAMPLES.search(m["content"]):
        return False
    m["content"] = _NEGATIVE_EXAMPLES.sub("\n", m["content"], count=1)
    return True


TRANSFORMS: dict[str, Callable[[dict, str | None], bool]] = {
    "node_slim": node_slim,
    "node_drop_audit": node_drop_audit,
    "edge_compact": edge_compact,
    "extraction_no_negative_examples": extraction_no_negative_examples,
}


def parse_schema_effort(spec: str | None) -> dict[str, str]:
    """'EdgeDuplicate=none,text=low' -> {'EdgeDuplicate': 'none', 'text': 'low'}."""
    out: dict[str, str] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        schema, sep, effort = part.partition("=")
        if not sep or not schema.strip() or not effort.strip():
            raise ValueError(f"bad schema effort entry {part!r}; expected Schema=effort")
        out[schema.strip()] = effort.strip()
    return out


def apply(request: dict, schema: str | None, names: tuple[str, ...], schema_effort: dict[str, str]) -> list[str]:
    """Apply named transforms and any per-schema reasoning effort; return what was applied."""
    applied = [n for n in names if TRANSFORMS[n](request, schema)]
    effort = schema_effort.get(schema or "")
    if effort:
        request["reasoning"] = {"effort": effort}
        applied.append(f"effort={effort}")
    return applied
