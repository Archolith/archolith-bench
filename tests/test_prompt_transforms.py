import json

import pytest

from archolith_bench.ci import prompt_transforms as pt
from archolith_bench.ci.budget_proxy import BudgetProxy


def _req(user: str, system: str = "sys") -> dict:
    return {"model": "m", "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


NODE_USER = (
    "<ENTITIES>\n[]\n</ENTITIES>\n\n<EXISTING ENTITIES>\n"
    + json.dumps([{"merge_audit": ["x" * 50], "namespace": "ns", "candidate_id": 0, "name": "shelf 1",
                   "entity_types": ["Entity"], "summary": "Shelf 1 holds pens."}])
    + "\n</EXISTING ENTITIES>\n\nFor each entity..."
)

EDGE_USER = """
NEVER mark facts as duplicates...

<EXISTING FACTS>
[{'idx': 0, 'fact': 'Safe 1 is open.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[{'idx': 1, 'fact': 'Safe 1 is closed.'}]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Safe 1 is closed now.
</NEW FACT>

You will receive TWO lists...
<EXAMPLE>
long example
</EXAMPLE>
"""


def test_node_slim_keeps_only_identity_fields():
    req = _req(NODE_USER)
    assert pt.node_slim(req, "NodeResolutions") is True
    user = req["messages"][1]["content"]
    block = user.split("<EXISTING ENTITIES>")[1].split("</EXISTING ENTITIES>")[0]
    assert json.loads(block) == [{"candidate_id": 0, "name": "shelf 1", "entity_types": ["Entity"],
                                  "summary": "Shelf 1 holds pens."}]
    assert user.startswith("<ENTITIES>") and user.endswith("For each entity...")


def test_node_slim_is_noop_on_other_schema_or_bad_json():
    req = _req(NODE_USER)
    assert pt.node_slim(req, "EdgeDuplicate") is False
    bad = _req("<EXISTING ENTITIES>\nnot json\n</EXISTING ENTITIES>")
    before = json.dumps(bad)
    assert pt.node_slim(bad, "NodeResolutions") is False
    assert json.dumps(bad) == before


def test_node_drop_audit_removes_only_merge_lineage():
    req = _req(NODE_USER.replace('"namespace": "ns"', '"namespace": "ns", "merged_from": ["u1"], "edge_count": 4'))
    assert pt.node_drop_audit(req, "NodeResolutions") is True
    block = req["messages"][1]["content"].split("<EXISTING ENTITIES>")[1].split("</EXISTING ENTITIES>")[0]
    entity = json.loads(block)[0]
    assert "merge_audit" not in entity and "merged_from" not in entity
    assert entity["namespace"] == "ns" and entity["edge_count"] == 4 and entity["name"] == "shelf 1"
    assert pt.node_drop_audit(req, "NodeResolutions") is False  # nothing left to drop


def test_edge_compact_preserves_data_blocks_and_drops_boilerplate():
    req = _req(EDGE_USER)
    assert pt.edge_compact(req, "EdgeDuplicate") is True
    user = req["messages"][1]["content"]
    assert "[{'idx': 0, 'fact': 'Safe 1 is open.'}]" in user
    assert "[{'idx': 1, 'fact': 'Safe 1 is closed.'}]" in user
    assert "<NEW FACT>\nSafe 1 is closed now.\n</NEW FACT>" in user
    assert "long example" not in user and "You will receive TWO lists" not in user
    assert user.startswith(pt.EDGE_COMPACT_INSTRUCTIONS)


def test_extraction_drops_negative_examples_only():
    req = _req("RULES\n<NEGATIVE EXAMPLES>\nA) skip\n</NEGATIVE EXAMPLES>\n<ENTITY TYPES>\nx\n</ENTITY TYPES>")
    assert pt.extraction_no_negative_examples(req, "CombinedExtraction") is True
    user = req["messages"][1]["content"]
    assert "NEGATIVE EXAMPLES" not in user and "RULES" in user and "<ENTITY TYPES>" in user
    assert pt.extraction_no_negative_examples(req, "CombinedExtraction") is False


def test_apply_sets_per_schema_effort_and_reports():
    req = _req(EDGE_USER)
    applied = pt.apply(req, "EdgeDuplicate", ("edge_compact", "node_slim"), {"EdgeDuplicate": "none"})
    assert applied == ["edge_compact", "effort=none"]
    assert req["reasoning"] == {"effort": "none"}
    other = _req("hi")
    assert pt.apply(other, "text", (), {"EdgeDuplicate": "none"}) == []
    assert "reasoning" not in other


def test_parse_schema_effort():
    assert pt.parse_schema_effort("EdgeDuplicate=none, text=low") == {"EdgeDuplicate": "none", "text": "low"}
    assert pt.parse_schema_effort("") == {}
    with pytest.raises(ValueError):
        pt.parse_schema_effort("EdgeDuplicate")


def test_budget_proxy_rejects_unknown_transform(tmp_path):
    with pytest.raises(ValueError, match="unknown prompt transforms"):
        BudgetProxy(api_key="k", trace_file=tmp_path / "t.jsonl", budget_file=tmp_path / "b.json",
                    transforms=("no_such_transform",))


def test_cached_tokens_billed_at_cache_rate(tmp_path):
    proxy = BudgetProxy(api_key="k", trace_file=tmp_path / "t.jsonl", budget_file=tmp_path / "b.json",
                        price_cache_read_per_1m=0.01)
    proxy.state.price_input_per_1m, proxy.state.price_output_per_1m = 0.10, 0.50
    proxy.state.record_usage(1_000_000, 0, cached_tokens=800_000)
    assert proxy.state.usd == pytest.approx(0.2 * 0.10 + 0.8 * 0.01)
    proxy.state.close()
