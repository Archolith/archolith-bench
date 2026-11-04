import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "replay_ab", Path(__file__).resolve().parents[1] / "scripts" / "ama" / "replay_ab.py")
replay_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay_ab)


def test_parse_arm():
    assert replay_ab.parse_arm("A:") == ("A", (), None)
    assert replay_ab.parse_arm("B:node_slim,+effort=none") == ("B", ("node_slim",), "none")
    with pytest.raises(SystemExit):
        replay_ab.parse_arm("C:bogus")
    assert replay_ab.parse_arm("R:+max_tokens=1024") == ("R", (), None)
    assert replay_ab.MAX_TOKENS["R"] == 1024


def test_edge_duplicate_agreement_is_order_insensitive():
    a = replay_ab.decisions("EdgeDuplicate", '{"duplicate_facts":[1,0],"contradicted_facts":[]}')
    b = replay_ab.decisions("EdgeDuplicate", '{"duplicate_facts":[0,1],"contradicted_facts":[2]}')
    assert replay_ab.agree("EdgeDuplicate", a, b) == (1, 2)


def test_node_resolution_agreement_per_entity():
    a = replay_ab.decisions("NodeResolutions",
                            '{"entity_resolutions":[{"id":0,"duplicate_candidate_id":-1},{"id":1,"duplicate_candidate_id":3}]}')
    b = replay_ab.decisions("NodeResolutions",
                            '{"entity_resolutions":[{"id":0,"duplicate_candidate_id":-1},{"id":1,"duplicate_candidate_id":2}]}')
    assert replay_ab.agree("NodeResolutions", a, b) == (1, 2)


def test_text_verdict_and_unparseable_response():
    assert replay_ab.decisions("text", "SAME.") == {"verdict": "SAME"}
    assert replay_ab.decisions("text", "") == {"verdict": None}
    assert replay_ab.agree("EdgeDuplicate", None, {"dup": (), "contra": ()}) == (0, 2)
