"""Citation locations resolve the same on case-sensitive (Linux) and Windows file systems."""

from __future__ import annotations

from pathlib import Path

from archolith_bench.beacon_eval.models import Gold
from archolith_bench.beacon_eval.scoring import citation_location_validity, score


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "Docs").mkdir(parents=True)
    (root / "AGENTS.md").write_bytes(b"one\ntwo\n")
    (root / "Docs" / "Guide.md").write_bytes(b"a\nb\nc\n")
    (tmp_path / "outside.md").write_bytes(b"x\n")
    return root


def _cite(*paths: str) -> dict:
    return {"citations": [{"path": p, "line_start": 1, "line_end": 2} for p in paths]}


def test_a_citation_matches_its_file_whatever_the_case(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    answer = _cite("AGENTS.md", "agents.md", "./docs/guide.md", "`DOCS/GUIDE.MD`")
    assert citation_location_validity(answer, root) == 1.0


def test_missing_files_and_paths_outside_the_checkout_are_invalid(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    answer = _cite("nope.md", "../outside.md", "docs/../../outside.md", "Docs")
    assert citation_location_validity(answer, root) == 0.0


def test_gold_matching_still_ignores_case(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    answer = {"docs": ["agents.md"], **_cite("agents.md")}
    scores = score(answer, Gold(docs=("AGENTS.md",)), root)
    assert scores["doc_recall"] == 1.0 and scores["citation_location_validity"] == 1.0
