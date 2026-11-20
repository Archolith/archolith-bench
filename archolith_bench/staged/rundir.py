"""On-disk run directory with append-only, resumable stage files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

STAGE_NAMES = ("search", "answers", "judged")


class RunConfigMismatch(RuntimeError):
    """The run directory already holds a different resolved config."""


class StageFile:
    """One JSONL stage file; a truncated trailing line is ignored."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        text = self.path.read_text(encoding="utf-8")
        for line in text.split("\n"):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def done_ids(self) -> set[str]:
        return {str(r["item_id"]) for r in self.records() if "item_id" in r}

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_nl = False
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_nl = f.read(1) != b"\n"
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            if needs_nl:
                f.write("\n")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


class RunDir:
    """`root/run_id` with a resolved config and stage files."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id
        self.path = self.root / run_id

    def write_config(self, resolved: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / "config.resolved.yaml"
        content = yaml.safe_dump(resolved, sort_keys=True)
        if target.exists():
            if target.read_text(encoding="utf-8") != content:
                raise RunConfigMismatch(f"{target} differs from the config being run")
            return
        target.write_text(content, encoding="utf-8")

    def stage(self, name: str) -> StageFile:
        if name not in STAGE_NAMES:
            raise ValueError(f"unknown stage: {name}")
        return StageFile(self.path / f"{name}.jsonl")

    def write_report(self, report: dict) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [f"# {self.run_id}", "", f"n: {report.get('n')}", f"accuracy: {report.get('accuracy')}", ""]
        for qtype, row in sorted((report.get("per_type") or {}).items()):
            lines.append(f"- {qtype}: n={row['n']} accuracy={row['accuracy']}")
        (self.path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
