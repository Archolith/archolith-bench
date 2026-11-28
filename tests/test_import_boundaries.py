"""Import-boundary report for the archolith-bench package reorg (migration step 1).

Finds every import from one benchmark package into another (or from a shared
package into a benchmark) and fails only when a coupling is not in ALLOWLIST, or
when an allowlisted coupling no longer exists. No file moves, no source changes.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

PACKAGE = "archolith_bench"
PKG_ROOT = Path(__file__).resolve().parents[1] / PACKAGE
REPO_ROOT = PKG_ROOT.parent

EXEMPT = {
    "archolith_bench/__init__.py",
    "archolith_bench/__main__.py",
    "archolith_bench/cli.py",
}

_SHARED = {"core", "clients", "progress", "procindex", "prompts"}

# Allowlist: (importer unit, importee unit) -> reason. Today's couplings; may only shrink.
_REGISTRY = "suites/menhir.py dispatches every Menhir ladder; step 4: each benchmark registers itself"
ALLOWLIST: dict[tuple[str, str], str] = {
    ("context_proxy", "bootstrap_hygiene"): _REGISTRY,
    ("context_proxy", "facet"): _REGISTRY,
    ("context_proxy", "intent"): _REGISTRY,
    ("context_proxy", "l4"): _REGISTRY,
    ("context_proxy", "oracle"): _REGISTRY,
    ("context_proxy", "r1"): _REGISTRY,
    ("context_proxy", "r3"): _REGISTRY,
    ("context_proxy", "r5"): _REGISTRY,
    ("engineering_evidence", "beacon_eval"): "Budget/DEFAULT_RUN_RESERVE -> core/budget.py; export_commit, ANSWER_INSTRUCTIONS, models, scoring need a shared agent-task core",
    ("engineering_evidence", "longmemeval"): "campaign.py uses staged.rundir; step 2: core/rundir.py",
    ("intent", "oracle"): "intent reuses oracle's combiner, executor and scorer; needs a shared ranking core",
    ("longmemeval", "context_proxy"): "harness/base.py imports the arms.py registry; step 4: classify arms.py",
}

_TOP_DIRS = {p.name for p in PKG_ROOT.iterdir() if p.is_dir()}
_TOP_MODS = {p.stem for p in PKG_ROOT.glob("*.py")}

DIR_UNITS = {
    "core": "core",
    "prompts": "prompts",
    "staged": "longmemeval",
    "build": "longmemeval",
    "ci": "longmemeval",
    "beacon_eval": "beacon_eval",
    "engineering_evidence": "engineering_evidence",
    "r1": "r1",
    "facet": "facet",
    "oracle": "oracle",
    "intent": "intent",
    "r3": "r3",
    "r5": "r5",
    "l4": "l4",
    "bootstrap_hygiene": "bootstrap_hygiene",
    "suites": "context_proxy",
}
for _name in _TOP_DIRS:
    DIR_UNITS.setdefault(_name, _name)

SINGLE_UNITS = {
    "progress": "progress",
    "procindex": "procindex",
    "arms": "context_proxy",
    "extraction_sim": "extraction",
    "dashboard": "longmemeval",
}


def _unit_for_file(rel: str) -> str | None:
    if rel in EXEMPT:
        return None
    rest = rel[len(PACKAGE) + 1:]
    first = rest.split("/")[0]
    if "/" not in rest:
        return SINGLE_UNITS.get(first, "structured_state")
    if first == "harness":
        return "clients" if rest == "harness/menhir_client.py" else "longmemeval"
    return DIR_UNITS.get(first, first)


def _unit_for_module(modname: str) -> str | None:
    if not modname.startswith(PACKAGE):
        return None
    rest = modname[len(PACKAGE):].lstrip(".")
    if not rest:
        return None
    first = rest.split(".")[0]
    if first == "harness":
        if rest == "harness.menhir_client" or rest.startswith("harness.menhir_client."):
            return "clients"
        return "longmemeval"
    if first in DIR_UNITS:
        return DIR_UNITS[first]
    if first in _TOP_MODS:
        return SINGLE_UNITS.get(first, "structured_state")
    return first


def _relative_base(rel: str, level: int, module: str | None) -> str | None:
    parts = rel.split("/")[:-1]
    if level <= 0:
        return module
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join(base + [module]) if module else ".".join(base)


def _scan_python_files() -> list[tuple[str, ast.AST]]:
    parsed: list[tuple[str, ast.AST]] = []
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        parsed.append((rel, ast.parse(path.read_text(encoding="utf-8"), filename=rel)))
    return parsed


_CACHE: dict[tuple[str, str], set[str]] | None = None


def _observe() -> dict[tuple[str, str], set[str]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    couplings: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rel, tree in _scan_python_files():
        importer = _unit_for_file(rel)
        if importer is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    _relative_base(rel, node.level, node.module)
                    if node.level
                    else (node.module or "")
                )
                if not base:
                    continue
                targets = [base] + [f"{base}.{alias.name}" for alias in node.names]
            else:
                continue
            for target in targets:
                importee = _unit_for_module(target)
                # Rule 4: any unit may import a shared unit, so only imports into a benchmark count.
                if importee is not None and importee != importer and importee not in _SHARED:
                    couplings[(importer, importee)].add(rel)
    _CACHE = dict(couplings)
    return _CACHE


def _format(pairs: list[tuple[str, str]]) -> str:
    observed = _observe()
    lines = []
    for pair in pairs:
        lines.append(f"{pair[0]} -> {pair[1]} ({len(observed.get(pair, ()))} files):")
        lines.extend(f"    {f}" for f in sorted(observed.get(pair, ())))
    return "\n".join(lines)


def test_no_new_cross_unit_imports() -> None:
    new = sorted(pair for pair in _observe() if pair not in ALLOWLIST)
    assert not new, "New cross-unit imports not in ALLOWLIST:\n" + _format(new)


def test_allowlist_has_no_stale_entries() -> None:
    observed = _observe()
    stale = sorted(pair for pair in ALLOWLIST if pair not in observed)
    assert not stale, "ALLOWLIST entries no longer observed (shrink the allowlist):\n" + "\n".join(
        f"    {a} -> {b}: {ALLOWLIST[(a, b)]}" for a, b in stale
    )


def test_shared_units_import_no_benchmark() -> None:
    observed = _observe()
    bad = sorted(
        pair
        for pair in observed
        if pair[0] in _SHARED and pair[1] not in _SHARED and pair not in ALLOWLIST
    )
    assert not bad, "Shared units must not import benchmark units:\n" + _format(bad)


def test_mapping_covers_every_module() -> None:
    unmapped = []
    for rel, _tree in _scan_python_files():
        unit = _unit_for_file(rel)
        if rel not in EXEMPT and not unit:
            unmapped.append(rel)
    assert not unmapped, "Modules mapped to no unit:\n" + "\n".join(f"    {m}" for m in unmapped)
