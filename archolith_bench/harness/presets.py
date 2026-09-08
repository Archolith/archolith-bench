"""Named smoke presets for the harness suite.

The adapters already work; what was missing was a cheap way to invoke one. The
only options were "nothing" or a full official run, so incremental evidence
never got gathered. A preset fixes the size knobs -- benchmark id, subset,
limit, arms -- at a deliberately small scale.

Presets set scale, never safety. No preset enables `--confirm-menhir-reset`,
picks a menhir URL, or otherwise authorises a mutating or paid action; those
stay explicit at the call site. A preset that could silently reset a graph
would be a footgun wearing a convenience label.

Explicit flags always win: `_apply_preset` only fills a value the caller left
at its parser default.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HarnessPreset:
    """A named, deliberately small harness configuration."""

    name: str
    benchmark_id: str
    description: str
    # Parsed-arg overrides applied only where the caller kept the default.
    overrides: dict[str, object] = field(default_factory=dict)
    # Extra flags the operator must supply; a preset never supplies them itself.
    requires: tuple[str, ...] = ()

    @property
    def cost(self) -> str:
        return "offline" if self.overrides.get("offline_fixture") else "live API"


# Item counts are chosen to be affordable rather than statistically meaningful.
# A smoke run answers "does this adapter still work end to end", not "how good
# is the proxy" -- which is why every description says so, and why none of
# these is a promotable evidence source on its own.
PRESETS: dict[str, HarnessPreset] = {
    "longbench-v2-smoke": HarnessPreset(
        name="longbench-v2-smoke",
        benchmark_id="longbench-v2",
        description="10 LongBench v2 tasks, direct vs proxy_only. Long-context sanity check.",
        overrides={"limit": 10, "arms": "direct,proxy_only"},
    ),
    "bigcodebench-hard-smoke": HarnessPreset(
        name="bigcodebench-hard-smoke",
        benchmark_id="bigcodebench-hard",
        description="10 BigCodeBench-Hard tasks, direct vs proxy_only. Proxy-overhead check.",
        overrides={"limit": 10, "arms": "direct,proxy_only"},
    ),
    "swe-bench-smoke": HarnessPreset(
        name="swe-bench-smoke",
        benchmark_id="swe-bench",
        description="5 SWE-bench Lite instances, direct vs proxy_only. Slowest preset; "
                    "wraps an external CLI.",
        overrides={"limit": 5, "subset": "lite", "arms": "direct,proxy_only"},
    ),
    "longmemeval-proxy-smoke": HarnessPreset(
        name="longmemeval-proxy-smoke",
        benchmark_id="longmemeval",
        description="10 LongMemEval items with the history IN-CONTEXT (Mode A). Tests proxy "
                    "context curation, NOT menhir persistent memory.",
        overrides={"limit": 10, "arms": "direct,proxy_only"},
    ),
    "longmemeval-menhir-smoke": HarnessPreset(
        name="longmemeval-menhir-smoke",
        benchmark_id="longmemeval-menhir",
        description="10 LongMemEval items through menhir memory (Mode B). Requires a "
                    "throwaway menhir; refuses production-looking targets.",
        # Arm strings, not constant names: SINGLE_RECALL == "menhir_recall".
        overrides={"limit": 10, "arms": "no_memory,menhir_recall"},
        requires=("--menhir-url", "--confirm-menhir-reset"),
    ),
}


def get_preset(name: str) -> HarnessPreset:
    """Return a preset by name, or raise with the known set."""
    try:
        return PRESETS[name]
    except KeyError as e:
        raise KeyError(
            f"unknown preset: {name!r} (available: {', '.join(sorted(PRESETS))})"
        ) from e


class PresetConflict(ValueError):
    """Raised when a preset and an explicit argument disagree."""


def _explicit_dests(parser, argv: list[str]) -> set[str]:
    """Which destinations the operator actually typed on the command line.

    Value equality cannot answer this: `--arms direct,proxy_only` is both an
    explicit choice and the parser default, and the default is the exact string
    a user copies out of `--help`. Asking argv distinguishes "omitted" from
    "passed, and happens to match the default".
    """
    opt_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            opt_to_dest[opt] = action.dest

    seen: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        name = token.split("=", 1)[0]
        dest = opt_to_dest.get(name)
        if dest is not None:
            seen.add(dest)
    return seen


def apply_preset(preset: HarnessPreset, args, parser, argv: list[str] | None = None) -> list[str]:
    """Apply *preset* to parsed *args* without clobbering explicit flags.

    Explicit flags always win, decided by what appears in *argv* rather than by
    comparing against the default. Returns the names of the options the preset
    actually set, for reporting.

    Raises PresetConflict when an explicit benchmark differs from the preset's:
    merging them runs one benchmark with another's scale and subset while the
    console prints the preset's description, so the terminal and the published
    evidence record disagree about what ran.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    explicit = _explicit_dests(parser, argv)
    applied: list[str] = []

    current_benchmark = getattr(args, "benchmark_id", None)
    if current_benchmark in (None, ""):
        args.benchmark_id = preset.benchmark_id
        applied.append("benchmark_id")
    elif current_benchmark != preset.benchmark_id:
        raise PresetConflict(
            f"preset {preset.name!r} is for benchmark {preset.benchmark_id!r}, but "
            f"{current_benchmark!r} was requested. Drop one: the preset's scale and "
            f"subset are calibrated for its own benchmark."
        )

    for key, value in preset.overrides.items():
        if not hasattr(args, key):
            continue
        if key in explicit:
            continue
        setattr(args, key, value)
        applied.append(key)

    return applied


def format_preset_list() -> str:
    """Render the preset table for `--list-presets`."""
    lines = ["Available harness smoke presets:"]
    for name in sorted(PRESETS):
        p = PRESETS[name]
        lines.append(f"  {name}  [{p.benchmark_id}] ({p.cost})")
        lines.append(f"    {p.description}")
        if p.requires:
            lines.append(f"    requires: {' '.join(p.requires)}")
    lines.append("")
    lines.append("Smoke presets are adapter sanity checks, not promotable evidence.")
    lines.append("Publish a run with --publish-evidence to record its provenance.")
    return "\n".join(lines)
