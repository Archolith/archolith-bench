"""Beacon agent-task evaluation (P3): does a Beacon endpoint help a coding agent orient?

Three conditions run the same task with the same model on the same pinned commit:

* ``A`` -- the checkout only (README, AGENTS.md, docs; normal file tools);
* ``B`` -- A plus the Beacon MCP endpoint (the only MCP server the agent has);
* ``C`` -- A plus the built ``beacon.generated.yaml`` pasted into the prompt;
* ``D`` -- Beacon MCP only: OpenCode's built-in file and shell tools are off (optional).

Answers are scored deterministically against human-reviewed gold answers. Plan:
``IdeaProjects/.agent/plans/beacon-p3-agent-task-evaluation-plan-2026-09-23.md``.
"""

from __future__ import annotations

CONDITIONS = ("A", "B", "C", "D")
#: What a run uses when --conditions is not given; D is opt-in.
DEFAULT_CONDITIONS = ("A", "B", "C")
