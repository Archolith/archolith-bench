"""Beacon agent-task evaluation (P3): does a Beacon endpoint help a coding agent orient?

Three conditions run the same task with the same model on the same pinned commit:

* ``A`` -- the checkout only (README, AGENTS.md, docs; normal file tools);
* ``B`` -- A plus the Beacon MCP endpoint (the only MCP server the agent has);
* ``C`` -- A plus the built ``beacon.generated.yaml`` pasted into the prompt.

Answers are scored deterministically against human-reviewed gold answers. Plan:
``IdeaProjects/.agent/plans/beacon-p3-agent-task-evaluation-plan-2026-09-23.md``.
"""

from __future__ import annotations

CONDITIONS = ("A", "B", "C")
