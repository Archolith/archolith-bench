"""Beacon agent-task evaluation (P3): does a Beacon endpoint help a coding agent orient?

Three conditions run the same task with the same model on the same pinned commit:

* ``A`` -- the checkout only (README, AGENTS.md, docs; normal file tools);
* ``B`` -- A plus the Beacon MCP endpoint (the only MCP server the agent has);
* ``C`` -- A plus the built ``beacon.generated.yaml`` pasted into the prompt;
* ``D`` -- Beacon MCP only: OpenCode's built-in file and shell tools are off (optional);
* ``H`` -- Beacon HTTP JSON only: no MCP and no checkout access, built-in tools off except
  ``webfetch``, which reads the loopback discovery document and routes (optional);
* ``M`` -- A plus a Menhir memory MCP endpoint (read-only key; optional, for "why" tasks);
* ``R`` -- Beacon MCP over Streamable HTTP only: like D, but the server is a remote
  ``http://127.0.0.1`` endpoint instead of a local stdio process (optional).

Answers are scored deterministically against human-reviewed gold answers. Plan:
``IdeaProjects/.agent/plans/beacon-p3-agent-task-evaluation-plan-2026-09-23.md``.
"""

from __future__ import annotations

CONDITIONS = ("A", "B", "C", "D", "H", "M", "R")
#: What a run uses when --conditions is not given; D, H, M and R are opt-in.
DEFAULT_CONDITIONS = ("A", "B", "C")
