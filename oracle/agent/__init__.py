"""Agentic orchestration for the a-rag QA type: trace, router, and loop.

- ``trace``  — structured per-step trace + token accounting (see Traceability in
  ``tools.md``).
- ``router`` — the local small-LLM tool router (see Routing Model).
- ``loop``   — the bounded, always-terminating orchestrator (see Termination &
  Loop Avoidance).
"""
from __future__ import annotations

from .trace import AgentTrace, TraceStep, TokenTotals, STOP_REASONS
from .router import Router, RouterDecision, LocalRouter
from .loop import AgentResult, run_agent

__all__ = [
    "AgentTrace", "TraceStep", "TokenTotals", "STOP_REASONS",
    "Router", "RouterDecision", "LocalRouter",
    "AgentResult", "run_agent",
]
