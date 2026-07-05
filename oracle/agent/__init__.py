"""Agentic orchestration for the a-rag QA type: trace, router, loop, planning.

- ``trace``    — structured per-step trace + token accounting (model id + input/
  output/reasoning tokens) — see Traceability in ``tools.md``.
- ``router``   — the local small-LLM tool router (see Routing Model).
- ``loop``     — the bounded, always-terminating orchestrator (Termination & Loop
  Avoidance).
- ``planner`` / ``multihop`` — multi-hop planning (plan → per-hop loop → compose).
"""
from __future__ import annotations

from .trace import AgentTrace, TraceStep, TokenTotals, STOP_REASONS
from .router import Router, RouterDecision, LocalRouter
from .loop import AgentResult, SynthesisResult, run_agent
from .planner import Planner, PlanResult, LocalPlanner, ProxyPlanner, SingleHopPlanner
from .verify import Verifier, VerifyResult, ProxyVerifier
from .multihop import run_multi_hop

__all__ = [
    "AgentTrace", "TraceStep", "TokenTotals", "STOP_REASONS",
    "Router", "RouterDecision", "LocalRouter",
    "AgentResult", "SynthesisResult", "run_agent",
    "Planner", "PlanResult", "LocalPlanner", "ProxyPlanner", "SingleHopPlanner",
    "Verifier", "VerifyResult", "ProxyVerifier",
    "run_multi_hop",
]
