"""Multi-hop orchestration: plan → answer each hop → compose, recursively.

Implements the *plan-then-execute with a facts scratchpad* idea from ``tools.md``,
now with two extensions:

1. **Recursive bounded sub-agents.** A hop's sub-question may itself be planned and
   decomposed, down to ``max_depth`` planning levels; each level's steps are tagged
   with their recursion ``depth`` so the trace nests. ``max_depth=1`` is the flat
   plan-once behaviour; deeper levels help genuinely multi-step questions. The base
   case (max depth reached, or a sub-question the planner leaves atomic) runs the
   bounded ``run_agent`` leaf loop.
2. **Verification / backtracking hop.** After the top-level answer is composed, an
   optional bounded verifier re-checks it against the gathered facts and, on a
   contradiction, returns a corrected answer (a one-shot backtrack). It runs once
   and cannot loop.

Depth × per-hop step budget × the wall-clock timeout bound the whole tree, so it
always terminates. ``run_hop`` (leaf), ``synthesize_final`` and ``verifier`` are
injected, so this is unit-testable with fakes and no model.
"""
from __future__ import annotations

import time
from typing import Callable

from .loop import AgentResult, _short
from .planner import Planner
from .trace import AgentTrace, TraceStep
from .verify import Verifier

RunHop = Callable[[str, list], AgentResult]
SynthesizeFinal = Callable[[str, list, list], "object"]


def _facts_message(facts: list[tuple[str, str]]) -> dict:
    body = "\n".join(f"- {q}: {a}" for q, a in facts)
    return {"role": "user", "content": f"Known facts so far:\n{body}"}


def run_multi_hop(
    *,
    question: str,
    history: list[dict],
    planner: Planner,
    run_hop: RunHop,
    synthesize_final: SynthesizeFinal,
    verifier: Verifier | None = None,
    verify: bool = False,
    max_hops: int = 3,
    max_depth: int = 1,
    depth: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentResult:
    history = history or []
    start = clock()
    trace = AgentTrace()
    idx = 0

    def _add(step: TraceStep) -> None:
        nonlocal idx
        step.index = idx
        trace.add(step)
        idx += 1

    # Plan only while we still have planning depth; otherwise treat as atomic.
    subs = [question]
    if depth < max_depth:
        plan = planner.plan(question, history)
        subs = plan.subquestions[:max_hops] or [question]
        # Always record the planning step (even a single-item "atomic" plan) so the
        # user can see that planning happened and what the planner decided.
        summary = ("plan: " + " | ".join(subs)) if len(subs) > 1 \
            else "atomic — no decomposition needed"
        _add(TraceStep(0, "planner", None, None, _short(summary),
                       prompt_tokens=plan.prompt_tokens,
                       completion_tokens=plan.completion_tokens,
                       reasoning_tokens=plan.reasoning_tokens,
                       model_id=plan.model_id, depth=depth))

    facts: list[tuple[str, str]] = []
    hop_stops: list[str] = []

    if len(subs) <= 1:
        # Atomic (or max depth reached): answer directly with the leaf loop.
        leaf = run_hop(question, history)
        for s in leaf.trace.steps:
            s.depth = depth
            _add(s)
        hop_stops.append(leaf.trace.stop_reason)
        answer = leaf.answer
    else:
        for hop_i, subq in enumerate(subs):
            hop_history = history + ([_facts_message(facts)] if facts else [])
            if depth + 1 < max_depth:
                child = run_multi_hop(
                    question=subq, history=hop_history, planner=planner,
                    run_hop=run_hop, synthesize_final=synthesize_final,
                    verifier=verifier, verify=False,  # verify only at the top level
                    max_hops=max_hops, max_depth=max_depth, depth=depth + 1, clock=clock,
                )
                child_steps = child.trace.steps  # already depth-tagged by the child
            else:
                child = run_hop(subq, hop_history)
                child_steps = child.trace.steps
                for s in child_steps:
                    s.depth = depth + 1
            for s in child_steps:
                s.hop = hop_i
                _add(s)
            hop_stops.append(child.trace.stop_reason)
            facts.append((subq, child.answer))

        fact_obs = [{"tool": "hop", "arguments": {"question": q}, "result": a}
                    for q, a in facts]
        syn = synthesize_final(question, history, fact_obs)
        _add(TraceStep(0, "synthesis", None, None, _short(syn.text),
                       prompt_tokens=syn.prompt_tokens, completion_tokens=syn.completion_tokens,
                       reasoning_tokens=syn.reasoning_tokens, elapsed_seconds=syn.elapsed_seconds,
                       model_id=syn.model_id, depth=depth))
        answer = syn.text.strip()

    # Verification / backtracking hop — top level only, bounded to a single check.
    if verify and verifier is not None and depth == 0:
        v = verifier.verify(question, answer, facts)
        note = v.note if v.ok else f"backtrack: {v.note}"
        _add(TraceStep(0, "verify", None, None, _short(note),
                       prompt_tokens=v.prompt_tokens, completion_tokens=v.completion_tokens,
                       reasoning_tokens=v.reasoning_tokens, model_id=v.model_id, depth=depth))
        if not v.ok and v.corrected_answer:
            answer = v.corrected_answer.strip()

    trace.stop_reason = next((r for r in hop_stops if r != "done"), "done")
    trace.elapsed_seconds = round(clock() - start, 3)
    return AgentResult(answer=answer.strip(), trace=trace)
