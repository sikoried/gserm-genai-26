"""Multi-hop orchestration: plan → answer each hop → compose.

Implements the *plan-then-execute with a facts scratchpad* idea from ``tools.md``:

1. the planner decomposes the question into ordered sub-questions;
2. each sub-question is answered by the bounded ``run_agent`` loop, with the
   answers of earlier hops threaded forward as **known facts**;
3. a final synthesis composes the hop answers into the answer to the original
   question (skipped when the plan is a single hop — that reproduces the plain,
   non-planning behaviour).

Depth is bounded by ``max_hops`` and each hop carries the loop's own step budget /
timeout, so the whole tree always terminates. Traces nest: a ``planner`` step, then
every hop's steps tagged with their hop index, then the final ``synthesis``. Token
accounting rolls the hops' cost up into the per-question total.

``run_hop`` and ``synthesize_final`` are injected, so this is unit-testable with
fakes and no model.
"""
from __future__ import annotations

import time
from typing import Callable

from .loop import AgentResult, SynthesisResult, _short
from .planner import Planner
from .trace import AgentTrace, TraceStep

# run_hop(subquestion, hop_history) -> AgentResult
RunHop = Callable[[str, list], AgentResult]
# synthesize_final(question, history, fact_observations) -> SynthesisResult
SynthesizeFinal = Callable[[str, list, list], SynthesisResult]


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
    max_hops: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentResult:
    history = history or []
    start = clock()
    trace = AgentTrace()

    plan = planner.plan(question, history)
    subs = plan.subquestions[:max_hops] or [question]
    trace.add(TraceStep(
        index=0, kind="planner", tool=None, arguments=None,
        result=_short("plan: " + " | ".join(subs)),
        prompt_tokens=plan.prompt_tokens, completion_tokens=plan.completion_tokens,
        reasoning_tokens=plan.reasoning_tokens, model_id=plan.model_id,
    ))

    idx = 1
    facts: list[tuple[str, str]] = []
    hop_stop_reasons: list[str] = []
    last_answer = ""

    for hop_i, subq in enumerate(subs):
        hop_history = list(history)
        if facts:
            hop_history = hop_history + [_facts_message(facts)]
        result = run_hop(subq, hop_history)
        for s in result.trace.steps:  # merge, renumber, tag with hop index
            s.index = idx
            s.hop = hop_i
            trace.add(s)
            idx += 1
        hop_stop_reasons.append(result.trace.stop_reason)
        facts.append((subq, result.answer))
        last_answer = result.answer

    if len(subs) > 1:
        fact_obs = [{"tool": "hop", "arguments": {"question": q}, "result": a}
                    for q, a in facts]
        syn = synthesize_final(question, history, fact_obs)
        trace.add(TraceStep(
            index=idx, kind="synthesis", tool=None, arguments=None,
            result=_short(syn.text), prompt_tokens=syn.prompt_tokens,
            completion_tokens=syn.completion_tokens, reasoning_tokens=syn.reasoning_tokens,
            elapsed_seconds=syn.elapsed_seconds, model_id=syn.model_id,
        ))
        answer = syn.text.strip()
    else:
        answer = last_answer.strip()

    # Surface the first non-"done" hop reason so a truncated hop is visible.
    trace.stop_reason = next((r for r in hop_stop_reasons if r != "done"), "done")
    trace.elapsed_seconds = round(clock() - start, 3)
    return AgentResult(answer=answer, trace=trace)
