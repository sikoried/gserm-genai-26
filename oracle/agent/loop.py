"""The bounded, always-terminating agent loop.

Contract (see *Termination & Loop Avoidance* in ``tools.md``): every call ends in
exactly one synthesis step and returns an answer. The router drives tool
selection; this loop enforces the guards that guarantee progress:

- **step budget** — at most ``max_steps`` tool calls; on exhaustion we stop and
  synthesize from whatever was gathered (``stop_reason="step-budget"``);
- **loop guard** — a repeated (tool, arguments) call, or a tool that returns an
  observation already seen, stops the loop (``"loop-guard"``);
- **timeout** — a wall-clock budget; on overrun we stop and synthesize
  (``"timeout"``);
- **done** — the router decided enough is known (``"done"``).

Everything heavy (the router model, the tools, the synthesis LLM call) is
injected, so the loop is fully unit-testable with fakes and a fake clock.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from .router import Router
from .trace import AgentTrace, TraceStep

# synthesize(question, history, observations) -> (answer_text, prompt_tokens,
# completion_tokens, elapsed_seconds)
Synthesize = Callable[[str, list, list], "tuple[str, int, int, float]"]


@dataclass
class AgentResult:
    answer: str
    trace: AgentTrace


def _short(text: str, limit: int = 240) -> str:
    s = " ".join(str(text).split())
    return s[:limit] + ("…" if len(s) > limit else "")


def _key(tool: str, arguments: dict | None) -> str:
    return tool + "|" + json.dumps(arguments or {}, sort_keys=True, default=str)


def run_agent(
    *,
    question: str,
    history: list[dict],
    router: Router,
    tools_by_name: dict,
    synthesize: Synthesize,
    reset_tool_tokens: Callable[[], None] = lambda: None,
    get_tool_tokens: Callable[[], "tuple[int, int]"] = lambda: (0, 0),
    max_steps: int = 6,
    timeout_seconds: float = 120.0,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentResult:
    history = history or []
    trace = AgentTrace()
    observations: list[dict] = []
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    start = clock()
    stop_reason = "done"
    idx = 0

    while True:
        if clock() - start > timeout_seconds:
            stop_reason = "timeout"
            break
        if len([s for s in trace.steps if s.kind == "tool"]) >= max_steps:
            stop_reason = "step-budget"
            break

        decision = router.decide(question, history, observations)
        trace.add(TraceStep(
            index=idx, kind="router", tool=decision.tool,
            arguments=decision.arguments,
            result="finish" if decision.finished else f"call {decision.tool}",
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
        ))
        idx += 1

        if decision.finished or not decision.tool:
            stop_reason = "done"
            break

        if decision.tool not in tools_by_name:
            # Unknown tool: record it and let the router try again (bounded by budget).
            trace.add(TraceStep(index=idx, kind="tool", tool=decision.tool,
                                arguments=decision.arguments,
                                result=f"Unknown tool: {decision.tool}"))
            idx += 1
            continue

        call_key = _key(decision.tool, decision.arguments)
        if call_key in seen_calls:
            stop_reason = "loop-guard"
            break
        seen_calls.add(call_key)

        reset_tool_tokens()
        t0 = clock()
        try:
            result = str(tools_by_name[decision.tool](**(decision.arguments or {})))
        except Exception as exc:  # a tool blowing up must not kill the answer
            result = f"Tool error: {exc}"
        elapsed = clock() - t0
        ptok, ctok = get_tool_tokens()

        trace.add(TraceStep(
            index=idx, kind="tool", tool=decision.tool, arguments=decision.arguments,
            result=_short(result), prompt_tokens=ptok, completion_tokens=ctok,
            elapsed_seconds=elapsed,
        ))
        idx += 1
        observations.append({"tool": decision.tool, "arguments": decision.arguments,
                             "result": result})

        norm = _short(result, 400)
        if norm in seen_results:
            stop_reason = "loop-guard"  # no new information — stop and answer
            break
        seen_results.add(norm)

    # Always synthesize the final answer (the single conclusion) on the big model.
    answer, sp, sc, selapsed = synthesize(question, history, observations)
    trace.add(TraceStep(index=idx, kind="synthesis", tool=None, arguments=None,
                        result=_short(answer), prompt_tokens=sp,
                        completion_tokens=sc, elapsed_seconds=selapsed))
    trace.stop_reason = stop_reason
    trace.elapsed_seconds = round(clock() - start, 3)
    return AgentResult(answer=answer.strip(), trace=trace)
