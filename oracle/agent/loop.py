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
injected, so the loop is fully unit-testable with fakes and a fake clock. Each
step records which model produced it and its input/output/reasoning tokens.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable

from .router import Router
from .trace import AgentTrace, TraceStep


@dataclass
class SynthesisResult:
    """What a synthesize(...) call returns: the answer text plus its token cost."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    elapsed_seconds: float = 0.0
    model_id: str | None = None


# synthesize(question, history, observations) -> SynthesisResult
Synthesize = Callable[[str, list, list], SynthesisResult]


@dataclass
class AgentResult:
    answer: str
    trace: AgentTrace


def _short(text: str, limit: int = 240) -> str:
    s = " ".join(str(text).split())
    return s[:limit] + ("…" if len(s) > limit else "")


def _key(tool: str, arguments: dict | None) -> str:
    return tool + "|" + json.dumps(arguments or {}, sort_keys=True, default=str)


def _arg_tokens(arguments: dict | None) -> frozenset:
    """Word tokens from a call's string arguments (for near-duplicate detection)."""
    text = " ".join(str(v) for v in (arguments or {}).values() if isinstance(v, str))
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _similar(a: frozenset, b: frozenset, threshold: float = 0.6) -> bool:
    """True if two token sets overlap heavily (Jaccard ≥ threshold)."""
    if not a or not b:
        return a == b
    return len(a & b) / len(a | b) >= threshold


_EMPTY_PREFIXES = ("no results", "no videos", "no article", "no query", "no expression",
                   "no candidates", "no title", "tool error", "unknown ")


def _is_empty(result: str) -> bool:
    """A tool result that carries no usable answer (so a rephrased retry is fair)."""
    t = (result or "").strip().lower()
    return (not t or "unavailable" in t or "no internet" in t
            or t.startswith(_EMPTY_PREFIXES))


def run_agent(
    *,
    question: str,
    history: list[dict],
    router: Router,
    tools_by_name: dict,
    synthesize: Synthesize,
    reset_tool_tokens: Callable[[], None] = lambda: None,
    get_tool_tokens: Callable[[], "tuple[int, int, int]"] = lambda: (0, 0, 0),
    get_tool_model: Callable[[], "str | None"] = lambda: None,
    max_steps: int = 6,
    timeout_seconds: float = 120.0,
    rag_first: bool = True,
    force_youtube: bool = False,
    call_cache: dict | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentResult:
    history = history or []
    trace = AgentTrace()
    observations: list[dict] = []
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    # Query token-sets already tried per tool that RETURNED a useful result, so a
    # reworded but equivalent re-query (e.g. two web searches for the same thing) is
    # caught even when the wording differs.
    productive_queries: dict[str, list[frozenset]] = {}
    start = clock()
    stop_reason = "done"
    idx = 0

    def _record_query(name: str, arguments: dict, result: str) -> None:
        if not _is_empty(result):
            productive_queries.setdefault(name, []).append(_arg_tokens(arguments))

    def _is_redundant(name: str, arguments: dict) -> bool:
        tokens = _arg_tokens(arguments)
        return any(_similar(tokens, prev) for prev in productive_queries.get(name, []))

    def _run_tool(name: str, arguments: dict) -> tuple[str, "TraceStep"]:
        nonlocal idx
        key = _key(name, arguments)
        # Shared result cache: never execute the same (tool, args) twice within one
        # question (e.g. across multi-hop hops) — reuse the earlier result.
        if call_cache is not None and key in call_cache:
            out = call_cache[key]
            step = TraceStep(index=idx, kind="tool", tool=name, arguments=arguments,
                             result=_short(out) + "  (cached)", model_id=None)
            idx += 1
            return out, step
        reset_tool_tokens()
        t0 = clock()
        try:
            out = str(tools_by_name[name](**(arguments or {})))
        except Exception as exc:  # a tool blowing up must not kill the answer
            out = f"Tool error: {exc}"
        p, c, r = get_tool_tokens()
        step = TraceStep(index=idx, kind="tool", tool=name, arguments=arguments,
                         result=_short(out), prompt_tokens=p, completion_tokens=c,
                         reasoning_tokens=r, elapsed_seconds=clock() - t0,
                         model_id=get_tool_model())
        idx += 1
        if call_cache is not None:
            call_cache[key] = out
        return out, step

    # RAG-first: always consult the local index before the router picks tools, so the
    # agent grounds on the corpus and only reaches for other tools if it's not enough.
    if rag_first and "search" in tools_by_name:
        args = {"query": question}
        result, step = _run_tool("search", args)
        trace.add(step)
        observations.append({"tool": "search", "arguments": args, "result": result})
        seen_calls.add(_key("search", args))
        seen_results.add(_short(result, 400))
        _record_query("search", args, result)

    # For questions that explicitly reference a video, force a YouTube lookup up front
    # so its transcript is in the evidence — the router otherwise settles for a web
    # result and never reaches `youtube`.
    if force_youtube and "youtube" in tools_by_name:
        args = {"query": question}
        result, step = _run_tool("youtube", args)
        trace.add(step)
        observations.append({"tool": "youtube", "arguments": args, "result": result})
        seen_calls.add(_key("youtube", args))
        seen_results.add(_short(result, 400))
        _record_query("youtube", args, result)

    while True:
        if clock() - start > timeout_seconds:
            stop_reason = "timeout"
            break
        if len([s for s in trace.steps if s.kind == "tool"]) >= max_steps:
            stop_reason = "step-budget"
            break

        decision = router.decide(question, history, observations)

        # The small router often re-requests a call it just made — exactly, or with a
        # reworded-but-equivalent query for a tool that already returned a useful
        # result. Both are no-ops: record it (so its tokens count) but as a clear
        # "finishing" step, not as another tool call, and stop cleanly.
        is_repeat = (not decision.finished and decision.tool in tools_by_name
                     and (_key(decision.tool, decision.arguments) in seen_calls
                          or _is_redundant(decision.tool, decision.arguments)))
        trace.add(TraceStep(
            index=idx, kind="router", tool=None if is_repeat else decision.tool,
            arguments=None if is_repeat else decision.arguments,
            result=(f"already ran {decision.tool} — finishing" if is_repeat
                    else ("finish" if decision.finished else f"call {decision.tool}")),
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
            reasoning_tokens=decision.reasoning_tokens,
            model_id=decision.model_id,
        ))
        idx += 1

        if is_repeat:
            stop_reason = "loop-guard"
            break
        if decision.finished or not decision.tool:
            stop_reason = "done"
            break

        if decision.tool not in tools_by_name:
            trace.add(TraceStep(index=idx, kind="tool", tool=decision.tool,
                                arguments=decision.arguments,
                                result=f"Unknown tool: {decision.tool}"))
            idx += 1
            continue

        call_key = _key(decision.tool, decision.arguments)
        seen_calls.add(call_key)

        result, step = _run_tool(decision.tool, decision.arguments or {})
        trace.add(step)
        observations.append({"tool": decision.tool, "arguments": decision.arguments,
                             "result": result})
        _record_query(decision.tool, decision.arguments or {}, result)

        norm = _short(result, 400)
        if norm in seen_results:
            stop_reason = "loop-guard"  # no new information — stop and answer
            break
        seen_results.add(norm)

    # Always synthesize the final answer (the single conclusion) on the big model.
    syn = synthesize(question, history, observations)
    trace.add(TraceStep(index=idx, kind="synthesis", tool=None, arguments=None,
                        result=_short(syn.text), prompt_tokens=syn.prompt_tokens,
                        completion_tokens=syn.completion_tokens,
                        reasoning_tokens=syn.reasoning_tokens,
                        elapsed_seconds=syn.elapsed_seconds, model_id=syn.model_id))
    trace.stop_reason = stop_reason
    trace.elapsed_seconds = round(clock() - start, 3)
    return AgentResult(answer=syn.text.strip(), trace=trace)
