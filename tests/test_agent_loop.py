"""Tests for the agentic loop: termination guards + token accounting.

All network-free: the router and synthesis are fakes, tools are plain callables,
and the wall clock is injected. The real per-tool token tally
(``oracle.tools.runtime``) is exercised so the accounting path is genuine.
"""
import pytest

from oracle.agent.loop import run_agent, SynthesisResult
from oracle.agent.router import Router, RouterDecision
from oracle.tools import runtime


class ScriptedRouter(Router):
    """Yields a pre-scripted list of decisions (last one repeats if exhausted)."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.i = 0

    def decide(self, question, history, observations):
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        return d


def _call(tool, args, p=4, c=1):
    return RouterDecision(finished=False, tool=tool, arguments=args,
                          prompt_tokens=p, completion_tokens=c, model_id="router/m")


def _finish(p=4, c=1):
    return RouterDecision(finished=True, prompt_tokens=p, completion_tokens=c,
                          model_id="router/m")


def _synth(text="FINAL", p=11, c=4, r=0, elapsed=0.05):
    def synthesize(question, history, observations):
        return SynthesisResult(text=text, prompt_tokens=p, completion_tokens=c,
                               reasoning_tokens=r, elapsed_seconds=elapsed,
                               model_id="big/synth")
    return synthesize


def _make_clock(values):
    vals = list(values)

    def clock():
        return vals.pop(0) if len(vals) > 1 else vals[0]
    return clock


@pytest.fixture(autouse=True)
def _reset_runtime_tokens():
    runtime.reset_tool_tokens()
    yield


def _echo_tools():
    return {"echo": lambda value="": f"echo:{value}"}


# --- termination guarantees ---------------------------------------------------

def test_done_when_router_finishes_immediately():
    res = run_agent(question="q", history=[], router=ScriptedRouter([_finish()]),
                    tools_by_name=_echo_tools(), synthesize=_synth())
    assert res.answer == "FINAL"
    assert res.trace.stop_reason == "done"
    # No tools, exactly one synthesis step.
    assert [s.kind for s in res.trace.steps] == ["router", "synthesis"]


def test_rag_first_forces_a_local_search_before_the_router():
    calls = []

    def search(query=""):
        calls.append(query)
        return f"local hit for {query}"

    # Router finishes immediately; the forced RAG search must still have run first.
    res = run_agent(question="capital of France?", history=[],
                    router=ScriptedRouter([_finish()]),
                    tools_by_name={"search": search}, synthesize=_synth(),
                    rag_first=True)
    assert calls == ["capital of France?"]
    kinds = [s.kind for s in res.trace.steps]
    assert kinds[0] == "tool" and res.trace.steps[0].tool == "search"
    assert kinds == ["tool", "router", "synthesis"]


def test_rag_first_can_be_disabled():
    calls = []
    res = run_agent(question="q", history=[], router=ScriptedRouter([_finish()]),
                    tools_by_name={"search": lambda query="": calls.append(query) or "x"},
                    synthesize=_synth(), rag_first=False)
    assert calls == []  # no forced search
    assert [s.kind for s in res.trace.steps] == ["router", "synthesis"]


def test_rag_first_no_op_when_no_search_tool():
    # rag_first is harmless when the toolset has no `search` (e.g. minimal setups).
    res = run_agent(question="q", history=[], router=ScriptedRouter([_finish()]),
                    tools_by_name=_echo_tools(), synthesize=_synth(), rag_first=True)
    assert [s.kind for s in res.trace.steps] == ["router", "synthesis"]


def test_step_budget_forces_synthesis():
    router = ScriptedRouter([_call("echo", {"value": str(i)}) for i in range(10)])
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name=_echo_tools(), synthesize=_synth(), max_steps=2)
    assert res.trace.stop_reason == "step-budget"
    assert len([s for s in res.trace.steps if s.kind == "tool"]) == 2
    assert sum(s.kind == "synthesis" for s in res.trace.steps) == 1  # still concludes


def test_loop_guard_on_repeated_call():
    router = ScriptedRouter([_call("echo", {"value": "x"}),
                             _call("echo", {"value": "x"})])  # identical → guard
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name=_echo_tools(), synthesize=_synth(), max_steps=6)
    assert res.trace.stop_reason == "loop-guard"
    assert len([s for s in res.trace.steps if s.kind == "tool"]) == 1


def test_loop_guard_on_repeated_observation():
    tools = {"a": lambda: "SAME", "b": lambda: "SAME"}  # different calls, same result
    router = ScriptedRouter([_call("a", {}), _call("b", {})])
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name=tools, synthesize=_synth(), max_steps=6)
    assert res.trace.stop_reason == "loop-guard"


def test_timeout_forces_synthesis():
    router = ScriptedRouter([_call("echo", {"value": "1"})])
    res = run_agent(question="q", history=[], router=router, tools_by_name=_echo_tools(),
                    synthesize=_synth(), timeout_seconds=10,
                    clock=_make_clock([0, 100, 100, 100]))
    assert res.trace.stop_reason == "timeout"
    assert sum(s.kind == "synthesis" for s in res.trace.steps) == 1


def test_unknown_tool_does_not_crash():
    router = ScriptedRouter([_call("nope", {}), _finish()])
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name=_echo_tools(), synthesize=_synth())
    assert res.answer == "FINAL"
    assert any("Unknown tool" in s.result for s in res.trace.steps)


def test_tool_exception_is_captured_not_raised():
    def boom():
        raise ValueError("kaboom")
    router = ScriptedRouter([_call("boom", {}), _finish()])
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name={"boom": boom}, synthesize=_synth())
    assert any("Tool error: kaboom" in s.result for s in res.trace.steps)
    assert res.answer == "FINAL"


# --- token accounting ---------------------------------------------------------

def test_token_accounting_attributes_by_kind_and_type():
    # One LLM-backed tool records 7 input + 3 output (1 of it reasoning) via runtime.
    def llm_tool(query=""):
        runtime.add_tool_tokens(7, 3, reasoning_tokens=1, model_id="tool/m")
        return "rewritten"

    router = ScriptedRouter([_call("rw", {"query": "x"}, p=5, c=2), _finish(p=5, c=2)])
    res = run_agent(question="q", history=[], router=router,
                    tools_by_name={"rw": llm_tool}, synthesize=_synth(p=11, c=4, r=2),
                    reset_tool_tokens=runtime.reset_tool_tokens,
                    get_tool_tokens=runtime.get_tool_tokens,
                    get_tool_model=runtime.get_tool_model)
    totals = res.trace.totals()
    # router: two decisions × (5 in, 2 out, 0 reasoning)
    assert totals.router.input == 10 and totals.router.output == 4 and totals.router.reasoning == 0
    # tools: the single rewrite call (output = completion - reasoning = 3 - 1 = 2)
    assert totals.tools.input == 7 and totals.tools.output == 2 and totals.tools.reasoning == 1
    # synthesis: 11 in, completion 4 of which 2 reasoning -> output 2
    assert totals.synthesis.input == 11 and totals.synthesis.output == 2 and totals.synthesis.reasoning == 2
    # grand totals by type
    assert totals.input_tokens == 10 + 7 + 11
    assert totals.reasoning_tokens == 0 + 1 + 2
    assert totals.total_tokens == totals.input_tokens + totals.output_tokens + totals.reasoning_tokens
    # model names surfaced
    assert res.trace.model_names() == {"router": "router/m", "synthesis": "big/synth"}


def test_local_tools_report_zero_tokens():
    router = ScriptedRouter([_call("echo", {"value": "1"}), _finish()])
    res = run_agent(question="q", history=[], router=router, tools_by_name=_echo_tools(),
                    synthesize=_synth(),
                    reset_tool_tokens=runtime.reset_tool_tokens,
                    get_tool_tokens=runtime.get_tool_tokens)
    tool_steps = [s for s in res.trace.steps if s.kind == "tool"]
    assert tool_steps and all(s.total_tokens == 0 for s in tool_steps)


def test_trace_serializes_to_dict():
    res = run_agent(question="q", history=[], router=ScriptedRouter([_finish()]),
                    tools_by_name=_echo_tools(), synthesize=_synth())
    d = res.trace.to_dict()
    assert d["stop_reason"] == "done"
    assert "totals" in d and "steps" in d
    assert d["totals"]["total_tokens"] == d["totals"]["prompt_tokens"] + d["totals"]["completion_tokens"]
