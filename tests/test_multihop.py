"""Tests for multi-hop planning: plan → per-hop loop → compose, all network-free."""
from oracle.agent.loop import AgentResult, SynthesisResult, run_agent
from oracle.agent.multihop import run_multi_hop
from oracle.agent.planner import (
    LocalPlanner, Planner, PlanResult, ProxyPlanner, SingleHopPlanner, parse_plan,
)
from oracle.agent.router import Router, RouterDecision
from oracle.agent.trace import AgentTrace, TraceStep


class FixedPlanner(Planner):
    def __init__(self, subs, p=6, c=3):
        self.subs, self.p, self.c = subs, p, c

    def plan(self, question, history):
        return PlanResult(subquestions=list(self.subs), prompt_tokens=self.p,
                          completion_tokens=self.c, model_id="router/m")


def _finish_router():
    class R(Router):
        def decide(self, q, h, o):
            return RouterDecision(finished=True, prompt_tokens=4, completion_tokens=1,
                                  model_id="router/m")
    return R()


def _synth(text):
    def s(q, h, o):
        return SynthesisResult(text=text, prompt_tokens=10, completion_tokens=3,
                               model_id="big/synth")
    return s


# --- parse_plan ---------------------------------------------------------------

def test_parse_plan_reads_json_array_and_bounds():
    assert parse_plan('["a", "b", "c", "d"]', "q", max_hops=2) == ["a", "b"]


def test_parse_plan_falls_back_to_single_hop():
    assert parse_plan("not json", "the question", max_hops=3) == ["the question"]


# --- planner model choice -----------------------------------------------------

def test_local_planner_uses_the_router_model():
    calls = {}

    def generate(messages):
        calls["messages"] = messages
        return '["a", "b"]', 12, 4

    p = LocalPlanner("qwen/router", generate, max_hops=3)
    res = p.plan("q", [])
    assert res.subquestions == ["a", "b"]
    assert res.model_id == "qwen/router" and res.prompt_tokens == 12


def test_proxy_planner_uses_the_big_answer_model():
    from oracle.llm import UsageMetrics
    seen = {}

    def fake_chat(client, model, messages, temperature=0.0):
        seen["model"] = model
        return '["step 1", "step 2", "step 3"]', UsageMetrics(30, 9, 39, 0.2, reasoning_tokens=3)

    p = ProxyPlanner(fake_chat, client="C", model_id="mistral/Big", max_hops=3)
    res = p.plan("hard q", [])
    assert res.subquestions == ["step 1", "step 2", "step 3"]
    assert res.model_id == "mistral/Big" and seen["model"] == "mistral/Big"
    assert res.prompt_tokens == 30 and res.reasoning_tokens == 3


# --- multi-hop orchestration --------------------------------------------------

def test_multi_hop_runs_each_hop_and_composes():
    hops_seen = []

    def run_hop(subq, history):
        hops_seen.append(subq)
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth(f"answer:{subq}"))

    res = run_multi_hop(question="deep q", history=[],
                        planner=FixedPlanner(["hop A", "hop B"]),
                        run_hop=run_hop, synthesize_final=_synth("FINAL"), max_hops=3)
    assert hops_seen == ["hop A", "hop B"]
    assert res.answer == "FINAL"
    kinds = [s.kind for s in res.trace.steps]
    assert kinds[0] == "planner"          # planning first
    assert kinds[-1] == "synthesis"       # final compose last
    # hop tagging present
    assert any(s.hop == 0 for s in res.trace.steps)
    assert any(s.hop == 1 for s in res.trace.steps)


def test_multi_hop_threads_prior_answers_as_facts():
    seen_histories = []

    def run_hop(subq, history):
        seen_histories.append(history)
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth(f"A[{subq}]"))

    run_multi_hop(question="q", history=[], planner=FixedPlanner(["first", "second"]),
                  run_hop=run_hop, synthesize_final=_synth("F"), max_hops=3)
    # First hop sees empty history; the second hop sees the first hop's fact.
    assert seen_histories[0] == []
    assert any("Known facts" in m.get("content", "") and "A[first]" in m.get("content", "")
               for m in seen_histories[1])


def test_planner_step_recorded_even_for_atomic_plan():
    # A single-item plan must still show a planner step, so the user can see that
    # planning happened (and was decided atomic) — the "no planning" fix.
    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("A"))

    res = run_multi_hop(question="q", history=[], planner=FixedPlanner(["q"], p=8, c=2),
                        run_hop=run_hop, synthesize_final=_synth("F"), max_hops=3)
    planner_steps = [s for s in res.trace.steps if s.kind == "planner"]
    assert len(planner_steps) == 1
    assert "atomic" in planner_steps[0].result
    assert res.trace.totals().planner.total == 10  # planner tokens still counted


def test_single_hop_plan_skips_final_synthesis():
    # A one-item plan reproduces plain behaviour: the hop answer is the final answer.
    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("only"))

    res = run_multi_hop(question="atomic", history=[],
                        planner=SingleHopPlanner(), run_hop=run_hop,
                        synthesize_final=_synth("SHOULD-NOT-BE-USED"), max_hops=3)
    assert res.answer == "only"
    # exactly one synthesis (the hop's), plus the planner step — no extra compose
    assert sum(s.kind == "synthesis" for s in res.trace.steps) == 1


def test_recursive_sub_agents_nest_by_depth():
    # Depth-2: the top plan yields 2 subs, and each sub is itself planned into 2.
    class DepthPlanner(Planner):
        def plan(self, question, history):
            # Top question -> two hard subs; each of those -> two atomic leaves.
            if question == "top":
                return PlanResult(["subA", "subB"], prompt_tokens=6, completion_tokens=2,
                                  model_id="router/m")
            return PlanResult([f"{question}-1", f"{question}-2"], prompt_tokens=4,
                              completion_tokens=1, model_id="router/m")

    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth(f"A[{subq}]"))

    res = run_multi_hop(question="top", history=[], planner=DepthPlanner(),
                        run_hop=run_hop, synthesize_final=_synth("FINAL"),
                        max_hops=3, max_depth=2)
    depths = {s.depth for s in res.trace.steps}
    assert 0 in depths and 1 in depths and 2 in depths  # nesting recorded
    # A planner step exists at depth 0 and at depth 1 (the recursion planned again).
    planner_depths = {s.depth for s in res.trace.steps if s.kind == "planner"}
    assert planner_depths == {0, 1}
    assert res.answer == "FINAL"


def test_max_depth_one_is_flat_no_recursion():
    class DepthPlanner(Planner):
        def plan(self, question, history):
            return PlanResult(["a", "b"], prompt_tokens=1, completion_tokens=1, model_id="m")

    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("x"))

    res = run_multi_hop(question="q", history=[], planner=DepthPlanner(), run_hop=run_hop,
                        synthesize_final=_synth("F"), max_hops=3, max_depth=1)
    # Flat: planned exactly once (no nested sub-plan), so no depth-2 nesting.
    assert sum(s.kind == "planner" for s in res.trace.steps) == 1
    assert max(s.depth for s in res.trace.steps) <= 1


def test_verification_hop_backtracks_on_contradiction():
    from oracle.agent.verify import Verifier, VerifyResult

    class Backtracker(Verifier):
        def verify(self, question, answer, facts):
            return VerifyResult(ok=False, corrected_answer="CORRECTED",
                                note="contradiction", prompt_tokens=5, completion_tokens=2,
                                model_id="big/verify")

    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("wrong"))

    res = run_multi_hop(question="q", history=[], planner=SingleHopPlanner(), run_hop=run_hop,
                        synthesize_final=_synth("wrong"), verifier=Backtracker(), verify=True)
    assert res.answer == "CORRECTED"
    verify_steps = [s for s in res.trace.steps if s.kind == "verify"]
    assert len(verify_steps) == 1  # bounded: runs once
    assert res.trace.totals().verify.total == 7


def test_verification_hop_keeps_answer_when_ok():
    from oracle.agent.verify import Verifier, VerifyResult

    class Passer(Verifier):
        def verify(self, question, answer, facts):
            return VerifyResult(ok=True, note="verified", prompt_tokens=3, completion_tokens=1)

    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("kept"))

    res = run_multi_hop(question="q", history=[], planner=SingleHopPlanner(), run_hop=run_hop,
                        synthesize_final=_synth("kept"), verifier=Passer(), verify=True)
    assert res.answer == "kept"
    assert sum(s.kind == "verify" for s in res.trace.steps) == 1


def test_multi_hop_token_totals_include_planner():
    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("x"))

    res = run_multi_hop(question="q", history=[], planner=FixedPlanner(["a", "b"], p=6, c=3),
                        run_hop=run_hop, synthesize_final=_synth("F"), max_hops=3)
    totals = res.trace.totals()
    assert totals.planner.input == 6 and totals.planner.output == 3
    assert totals.total_tokens > 0
