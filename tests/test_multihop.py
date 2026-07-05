"""Tests for multi-hop planning: plan → per-hop loop → compose, all network-free."""
from oracle.agent.loop import AgentResult, SynthesisResult, run_agent
from oracle.agent.multihop import run_multi_hop
from oracle.agent.planner import Planner, PlanResult, SingleHopPlanner, parse_plan
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


def test_multi_hop_token_totals_include_planner():
    def run_hop(subq, history):
        return run_agent(question=subq, history=history, router=_finish_router(),
                         tools_by_name={}, synthesize=_synth("x"))

    res = run_multi_hop(question="q", history=[], planner=FixedPlanner(["a", "b"], p=6, c=3),
                        run_hop=run_hop, synthesize_final=_synth("F"), max_hops=3)
    totals = res.trace.totals()
    assert totals.planner.input == 6 and totals.planner.output == 3
    assert totals.total_tokens > 0
