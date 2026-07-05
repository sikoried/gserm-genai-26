"""`a-rag` (agentic RAG): a local tool router drives tools, the proxy synthesizes.

Division of labour (see ``tools.md``):

- a small **local** LLM (``agent.LocalRouter``, default Qwen2.5-1.5B) decides, per
  step, which tool to call — search, query-rewrite, calculator, date, unit
  convert, wiki-lookup, list-pick — to answer bar-quiz questions;
- the bounded, always-terminating ``agent.run_agent`` loop enforces the step
  budget, loop guards and timeout;
- the large proxy model is invoked once at the end to **synthesize** the final
  answer from the evidence the tools gathered.

The smolagents tools live in ``oracle.tools``. The run produces a structured
``AgentTrace`` (per-step tool calls + token accounting + stop reason) carried on
the ``Answer``.
"""
from __future__ import annotations

from .base import Answer, QASystem
from .rag import DEFAULT_SYSTEM_PROMPT
from ..config import QAConfig
from ..llm import UsageMetrics, chat_with_metrics, make_client
from ..models import reasoning_request_kwargs
from ..retrieval import Embedder, load_retriever
from .. import tools
from ..agent import (
    LocalPlanner, LocalRouter, ProxyPlanner, ProxyVerifier, SynthesisResult,
    run_agent, run_multi_hop,
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are answering a bar-quiz question. Use the evidence gathered by the tools "
    "below, plus your own knowledge, to give a SHORT, exact answer (a name, number, "
    "year, or phrase) — not an essay. If the evidence is empty or unhelpful, answer "
    "from your own knowledge; if you genuinely cannot determine it, say so briefly. "
    "If a tool result says the web or YouTube lookup was unavailable (e.g. 'No "
    "internet connection'), briefly tell the user that the online lookup could not be "
    "performed, then answer from your own knowledge as best you can."
)


# --- message helpers (retained: sanitize an empty assistant turn if a proxy model
# ever drives the loop; also covered by tests/test_arag.py) ---------------------

def _message_text(content) -> str:
    """Flatten string- or list-of-parts message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type", "text") == "text")
    return "" if content is None else str(content)


def _sanitize_messages(messages):
    """Give an empty assistant message a placeholder (Mistral rejects empty turns)."""
    for m in messages or []:
        if not isinstance(m, dict) or "assistant" not in str(m.get("role", "")).lower():
            continue
        if m.get("tool_calls") or _message_text(m.get("content")).strip():
            continue
        content = m.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            content[0]["text"] = "(no output)"  # keep the list-of-parts shape
        else:
            m["content"] = "(no output)"
    return messages


def _fmt_args(args) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v!r}" for k, v in args.items())
    return str(args)


def _build_trace(steps) -> str:
    """A compact, human-readable trace of tool use (legacy smolagents-step form)."""
    lines: list[str] = []
    for step in steps:
        if type(step).__name__ != "ActionStep":
            continue
        calls = [tc for tc in (getattr(step, "tool_calls", None) or [])
                 if getattr(tc, "name", "") != "final_answer"]
        for tc in calls:
            lines.append(f"• {tc.name}({_fmt_args(getattr(tc, 'arguments', None))})")
        obs = getattr(step, "observations", None)
        if calls and obs:
            text = " ".join(str(obs).split())
            lines.append(f"    → {text[:200]}{'…' if len(text) > 200 else ''}")
    return "\n".join(lines) if lines else "Answered directly — no retrieval was needed."


def _last_user_question(history: list[dict]) -> str:
    return next((m.get("content", "") for m in reversed(history)
                 if m.get("role") == "user"), "")


class AgenticRagQA(QASystem):
    # Index and embedder are heavy; share them across instances (router weights are
    # shared via oracle.agent.router's model cache).
    _retriever = None
    _embedder = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        if AgenticRagQA._retriever is None:
            AgenticRagQA._retriever = load_retriever()
        if AgenticRagQA._embedder is None:
            AgenticRagQA._embedder = Embedder(config.embedding_model)
        self.client = make_client(config.endpoint)
        tools.configure(AgenticRagQA._retriever, AgenticRagQA._embedder,
                        self.client, config.model)
        # Compose the active toolset (online tools only when opted in).
        self.active_tools = tools.active_tools(config.enable_online_tools)
        self.tools_by_name = tools.tools_by_name(self.active_tools)
        self.router = LocalRouter(
            config.router_model, tools.tool_metas(self.active_tools),
            device=config.router_device, max_gb=config.router_max_gb,
        )
        self.planner = None
        if config.multi_hop:
            if config.planner_model == "answer":
                # Delegate planning to the big model chosen in 'Model' — better on hard
                # questions than the small router.
                self.planner = ProxyPlanner(
                    chat_with_metrics, self.client, config.model,
                    max_hops=config.max_hops, temperature=config.temperature,
                )
            else:
                self.planner = LocalPlanner(config.router_model, self.router.generate,
                                            max_hops=config.max_hops)
        self.verifier = None
        if config.verify:
            self.verifier = ProxyVerifier(chat_with_metrics, self.client, config.model,
                                          temperature=config.temperature)

    def _synthesize(self, question: str, history: list[dict],
                    observations: list[dict]) -> SynthesisResult:
        context = "\n\n".join(f"[{o['tool']}] {o['result']}" for o in observations) \
            or "(no tool results)"
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            *history,
            {"role": "user",
             "content": f"{question}\n\nEvidence gathered by tools:\n{context}"},
        ]
        content, metrics = chat_with_metrics(
            self.client, self.config.model, messages,
            temperature=self.config.temperature,
            **reasoning_request_kwargs(self.config.model, self.config.reasoning_effort),
        )
        return SynthesisResult(
            text=content.strip(), prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            elapsed_seconds=metrics.elapsed_seconds, model_id=self.config.model,
        )

    def _run_hop(self, question: str, history: list[dict]):
        return run_agent(
            question=question, history=history, router=self.router,
            tools_by_name=self.tools_by_name, synthesize=self._synthesize,
            reset_tool_tokens=tools.reset_tool_tokens,
            get_tool_tokens=tools.get_tool_tokens, get_tool_model=tools.get_tool_model,
            max_steps=self.config.max_steps,
            timeout_seconds=self.config.agent_timeout_seconds,
            rag_first=self.config.rag_first, call_cache=self._call_cache,
        )

    def _run(self, question: str, history: list[dict]) -> Answer:
        # One result cache per question, shared across all hops / recursion.
        self._call_cache: dict = {}
        if self.planner is not None:
            result = run_multi_hop(
                question=question, history=history, planner=self.planner,
                run_hop=self._run_hop, synthesize_final=self._synthesize,
                verifier=self.verifier, verify=self.config.verify,
                max_hops=self.config.max_hops, max_depth=self.config.max_depth,
            )
        else:
            result = self._run_hop(question, history)
        totals = result.trace.totals()
        metrics = UsageMetrics(
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            total_tokens=totals.total_tokens,
            elapsed_seconds=result.trace.elapsed_seconds,
            reasoning_tokens=totals.reasoning_tokens,
        )
        return Answer(content=result.answer, metrics=metrics,
                      reasoning=result.trace.as_text(), trace=result.trace)

    def answer(self, question: str) -> Answer:
        return self._run(question, [])

    def answer_chat(self, history: list[dict]) -> Answer:
        """Multi-turn: the last user message is the question, prior turns are context."""
        question = _last_user_question(history)
        prior = history[:-1] if history and history[-1].get("role") == "user" else history
        return self._run(question, prior)
