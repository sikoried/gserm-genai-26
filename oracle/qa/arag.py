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
from ..agent import LocalRouter, run_agent

SYNTHESIS_SYSTEM_PROMPT = (
    "You are answering a bar-quiz question. Use the evidence gathered by the tools "
    "below, plus your own knowledge, to give a SHORT, exact answer (a name, number, "
    "year, or phrase) — not an essay. If the evidence is empty or unhelpful, answer "
    "from your own knowledge; if you genuinely cannot determine it, say so briefly."
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
    # Index, embedder, and router are heavy; share them across instances.
    _retriever = None
    _embedder = None
    _router = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        if AgenticRagQA._retriever is None:
            AgenticRagQA._retriever = load_retriever()
        if AgenticRagQA._embedder is None:
            AgenticRagQA._embedder = Embedder(config.embedding_model)
        self.client = make_client(config.endpoint)
        tools.configure(AgenticRagQA._retriever, AgenticRagQA._embedder,
                        self.client, config.model)
        if AgenticRagQA._router is None:
            AgenticRagQA._router = LocalRouter(
                config.router_model, tools.tool_metas(),
                device=config.router_device, max_gb=config.router_max_gb,
            )

    def _synthesize(self, question: str, history: list[dict],
                    observations: list[dict]):
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
        return (content.strip(), metrics.prompt_tokens,
                metrics.completion_tokens, metrics.elapsed_seconds)

    def _run(self, question: str, history: list[dict]) -> Answer:
        result = run_agent(
            question=question, history=history, router=AgenticRagQA._router,
            tools_by_name=tools.TOOLS_BY_NAME, synthesize=self._synthesize,
            reset_tool_tokens=tools.reset_tool_tokens,
            get_tool_tokens=tools.get_tool_tokens,
            max_steps=self.config.max_steps,
            timeout_seconds=self.config.agent_timeout_seconds,
        )
        totals = result.trace.totals()
        metrics = UsageMetrics(
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            total_tokens=totals.total_tokens,
            elapsed_seconds=result.trace.elapsed_seconds,
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
