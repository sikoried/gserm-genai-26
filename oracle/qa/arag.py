"""`a-rag` (agentic RAG): a smolagents agent that decides whether to retrieve.

The agent first reasons whether a retrieval step is needed; when it is, it searches
the local wiki index (optionally rewriting a short query first) and, if a search comes
back empty, rephrases once and retries — never more than twice. The general prompt
structure is reused from the basic ``rag`` system.
"""
from __future__ import annotations

from smolagents import OpenAIServerModel, ToolCallingAgent

from .base import Answer, QASystem
from .rag import DEFAULT_SYSTEM_PROMPT
from ..config import QAConfig
from ..llm import UsageMetrics, load_token, make_client
from ..retrieval import Embedder, load_retriever
from .. import tools

AGENT_INSTRUCTIONS = (
    f"{DEFAULT_SYSTEM_PROMPT}\n\n"
    "You have a `search` tool over a local wiki knowledge base and a `query_rewrite` tool.\n"
    "First reason whether retrieval is needed: if the latest message can be answered from "
    "the conversation so far, answer directly without searching.\n"
    "When you do need information, call `search` with a descriptive sentence; if the query "
    "is short, call `query_rewrite` first so its embedding is more characteristic.\n"
    "If a search returns no results, rephrase the query once with `query_rewrite` and search "
    "again — never search more than twice."
)


def _message_text(content) -> str:
    """Flatten string- or list-of-parts message content to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type", "text") == "text")
    return "" if content is None else str(content)


def _sanitize_messages(messages):
    """Mistral rejects assistant messages with empty content and no tool calls.

    smolagents can emit such a (model-produced) empty step mid-loop and re-send it,
    crashing the run. The role may be a ``MessageRole`` enum and the content a list of
    parts (e.g. ``[{"type": "text", "text": ""}]``), so check both shapes and give an
    empty assistant message a placeholder so the agent loop survives.
    """
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
    """A compact, human-readable trace of the agent's tool use (its reasoning)."""
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


class AgenticRagQA(QASystem):
    # Index + embedder are heavy; share them across instances in the process.
    _retriever = None
    _embedder = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        if AgenticRagQA._retriever is None:
            AgenticRagQA._retriever = load_retriever()
        if AgenticRagQA._embedder is None:
            AgenticRagQA._embedder = Embedder(config.embedding_model)
        tools.configure(AgenticRagQA._retriever, AgenticRagQA._embedder,
                        make_client(config.endpoint), config.model)
        self.model = OpenAIServerModel(
            model_id=config.model, api_base=config.endpoint, api_key=load_token(),
            temperature=config.temperature, max_tokens=2048,
            # Bound the per-call wait and generation length: smolagents' default client
            # allows a 600s read with 2 retries (~30 min), so a stalled or runaway-reasoning
            # Mistral call reads as a hang in the GUI.
            client_kwargs={"timeout": 90.0, "max_retries": 1},
        )
        self._patch_model_client()
        self.agent = ToolCallingAgent(
            tools=[tools.query_rewrite, tools.search], model=self.model,
            instructions=AGENT_INSTRUCTIONS, max_steps=6,
        )

    def _patch_model_client(self) -> None:
        """Sanitize outgoing messages so an empty assistant turn can't crash the run."""
        create = self.model.client.chat.completions.create

        def patched(*args, **kwargs):
            if "messages" in kwargs:
                kwargs["messages"] = _sanitize_messages(kwargs["messages"])
            return create(*args, **kwargs)

        self.model.client.chat.completions.create = patched

    def _run(self, task: str) -> Answer:
        try:
            result = self.agent.run(task, reset=True, return_full_result=True)
        except Exception as exc:
            raise RuntimeError(
                f"Agentic RAG did not complete — the model stalled or errored: {exc}"
            ) from exc
        usage, timing = result.token_usage, result.timing
        elapsed = 0.0
        if timing and timing.start_time and timing.end_time:
            elapsed = round(timing.end_time - timing.start_time, 3)
        metrics = UsageMetrics(
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            elapsed_seconds=elapsed,
        )
        # Use memory.steps (TaskStep/ActionStep objects) — result.steps are plain dicts.
        return Answer(content=str(result.output).strip(), metrics=metrics,
                      reasoning=_build_trace(self.agent.memory.steps))

    def answer(self, question: str) -> Answer:
        return self._run(question)

    def answer_chat(self, history: list[dict]) -> Answer:
        convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history)
        return self._run(
            f"Conversation so far:\n{convo}\n\nReason whether new retrieval is needed, "
            "then answer the latest user message."
        )
