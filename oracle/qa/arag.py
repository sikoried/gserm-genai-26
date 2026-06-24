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
        self.model = OpenAIServerModel(model_id=config.model, api_base=config.endpoint,
                                       api_key=load_token(), temperature=config.temperature)
        self.agent = ToolCallingAgent(
            tools=[tools.query_rewrite, tools.search], model=self.model,
            instructions=AGENT_INSTRUCTIONS, max_steps=6,
        )

    def _run(self, task: str) -> Answer:
        result = self.agent.run(task, reset=True, return_full_result=True)
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
        return Answer(content=str(result.output).strip(), metrics=metrics)

    def answer(self, question: str) -> Answer:
        return self._run(question)

    def answer_chat(self, history: list[dict]) -> Answer:
        convo = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history)
        return self._run(
            f"Conversation so far:\n{convo}\n\nReason whether new retrieval is needed, "
            "then answer the latest user message."
        )
