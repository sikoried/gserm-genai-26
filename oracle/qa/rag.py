"""`rag` QA system: retrieve from the local wiki-10k index, then answer with context.

Deliberately *not* agentic: a single retrieval per conversation (on the first user
message), the message is embedded directly without reformulation, and the retrieved
chunks are rendered into the prompt via a Jinja2 template.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .base import Answer, QASystem
from ..config import QAConfig
from ..llm import chat_with_metrics, make_client
from ..models import reasoning_request_kwargs
from ..retrieval import Embedder, Hit, load_retriever

DEFAULT_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Use only the context provided below to "
    "answer the question. If the answer is not contained in the context, say so briefly "
    "rather than guessing."
)

_env = Environment(
    loader=FileSystemLoader(str(Path(__file__).resolve().parent)),
    autoescape=False, trim_blocks=True, lstrip_blocks=True,
)
_context_template = _env.get_template("rag_prompt.j2")


def _first_user_message(history: list[dict]) -> str:
    return next((m.get("content", "") for m in history if m.get("role") == "user"), "")


class RagQA(QASystem):
    # Index + embedder are heavy; share them across instances in the process.
    _retriever = None
    _embedder = None

    def __init__(self, config: QAConfig):
        super().__init__(config)
        self.client = make_client(config.endpoint)
        self.system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.top_k = config.top_k
        if RagQA._retriever is None:
            RagQA._retriever = load_retriever()
        if RagQA._embedder is None:
            RagQA._embedder = Embedder(config.embedding_model)

    def retrieve(self, query: str) -> list[Hit]:
        # Embed the message directly (no reformulation), then cosine-search.
        embedding = RagQA._embedder.encode([query])[0]
        return RagQA._retriever.search(embedding, self.top_k)

    def _system_message(self, query: str) -> dict:
        context = _context_template.render(hits=self.retrieve(query))
        return {"role": "system", "content": f"{self.system_prompt}\n\nContext:\n{context}"}

    def _complete(self, messages: list[dict]) -> Answer:
        content, metrics = chat_with_metrics(
            self.client, self.config.model, messages,
            temperature=self.config.temperature,
            **reasoning_request_kwargs(self.config.model, self.config.reasoning_effort),
        )
        return Answer(content=content.strip(), metrics=metrics)

    def answer(self, question: str) -> Answer:
        return self._complete(
            [self._system_message(question), {"role": "user", "content": question}]
        )

    def answer_chat(self, history: list[dict]) -> Answer:
        """Multi-turn: retrieve only on the first user message; later turns extend context."""
        return self._complete([self._system_message(_first_user_message(history)), *history])
