"""`world` QA system: answer from the model's own world knowledge (no retrieval)."""
from __future__ import annotations

from .base import QASystem
from ..config import QAConfig
from ..llm import chat, make_client

DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable question-answering assistant. "
    "Answer the question directly and concisely from your own knowledge. "
    "If you do not know, say so briefly rather than guessing."
)


class WorldQA(QASystem):
    def __init__(self, config: QAConfig):
        super().__init__(config)
        self.client = make_client(config.endpoint)
        self.system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT

    def answer(self, question: str) -> str:
        return chat(
            self.client,
            self.config.model,
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=self.config.temperature,
        ).strip()
