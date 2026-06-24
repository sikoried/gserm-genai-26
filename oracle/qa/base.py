"""Base interface shared by all QA systems."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import QAConfig
from ..llm import UsageMetrics


@dataclass
class Answer:
    """The result of a QA system call: the answer text plus usage metrics."""
    content: str
    metrics: UsageMetrics
    reasoning: str | None = None  # agent reasoning trace (agentic RAG); None otherwise


class QASystem(ABC):
    """A question-answering system: maps a question to an Answer."""

    def __init__(self, config: QAConfig):
        self.config = config

    @abstractmethod
    def answer(self, question: str) -> Answer:
        ...
