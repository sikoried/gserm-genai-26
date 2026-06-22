"""Base interface shared by all QA systems."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import QAConfig


class QASystem(ABC):
    """A question-answering system: maps a question to an answer string."""

    def __init__(self, config: QAConfig):
        self.config = config

    @abstractmethod
    def answer(self, question: str) -> str:
        ...
