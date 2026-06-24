"""QA systems and a factory that builds one from a config."""
from __future__ import annotations

from ..config import QAConfig
from .base import QASystem
from .rag import RagQA
from .world import WorldQA


def build_qa_system(config: QAConfig) -> QASystem:
    if config.type == "world":
        return WorldQA(config)
    if config.type == "rag":
        return RagQA(config)
    if config.type == "a-rag":
        from .arag import AgenticRagQA  # lazy: pulls smolagents only when used
        return AgenticRagQA(config)
    raise NotImplementedError(f"QA type {config.type!r} is not implemented yet")


__all__ = ["QASystem", "WorldQA", "RagQA", "build_qa_system"]
