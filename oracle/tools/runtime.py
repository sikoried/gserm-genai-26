"""Shared runtime the agentic RAG system wires up for the agent's tools.

smolagents ``@tool`` functions are module-level, so the retriever / embedder / LLM
client they need are held here and configured per process by ``AgenticRagQA``.
"""
from __future__ import annotations

_retriever = None
_embedder = None
_client = None
_model = None


def configure(retriever, embedder, client, model) -> None:
    global _retriever, _embedder, _client, _model
    _retriever, _embedder, _client, _model = retriever, embedder, client, model


def retriever():
    if _retriever is None:
        raise RuntimeError("tools runtime not configured (build the agentic RAG system first)")
    return _retriever


def embedder():
    return _embedder


def llm():
    return _client, _model
