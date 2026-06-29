"""Shared runtime the agentic RAG system wires up for the agent's tools.

The tool functions are module-level, so the retriever / embedder / LLM client
they need are held here and configured per process by ``AgenticRagQA``.

It also keeps a small **per-tool-call token tally**: a tool that calls the LLM
(e.g. ``query_rewrite``) records its usage here, and the orchestrator reads it
right after the call to attribute those tokens to that step of the trace. Local
tools never touch it, so they report 0 tokens.
"""
from __future__ import annotations

_retriever = None
_embedder = None
_client = None
_model = None

# Token tally for the tool call currently in flight (prompt, completion).
_tool_tokens = {"prompt": 0, "completion": 0}


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


# --- per-tool-call token accounting -------------------------------------------

def reset_tool_tokens() -> None:
    _tool_tokens["prompt"] = 0
    _tool_tokens["completion"] = 0


def add_tool_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    _tool_tokens["prompt"] += int(prompt_tokens or 0)
    _tool_tokens["completion"] += int(completion_tokens or 0)


def get_tool_tokens() -> tuple[int, int]:
    return _tool_tokens["prompt"], _tool_tokens["completion"]
