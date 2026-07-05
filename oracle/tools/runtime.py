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

# Token tally for the tool call currently in flight. Reasoning is a subset of
# completion; model is the id of the LLM a tool used (None for local tools).
_tool_tokens = {"prompt": 0, "completion": 0, "reasoning": 0, "model": None}


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
    _tool_tokens.update(prompt=0, completion=0, reasoning=0, model=None)


def add_tool_tokens(prompt_tokens: int, completion_tokens: int,
                    reasoning_tokens: int = 0, model_id: str | None = None) -> None:
    _tool_tokens["prompt"] += int(prompt_tokens or 0)
    _tool_tokens["completion"] += int(completion_tokens or 0)
    _tool_tokens["reasoning"] += int(reasoning_tokens or 0)
    if model_id:
        _tool_tokens["model"] = model_id


def get_tool_tokens() -> tuple[int, int, int]:
    """(prompt, completion, reasoning) for the tool call just executed."""
    return _tool_tokens["prompt"], _tool_tokens["completion"], _tool_tokens["reasoning"]


def get_tool_model() -> str | None:
    return _tool_tokens["model"]
