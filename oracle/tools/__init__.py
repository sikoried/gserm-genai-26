"""Tools for the agentic RAG system.

Each tool's logic lives in its own module as a plain, unit-testable function;
here we wrap them as smolagents ``Tool`` objects (which expose name / description
/ input schema to the router and are callable by the orchestrator).

``ALL_TOOLS`` / ``TOOLS_BY_NAME`` are the registry the agent loads; ``tool_metas``
renders the schema the local router sees when choosing a tool.
"""
from __future__ import annotations

from smolagents import tool as _tool

from .runtime import (
    configure, retriever, embedder, llm,
    reset_tool_tokens, add_tool_tokens, get_tool_tokens,
)
from .calculator import calculator as _calculator
from .convert import unit_convert as _unit_convert
from .datetool import date_tool as _date_tool
from .pick import list_pick as _list_pick
from .rewrite import query_rewrite as _query_rewrite
from .search import search as _search
from .wiki_lookup import wiki_lookup as _wiki_lookup

# smolagents Tool objects (the agent/router use these).
search = _tool(_search)
query_rewrite = _tool(_query_rewrite)
calculator = _tool(_calculator)
date_tool = _tool(_date_tool)
unit_convert = _tool(_unit_convert)
wiki_lookup = _tool(_wiki_lookup)
list_pick = _tool(_list_pick)

ALL_TOOLS = [
    search, query_rewrite, calculator, date_tool, unit_convert, wiki_lookup, list_pick,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def tool_metas(tools=None) -> list[dict]:
    """Render ``{name, description, inputs}`` for the router prompt."""
    return [
        {"name": t.name, "description": t.description, "inputs": t.inputs}
        for t in (tools or ALL_TOOLS)
    ]


__all__ = [
    "configure", "retriever", "embedder", "llm",
    "reset_tool_tokens", "add_tool_tokens", "get_tool_tokens",
    "search", "query_rewrite", "calculator", "date_tool", "unit_convert",
    "wiki_lookup", "list_pick",
    "ALL_TOOLS", "TOOLS_BY_NAME", "tool_metas",
]
