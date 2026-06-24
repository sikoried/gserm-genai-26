"""Query-rewrite tool: expand a short query so its embedding is more characteristic."""
from __future__ import annotations

from smolagents import tool

from . import runtime
from ..llm import chat

_SYSTEM = (
    "Rewrite the user's search query into a single, more descriptive sentence that "
    "captures its intent for semantic retrieval. Return only the rewritten query."
)


@tool
def query_rewrite(query: str) -> str:
    """Rewrite a short query into a fuller, more descriptive search sentence.

    Use this when a query is terse so its embedding becomes more characteristic for
    retrieval, and to rephrase before retrying a search that returned nothing.

    Args:
        query: The short query to expand.
    """
    client, model = runtime.llm()
    return chat(
        client, model,
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": query}],
        temperature=0.0,
    ).strip()
