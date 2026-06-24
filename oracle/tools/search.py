"""Search tool: embed a query and retrieve passages from the local wiki index."""
from __future__ import annotations

from smolagents import tool

from . import runtime


@tool
def search(query: str, k: int = 10, min_similarity: float = 0.0) -> str:
    """Search the local wiki knowledge base for passages relevant to a query.

    The query sentence is embedded and compared against the indexed passages. The
    result may be empty if nothing reaches the similarity threshold.

    Args:
        query: A sentence describing what to look for.
        k: Maximum number of passages to return.
        min_similarity: Minimum cosine similarity (0-1) a passage must reach to be returned.
    """
    embedding = runtime.embedder().encode([query])[0]
    hits = [h for h in runtime.retriever().search(embedding, k) if h.score >= min_similarity]
    if not hits:
        return "No results found."
    return "\n\n".join(f"[{i}] {h.title}\n{h.text}" for i, h in enumerate(hits, 1))
