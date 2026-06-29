"""Wiki-lookup tool: fetch a specific article by title from the local index.

Unlike `search` (fuzzy semantic retrieval), this matches the chunk *title* so
"what is the capital of France" style questions can pull the France article
directly. Offline (local index, 0 tokens).
"""
from __future__ import annotations

from . import runtime

_MAX_CHARS = 1500


def lookup(title: str, chunks, limit: int = 3) -> str:
    """Return text of indexed chunks whose title matches `title` (pure helper).

    Prefers an exact (case-insensitive) title match; falls back to a substring
    match. `chunks` is any iterable of objects with `.title` and `.text`.
    """
    needle = (title or "").strip().lower()
    if not needle:
        return "No title given."
    exact = [c for c in chunks if c.title.strip().lower() == needle]
    matches = exact or [c for c in chunks if needle in c.title.strip().lower()]
    if not matches:
        return f"No article titled {title!r} found."
    body = "\n".join(c.text for c in matches[:limit])
    body = body[:_MAX_CHARS]
    return f"{matches[0].title}\n{body}"


def wiki_lookup(title: str) -> str:
    """Look up a specific wiki article by its title and return its opening text.

    Use this for direct factual lookups about a named entity (a country, person,
    work, …) when you already know what it is called. Returns a not-found message
    if no article carries that title.

    Args:
        title: The article title to look up, e.g. "France" or "Romeo and Juliet".
    """
    return lookup(title, runtime.retriever().chunks)
