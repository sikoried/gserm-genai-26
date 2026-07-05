"""Web-search tool (online, opt-in): DuckDuckGo via the cost-free ``ddgs`` library.

No API key and no paid service — ``ddgs`` scrapes public results. This tool reaches
the network, so it is only registered when ``enable_online_tools`` is set. It calls
no LLM, so it records 0 tokens (its cost is the network round-trip, shown in the
trace timing).

The actual HTTP call lives in ``_backend`` so tests can monkeypatch it and stay
network-free.
"""
from __future__ import annotations

from .net import is_offline_error, offline_message

_MAX_SNIPPET = 240


def _backend(query: str, k: int) -> list[dict]:
    """Return up to `k` results as {title, href, body} dicts (lazy import)."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=k))


def google_search(query: str, k: int = 10) -> str:
    """Search the web and return the best results as title · url · snippet.

    Use for facts unlikely to be in the local wiki index (recent events, niche
    trivia). Returns up to `k` results (default 10); may be empty. Online tool.

    Args:
        query: What to search the web for.
        k: Maximum number of results to return (capped at 10).
    """
    q = (query or "").strip()
    if not q:
        return "No query given."
    k = max(1, min(int(k or 10), 10))
    try:
        results = _backend(q, k) or []
    except Exception as exc:  # network / parse failure must not raise
        if is_offline_error(exc):
            return offline_message("web search")
        return f"Web search failed: {exc}"
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:k], 1):
        title = (r.get("title") or "").strip()
        url = (r.get("href") or r.get("url") or "").strip()
        body = " ".join((r.get("body") or "").split())[:_MAX_SNIPPET]
        lines.append(f"[{i}] {title}\n{url}\n{body}")
    return "\n\n".join(lines)
