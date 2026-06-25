"""Query transformation before retrieval (requirements §F6).

Strategies (default ``none`` reproduces the baseline exactly):

* ``none``    — embed the user message verbatim.
* ``rewrite`` — one LLM-reformulated, more descriptive query.
* ``multi``   — the original plus several LLM paraphrases; their candidate sets
                are merged + deduplicated before ranking, expanding recall.
* ``hyde``    — a hypothetical answer passage whose embedding is used to retrieve
                (Hypothetical Document Embeddings).

The LLM call is injected as a ``generate(system, user) -> str`` callable so the
transforms are unit-testable with a stub and carry no network at import time.
"""
from __future__ import annotations

from typing import Callable

from ..llm import chat
from ..retrieval.store import Hit

GenerateFn = Callable[[str, str], str]


def make_generator(client, model: str, temperature: float = 0.0) -> "GenerateFn":
    """Build a ``generate(system, user) -> str`` bound to a client + model.

    Central helper for auxiliary LLM tasks (query transforms, llm compression):
    pass ``config.aux`` as ``model`` to route helper calls to the smaller model
    while the final answer keeps using the main model (requirements §F5).
    """
    def generate(system: str, user: str) -> str:
        return chat(
            client, model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        ).strip()
    return generate

REWRITE_SYS = (
    "Rewrite the user's search query into a single, more descriptive sentence that "
    "captures its intent for semantic retrieval. Return only the rewritten query."
)
MULTI_SYS = (
    "Generate {n} alternative search queries that rephrase the user's question from "
    "different angles for document retrieval. Return one query per line, no numbering."
)
HYDE_SYS = (
    "Write a short, factual paragraph that would directly answer the user's question, "
    "as if quoting an encyclopedia. Return only the paragraph."
)


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving, case-insensitive de-duplication of non-empty strings."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = (it or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out


def _parse_lines(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*0123456789.) ").strip()
        if line:
            out.append(line)
    return out


def transform_queries(query: str, qcfg, generate: GenerateFn) -> list[str]:
    """Return the list of query strings to embed for retrieval."""
    transform = getattr(qcfg, "transform", "none")
    if transform == "none":
        return [query]
    if transform == "rewrite":
        return [(generate(REWRITE_SYS, query) or query).strip()]
    if transform == "hyde":
        return [(generate(HYDE_SYS, query) or query).strip()]
    if transform == "multi":
        n = max(1, getattr(qcfg, "num_queries", 3))
        extra = _parse_lines(generate(MULTI_SYS.format(n=n), query))
        return _dedupe([query, *extra])[:n + 1]
    return [query]


def merge_candidates(hit_lists: list[list[Hit]]) -> list[Hit]:
    """Merge candidate sets from several queries, deduped by ``chunk_id``.

    Keeps the highest score seen for each chunk; the result is deterministically
    ordered by score (desc) then ``chunk_id`` so multi-query retrieval is stable.
    """
    best: dict[str, Hit] = {}
    for hits in hit_lists:
        for h in hits:
            key = h.chunk_id or f"{h.doc_id}:{h.text[:32]}"
            cur = best.get(key)
            if cur is None or h.score > cur.score:
                best[key] = h
    return sorted(best.values(), key=lambda h: (-h.score, h.chunk_id))
