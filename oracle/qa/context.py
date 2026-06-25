"""Prompt-context assembly: token budgeting + extractive compression (§F4).

Given ranked retrieval hits, this builds the context block that goes into the
prompt while (a) never exceeding a token budget and (b) optionally shrinking each
passage to only its query-relevant sentences. Citation handles stay valid: the
``[i]`` index is the block's position in the rendered list, compression only
shortens a block's text (never removes the block), and budget enforcement drops
whole trailing blocks (or truncates the last one) so the remaining ``[1..n]`` are
exactly what the prompt lists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..retrieval.tokens import count_tokens, truncate_to_tokens

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass
class ContextBlock:
    """One numbered source in the prompt context."""
    title: str
    text: str
    url: str
    score: float


def split_sentences(text: str) -> list[str]:
    """Split into sentences on terminal punctuation (keeps order, drops empties)."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def compress_extractive(text: str, query_vec: np.ndarray, embed_fn: EmbedFn,
                        keep_sentences: int = 3) -> str:
    """Keep the ``keep_sentences`` most query-relevant sentences, in original order.

    ``embed_fn`` must return L2-normalized row vectors (the local Embedder does);
    relevance is cosine similarity to ``query_vec`` (also normalized).
    """
    sentences = split_sentences(text)
    if len(sentences) <= keep_sentences:
        return text.strip()
    vecs = np.asarray(embed_fn(sentences), dtype="float32")
    scores = vecs @ np.asarray(query_vec, dtype="float32").reshape(-1)
    keep_idx = sorted(np.argsort(scores)[::-1][:keep_sentences].tolist())
    return " ".join(sentences[i] for i in keep_idx)


def _block_tokens(block: ContextBlock, tokenizer=None) -> int:
    # Mirror the rendered form so the budget reflects what the LLM actually sees.
    return count_tokens(f"[0] {block.title}\n{block.text}", tokenizer)


def assemble_context(blocks: list[ContextBlock], budget_tokens: int,
                     tokenizer=None) -> list[ContextBlock]:
    """Trim ``blocks`` (already relevance-ordered) to fit ``budget_tokens``.

    Whole blocks are kept until the budget would be exceeded; the first block
    that does not fit is truncated to the remaining budget (if any room is left)
    and becomes the last kept block. The rendered context never exceeds budget.
    """
    if budget_tokens <= 0:
        return []
    kept: list[ContextBlock] = []
    used = 0
    for b in blocks:
        cost = _block_tokens(b, tokenizer)
        if used + cost <= budget_tokens:
            kept.append(b)
            used += cost
            continue
        # Try to fit a truncated version of this block in the remaining room.
        room = budget_tokens - used
        overhead = count_tokens(f"[0] {b.title}\n", tokenizer)
        if room - overhead >= 1:
            truncated = truncate_to_tokens(b.text, room - overhead, tokenizer)
            if truncated:
                kept.append(ContextBlock(b.title, truncated, b.url, b.score))
        break
    return kept


def render_context(blocks: list[ContextBlock]) -> str:
    """Render numbered ``[i]`` blocks (same shape as ``rag_prompt.j2``)."""
    return "\n".join(f"[{i}] {b.title}\n{b.text}\n" for i, b in enumerate(blocks, 1))


def blocks_from_hits(hits) -> list[ContextBlock]:
    return [ContextBlock(title=h.title, text=h.text, url=h.url, score=h.score)
            for h in hits]


def compress_blocks(blocks: list[ContextBlock], query_vec: np.ndarray,
                    embed_fn: EmbedFn, keep_sentences: int) -> list[ContextBlock]:
    """Apply extractive compression to each block's text (citation-stable)."""
    return [
        ContextBlock(b.title, compress_extractive(b.text, query_vec, embed_fn, keep_sentences),
                     b.url, b.score)
        for b in blocks
    ]
