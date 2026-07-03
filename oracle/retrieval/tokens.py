"""Offline, dependency-light token counting and truncation (requirements §3.5).

`count_tokens` / `truncate_to_tokens` work without any network access. They use
the embedding model's own tokenizer when one is supplied (most accurate), and
otherwise fall back to a deterministic word/punctuation approximation so unit
tests and the structural chunker run with no model load and no network.

The approximation counts word and punctuation runs (``\\w+`` or a single
non-word, non-space char). For English prose this tracks sub-word tokenizers
closely enough to keep chunks under the MiniLM 256-token window with the default
220-token target and its safety margin.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _approx_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _TOKEN_RE.finditer(text)]


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the approximate tokens in ``text`` (offline)."""
    return _approx_spans(text)


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    """Number of tokens in ``text``.

    If ``tokenizer`` (a Hugging Face / sentence-transformers tokenizer exposing
    ``encode``) is given, use it; otherwise use the deterministic approximation.
    """
    if not text:
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))
    return len(_approx_spans(text))


def truncate_to_tokens(text: str, max_tokens: int, tokenizer: Any | None = None) -> str:
    """Return the longest prefix of ``text`` with at most ``max_tokens`` tokens.

    With a real tokenizer this round-trips through encode/decode; with the
    approximation it cuts at the end of the ``max_tokens``-th token span so the
    result never splits a word.
    """
    if max_tokens <= 0 or not text:
        return ""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        if len(ids) <= max_tokens:
            return text
        try:
            return tokenizer.decode(ids[:max_tokens]).strip()
        except Exception:
            pass  # fall through to the approximation
    spans = _approx_spans(text)
    if len(spans) <= max_tokens:
        return text
    return text[: spans[max_tokens - 1][1]].rstrip()
