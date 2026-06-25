"""Chunk wiki-10k articles for retrieval.

Two strategies behind a small interface (requirements §F1):

* ``FixedChunker`` — the original fixed char-window split (default-compatible).
* ``StructuralChunker`` — token-budget-aware, paragraph/sentence-boundary aware,
  and it emits a small→big hierarchy (child chunks for precise retrieval, parent
  chunks for broader answer context — requirements §F2).

Every chunk carries hierarchy fields (``chunk_id``, ``parent_id``, ``level``,
``start_char``, ``end_char``). A chunker's ``split`` returns ``(children,
parents)``; ``parents`` is empty when the strategy has no hierarchy (fixed).

The module-level ``chunk_text`` / ``chunk_document`` helpers preserve the old
fixed-window behavior and imports.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .tokens import count_tokens, token_spans, truncate_to_tokens

# A "paragraph" in wiki-10k is a single non-empty line; sentences end at . ? !
# followed by whitespace. Both are deliberately simple and offset-preserving.
_PARAGRAPH_RE = re.compile(r"[^\n]+")
_SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s|$)|$)", re.DOTALL)


@dataclass
class Chunk:
    """A retrievable slice of a source document (with small→big hierarchy info)."""
    text: str
    doc_id: str
    title: str
    url: str
    chunk_id: str = ""
    parent_id: str | None = None
    level: int = 0  # 0 = child (retrieval unit), 1 = parent (context unit)
    start_char: int = 0
    end_char: int = 0


# ---------------------------------------------------------------------------
# Fixed char-window chunking (legacy default)
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split `text` into ~`size`-char windows overlapping by `overlap` chars.

    ``size`` defaults below the embedding model's token window (all-MiniLM-L6-v2
    truncates at 256 tokens ≈ ~1000 chars) so chunks embed without truncation.
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _char_windows(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Like ``chunk_text`` but keep each window's (start, end) char offsets."""
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    out: list[tuple[str, int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        out.append((text[start:end], start, end))
        if end >= n:
            break
        start = end - overlap
    return out


def chunk_document(doc: dict, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk one wiki-10k row with the fixed strategy (back-compat helper)."""
    children, _ = FixedChunker(size=size, overlap=overlap).split(doc)
    return children


# ---------------------------------------------------------------------------
# Chunker strategy interface
# ---------------------------------------------------------------------------

class Chunker(ABC):
    """Splits a wiki-10k row into ``(children, parents)`` chunk lists."""

    @abstractmethod
    def split(self, doc: dict) -> tuple[list[Chunk], list[Chunk]]:
        ...


def _doc_fields(doc: dict) -> tuple[str, str, str, str]:
    return (
        (doc.get("text", "") or "").strip(),
        str(doc.get("id", "")),
        doc.get("title", ""),
        doc.get("url", ""),
    )


class FixedChunker(Chunker):
    """Original fixed char-window split. Flat (no parent hierarchy)."""

    def __init__(self, size: int = 800, overlap: int = 150):
        self.size = size
        self.overlap = overlap

    def split(self, doc: dict) -> tuple[list[Chunk], list[Chunk]]:
        text, doc_id, title, url = _doc_fields(doc)
        children = [
            Chunk(text=piece, doc_id=doc_id, title=title, url=url,
                  chunk_id=f"{doc_id}#c{i}", parent_id=None, level=0,
                  start_char=s, end_char=e)
            for i, (piece, s, e) in enumerate(_char_windows(text, self.size, self.overlap))
        ]
        return children, []


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences within ``text`` (no empties)."""
    spans = [(m.start(), m.end()) for m in _SENTENCE_RE.finditer(text) if m.group().strip()]
    return spans or ([(0, len(text))] if text else [])


class StructuralChunker(Chunker):
    """Token-budget, paragraph/sentence-aware chunking with a small→big hierarchy.

    Each non-empty line is a *parent* (a section/paragraph). Within a parent,
    children pack whole sentences up to ``target_tokens``; if a parent already
    fits the budget the single child equals the parent. Child overlap reuses
    trailing sentences summing to ~``overlap_tokens`` so context bleeds across a
    cut without splitting a sentence. No child exceeds ``target_tokens`` (a lone
    over-budget sentence is hard-truncated by token count).
    """

    def __init__(self, target_tokens: int = 220, overlap_tokens: int = 40,
                 tokenizer=None):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer

    def _ntoks(self, text: str) -> int:
        return count_tokens(text, self.tokenizer)

    def split(self, doc: dict) -> tuple[list[Chunk], list[Chunk]]:
        text, doc_id, title, url = _doc_fields(doc)
        children: list[Chunk] = []
        parents: list[Chunk] = []
        ci = pi = 0
        for m in _PARAGRAPH_RE.finditer(text):
            para, p_start = m.group(), m.start()
            if not para.strip():
                continue
            parent_id = f"{doc_id}#p{pi}"
            parents.append(Chunk(
                text=para, doc_id=doc_id, title=title, url=url,
                chunk_id=parent_id, parent_id=None, level=1,
                start_char=p_start, end_char=p_start + len(para),
            ))
            pi += 1
            for c_text, c_start, c_end in self._children_of(para):
                children.append(Chunk(
                    text=c_text, doc_id=doc_id, title=title, url=url,
                    chunk_id=f"{doc_id}#c{ci}", parent_id=parent_id, level=0,
                    start_char=p_start + c_start, end_char=p_start + c_end,
                ))
                ci += 1
        return children, parents

    def _children_of(self, para: str) -> list[tuple[str, int, int]]:
        """Pack a paragraph's sentences into token-budgeted, overlapping children."""
        if self._ntoks(para) <= self.target_tokens:
            return [self._clip(para, 0, len(para))]
        spans = self._units(para)  # each unit already fits the token budget
        n = len(spans)
        out: list[tuple[str, int, int]] = []
        i = 0
        while i < n:
            j, tok = i, 0
            while j < n:
                stoks = self._ntoks(para[spans[j][0]:spans[j][1]])
                if j > i and tok + stoks > self.target_tokens:
                    break
                tok += stoks
                j += 1
            out.append(self._clip(para, spans[i][0], spans[j - 1][1]))
            if j >= n:
                break
            i = self._next_start(para, spans, i, j)
        return out

    def _units(self, para: str) -> list[tuple[int, int]]:
        """Sentence spans, with any over-budget sentence pre-split into token windows.

        Guarantees every unit fits ``target_tokens`` so the packer never has to
        drop content (a lone giant sentence is windowed, not truncated).
        """
        units: list[tuple[int, int]] = []
        for s, e in _sentence_spans(para):
            if self._ntoks(para[s:e]) <= self.target_tokens:
                units.append((s, e))
            else:
                units.extend(self._token_windows(para, s, e))
        return units

    def _token_windows(self, para: str, s: int, e: int) -> list[tuple[int, int]]:
        """Split ``para[s:e]`` into overlapping ≤target_tokens token windows."""
        spans = token_spans(para[s:e])
        out: list[tuple[int, int]] = []
        i, n = 0, len(spans)
        step = max(1, self.target_tokens - self.overlap_tokens)
        while i < n:
            j = min(i + self.target_tokens, n)
            out.append((s + spans[i][0], s + spans[j - 1][1]))
            if j >= n:
                break
            i += step
        return out

    def _clip(self, para: str, a: int, b: int) -> tuple[str, int, int]:
        """Slice ``para[a:b]``, drop leading space, and hard-cap to the budget."""
        while a < b and para[a].isspace():
            a += 1
        text = para[a:b]
        if self._ntoks(text) > self.target_tokens:
            text = truncate_to_tokens(text, self.target_tokens, self.tokenizer)
            b = a + len(text)
        return (text, a, b)

    def _next_start(self, para: str, spans: list[tuple[int, int]], i: int, j: int) -> int:
        """Sentence index where the next child begins — back up ~overlap_tokens.

        Never backs up to or past ``i`` so each child advances at least one
        sentence (forward progress is guaranteed).
        """
        if self.overlap_tokens <= 0:
            return j
        tok, k = 0, j
        while k - 1 > i:
            s = self._ntoks(para[spans[k - 1][0]:spans[k - 1][1]])
            if tok and tok + s > self.overlap_tokens:
                break
            tok += s
            k -= 1
        return max(k, i + 1)


def build_chunker(chunking) -> Chunker:
    """Construct the chunker named by a ``ChunkingConfig`` (defaults to fixed)."""
    strategy = getattr(chunking, "strategy", "fixed")
    if strategy in ("structural", "semantic", "llm"):
        # semantic/llm fall back to structural's boundary logic for the no-network
        # path; their extra behavior is layered on at index build time when enabled.
        return StructuralChunker(
            target_tokens=chunking.target_tokens,
            overlap_tokens=chunking.overlap_tokens,
        )
    return FixedChunker(size=chunking.size, overlap=chunking.overlap)
