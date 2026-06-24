"""Split wiki-10k articles into overlapping character chunks for retrieval."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    """A retrievable slice of a source document."""
    text: str
    doc_id: str
    title: str
    url: str


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


def chunk_document(doc: dict, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk one wiki-10k row (`id`, `title`, `url`, `text`)."""
    return [
        Chunk(text=piece, doc_id=str(doc.get("id", "")),
              title=doc.get("title", ""), url=doc.get("url", ""))
        for piece in chunk_text(doc.get("text", ""), size, overlap)
    ]
