"""Local document retrieval for RAG: chunking, embeddings, and a FAISS store."""
from __future__ import annotations

from .chunking import Chunk, chunk_document, chunk_text
from .index import INDEX_DIR, build_index, load_retriever
from .store import Embedder, FaissRetriever, Hit, Retriever

__all__ = [
    "Chunk", "chunk_document", "chunk_text",
    "Embedder", "Retriever", "FaissRetriever", "Hit",
    "INDEX_DIR", "build_index", "load_retriever",
]
