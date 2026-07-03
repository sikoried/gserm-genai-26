"""Local document retrieval for RAG: chunking, embeddings, and a FAISS store."""
from __future__ import annotations

from .chunking import (
    Chunk, Chunker, FixedChunker, SemanticChunker, StructuralChunker,
    build_chunker, chunk_document, chunk_text,
)
from .index import (
    INDEX_DIR, LoadedIndex, build_index, check_chunking_mismatch,
    load_index, load_retriever,
)
from .rerank import CrossEncoderReranker, MmrReranker, Reranker, mmr_select
from .store import Embedder, FaissRetriever, Hit, Retriever
from .tokens import count_tokens, truncate_to_tokens

__all__ = [
    "Chunk", "Chunker", "FixedChunker", "SemanticChunker", "StructuralChunker",
    "build_chunker", "chunk_document", "chunk_text",
    "Embedder", "Retriever", "FaissRetriever", "Hit",
    "INDEX_DIR", "LoadedIndex", "build_index", "load_index", "load_retriever",
    "check_chunking_mismatch",
    "Reranker", "CrossEncoderReranker", "MmrReranker", "mmr_select",
    "count_tokens", "truncate_to_tokens",
]
