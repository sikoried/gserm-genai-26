"""QA-system configuration (see CLAUDE.md `Configuration Schema`).

The schema is backward compatible: a flat baseline config (``type``/``model``/
``top_k``/``embedding_model``) still loads unchanged, while four optional nested
blocks add the advanced-RAG controls:

  chunking   — index-build-time only (changing it **requires an index rebuild**)
  retrieval  — query-time candidate depth, filtering, ranking
  context    — query-time token budget + compression
  query      — query-time transforms (rewrite / multi / hyde)

The legacy flat keys ``top_k`` and ``embedding_model`` remain accepted and are
folded into the nested blocks, so old YAML keeps working (see ``Cross-cutting``
in requirements-rag-simon.md §3.1).
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

DEFAULT_MODEL = "mistralai/Mistral-Medium-3.5-128B"
DEFAULT_ENDPOINT = "https://kiz1.in.ohmportal.de/llmproxy/v1"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ChunkingConfig(BaseModel):
    """How documents are split before embedding. **Requires an index rebuild.**"""

    strategy: str = "fixed"  # fixed | structural | semantic | llm
    target_tokens: int = 220  # token budget per child chunk (structural/semantic)
    overlap_tokens: int = 40  # sentence-aware overlap between child chunks
    semantic_threshold: float = 0.55  # cosine break threshold (semantic strategy)
    llm_model: str | None = None  # model for the (gated) llm strategy
    # Legacy char-window knobs, used by the `fixed` strategy for exact back-compat.
    size: int = 800
    overlap: int = 150


class RetrievalConfig(BaseModel):
    """Query-time candidate depth, filtering, ranking, and multi-resolution."""

    top_k: int = 10  # chunks kept as context after ranking/filtering
    fetch_k: int = 30  # candidates pulled before rerank/filter (>= top_k)
    min_similarity: float = 0.0  # drop candidates below this cosine score
    rerank: str = "off"  # off | cross_encoder
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mmr: bool = False  # diversify the kept set with MMR
    mmr_lambda: float = 0.5  # MMR relevance/diversity trade-off (1.0 = pure relevance)
    # --- multi-resolution (small-to-big) ---
    multi_resolution: str = "off"  # off | small_to_big | multi_index
    parent_level: str = "section"  # section | neighbors
    parent_window: int = 1  # neighbor radius when parent_level == neighbors


class ContextConfig(BaseModel):
    """Query-time prompt-context budget and compression."""

    budget_tokens: int = 1500  # hard ceiling on rendered context tokens
    compression: str = "off"  # off | extractive | llm
    keep_sentences: int = 3  # sentences kept per passage (extractive)
    compress_model: str | None = None  # model for llm compression (gated)


class QueryConfig(BaseModel):
    """Query transformation applied before retrieval."""

    transform: str = "none"  # none | rewrite | multi | hyde
    num_queries: int = 3  # variants generated for the `multi` transform


class QAConfig(BaseModel):
    """Configuration for a single QA system."""

    type: str = "world"  # world | rag | a-rag
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    temperature: float = 0.0
    system_prompt: str | None = None  # overrides the QA system's built-in prompt
    reasoning_effort: str | None = None  # "low"/"medium"/"high"; passed to the model when set
    # --- RAG ---
    embedding_model: str = DEFAULT_EMBEDDING_MODEL  # local embedding model for retrieval
    aux_model: str | None = None  # smaller helper model for rewrite/compression; None -> `model`
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    query: QueryConfig = QueryConfig()

    @model_validator(mode="before")
    @classmethod
    def _fold_legacy_aliases(cls, data):
        """Accept the legacy flat ``top_k`` key and fold it into ``retrieval``."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "top_k" in data:
            legacy_top_k = data.pop("top_k")
            retrieval = dict(data.get("retrieval") or {})
            retrieval.setdefault("top_k", legacy_top_k)
            data["retrieval"] = retrieval
        return data

    @property
    def top_k(self) -> int:
        """Back-compat alias: the number of context chunks (``retrieval.top_k``)."""
        return self.retrieval.top_k

    @property
    def aux(self) -> str:
        """Resolved auxiliary model id — falls back to the main ``model`` when unset."""
        return self.aux_model or self.model

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QAConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
