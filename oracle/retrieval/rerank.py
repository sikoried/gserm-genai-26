"""Rerankers for the RAG candidate set (requirements §F3).

Two abstractions behind a tiny ``Reranker`` interface:

* ``CrossEncoderReranker`` — a cross-encoder relevance model. **Lazy-loaded**:
  the model is only imported/instantiated on first ``rerank`` call, so an
  ``off`` pipeline never pulls the dependency (and a missing model degrades to a
  clear error, not an import-time crash).
* ``MmrReranker`` — Maximal Marginal Relevance over candidate embeddings
  (pure numpy), to diversify the kept set. The core ``mmr_select`` is a free
  function so it is unit-testable with canned vectors and no model load.

Both operate on ``list[Hit]`` and return a re-ordered ``list[Hit]``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .store import Hit


class Reranker(ABC):
    """Re-orders retrieved hits for a query; may rewrite their scores."""

    @abstractmethod
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        ...


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).

    The model is loaded lazily on first use (never at import / config time).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "cross-encoder rerank requested but sentence-transformers "
                    f"CrossEncoder is unavailable: {exc}"
                ) from exc
            try:
                self._model = CrossEncoder(self.model_name, device=self.device,
                                           local_files_only=True)
            except Exception:
                self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        model = self._load()
        raw = np.asarray(model.predict([(query, h.text) for h in hits]),
                         dtype="float32")
        # ms-marco cross-encoders emit unbounded logits; squash with a sigmoid so
        # the stored score is a (0,1) relevance probability, comparable to the
        # cosine scores used elsewhere. Sigmoid is monotonic, so the order below
        # is identical to ordering by the raw logits.
        scores = 1.0 / (1.0 + np.exp(-raw))
        order = np.argsort(scores)[::-1]
        out: list[Hit] = []
        for rank in order:
            h = hits[int(rank)]
            out.append(Hit(text=h.text, title=h.title, url=h.url,
                           score=float(scores[int(rank)]), chunk_id=h.chunk_id,
                           parent_id=h.parent_id, doc_id=h.doc_id, idx=h.idx))
        return out


def mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int,
               lambda_: float = 0.5) -> list[int]:
    """Greedy Maximal Marginal Relevance selection over (normalized) vectors.

    Returns up to ``k`` row indices into ``doc_vecs`` ordered by selection.
    Score balances relevance to the query against redundancy with already
    selected docs: ``lambda_ * sim(q, d) - (1 - lambda_) * max sim(d, sel)``.
    """
    q = np.asarray(query_vec, dtype="float32").reshape(-1)
    d = np.asarray(doc_vecs, dtype="float32")
    n = d.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    rel = d @ q
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < k and remaining:
        best_i, best_score = None, None
        for i in remaining:
            if selected:
                redundancy = float(np.max(d[selected] @ d[i]))
            else:
                redundancy = 0.0
            score = lambda_ * float(rel[i]) - (1.0 - lambda_) * redundancy
            if best_score is None or score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        remaining.discard(best_i)
    return selected


class MmrReranker(Reranker):
    """Diversify hits with MMR. Needs per-hit vectors via ``with_vectors``."""

    def __init__(self, lambda_: float = 0.5, k: int | None = None):
        self.lambda_ = lambda_
        self.k = k

    def rerank_vectors(self, query_vec: np.ndarray, hits: list[Hit],
                       doc_vecs: np.ndarray, k: int | None = None) -> list[Hit]:
        k = k or self.k or len(hits)
        order = mmr_select(query_vec, doc_vecs, k, self.lambda_)
        return [hits[i] for i in order]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:  # pragma: no cover
        raise NotImplementedError("MmrReranker needs vectors; use rerank_vectors")
