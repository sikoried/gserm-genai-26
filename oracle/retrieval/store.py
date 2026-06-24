"""Local embeddings (sentence-transformers on MPS) + an in-memory FAISS retriever."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    """A retrieved chunk plus its similarity score."""
    text: str
    title: str
    url: str
    score: float


def _pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """Wraps a sentence-transformers model; encodes to L2-normalized float32."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.device = device or _pick_device()
        try:
            # Prefer the local cache so a query never phones home to HF — only the LLM
            # proxy (kiz1) should be contacted at request time.
            self.model = SentenceTransformer(model_name, device=self.device, local_files_only=True)
        except Exception:
            # Not cached yet (e.g. fresh checkout) — allow a one-time download.
            self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype("float32")


class Retriever(ABC):
    """Swappable retrieval interface (see CLAUDE.md `Embeddings & Vector Store`)."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        ...


class FaissRetriever(Retriever):
    """Exact in-memory cosine search via FAISS `IndexFlatIP` over normalized vectors."""

    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]):
        import faiss
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("embeddings and chunks length mismatch")
        self.chunks = chunks
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))

    def search(self, query_embedding: np.ndarray, k: int) -> list[Hit]:
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.chunks)))
        hits: list[Hit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            c = self.chunks[idx]
            hits.append(Hit(text=c.text, title=c.title, url=c.url, score=float(score)))
        return hits

    def __len__(self) -> int:
        return len(self.chunks)
