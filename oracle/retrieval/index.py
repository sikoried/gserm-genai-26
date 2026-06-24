"""Build, cache, and load the wiki-10k retrieval index.

Embeddings are expensive, so an index is built once and cached under
``data/rag-index/`` (gitignored): an ``embeddings.npy`` matrix, a ``chunks.jsonl``
sidecar, and ``meta.json``. ``load_retriever`` rebuilds the in-memory FAISS index
from the cache (fast); ``build_index`` (re)creates the cache from the dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import Chunk, chunk_document
from .store import Embedder, FaissRetriever

_REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = _REPO_ROOT / "data" / "rag-index"
_HF_CACHE = _REPO_ROOT / "data" / "hf-cache"
DATASET = "NeelNanda/wiki-10k"


def build_index(max_docs: int | None = None, chunk_size: int = 800, overlap: int = 150,
                model_name: str = "all-MiniLM-L6-v2", cache_dir: Path | None = None,
                log: Callable[[str], None] = print) -> int:
    """Chunk + embed (up to `max_docs`) the corpus and cache the index. Returns chunk count."""
    from datasets import load_dataset

    cache_dir = Path(cache_dir or INDEX_DIR)
    log(f"loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train", cache_dir=str(_HF_CACHE))
    if max_docs:
        ds = ds.select(range(min(max_docs, ds.num_rows)))

    log(f"chunking {ds.num_rows:,} documents (size={chunk_size}, overlap={overlap}) ...")
    chunks: list[Chunk] = []
    for row in ds:
        chunks.extend(chunk_document(row, chunk_size, overlap))
    log(f"  -> {len(chunks):,} chunks")

    log(f"embedding with {model_name} ...")
    embedder = Embedder(model_name)
    log(f"  device: {embedder.device}")
    embeddings = embedder.encode([c.text for c in chunks])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)
    with (cache_dir / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.text, "doc_id": c.doc_id,
                                "title": c.title, "url": c.url}) + "\n")
    (cache_dir / "meta.json").write_text(json.dumps({
        "dataset": DATASET, "model": model_name, "chunk_size": chunk_size,
        "overlap": overlap, "max_docs": max_docs, "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }, indent=2))
    log(f"cached index to {cache_dir}")
    return len(chunks)


def load_retriever(cache_dir: Path | None = None) -> FaissRetriever:
    """Load the cached index and rebuild the in-memory FAISS retriever."""
    cache_dir = Path(cache_dir or INDEX_DIR)
    emb_path = cache_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"No RAG index at {cache_dir}. Build it first: "
            f".venv/bin/python bin/build_index.py"
        )
    embeddings = np.load(emb_path)
    chunks: list[Chunk] = []
    with (cache_dir / "chunks.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(d["text"], d["doc_id"], d["title"], d["url"]))
    return FaissRetriever(embeddings, chunks)
