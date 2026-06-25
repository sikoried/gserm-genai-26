"""Build, cache, and load the wiki-10k retrieval index.

Embeddings are expensive, so an index is built once and cached under
``data/rag-index/`` (gitignored): an ``embeddings.npy`` matrix, a ``chunks.jsonl``
sidecar (child chunks), a ``parents.jsonl`` sidecar (parent/section chunks for
small→big expansion), and ``meta.json`` (dataset, embedder, and the chunking /
resolution parameters the cache was built with).

``load_index`` rebuilds the in-memory FAISS retriever plus the parent lookup from
the cache (fast); ``build_index`` (re)creates the cache from the dataset.
``check_chunking_mismatch`` compares a runtime config against the cache metadata
so a config/index drift is warned about rather than silently wrong (§3.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import ChunkingConfig, QAConfig
from .chunking import Chunk, build_chunker
from .store import Embedder, FaissRetriever

_REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = _REPO_ROOT / "data" / "rag-index"
_HF_CACHE = _REPO_ROOT / "data" / "hf-cache"
DATASET = "NeelNanda/wiki-10k"

_CHUNK_FIELDS = ("text", "doc_id", "title", "url", "chunk_id",
                 "parent_id", "level", "start_char", "end_char")


@dataclass
class LoadedIndex:
    """A loaded RAG index: the retriever, parent lookup, and build metadata."""
    retriever: FaissRetriever
    parents: dict[str, Chunk]  # parent chunk_id -> parent Chunk
    meta: dict


def _chunk_to_record(c: Chunk) -> dict:
    return {k: getattr(c, k) for k in _CHUNK_FIELDS}


def _record_to_chunk(d: dict) -> Chunk:
    return Chunk(
        text=d["text"], doc_id=d.get("doc_id", ""), title=d.get("title", ""),
        url=d.get("url", ""), chunk_id=d.get("chunk_id", ""),
        parent_id=d.get("parent_id"), level=d.get("level", 0),
        start_char=d.get("start_char", 0), end_char=d.get("end_char", 0),
    )


def build_index(config: QAConfig | None = None, *, max_docs: int | None = None,
                cache_dir: Path | None = None, log: Callable[[str], None] = print) -> int:
    """Chunk + embed (up to `max_docs`) the corpus and cache the index.

    The chunking strategy and the embedding model are taken from ``config``
    (defaults reproduce the legacy fixed-window build). Returns the child count.
    """
    from datasets import load_dataset

    config = config or QAConfig(type="rag")
    chunking = config.chunking
    model_name = config.embedding_model
    cache_dir = Path(cache_dir or INDEX_DIR)

    log(f"loading {DATASET} ...")
    ds = load_dataset(DATASET, split="train", cache_dir=str(_HF_CACHE))
    if max_docs:
        ds = ds.select(range(min(max_docs, ds.num_rows)))

    chunker = build_chunker(chunking)
    log(f"chunking {ds.num_rows:,} documents (strategy={chunking.strategy}) ...")
    children: list[Chunk] = []
    parents: list[Chunk] = []
    for row in ds:
        c, p = chunker.split(row)
        children.extend(c)
        parents.extend(p)
    log(f"  -> {len(children):,} child chunks, {len(parents):,} parents")

    log(f"embedding with {model_name} ...")
    embedder = Embedder(model_name)
    log(f"  device: {embedder.device}")
    embeddings = embedder.encode([c.text for c in children])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)
    with (cache_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in children:
            f.write(json.dumps(_chunk_to_record(c), ensure_ascii=False) + "\n")
    with (cache_dir / "parents.jsonl").open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(_chunk_to_record(p), ensure_ascii=False) + "\n")
    (cache_dir / "meta.json").write_text(json.dumps({
        "dataset": DATASET,
        "model": model_name,
        "chunking": chunking.model_dump(),
        "multi_resolution": config.retrieval.multi_resolution,
        "parent_level": config.retrieval.parent_level,
        "max_docs": max_docs,
        "num_chunks": len(children),
        "num_parents": len(parents),
        "has_parents": bool(parents),
        "dim": int(embeddings.shape[1]),
    }, indent=2))
    log(f"cached index to {cache_dir}")
    return len(children)


def load_index(cache_dir: Path | None = None) -> LoadedIndex:
    """Load the cached index: FAISS retriever + parent lookup + build metadata."""
    cache_dir = Path(cache_dir or INDEX_DIR)
    emb_path = cache_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"No RAG index at {cache_dir}. Build it first: "
            f".venv/Scripts/python bin/build_index.py --config configs/rag.yaml"
        )
    embeddings = np.load(emb_path)
    children: list[Chunk] = []
    with (cache_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            children.append(_record_to_chunk(json.loads(line)))
    parents: dict[str, Chunk] = {}
    parents_path = cache_dir / "parents.jsonl"
    if parents_path.exists():
        with parents_path.open(encoding="utf-8") as f:
            for line in f:
                p = _record_to_chunk(json.loads(line))
                parents[p.chunk_id] = p
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return LoadedIndex(FaissRetriever(embeddings, children), parents, meta)


def load_retriever(cache_dir: Path | None = None) -> FaissRetriever:
    """Load the cached index and rebuild the in-memory FAISS retriever (back-compat)."""
    return load_index(cache_dir).retriever


def check_chunking_mismatch(config: QAConfig, meta: dict) -> str | None:
    """Return a human-readable warning if the config's chunking differs from the
    cache metadata, else ``None`` (§3.2). Compares only rebuild-boundary fields.
    """
    if not meta:
        return None
    cached = meta.get("chunking") or {}
    want: ChunkingConfig = config.chunking
    fields = ["strategy", "target_tokens", "overlap_tokens", "size", "overlap"]
    diffs = []
    for field in fields:
        cv = cached.get(field)
        wv = getattr(want, field, None)
        if cv is not None and wv is not None and cv != wv:
            diffs.append(f"{field}: index={cv!r} config={wv!r}")
    if cached.get("strategy") and cached["strategy"] != want.strategy:
        pass  # already captured above
    if config.embedding_model and meta.get("model") and config.embedding_model != meta["model"]:
        diffs.append(f"embedding_model: index={meta['model']!r} config={config.embedding_model!r}")
    if not diffs:
        return None
    return ("RAG index was built with different chunking/embedding settings than "
            "this config — rebuild the index (bin/build_index.py) to match. "
            + "; ".join(diffs))
