#!/usr/bin/env python3
"""Build and cache the wiki-10k RAG index (chunks + local embeddings).

The chunking strategy and embedding model come from a QA-system config (so the
index matches the profile it will serve). The cache lands under ``data/rag-index/``
and is loaded at query time by the `rag` system. Re-run this after changing any
``chunking.*`` setting (it is a rebuild boundary — see requirements §3.1).

Usage:
    .venv/Scripts/python bin/build_index.py --config configs/rag.yaml
    .venv/Scripts/python bin/build_index.py --config configs/rag_structural.yaml --max-docs 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.config import QAConfig  # noqa: E402
from oracle.retrieval import INDEX_DIR, build_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build the wiki-10k RAG index.")
    p.add_argument("--config", default=None,
                   help="QA config YAML (uses its chunking + embedding_model)")
    p.add_argument("--max-docs", type=int, default=None,
                   help="limit number of articles (default: all 10k)")
    p.add_argument("--strategy", default=None,
                   help="override chunking strategy (fixed | structural | semantic | llm)")
    p.add_argument("--model", default=None, help="override sentence-transformers model")
    args = p.parse_args()

    config = QAConfig.from_yaml(args.config) if args.config else QAConfig(type="rag")
    if args.strategy:
        config.chunking.strategy = args.strategy
    if args.model:
        config.embedding_model = args.model

    print(f"strategy  : {config.chunking.strategy}")
    print(f"embedder  : {config.embedding_model}")
    t0 = time.perf_counter()
    n = build_index(config, max_docs=args.max_docs)
    print(f"Done: {n:,} chunks indexed at {INDEX_DIR} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
