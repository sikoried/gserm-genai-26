#!/usr/bin/env python3
"""Build and cache the wiki-10k RAG index (chunks + local embeddings).

The index is cached under ``data/rag-index/`` and loaded at query time by the
`rag` QA system. Re-run this to rebuild (e.g. after changing chunking or model).

Usage:
    .venv/bin/python bin/build_index.py                 # all 10k articles
    .venv/bin/python bin/build_index.py --max-docs 500  # a subset (faster)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.retrieval import INDEX_DIR, build_index  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build the wiki-10k RAG index.")
    p.add_argument("--max-docs", type=int, default=None,
                   help="limit number of articles (default: all 10k)")
    p.add_argument("--chunk-size", type=int, default=800, help="chars per chunk")
    p.add_argument("--overlap", type=int, default=150, help="char overlap between chunks")
    p.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    args = p.parse_args()

    t0 = time.perf_counter()
    n = build_index(max_docs=args.max_docs, chunk_size=args.chunk_size,
                    overlap=args.overlap, model_name=args.model)
    print(f"Done: {n:,} chunks indexed at {INDEX_DIR} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
