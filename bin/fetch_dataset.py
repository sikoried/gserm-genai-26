#!/usr/bin/env python3
"""Fetch and explore the NeelNanda/wiki-10k dataset.

Familiarization / helper script for the Oracle project. It downloads the
corpus that the RAG variants will retrieve over and prints a quick summary
(schema, row count, basic length stats, a few sample rows).

Usage:
    .venv/bin/python bin/fetch_dataset.py            # download + show 3 rows
    .venv/bin/python bin/fetch_dataset.py -n 5       # show 5 sample rows
    .venv/bin/python bin/fetch_dataset.py --chars 1000
"""
from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

DATASET = "NeelNanda/wiki-10k"
# Keep downloads inside the repo (data/ is gitignored) so the corpus is
# self-contained and easy to inspect.
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "hf-cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Fetch and explore {DATASET}.")
    p.add_argument("-n", "--rows", type=int, default=3,
                   help="number of sample rows to print (default: 3)")
    p.add_argument("--chars", type=int, default=500,
                   help="max characters of text to show per row (default: 500)")
    p.add_argument("--split", default="train",
                   help="dataset split to load (default: train)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                   help="HF datasets cache directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    cache_dir = os.path.expanduser(args.cache_dir)
    print(f"Loading {DATASET} (split={args.split}) into {cache_dir} ...")
    ds = load_dataset(DATASET, split=args.split, cache_dir=cache_dir)

    print("\n=== schema / features ===")
    for name, feature in ds.features.items():
        print(f"  {name}: {feature}")

    print("\n=== size ===")
    print(f"  rows: {ds.num_rows:,}")
    print(f"  columns: {ds.column_names}")

    # Length stats over the primary text column.
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    lengths = [len(t) for t in ds[text_col]]
    n = len(lengths)
    print(f"\n=== '{text_col}' length (chars) ===")
    print(f"  min={min(lengths):,}  max={max(lengths):,}  mean={sum(lengths) // n:,}")

    print(f"\n=== first {args.rows} rows ===")
    for i in range(min(args.rows, ds.num_rows)):
        text = str(ds[i][text_col])
        snippet = textwrap.shorten(text.replace("\n", " "), width=args.chars,
                                   placeholder=" …")
        print(f"\n--- row {i} ({len(text):,} chars) ---")
        print(snippet)


if __name__ == "__main__":
    main()
