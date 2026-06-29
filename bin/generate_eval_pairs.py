#!/usr/bin/env python3
"""Generate corpus-grounded evaluation Q&A pairs from wiki-10k articles.

Samples articles from the dataset, sends excerpts to the LLM to produce
question/answer pairs at varying difficulty levels, and writes them to
eval/qa_pairs.yaml.

Usage:
    .venv/Scripts/python.exe bin/generate_eval_pairs.py
"""
from __future__ import annotations

import json
import random
import sys
import yaml
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oracle.config import DEFAULT_ENDPOINT, DEFAULT_MODEL  # noqa: E402
from oracle.llm import chat, make_client  # noqa: E402

OUTPUT = REPO / "eval" / "qa_pairs.yaml"
NUM_ARTICLES = 10
PAIRS_PER_ARTICLE = 3

SYSTEM_PROMPT = """\
You are an evaluation dataset generator. Given a Wikipedia article excerpt, \
produce exactly {n} question-answer pairs that can ONLY be answered using \
information in the excerpt. Return valid JSON only — no markdown fences.

Requirements:
- Each pair has: "question", "reference_answer", "difficulty", "type"
- difficulty: one of "easy", "medium", "hard"
- type: one of "factoid" (single fact lookup), "multi-hop" (combining 2+ facts \
from the text), "comparison" (comparing entities/numbers in the text)
- Include at least one of each difficulty level across the {n} pairs
- reference_answer must be concise (1-2 sentences max) and grounded in the text
- Questions must be self-contained (don't say "in the article" or "according to the text")

Output format (JSON array):
[{{"question": "...", "reference_answer": "...", "difficulty": "...", "type": "..."}}]"""


def load_articles(n: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    hf_cache = str(REPO / "data" / "hf-cache")
    ds = load_dataset("NeelNanda/wiki-10k", split="train", cache_dir=hf_cache)
    random.seed(seed)
    indices = random.sample(range(ds.num_rows), min(n, ds.num_rows))
    articles = []
    for i in indices:
        row = ds[i]
        articles.append({
            "id": row.get("id", i),
            "title": row["title"],
            "text": row["text"][:4000],
        })
    return articles


def generate_pairs(client, article: dict, n: int) -> list[dict]:
    excerpt = f"Title: {article['title']}\n\n{article['text']}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(n=n)},
        {"role": "user", "content": f"Generate {n} Q&A pairs from this article:\n\n{excerpt}"},
    ]
    raw = chat(client, DEFAULT_MODEL, messages, temperature=0.3)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    pairs = json.loads(raw)
    for p in pairs:
        p["source_article"] = article["title"]
    return pairs


def main() -> None:
    client = make_client(DEFAULT_ENDPOINT)
    print(f"Loading {NUM_ARTICLES} articles from wiki-10k ...")
    articles = load_articles(NUM_ARTICLES)
    print(f"Sampled articles: {[a['title'] for a in articles]}\n")

    all_pairs = []
    for i, article in enumerate(articles):
        print(f"[{i+1}/{NUM_ARTICLES}] Generating from: {article['title']} ...", end=" ", flush=True)
        try:
            pairs = generate_pairs(client, article, PAIRS_PER_ARTICLE)
            all_pairs.extend(pairs)
            print(f"{len(pairs)} pairs")
        except Exception as exc:
            print(f"FAILED: {exc}")

    yaml_pairs = []
    for i, p in enumerate(all_pairs, 1):
        yaml_pairs.append({
            "id": f"wiki-{i:03d}",
            "question": p["question"],
            "reference_answer": p["reference_answer"],
            "difficulty": p.get("difficulty", "medium"),
            "type": p.get("type", "factoid"),
            "source_article": p.get("source_article", ""),
        })

    OUTPUT.write_text(yaml.dump(yaml_pairs, default_flow_style=False,
                                allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    print(f"\nWrote {len(yaml_pairs)} pairs to {OUTPUT}")

    counts = {}
    for p in yaml_pairs:
        d = p["difficulty"]
        counts[d] = counts.get(d, 0) + 1
    print(f"Difficulty distribution: {counts}")
    type_counts = {}
    for p in yaml_pairs:
        t = p["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Type distribution: {type_counts}")


if __name__ == "__main__":
    main()
