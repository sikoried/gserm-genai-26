#!/usr/bin/env python3
"""Run evaluation across all QA modes (world, rag, a-rag) and produce a KPI comparison.

Outputs per-item results as JSON and prints a summary table comparing accuracy,
token usage, and round-trip times across modes — useful as a baseline before
feature changes.

Usage:
    .venv/Scripts/python.exe bin/run_eval_all.py
    .venv/Scripts/python.exe bin/run_eval_all.py --out data/eval-results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from oracle.config import QAConfig  # noqa: E402
from oracle.eval.judge import VERDICTS  # noqa: E402
from oracle.eval.runner import ItemResult, load_pairs, run_eval, summarize  # noqa: E402

CONFIGS = {
    "world": REPO / "configs" / "world.yaml",
    "rag": REPO / "configs" / "rag.yaml",
    "a-rag": REPO / "configs" / "arag.yaml",
}

LABEL_ICON = {"correct": "+", "wrong": "X", "orthogonal": "~", "error": "!"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all QA modes and compare KPIs.")
    p.add_argument("--pairs", default=str(REPO / "eval" / "qa_pairs.yaml"))
    p.add_argument("--out", default=None, help="path to write combined JSON results")
    return p.parse_args()


def print_progress(mode: str):
    def _cb(r: ItemResult) -> None:
        icon = LABEL_ICON.get(r.verdict, "?")
        print(f"  [{icon}] {r.id}: {r.verdict.upper()}  "
              f"({r.elapsed_seconds:.2f}s, {r.total_tokens} tok)")
    return _cb


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args.pairs)
    all_results: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}

    for mode, config_path in CONFIGS.items():
        if not config_path.exists():
            print(f"⚠ Skipping {mode}: {config_path} not found")
            continue

        config = QAConfig.from_yaml(config_path)
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}  |  Model: {config.model}")
        print(f"{'='*60}")

        t0 = time.perf_counter()
        results = run_eval(config, pairs, progress=print_progress(mode))
        wall_time = time.perf_counter() - t0

        counts = summarize(results)
        n = len(results)

        total_prompt = sum(r.prompt_tokens for r in results)
        total_completion = sum(r.completion_tokens for r in results)
        total_tokens = sum(r.total_tokens for r in results)
        total_elapsed = sum(r.elapsed_seconds for r in results)
        avg_elapsed = total_elapsed / n if n else 0

        summaries[mode] = {
            "n": n,
            "correct": counts["correct"],
            "wrong": counts["wrong"],
            "orthogonal": counts["orthogonal"],
            "error": counts["error"],
            "accuracy": (counts["correct"] / (counts["correct"] + counts["wrong"])
                         if (counts["correct"] + counts["wrong"]) else None),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": total_prompt / n if n else 0,
            "avg_completion_tokens": total_completion / n if n else 0,
            "avg_total_tokens": total_tokens / n if n else 0,
            "avg_elapsed_seconds": avg_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "wall_time_seconds": wall_time,
        }

        all_results[mode] = [asdict(r) for r in results]

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  KPI COMPARISON ACROSS MODES")
    print(f"{'='*80}\n")

    header = f"{'KPI':<30s}"
    for mode in summaries:
        header += f"  {mode:>12s}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Questions", "n", "{:>12d}"),
        ("Correct", "correct", "{:>12d}"),
        ("Wrong", "wrong", "{:>12d}"),
        ("Orthogonal", "orthogonal", "{:>12d}"),
        ("Errors", "error", "{:>12d}"),
        ("Accuracy", "accuracy", "{:>11.0%} "),
        ("Avg prompt tokens", "avg_prompt_tokens", "{:>12.0f}"),
        ("Avg completion tokens", "avg_completion_tokens", "{:>12.0f}"),
        ("Avg total tokens", "avg_total_tokens", "{:>12.0f}"),
        ("Total tokens", "total_tokens", "{:>12,d}"),
        ("Avg latency (s)", "avg_elapsed_seconds", "{:>12.2f}"),
        ("Total latency (s)", "total_elapsed_seconds", "{:>12.2f}"),
        ("Wall time (s)", "wall_time_seconds", "{:>12.2f}"),
    ]

    for label, key, fmt in rows:
        line = f"{label:<30s}"
        for mode in summaries:
            val = summaries[mode][key]
            if val is None:
                line += f"  {'N/A':>12s}"
            else:
                line += f"  {fmt.format(val)}"
        print(line)

    # ── Write JSON ────────────────────────────────────────────────────
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summaries": summaries, "results": all_results}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {out}")

    print()


if __name__ == "__main__":
    main()
