#!/usr/bin/env python3
"""Run an end-to-end evaluation of a QA system.

Loads a QA-system config (YAML) and a set of query/answer pairs, answers each
question with the configured system, and judges each answer with LLM-as-a-judge
(verdict: correct | wrong | orthogonal).

Usage:
    .venv/bin/python bin/run_eval.py
    .venv/bin/python bin/run_eval.py --config configs/world.yaml --pairs eval/qa_pairs.yaml
    .venv/bin/python bin/run_eval.py --out data/eval-results/world.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when run directly as a script.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# The progress printer uses ✓/✗/≈ icons; force UTF-8 so a cp1252 console
# (Windows default) doesn't crash on them.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from oracle.config import QAConfig  # noqa: E402
from oracle.eval.judge import JUDGE_MODEL, VERDICTS  # noqa: E402
from oracle.eval.runner import (  # noqa: E402
    load_pairs, results_as_dicts, run_eval, summarize,
)

LABEL_ICON = {"correct": "✓", "wrong": "✗", "orthogonal": "≈", "error": "!"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end QA evaluation.")
    p.add_argument("--config", default=str(REPO / "configs" / "world.yaml"))
    p.add_argument("--pairs", default=str(REPO / "eval" / "qa_pairs.yaml"))
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--out", default=None, help="optional path to write JSON results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = QAConfig.from_yaml(args.config)
    pairs = load_pairs(args.pairs)

    print(f"QA system : {config.type}  (model={config.model}, temp={config.temperature})")
    print(f"Judge     : {args.judge_model}")
    print(f"Endpoint  : {config.endpoint}")
    print(f"Pairs     : {len(pairs)} from {args.pairs}\n")

    def progress(r) -> None:
        icon = LABEL_ICON.get(r.verdict, "?")
        snippet = r.answer if len(r.answer) <= 160 else r.answer[:160] + "…"
        print(f"  [{icon}] {r.id}: {r.verdict.upper()}")
        print(f"      Q:   {r.question}")
        print(f"      A:   {snippet}")
        print(f"      ref: {r.reference}  |  {r.reasoning}")
        print(f"      ⏱  {r.elapsed_seconds:.2f}s  "
              f"tokens: {r.prompt_tokens}+{r.completion_tokens}={r.total_tokens}\n")

    results = run_eval(config, pairs, judge_model=args.judge_model, progress=progress)

    counts = summarize(results)
    n = len(results)
    print("=== summary ===")
    for label in VERDICTS:
        c = counts[label]
        pct = f"{100 * c / n:.0f}%" if n else "—"
        print(f"  {label:11s}: {c}/{n} ({pct})")
    if counts["error"]:
        print(f"  {'error':11s}: {counts['error']}/{n}")
    decided = counts["correct"] + counts["wrong"]
    if decided:
        print(f"  accuracy   : {100 * counts['correct'] / decided:.0f}%  (correct / [correct+wrong])")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results_as_dicts(results), indent=2, ensure_ascii=False))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
