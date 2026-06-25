#!/usr/bin/env python3
"""Evaluate a set of RAG profiles and print a side-by-side comparison table.

Runs ``run_eval`` for each profile over the same query/answer set, writes the raw
per-profile results to ``data/eval/<profile>.json``, and prints a summary table
with verdict counts, accuracy, average token usage, and a retrieval-confidence
proxy — including deltas against the baseline profile (requirements §5).

Each profile is evaluated in a fresh subprocess so the per-process index cache is
isolated (important when ``--rebuild`` swaps the index between chunking
strategies, a rebuild boundary per §3.1).

Usage:
    .venv/Scripts/python bin/compare_rag.py
    .venv/Scripts/python bin/compare_rag.py --profiles rag rag_rerank rag_full
    .venv/Scripts/python bin/compare_rag.py --rebuild --max-docs 40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:  # the table uses a Δ glyph; force UTF-8 on cp1252 consoles (Windows)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from oracle.config import QAConfig  # noqa: E402
from oracle.eval.judge import JUDGE_MODEL  # noqa: E402

DEFAULT_PROFILES = [
    "rag", "rag_structural", "rag_multi", "rag_rerank", "rag_compress", "rag_full",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RAG profiles head-to-head.")
    p.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES,
                   help="config stems under configs/ (default: the 6 rag profiles)")
    p.add_argument("--pairs", default=str(REPO / "eval" / "qa_pairs.yaml"))
    p.add_argument("--out-dir", default=str(REPO / "data" / "eval"))
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--baseline", default="rag", help="profile used as the delta reference")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild the index before each distinct chunking strategy")
    p.add_argument("--max-docs", type=int, default=None, help="--max-docs for --rebuild")
    p.add_argument("--python", default=sys.executable, help="interpreter for subprocess runs")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    subprocess.run(cmd, check=True, env=env)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_results(items: list[dict]) -> dict:
    n = len(items)
    counts = {"correct": 0, "wrong": 0, "orthogonal": 0, "error": 0}
    for it in items:
        counts[it.get("verdict", "error")] = counts.get(it.get("verdict", "error"), 0) + 1
    decided = counts["correct"] + counts["wrong"]
    return {
        "n": n,
        **counts,
        "accuracy": (counts["correct"] / decided) if decided else 0.0,
        "avg_prompt": _avg([it.get("prompt_tokens", 0) for it in items]),
        "avg_completion": _avg([it.get("completion_tokens", 0) for it in items]),
        "avg_total": _avg([it.get("total_tokens", 0) for it in items]),
        "avg_top_score": _avg([it.get("top_score", 0.0) for it in items]),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    built_strategies: set[str] = set()

    for profile in args.profiles:
        cfg_path = REPO / "configs" / f"{profile}.yaml"
        if not cfg_path.exists():
            print(f"!! skipping {profile}: {cfg_path} not found")
            continue
        print(f"\n=== profile: {profile} ===")
        if args.rebuild:
            strategy = QAConfig.from_yaml(cfg_path).chunking.strategy
            # Rebuild once per distinct chunking strategy (a rebuild boundary).
            if strategy not in built_strategies:
                build_cmd = [args.python, str(REPO / "bin" / "build_index.py"),
                             "--config", str(cfg_path)]
                if args.max_docs:
                    build_cmd += ["--max-docs", str(args.max_docs)]
                _run(build_cmd)
                built_strategies.add(strategy)

        out_json = out_dir / f"{profile}.json"
        eval_cmd = [args.python, str(REPO / "bin" / "run_eval.py"),
                    "--config", str(cfg_path), "--pairs", args.pairs,
                    "--judge-model", args.judge_model, "--out", str(out_json)]
        try:
            _run(eval_cmd)
        except subprocess.CalledProcessError as exc:
            print(f"!! {profile} eval failed: {exc}")
            continue
        items = json.loads(out_json.read_text(encoding="utf-8"))
        summaries[profile] = summarize_results(items)

    _print_table(summaries, args.baseline)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.json'}")


def _print_table(summaries: dict[str, dict], baseline: str) -> None:
    if not summaries:
        print("\nNo profiles evaluated.")
        return
    base = summaries.get(baseline)
    header = (f"\n{'profile':16s} {'cor':>4s} {'wrg':>4s} {'ort':>4s} {'err':>4s} "
              f"{'acc':>6s} {'p_tok':>7s} {'c_tok':>7s} {'tot':>7s} {'Δtot':>7s} {'top_s':>6s}")
    print(header)
    print("-" * len(header))
    for profile, s in summaries.items():
        dtot = (s["avg_total"] - base["avg_total"]) if base else 0.0
        dtot_str = f"{dtot:+.0f}" if base and profile != baseline else "—"
        print(f"{profile:16s} {s['correct']:4d} {s['wrong']:4d} {s['orthogonal']:4d} "
              f"{s['error']:4d} {s['accuracy']*100:5.0f}% {s['avg_prompt']:7.0f} "
              f"{s['avg_completion']:7.0f} {s['avg_total']:7.0f} {dtot_str:>7s} "
              f"{s['avg_top_score']:6.3f}")
    print("\nLegend: cor/wrg/ort/err = correct/wrong/orthogonal/error; "
          "acc = correct/(correct+wrong); p_tok/c_tok/tot = avg prompt/completion/"
          "total tokens; Δtot = total-token delta vs baseline; top_s = avg top "
          "retrieval score (proxy for retrieval confidence).")


if __name__ == "__main__":
    main()
