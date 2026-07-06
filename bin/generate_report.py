#!/usr/bin/env python3
"""Generate a PDF report from an eval-results JSON file.

Usage:
    .venv/Scripts/python.exe bin/generate_report.py
    .venv/Scripts/python.exe bin/generate_report.py --input data/eval-results/baseline.json --output data/eval-results/baseline.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PDF report from eval results.")
    p.add_argument("--input", default=str(REPO / "data" / "eval-results" / "baseline.json"))
    p.add_argument("--output", default=None, help="output PDF path (default: same name as input)")
    return p.parse_args()


def load_data(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_summary_table(ax, summaries: dict) -> None:
    ax.axis("off")
    ax.set_title("KPI Summary Table", fontsize=14, fontweight="bold", pad=20)

    modes = list(summaries.keys())
    rows = [
        ("Questions", "n", "d"),
        ("Correct", "correct", "d"),
        ("Wrong", "wrong", "d"),
        ("Orthogonal", "orthogonal", "d"),
        ("Errors", "error", "d"),
        ("Accuracy", "accuracy", "%"),
        ("Avg Prompt Tokens", "avg_prompt_tokens", ".0f"),
        ("Avg Completion Tokens", "avg_completion_tokens", ".0f"),
        ("Avg Total Tokens", "avg_total_tokens", ".0f"),
        ("Total Tokens", "total_tokens", ",d"),
        ("Avg Latency (s)", "avg_elapsed_seconds", ".2f"),
        ("Total Latency (s)", "total_elapsed_seconds", ".2f"),
        ("Wall Time (s)", "wall_time_seconds", ".2f"),
    ]

    cell_text = []
    row_labels = []
    for label, key, fmt in rows:
        row_labels.append(label)
        row = []
        for mode in modes:
            val = summaries[mode][key]
            if val is None:
                row.append("N/A")
            elif fmt == "%":
                row.append(f"{val:.0%}")
            else:
                row.append(f"{val:{fmt}}")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[m.upper() for m in modes],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#D9E2F3")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F2F2F2" if r % 2 == 0 else "white")


def make_accuracy_chart(ax, summaries: dict) -> None:
    modes = list(summaries.keys())
    colors = ["#4472C4", "#ED7D31", "#70AD47"]

    for mode in modes:
        s = summaries[mode]
        total = s["correct"] + s["wrong"] + s["orthogonal"] + s["error"]
        if total == 0:
            continue

    x = np.arange(len(modes))
    width = 0.2
    categories = ["correct", "wrong", "orthogonal", "error"]
    cat_colors = ["#70AD47", "#FF4444", "#FFC000", "#A5A5A5"]

    for i, (cat, col) in enumerate(zip(categories, cat_colors)):
        vals = [summaries[m][cat] for m in modes]
        ax.bar(x + i * width, vals, width, label=cat.capitalize(), color=col)

    ax.set_xlabel("Mode")
    ax.set_ylabel("Count")
    ax.set_title("Verdict Distribution by Mode", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.legend()
    ax.set_ylim(bottom=0)


def make_token_chart(ax, summaries: dict) -> None:
    modes = list(summaries.keys())
    x = np.arange(len(modes))
    width = 0.35

    prompt = [summaries[m]["avg_prompt_tokens"] for m in modes]
    completion = [summaries[m]["avg_completion_tokens"] for m in modes]

    ax.bar(x - width / 2, prompt, width, label="Prompt", color="#4472C4")
    ax.bar(x + width / 2, completion, width, label="Completion", color="#ED7D31")

    ax.set_xlabel("Mode")
    ax.set_ylabel("Avg Tokens per Question")
    ax.set_title("Average Token Usage by Mode", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.legend()
    ax.set_ylim(bottom=0)


def make_latency_chart(ax, summaries: dict) -> None:
    modes = list(summaries.keys())
    x = np.arange(len(modes))
    colors = ["#4472C4", "#ED7D31", "#70AD47"]

    avg_latency = [summaries[m]["avg_elapsed_seconds"] for m in modes]
    bars = ax.bar(x, avg_latency, color=colors[:len(modes)])

    for bar, val in zip(bars, avg_latency):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Mode")
    ax.set_ylabel("Avg Latency (seconds)")
    ax.set_title("Average Latency per Question", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in modes])
    ax.set_ylim(bottom=0)


def make_per_item_table(ax, results: dict) -> None:
    ax.axis("off")
    ax.set_title("Per-Question Results", fontsize=14, fontweight="bold", pad=20)

    modes = list(results.keys())
    first_mode = modes[0]
    questions = [r["id"] for r in results[first_mode]]

    col_labels = ["ID", "Question"]
    for mode in modes:
        col_labels.extend([f"{mode.upper()} Verdict", f"{mode.upper()} Tokens", f"{mode.upper()} Time"])

    cell_text = []
    for i, qid in enumerate(questions):
        row = [qid, results[first_mode][i]["question"][:40]]
        for mode in modes:
            r = results[mode][i]
            row.append(r["verdict"].upper())
            row.append(str(r["total_tokens"]))
            row.append(f"{r['elapsed_seconds']:.2f}s")
        cell_text.append(row)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.5)

    verdict_cols = {j for j, lbl in enumerate(col_labels) if "Verdict" in lbl}
    verdict_colors = {"CORRECT": "#C6EFCE", "WRONG": "#FFC7CE", "ORTHOGONAL": "#FFEB9C", "ERROR": "#D9D9D9"}

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold", fontsize=6)
        elif c in verdict_cols:
            text = cell.get_text().get_text()
            cell.set_facecolor(verdict_colors.get(text, "white"))


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    summaries = data["summaries"]
    results = data["results"]

    output = args.output or str(Path(args.input).with_suffix(".pdf"))

    with PdfPages(output) as pdf:
        # Page 1: Title + summary table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.suptitle("Oracle Evaluation Report — Baseline", fontsize=18, fontweight="bold", y=0.97)
        fig.text(0.5, 0.93, f"Model: mistralai/Mistral-Medium-3.5-128B  |  Questions: {summaries[list(summaries.keys())[0]]['n']}",
                 ha="center", fontsize=10, color="gray")
        make_summary_table(ax, summaries)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        pdf.savefig(fig)
        plt.close()

        # Page 2: Charts
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("KPI Charts", fontsize=16, fontweight="bold")

        make_accuracy_chart(axes[0, 0], summaries)
        make_token_chart(axes[0, 1], summaries)
        make_latency_chart(axes[1, 0], summaries)

        # Total tokens bar chart
        modes = list(summaries.keys())
        x = np.arange(len(modes))
        colors = ["#4472C4", "#ED7D31", "#70AD47"]
        totals = [summaries[m]["total_tokens"] for m in modes]
        bars = axes[1, 1].bar(x, totals, color=colors[:len(modes)])
        for bar, val in zip(bars, totals):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                            f"{val:,}", ha="center", va="bottom", fontsize=9)
        axes[1, 1].set_xlabel("Mode")
        axes[1, 1].set_ylabel("Total Tokens")
        axes[1, 1].set_title("Total Token Usage", fontsize=12, fontweight="bold")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([m.upper() for m in modes])
        axes[1, 1].set_ylim(bottom=0)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # Page 3: Per-item results
        fig, ax = plt.subplots(figsize=(11, 8.5))
        make_per_item_table(ax, results)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
