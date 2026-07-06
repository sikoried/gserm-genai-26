#!/usr/bin/env python3
"""Generate a PDF report comparing two evaluation runs side by side.

Usage:
    .venv/Scripts/python.exe bin/generate_comparison_report.py
    .venv/Scripts/python.exe bin/generate_comparison_report.py --baseline data/eval-results/baseline_v2.json --current data/eval-results/split_model_v1.json --output data/eval-results/comparison.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

REPO = Path(__file__).resolve().parent.parent

COLORS = {
    "baseline": "#4472C4",
    "current": "#ED7D31",
    "green": "#70AD47",
    "red": "#FF4444",
    "yellow": "#FFC000",
    "gray": "#A5A5A5",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two eval runs as a PDF report.")
    p.add_argument("--baseline", default=str(REPO / "data" / "eval-results" / "baseline_v2.json"))
    p.add_argument("--current", default=str(REPO / "data" / "eval-results" / "split_model_v1.json"))
    p.add_argument("--output", default=str(REPO / "data" / "eval-results" / "comparison.pdf"))
    p.add_argument("--baseline-label", default="Baseline (smolagents)")
    p.add_argument("--current-label", default="Split-Model (local router)")
    return p.parse_args()


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct_change(old: float, new: float) -> str:
    if old == 0:
        return "N/A"
    change = (new - old) / old * 100
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.0f}%"


def arrow(old: float, new: float, lower_is_better: bool = False) -> str:
    if new < old:
        return "v" if lower_is_better else "v"
    if new > old:
        return "^" if not lower_is_better else "^"
    return "="


# ── Page 1: Title + side-by-side summary table ────────────────────────
def page_summary_table(pdf, bl, cur, bl_label, cur_label):
    modes = list(bl["summaries"].keys())
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Oracle Evaluation — Baseline vs. Split-Model Comparison",
                 fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.93, f"Modes: {', '.join(m.upper() for m in modes)}  |  "
             f"30 corpus-grounded questions", ha="center", fontsize=10, color="gray")

    for idx, mode in enumerate(modes):
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.axis("off")
        ax.set_title(mode.upper(), fontsize=13, fontweight="bold", pad=15)

        bs = bl["summaries"].get(mode, {})
        cs = cur["summaries"].get(mode, {})

        rows = [
            ("Correct", "correct", "d", False),
            ("Wrong", "wrong", "d", True),
            ("Orthogonal", "orthogonal", "d", True),
            ("Accuracy", "accuracy", "%", False),
            ("Avg Tokens", "avg_total_tokens", ".0f", True),
            ("Total Tokens", "total_tokens", ",d", True),
            ("Avg Latency (s)", "avg_elapsed_seconds", ".2f", True),
            ("Wall Time (s)", "wall_time_seconds", ".2f", True),
        ]

        row_labels = []
        cell_text = []
        for label, key, fmt, lower_better in rows:
            row_labels.append(label)
            bv = bs.get(key)
            cv = cs.get(key)
            if bv is None and cv is None:
                cell_text.append(["N/A", "N/A", ""])
                continue
            if fmt == "%":
                b_str = f"{bv:.0%}" if bv is not None else "N/A"
                c_str = f"{cv:.0%}" if cv is not None else "N/A"
                delta = pct_change(bv * 100 if bv else 0, cv * 100 if cv else 0) if bv and cv else ""
            else:
                b_str = f"{bv:{fmt}}" if bv is not None else "N/A"
                c_str = f"{cv:{fmt}}" if cv is not None else "N/A"
                delta = pct_change(float(bv or 0), float(cv or 0))
            cell_text.append([b_str, c_str, delta])

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=["Baseline", "Split", "Delta"],
            cellLoc="center", rowLoc="right", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(COLORS["baseline"] if c == 0 else
                                   COLORS["current"] if c == 1 else "#888888")
                cell.set_text_props(color="white", fontweight="bold")
            elif c == -1:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold", fontsize=7)
            elif c == 2:
                text = cell.get_text().get_text()
                if text.startswith("-"):
                    cell.set_text_props(color="#228B22")
                elif text.startswith("+"):
                    cell.set_text_props(color="#CC0000")
            else:
                cell.set_facecolor("white" if r % 2 else "#F8F8F8")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig)
    plt.close()


# ── Page 2: aRAG-focused comparison charts ────────────────────────────
def page_arag_charts(pdf, bl, cur, bl_label, cur_label):
    bs = bl["summaries"].get("a-rag", {})
    cs = cur["summaries"].get("a-rag", {})

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Agentic RAG — Baseline vs. Split-Model", fontsize=16, fontweight="bold")

    # Chart 1: Verdict comparison
    ax = axes[0, 0]
    categories = ["Correct", "Wrong", "Orthogonal", "Error"]
    keys = ["correct", "wrong", "orthogonal", "error"]
    x = np.arange(len(categories))
    width = 0.35
    b_vals = [bs.get(k, 0) for k in keys]
    c_vals = [cs.get(k, 0) for k in keys]
    ax.bar(x - width/2, b_vals, width, label=bl_label, color=COLORS["baseline"])
    ax.bar(x + width/2, c_vals, width, label=cur_label, color=COLORS["current"])
    ax.set_ylabel("Count")
    ax.set_title("Verdict Distribution", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Chart 2: Token usage
    ax = axes[0, 1]
    metrics = ["Avg Prompt", "Avg Completion", "Avg Total"]
    mkeys = ["avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens"]
    x = np.arange(len(metrics))
    b_vals = [bs.get(k, 0) for k in mkeys]
    c_vals = [cs.get(k, 0) for k in mkeys]
    ax.bar(x - width/2, b_vals, width, label=bl_label, color=COLORS["baseline"])
    ax.bar(x + width/2, c_vals, width, label=cur_label, color=COLORS["current"])
    ax.set_ylabel("Tokens")
    ax.set_title("Token Usage per Question", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Chart 3: Latency
    ax = axes[1, 0]
    metrics = ["Avg Latency", "Wall Time"]
    mkeys = ["avg_elapsed_seconds", "wall_time_seconds"]
    x = np.arange(len(metrics))
    b_vals = [bs.get(k, 0) for k in mkeys]
    c_vals = [cs.get(k, 0) for k in mkeys]
    bars_b = ax.bar(x - width/2, b_vals, width, label=bl_label, color=COLORS["baseline"])
    bars_c = ax.bar(x + width/2, c_vals, width, label=cur_label, color=COLORS["current"])
    for bar, val in zip(list(bars_b) + list(bars_c), b_vals + c_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Seconds")
    ax.set_title("Latency Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Chart 4: Total tokens
    ax = axes[1, 1]
    x = np.arange(1)
    b_val = bs.get("total_tokens", 0)
    c_val = cs.get("total_tokens", 0)
    bars_b = ax.bar(x - width/2, [b_val], width, label=bl_label, color=COLORS["baseline"])
    bars_c = ax.bar(x + width/2, [c_val], width, label=cur_label, color=COLORS["current"])
    ax.text(bars_b[0].get_x() + bars_b[0].get_width()/2, bars_b[0].get_height() + 1000,
            f"{b_val:,}", ha="center", va="bottom", fontsize=9)
    ax.text(bars_c[0].get_x() + bars_c[0].get_width()/2, bars_c[0].get_height() + 1000,
            f"{c_val:,}", ha="center", va="bottom", fontsize=9)
    reduction = (1 - c_val / b_val) * 100 if b_val else 0
    ax.set_title(f"Total Token Usage ({reduction:.0f}% reduction)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["a-rag"])
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()


# ── Page 3: All modes comparison charts ───────────────────────────────
def page_all_modes(pdf, bl, cur, bl_label, cur_label):
    modes = list(bl["summaries"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("All Modes — Baseline vs. Split-Model", fontsize=16, fontweight="bold")

    width = 0.35
    x = np.arange(len(modes))
    mode_labels = [m.upper() for m in modes]

    # Accuracy
    ax = axes[0, 0]
    b_acc = [bl["summaries"][m].get("accuracy") or 0 for m in modes]
    c_acc = [cur["summaries"][m].get("accuracy") or 0 for m in modes]
    ax.bar(x - width/2, [a*100 for a in b_acc], width, label=bl_label, color=COLORS["baseline"])
    ax.bar(x + width/2, [a*100 for a in c_acc], width, label=cur_label, color=COLORS["current"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Mode", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)

    # Avg tokens
    ax = axes[0, 1]
    b_tok = [bl["summaries"][m].get("avg_total_tokens", 0) for m in modes]
    c_tok = [cur["summaries"][m].get("avg_total_tokens", 0) for m in modes]
    ax.bar(x - width/2, b_tok, width, label=bl_label, color=COLORS["baseline"])
    ax.bar(x + width/2, c_tok, width, label=cur_label, color=COLORS["current"])
    ax.set_ylabel("Avg Tokens")
    ax.set_title("Avg Token Usage by Mode", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Avg latency
    ax = axes[1, 0]
    b_lat = [bl["summaries"][m].get("avg_elapsed_seconds", 0) for m in modes]
    c_lat = [cur["summaries"][m].get("avg_elapsed_seconds", 0) for m in modes]
    bars_b = ax.bar(x - width/2, b_lat, width, label=bl_label, color=COLORS["baseline"])
    bars_c = ax.bar(x + width/2, c_lat, width, label=cur_label, color=COLORS["current"])
    for bar, val in zip(list(bars_b) + list(bars_c), b_lat + c_lat):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Avg Latency by Mode", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Total tokens
    ax = axes[1, 1]
    b_tot = [bl["summaries"][m].get("total_tokens", 0) for m in modes]
    c_tot = [cur["summaries"][m].get("total_tokens", 0) for m in modes]
    ax.bar(x - width/2, b_tot, width, label=bl_label, color=COLORS["baseline"])
    ax.bar(x + width/2, c_tot, width, label=cur_label, color=COLORS["current"])
    ax.set_ylabel("Total Tokens")
    ax.set_title("Total Token Usage by Mode", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close()


# ── Page 4: Per-item aRAG comparison ──────────────────────────────────
def page_per_item(pdf, bl, cur, bl_label, cur_label):
    b_results = bl.get("results", {}).get("a-rag", [])
    c_results = cur.get("results", {}).get("a-rag", [])
    if not b_results or not c_results or len(b_results) != len(c_results):
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Agentic RAG — Per-Question Comparison", fontsize=14, fontweight="bold", pad=20)

    col_labels = ["ID", "Question",
                  "BL Verdict", "BL Tokens", "BL Time",
                  "SM Verdict", "SM Tokens", "SM Time"]

    cell_text = []
    for br, cr in zip(b_results, c_results):
        cell_text.append([
            br["id"],
            br["question"][:35],
            br["verdict"].upper(),
            str(br["total_tokens"]),
            f"{br['elapsed_seconds']:.1f}s",
            cr["verdict"].upper(),
            str(cr["total_tokens"]),
            f"{cr['elapsed_seconds']:.1f}s",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.3)

    verdict_colors = {"CORRECT": "#C6EFCE", "WRONG": "#FFC7CE",
                      "ORTHOGONAL": "#FFEB9C", "ERROR": "#D9D9D9"}
    verdict_cols = {2, 5}

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            if c in (2, 3, 4):
                cell.set_facecolor(COLORS["baseline"])
            elif c in (5, 6, 7):
                cell.set_facecolor(COLORS["current"])
            else:
                cell.set_facecolor("#555555")
            cell.set_text_props(color="white", fontweight="bold", fontsize=6)
        elif c in verdict_cols:
            text = cell.get_text().get_text()
            cell.set_facecolor(verdict_colors.get(text, "white"))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


# ── Page 5: Key findings ─────────────────────────────────────────────
def page_findings(pdf, bl, cur, bl_label, cur_label):
    bs = bl["summaries"].get("a-rag", {})
    cs = cur["summaries"].get("a-rag", {})

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.set_title("Key Findings — Split-Model aRAG", fontsize=16, fontweight="bold", pad=30)

    latency_change = (1 - cs.get("avg_elapsed_seconds", 0) / bs.get("avg_elapsed_seconds", 1)) * 100
    token_change = (1 - cs.get("total_tokens", 0) / bs.get("total_tokens", 1)) * 100
    wall_change = (1 - cs.get("wall_time_seconds", 0) / bs.get("wall_time_seconds", 1)) * 100

    b_acc = bs.get("accuracy")
    c_acc = cs.get("accuracy")
    b_acc_str = f"{b_acc:.0%}" if b_acc is not None else "N/A"
    c_acc_str = f"{c_acc:.0%}" if c_acc is not None else "N/A"

    findings = [
        f"LATENCY: {latency_change:.0f}% reduction in average latency "
        f"({bs.get('avg_elapsed_seconds', 0):.1f}s -> {cs.get('avg_elapsed_seconds', 0):.1f}s)",

        f"TOKENS: {token_change:.0f}% reduction in total token usage "
        f"({bs.get('total_tokens', 0):,} -> {cs.get('total_tokens', 0):,})",

        f"WALL TIME: {wall_change:.0f}% reduction "
        f"({bs.get('wall_time_seconds', 0):.0f}s -> {cs.get('wall_time_seconds', 0):.0f}s)",

        f"ACCURACY: {b_acc_str} -> {c_acc_str} "
        f"(on decided answers, excluding orthogonal)",

        f"CORRECT: {bs.get('correct', 0)} -> {cs.get('correct', 0)} correct answers",

        f"ORTHOGONAL: {bs.get('orthogonal', 0)} -> {cs.get('orthogonal', 0)} "
        f"(router sends some corpus questions to 'world', producing no-answer responses)",

        "TRADE-OFF: The split-model approach dramatically cuts cost and latency, "
        "but the local router sometimes misclassifies corpus questions as 'world', "
        "leading to more orthogonal (unanswered) responses.",

        "RECOMMENDATION: Tune the router prompt to bias toward RAG for borderline "
        "questions, or use a slightly larger local model for better classification.",
    ]

    y = 0.82
    for i, finding in enumerate(findings):
        style = "bold" if i < 3 else "normal"
        color = COLORS["green"] if i < 3 else "#333333"
        if i >= 4:
            color = "#555555"
        ax.text(0.08, y, f"{i+1}.", fontsize=11, fontweight="bold", color=color,
                transform=ax.transAxes, va="top")
        ax.text(0.12, y, finding, fontsize=10, fontweight=style, color=color,
                transform=ax.transAxes, va="top", wrap=True,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="#DDDDDD")
                if i < 3 else None)
        y -= 0.10

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()


def main() -> None:
    args = parse_args()
    bl = load(args.baseline)
    cur = load(args.current)

    with PdfPages(args.output) as pdf:
        page_summary_table(pdf, bl, cur, args.baseline_label, args.current_label)
        page_arag_charts(pdf, bl, cur, args.baseline_label, args.current_label)
        page_all_modes(pdf, bl, cur, args.baseline_label, args.current_label)
        page_per_item(pdf, bl, cur, args.baseline_label, args.current_label)
        page_findings(pdf, bl, cur, args.baseline_label, args.current_label)

    print(f"Comparison report saved to {args.output}")


if __name__ == "__main__":
    main()
