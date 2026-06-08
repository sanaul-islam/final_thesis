#!/usr/bin/env python3
"""
Render Question-Answer pairs (GRAPES-SHAP vs Advanced Baseline RAG) as images.

Reads outputs/comparison_results.json and produces, under outputs/figures/qa_pairs/:
  • qa_<id>_<slug>.png  — one high-res (300 DPI) side-by-side card per prompt
  • qa_overview.png     — single contact-sheet summarising all prompts + scores

Run: python scripts/render_qa_pairs.py
"""

import sys
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "outputs" / "comparison_results.json"
OUT_DIR = ROOT / "outputs" / "figures" / "qa_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette (matches the project's dark research-grade theme)
BG = "#0f1117"
CARD = "#171a21"
CARD2 = "#1d2230"
CYAN = "#22d3ee"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
ROSE = "#fb7185"
VIOLET = "#a78bfa"
TEXT = "#e2e8f0"
MUTED = "#8892a4"


def slug(s: str, n: int = 28) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in s.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")[:n]


def wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(out)


def metric_line(side: dict) -> str:
    return (f"kw={side.get('kw_score', 0):.2f}   "
            f"cite={side.get('citations', 0)}   "
            f"struct={side.get('structure', 0):.2f}   "
            f"conf={side.get('confidence', 0):.2f}")


def render_card(rec: dict):
    q = rec["query"]
    cat = rec["category"]
    b = rec["baseline"]
    g = rec["grapes"]

    wrap_w = 64
    b_text = wrap(b["answer"], wrap_w)
    g_text = wrap(g["answer"], wrap_w)
    max_lines = max(b_text.count("\n"), g_text.count("\n")) + 1
    # Dynamic height based on content
    height = max(9.0, 3.2 + max_lines * 0.135)

    fig = plt.figure(figsize=(16, height), dpi=300)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Header
    ax.text(0.012, 0.975, f"[{rec['id']}] {cat}", color=CYAN,
            fontsize=15, fontweight="bold", va="top")
    ax.text(0.012, 0.945, "Clinical Query", color=MUTED, fontsize=9,
            va="top", fontweight="bold")
    ax.text(0.012, 0.928, wrap(q, 150), color=TEXT, fontsize=9.5, va="top")

    q_lines = wrap(q, 150).count("\n") + 1
    body_top = 0.928 - q_lines * 0.020 - 0.02

    col_w = 0.475
    gap = 0.02
    left_x = 0.012
    right_x = left_x + col_w + gap

    # Column backgrounds
    ax.add_patch(FancyBboxPatch(
        (left_x - 0.006, 0.03), col_w, body_top - 0.03,
        boxstyle="round,pad=0.004", linewidth=1.2,
        edgecolor=ROSE, facecolor=CARD, transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch(
        (right_x - 0.006, 0.03), col_w, body_top - 0.03,
        boxstyle="round,pad=0.004", linewidth=1.4,
        edgecolor=TEAL, facecolor=CARD2, transform=ax.transAxes))

    # Column titles
    ax.text(left_x + col_w / 2 - 0.006, body_top - 0.012,
            "Advanced Baseline RAG", color=ROSE, fontsize=12,
            fontweight="bold", ha="center", va="top")
    ax.text(right_x + col_w / 2 - 0.006, body_top - 0.012,
            "GRAPES-SHAP (ours)", color=TEAL, fontsize=12,
            fontweight="bold", ha="center", va="top")

    # Metric lines
    ax.text(left_x + 0.006, body_top - 0.038, metric_line(b),
            color=AMBER, fontsize=8.5, va="top", family="monospace")
    ax.text(right_x + 0.006, body_top - 0.038, metric_line(g),
            color=AMBER, fontsize=8.5, va="top", family="monospace")

    # Answers
    ax.text(left_x + 0.006, body_top - 0.062, b_text, color=TEXT,
            fontsize=7.6, va="top", family="monospace", linespacing=1.25)
    ax.text(right_x + 0.006, body_top - 0.062, g_text, color=TEXT,
            fontsize=7.6, va="top", family="monospace", linespacing=1.25)

    fname = OUT_DIR / f"qa_{rec['id']:02d}_{slug(cat)}.png"
    fig.savefig(fname, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return fname


def render_overview(data: dict):
    results = data["results"]
    summ = data.get("summary", {})

    fig = plt.figure(figsize=(16, 2.0 + len(results) * 0.62), dpi=300)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.975, "GRAPES-SHAP vs Advanced Baseline RAG  —  Q&A Scorecard",
            color=CYAN, fontsize=16, fontweight="bold", ha="center", va="top")

    n = len(results)
    top = 0.90
    bot = 0.10
    row_h = (top - bot) / n

    # Header row
    cols_x = [0.03, 0.30, 0.55, 0.80]
    headers = ["Clinical Scenario", "Baseline (kw / cite / struct)",
               "GRAPES (kw / cite / struct)", "Winner"]
    for x, h in zip(cols_x, headers):
        ax.text(x, top + 0.025, h, color=MUTED, fontsize=10,
                fontweight="bold", va="top")

    for i, r in enumerate(results):
        y = top - i * row_h
        b, g = r["baseline"], r["grapes"]
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch(
                (0.02, y - row_h + 0.008), 0.96, row_h - 0.008,
                boxstyle="round,pad=0.002", linewidth=0,
                facecolor=CARD, transform=ax.transAxes))
        ax.text(cols_x[0], y - 0.012, f"[{r['id']}] {r['category']}",
                color=TEXT, fontsize=9.5, va="top")
        ax.text(cols_x[1], y - 0.012,
                f"{b['kw_score']:.2f} / {b['citations']} / {b['structure']:.2f}",
                color=ROSE, fontsize=9.5, va="top", family="monospace")
        ax.text(cols_x[2], y - 0.012,
                f"{g['kw_score']:.2f} / {g['citations']} / {g['structure']:.2f}",
                color=TEAL, fontsize=9.5, va="top", family="monospace")
        g_better = (g["kw_score"] + g["structure"]) >= (b["kw_score"] + b["structure"])
        ax.text(cols_x[3], y - 0.012, "GRAPES" if g_better else "Baseline",
                color=(TEAL if g_better else ROSE), fontsize=9.5,
                fontweight="bold", va="top")

    # Summary footer
    if summ:
        bs, gs = summ.get("baseline", {}), summ.get("grapes", {})
        foot = (f"MEAN   |   Baseline: kw={bs.get('kw_score', 0):.2f} "
                f"cite={bs.get('citations', 0):.1f} struct={bs.get('structure', 0):.2f}"
                f"      GRAPES: kw={gs.get('kw_score', 0):.2f} "
                f"cite={gs.get('citations', 0):.1f} struct={gs.get('structure', 0):.2f} "
                f"shap={gs.get('mean_abs_shap', 0):.2f}")
        ax.text(0.5, 0.05, foot, color=AMBER, fontsize=10.5,
                ha="center", va="center", family="monospace")

    fname = OUT_DIR / "qa_overview.png"
    fig.savefig(fname, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return fname


def main():
    if not RESULTS.exists():
        sys.exit(f"Missing {RESULTS} — run scripts/compare_rag.py first.")
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    results = data["results"] if isinstance(data, dict) else data
    if isinstance(data, list):
        data = {"results": results, "summary": {}}

    print(f"Rendering {len(results)} Q&A pair cards...")
    for r in results:
        f = render_card(r)
        print(f"  saved {f.name}")
    f = render_overview(data)
    print(f"  saved {f.name}")
    print("Done.")


if __name__ == "__main__":
    main()
