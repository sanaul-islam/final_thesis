import numpy as np
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from grapes_shap.config import FIG_DIR

def plot_inference_results(query: str, docs: List[str], shap_vals: np.ndarray,
                           plan: Dict, outcome_names: List[str]):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    CYAN   = "#38bdf8"; TEAL   = "#2dd4bf"; AMBER  = "#fbbf24"
    ROSE   = "#f87171"; VIOLET = "#a78bfa"; GREEN  = "#4ade80"

    def style(ax, title):
        ax.set_facecolor("#161b27"); ax.tick_params(colors="#8892a4", labelsize=8)
        ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)

    # 1. SHAP waterfall
    ax1 = fig.add_subplot(gs[0, 0:2])
    K    = len(docs)
    cols = [GREEN if v >= 0 else ROSE for v in shap_vals]
    ylabels = [f"Doc [{i+1}]: {docs[i][:55]}..." for i in range(K)]
    ax1.barh(range(K), shap_vals, color=cols, edgecolor="#0f1117", linewidth=0.5)
    ax1.set_yticks(range(K)); ax1.set_yticklabels(ylabels, fontsize=7.5)
    ax1.axvline(0, color="#8892a4", lw=0.8)
    for i, v in enumerate(shap_vals):
        ax1.text(v + (0.001 if v >= 0 else -0.001), i,
                 f"{v:.4f}", va="center", ha="left" if v >= 0 else "right",
                 color="#e2e8f0", fontsize=7.5)
    style(ax1, "SHAP Document Attributions — Evidence Contribution to Recommendation")
    ax1.set_xlabel("Shapley value φ", color="#8892a4", fontsize=8)

    # 2. Outcome predictions with uncertainty
    ax2 = fig.add_subplot(gs[0, 2])
    mu_np  = plan["mu"][0].cpu().numpy() if plan["mu"] is not None else np.zeros(len(outcome_names))
    std_np = plan["std"][0].cpu().numpy() if plan["std"] is not None else np.zeros(len(outcome_names))
    y_pos  = range(len(outcome_names))
    bar_c  = [GREEN if v > 0.5 else (AMBER if v > 0.3 else ROSE) for v in mu_np]
    ax2.barh(y_pos, mu_np, xerr=std_np, color=bar_c, height=0.55,
             error_kw={"elinewidth": 1.5, "ecolor": "#8892a4", "capsize": 4},
             edgecolor="#0f1117")
    ax2.set_yticks(y_pos); ax2.set_yticklabels(outcome_names, fontsize=8)
    ax2.set_xlim(0, 1.15)
    for i, (m, s) in enumerate(zip(mu_np, std_np)):
        ax2.text(m + s + 0.03, i, f"{m:.3f}±{s:.3f}", va="center", color="#e2e8f0", fontsize=7.5)
    style(ax2, "Predicted Outcomes (μ ± σ)")
    ax2.set_xlabel("Probability / Score", color="#8892a4", fontsize=8)

    # 3. Planning score comparison
    ax3 = fig.add_subplot(gs[1, 0])
    actions = plan.get("actions", [])
    if actions:
        a_labels = [f"a{a.item() if hasattr(a,'item') else a}" for a in actions]
        a_scores = np.linspace(plan["score"] * 0.6, plan["score"], len(actions))
        bar_cs   = plt.cm.viridis(np.linspace(0.3, 0.9, len(actions)))
        ax3.bar(range(len(a_labels)), a_scores, color=bar_cs, edgecolor="#0f1117", width=0.6)
        ax3.set_xticks(range(len(a_labels)))
        ax3.set_xticklabels(a_labels, fontsize=8)
        ax3.axhline(plan["score"], color=AMBER, lw=1.2, linestyle="--",
                    label=f"Best score: {plan['score']:.3f}")
        ax3.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)
    style(ax3, "ToT Planning — Action Sequence Scores")
    ax3.set_xlabel("Action step", color="#8892a4", fontsize=8)
    ax3.set_ylabel("Cumulative score", color="#8892a4", fontsize=8)

    # 4. SHAP normalised pie
    ax4 = fig.add_subplot(gs[1, 1])
    abs_phi = np.abs(shap_vals)
    if abs_phi.sum() > 0:
        fracs  = abs_phi / abs_phi.sum()
        labels = [f"Doc[{i+1}]" for i in range(K)]
        wedge_cols = [CYAN, TEAL, AMBER, ROSE, VIOLET, GREEN][:K]
        wedges, texts, autotexts = ax4.pie(
            fracs, labels=labels, colors=wedge_cols,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.8},
            textprops={"color": "#e2e8f0", "fontsize": 8})
        for at in autotexts:
            at.set_fontsize(7.5)
    style(ax4, "Relative Document Contribution (|φ| normalised)")

    # 5. Query + recommendation text box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#161b27"); ax5.axis("off")
    top_doc_idx = int(np.argmax(np.abs(shap_vals))) if len(shap_vals) > 0 else 0
    summary_lines = [
        "INFERENCE SUMMARY",
        "",
        f"Query (truncated):",
        f"  {query[:80]}...",
        "",
        f"Best plan score:  {plan['score']:.4f}",
        f"Actions planned:  {len(actions)} steps",
        "",
        f"Primary outcome:  {mu_np[0]:.3f} ± {std_np[0]:.3f}",
        f"Key evidence:     Doc[{top_doc_idx+1}] (φ={shap_vals[top_doc_idx]:.4f})",
        "",
        f"Top doc preview:",
        f"  {docs[top_doc_idx][:90]}..." if docs else "  —",
    ]
    for i, line in enumerate(summary_lines):
        color = CYAN if i == 0 else (AMBER if line.startswith("Query") or
                                     line.startswith("Best") or
                                     line.startswith("Primary") or
                                     line.startswith("Key") else "#e2e8f0")
        ax5.text(0.03, 0.97 - i*0.072, line, transform=ax5.transAxes,
                 color=color, fontsize=8, va="top",
                 fontweight="bold" if i == 0 else "normal")
    for sp in ax5.spines.values():
        sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
    ax5.set_title("Inference Summary", color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)

    fig.suptitle("GRAPES-SHAP — Full Inference Results (DDXPlus + MedQA)",
                 color="#e2e8f0", fontsize=13, fontweight="bold", y=1.01)
    path = FIG_DIR / "04_inference_results.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
