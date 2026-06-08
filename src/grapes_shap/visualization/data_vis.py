import numpy as np
from collections import Counter
from typing import List, Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
import seaborn as sns
from grapes_shap.config import Config, FIG_DIR
from grapes_shap.data.preprocessor import DDXPlusPreprocessor

def plot_dataset_overview(processed_train: List[Dict],
                          processed_val: List[Dict],
                          processed_test: List[Dict],
                          preprocessor: DDXPlusPreprocessor,
                          cfg: Config):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    CYAN = "#38bdf8"; TEAL = "#2dd4bf"; AMBER = "#fbbf24"
    ROSE = "#f87171"; VIOLET = "#a78bfa"; GREEN = "#4ade80"

    def ax_style(ax, title):
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8892a4", labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_edgecolor("#2a3348")
        ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    # 1. Split sizes
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ["Train", "Validation", "Test"]
    sizes  = [len(processed_train), len(processed_val), len(processed_test)]
    colors = [CYAN, TEAL, VIOLET]
    bars = ax1.bar(splits, sizes, color=colors, width=0.55, edgecolor="#0f1117", linewidth=0.8)
    for b, s in zip(bars, sizes):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 200,
                 f"{s:,}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)
    ax_style(ax1, "Dataset Split Sizes")
    ax1.set_ylabel("Samples", color="#8892a4", fontsize=8)

    # 2. Age distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ages = [r["obs"][0, 0] * 100 for r in processed_train[:5000]]
    ax2.hist(ages, bins=30, color=TEAL, edgecolor="#0f1117", linewidth=0.5, alpha=0.85)
    ax_style(ax2, "Patient Age Distribution")
    ax2.set_xlabel("Age (years)", color="#8892a4", fontsize=8)
    ax2.set_ylabel("Count", color="#8892a4", fontsize=8)

    # 3. Sex distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sexes  = [r["obs"][0, 1] for r in processed_train[:5000]]
    male   = sum(1 for s in sexes if s > 0.5)
    female = len(sexes) - male
    ax3.pie([male, female], labels=["Male", "Female"],
            colors=[CYAN, ROSE], autopct="%1.1f%%",
            textprops={"color": "#e2e8f0", "fontsize": 8},
            wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.8})
    ax_style(ax3, "Sex Distribution")

    # 4. Pathology distribution (top 15)
    ax4 = fig.add_subplot(gs[0, 3])
    path_ids = [r["pathology_id"] for r in processed_train[:10000]]
    path_counts = Counter(path_ids)
    top15 = path_counts.most_common(15)
    inv_vocab = {v: k for k, v in preprocessor.pathology_vocab.items()}
    labels = [inv_vocab.get(p, f"P{p}")[:18] for p, _ in top15]
    counts = [c for _, c in top15]
    ax4.barh(labels[::-1], counts[::-1], color=AMBER, edgecolor="#0f1117", linewidth=0.5)
    ax_style(ax4, "Top-15 Pathologies (Train)")
    ax4.set_xlabel("Count", color="#8892a4", fontsize=8)
    ax4.tick_params(axis="y", labelsize=6.5)

    # 5. Outcome probability heatmap
    ax5 = fig.add_subplot(gs[1, 0:2])
    outcomes_arr = np.array([r["outcomes"] for r in processed_train[:500]])
    im = ax5.imshow(outcomes_arr[:60].T, aspect="auto", cmap="viridis")
    ax5.set_xlabel("Patient index", color="#8892a4", fontsize=8)
    ax5.set_ylabel("Diagnosis rank", color="#8892a4", fontsize=8)
    ax5.set_yticks(range(5))
    ax5.set_yticklabels(["Dx-1","Dx-2","Dx-3","Dx-4","Dx-5"], color="#8892a4", fontsize=7)
    plt.colorbar(im, ax=ax5).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
    ax_style(ax5, "Differential Diagnosis Probability Heatmap (first 60 patients)")

    # 6. Symptom activation frequency
    ax6 = fig.add_subplot(gs[1, 2:4])
    sym_slots = np.array([r["obs"][0, 2:34] for r in processed_train[:5000]])
    sym_freq  = sym_slots.mean(axis=0)
    ax6.bar(range(len(sym_freq)), sym_freq, color=VIOLET, edgecolor="#0f1117", linewidth=0.3)
    ax_style(ax6, "Symptom Feature Activation Frequency (Train Set)")
    ax6.set_xlabel("Symptom slot index", color="#8892a4", fontsize=8)
    ax6.set_ylabel("Activation rate", color="#8892a4", fontsize=8)

    # 7. Trajectory length histogram
    ax7 = fig.add_subplot(gs[2, 0])
    ev_lengths = []
    for r in processed_train[:5000]:
        acts = r["actions"]
        nz = int((acts != 0).sum())
        ev_lengths.append(nz)
    ax7.hist(ev_lengths, bins=cfg.seq_len, color=GREEN, edgecolor="#0f1117", linewidth=0.5, alpha=0.85)
    ax_style(ax7, "Evidence Collection Length Distribution")
    ax7.set_xlabel("Steps with non-zero action", color="#8892a4", fontsize=8)
    ax7.set_ylabel("Count", color="#8892a4", fontsize=8)

    # 8. Observation feature correlation matrix
    ax8 = fig.add_subplot(gs[2, 1:3])
    obs_sample = np.array([r["obs"][0, :16] for r in processed_train[:1000]])
    corr = np.corrcoef(obs_sample.T)
    im2 = ax8.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax8.set_xticks(range(16)); ax8.set_yticks(range(16))
    ax8.set_xticklabels(range(16), fontsize=6); ax8.set_yticklabels(range(16), fontsize=6)
    ax8.tick_params(colors="#8892a4")
    plt.colorbar(im2, ax=ax8).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
    ax_style(ax8, "Observation Feature Correlation Matrix (first 16 features)")

    # 9. Dataset statistics summary
    ax9 = fig.add_subplot(gs[2, 3])
    ax9.set_facecolor("#161b27")
    ax9.axis("off")
    stats = [
        ("DDXPlus patients", f"{len(processed_train)+len(processed_val)+len(processed_test):,}"),
        ("Unique pathologies", f"{len(preprocessor.pathology_vocab)}"),
        ("Unique symptoms", f"{len(preprocessor.symptom_vocab)}"),
        ("Unique antecedents", f"{len(preprocessor.antecedent_vocab)}"),
        ("Evidence types", f"{len(preprocessor.evidence_vocab)}"),
        ("Trajectory length", f"{cfg.seq_len} steps"),
        ("Obs dimensions", f"{cfg.obs_dim}"),
        ("Outcome dimensions", f"{cfg.n_outcomes}"),
    ]
    for i, (k, v) in enumerate(stats):
        y = 0.93 - i * 0.115
        ax9.text(0.02, y, k, color="#8892a4", fontsize=8, transform=ax9.transAxes)
        ax9.text(0.98, y, v, color=CYAN, fontsize=8, fontweight="bold",
                 ha="right", transform=ax9.transAxes)
    ax9.set_title("Dataset Statistics", color="#e2e8f0", fontsize=9,
                  fontweight="bold", pad=8)
    ax9.spines[["top","right","left","bottom"]].set_edgecolor("#2a3348")

    fig.suptitle("GRAPES-SHAP — Data Exploration Dashboard (DDXPlus)",
                 color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01)

    path = FIG_DIR / "01_data_exploration.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
