import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grapes_shap.config import FIG_DIR

def plot_training_history(wm_hist, ens_hist):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f1117")
    CYAN = "#38bdf8"; TEAL = "#2dd4bf"; AMBER = "#fbbf24"

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8892a4", labelsize=8)
        ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color="#8892a4", fontsize=8)
        ax.set_ylabel(ylabel, color="#8892a4", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
        ax.grid(True, color="#2a3348", linewidth=0.4, alpha=0.5)
        ax.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    epochs_wm  = range(1, len(wm_hist["loss"]) + 1)
    axes[0].plot(epochs_wm, wm_hist["loss"],   color=CYAN,  lw=1.8, label="Total loss", marker="o", ms=3)
    axes[0].plot(epochs_wm, wm_hist["recon_loss"], color=AMBER, lw=1.5, linestyle="--", label="Recon loss", marker="s", ms=2.5)
    style(axes[0], "World Model Training Loss", "Epoch", "Loss")

    ax2 = axes[0].twinx()
    ax2.plot(epochs_wm, wm_hist["lr"], color=TEAL, lw=1.2, linestyle=":", label="LR", alpha=0.7)
    ax2.set_ylabel("Learning rate", color=TEAL, fontsize=8)
    ax2.tick_params(colors=TEAL, labelsize=7)

    epochs_ens = range(1, len(ens_hist["loss"]) + 1)
    axes[1].plot(epochs_ens, ens_hist["loss"], color=TEAL, lw=1.8, marker="o", ms=3, label="NLL loss")
    style(axes[1], "Deep Ensemble Training Loss", "Epoch", "NLL Loss")

    # Loss decay rate
    wm_smooth  = pd.Series(wm_hist["loss"]).ewm(span=3).mean()
    ens_smooth = pd.Series(ens_hist["loss"]).ewm(span=3).mean()
    axes[2].plot(range(1, len(wm_smooth)+1),  wm_smooth,  color=CYAN,  lw=1.8, label="WM (smoothed)")
    axes[2].plot(range(1, len(ens_smooth)+1), ens_smooth, color=TEAL,  lw=1.8, label="Ensemble (smoothed)")
    style(axes[2], "Training Convergence (EMA smoothed)", "Epoch", "Loss")

    fig.suptitle("GRAPES-SHAP — Training History", color="#e2e8f0", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "02_training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
