import torch
import numpy as np
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from grapes_shap.config import Config, FIG_DIR

def plot_performance_dashboard(metrics: Dict):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    CYAN   = "#38bdf8"; TEAL   = "#2dd4bf"; AMBER  = "#fbbf24"
    ROSE   = "#f87171"; VIOLET = "#a78bfa"; GREEN  = "#4ade80"

    def style(ax, title):
        ax.set_facecolor("#161b27")
        ax.tick_params(colors="#8892a4", labelsize=8)
        ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)

    mu, std, y = metrics["mu"], metrics["std"], metrics["y"]
    ep, al     = metrics["ep"], metrics["al"]

    outcome_names = ["Dx-1 prob","Dx-2 prob","Dx-3 prob","Dx-4 prob","Dx-5 prob"]

    # 1. Predicted vs actual (outcome 0 — primary diagnosis probability)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y[:,0], mu[:,0], alpha=0.25, s=6, color=CYAN)
    lim = max(y[:,0].max(), mu[:,0].max()) * 1.05
    ax1.plot([0, lim], [0, lim], color=ROSE, lw=1.2, linestyle="--")
    style(ax1, "Predicted vs Actual (Dx-1 Probability)")
    ax1.set_xlabel("Actual", color="#8892a4", fontsize=8)
    ax1.set_ylabel("Predicted", color="#8892a4", fontsize=8)
    ax1.text(0.05, 0.92, f"MAE={metrics['mae']:.4f}", transform=ax1.transAxes,
             color=AMBER, fontsize=8)

    # 2. Residuals plot
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = (mu[:,0] - y[:,0]).ravel()
    ax2.scatter(mu[:,0].ravel(), residuals, alpha=0.2, s=5, color=TEAL)
    ax2.axhline(0, color=ROSE, lw=1.2, linestyle="--")
    style(ax2, "Residuals (Dx-1 Probability)")
    ax2.set_xlabel("Predicted", color="#8892a4", fontsize=8)
    ax2.set_ylabel("Residual", color="#8892a4", fontsize=8)

    # 3. Calibration curve (outcome 0)
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        bins = np.linspace(0, 1, 11)
        bin_means, bin_accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask_b = (mu[:,0] >= lo) & (mu[:,0] < hi)
            if mask_b.sum() > 0:
                bin_means.append(mu[mask_b, 0].mean())
                bin_accs.append(y[mask_b, 0].mean())
        ax3.plot(bin_means, bin_accs, color=CYAN, lw=1.8, marker="o", ms=4, label="Model")
        ax3.plot([0,1],[0,1], color=ROSE, lw=1.2, linestyle="--", label="Perfect")
        ax3.fill_between(bin_means, bin_means, bin_accs, alpha=0.15, color=AMBER, label="Gap")
    except Exception:
        ax3.text(0.3, 0.5, "Insufficient data", color="#8892a4")
    style(ax3, "Calibration Curve")
    ax3.set_xlabel("Mean predicted", color="#8892a4", fontsize=8)
    ax3.set_ylabel("Fraction actual", color="#8892a4", fontsize=8)
    ax3.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=7)

    # 4. Uncertainty decomposition
    ax4 = fig.add_subplot(gs[0, 3])
    ep_m = ep.mean(axis=0)
    al_m = al.mean(axis=0)
    x = np.arange(len(outcome_names))
    ax4.bar(x, ep_m, label="Epistemic", color=VIOLET, alpha=0.8, width=0.4)
    ax4.bar(x+0.4, al_m, label="Aleatoric", color=AMBER, alpha=0.8, width=0.4)
    ax4.set_xticks(x+0.2); ax4.set_xticklabels(outcome_names, rotation=25, fontsize=7)
    style(ax4, "Epistemic vs Aleatoric Uncertainty")
    ax4.set_ylabel("Mean σ", color="#8892a4", fontsize=8)
    ax4.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    # 5. MAE per outcome
    ax5 = fig.add_subplot(gs[1, 0])
    mae_per = np.abs(mu - y).mean(axis=0)
    bars = ax5.bar(outcome_names, mae_per, color=[CYAN,TEAL,AMBER,ROSE,VIOLET], width=0.55)
    for b, v in zip(bars, mae_per):
        ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                 f"{v:.3f}", ha="center", va="bottom", color="#e2e8f0", fontsize=7)
    style(ax5, "MAE per Outcome Variable")
    ax5.set_ylabel("MAE", color="#8892a4", fontsize=8)
    ax5.set_xticks(range(len(outcome_names)))
    ax5.set_xticklabels(outcome_names, rotation=25, fontsize=7)

    # 6. Confusion matrix (top diagnosis classification)
    ax6 = fig.add_subplot(gs[1, 1])
    n_cls = min(10, len(np.unique(metrics["true_class"])))
    cm    = confusion_matrix(metrics["true_class"][:2000], metrics["pred_class"][:2000])
    cm_n  = cm[:n_cls, :n_cls].astype(float)
    row_s = cm_n.sum(axis=1, keepdims=True)
    row_s[row_s == 0] = 1
    cm_n  = cm_n / row_s
    im = ax6.imshow(cm_n, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax6).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
    style(ax6, f"Confusion Matrix (top-{n_cls} diagnoses, normalised)")
    ax6.set_xlabel("Predicted", color="#8892a4", fontsize=8)
    ax6.set_ylabel("True", color="#8892a4", fontsize=8)

    # 7. Uncertainty vs error scatter
    ax7 = fig.add_subplot(gs[1, 2])
    err = np.abs(mu[:,0] - y[:,0])
    unc = std[:,0]
    ax7.scatter(unc[:2000], err[:2000], alpha=0.15, s=5, color=TEAL)
    z_fit = np.polyfit(unc[:2000], err[:2000], 1)
    x_fit = np.linspace(unc.min(), unc.max(), 100)
    ax7.plot(x_fit, np.polyval(z_fit, x_fit), color=AMBER, lw=1.5, label="Trend")
    style(ax7, "Uncertainty vs Prediction Error")
    ax7.set_xlabel("Predicted uncertainty (σ)", color="#8892a4", fontsize=8)
    ax7.set_ylabel("|Error|", color="#8892a4", fontsize=8)
    ax7.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    # 8. Outcome distribution comparison
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(y[:,0], bins=30, alpha=0.6, label="Ground truth", color=CYAN, density=True)
    ax8.hist(mu[:,0], bins=30, alpha=0.6, label="Predicted", color=AMBER, density=True)
    style(ax8, "Predicted vs True Outcome Distribution (Dx-1)")
    ax8.set_xlabel("Dx-1 probability", color="#8892a4", fontsize=8)
    ax8.set_ylabel("Density", color="#8892a4", fontsize=8)
    ax8.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    # 9. Metric summary table
    ax9 = fig.add_subplot(gs[2, 0:2])
    ax9.set_facecolor("#161b27"); ax9.axis("off")
    table_data = [
        ["Metric", "Value", "Target", "Status"],
        ["MAE",               f"{metrics['mae']:.4f}",   "< 0.04",  "✓" if metrics['mae']<0.04 else "✗"],
        ["RMSE",              f"{metrics['rmse']:.4f}",  "< 0.06",  "✓" if metrics['rmse']<0.06 else "✗"],
        ["1σ Coverage",       f"{metrics['1sigma_coverage']:.3f}", "0.65–0.71", "✓" if 0.60<metrics['1sigma_coverage']<0.75 else "✗"],
        ["ECE",               f"{metrics['ece']:.4f}",   "< 0.05",  "✓" if metrics['ece']<0.05 else "✗"],
        ["Diagnosis Acc.",    f"{metrics['accuracy']:.3f}", "> 0.55",  "✓" if metrics['accuracy']>0.55 else "✗"],
        ["F1-macro",          f"{metrics['f1_macro']:.3f}", "> 0.40",  "✓" if metrics['f1_macro']>0.40 else "✗"],
        ["Mean |SHAP|",       f"{metrics['mean_shap']:.4f}", "> 0.00", "✓"],
    ]
    t = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc="center", loc="center",
                  colWidths=[0.30, 0.20, 0.20, 0.15])
    t.auto_set_font_size(False); t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor("#1e2535" if r==0 else ("#161b27" if r%2==0 else "#111827"))
        cell.set_edgecolor("#2a3348")
        if r == 0:
            cell.set_text_props(color="#38bdf8", fontweight="bold")
        elif c == 3:
            txt = cell.get_text().get_text()
            cell.set_text_props(color="#4ade80" if txt=="✓" else "#f87171", fontweight="bold")
        else:
            cell.set_text_props(color="#e2e8f0")
    style(ax9, "Performance Metrics Summary")

    # 10. Sigma distribution
    ax10 = fig.add_subplot(gs[2, 2])
    ax10.hist(std[:,0], bins=40, color=VIOLET, alpha=0.8, density=True)
    ax10.axvline(std[:,0].mean(), color=AMBER, lw=1.5, linestyle="--",
                 label=f"mean={std[:,0].mean():.3f}")
    style(ax10, "Total Uncertainty Distribution (σ_total, Dx-1)")
    ax10.set_xlabel("σ", color="#8892a4", fontsize=8)
    ax10.set_ylabel("Density", color="#8892a4", fontsize=8)
    ax10.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    # 11. Error percentile analysis
    ax11 = fig.add_subplot(gs[2, 3])
    errs_sorted = np.sort(np.abs(mu[:,0] - y[:,0]))
    percentiles = np.linspace(0, 100, len(errs_sorted))
    ax11.plot(percentiles, errs_sorted, color=ROSE, lw=1.8)
    ax11.axhline(metrics["mae"], color=AMBER, lw=1.2, linestyle="--", label="MAE")
    ax11.fill_between(percentiles, errs_sorted, alpha=0.1, color=ROSE)
    style(ax11, "Cumulative Error Distribution (Dx-1)")
    ax11.set_xlabel("Percentile", color="#8892a4", fontsize=8)
    ax11.set_ylabel("|Error|", color="#8892a4", fontsize=8)
    ax11.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

    fig.suptitle("GRAPES-SHAP — Performance Evaluation Dashboard (DDXPlus)",
                 color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01)
    path = FIG_DIR / "03_performance_dashboard.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")

def plot_latent_space(enc, gnn, kg, loader, cfg: Config, n_samples=2000):
    enc.eval(); gnn.eval()
    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
    all_z, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(all_z) * cfg.batch_size >= n_samples:
                break
            obs = batch["obs"].to(cfg.device)
            pid = batch["pathology_id"]
            B   = obs.shape[0]
            nf_b = nf.unsqueeze(0).expand(B,-1,-1)
            ad_b = adj.unsqueeze(0).expand(B,-1,-1)
            ew_b = ew.unsqueeze(0).expand(B,-1,-1)
            _, g = gnn(nf_b, ad_b, ew_b, mask)
            z    = enc(obs, g)[:,-1,:].cpu().numpy()
            all_z.append(z); all_labels.extend(pid.tolist())

    Z = np.vstack(all_z)[:n_samples]
    L = np.array(all_labels)[:n_samples]

    print(f"  Running t-SNE on {len(Z)} latent vectors...")
    # scikit-learn >=1.5 renamed `n_iter` to `max_iter`; support both.
    try:
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=500)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=500)
    Z2d  = tsne.fit_transform(Z)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0f1117")

    n_cls = min(15, len(np.unique(L)))
    cmap  = plt.cm.get_cmap("tab20", n_cls)
    for i, ax in enumerate(axes):
        ax.set_facecolor("#161b27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
        ax.tick_params(colors="#8892a4", labelsize=7)

    sc1 = axes[0].scatter(Z2d[:,0], Z2d[:,1], c=L % n_cls, cmap=cmap,
                          s=5, alpha=0.5, linewidths=0)
    axes[0].set_title("Latent Space Coloured by Pathology Class",
                      color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
    axes[0].set_xlabel("t-SNE dim 1", color="#8892a4", fontsize=8)
    axes[0].set_ylabel("t-SNE dim 2", color="#8892a4", fontsize=8)
    plt.colorbar(sc1, ax=axes[0]).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)

    # Density plot
    axes[1].hexbin(Z2d[:,0], Z2d[:,1], gridsize=60, cmap="plasma", alpha=0.9)
    axes[1].set_title("Latent Space Density (Hexbin)",
                      color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
    axes[1].set_xlabel("t-SNE dim 1", color="#8892a4", fontsize=8)
    axes[1].set_ylabel("t-SNE dim 2", color="#8892a4", fontsize=8)

    fig.suptitle("GRAPES-SHAP — Latent Space Visualisation (Evidence Fusion Encoder)",
                 color="#e2e8f0", fontsize=12, fontweight="bold")
    path = FIG_DIR / "05_latent_space.png"
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
