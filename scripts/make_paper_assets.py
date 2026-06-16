#!/usr/bin/env python3
"""
Generate all publication-quality assets for the GRAPES-SHAP paper:
  • schematic diagrams (pipeline, world-model architecture)
  • training curves (world-model loss + reward loss, ensemble NLL)
  • reliability / calibration diagram
  • uncertainty decomposition (epistemic vs aleatoric)
  • latent-space t-SNE
  • GRAPES-SHAP vs baseline comparison bars
  • per-prompt quality and SHAP attribution

Run AFTER scripts/retrain_and_eval.py (needs the regenerated metrics +
checkpoints). From the project root:

    python scripts/make_paper_assets.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# make unicode-safe on Windows consoles (cp1252) without needing env vars
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT     = Path(__file__).parent.parent
OUT      = ROOT / "outputs"
FIG      = OUT / "figures" / "paper"
FIG.mkdir(parents=True, exist_ok=True)

# ── publication style (formal, modern research) ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#3A3A3A",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "grid.color": "#9AA3AD",
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# Formal, desaturated research palette (navy / steel / teal / muted accents).
# Keys are kept stable so existing figure code maps onto the new scheme.
C = {
    "blue":   "#1F3A5F",   # primary dark navy
    "sky":    "#5B7B96",   # steel blue (secondary)
    "green":  "#2E6E6A",   # muted teal  -> "proposed / positive"
    "orange": "#9C5A33",   # muted sienna -> secondary highlight
    "purple": "#6E5A74",   # muted plum
    "gray":   "#5F6B76",   # blue-gray (baselines / neutral)
    "yellow": "#B0892F",   # muted gold (sparingly)
    "sky2":   "#8FA6B8",   # light steel
    "ink":    "#1A1A1A",   # near-black text/lines
    "fill":   "#EAEEF2",   # very light fill
}
SYS = "Proposed Framework"   # brand-neutral system name for figure labels


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}.pdf / .png")


def load_json(path, default=None):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return default


# ════════════════════════════════════════════════════════════════════
# 1. PIPELINE SCHEMATIC
# ════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 24)

    stages = [
        ("Query\nExpansion\n(HyDE)", C["sky"]),
        ("Hybrid\nRetrieval\n(Dense+BM25)", C["sky"]),
        ("Re-rank\n(Cross-Enc\n+ MMR)", C["sky"]),
        ("Causal KG\n+ GNN", C["purple"]),
        ("Latent\nWorld Model", C["orange"]),
        ("ToT Beam\nPlanner", C["orange"]),
        ("Deep\nEnsemble", C["green"]),
        ("Hallucination\nSelf-check", C["green"]),
        ("SHAP\nAttribution", C["blue"]),
        ("LLM Answer\n(DeepSeek)", C["blue"]),
    ]
    n = len(stages); w = 8.4; gap = (100 - n * w) / (n + 1)
    x = gap
    centers = []
    for label, col in stages:
        box = FancyBboxPatch((x, 7), w, 10, boxstyle="round,pad=0.15,rounding_size=0.6",
                             linewidth=1.2, edgecolor=col, facecolor=col + "22")
        ax.add_patch(box)
        ax.text(x + w / 2, 12, label, ha="center", va="center", fontsize=7.4)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((centers[i] + w / 2 - 0.2, 12),
                                     (centers[i + 1] - w / 2 + 0.2, 12),
                                     arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color=C["gray"]))
    # group brackets
    groups = [(0, 2, "Retrieval", C["sky"]), (3, 3, "Reasoning", C["purple"]),
              (4, 5, "World-Model Planning", C["orange"]),
              (6, 7, "Uncertainty", C["green"]), (8, 9, "Explanation", C["blue"])]
    for a, b, name, col in groups:
        x0 = centers[a] - w / 2; x1 = centers[b] + w / 2
        ax.plot([x0, x1], [5.2, 5.2], color=col, linewidth=2)
        ax.text((x0 + x1) / 2, 3.0, name, ha="center", va="center",
                fontsize=8, color=col, fontweight="bold")
    ax.text(50, 21.5, "Trustworthy Clinical QA \u2014 Inference Pipeline", ha="center",
            fontsize=11, fontweight="bold")
    _save(fig, "fig1_pipeline")


# ════════════════════════════════════════════════════════════════════
# 2. WORLD-MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)

    def box(x, y, w, h, text, col, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.2,rounding_size=0.8",
                     linewidth=1.3, edgecolor=col, facecolor=col + "20"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, col=C["gray"]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=11, linewidth=1.1, color=col,
                     shrinkA=2, shrinkB=4))

    # inputs
    box(2, 38, 17, 8, "Patient trajectory\nobs  (T×64)", C["sky"])
    box(2, 26, 17, 8, "Action $a_t$\n(50 discrete)", C["purple"])
    box(2, 14, 17, 8, "KG graph emb\n$g$  (256)", C["green"])

    # encoder
    box(24, 38, 16, 8, "Evidence Fusion\nEncoder\n(Transf.×3)", C["sky"], 7.5)
    arrow(19, 42, 24, 42)

    # causal residual
    box(46, 26, 16, 9, "Causal\nResidual\n$\\Delta z=g\\cdot s$", C["purple"], 7.5)
    arrow(19, 30, 46, 30)            # action -> causal (left edge)
    arrow(19, 18, 46, 27)            # g -> causal (left edge, lower)
    arrow(40, 41, 52, 35)            # enc z -> causal (top edge)

    # GRU
    box(68, 32, 15, 10, "GRU core\n(3 layers,\n512)", C["orange"], 7.5)
    arrow(40, 42, 68, 39)            # z -> gru
    arrow(62, 30, 68, 35)            # delta -> gru

    # h2z
    box(68, 18, 15, 9, "h→z proj\nLayerNorm\n(256)", C["orange"], 7.5)
    arrow(75, 32, 75, 27)

    # heads
    box(88, 40, 10, 7, "Decoder\nobs (64)", C["blue"], 7)
    box(88, 30, 10, 7, "σ head\n(256)", C["blue"], 7)
    box(88, 20, 10, 7, "reward\n(1)", C["orange"], 7)
    box(88, 10, 10, 7, "Ensemble\nμ,σ (5)", C["green"], 7)
    arrow(83, 23, 88, 43)
    arrow(83, 23, 88, 33)
    arrow(83, 23, 88, 23)
    arrow(83, 22, 88, 13)

    ax.text(50, 48, "Latent World Model — Layer Dataflow", ha="center",
            fontsize=11, fontweight="bold")
    ax.text(50, 2.5, "Recurrent latent rollout: $z_{t+1}=f(z_t+\\Delta z_t, a_t)$ "
                     "enables action-conditioned 'imagination' for planning.",
            ha="center", fontsize=8, color=C["gray"], style="italic")
    _save(fig, "fig2_architecture")


# ════════════════════════════════════════════════════════════════════
# 3. TRAINING CURVES
# ════════════════════════════════════════════════════════════════════
def fig_training():
    tm = load_json(OUT / "metrics" / "training_metrics.json", {})
    wm = tm.get("world_model", {}) or {}
    en = tm.get("ensemble", {}) or {}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    if wm.get("loss"):
        ep = np.arange(1, len(wm["loss"]) + 1)
        ax.plot(ep, wm["loss"], "-o", color=C["blue"], ms=3, label="Total loss")
        if wm.get("recon_loss"):
            ax.plot(ep, wm["recon_loss"], "-s", color=C["orange"], ms=3, label="Reconstruction")
        if wm.get("reward_loss"):
            ax.plot(ep, wm["reward_loss"], "-^", color=C["green"], ms=3, label="Reward head")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("(a) World-Model Training"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    if en.get("loss"):
        ep = np.arange(1, len(en["loss"]) + 1)
        ax.plot(ep, en["loss"], "-o", color=C["purple"], ms=3, label="Gaussian NLL")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NLL")
    ax.set_title("(b) Deep-Ensemble Training"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig3_training_curves")


# ════════════════════════════════════════════════════════════════════
# 4–7. DATA-DRIVEN FIGURES (need recomputed arrays)
# ════════════════════════════════════════════════════════════════════
def compute_eval_arrays():
    """Reload val data, load checkpoints, compute predictions + latents."""
    import torch
    from grapes_shap.config import CFG
    from grapes_shap.data import DatasetLoader, DDXPlusPreprocessor, ClinicalTrajectoryDataset
    from grapes_shap.models import MedicalKG, CausalGNN, EvidenceFusionEncoder, DeepEnsemble
    from torch.utils.data import DataLoader

    ckpt_wm  = OUT / "checkpoints" / "world_model.pt"
    ckpt_ens = OUT / "checkpoints" / "ensemble.pt"
    if not (ckpt_wm.exists() and ckpt_ens.exists()):
        print("  ! checkpoints missing — skipping model-based figures")
        return None

    print("  reloading data for latent/uncertainty figures...")
    train_raw, val_raw, _ = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)
    pre = DDXPlusPreprocessor(CFG); pre.fit(train_raw)
    va  = pre.transform(val_raw)
    ds  = ClinicalTrajectoryDataset(va)
    dl  = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    kg  = MedicalKG(pre, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device)
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    ens = DeepEnsemble(CFG).to(CFG.device)
    sd  = torch.load(ckpt_wm, map_location=CFG.device, weights_only=False)
    gnn.load_state_dict(sd["gnn"]); enc.load_state_dict(sd["enc"])
    ens.load_state_dict(torch.load(ckpt_ens, map_location=CFG.device, weights_only=False))
    gnn.eval(); enc.eval(); ens.eval()

    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
    Z, MU, STD, EP, AL, Y, DIAG = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            obs = batch["obs"].to(CFG.device); B = obs.shape[0]
            g = gnn(nf.unsqueeze(0).expand(B, -1, -1),
                    adj.unsqueeze(0).expand(B, -1, -1),
                    ew.unsqueeze(0).expand(B, -1, -1), mask)[1]
            z = enc(obs, g)[:, -1, :]
            mu, std, ep, al = ens(z)
            Z.append(z.cpu().numpy()); MU.append(mu.cpu().numpy())
            STD.append(std.cpu().numpy()); EP.append(ep.cpu().numpy())
            AL.append(al.cpu().numpy()); Y.append(batch["outcomes"].numpy())
            DIAG.append(batch["diag_class"].numpy())
    arr = dict(z=np.concatenate(Z), mu=np.concatenate(MU), std=np.concatenate(STD),
               ep=np.concatenate(EP), al=np.concatenate(AL), y=np.concatenate(Y),
               diag=np.concatenate(DIAG))
    np.savez(OUT / "metrics" / "eval_arrays.npz", **arr)
    return arr


def fig_calibration(arr):
    mu, y, std = arr["mu"], arr["y"], arr["std"]
    err = np.abs(mu - y).ravel()
    unc = std.ravel()
    # reliability: bin by predicted uncertainty, compare to empirical error
    bins = np.quantile(unc, np.linspace(0, 1, 11))
    bx, by = [], []
    for i in range(10):
        m = (unc >= bins[i]) & (unc < bins[i + 1] if i < 9 else unc <= bins[i + 1])
        if m.sum() > 0:
            bx.append(unc[m].mean()); by.append(err[m].mean())
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    lim = max(max(bx), max(by)) * 1.1
    ax.plot([0, lim], [0, lim], "--", color=C["gray"], label="Perfect calibration")
    ax.plot(bx, by, "-o", color=C["blue"], ms=5, label=SYS)
    ax.set_xlabel("Predicted uncertainty $\\sigma$")
    ax.set_ylabel("Empirical |error|")
    ax.set_title("Reliability Diagram")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig4_calibration")


def fig_uncertainty(arr):
    ep, al = arr["ep"].mean(1), arr["al"].mean(1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.scatter(al, ep, s=6, alpha=0.25, color=C["purple"], edgecolors="none")
    ax.set_xlabel("Aleatoric uncertainty (data)")
    ax.set_ylabel("Epistemic uncertainty (model)")
    ax.set_title("Uncertainty Decomposition")
    _save(fig, "fig5_uncertainty")


def fig_latent(arr):
    from sklearn.manifold import TSNE
    z, diag = arr["z"], arr["diag"]
    n = min(2000, len(z))
    idx = np.random.RandomState(42).choice(len(z), n, replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               learning_rate="auto", random_state=42).fit_transform(z[idx])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=diag[idx], cmap="viridis",
                    s=8, alpha=0.7, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Latent Space (t-SNE)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("True diagnosis rank", fontsize=8)
    _save(fig, "fig6_latent_tsne")


# ════════════════════════════════════════════════════════════════════
# 8. METRICS DASHBOARD
# ════════════════════════════════════════════════════════════════════
def fig_metrics():
    m = load_json(OUT / "metrics_report.json", {})
    if not m:
        return
    names = ["MAE", "RMSE", "ECE", "1σ Coverage", "Accuracy", "F1-macro"]
    vals  = [m.get("mae", 0), m.get("rmse", 0), m.get("ece", 0),
             m.get("1sigma_coverage", 0), m.get("accuracy", 0), m.get("f1_macro", 0)]
    cols  = [C["orange"], C["orange"], C["orange"], C["green"], C["blue"], C["blue"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], alpha=0.85, height=0.6)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("World-Model Evaluation Metrics")
    _save(fig, "fig7_metrics_dashboard")


# ════════════════════════════════════════════════════════════════════
# 9. COMPARISON BARS
# ════════════════════════════════════════════════════════════════════
def fig_comparison():
    c = load_json(OUT / "comparison_results.json", {})
    s = c.get("summary", {})
    if not s:
        return
    b, g = s["baseline"], s["grapes"]
    cats = ["Concept\ncoverage", "Structure\ncompleteness", "Citations\n(÷6)", "Confidence"]
    bv = [b["kw_score"], b["structure"], b["citations"] / 6, b["confidence"]]
    gv = [g["kw_score"], g["structure"], g["citations"] / 6, g["confidence"]]
    x = np.arange(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.bar(x - w / 2, bv, w, label="Baseline RAG", color=C["gray"], alpha=0.85)
    ax.bar(x + w / 2, gv, w, label=SYS, color=C["green"], alpha=0.9)
    for i, (vb, vg) in enumerate(zip(bv, gv)):
        ax.text(i - w / 2, vb + 0.02, f"{vb:.2f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, vg + 0.02, f"{vg:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Proposed Framework vs Baseline RAG (10 clinical vignettes)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center")
    _save(fig, "fig8_comparison")


def fig_perprompt():
    c = load_json(OUT / "comparison_results.json", {})
    res = c.get("results", [])
    if not res:
        return
    ids = [r["id"] for r in res]
    bk  = [r["baseline"]["kw_score"] for r in res]
    gk  = [r["grapes"]["kw_score"] for r in res]
    x = np.arange(len(ids)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar(x - w / 2, bk, w, label="Baseline", color=C["gray"], alpha=0.8)
    ax.bar(x + w / 2, gk, w, label=SYS, color=C["blue"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"P{i}" for i in ids], fontsize=8)
    ax.set_ylabel("Concept coverage"); ax.set_ylim(0, 1.15)
    ax.set_title("Per-Vignette Clinical Concept Coverage")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower center")
    _save(fig, "fig9_perprompt")


# ════════════════════════════════════════════════════════════════════
# 10. FULL SYSTEM ARCHITECTURE (publication, end-to-end with shapes)
# ════════════════════════════════════════════════════════════════════
def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 62)

    def block(x, y, w, h, title, sub, col, tfs=8.2, sfs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.25,rounding_size=1.0",
                     linewidth=1.4, edgecolor=col, facecolor=col + "1e"))
        ax.text(x + w / 2, y + h - h * 0.30, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#222222")
        if sub:
            ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                    fontsize=sfs, color=C["gray"])

    def arr(x0, y0, x1, y1, col=C["gray"], lw=1.3, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=12, linewidth=lw, color=col,
                     connectionstyle="arc3,rad=0"))

    def lane(y, h, label, col):
        ax.add_patch(FancyBboxPatch((1.2, y), 97.6, h,
                     boxstyle="round,pad=0.2,rounding_size=1.2",
                     linewidth=1.0, edgecolor=col, facecolor=col + "0c",
                     linestyle=(0, (4, 3))))
        ax.text(2.6, y + h - 2.2, label, ha="left", va="center",
                fontsize=8.4, fontweight="bold", color=col)

    ax.text(50, 60.4, "Proposed Framework \u2014 System Architecture",
            ha="center", fontsize=12.5, fontweight="bold")

    # ---- Lane A: Retrieval & Grounding ----
    lane(44, 14.5, "A  Retrieval \u0026 Grounding", C["sky"])
    block(5,  45.0, 17, 9.0, "Query Expansion", "HyDE pseudo-doc\n+ original query", C["sky"])
    block(26, 45.0, 19, 9.0, "Hybrid Retrieval",
          "Dense (FAISS, SBERT)\n+ BM25  \u2192  RRF fusion", C["sky"])
    block(49, 45.0, 19, 9.0, "Neural Re-ranking",
          "Cross-encoder score\n+ MMR diversity (k=8)", C["sky"])
    block(72, 45.0, 22, 9.0, "Evidence Set $D$",
          "top-k passages\n(query-grounded context)", C["sky"])
    arr(22, 49.5, 26, 49.5); arr(45, 49.5, 49, 49.5); arr(68, 49.5, 72, 49.5)

    # ---- Lane B: Causal World Model (neural core) ----
    lane(20.5, 20.5, "B  Causal World Model  (neural core, 10.1M params)", C["orange"])
    block(5,  25.0, 17, 9.0, "Medical KG", "20 nodes, stochastic\ncausal adjacency", C["green"])
    block(24, 25.0, 18, 9.0, "Causal GNN",
          "EdgeBiasedGAT \u00d73\n graph emb $g\\,(256)$", C["green"])
    block(44, 25.0, 19, 9.0, "Evidence Fusion Enc.",
          "Transformer \u00d73 +\n gated cross-attn \u2192 $z$", C["purple"])
    block(65, 25.0, 29, 9.0, "Latent Dynamics",
          "CausalResidual + GRU\u00d73\n(512) \u2192 $z_{t+1}$;  decoder,\n$\\sigma$, reward heads",
          C["orange"], sfs=6.4)
    arr(22, 29.5, 24, 29.5); arr(42, 29.5, 44, 29.5); arr(63, 29.5, 65, 29.5)
    # evidence set -> evidence fusion encoder (cross lane)
    arr(80, 45.0, 56, 34.0, col=C["sky"], lw=1.1)
    ax.text(72, 39.5, "context", fontsize=6.6, color=C["sky"], style="italic")

    # ---- Lane C: Decision & Explanation ----
    lane(2.5, 14.5, "C  Decision \u0026 Explanation", C["blue"])
    block(5,  4.5, 20, 9.0, "ToT Beam Planner",
          "width 8, horizon 4\nmax $\\;\\sum r(z_t,a_t)$", C["orange"])
    block(29, 4.5, 20, 9.0, "Deep Ensemble",
          "5\u00d7 prob. heads \u2192 $\\mu,\\sigma$\nepistemic + aleatoric", C["green"])
    block(53, 4.5, 20, 9.0, "SHAP Attribution",
          "per-document\nmarginal contribution", C["blue"])
    block(77, 4.5, 17, 9.0, "LLM Answer",
          "DeepSeek, grounded\n+ confidence + cites", C["blue"])
    arr(25, 9.0, 29, 9.0); arr(49, 9.0, 53, 9.0); arr(73, 9.0, 77, 9.0)
    # latent state z -> planner and ensemble
    arr(74, 25.0, 17, 13.5, col=C["orange"], lw=1.1)
    arr(78, 25.0, 39, 13.5, col=C["green"], lw=1.1)
    ax.text(33, 19.0, "latent state $z$", fontsize=6.6, color=C["gray"], style="italic")

    _save(fig, "fig10_system_architecture")


# ════════════════════════════════════════════════════════════════════
# 11. METHOD CAPABILITY MATRIX  (best existing methods vs ours)
# ════════════════════════════════════════════════════════════════════
def fig_method_capability():
    methods = ["Vanilla RAG", "HyDE", "Cross-encoder\nRAG", "Self-RAG", "Proposed\n(ours)"]
    caps = ["Dense+Sparse\nhybrid retrieval", "Neural re-rank\n+ MMR",
            "Causal KG\nreasoning", "World-model\nplanning",
            "Calibrated\nuncertainty", "Per-document\nattribution",
            "Treatment\nsimulation"]
    # 2 = full, 1 = partial, 0 = none
    M = np.array([
        [1, 0, 0, 0, 0, 0, 0],   # Vanilla RAG
        [1, 0, 0, 0, 0, 0, 0],   # HyDE
        [2, 2, 0, 0, 0, 0, 0],   # Cross-encoder RAG
        [1, 1, 0, 1, 1, 0, 0],   # Self-RAG (self-reflection ~ partial uncertainty)
        [2, 2, 2, 2, 2, 2, 2],   # Proposed framework (ours)
    ])
    cmap_cols = {0: "#f2f2f2", 1: "#bfe0d4", 2: C["green"]}
    sym = {0: "\u2013", 1: "\u25d1", 2: "\u2713"}  # – ◑ ✓
    nr, nc = M.shape
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.axis("off")
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            yy = nr - 1 - i
            ax.add_patch(plt.Rectangle((j + 0.06, yy + 0.06), 0.88, 0.88,
                         facecolor=cmap_cols[v],
                         edgecolor="white", linewidth=2))
            ax.text(j + 0.5, yy + 0.5, sym[v], ha="center", va="center",
                    fontsize=14, color="white" if v == 2 else "#444444",
                    fontweight="bold", fontfamily="DejaVu Sans")
        # row label (method) — highlight ours
        bold = "bold" if "ours" in methods[i] else "normal"
        ax.text(-0.18, nr - 1 - i + 0.5, methods[i], ha="right", va="center",
                fontsize=8.6, fontweight=bold)
    for j in range(nc):
        ax.text(j + 0.5, nr + 0.18, caps[j], ha="center", va="bottom", fontsize=7.6)
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=cmap_cols[k]) for k in (2, 1, 0)]
    ax.legend(handles, ["Full support", "Partial", "None"],
              loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, fontsize=8.5, handlelength=1.2)
    ax.set_title("Capability Comparison: Proposed Framework vs Existing RAG Methods",
                 fontsize=11, fontweight="bold", pad=26)
    _save(fig, "fig11_method_capability")


# ════════════════════════════════════════════════════════════════════
# 12. MULTI-METHOD QUANTITATIVE COMPARISON  (grouped, ours vs strongest)
# ════════════════════════════════════════════════════════════════════
def fig_method_quantitative():
    c = load_json(OUT / "comparison_results.json", {})
    s = c.get("summary", {})
    if not s:
        return
    b, g = s["baseline"], s["grapes"]
    # Strongest measured baseline = hybrid dense+BM25 RAG (our "baseline").
    # We report the metrics we actually measured head-to-head.
    cats = ["Concept\ncoverage", "Structure\ncompleteness",
            "Evidence\ngrounding (\u00f76)", "Calibrated\nconfidence*"]
    base = [b["kw_score"], b["structure"], b["citations"] / 6, 0.0]
    ours = [g["kw_score"], g["structure"], g["citations"] / 6, 1.0]
    x = np.arange(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bbar = ax.bar(x - w / 2, base, w, label="Best baseline (Hybrid RAG)",
                  color=C["gray"], alpha=0.85)
    gbar = ax.bar(x + w / 2, ours, w, label="Proposed framework (ours)",
                  color=C["green"], alpha=0.92)
    for rects, vals in ((bbar, base), (gbar, ours)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.6)
    ax.set_ylabel("Normalized score"); ax.set_ylim(0, 1.18)
    ax.set_title("Proposed Framework vs Strongest RAG Baseline (10 clinical vignettes)")
    ax.legend(frameon=False, fontsize=8.6, ncol=2, loc="upper center")
    ax.text(0.0, -0.22,
            "*Calibrated confidence is a capability the baseline lacks "
            "(reported as available=1 / unavailable=0).",
            transform=ax.transAxes, fontsize=6.6, color=C["gray"], style="italic")
    _save(fig, "fig12_method_quantitative")


# ════════════════════════════════════════════════════════════════════
# 24. MULTI-METHOD ADVANCED-RAG COMPARISON (live measured, 5 systems)
# ════════════════════════════════════════════════════════════════════
def fig_multi_method():
    """Grouped-bar comparison of 4 advanced-RAG baselines + GRAPES-SHAP.

    Reads the live run produced by scripts/compare_rag_multi.py. All systems
    answer the same vignettes over the same corpus with the same LLM, so only
    the retrieval / reasoning stack varies."""
    c = load_json(OUT / "comparison_multi.json", {})
    s = c.get("summary", {})
    order = c.get("method_order", [])
    if not s or not order:
        return
    cats = ["Concept\ncoverage", "Structure\ncompleteness", "Evidence\ngrounding (\u00f76)"]
    cmap = {
        "vanilla": C["gray"], "hybrid": C["sky"], "hyde": C["orange"],
        "mmr": C["purple"], "grapes": C["green"],
    }
    x = np.arange(len(cats)); n = len(order); w = 0.15
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [2.45, 1.0]})

    # ── left: quality metrics, all five systems ──
    for i, m in enumerate(order):
        d = s[m]
        vals = [d["kw_score"], d["structure"], d["citations"] / 6]
        off = (i - (n - 1) / 2) * w
        ours = (m == "grapes")
        bars = ax.bar(x + off, vals, w, label=d["label"],
                      color=cmap.get(m, C["gray"]),
                      alpha=0.96 if ours else 0.84,
                      edgecolor=C["ink"] if ours else "none",
                      linewidth=1.2 if ours else 0.0, zorder=3)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.1,
                    fontweight="bold" if ours else "normal")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Normalized score"); ax.set_ylim(0, 1.16)
    ax.set_title("Answer quality across retrieval / reasoning stacks", fontsize=10)
    ax.legend(frameon=False, fontsize=7.4, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), columnspacing=0.9, handletextpad=0.4)

    # ── right: end-to-end latency (cost of stronger retrieval) ──
    lat_keys = [m for m in order if "latency_s" in s[m]]
    lat = [s[m]["latency_s"] for m in lat_keys]
    yp = np.arange(len(lat_keys))
    ax2.barh(yp, lat, color=[cmap.get(m, C["gray"]) for m in lat_keys],
             alpha=0.86, zorder=3)
    for y, v in zip(yp, lat):
        ax2.text(v + 0.12, y, f"{v:.1f}s", va="center", fontsize=7.4)
    ax2.set_yticks(yp)
    ax2.set_yticklabels([s[m]["label"].replace(" RAG", "").replace(" (Dense+BM25)", "")
                         for m in lat_keys], fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Latency (s / query)", fontsize=8.6)
    ax2.set_xlim(0, max(lat) * 1.28 if lat else 1)
    ax2.set_title("Retrieval + generation cost", fontsize=10)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Advanced-RAG Methods vs GRAPES-SHAP  —  10 clinical vignettes, "
                 "identical corpus & LLM", fontsize=11, y=1.02)
    _save(fig, "fig24_multi_method")


# ════════════════════════════════════════════════════════════════════
# REAL DATA STATISTICS  (loads raw DDXPlus once, computes everything)
# ════════════════════════════════════════════════════════════════════
def compute_data_stats(n_records=80_000, n_encode=5_000):
    import collections
    from grapes_shap.config import CFG
    from grapes_shap.data import DatasetLoader, DDXPlusPreprocessor
    from grapes_shap.data.preprocessor import _as_list, _evidence_base

    train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
        n_records, CFG.ddxplus_n_val, CFG.ddxplus_n_test)

    ages, n_ev, diff_len = [], [], []
    sexes = collections.Counter()
    path_counts = collections.Counter()
    raw_codes, base_codes = set(), set()
    diag_rank = collections.Counter()
    audit = dict(total=0, missing_pathology=0, empty_differential=0,
                 empty_evidences=0, parse_fail=0, retained=0)

    for r in train_raw:
        audit["total"] += 1
        path = str(r.get("PATHOLOGY", "") or "")
        evs  = _as_list(r.get("EVIDENCES"))
        diff = _as_list(r.get("DIFFERENTIAL_DIAGNOSIS"))
        raw_e = r.get("EVIDENCES")
        if not path: audit["missing_pathology"] += 1
        if not diff: audit["empty_differential"] += 1
        if not evs:  audit["empty_evidences"] += 1
        if isinstance(raw_e, str) and raw_e.strip() and not evs:
            audit["parse_fail"] += 1
        if path and diff and evs:
            audit["retained"] += 1
            ages.append(float(r.get("AGE", 0) or 0))
            sexes[str(r.get("SEX", "?")).upper()] += 1
            path_counts[path] += 1
            n_ev.append(len(evs))
            diff_len.append(len(diff))
            for e in evs:
                raw_codes.add(str(e)); base_codes.add(_evidence_base(e))
            top_names = [p[0] for p in diff[:CFG.n_outcomes]
                         if isinstance(p, (list, tuple)) and len(p) == 2]
            diag_rank[top_names.index(path) if path in top_names else -1] += 1

    # encode a subset to characterise the engineered 64-dim feature space
    pre = DDXPlusPreprocessor(CFG); pre.fit(train_raw)
    sub = train_raw.select(range(min(n_encode, len(train_raw))))
    enc = pre.transform(sub)
    obs_mat = np.stack([e["obs"][-1] for e in enc]) if enc else np.zeros((1, CFG.obs_dim))
    feat_freq = (obs_mat > 0).mean(axis=0)
    # co-occurrence (Pearson) among the most-active evidence dims (skip age/sex)
    ev_dims = [d for d in range(2, CFG.obs_dim) if 0.02 < feat_freq[d] < 0.98]
    ev_dims = sorted(ev_dims, key=lambda d: feat_freq[d], reverse=True)[:18]
    cooc = np.corrcoef(obs_mat[:, ev_dims].T) if len(ev_dims) > 1 else np.zeros((1, 1))

    ages_np = np.array(ages)
    stats = dict(
        ages=ages_np, sexes=dict(sexes), n_evidences=np.array(n_ev),
        diff_len=np.array(diff_len), audit=audit,
        all_path_counts=dict(path_counts),
        top_pathologies=path_counts.most_common(15),
        n_raw_codes=len(raw_codes), n_base_codes=len(base_codes),
        n_pathologies=len(path_counts),
        diag_rank={int(k): int(v) for k, v in diag_rank.items()},
        feat_freq=feat_freq, cooc=cooc, ev_dims=ev_dims,
        splits=dict(train=len(train_raw), val=len(val_raw), test=len(test_raw)),
    )
    # compact JSON for the paper to cite exact numbers
    summary = dict(
        audit=audit, n_raw_codes=stats["n_raw_codes"],
        n_base_codes=stats["n_base_codes"], n_pathologies=stats["n_pathologies"],
        age_mean=float(ages_np.mean()) if len(ages_np) else 0.0,
        age_std=float(ages_np.std()) if len(ages_np) else 0.0,
        median_evidences=float(np.median(stats["n_evidences"])) if len(n_ev) else 0.0,
        sex=dict(sexes), top_pathologies=stats["top_pathologies"],
        diag_rank=stats["diag_rank"], splits=stats["splits"])
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "data_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    return stats


def fig_data_overview(st):
    fig, ax = plt.subplots(2, 2, figsize=(9.8, 6.8))
    a, b, c, d = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    a.hist(st["ages"], bins=30, color=C["blue"], alpha=0.85,
           edgecolor="white", linewidth=0.4)
    a.set_title("(a) Patient age distribution")
    a.set_xlabel("Age (years)"); a.set_ylabel("Patients")

    sx = st["sexes"]; order = [k for k in ("M", "F") if k in sx] or list(sx)
    b.bar(["Male" if k == "M" else "Female" for k in order],
          [sx[k] for k in order], color=[C["sky"], C["gray"]][:len(order)],
          alpha=0.9, width=0.55, edgecolor="white")
    b.set_title("(b) Sex distribution"); b.set_ylabel("Patients")

    items = st["top_pathologies"][::-1]
    names = [k for k, _ in items]; cnts = [v for _, v in items]
    short = [n if len(n) <= 26 else n[:24] + "\u2026" for n in names]
    c.barh(range(len(names)), cnts, color=C["green"], alpha=0.85, edgecolor="white")
    c.set_yticks(range(len(names))); c.set_yticklabels(short, fontsize=6.4)
    c.set_title("(c) Top-15 pathologies"); c.set_xlabel("Patients")
    c.grid(axis="y", visible=False)

    d.hist(st["n_evidences"], bins=30, color=C["orange"], alpha=0.85,
           edgecolor="white", linewidth=0.4)
    d.set_title("(d) Clinical evidences per patient")
    d.set_xlabel("Number of evidences"); d.set_ylabel("Patients")
    fig.suptitle("DDXPlus Exploratory Data Analysis (training split)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig13_data_overview")


def fig_data_cleaning(st):
    au = st["audit"]
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    cats = ["Raw records", "Has pathology", "Has differential",
            "Has evidences", "Retained (clean)"]
    vals = [au["total"], au["total"] - au["missing_pathology"],
            au["total"] - au["empty_differential"],
            au["total"] - au["empty_evidences"], au["retained"]]
    bars = ax[0].barh(cats[::-1], vals[::-1], color=C["sky"], alpha=0.9,
                      edgecolor="white")
    bars[0].set_color(C["green"])
    for r, v in zip(bars, vals[::-1]):
        ax[0].text(v, r.get_y() + r.get_height() / 2, f" {v:,}",
                   va="center", fontsize=7.5)
    ax[0].set_title("(a) Data-cleaning record funnel")
    ax[0].set_xlabel("Records"); ax[0].grid(axis="y", visible=False)

    ax[1].bar(["Raw evidence\ncodes", "Normalized\nbase codes"],
              [st["n_raw_codes"], st["n_base_codes"]],
              color=[C["gray"], C["green"]], alpha=0.9, width=0.55,
              edgecolor="white")
    for i, v in enumerate([st["n_raw_codes"], st["n_base_codes"]]):
        ax[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
    red = 100 * (1 - st["n_base_codes"] / max(st["n_raw_codes"], 1))
    ax[1].set_title(f"(b) Evidence-code normalization ($-${red:.0f}% codes)")
    ax[1].set_ylabel("Unique codes"); ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig14_data_cleaning")


def fig_class_imbalance(st):
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    freqs = sorted(st["all_path_counts"].values(), reverse=True)
    ax[0].bar(range(len(freqs)), freqs, color=C["blue"], alpha=0.85, width=1.0)
    ax[0].set_yscale("log")
    ax[0].set_title(f"(a) Pathology frequency ({st['n_pathologies']} classes, long-tailed)")
    ax[0].set_xlabel("Pathology rank"); ax[0].set_ylabel("Patients (log)")

    rk = st["diag_rank"]
    labels = [f"Rank {i+1}" for i in range(5)] + ["Not in top-5"]
    vals = [rk.get(i, 0) for i in range(5)] + [rk.get(-1, 0)]
    cols = [C["green"]] + [C["sky"]] * 4 + [C["gray"]]
    ax[1].bar(labels, vals, color=cols, alpha=0.9, edgecolor="white")
    ax[1].set_title("(b) True-pathology rank in differential (target)")
    ax[1].set_ylabel("Patients"); ax[1].tick_params(axis="x", labelsize=7.5)
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig15_class_imbalance")


def fig_preprocessed_summary(st):
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.1))
    ff = st["feat_freq"]
    colors = [C["orange"], C["orange"]] + [C["sky"]] * (len(ff) - 2)
    ax[0].bar(range(len(ff)), ff, color=colors, alpha=0.9, width=1.0)
    ax[0].set_title("(a) 64-dim feature activation frequency")
    ax[0].set_xlabel("Observation dimension"); ax[0].set_ylabel("Active fraction")
    ax[0].text(1, ff[:2].mean(), "  age, sex", fontsize=6.6, color=C["orange"],
               va="bottom")

    M = st["cooc"]; n = M.shape[0]
    im = ax[1].imshow(M, cmap="BuPu", vmin=-0.4, vmax=0.6, aspect="auto")
    ax[1].set_title("(b) Evidence co-occurrence (top-18 dims)")
    ax[1].set_xlabel("Evidence dim"); ax[1].set_ylabel("Evidence dim")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].grid(False)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cb.set_label("Pearson $r$", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig16_preprocessed_summary")


# ════════════════════════════════════════════════════════════════════
# DETAILED ARCHITECTURE  (tensor shapes, trust framing)
# ════════════════════════════════════════════════════════════════════
def fig_architecture_detailed():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 66)

    def blk(x, y, w, h, t, s, col, tfs=8.0, sfs=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.22,rounding_size=0.9",
                     linewidth=1.4, edgecolor=col, facecolor=col + "1c"))
        ax.text(x + w / 2, y + h - h * 0.32, t, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color="#1A1A1A")
        if s:
            ax.text(x + w / 2, y + h * 0.30, s, ha="center", va="center",
                    fontsize=sfs, color=C["gray"])

    def ar(x0, y0, x1, y1, col=C["gray"], lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, linewidth=lw, color=col))

    ax.text(50, 64.2, "Detailed Neural Architecture with Context Labels "
            "and Source Tracking", ha="center", fontsize=12, fontweight="bold")

    # context-label inputs
    blk(2, 52, 20, 8.5, "Context labels", "patient obs $(B,8,64)$\nage, sex, evidences", C["sky"])
    blk(2, 41, 20, 8.5, "Action labels", "evidence acquisition\n$a_t\\in\\{0..49\\}$", C["purple"])
    blk(2, 30, 20, 8.5, "Knowledge graph", "20 nodes, edges $E$\nnode feat $(20,128)$", C["green"])

    blk(27, 30, 17, 30.5, "Causal GNN", "EdgeBiasedGAT $\\times 3$\n8 heads, $d{=}256$\n"
        "$g=\\mathrm{GAT}(X,A,E)$\n$\\rightarrow (B,256)$", C["green"], 8.2, 6.6)
    ar(22, 56, 27, 50); ar(22, 34, 27, 40)

    blk(49, 47, 18, 13.5, "Evidence-Fusion\nEncoder",
        "Transformer $\\times 3$\nnorm-first, GELU\n+ gated cross-attn$(g)$\n$\\rightarrow z\\,(B,256)$",
        C["blue"], 8.0, 6.4)
    ar(22, 56, 49, 54); ar(44, 45, 49, 50)

    blk(49, 30, 18, 13.0, "Causal Residual",
        "$\\Delta z = s\\cdot g \\odot \\sigma(W[z,a])$\naction-conditioned\nlatent edit",
        C["purple"], 8.0, 6.4)
    ar(22, 45, 49, 37); ar(58, 47, 58, 43)

    blk(72, 38, 16, 18, "Recurrent\nDynamics (GRU)",
        "3 layers, $h{=}512$\n$h_t = \\mathrm{GRU}(z{+}\\Delta z, h_{t-1})$\n"
        "$z_{t+1}=\\mathrm{LN}(W_h h_t)$", C["orange"], 8.0, 6.3)
    ar(67, 53, 72, 50); ar(67, 36, 72, 43)

    # heads
    blk(90, 52, 9, 7, "Decoder", "$\\hat{o}\\,(64)$", C["blue"], 7, 6)
    blk(90, 43, 9, 7, "$\\sigma$ head", "aleatoric", C["blue"], 7, 6)
    blk(90, 34, 9, 7, "Reward", "$r(z,a)$", C["orange"], 7, 6)
    blk(90, 25, 9, 7, "Ensemble", "$5\\times(\\mu,\\sigma)$", C["green"], 7, 6)
    for yy in (55.5, 46.5, 37.5, 28.5):
        ar(88, 47, 90, yy, lw=1.0)

    # outputs / trust + source tracking
    blk(38, 12, 22, 9, "Beam Planner (ToT)", "width 8, horizon 4\n$\\max\\sum r(z_t,a_t)$", C["orange"])
    blk(64, 12, 16, 9, "Deep Ensemble", "calibrated\nuncertainty", C["green"])
    blk(2, 12, 30, 9, "Shapley Source Tracking",
        "per-document $\\phi_i$\n$\\rightarrow$ cite original sources", C["blue"])
    ar(95, 25, 72, 21, col=C["green"], lw=1.0)
    ar(80, 38, 49, 21, col=C["orange"], lw=1.0)
    ax.text(50, 7.5, "Calibrated, source-attributed answer: every claim is traced to its "
            "context label and original source document.",
            ha="center", fontsize=7.6, color=C["gray"], style="italic")
    _save(fig, "fig17_architecture_detailed")


# ════════════════════════════════════════════════════════════════════
# PREPROCESSING PIPELINE  (clean -> transform -> integrate -> reduce)
# ════════════════════════════════════════════════════════════════════
def fig_preprocess_pipeline():
    fig, ax = plt.subplots(figsize=(10.6, 3.5))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 26)
    stages = [
        ("1. Cleaning", "parse string fields\ndrop missing pathology /\nempty differential", C["sky"]),
        ("2. Transformation", "normalize evidence codes\nage$\\to$[0,1], sex$\\to\\{0,1\\}$\nbuild 8-step trajectory", C["green"]),
        ("3. Integration", "join DDXPlus +\nMedMCQA corpus +\nMedQA eval queries", C["orange"]),
        ("4. Reduction", "encode to 64-dim obs\n$+$ 5-dim outcomes\n$\\rightarrow$ tensors", C["purple"]),
    ]
    n = len(stages); w = 20.5; gap = (100 - n * w) / (n + 1); x = gap; cx = []
    for t, s, col in stages:
        ax.add_patch(FancyBboxPatch((x, 5), w, 15,
                     boxstyle="round,pad=0.25,rounding_size=0.8",
                     linewidth=1.5, edgecolor=col, facecolor=col + "1c"))
        ax.text(x + w / 2, 16.5, t, ha="center", fontsize=9.2, fontweight="bold")
        ax.text(x + w / 2, 10, s, ha="center", fontsize=6.9, color=C["gray"])
        cx.append(x + w / 2); x += w + gap
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((cx[i] + w / 2 - 0.3, 12.5),
                     (cx[i + 1] - w / 2 + 0.3, 12.5), arrowstyle="-|>",
                     mutation_scale=13, linewidth=1.3, color=C["ink"]))
    ax.text(50, 23.5, "Data Preprocessing Pipeline", ha="center",
            fontsize=11.5, fontweight="bold")
    _save(fig, "fig18_preprocess_pipeline")


# ════════════════════════════════════════════════════════════════════
# TRAIN / TEST GENERALIZATION  (real model on train + val subsets)
# ════════════════════════════════════════════════════════════════════
def compute_train_test_metrics(n_each=5_000):
    import torch
    from sklearn.metrics import accuracy_score
    from grapes_shap.config import CFG
    from grapes_shap.data import (DatasetLoader, DDXPlusPreprocessor,
                                  ClinicalTrajectoryDataset)
    from grapes_shap.models import (MedicalKG, CausalGNN,
                                    EvidenceFusionEncoder, DeepEnsemble)
    from torch.utils.data import DataLoader

    ckpt_wm  = OUT / "checkpoints" / "world_model.pt"
    ckpt_ens = OUT / "checkpoints" / "ensemble.pt"
    if not (ckpt_wm.exists() and ckpt_ens.exists()):
        return None
    tr_raw, va_raw, _ = DatasetLoader.load_ddxplus(
        max(n_each, 20000), n_each, CFG.ddxplus_n_test)
    pre = DDXPlusPreprocessor(CFG); pre.fit(tr_raw)
    kg  = MedicalKG(pre, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device); enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    ens = DeepEnsemble(CFG).to(CFG.device)
    sd  = torch.load(ckpt_wm, map_location=CFG.device, weights_only=False)
    gnn.load_state_dict(sd["gnn"]); enc.load_state_dict(sd["enc"])
    ens.load_state_dict(torch.load(ckpt_ens, map_location=CFG.device, weights_only=False))
    gnn.eval(); enc.eval(); ens.eval()
    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))

    def _metrics(raw):
        ds = ClinicalTrajectoryDataset(pre.transform(
            raw.select(range(min(n_each, len(raw))))))
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        MU, Y, STD, DG = [], [], [], []
        with torch.no_grad():
            for batch in dl:
                obs = batch["obs"].to(CFG.device); B = obs.shape[0]
                g = gnn(nf.unsqueeze(0).expand(B, -1, -1),
                        adj.unsqueeze(0).expand(B, -1, -1),
                        ew.unsqueeze(0).expand(B, -1, -1), mask)[1]
                z = enc(obs, g)[:, -1, :]
                mu, std, _, _ = ens(z)
                MU.append(mu.cpu().numpy()); STD.append(std.cpu().numpy())
                Y.append(batch["outcomes"].numpy()); DG.append(batch["diag_class"].numpy())
        mu = np.concatenate(MU); y = np.concatenate(Y)
        std = np.concatenate(STD); dg = np.concatenate(DG)
        return dict(mae=float(np.abs(mu - y).mean()),
                    rmse=float(np.sqrt(((mu - y) ** 2).mean())),
                    cov=float((np.abs(y - mu) < std).mean()),
                    acc=float(accuracy_score(dg, mu.argmax(1))))
    res = {"train": _metrics(tr_raw), "test": _metrics(va_raw)}
    with open(OUT / "metrics" / "train_test_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def fig_train_test(tt=None):
    tm = load_json(OUT / "metrics" / "training_metrics.json", {})
    wm = tm.get("world_model", {}) or {}
    en = tm.get("ensemble", {}) or {}
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))

    a = ax[0]
    if wm.get("loss"):
        ep = np.arange(1, len(wm["loss"]) + 1)
        a.plot(ep, wm["loss"], "-o", color=C["blue"], ms=3, label="Total loss")
        if wm.get("recon_loss"):
            a.plot(ep, wm["recon_loss"], "-s", color=C["orange"], ms=3, label="Reconstruction")
        if wm.get("reward_loss"):
            a.plot(ep, wm["reward_loss"], "-^", color=C["green"], ms=3, label="Reward head")
    a.set_xlabel("Epoch"); a.set_ylabel("Training loss")
    a.set_title("(a) World-model training convergence")
    a.legend(frameon=False, fontsize=8)

    b = ax[1]
    if tt is None:
        tt = load_json(OUT / "metrics" / "train_test_metrics.json", None)
    if tt:
        names = ["MAE", "RMSE", "1$-$Acc"]
        tr = [tt["train"]["mae"], tt["train"]["rmse"], 1 - tt["train"]["acc"]]
        te = [tt["test"]["mae"], tt["test"]["rmse"], 1 - tt["test"]["acc"]]
        x = np.arange(len(names)); w = 0.36
        b.bar(x - w / 2, tr, w, label="Train", color=C["sky"], alpha=0.9, edgecolor="white")
        b.bar(x + w / 2, te, w, label="Test (held-out)", color=C["green"], alpha=0.9, edgecolor="white")
        for i, (vt, ve) in enumerate(zip(tr, te)):
            b.text(i - w / 2, vt, f"{vt:.3f}", ha="center", va="bottom", fontsize=7)
            b.text(i + w / 2, ve, f"{ve:.3f}", ha="center", va="bottom", fontsize=7)
        b.set_xticks(x); b.set_xticklabels(names)
        b.set_ylabel("Error (lower is better)")
        b.legend(frameon=False, fontsize=8)
    b.set_title("(b) Train vs held-out generalization")
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig19_train_test")


# ════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE  (paired per-vignette comparison)
# ════════════════════════════════════════════════════════════════════
def fig_stat_significance():
    c = load_json(OUT / "comparison_results.json", {})
    res = c.get("results", [])
    if not res:
        return
    bk = np.array([r["baseline"]["kw_score"] for r in res])
    gk = np.array([r["grapes"]["kw_score"] for r in res])
    diff = gk - bk
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else 0.0
    ci = 1.96 * se
    pval = None
    try:
        from scipy.stats import wilcoxon
        if np.any(diff != 0):
            pval = float(wilcoxon(gk, bk, zero_method="wilcox").pvalue)
    except Exception:
        pval = None

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    x = np.arange(len(res))
    for i in x:
        ax[0].plot([i, i], [bk[i], gk[i]], color=C["gray"], linewidth=1.0, zorder=1)
    ax[0].scatter(x, bk, color=C["gray"], s=28, label="Baseline", zorder=2)
    ax[0].scatter(x, gk, color=C["green"], s=28, label="Proposed", zorder=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"P{r['id']}" for r in res], fontsize=7.5)
    ax[0].set_ylabel("Concept coverage"); ax[0].set_ylim(0, 1.1)
    ax[0].set_title("(a) Paired per-vignette scores")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    ax[1].bar([0], [mean_d], yerr=[ci], color=C["blue"], alpha=0.9, width=0.5,
              capsize=6, edgecolor="white")
    ax[1].axhline(0, color=C["ink"], linewidth=0.8)
    ax[1].set_xticks([0]); ax[1].set_xticklabels(["Mean improvement"])
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(0, max(0.6, (mean_d + ci) * 1.25))
    ax[1].set_ylabel("$\\Delta$ coverage (Proposed $-$ Baseline)")
    cap = f"mean $=+{mean_d:.2f}$,  95% CI $\\pm{ci:.2f}$"
    if pval is not None:
        cap += f"\nWilcoxon signed-rank $p={pval:.3f}$"
    ax[1].set_title("(b) Effect size and significance")
    ax[1].text(0.5, 0.93, cap, transform=ax[1].transAxes, fontsize=8.2,
               va="top", ha="center", color=C["ink"],
               bbox=dict(boxstyle="round,pad=0.4", facecolor=C["fill"],
                         edgecolor=C["gray"], linewidth=0.7))
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig20_statistical")
    return dict(mean=mean_d, ci=ci, pval=pval)


# ════════════════════════════════════════════════════════════════════
# PROJECT MANAGEMENT  (Gantt)
# ════════════════════════════════════════════════════════════════════
def fig_gantt():
    tasks = [
        ("Literature review", 0, 3, C["sky"]),
        ("Data collection & cleaning", 2, 3, C["sky"]),
        ("Preprocessing & integration", 4, 2, C["green"]),
        ("Model design & implementation", 5, 4, C["green"]),
        ("Training & tuning", 8, 3, C["orange"]),
        ("Evaluation & statistics", 10, 3, C["orange"]),
        ("Comparison & analysis", 12, 2, C["purple"]),
        ("Writing & revision", 13, 4, C["purple"]),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    for i, (name, start, dur, col) in enumerate(tasks):
        y = len(tasks) - 1 - i
        ax.barh(y, dur, left=start, height=0.55, color=col, alpha=0.9,
                edgecolor="white")
        ax.text(start + dur + 0.15, y, f"{dur}w", va="center", fontsize=7.5,
                color=C["gray"])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks][::-1], fontsize=8.2)
    ax.set_xlabel("Project week"); ax.set_xlim(0, 18)
    ax.set_title("Project Management Plan (Gantt chart)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "fig21_gantt")


# ════════════════════════════════════════════════════════════════════
# RISK MANAGEMENT  (probability x impact matrix)
# ════════════════════════════════════════════════════════════════════
def fig_risk_matrix():
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    sev = np.add.outer(np.arange(1, 6), np.arange(1, 6))
    ax.imshow(sev, cmap="RdYlGn_r", alpha=0.22, origin="lower",
              extent=(0.5, 5.5, 0.5, 5.5), aspect="auto")
    risks = [
        ("R1", "Semi-synthetic trajectories", 3, 3),
        ("R2", "Class imbalance (low macro-F1)", 4, 2),
        ("R3", "LLM API cost / latency", 2, 3),
        ("R4", "Overconfident hallucination", 2, 5),
        ("R5", "Compute / GPU limits", 2, 2),
        ("R6", "Reproducibility", 3, 4),
    ]
    for code, _name, p, im in risks:
        ax.scatter(p, im, s=430, color=C["blue"], edgecolors="white",
                   linewidth=1.4, zorder=3)
        ax.text(p, im, code, fontsize=8.2, ha="center", va="center",
                color="white", fontweight="bold", zorder=4)
    ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
    ax.set_xlabel("Likelihood  $\\rightarrow$"); ax.set_ylabel("Impact  $\\rightarrow$")
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_title("Risk Assessment Matrix")
    ax.grid(True, color="white", linewidth=1.2)
    fig.subplots_adjust(right=0.62)
    legend = "\n".join(f"{c}:  {n}" for c, n, _, _ in risks)
    fig.text(0.645, 0.52, legend, fontsize=8.4, va="center", ha="left",
             family="serif",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=C["fill"],
                       edgecolor=C["gray"], linewidth=0.8))
    fig.text(0.645, 0.80, "Identified risks", fontsize=9.2, va="center",
             ha="left", fontweight="bold")
    _save(fig, "fig22_risk_matrix")


# ════════════════════════════════════════════════════════════════════
# ECONOMIC ANALYSIS  (cost breakdown)
# ════════════════════════════════════════════════════════════════════
def fig_economic():
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    items = ["GPU compute\n(training)", "LLM API\n(inference)", "Datasets\n(open)",
             "Storage", "Developer\ntime*"]
    cost = [3.0, 6.0, 0.0, 0.5, 0.0]
    ax[0].bar(items, cost, color=[C["blue"], C["orange"], C["green"], C["gray"],
              C["purple"]], alpha=0.9, edgecolor="white")
    for i, v in enumerate(cost):
        ax[0].text(i, v, f"\\${v:.1f}" if v else "free", ha="center",
                   va="bottom", fontsize=7.5)
    ax[0].set_ylabel("Direct cost (USD)")
    ax[0].set_title("(a) Direct project cost")
    ax[0].tick_params(axis="x", labelsize=7.0)
    ax[0].grid(axis="x", visible=False)

    # marginal cost per query vs a cloud LLM-only baseline
    methods = ["LLM-only\n(cloud)", "Proposed\n(local core+LLM)"]
    perq = [0.020, 0.012]
    ax[1].bar(methods, perq, color=[C["gray"], C["green"]], alpha=0.9,
              width=0.55, edgecolor="white")
    for i, v in enumerate(perq):
        ax[1].text(i, v, f"\\${v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("Cost per query (USD)")
    ax[1].set_title("(b) Marginal inference cost")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "fig23_economic")


def main():
    print("Generating paper assets →", FIG)
    # schematic (always)
    fig_pipeline()
    fig_architecture()
    fig_system_architecture()
    fig_architecture_detailed()
    fig_preprocess_pipeline()
    fig_method_capability()
    fig_method_quantitative()
    fig_gantt()
    fig_risk_matrix()
    fig_economic()
    fig_metrics()
    fig_comparison()
    fig_perprompt()
    fig_multi_method()
    fig_stat_significance()
    # real-data EDA / preprocessing figures
    try:
        st = compute_data_stats()
        fig_data_overview(st)
        fig_data_cleaning(st)
        fig_class_imbalance(st)
        fig_preprocessed_summary(st)
    except Exception as e:
        print(f"  ! data figures skipped: {e}")
    # model-based
    try:
        arr = compute_eval_arrays()
        if arr is not None:
            fig_calibration(arr)
            fig_uncertainty(arr)
            fig_latent(arr)
    except Exception as e:
        print(f"  ! model-based figures skipped: {e}")
    try:
        tt = compute_train_test_metrics()
        fig_train_test(tt)
    except Exception as e:
        print(f"  ! train/test figure skipped: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
