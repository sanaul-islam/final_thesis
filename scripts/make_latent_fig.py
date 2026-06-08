#!/usr/bin/env python3
"""
Generate Figure 05 — latent-space t-SNE visualisation of the Evidence-Fusion
Encoder, using the trained checkpoint. Standalone so it can be regenerated
without re-running the full pipeline.

Run: python scripts/make_latent_fig.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from grapes_shap.config import CFG, CKPT_DIR
from grapes_shap.data import (DatasetLoader, DDXPlusPreprocessor,
                              ClinicalTrajectoryDataset)
from grapes_shap.models import (MedicalKG, CausalGNN, EvidenceFusionEncoder)
from grapes_shap.visualization import plot_latent_space


def main():
    print("Figure 05 — Latent-space t-SNE\n")

    print("[1/4] Loading DDXPlus (train for vocab + val for plotting)...")
    train_raw, val_raw, _ = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)

    print("[2/4] Fitting preprocessor + transforming val split...")
    pre = DDXPlusPreprocessor(CFG)
    pre.fit(train_raw)
    processed_val = pre.transform(val_raw)
    ds_val = ClinicalTrajectoryDataset(processed_val)
    dl_val = DataLoader(ds_val, batch_size=128, shuffle=False)

    print("[3/4] Building models + loading checkpoint...")
    kg  = MedicalKG(pre, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device)
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)

    wm_ckpt = CKPT_DIR / "world_model.pt"
    if not wm_ckpt.exists():
        sys.exit(f"Missing {wm_ckpt} — run training (run.py) first.")
    sd = torch.load(wm_ckpt, map_location=CFG.device)
    enc.load_state_dict(sd["enc"])
    gnn.load_state_dict(sd["gnn"])
    print(f"  Loaded encoder + GNN from {wm_ckpt.name}")

    print("[4/4] Computing latent embeddings + t-SNE...")
    plot_latent_space(enc, gnn, kg, dl_val, CFG, n_samples=2000)
    print("Done.")


if __name__ == "__main__":
    main()
