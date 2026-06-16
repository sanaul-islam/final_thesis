#!/usr/bin/env python3
"""
Retrain GRAPES-SHAP (world model + deep ensemble) with the corrected
reward-head supervision and beam-search planner, then run the full
quantitative evaluation and persist real metrics for the paper.

This script deliberately avoids the DeepSeek LLM stage so it can run
end-to-end without any API key. Run from the project root:

    python scripts/retrain_and_eval.py
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from grapes_shap.config import CFG, SAVE_DIR, CKPT_DIR
from grapes_shap.data import (
    DatasetLoader, DDXPlusPreprocessor, MedMCQAPreprocessor,
    MedQAPreprocessor, ClinicalTrajectoryDataset,
)
from grapes_shap.models import (
    MedicalKG, CausalGNN, EvidenceFusionEncoder, LatentWorldModel, DeepEnsemble,
)
from grapes_shap.inference import HybridRetriever, SHAPAttributor
from grapes_shap.training import train_world_model, train_ensemble, evaluate_all

METRICS_DIR = SAVE_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "=" * 68)
    print("  GRAPES-SHAP  |  Retrain + Evaluate (reward head + beam planner)")
    print("=" * 68 + "\n")
    t0 = time.time()

    # ── 1. Load data ──
    print("[1/6] Loading datasets...")
    train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)
    medmcqa_raw = DatasetLoader.load_medmcqa(CFG.medmcqa_n_train)
    medqa_raw   = DatasetLoader.load_medqa(CFG.medqa_n_test)

    # ── 2. Preprocess ──
    print("\n[2/6] Preprocessing...")
    pre = DDXPlusPreprocessor(CFG)
    pre.fit(train_raw)
    tr = pre.transform(train_raw)
    va = pre.transform(val_raw)
    print(f"  train:{len(tr)} val:{len(va)}")
    docs    = MedMCQAPreprocessor.to_documents(medmcqa_raw)
    queries = MedQAPreprocessor.to_queries(medqa_raw)
    print(f"  RAG docs:{len(docs):,}  eval queries:{len(queries)}")

    # ── 3. Retrieval index (for SHAP eval only; no LLM) ──
    print("\n[3/6] Building hybrid retriever (subset for SHAP eval)...")
    retriever = HybridRetriever(CFG)
    retriever.build(docs[:20_000])

    # ── 4. Build architecture ──
    print("\n[4/6] Initialising architecture...")
    kg  = MedicalKG(pre, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device)
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    wm  = LatentWorldModel(CFG).to(CFG.device)
    ens = DeepEnsemble(CFG).to(CFG.device)
    total_params = sum(p.numel() for m in [gnn, enc, wm, ens] for p in m.parameters())
    print(f"  Total parameters: {total_params:,}")

    ds_tr = ClinicalTrajectoryDataset(tr)
    ds_va = ClinicalTrajectoryDataset(va)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.batch_size, shuffle=True,
                       num_workers=2, pin_memory=torch.cuda.is_available())
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False,
                       num_workers=2, pin_memory=torch.cuda.is_available())

    # ── 5. Train ──
    print("\n[5/6] Training world model (with reward-head supervision)...")
    wm_hist  = train_world_model(wm, enc, gnn, kg, dl_tr, CFG)
    print("\n       Training deep ensemble...")
    ens_hist = train_ensemble(ens, enc, gnn, kg, dl_tr, CFG)

    # ── 6. Evaluate ──
    print("\n[6/6] Evaluating...")
    shap_attr = SHAPAttributor(CFG)
    metrics = evaluate_all(ens, enc, gnn, kg, dl_va, queries,
                           retriever, shap_attr, CFG)

    train_min = round((time.time() - t0) / 60, 2)
    report = {
        "mae":               float(metrics["mae"]),
        "rmse":              float(metrics["rmse"]),
        "1sigma_coverage":   float(metrics["1sigma_coverage"]),
        "ece":               float(metrics["ece"]),
        "accuracy":          float(metrics["accuracy"]),
        "f1_macro":          float(metrics["f1_macro"]),
        "mean_shap":         float(metrics["mean_shap"]),
        "total_params":      total_params,
        "training_time_min": train_min,
        "dataset":           "DDXPlus + MedMCQA + MedQA",
        "device":            CFG.device,
    }
    with open(SAVE_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Persist training curves for the paper figures.
    with open(METRICS_DIR / "training_metrics.json", "w") as f:
        json.dump({"world_model": wm_hist, "ensemble": ens_hist,
                   "summary": report}, f, indent=2)

    print("\n" + "=" * 68)
    print("  FINAL METRICS")
    print("=" * 68)
    for k, v in report.items():
        print(f"  {k:<22s} {v:.4f}" if isinstance(v, float) else f"  {k:<22s} {v}")
    print(f"\n  Checkpoints: {CKPT_DIR}")
    print(f"  Metrics:     {SAVE_DIR/'metrics_report.json'}")
    print(f"  Curves:      {METRICS_DIR/'training_metrics.json'}")
    print(f"  Runtime:     {train_min:.1f} min")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
