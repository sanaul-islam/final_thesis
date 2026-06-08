import sys
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from grapes_shap.config import CFG, SAVE_DIR, FIG_DIR, CKPT_DIR
from grapes_shap.data import DatasetLoader, DDXPlusPreprocessor, MedMCQAPreprocessor, MedQAPreprocessor, ClinicalTrajectoryDataset
from grapes_shap.models import MedicalKG, CausalGNN, EvidenceFusionEncoder, LatentWorldModel, DeepEnsemble
from grapes_shap.inference import HybridRetriever, SHAPAttributor, full_inference_pipeline
from grapes_shap.training import train_world_model, train_ensemble, evaluate_all
from grapes_shap.visualization import (
    plot_dataset_overview,
    plot_training_history,
    plot_performance_dashboard,
    plot_latent_space,
    plot_inference_results
)

def main():
    print("\n" + "="*68)
    print("  GRAPES-SHAP  |  Full Research Pipeline")
    print("  Datasets: DDXPlus + MedMCQA + MedQA (zero-barrier)")
    print("="*68 + "\n")
    t0 = time.time()

    # ── Step 1: Load datasets ──
    print("[1/9] Loading datasets...")
    train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)
    medmcqa_raw  = DatasetLoader.load_medmcqa(CFG.medmcqa_n_train)
    medqa_raw    = DatasetLoader.load_medqa(CFG.medqa_n_test)

    # ── Step 2: Preprocess ──
    print("\n[2/9] Preprocessing...")
    preprocessor = DDXPlusPreprocessor(CFG)
    preprocessor.fit(train_raw)

    processed_train = preprocessor.transform(train_raw)
    processed_val   = preprocessor.transform(val_raw)
    processed_test  = preprocessor.transform(test_raw)
    print(f"  Processed - train:{len(processed_train)} val:{len(processed_val)} test:{len(processed_test)}")

    medmcqa_docs  = MedMCQAPreprocessor.to_documents(medmcqa_raw)
    medqa_queries = MedQAPreprocessor.to_queries(medqa_raw)
    print(f"  RAG corpus: {len(medmcqa_docs):,} MedMCQA documents")
    print(f"  Eval queries: {len(medqa_queries)} MedQA USMLE questions")

    # ── Step 3: Data exploration visualisation ──
    print("\n[3/9] Data exploration visualisation...")
    plot_dataset_overview(processed_train, processed_val, processed_test, preprocessor, CFG)

    # ── Step 4: Build retrieval index ──
    print("\n[4/9] Building hybrid retrieval index (Dense + BM25)...")
    retriever = HybridRetriever(CFG)
    retriever.build(medmcqa_docs[:30_000])

    # ── Step 5: Initialise architecture ──
    print("\n[5/9] Initialising GRAPES-SHAP architecture...")
    kg  = MedicalKG(preprocessor, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device)
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    wm  = LatentWorldModel(CFG).to(CFG.device)
    ens = DeepEnsemble(CFG).to(CFG.device)

    total_params = sum(p.numel() for m in [gnn,enc,wm,ens] for p in m.parameters())
    print(f"  Total neural parameters: {total_params:,}")
    if torch.cuda.is_available():
        print(f"  VRAM after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # DataLoaders
    ds_train = ClinicalTrajectoryDataset(processed_train)
    ds_val   = ClinicalTrajectoryDataset(processed_val)
    dl_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=128, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ── Step 6: Training ──
    print("\n[6/9] Training World Model...")
    wm_hist  = train_world_model(wm, enc, gnn, kg, dl_train, CFG)

    print("\n       Training Deep Ensemble...")
    ens_hist = train_ensemble(ens, enc, gnn, kg, dl_train, CFG)

    plot_training_history(wm_hist, ens_hist)

    # ── Step 7: Evaluation ──
    print("\n[7/9] Full evaluation...")
    shap_attr = SHAPAttributor(CFG)
    metrics   = evaluate_all(ens, enc, gnn, kg, dl_val,
                              medqa_queries, retriever, shap_attr, CFG)
    plot_performance_dashboard(metrics)

    # ── Step 8: Latent space visualisation ──
    print("\n[8/9] Latent space t-SNE visualisation...")
    try:
        plot_latent_space(enc, gnn, kg, dl_val, CFG, n_samples=2000)
    except Exception as e:
        print(f"  t-SNE skipped: {e}")

    # ── Step 9: Full inference on MedQA sample + SHAP ──
    print("\n[9/9] Full inference pipeline demonstration...")
    sample_query = medqa_queries[0]["question"] if medqa_queries else (
        "A 62-year-old woman presents with sudden onset chest pain radiating to her left arm. "
        "ECG shows ST elevation in leads V1-V4. Troponin is elevated. "
        "She has a 10-year history of hypertension. What is the best immediate treatment?")
    print(f"  Query: {sample_query[:100]}...")

    outcome_names = ["Dx-1 prob","Dx-2 prob","Dx-3 prob","Dx-4 prob","Dx-5 prob"]
    result = full_inference_pipeline(
        sample_query, wm, enc, gnn, kg, ens, retriever, shap_attr, CFG)
    plan = result["plan"]
    docs = result["docs"]
    shap_vals = result["shap_vals"]

    print(f"\n  Retrieved {len(docs)} documents:")
    for i, d in enumerate(docs):
        print(f"    [{i+1}] {d[:75]}...")

    print(f"\n  ToT best plan score: {plan['score']:.4f}")
    print(f"  Action sequence: {[a.item() for a in plan['actions']] if plan['actions'] else []}")

    if plan["mu"] is not None:
        mu_np  = plan["mu"][0].cpu().numpy()
        std_np = plan["std"][0].cpu().numpy()
        print(f"\n  Predicted outcomes:")
        for n, m, s in zip(outcome_names, mu_np, std_np):
            bar = "#" * int(m*25) + "-" * (25 - int(m*25))
            print(f"    {n:<14s} {m:.3f} +/- {s:.3f}  [{bar}]")

    if len(shap_vals) > 0:
        print(f"\n  SHAP document attributions:")
        ranked = sorted(zip(range(len(docs)), shap_vals.tolist()),
                        key=lambda x: -abs(x[1]))
        for idx, phi in ranked:
            print(f"    Doc[{idx+1}] phi={phi:+.4f}  {docs[idx][:65]}...")

    if docs and len(shap_vals) > 0:
        plot_inference_results(sample_query, docs, shap_vals, plan, outcome_names)

    # ── Save full metrics report ──
    report = {
        "mae":               float(metrics["mae"]),
        "rmse":              float(metrics["rmse"]),
        "1sigma_coverage":   float(metrics["1sigma_coverage"]),
        "ece":               float(metrics["ece"]),
        "accuracy":          float(metrics["accuracy"]),
        "f1_macro":          float(metrics["f1_macro"]),
        "mean_shap":         float(metrics["mean_shap"]),
        "total_params":      total_params,
        "training_time_min": round((time.time()-t0)/60, 2),
        "dataset":           "DDXPlus + MedMCQA + MedQA",
        "device":            CFG.device,
    }
    with open(SAVE_DIR / "metrics_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*68)
    print("  FINAL METRICS SUMMARY")
    print("="*68)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:<28s} {v:.4f}")
        else:
            print(f"  {k:<28s} {v}")
    print(f"\n  Figures saved to: {FIG_DIR.resolve()}")
    print(f"  Checkpoints:      {CKPT_DIR.resolve()}")
    print(f"  Metrics report:   {SAVE_DIR/'metrics_report.json'}")
    print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} minutes")
    print("="*68 + "\n")

if __name__ == "__main__":
    main()
