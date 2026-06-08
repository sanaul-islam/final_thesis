"""Reduced-scale end-to-end smoke test to catch errors fast on GPU before a full run."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from grapes_shap.config import CFG, SAVE_DIR, FIG_DIR, CKPT_DIR

# ---- shrink everything for a fast smoke test ----
CFG.ddxplus_n_train = 800
CFG.ddxplus_n_val = 200
CFG.ddxplus_n_test = 200
CFG.medmcqa_n_train = 2000
CFG.medqa_n_test = 50
CFG.wm_epochs = 2
CFG.pred_epochs = 2
CFG.batch_size = 64

from grapes_shap.data import (
    DatasetLoader, DDXPlusPreprocessor, MedMCQAPreprocessor,
    MedQAPreprocessor, ClinicalTrajectoryDataset,
)
from grapes_shap.models import MedicalKG, CausalGNN, EvidenceFusionEncoder, LatentWorldModel, DeepEnsemble
from grapes_shap.inference import HybridRetriever, SHAPAttributor, full_inference_pipeline
from grapes_shap.training import train_world_model, train_ensemble, evaluate_all


def main():
    print("=" * 60)
    print("SMOKE TEST  |  device:", CFG.device)
    print("=" * 60)
    t0 = time.time()

    print("[1] Loading datasets (reduced)...")
    train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)
    medmcqa_raw = DatasetLoader.load_medmcqa(CFG.medmcqa_n_train)
    medqa_raw = DatasetLoader.load_medqa(CFG.medqa_n_test)

    print("[2] Preprocessing...")
    pre = DDXPlusPreprocessor(CFG)
    pre.fit(train_raw)
    tr = pre.transform(train_raw)
    va = pre.transform(val_raw)
    te = pre.transform(test_raw)
    print(f"  processed train:{len(tr)} val:{len(va)} test:{len(te)}")

    docs = MedMCQAPreprocessor.to_documents(medmcqa_raw)
    queries = MedQAPreprocessor.to_queries(medqa_raw)
    print(f"  docs:{len(docs)} queries:{len(queries)}")

    print("[3] Building retriever...")
    retr = HybridRetriever(CFG)
    retr.build(docs[:2000])

    print("[4] Init models...")
    kg = MedicalKG(pre, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
    gnn = CausalGNN(CFG).to(CFG.device)
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    wm = LatentWorldModel(CFG).to(CFG.device)
    ens = DeepEnsemble(CFG).to(CFG.device)
    total = sum(p.numel() for m in [gnn, enc, wm, ens] for p in m.parameters())
    print(f"  params:{total:,}")
    if torch.cuda.is_available():
        print(f"  VRAM:{torch.cuda.memory_allocated()/1e9:.2f} GB")

    ds_tr = ClinicalTrajectoryDataset(tr)
    ds_va = ClinicalTrajectoryDataset(va)
    dl_tr = DataLoader(ds_tr, batch_size=CFG.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    print("[5] Train world model...")
    wm_hist = train_world_model(wm, enc, gnn, kg, dl_tr, CFG)
    print("  wm_hist keys:", list(wm_hist.keys()))

    print("[6] Train ensemble...")
    ens_hist = train_ensemble(ens, enc, gnn, kg, dl_tr, CFG)
    print("  ens_hist keys:", list(ens_hist.keys()))

    print("[7] Evaluate...")
    shap_attr = SHAPAttributor(CFG)
    metrics = evaluate_all(ens, enc, gnn, kg, dl_va, queries, retr, shap_attr, CFG)
    print("  metrics:", {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, (int, float))})

    print("[8] Full inference pipeline...")
    q = queries[0]["question"] if queries else "Patient with chest pain and ST elevation. Best treatment?"
    res = full_inference_pipeline(q, wm, enc, gnn, kg, ens, retr, shap_attr, CFG)
    print("  result keys:", list(res.keys()))
    print("  n docs:", len(res["docs"]), "plan score:", round(res["plan"]["score"], 4))

    print(f"\nSMOKE TEST PASSED in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
