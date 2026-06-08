import torch
import numpy as np
from typing import Dict

from grapes_shap.config import Config
from grapes_shap.models.kg import MedicalKG
from grapes_shap.models.gnn import CausalGNN
from grapes_shap.models.encoder import EvidenceFusionEncoder
from grapes_shap.models.world_model import LatentWorldModel
from grapes_shap.models.ensemble import DeepEnsemble
from grapes_shap.inference.retriever import HybridRetriever
from grapes_shap.inference.shap import SHAPAttributor
from grapes_shap.inference.planner import ToTPlanner

def full_inference_pipeline(query: str, wm: LatentWorldModel, enc: EvidenceFusionEncoder,
                             gnn: CausalGNN, kg: MedicalKG, ens: DeepEnsemble,
                             retriever: HybridRetriever,
                             shap_attr: SHAPAttributor,
                             cfg: Config) -> Dict:
    dev = cfg.device
    wm.eval(); enc.eval(); gnn.eval(); ens.eval()

    docs = retriever.retrieve(query, k=cfg.top_k)

    seed_ids = list(range(min(5, kg.n)))
    nf, adj, ew, mask = kg.subgraph(seed_ids)
    _, g_emb = gnn(nf.unsqueeze(0), adj.unsqueeze(0), ew.unsqueeze(0), mask)

    init_obs = torch.randn(1, cfg.seq_len, cfg.obs_dim, device=dev) * 0.25
    with torch.no_grad():
        z_seq = enc(init_obs, g_emb)
    z0 = z_seq[:, -1, :]

    planner = ToTPlanner(wm, ens, cfg)
    plan    = planner.plan(z0, g_emb.squeeze(0))

    shap_vals = shap_attr.shapley(query, docs) if docs else np.array([])

    return {"query": query, "docs": docs, "plan": plan,
            "shap_vals": shap_vals, "g_emb": g_emb}
