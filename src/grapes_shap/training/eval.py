import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from grapes_shap.config import Config

def evaluate_all(ens, enc, gnn, kg, val_loader, medqa_queries,
                 retriever, shap_attr, cfg: Config):
    ens.eval(); enc.eval(); gnn.eval()
    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
    all_mu, all_std, all_y = [], [], []
    all_ep, all_al = [], []
    all_diag = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Evaluating", leave=False):
            obs      = batch["obs"].to(cfg.device)
            outcomes = batch["outcomes"].to(cfg.device)
            B = obs.shape[0]
            nf_b = nf.unsqueeze(0).expand(B,-1,-1)
            ad_b = adj.unsqueeze(0).expand(B,-1,-1)
            ew_b = ew.unsqueeze(0).expand(B,-1,-1)
            _, g  = gnn(nf_b, ad_b, ew_b, mask)
            z_last = enc(obs, g)[:,-1,:]
            mu, std, ep, al = ens(z_last)
            all_mu.append(mu.cpu()); all_std.append(std.cpu())
            all_y.append(outcomes.cpu())
            all_ep.append(ep.cpu()); all_al.append(al.cpu())
            all_diag.append(batch["diag_class"].cpu())

    mu   = torch.cat(all_mu).numpy()
    std  = torch.cat(all_std).numpy()
    y    = torch.cat(all_y).numpy()
    ep   = torch.cat(all_ep).numpy()
    al   = torch.cat(all_al).numpy()
    diag = torch.cat(all_diag).numpy()

    # Regression metrics per outcome
    mae    = np.abs(mu - y).mean()
    rmse   = np.sqrt(((mu - y)**2).mean())
    cov_1s = float((np.abs(y - mu) < std).mean())
    ece    = float(np.abs(np.abs(y - mu) - std).mean())

    # Diagnosis classification: does the model rank the TRUE pathology highest?
    # `true_class` = rank of the ground-truth pathology within the differential.
    pred_class  = mu.argmax(axis=1)
    true_class  = diag
    acc = accuracy_score(true_class, pred_class)
    f1  = f1_score(true_class, pred_class, average="macro", zero_division=0)

    # SHAP evaluation on 20 MedQA queries
    shap_scores = []
    if medqa_queries and retriever.index is not None:
        for q_item in medqa_queries[:20]:
            q  = q_item["question"]
            docs = retriever.retrieve(q, k=6)
            if docs:
                phi = shap_attr.shapley(q, docs)
                shap_scores.append(float(np.abs(phi).mean()))

    metrics = {
        "mae": mae, "rmse": rmse,
        "1sigma_coverage": cov_1s, "ece": ece,
        "accuracy": acc, "f1_macro": f1,
        "mean_shap": float(np.mean(shap_scores)) if shap_scores else 0.0,
        "mu": mu, "std": std, "y": y,
        "ep": ep, "al": al,
        "pred_class": pred_class, "true_class": true_class,
    }
    print(f"\n  -- Evaluation Results --")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  1-sigma Coverage: {cov_1s:.3f}  (target ~ 0.68)")
    print(f"  ECE:              {ece:.4f}")
    print(f"  Diagnosis Acc:    {acc:.3f}")
    print(f"  F1-macro:         {f1:.3f}")
    print(f"  Mean |SHAP|:      {metrics['mean_shap']:.4f}")
    return metrics
