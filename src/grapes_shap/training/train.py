import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from grapes_shap.config import Config, CKPT_DIR

def train_world_model(wm, enc, gnn, kg, loader, cfg: Config):
    params = list(wm.parameters()) + list(enc.parameters()) + list(gnn.parameters())
    opt   = torch.optim.AdamW(params, lr=cfg.wm_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.wm_lr, steps_per_epoch=len(loader), epochs=cfg.wm_epochs)
    scaler = GradScaler("cuda")
    wm.train(); enc.train(); gnn.train()
    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
    history = {"loss": [], "recon_loss": [], "lr": []}

    for ep in range(cfg.wm_epochs):
        ep_loss, ep_recon = 0.0, 0.0
        for batch in tqdm(loader, desc=f"  WM {ep+1}/{cfg.wm_epochs}", leave=False):
            obs  = batch["obs"].to(cfg.device)
            acts = batch["actions"].to(cfg.device)
            nobs = batch["next_obs"].to(cfg.device)
            B    = obs.shape[0]
            nf_b = nf.unsqueeze(0).expand(B,-1,-1)
            ad_b = adj.unsqueeze(0).expand(B,-1,-1)
            ew_b = ew.unsqueeze(0).expand(B,-1,-1)
            with autocast("cuda", dtype=cfg.amp_dtype):
                _, g = gnn(nf_b, ad_b, ew_b, mask)
                z_seq = enc(obs, g)
                obs_pred, z_preds, sigmas = wm(z_seq, acts, g)
                recon  = F.mse_loss(obs_pred, nobs)
                smooth = (z_preds[:,1:] - z_preds[:,:-1]).pow(2).mean()
                loss   = recon + 0.01*smooth + 0.001*sigmas.mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss  += loss.item()
            ep_recon += recon.item()
        avg_loss  = ep_loss  / len(loader)
        avg_recon = ep_recon / len(loader)
        history["loss"].append(avg_loss)
        history["recon_loss"].append(avg_recon)
        history["lr"].append(sched.get_last_lr()[0])
        print(f"  WM epoch {ep+1:2d} | loss={avg_loss:.5f} | recon={avg_recon:.5f}")

    torch.save({"wm": wm.state_dict(), "enc": enc.state_dict(),
                "gnn": gnn.state_dict()}, CKPT_DIR / "world_model.pt")
    return history


def train_ensemble(ens, enc, gnn, kg, loader, cfg: Config):
    opt   = torch.optim.AdamW(ens.parameters(), lr=cfg.pred_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pred_epochs)
    scaler = GradScaler("cuda")
    ens.train(); enc.eval(); gnn.eval()
    nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
    history = {"loss": []}

    for ep in range(cfg.pred_epochs):
        ep_loss = 0.0
        for batch in tqdm(loader, desc=f"  Ens {ep+1}/{cfg.pred_epochs}", leave=False):
            obs      = batch["obs"].to(cfg.device)
            outcomes = batch["outcomes"].to(cfg.device)
            B = obs.shape[0]
            nf_b = nf.unsqueeze(0).expand(B,-1,-1)
            ad_b = adj.unsqueeze(0).expand(B,-1,-1)
            ew_b = ew.unsqueeze(0).expand(B,-1,-1)
            with torch.no_grad():
                _, g  = gnn(nf_b, ad_b, ew_b, mask)
                z_seq = enc(obs, g)
            z_last = z_seq[:,-1,:]
            with autocast("cuda", dtype=cfg.amp_dtype):
                loss = ens.nll_loss(z_last, outcomes)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(ens.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()
        sched.step()
        avg = ep_loss / len(loader)
        history["loss"].append(avg)
        print(f"  Ens epoch {ep+1:2d} | nll={avg:.5f}")

    torch.save(ens.state_dict(), CKPT_DIR / "ensemble.pt")
    return history
