import torch
import torch.nn as nn
import torch.nn.functional as F
from grapes_shap.config import Config

class CausalResidual(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.a_emb = nn.Embedding(cfg.action_dim, cfg.latent_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(cfg.latent_dim*3, cfg.latent_dim*2), nn.GELU(),
            nn.Linear(cfg.latent_dim*2, cfg.latent_dim))
        self.gate  = nn.Sequential(nn.Linear(cfg.latent_dim*2, cfg.latent_dim), nn.Sigmoid())
        self.scale = nn.Parameter(torch.ones(1)*0.1)

    def forward(self, z, a, g):
        ae  = self.a_emb(a)
        h   = self.mlp(torch.cat([z, g, ae], dim=-1))
        gv  = self.gate(torch.cat([z, h], dim=-1))
        return self.scale * gv * h


class LatentWorldModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.causal_res = CausalResidual(cfg)
        self.a_emb      = nn.Embedding(cfg.action_dim, cfg.latent_dim)
        self.gru        = nn.GRU(cfg.latent_dim*2, cfg.hidden_dim,
                                  num_layers=3, batch_first=True, dropout=cfg.dropout)
        self.h2z        = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.latent_dim),
                                         nn.LayerNorm(cfg.latent_dim), nn.SiLU())
        self.decoder    = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim//2), nn.GELU(),
            nn.Linear(cfg.hidden_dim//2, cfg.obs_dim))
        self.sigma_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim//2), nn.GELU(),
            nn.Linear(cfg.latent_dim//2, cfg.latent_dim), nn.Softplus())
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.latent_dim*2, 128), nn.GELU(), nn.Linear(128, 1))

    def step(self, z, a, g, h=None):
        delta = self.causal_res(z, a, g)
        ae    = self.a_emb(a)
        inp   = torch.cat([z+delta, ae], dim=-1).unsqueeze(1)
        out, h_new = self.gru(inp, h)
        z_next = self.h2z(out.squeeze(1))
        sigma  = self.sigma_head(z_next)
        return z_next, h_new, sigma

    def forward(self, z_seq, actions, g):
        B, T, _ = z_seq.shape
        preds, sigmas, h = [], [], None
        for t in range(T):
            g_t = g.expand(B,-1) if g.dim()==1 else g
            z_next, h, sigma = self.step(z_seq[:,t], actions[:,t], g_t, h)
            preds.append(z_next); sigmas.append(sigma)
        z_preds  = torch.stack(preds, 1)
        obs_pred = self.decoder(z_preds)
        return obs_pred, z_preds, torch.stack(sigmas, 1)

    @torch.no_grad()
    def rollout(self, z0, action_seq, g):
        z, h, traj, rewards = z0, None, [z0], []
        for a in action_seq:
            a_t = a.view(z.shape[0]) if a.dim()>0 else a.unsqueeze(0).expand(z.shape[0])
            z_prev = z
            z, h, _ = self.step(z, a_t, g.expand(z.shape[0],-1), h)
            traj.append(z)
            rewards.append(self.reward_head(torch.cat([z_prev, z], -1)))
        return traj, torch.stack(rewards)
