import torch
import torch.nn as nn
from grapes_shap.config import Config

class EvidenceFusionEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.obs_proj = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim), nn.SiLU())
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim, nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_transformer_layers)
        self.cross_attn  = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads,
                                                  dropout=cfg.dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(cfg.latent_dim*2, cfg.latent_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, obs, g_emb=None):
        z = self.transformer(self.obs_proj(obs))
        if g_emb is not None:
            kv = g_emb.unsqueeze(1)
            fused, _ = self.cross_attn(z, kv, kv)
            gate = self.gate(torch.cat([z, fused], dim=-1))
            z = self.norm(z + gate * fused)
        return z
