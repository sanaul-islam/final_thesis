import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from grapes_shap.config import Config

class EdgeBiasedGAT(nn.Module):
    def __init__(self, in_d, out_d, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dh = out_d // heads
        self.Wq = nn.Linear(in_d, out_d)
        self.Wk = nn.Linear(in_d, out_d)
        self.Wv = nn.Linear(in_d, out_d)
        self.We = nn.Linear(1, heads)
        self.proj = nn.Linear(out_d, out_d)
        self.norm = nn.LayerNorm(out_d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, ew):
        B, N, _ = x.shape
        Q = self.Wq(x).view(B,N,self.heads,self.dh).transpose(1,2)
        K = self.Wk(x).view(B,N,self.heads,self.dh).transpose(1,2)
        V = self.Wv(x).view(B,N,self.heads,self.dh).transpose(1,2)
        s = Q @ K.transpose(-2,-1) / math.sqrt(self.dh)
        s = s + self.We(ew.unsqueeze(-1)).permute(0,3,1,2)
        # Add self-loops so every node always attends to itself. Without this,
        # nodes with no incoming edges produce an all-masked row whose softmax
        # is NaN (which then propagates through the whole network).
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj_sl = (adj + eye).clamp(max=1.0)
        # Use a large finite negative value instead of -inf for fp16 safety.
        neg = torch.finfo(s.dtype).min
        s = s.masked_fill((adj_sl == 0).unsqueeze(1), neg)
        a = self.drop(F.softmax(s, dim=-1))
        out = (a @ V).transpose(1,2).contiguous().view(B,N,-1)
        return self.norm(x + self.proj(out))


class CausalGNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Linear(cfg.graph_node_dim, cfg.latent_dim)
        self.layers = nn.ModuleList([
            EdgeBiasedGAT(cfg.latent_dim, cfg.latent_dim, heads=4, dropout=cfg.dropout)
            for _ in range(3)
        ])
        self.pool = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, feats, adj, ew, mask):
        B = feats.shape[0] if feats.dim()==3 else 1
        if feats.dim()==2:
            feats = feats.unsqueeze(0).expand(B,-1,-1)
            adj   = adj.unsqueeze(0).expand(B,-1,-1)
            ew    = ew.unsqueeze(0).expand(B,-1,-1)
        x = self.proj(feats)
        for l in self.layers:
            x = l(x, adj, ew)
        m  = mask.view(1,-1,1)
        g  = self.pool(self.norm((x*m).sum(1) / (mask.sum()+1e-8)))
        return x, g
