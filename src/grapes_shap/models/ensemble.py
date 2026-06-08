import torch
import torch.nn as nn
from grapes_shap.config import Config

class ProbHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 128), nn.GELU())
        self.mu     = nn.Linear(128, cfg.n_outcomes)
        self.logvar = nn.Linear(128, cfg.n_outcomes)

    def forward(self, z):
        h = self.net(z)
        return self.mu(h), self.logvar(h).clamp(-10, 4)


class DeepEnsemble(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.members = nn.ModuleList([ProbHead(cfg) for _ in range(cfg.n_ensemble)])

    def forward(self, z):
        mus, lvs = zip(*[m(z) for m in self.members])
        mus = torch.stack(mus)
        mu  = mus.mean(0)
        ep  = mus.var(0)
        al  = torch.stack(lvs).exp().mean(0)
        return mu, (ep+al).sqrt(), ep.sqrt(), al.sqrt()

    def nll_loss(self, z, targets):
        total = torch.tensor(0., device=z.device)
        for m in self.members:
            mu, lv = m(z)
            total = total + (0.5*(lv + (targets-mu).pow(2)/lv.exp())).mean()
        return total / len(self.members)
