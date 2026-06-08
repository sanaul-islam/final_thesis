import torch
from grapes_shap.config import Config
from grapes_shap.models.world_model import LatentWorldModel
from grapes_shap.models.ensemble import DeepEnsemble

class ToTPlanner:
    def __init__(self, wm: LatentWorldModel, ens: DeepEnsemble, cfg: Config):
        self.wm, self.ens, self.cfg = wm, ens, cfg

    @torch.no_grad()
    def plan(self, z0, g):
        dev = z0.device
        best = {"score": float("-inf"), "actions": None, "mu": None, "std": None}
        for fa in range(min(self.cfg.action_dim, self.cfg.plan_candidates)):
            seq = [torch.tensor(fa, device=dev)]
            for _ in range(self.cfg.plan_horizon - 1):
                seq.append(torch.randint(0, self.cfg.action_dim, (1,), device=dev).squeeze())
            traj, rewards = self.wm.rollout(z0, seq, g)
            mu, std, ep, al = self.ens(traj[-1])
            val   = mu[0,0] - 0.3*mu[0,1] - 0.2*mu[0,2]
            pen   = 0.1 * std[0].mean()
            score = float(val + 0.5*rewards.sum() - pen)
            if score > best["score"]:
                best.update({"score":score,"actions":seq,"mu":mu,"std":std})
        return best
