import torch
from grapes_shap.config import Config
from grapes_shap.models.world_model import LatentWorldModel
from grapes_shap.models.ensemble import DeepEnsemble

class ToTPlanner:
    """Tree-of-Thought planner.

    Performs a beam search over the action tree using the trained world model
    as the transition function and its reward head as the step-value. Each
    depth expands every beam node over the action space, scores the resulting
    latent transition with the (trained) reward head, and keeps the top-`W`
    partial plans. Terminal plans are ranked by the deep-ensemble outcome value
    minus an uncertainty penalty plus the accumulated reward.
    """
    def __init__(self, wm: LatentWorldModel, ens: DeepEnsemble, cfg: Config):
        self.wm, self.ens, self.cfg = wm, ens, cfg

    @torch.no_grad()
    def plan(self, z0, g):
        dev = z0.device
        A   = self.cfg.action_dim
        W   = max(1, self.cfg.plan_candidates)        # beam width
        H   = self.cfg.plan_horizon                   # search depth
        n_branch = min(A, max(W, self.cfg.plan_candidates))
        g1  = g.expand(z0.shape[0], -1)

        # Beam entries: (cumulative_reward, z_state, gru_hidden, action_list)
        beam = [(0.0, z0, None, [])]
        for _ in range(H):
            cand = []
            for cum, z, h, acts in beam:
                for a in range(n_branch):
                    a_t = torch.tensor([a], device=dev).expand(z.shape[0])
                    z_next, h_new, _ = self.wm.step(z, a_t, g1, h)
                    r = self.wm.reward_head(
                        torch.cat([z, z_next], dim=-1)).mean().item()
                    cand.append((cum + r, z_next, h_new,
                                 acts + [torch.tensor(a, device=dev)]))
            cand.sort(key=lambda x: -x[0])
            beam = cand[:W]

        # Rank terminal plans with ensemble value − uncertainty + accumulated reward.
        best = {"score": float("-inf"), "actions": None, "mu": None, "std": None}
        for cum, z, h, acts in beam:
            mu, std, ep, al = self.ens(z)
            val   = mu[0,0] - 0.3*mu[0,1] - 0.2*mu[0,2]
            pen   = 0.1 * std[0].mean()
            score = float(val + 0.5*cum - pen)
            if score > best["score"]:
                best.update({"score":score, "actions":acts, "mu":mu, "std":std})
        return best
