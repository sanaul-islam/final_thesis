import numpy as np
from grapes_shap.config import Config

class SHAPAttributor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._proxy = self._load_proxy()

    def _load_proxy(self):
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        except Exception:
            return None

    def _score(self, query, subset):
        if not subset:
            return 0.0
        if self._proxy:
            try:
                return float(self._proxy.predict([(query, " ".join(subset))])[0])
            except Exception:
                pass
        q_tok = set(query.lower().split())
        d_tok = set(" ".join(subset).lower().split())
        return len(q_tok & d_tok) / (len(q_tok) + 1)

    def shapley(self, query, docs):
        K   = len(docs)
        phi = np.zeros(K)
        b   = self._score(query, [])
        for _ in range(self.cfg.shap_perms):
            perm = np.random.permutation(K)
            S, v = [], b
            for idx in perm:
                S.append(idx)
                v_new = self._score(query, [docs[j] for j in S])
                phi[idx] += v_new - v
                v = v_new
        return phi / self.cfg.shap_perms
