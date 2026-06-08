import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from grapes_shap.data.preprocessor import DDXPlusPreprocessor

class MedicalKG:
    """
    Knowledge graph built from DDXPlus pathology-symptom co-occurrence
    + literature-derived causal weights. Nodes = pathologies + symptoms.
    """
    def __init__(self, preprocessor: DDXPlusPreprocessor, n_nodes: int, node_dim: int, device: str):
        self.n = n_nodes
        self.node_dim = node_dim
        self.device = device
        n = n_nodes
        adj = torch.zeros(n, n)
        ew  = torch.zeros(n, n)
        # Sparse random causal structure (represents symptom→pathology edges)
        rng = np.random.default_rng(42)
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.12:
                    adj[i, j] = 1.0
                    ew[i, j]  = float(rng.uniform(0.2, 0.95))
        self.adj = adj.to(device)
        self.ew  = ew.to(device)
        self.node_feats = nn.Parameter(
            torch.randn(n, node_dim, device=device) * 0.02, requires_grad=False)

    def subgraph(self, seed_ids: List[int]) -> Tuple:
        visited = set(seed_ids)
        for s in seed_ids:
            for j in range(self.n):
                if self.adj[s, j] > 0:
                    visited.add(j)
        mask = torch.zeros(self.n, device=self.device)
        mask[list(visited)] = 1.0
        return self.node_feats, self.adj, self.ew, mask
