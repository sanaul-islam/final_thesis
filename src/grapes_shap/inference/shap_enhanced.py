"""
SHAP Attribution with Pairwise Interactions

Enhanced SHAP implementation that includes:
1. Shapley values (individual document contribution)
2. Pairwise interaction effects (synergies between documents)
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from grapes_shap.config import Config
from itertools import combinations


@dataclass
class SHAPResult:
    """SHAP attribution results."""
    document_shapley: np.ndarray  # Shapley values for each doc
    pairwise_interactions: Dict[Tuple[int, int], float]  # (i, j) -> interaction value
    total_attribution_check: float  # Sum of all attributions (should ≈ model output)


class SHAPAttributor:
    """Enhanced SHAP with pairwise interactions for RAG interpretability."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_proxy()
    
    def _load_proxy(self):
        """Load cross-encoder as proxy scoring model."""
        try:
            from sentence_transformers import CrossEncoder
            self.proxy_model = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.cfg.device
            )
        except Exception as e:
            print(f"Warning: CrossEncoder not loaded for SHAP: {e}")
            self.proxy_model = None
    
    def _score_subset(
        self,
        query: str,
        doc_indices: List[int],
        all_docs: List[str]
    ) -> float:
        """
        Score a subset of documents for a query.
        
        This is the scoring function used for SHAP computation.
        Returns a value in [0, 1] representing how well this doc subset
        answers the query.
        
        Args:
            query: Clinical query
            doc_indices: Indices of selected documents
            all_docs: All available documents
            
        Returns:
            Score in [0, 1]
        """
        if not doc_indices:
            return 0.0
        
        # Select documents
        selected_docs = [all_docs[i] for i in doc_indices]
        subset_text = " ".join(selected_docs)
        
        # Score using cross-encoder if available
        if self.proxy_model is not None:
            try:
                score = float(self.proxy_model.predict([(query, subset_text)])[0])
                # Normalize to [0, 1] if needed
                return (score + 1.0) / 2.0  # Approximate normalization
            except Exception:
                pass
        
        # Fallback: keyword overlap
        query_tokens = set(query.lower().split())
        doc_tokens = set(subset_text.lower().split())
        overlap = len(query_tokens & doc_tokens) / (len(query_tokens) + 1e-8)
        
        return min(1.0, overlap)
    
    def compute_shapley_values(
        self,
        query: str,
        documents: List[str],
        n_permutations: int = 32
    ) -> np.ndarray:
        """
        Compute Shapley values for each document.
        
        Shapley value = average marginal contribution of a document
        across all possible orderings of the other documents.
        
        Formula:
        φ_i = (1/P) * Σ_{perm} [f(S_perm ∪ {i}) - f(S_perm)]
        where S_perm = documents before i in random permutation
        
        Args:
            query: Clinical query
            documents: List of documents
            n_permutations: Number of random permutations (default 32)
            
        Returns:
            Shapley values array (shape: num_docs)
        """
        K = len(documents)
        phi = np.zeros(K)
        
        baseline_score = self._score_subset(query, [], documents)
        
        for _ in range(n_permutations):
            # Random permutation of document indices
            perm = np.random.permutation(K)
            
            # Build up subsets and compute marginal contributions
            S = []  # Currently selected
            v_prev = baseline_score
            
            for idx in perm:
                S.append(idx)
                v_new = self._score_subset(query, S, documents)
                phi[idx] += (v_new - v_prev)
                v_prev = v_new
        
        # Average across permutations
        phi /= n_permutations
        
        return phi
    
    def compute_pairwise_interactions(
        self,
        query: str,
        documents: List[str],
        shapley_values: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute pairwise interaction effects (synergies).
        
        For documents i and j:
        I(i,j) = f({i,j}) - f({i}) - f({j}) + f({})
        
        This measures how much more (or less) value i and j provide together
        compared to independent contributions.
        
        Positive I(i,j) = synergistic (work well together)
        Negative I(i,j) = redundant (overlap in coverage)
        
        Args:
            query: Clinical query
            documents: List of documents
            shapley_values: Precomputed Shapley values
            
        Returns:
            Dict mapping (i, j) -> interaction value
        """
        K = len(documents)
        interactions = {}
        
        # Baseline score (no docs)
        f_empty = self._score_subset(query, [], documents)
        
        # For computational efficiency, only compute interactions for
        # top-contributing documents
        top_k = min(6, K)
        top_indices = np.argsort(np.abs(shapley_values))[-top_k:]
        
        for i, j in combinations(top_indices, 2):
            # Score individual docs
            f_i = self._score_subset(query, [i], documents)
            f_j = self._score_subset(query, [j], documents)
            
            # Score together
            f_ij = self._score_subset(query, [i, j], documents)
            
            # Interaction effect
            interaction = f_ij - f_i - f_j + f_empty
            interactions[(i, j)] = interaction
        
        return interactions
    
    def shapley_with_interactions(
        self,
        query: str,
        documents: List[str],
        n_permutations: int = 32
    ) -> SHAPResult:
        """
        Compute complete SHAP with pairwise interactions.
        
        Args:
            query: Clinical query
            documents: List of documents
            n_permutations: Number of Monte Carlo permutations
            
        Returns:
            SHAPResult with Shapley values and interactions
        """
        # Compute Shapley values
        shapley = self.compute_shapley_values(
            query,
            documents,
            n_permutations=n_permutations
        )
        
        # Compute pairwise interactions
        interactions = self.compute_pairwise_interactions(
            query,
            documents,
            shapley
        )
        
        # Compute total attribution for verification
        # (should approximately equal model output)
        baseline = self._score_subset(query, [], documents)
        total = baseline + np.sum(shapley)
        
        return SHAPResult(
            document_shapley=shapley,
            pairwise_interactions=interactions,
            total_attribution_check=total
        )
    
    def interpret_attribution(
        self,
        query: str,
        documents: List[str],
        shap_result: SHAPResult,
        top_k: int = 3
    ) -> Dict:
        """
        Generate human-readable interpretation of SHAP results.
        
        Args:
            query: Clinical query
            documents: Document list
            shap_result: SHAP computation result
            top_k: Number of top documents to highlight
            
        Returns:
            Dict with interpretation
        """
        shapley = shap_result.document_shapley
        interactions = shap_result.pairwise_interactions
        
        # Rank documents by absolute Shapley value
        doc_rankings = np.argsort(np.abs(shapley))[::-1]
        
        interpretation = {
            "summary": [],
            "top_documents": [],
            "synergies": [],
            "redundancies": []
        }
        
        # Top contributing documents
        for rank, idx in enumerate(doc_rankings[:top_k]):
            contribution = shapley[idx]
            direction = "positive" if contribution > 0 else "negative"
            magnitude = abs(contribution)
            
            doc_preview = documents[idx][:100] + "..." \
                         if len(documents[idx]) > 100 else documents[idx]
            
            interpretation["top_documents"].append({
                "rank": rank + 1,
                "index": int(idx),
                "document": doc_preview,
                "shapley_value": float(contribution),
                "direction": direction,
                "magnitude": float(magnitude)
            })
        
        # Pairwise synergies and redundancies
        for (i, j), interaction in sorted(
            interactions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ):
            if abs(interaction) > 0.05:  # Only report meaningful interactions
                if interaction > 0:
                    interpretation["synergies"].append({
                        "doc1_index": int(i),
                        "doc2_index": int(j),
                        "interaction_value": float(interaction),
                        "interpretation": f"Documents {i} and {j} work well together"
                    })
                else:
                    interpretation["redundancies"].append({
                        "doc1_index": int(i),
                        "doc2_index": int(j),
                        "interaction_value": float(interaction),
                        "interpretation": f"Documents {i} and {j} are redundant"
                    })
        
        # Summary
        total_positive = np.sum(shapley[shapley > 0])
        total_negative = np.sum(shapley[shapley < 0])
        
        interpretation["summary"] = [
            f"Total documents analyzed: {len(documents)}",
            f"Top contributing document (φ={float(shapley[doc_rankings[0]]):.3f}): {documents[doc_rankings[0]][:80]}",
            f"Positive contributions: {float(total_positive):.3f}",
            f"Negative contributions: {float(total_negative):.3f}",
            f"Key synergies found: {len(interpretation['synergies'])}",
            f"Redundancies found: {len(interpretation['redundancies'])}"
        ]
        
        return interpretation
    
    # Legacy interface (for backward compatibility)
    def shapley(self, query: str, docs: List[str]) -> np.ndarray:
        """Legacy interface - compute Shapley values only."""
        return self.compute_shapley_values(
            query,
            docs,
            n_permutations=self.cfg.shap_perms
        )
