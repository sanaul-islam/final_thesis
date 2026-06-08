"""
Re-ranking and Diversity Module - Cross-encoder + Maximum Marginal Relevance

This module implements:
1. Cross-encoder re-ranking for precision (RRF candidates → top-k)
2. Maximum Marginal Relevance (MMR) for diversity
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from grapes_shap.config import Config


@dataclass
class RankedDocument:
    doc: str
    score: float
    rank: int
    provider: str  # "dense", "bm25", or "cross_encoder"


class ReRankerMMR:
    """Cross-encoder re-ranking + MMR diversity sampling."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_cross_encoder()
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for precise relevance scoring."""
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.cfg.device
            )
        except Exception as e:
            print(f"Warning: CrossEncoder not loaded: {e}")
            self.cross_encoder = None
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 20,
        fusion_k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine dense and BM25 rankings using Reciprocal Rank Fusion.
        
        RRF formula: score = Σ 1/(k + rank)
        - k=60 prevents top-1 from dominating
        - Rewards documents appearing highly in BOTH lists
        
        Args:
            dense_results: List of (doc, score) from dense search
            bm25_results: List of (doc, score) from BM25 search
            k: Number of top results to consider
            fusion_k: RRF k parameter (default 60)
            
        Returns:
            Combined ranked list of (doc, rrf_score)
        """
        # Create rank dictionaries
        rrf_scores = {}
        
        # Add dense results
        for rank, (doc, _) in enumerate(dense_results[:k], 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (fusion_k + rank)
        
        # Add BM25 results
        for rank, (doc, _) in enumerate(bm25_results[:k], 1):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (fusion_k + rank)
        
        # Sort by RRF score
        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return combined
    
    def cross_encoder_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 6
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using cross-encoder.
        
        Cross-encoder reads query+document together (full interaction),
        unlike bi-encoder which embeds separately.
        
        Args:
            query: Clinical query
            documents: List of candidate documents
            top_k: Return top k documents
            
        Returns:
            Top-k re-ranked documents with scores
        """
        if self.cross_encoder is None or not documents:
            # Fallback: return first k docs
            return [(doc, 1.0) for doc in documents[:top_k]]
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Score all pairs
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by score (descending)
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return ranked[:top_k]
        
        except Exception as e:
            print(f"Cross-encoder re-ranking failed: {e}")
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    def maximum_marginal_relevance(
        self,
        query: str,
        candidates: List[str],
        embeddings: np.ndarray,
        lambda_param: float = 0.6,
        top_k: int = 6
    ) -> List[Tuple[str, float, float]]:
        """
        Select top_k documents balancing relevance and diversity.
        
        MMR formula: MMR(d) = λ·Sim(d, query) - (1-λ)·max(Sim(d, S))
        where S = already selected documents
        
        Args:
            query: Clinical query
            candidates: Candidate documents (pre-ranked by relevance)
            embeddings: Document embeddings (batch, dim)
            lambda_param: Relevance weight (0.6 = 60% relevance, 40% diversity)
            top_k: Number of documents to select
            
        Returns:
            List of (doc, relevance_score, mmr_score)
        """
        if len(candidates) <= top_k:
            return [(doc, 1.0, 1.0) for doc in candidates]
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Embed query
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device=self.cfg.device
                )
                query_emb = encoder.encode(query, convert_to_numpy=True).reshape(1, -1)
            except:
                # Fallback: use first embedding as proxy
                query_emb = embeddings[0:1]
            
            # Calculate query-document similarities (relevance)
            relevance = cosine_similarity(query_emb, embeddings).flatten()
            
            selected_indices = []
            selected_docs = []
            remaining_indices = list(range(len(candidates)))
            
            # Greedy selection: iteratively pick best MMR score
            for _ in range(min(top_k, len(candidates))):
                best_mmr = -np.inf
                best_idx = 0
                
                for i, idx in enumerate(remaining_indices):
                    # Relevance component
                    rel_score = relevance[idx]
                    
                    # Diversity component: minimize similarity to selected docs
                    if selected_indices:
                        selected_embs = embeddings[selected_indices]
                        similarities = cosine_similarity(
                            embeddings[idx:idx+1],
                            selected_embs
                        ).flatten()
                        diversity = np.max(similarities) if len(similarities) > 0 else 0
                    else:
                        diversity = 0
                    
                    # MMR score
                    mmr_score = lambda_param * rel_score - (1 - lambda_param) * diversity
                    
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = i
                
                selected_idx = remaining_indices.pop(best_idx)
                selected_indices.append(selected_idx)
                selected_docs.append(candidates[selected_idx])
            
            return [
                (doc, relevance[idx], 1.0)
                for doc, idx in zip(selected_docs, selected_indices)
            ]
        
        except Exception as e:
            print(f"MMR selection failed: {e}")
            # Fallback: return first k documents
            return [(doc, 1.0, 1.0) for doc in candidates[:top_k]]
    
    def rerank_pipeline(
        self,
        query: str,
        dense_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        embeddings: np.ndarray,
        lambda_param: float = 0.6,
        top_k: int = 6
    ) -> List[Tuple[str, float, str]]:
        """
        Complete re-ranking pipeline: RRF → Cross-encoder → MMR.
        
        Args:
            query: Clinical query
            dense_results: Results from dense search
            bm25_results: Results from BM25 search
            embeddings: Document embeddings for MMR
            lambda_param: MMR relevance weight
            top_k: Final number of documents to return
            
        Returns:
            List of (doc, final_score, method)
        """
        # Step 1: RRF combination
        rrf_combined = self.reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            k=20
        )
        rrf_docs = [doc for doc, _ in rrf_combined]
        
        # Step 2: Cross-encoder re-ranking (precision)
        cross_ranked = self.cross_encoder_rerank(
            query,
            rrf_docs,
            top_k=top_k * 2  # Get more for MMR selection
        )
        
        # Step 3: MMR for diversity
        mmr_selected = self.maximum_marginal_relevance(
            query,
            [doc for doc, _ in cross_ranked],
            embeddings[:len(cross_ranked)],
            lambda_param=lambda_param,
            top_k=top_k
        )
        
        # Return with scores
        results = [
            (doc, score, "mmr+cross_encoder")
            for doc, rel_score, mmr_score in mmr_selected
        ]
        
        return results[:top_k]
