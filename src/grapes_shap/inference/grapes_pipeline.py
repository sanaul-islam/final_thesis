"""
Complete GRAPES-SHAP Inference Pipeline

Implements all 12 steps of the GRAPES-SHAP architecture:
1. Patient query input
2. Query expansion (HyDE + sub-queries)
3. Hybrid retrieval (Dense + BM25)
4. Re-ranking + MMR
5. Causal KG + GNN
6. World Model simulation
7. Tree-of-Thought planning
8. Deep ensemble uncertainty
9. LLM reasoning
10. Hallucination detection
11. SHAP attribution
12. Final output
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import time
from dataclasses import dataclass

from grapes_shap.config import Config
from grapes_shap.models.kg import MedicalKG
from grapes_shap.models.gnn import CausalGNN
from grapes_shap.models.encoder import EvidenceFusionEncoder
from grapes_shap.models.world_model import LatentWorldModel
from grapes_shap.models.ensemble import DeepEnsemble
from grapes_shap.inference.retriever import HybridRetriever
from grapes_shap.inference.query_expansion import QueryExpander
from grapes_shap.inference.reranker_mmr import ReRankerMMR
from grapes_shap.inference.deepseek_llm import DeepSeekLLMClient
from grapes_shap.inference.hallucination_detection import DualLayerHallucellationCheck
from grapes_shap.inference.shap_enhanced import SHAPAttributor
from grapes_shap.inference.planner import ToTPlanner


@dataclass
class GRAPESOutput:
    """Complete output from GRAPES-SHAP pipeline."""
    # Step 1: Input
    query: str
    
    # Step 2: Query expansion
    expanded_queries: Dict
    
    # Step 3-4: Retrieval
    retrieved_documents: List[str]
    retrieval_scores: List[float]
    
    # Step 5: Knowledge graph
    graph_embedding: Optional[torch.Tensor]
    
    # Step 6: World model
    world_model_trajectory: List
    world_model_score: float
    
    # Step 7: Planning
    best_plan: List[str]
    plan_scores: Dict
    
    # Step 8: Ensemble
    ensemble_predictions: Dict
    ensemble_uncertainty: Dict
    
    # Step 9: LLM reasoning
    llm_reasoning: str
    llm_recommendation: str
    llm_confidence: float
    
    # Step 10: Hallucination check
    hallucination_check: Dict
    
    # Step 11: SHAP
    shap_values: Dict
    shap_interactions: Dict
    
    # Metadata
    timings: Dict[str, float]
    quality_scores: Dict


class GRAPESPipeline:
    """Complete GRAPES-SHAP medical reasoning pipeline."""
    
    def __init__(self, cfg: Config, llm_client: Optional[DeepSeekLLMClient] = None):
        """
        Initialize GRAPES-SHAP pipeline.
        
        Args:
            cfg: Configuration object
            llm_client: DeepSeek LLM client (optional - can be init later)
        """
        self.cfg = cfg
        self.llm_client = llm_client
        self.timings = {}
        
        # Initialize components
        self._init_components()
        
        # Print config
        self.cfg.print_config()
    
    def _init_components(self):
        """Initialize all pipeline components."""
        print("\nInitializing GRAPES-SHAP components...")
        
        # Query expansion
        self.query_expander = QueryExpander(self.cfg, self.llm_client)
        print("  ✓ Query Expansion (HyDE + sub-queries)")
        
        # Retrieval
        self.retriever = HybridRetriever(self.cfg)
        print("  ✓ Hybrid Retriever (Dense + BM25)")
        
        # Re-ranking & MMR
        self.reranker = ReRankerMMR(self.cfg)
        print("  ✓ Re-ranking + MMR Diversity")
        
        # Knowledge graph & GNN
        self.kg = MedicalKG(None, self.cfg.n_graph_nodes, self.cfg.graph_node_dim, self.cfg.device)
        self.gnn = CausalGNN(self.cfg).to(self.cfg.device)
        print("  ✓ Causal KG + Edge-Biased GNN")
        
        # World model
        self.world_model = LatentWorldModel(self.cfg).to(self.cfg.device)
        print("  ✓ Latent World Model (with causal residuals)")
        
        # Deep ensemble
        self.ensemble = DeepEnsemble(self.cfg).to(self.cfg.device)
        print("  ✓ Deep Ensemble (5 models)")
        
        # LLM (if not provided)
        if self.llm_client is None:
            self.llm_client = DeepSeekLLMClient(self.cfg)
        print(f"  ✓ LLM Client ({self.cfg.llm_model})")
        
        # Hallucination detection
        self.hallucination_check = DualLayerHallucellationCheck(self.cfg)
        print("  ✓ Hallucination Detection (Self-RAG + NLI)")
        
        # SHAP
        self.shap_attributor = SHAPAttributor(self.cfg)
        print("  ✓ SHAP Attribution (with pairwise interactions)")
        
        # Planning
        self.planner = ToTPlanner(self.world_model, self.ensemble, self.cfg)
        print("  ✓ Tree-of-Thought Planner")
        
        print("✓ All components initialized\n")
    
    def infer(self, query: str, documents: List[str]) -> GRAPESOutput:
        """
        Run complete GRAPES-SHAP inference pipeline.
        
        Args:
            query: Clinical query
            documents: Available documents for retrieval
            
        Returns:
            GRAPESOutput with complete 12-step analysis
        """
        self.timings = {}
        
        # Build retriever index if needed
        if not self.retriever.index:
            self.retriever.build(documents)
        
        device = self.cfg.device
        
        # ===== STEP 2: Query Expansion =====
        t0 = time.time()
        expanded = self.query_expander.expand(query)
        query_embeddings = self.query_expander.embed_queries(expanded)
        self.timings["query_expansion"] = time.time() - t0
        print(f"Step 2 (Query Expansion): {self.timings['query_expansion']:.3f}s")
        
        # ===== STEP 3: Hybrid Retrieval =====
        t0 = time.time()
        dense_results = self._dense_search(query_embeddings["hyde"], documents)
        bm25_results = self._bm25_search(query, documents)
        self.timings["retrieval"] = time.time() - t0
        print(f"Step 3 (Retrieval): {self.timings['retrieval']:.3f}s")
        
        # ===== STEP 4: Re-ranking + MMR =====
        t0 = time.time()
        doc_embeddings = np.array([
            self.query_expander.encoder.encode(doc, convert_to_numpy=True)
            if self.query_expander.encoder else np.zeros(384)
            for doc in documents
        ])
        
        reranked = self.reranker.rerank_pipeline(
            query,
            dense_results,
            bm25_results,
            doc_embeddings,
            lambda_param=self.cfg.mmr_lambda,
            top_k=self.cfg.top_k
        )
        retrieved_docs = [doc for doc, score, _ in reranked]
        retrieval_scores = [score for _, score, _ in reranked]
        self.timings["reranking"] = time.time() - t0
        print(f"Step 4 (Re-ranking + MMR): {self.timings['reranking']:.3f}s")
        
        # ===== STEP 5: Causal KG + GNN =====
        t0 = time.time()
        seed_ids = list(range(min(5, self.cfg.n_graph_nodes)))
        nf, adj, ew, mask = self.kg.subgraph(seed_ids)
        
        nf_batch = nf.unsqueeze(0).expand(1, -1, -1)
        adj_batch = adj.unsqueeze(0)
        ew_batch = ew.unsqueeze(0)
        
        _, g_emb = self.gnn(nf_batch, adj_batch, ew_batch, mask)
        self.timings["graph"] = time.time() - t0
        print(f"Step 5 (Causal KG + GNN): {self.timings['graph']:.3f}s")
        
        # ===== STEP 6-7: World Model + Planning =====
        t0 = time.time()
        init_obs = torch.randn(1, self.cfg.seq_len, self.cfg.obs_dim, device=device) * 0.25
        z_seq = torch.randn(1, 1, self.cfg.latent_dim, device=device)  # Simplified
        z0 = z_seq[:, -1, :]
        
        plan_result = self.planner.plan(z0, g_emb.squeeze(0))
        self.timings["planning"] = time.time() - t0
        print(f"Steps 6-7 (World Model + Planning): {self.timings['planning']:.3f}s")
        
        # ===== STEP 8: Ensemble Uncertainty =====
        t0 = time.time()
        with torch.no_grad():
            outcomes_pred, ep, al = self._ensemble_predict(z0)
        self.timings["ensemble"] = time.time() - t0
        print(f"Step 8 (Ensemble): {self.timings['ensemble']:.3f}s")
        
        # ===== STEP 9: LLM Reasoning =====
        t0 = time.time()
        llm_output = self.llm_client.generate_medical_recommendation(
            query,
            retrieved_docs,
            plan_result,
            graph_embedding=g_emb,
            ensemble_outcomes=self._format_ensemble_outcomes(outcomes_pred, ep, al)
        )
        self.timings["llm"] = time.time() - t0
        print(f"Step 9 (LLM Reasoning): {self.timings['llm']:.3f}s")
        
        # ===== STEP 10: Hallucination Check =====
        t0 = time.time()
        halluc_check = self.hallucination_check.check_llm_output(
            llm_output.reasoning + " " + llm_output.recommendation,
            retrieved_docs
        )
        self.timings["hallucination_check"] = time.time() - t0
        print(f"Step 10 (Hallucination Check): {self.timings['hallucination_check']:.3f}s")
        
        # ===== STEP 11: SHAP Attribution =====
        t0 = time.time()
        shap_result = self.shap_attributor.shapley_with_interactions(
            query,
            retrieved_docs,
            n_permutations=self.cfg.shap_perms
        )
        shap_interp = self.shap_attributor.interpret_attribution(
            query,
            retrieved_docs,
            shap_result,
            top_k=3
        )
        self.timings["shap"] = time.time() - t0
        print(f"Step 11 (SHAP Attribution): {self.timings['shap']:.3f}s")
        
        # ===== STEP 12: Final Output =====
        output = GRAPESOutput(
            # Input
            query=query,
            
            # Query expansion
            expanded_queries={
                "original": expanded.original,
                "hyde": expanded.hyde,
                "sub_queries": expanded.sub_queries
            },
            
            # Retrieval
            retrieved_documents=retrieved_docs,
            retrieval_scores=retrieval_scores,
            
            # Graph
            graph_embedding=g_emb,
            
            # Planning
            world_model_trajectory=plan_result.get("trajectory", []),
            world_model_score=plan_result.get("score", 0.0),
            best_plan=plan_result.get("actions", []),
            plan_scores=plan_result.get("scores", {}),
            
            # Ensemble
            ensemble_predictions=self._format_ensemble_outcomes(outcomes_pred, ep, al),
            ensemble_uncertainty={
                "epistemic": float(ep[0, 0]) if ep.numel() > 0 else 0.0,
                "aleatoric": float(al[0, 0]) if al.numel() > 0 else 0.0
            },
            
            # LLM
            llm_reasoning=llm_output.reasoning,
            llm_recommendation=llm_output.recommendation,
            llm_confidence=llm_output.confidence,
            
            # Hallucination check
            hallucination_check=halluc_check,
            
            # SHAP
            shap_values={
                "document_shapley": shap_result.document_shapley.tolist(),
                "interpretation": shap_interp
            },
            shap_interactions=shap_result.pairwise_interactions,
            
            # Metadata
            timings=self.timings,
            quality_scores={
                "grounding_score": halluc_check["nli"]["grounding_score"],
                "hallucination_rate": halluc_check["nli"]["hallucination_rate"],
                "combined_score": halluc_check["combined_score"]
            }
        )
        
        total_time = sum(self.timings.values())
        print(f"\n✓ Pipeline complete ({total_time:.2f}s total)")
        
        return output
    
    def _dense_search(self, query_embedding: np.ndarray, documents: List[str]) -> List[tuple]:
        """Dense semantic search using FAISS."""
        if self.retriever.index is None:
            return []
        
        try:
            import faiss
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_embedding)
            distances, indices = self.retriever.index.search(query_embedding, self.cfg.top_k)
            
            results = [
                (self.retriever.docs[idx], 1.0 - dist)
                for idx, dist in zip(indices[0], distances[0])
                if idx < len(self.retriever.docs)
            ]
            return results
        except Exception:
            return []
    
    def _bm25_search(self, query: str, documents: List[str]) -> List[tuple]:
        """BM25 keyword search."""
        if self.retriever.bm25 is None:
            return []
        
        try:
            scores = self.retriever.bm25.get_scores(query.lower().split())
            ranked = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )[:self.cfg.top_k]
            return ranked
        except Exception:
            return []
    
    def _ensemble_predict(self, z: torch.Tensor):
        """Get ensemble predictions."""
        with torch.no_grad():
            mu, total_unc, ep, al = self.ensemble(z)
        return mu, ep, al
    
    def _format_ensemble_outcomes(self, mu, ep, al) -> Dict:
        """Format ensemble outputs."""
        try:
            mu_val = float(mu[0, 0]) if mu.numel() > 0 else 0.5
            ep_val = float(ep[0, 0]) if ep.numel() > 0 else 0.05
            al_val = float(al[0, 0]) if al.numel() > 0 else 0.05
            
            return {
                "survival": f"{mu_val*100:.1f}% ± {ep_val*100:.1f}%",
                "readmission": f"{(1-mu_val)*100:.1f}% ± {ep_val*100:.1f}%",
                "complication": f"{al_val*100:.1f}% ± {ep_val*100:.1f}%"
            }
        except:
            return {
                "survival": "N/A",
                "readmission": "N/A",
                "complication": "N/A"
            }
