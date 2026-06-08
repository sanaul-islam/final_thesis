"""
Query Expansion Module - Implements HyDE + Sub-Query Decomposition

This module creates 3 versions of a user query to maximize retrieval recall:
1. Original query (clinical context)
2. HyDE - Hypothetical document embedding (LLM-generated ideal answer)
3. Sub-query decomposition (atomic clinical concepts)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from grapes_shap.config import Config


@dataclass
class ExpandedQuery:
    original: str
    hyde: str
    sub_queries: List[str]


class QueryExpander:
    """Expands clinical queries for better retrieval coverage."""
    
    def __init__(self, cfg: Config, llm_client=None):
        self.cfg = cfg
        self.llm_client = llm_client  # DeepSeek API client
        self._load_encoder()
    
    def _load_encoder(self):
        """Load sentence transformer for embedding generation."""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=self.cfg.device
            )
        except Exception as e:
            print(f"Warning: SentenceTransformer not loaded: {e}")
            self.encoder = None
    
    def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query into 3 representations.
        
        Args:
            query: Original clinical question
            
        Returns:
            ExpandedQuery with original, hyde, and sub_queries
        """
        # 1. Keep original
        original = query
        
        # 2. Generate HyDE (hypothetical document embedding)
        hyde = self._generate_hyde(query)
        
        # 3. Decompose into atomic sub-queries
        sub_queries = self._decompose_subqueries(query)
        
        return ExpandedQuery(
            original=original,
            hyde=hyde,
            sub_queries=sub_queries
        )
    
    def _generate_hyde(self, query: str) -> str:
        """
        Generate hypothetical document using LLM.
        
        This simulates what an ideal answer document would look like,
        allowing semantic search to find answers using answer vocabulary.
        
        Args:
            query: Clinical query
            
        Returns:
            Hypothetical answer text
        """
        if self.llm_client is None:
            # Fallback: simple expansion
            return self._fallback_hyde(query)
        
        try:
            prompt = f"""Given this clinical query, write a short hypothetical answer document 
that would perfectly address it. Focus on clinical facts, guidelines, and evidence-based recommendations.
Query: {query}

Hypothetical answer document:"""
            
            # Use DeepSeek API
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                top_p=0.9
            )
            
            hyde = response.choices[0].message.content.strip()
            return hyde
        
        except Exception as e:
            print(f"HyDE generation failed: {e}. Using fallback.")
            return self._fallback_hyde(query)
    
    def _fallback_hyde(self, query: str) -> str:
        """Fallback HyDE when LLM unavailable."""
        # Extract key medical terms and rephrase as an answer
        keywords = query.lower().split()
        clinical_keywords = [w for w in keywords if len(w) > 4]
        
        if len(clinical_keywords) >= 2:
            return f"Evidence shows that {' and '.join(clinical_keywords[:3])} are " \
                   f"associated with improved clinical outcomes in this patient population. " \
                   f"Treatment guidelines recommend careful consideration of {clinical_keywords[0]} " \
                   f"status when making clinical decisions."
        else:
            return f"Clinical evidence and expert guidelines provide the following " \
                   f"recommendations for patients with {query}. Key considerations include " \
                   f"baseline characteristics, prior treatments, and comorbidities."
    
    def _decompose_subqueries(self, query: str) -> List[str]:
        """
        Decompose complex query into atomic sub-queries.
        
        Examples:
        "70-year-old male with EGFR+ NSCLC and brain metastases" →
        ["EGFR positive NSCLC treatment", "brain metastases management", 
         "elderly patient lung cancer therapy"]
        
        Args:
            query: Original query
            
        Returns:
            List of focused sub-queries
        """
        sub_queries = []
        
        # Clinical entity extraction patterns
        patterns = {
            "mutations": ["EGFR", "ALK", "ROS1", "KRAS", "TP53", "BRAF"],
            "cancer_types": ["NSCLC", "lung cancer", "SCLC", "mesothelioma"],
            "conditions": ["brain metastases", "CNS", "resistance", "recurrent"],
            "treatments": ["osimertinib", "chemotherapy", "immunotherapy", "radiation"]
        }
        
        query_lower = query.lower()
        
        # Find patterns in query
        findings = []
        for category, terms in patterns.items():
            for term in terms:
                if term.lower() in query_lower:
                    findings.append((category, term))
        
        # Generate sub-queries from findings
        if findings:
            # Sub-query 1: Treatment + mutation
            treatment_terms = [t for c, t in findings if c == "treatments"]
            mutation_terms = [t for c, t in findings if c == "mutations"]
            if mutation_terms and treatment_terms:
                sub_queries.append(
                    f"{mutation_terms[0]} mutation treatment with {treatment_terms[0]}"
                )
            
            # Sub-query 2: Cancer type + special conditions
            cancer_terms = [t for c, t in findings if c == "cancer_types"]
            condition_terms = [t for c, t in findings if c == "conditions"]
            if cancer_terms and condition_terms:
                sub_queries.append(
                    f"{condition_terms[0]} management in {cancer_terms[0]}"
                )
            
            # Sub-query 3: Age/performance status + clinical scenario
            if "age" in query_lower or "elderly" in query_lower or "70" in query:
                cancer = cancer_terms[0] if cancer_terms else "lung cancer"
                sub_queries.append(f"elderly patient {cancer} treatment safety")
        
        # Fallback: generic sub-queries
        if not sub_queries:
            sub_queries = [
                query,  # Full query
                query.split(",")[0] if "," in query else query[:50],  # First clause
                " ".join(query.split()[-5:])  # Last 5 words
            ]
        
        # Ensure exactly 3 diverse sub-queries
        while len(sub_queries) < 3:
            sub_queries.append(query)
        
        return sub_queries[:3]
    
    def embed_queries(self, expanded: ExpandedQuery) -> Dict[str, np.ndarray]:
        """
        Embed all 3 query versions.
        
        Args:
            expanded: ExpandedQuery object
            
        Returns:
            Dict with embeddings for original, hyde, and each sub-query
        """
        if self.encoder is None:
            # Return dummy embeddings
            dummy = np.zeros(384)
            return {
                "original": dummy,
                "hyde": dummy,
                "sub_queries": [dummy, dummy, dummy]
            }
        
        embeddings = {}
        
        with torch.no_grad():
            # Embed original
            embeddings["original"] = self.encoder.encode(
                expanded.original,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Embed HyDE
            embeddings["hyde"] = self.encoder.encode(
                expanded.hyde,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Embed sub-queries
            embeddings["sub_queries"] = self.encoder.encode(
                expanded.sub_queries,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        return embeddings
