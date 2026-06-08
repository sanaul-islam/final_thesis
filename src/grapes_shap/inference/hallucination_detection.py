"""
Hallucination Detection - Self-RAG + NLI verification

Implements two-layer fact-checking:
1. Self-RAG: LLM generates critique tokens during generation
2. NLI (Natural Language Inference): Post-generation verification against evidence
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from grapes_shap.config import Config


class FactCheckResult(Enum):
    """Result of fact-checking a claim."""
    ENTAILMENT = "entailed_by_evidence"  # Fully supported
    PARTIAL = "partially_supported"       # Partially supported
    NEUTRAL = "neutral"                   # No contradiction or support
    CONTRADICTION = "contradicted"        # Contradicted by evidence
    UNKNOWN = "unknown"                   # Cannot determine


@dataclass
class HallucellationCheckReport:
    total_claims: int
    supported_claims: int
    partial_claims: int
    flagged_claims: List[str]
    hallucination_rate: float
    grounding_score: float  # 0-1: fraction of supported claims


class SelfRAGCritique:
    """Self-RAG: LLM internal fact-checking during generation."""
    
    @staticmethod
    def critique_tokens() -> Dict[str, str]:
        """Get critique token templates."""
        return {
            "IsREL": {
                "YES": "[IsREL=YES]",
                "NO": "[IsREL=NO]"
            },
            "IsSUP": {
                "fully": "[IsSUP=fully]",
                "partial": "[IsSUP=partial]",
                "none": "[IsSUP=none]"
            }
        }
    
    @staticmethod
    def extract_critique_from_output(text: str) -> List[Tuple[str, str]]:
        """Extract critique tokens from LLM output."""
        import re
        
        critiques = []
        
        # Extract IsREL tokens
        isrel_matches = re.findall(r'\[IsREL=(YES|NO)\]', text)
        for match in isrel_matches:
            critiques.append(("IsREL", match))
        
        # Extract IsSUP tokens
        issup_matches = re.findall(r'\[IsSUP=(fully|partial|none)\]', text)
        for match in issup_matches:
            critiques.append(("IsSUP", match))
        
        return critiques
    
    @staticmethod
    def should_requery(text: str) -> bool:
        """Check if low-confidence output should trigger re-retrieval."""
        # Trigger re-retrieval if:
        # 1. IsSUP=partial found
        # 2. No IsREL or IsSUP tokens found (uncertainty)
        
        has_partial = "[IsSUP=partial]" in text
        has_issup = "[IsSUP=" in text
        
        return has_partial or not has_issup


class NLIVerifier:
    """Natural Language Inference verification against evidence."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load_nli_model()
    
    def _load_nli_model(self):
        """Load NLI model for entailment checking."""
        try:
            from transformers import pipeline
            self.nli = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.cfg.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: NLI model not loaded: {e}")
            self.nli = None
    
    def extract_claims(self, text: str, max_claims: int = 20) -> List[str]:
        """
        Extract factual claims from text.
        
        Split by sentences and filter for factual content (not subjective).
        
        Args:
            text: Text to extract claims from
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of claims
        """
        import re
        
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip very short sentences (likely not claims)
            if len(sentence.split()) < 4:
                continue
            
            # Skip opinion phrases
            opinion_phrases = [
                "i think", "i believe", "in my opinion",
                "seems like", "appears to", "might be",
                "could be", "possibly", "perhaps"
            ]
            
            if any(phrase in sentence.lower() for phrase in opinion_phrases):
                continue
            
            claims.append(sentence)
        
        return claims[:max_claims]
    
    def verify_claim(
        self,
        claim: str,
        evidence_docs: List[str]
    ) -> Tuple[FactCheckResult, float]:
        """
        Verify a single claim against evidence documents.
        
        Uses NLI to check entailment relationship:
        - Evidence → Claim: ENTAILMENT (supported)
        - Evidence ⊥ Claim: CONTRADICTION (contradicted)
        - Otherwise: NEUTRAL or UNKNOWN
        
        Args:
            claim: Factual claim to verify
            evidence_docs: List of evidence documents
            
        Returns:
            (FactCheckResult, confidence_score)
        """
        if self.nli is None:
            return FactCheckResult.UNKNOWN, 0.5
        
        if not evidence_docs:
            return FactCheckResult.UNKNOWN, 0.5
        
        try:
            best_result = FactCheckResult.NEUTRAL
            best_score = 0.0
            
            # Check claim against each evidence document
            for doc in evidence_docs:
                # Truncate very long documents
                doc_text = doc[:500] if len(doc) > 500 else doc
                
                # NLI: does doc entail claim?
                try:
                    result = self.nli(
                        doc_text,
                        [claim],
                        multi_class=False
                    )
                    
                    label = result["labels"][0]  # "entailment", "neutral", "contradiction"
                    score = result["scores"][0]
                    
                    # Map to our result enum
                    if label == "entailment" and score > 0.7:
                        return FactCheckResult.ENTAILMENT, score
                    elif label == "contradiction" and score > 0.7:
                        best_result = FactCheckResult.CONTRADICTION
                        best_score = max(best_score, score)
                    elif label == "neutral" and score > best_score:
                        if best_result == FactCheckResult.NEUTRAL:
                            best_score = score
                
                except Exception as e:
                    print(f"NLI check failed for claim: {e}")
                    continue
            
            return best_result, best_score
        
        except Exception as e:
            print(f"Verification failed: {e}")
            return FactCheckResult.UNKNOWN, 0.5
    
    def verify_text(
        self,
        text: str,
        evidence_docs: List[str]
    ) -> HallucellationCheckReport:
        """
        Verify all claims in a text against evidence.
        
        Args:
            text: Generated text to verify
            evidence_docs: Evidence documents for fact-checking
            
        Returns:
            HallucellationCheckReport with statistics
        """
        # Extract claims
        claims = self.extract_claims(text)
        
        if not claims:
            return HallucellationCheckReport(
                total_claims=0,
                supported_claims=0,
                partial_claims=0,
                flagged_claims=[],
                hallucination_rate=0.0,
                grounding_score=1.0
            )
        
        # Verify each claim
        supported = 0
        partial = 0
        flagged = []
        
        for claim in claims:
            result, confidence = self.verify_claim(claim, evidence_docs)
            
            if result == FactCheckResult.ENTAILMENT:
                supported += 1
            elif result == FactCheckResult.PARTIAL:
                partial += 1
            elif result == FactCheckResult.CONTRADICTION:
                flagged.append(f"CONTRADICTION: {claim}")
            elif result == FactCheckResult.NEUTRAL:
                # Treat neutral as unsupported
                flagged.append(f"UNSUPPORTED: {claim}")
        
        total = len(claims)
        hallucination_rate = len(flagged) / max(1, total)
        grounding_score = (supported + partial / 2) / max(1, total)
        
        return HallucellationCheckReport(
            total_claims=total,
            supported_claims=supported,
            partial_claims=partial,
            flagged_claims=flagged,
            hallucination_rate=hallucination_rate,
            grounding_score=grounding_score
        )


class DualLayerHallucellationCheck:
    """Combined Self-RAG + NLI hallucination detection."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.self_rag = SelfRAGCritique()
        self.nli_verifier = NLIVerifier(cfg)
    
    def check_llm_output(
        self,
        llm_output: str,
        evidence_docs: List[str]
    ) -> Dict:
        """
        Run dual-layer hallucination check on LLM output.
        
        Args:
            llm_output: Generated text from LLM
            evidence_docs: Evidence documents
            
        Returns:
            Dict with check results and recommendations
        """
        # Layer 1: Self-RAG critique tokens
        critique_tokens = self.self_rag.extract_critique_from_output(llm_output)
        should_requery = self.self_rag.should_requery(llm_output)
        
        # Layer 2: NLI verification
        nli_report = self.nli_verifier.verify_text(llm_output, evidence_docs)
        
        # Combine results
        return {
            "self_rag": {
                "critique_tokens": critique_tokens,
                "should_requery": should_requery
            },
            "nli": {
                "total_claims": nli_report.total_claims,
                "supported_claims": nli_report.supported_claims,
                "partial_claims": nli_report.partial_claims,
                "flagged_claims": nli_report.flagged_claims,
                "hallucination_rate": nli_report.hallucination_rate,
                "grounding_score": nli_report.grounding_score
            },
            "combined_score": (
                (1 - nli_report.hallucination_rate) +
                nli_report.grounding_score
            ) / 2,
            "recommended_action": self._recommend_action(
                should_requery,
                nli_report.hallucination_rate
            )
        }
    
    def _recommend_action(
        self,
        self_rag_requery: bool,
        hallucination_rate: float
    ) -> str:
        """Recommend action based on checks."""
        if self_rag_requery and hallucination_rate > 0.2:
            return "HIGH: Re-query evidence and regenerate response"
        elif hallucination_rate > 0.15:
            return "MEDIUM: Flag specific claims for clinician review"
        elif hallucination_rate > 0.05:
            return "LOW: Minor unsupported claims, acceptable with caveats"
        else:
            return "PASS: Output well-grounded in evidence"
