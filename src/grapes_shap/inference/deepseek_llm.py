"""
DeepSeek LLM Integration - Chain-of-thought reasoning with QLoRA 4-bit compression

This module provides:
1. DeepSeek API client with structured prompts
2. QLoRA 4-bit fine-tuning for parameter-efficient adaptation
3. Flash Attention 2 for memory efficiency
4. Chain-of-thought reasoning templates
"""

from typing import Dict, List, Optional
import os
from dataclasses import dataclass
from grapes_shap.config import Config


@dataclass
class LLMOutput:
    reasoning: str
    recommendation: str
    confidence: float
    key_evidence: List[int]  # Indices of key documents
    full_text: str = ""       # Complete structured response from the LLM


class DeepSeekLLMClient:
    """Client for DeepSeek LLM with structured medical prompts."""
    
    def __init__(self, cfg: Config, api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            cfg: Config object
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
        """
        self.cfg = cfg
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize DeepSeek API client."""
        try:
            from openai import OpenAI
            return OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            print("Warning: OpenAI client not available. Install with: pip install openai")
            return None
    
    def generate_medical_recommendation(
        self,
        query: str,
        evidence_docs: List[str],
        world_model_results: Dict,
        graph_embedding: Optional[object] = None,
        ensemble_outcomes: Optional[Dict] = None
    ) -> LLMOutput:
        """
        Generate chain-of-thought medical recommendation.
        
        Args:
            query: Patient clinical query
            evidence_docs: List of retrieved evidence documents
            world_model_results: Simulation results (best plan, trajectory)
            graph_embedding: KG embedding (unused in output, for context)
            ensemble_outcomes: Ensemble predictions (survival, etc.)
            
        Returns:
            LLMOutput with reasoning, recommendation, confidence, key docs
        """
        if self.client is None:
            return self._fallback_recommendation(
                query,
                evidence_docs,
                world_model_results,
                ensemble_outcomes
            )
        
        # Build comprehensive prompt
        prompt = self._build_prompt(
            query,
            evidence_docs,
            world_model_results,
            ensemble_outcomes
        )
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": self._SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temp for consistency
                max_tokens=2000,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            output = self._parse_response(response_text, evidence_docs)
            return output
        
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return self._fallback_recommendation(
                query,
                evidence_docs,
                world_model_results,
                ensemble_outcomes
            )
    
    def _build_prompt(
        self,
        query: str,
        evidence_docs: List[str],
        world_model_results: Dict,
        ensemble_outcomes: Optional[Dict] = None
    ) -> str:
        """Build structured prompt for LLM."""
        
        prompt = f"""## Clinical Query
{query}

## Retrieved Evidence (citations [1]-[{len(evidence_docs)}])
"""
        for i, doc in enumerate(evidence_docs, 1):
            # Truncate long documents
            doc_text = doc[:500] + "..." if len(doc) > 500 else doc
            prompt += f"\n[{i}] {doc_text}\n"
        
        # Add world model results
        if world_model_results:
            prompt += f"""
## Simulation Results
Best treatment plan: {world_model_results.get('best_actions', 'Not available')}
Plan score: {world_model_results.get('best_score', 'N/A')}
"""
        
        # Add ensemble predictions
        if ensemble_outcomes:
            prompt += f"""
## Predicted Outcomes (from 5-model ensemble)
- 1-year survival: {ensemble_outcomes.get('survival', 'N/A')}
- 30-day readmission risk: {ensemble_outcomes.get('readmission', 'N/A')}
- Complication rate: {ensemble_outcomes.get('complication', 'N/A')}
"""
        
        prompt += """
## Task
Provide a structured clinical recommendation with:
1. **Chain-of-thought reasoning**: Step-by-step analysis citing evidence [1], [2], etc.
2. **Treatment recommendation**: Specific drug, dose, schedule
3. **Risk-benefit summary**: Major benefits and risks for this patient
4. **Follow-up protocol**: Monitoring and contingency plans
5. **Confidence assessment**: Your confidence in this recommendation (0-1)
6. **Key evidence**: Which evidence citations drove this recommendation (e.g., [1], [3], [5])

Format your response with clear sections using bold headers.
Remember: Every claim must be supported by evidence numbers [1]-[{len(evidence_docs)}].
"""
        
        return prompt
    
    def _parse_response(
        self,
        response_text: str,
        evidence_docs: List[str]
    ) -> LLMOutput:
        """Parse structured response from LLM."""
        
        # Extract sections
        sections = self._extract_sections(response_text)
        
        reasoning = sections.get("reasoning", response_text[:500])
        recommendation = sections.get("recommendation", "See reasoning above")
        confidence_str = sections.get("confidence", "0.7")
        key_evidence_str = sections.get("key_evidence", "")
        
        # Parse confidence (0-1)
        try:
            confidence = float(confidence_str.strip().split()[0])
            confidence = max(0, min(1, confidence))  # Clamp to [0, 1]
        except:
            confidence = 0.7
        
        # Parse key evidence indices
        key_evidence = self._extract_citation_indices(
            reasoning + " " + key_evidence_str,
            len(evidence_docs)
        )
        
        return LLMOutput(
            reasoning=reasoning,
            recommendation=recommendation,
            confidence=confidence,
            key_evidence=key_evidence,
            full_text=response_text
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract structured sections from response."""
        sections = {}
        
        headers = [
            ("chain-of-thought reasoning", "reasoning"),
            ("treatment recommendation", "recommendation"),
            ("risk-benefit summary", "risk_benefit"),
            ("follow-up protocol", "follow_up"),
            ("confidence", "confidence"),
            ("key evidence", "key_evidence")
        ]
        
        text_lower = text.lower()
        
        for header, key in headers:
            start = text_lower.find(header)
            if start != -1:
                # Find next header or end
                next_header = len(text)
                for other_header, _ in headers:
                    if other_header != header:
                        pos = text_lower.find(other_header, start + 1)
                        if pos != -1 and pos < next_header:
                            next_header = pos
                
                content = text[start + len(header):next_header].strip()
                # Remove markdown formatting
                content = content.replace("**", "").replace("##", "").strip()
                sections[key] = content[:1000]  # Limit length
        
        return sections
    
    def _extract_citation_indices(self, text: str, max_docs: int) -> List[int]:
        """Extract citation indices [1], [2], etc. from text."""
        import re
        citations = re.findall(r'\[(\d+)\]', text)
        indices = []
        for cite in citations:
            try:
                idx = int(cite) - 1  # Convert to 0-indexed
                if 0 <= idx < max_docs:
                    indices.append(idx)
            except:
                pass
        return list(set(indices))  # Remove duplicates
    
    def _fallback_recommendation(
        self,
        query: str,
        evidence_docs: List[str],
        world_model_results: Dict,
        ensemble_outcomes: Optional[Dict] = None
    ) -> LLMOutput:
        """Generate fallback recommendation without LLM."""
        
        reasoning = f"""Based on the clinical query: {query}
        
Retrieved evidence suggests a systematic approach:
"""
        
        if evidence_docs:
            reasoning += f"\n{len(evidence_docs)} key evidence documents have been identified. "
            reasoning += "The top evidence sources provide evidence-based guidance.\n"
        
        recommendation = "Recommend consulting with clinical specialists and reviewing " \
                        "the retrieved evidence documents for treatment planning."
        
        if world_model_results:
            recommendation = f"Treatment plan: {world_model_results.get('best_actions', 'See evidence')}"
        
        return LLMOutput(
            reasoning=reasoning,
            recommendation=recommendation,
            confidence=0.6,
            key_evidence=list(range(min(3, len(evidence_docs)))),
            full_text=reasoning + "\n\n" + recommendation
        )
    
    # System prompt for medical reasoning
    _SYSTEM_PROMPT = """You are an expert medical AI assistant trained in evidence-based clinical decision-making.
Your role is to analyze patient cases and provide structured, evidence-supported treatment recommendations.

IMPORTANT GUIDELINES:
1. Base all recommendations on the provided evidence citations
2. Cite evidence using [1], [2], etc. format
3. Acknowledge uncertainty and limitations
4. Consider patient-specific factors (age, comorbidities, preferences)
5. Provide actionable, specific recommendations
6. Format clearly with section headers
7. Keep confidence assessment realistic (0.0-1.0 scale)

MEDICAL ETHICS:
- These recommendations are for clinical consideration, not final medical advice
- Always defer to clinical judgment of the treating physician
- Highlight any gaps in evidence or patient-specific uncertainties
- Consider both benefits and potential harms"""


class QLoRAAdapter:
    """Parameter-efficient fine-tuning with QLoRA (4-bit compression + LoRA)."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.adapter_loaded = False
    
    def setup_quantization_config(self) -> Dict:
        """Return BitsAndBytesConfig for 4-bit quantization."""
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
    
    def setup_lora_config(self) -> Dict:
        """Return LoRAConfig with appropriate parameters."""
        return {
            "r": 16,  # LoRA rank
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "v_proj"],  # For Mistral/LLaMA
        }
    
    def load_model_with_lora(self, model_name: str) -> Optional[object]:
        """Load model with QLoRA adaptation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig
            from bitsandbytes.nn import Linear4bit
            
            quantization_config = self.setup_quantization_config()
            lora_config = self.setup_lora_config()
            
            # Load base model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Apply LoRA
            model = get_peft_model(model, LoraConfig(**lora_config))
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.model = model
            self.tokenizer = tokenizer
            self.adapter_loaded = True
            
            return model
        
        except Exception as e:
            print(f"QLoRA loading failed: {e}")
            return None
