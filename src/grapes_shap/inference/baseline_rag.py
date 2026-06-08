"""
Baseline Advanced RAG (comparison reference for GRAPES-SHAP).

This is a strong, modern Retrieval-Augmented Generation baseline:
  • Hybrid dense (MiniLM + FAISS) + sparse (BM25) retrieval with RRF fusion
  • Direct large-language-model answer generation (DeepSeek)

It deliberately OMITS the novel GRAPES-SHAP components so the two systems can
be compared on identical questions:
  - no query expansion / HyDE
  - no MMR diversity re-ranking
  - no causal knowledge graph / GNN
  - no latent world-model simulation
  - no Tree-of-Thought planning
  - no deep-ensemble uncertainty
  - no hallucination self-check
  - no SHAP attribution

The contrast isolates the contribution of the GRAPES-SHAP reasoning stack.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from grapes_shap.config import Config
from grapes_shap.inference.retriever import HybridRetriever
from grapes_shap.inference.deepseek_llm import DeepSeekLLMClient


@dataclass
class BaselineRAGOutput:
    query: str
    answer: str
    retrieved_documents: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_s: float = 0.0


class BaselineRAG:
    """Vanilla hybrid-retrieval + LLM RAG pipeline."""

    _SYSTEM_PROMPT = (
        "You are a clinical decision-support assistant. Using ONLY the retrieved "
        "evidence provided, give a concise, structured answer to the clinical "
        "question. Include: (1) most likely diagnosis, (2) immediate management, "
        "and (3) a one-line rationale citing evidence numbers like [1], [2]. "
        "Do not fabricate facts beyond the evidence."
    )

    def __init__(self, cfg: Config,
                 retriever: Optional[HybridRetriever] = None,
                 llm_client: Optional[DeepSeekLLMClient] = None):
        self.cfg = cfg
        self.retriever = retriever or HybridRetriever(cfg)
        self.llm = llm_client or DeepSeekLLMClient(cfg)

    def build(self, documents: List[str]):
        """Build the retrieval index over the document corpus."""
        if not self.retriever.index:
            self.retriever.build(documents)

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        prompt = f"## Clinical Question\n{query}\n\n## Retrieved Evidence\n"
        for i, doc in enumerate(docs, 1):
            doc_text = doc[:500] + "..." if len(doc) > 500 else doc
            prompt += f"[{i}] {doc_text}\n"
        prompt += (
            "\n## Task\nProvide your structured answer now. End with a line "
            "'Confidence: X' where X is between 0 and 1."
        )
        return prompt

    def answer(self, query: str, k: Optional[int] = None) -> BaselineRAGOutput:
        t0 = time.time()
        docs = self.retriever.retrieve(query, k=k or self.cfg.top_k)

        answer_text, confidence = self._generate(query, docs)
        return BaselineRAGOutput(
            query=query,
            answer=answer_text,
            retrieved_documents=docs,
            confidence=confidence,
            latency_s=time.time() - t0,
        )

    def _generate(self, query: str, docs: List[str]):
        if self.llm.client is None:
            return ("[LLM unavailable] Based on retrieved evidence, consult the "
                    "documents above for management guidance.", 0.5)
        prompt = self._build_prompt(query, docs)
        try:
            resp = self.llm.client.chat.completions.create(
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1200,
                top_p=0.9,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            return (f"[LLM error: {e}]", 0.5)

        confidence = 0.6
        low = text.lower()
        if "confidence:" in low:
            try:
                tail = low.split("confidence:")[-1].strip()
                confidence = float(tail.split()[0].rstrip("."))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                pass
        return text, confidence
