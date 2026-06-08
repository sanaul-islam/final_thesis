import random
from typing import List
from collections import defaultdict
from grapes_shap.config import Config

class HybridRetriever:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.docs: List[str] = []
        self.index = None
        self.bm25  = None
        self.encoder = None
        self._try_load_encoder()

    def _try_load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception as e:
            print(f"  SentenceTransformer not available: {e}")

    def build(self, docs: List[str]):
        self.docs = docs
        if self.encoder is None:
            return
        try:
            import faiss
            embs = self.encoder.encode(docs, convert_to_numpy=True,
                                        show_progress_bar=True, batch_size=256)
            faiss.normalize_L2(embs)
            idx = faiss.IndexHNSWFlat(embs.shape[1], 32)
            idx.hnsw.efConstruction = 200
            idx.add(embs)
            self.index = idx
        except Exception as e:
            print(f"  FAISS build failed: {e}")
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([d.lower().split() for d in docs])
        except Exception:
            pass
        print(f"  RAG index built: {len(docs):,} documents")

    def retrieve(self, query: str, k: int = None) -> List[str]:
        k = k or self.cfg.top_k
        if len(self.docs) == 0:
            return []
        if self.index is None:
            return random.sample(self.docs, min(k, len(self.docs)))
        try:
            import faiss
            q = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q)
            dense_scores, dense_idxs = self.index.search(q, min(20, len(self.docs)))
            dense_ranked = list(zip(dense_idxs[0].tolist(), dense_scores[0].tolist()))
        except Exception:
            dense_ranked = []
        bm25_ranked = []
        if self.bm25 is not None:
            try:
                scores = self.bm25.get_scores(query.lower().split())
                bm25_ranked = sorted(enumerate(scores.tolist()), key=lambda x:-x[1])[:20]
            except Exception:
                pass
        rrf = defaultdict(float)
        for rank, (idx, _) in enumerate(dense_ranked):
            rrf[idx] += 1.0 / (60 + rank + 1)
        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf[idx] += 1.0 / (60 + rank + 1)
        if rrf:
            top_idxs = sorted(rrf, key=lambda x: -rrf[x])[:k]
        else:
            top_idxs = list(range(min(k, len(self.docs))))
        return [self.docs[i] for i in top_idxs if i < len(self.docs)]
