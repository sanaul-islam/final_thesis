# GRAPES-SHAP — Models, Parameters & Key Results (One-Page Summary)

## 1. Models Used

| Component | Model / Architecture | Role |
|---|---|---|
| **Generator LLM** | DeepSeek `deepseek-chat` (QLoRA r=16, α=32) | Final answer generation |
| **Dense retriever** | MiniLM sentence-embeddings + FAISS | Semantic evidence retrieval |
| **Sparse retriever** | BM25 + Reciprocal-Rank Fusion | Keyword retrieval |
| **Re-ranker** | Cross-Encoder `ms-marco-MiniLM-L-6-v2` + MMR | Relevance/diversity re-ranking |
| **Query expansion** | HyDE (3 sub-queries) | Recall boost |
| **Latent World Model** | GRU(3-layer) + Causal-Residual + σ/reward heads | Treatment-plan simulation |
| **Knowledge graph** | Causal KG + edge-biased GNN | Structured reasoning |
| **Uncertainty** | Deep ensemble (5 members) | Calibrated confidence |
| **Planner** | Tree-of-Thought (horizon 4, 8 candidates) | Action planning |
| **Attribution** | SHAP (32 permutations, pairwise interactions) | Evidence interpretability |
| **Hallucination check** | Self-RAG + NLI verification (thr. 0.15) | Faithfulness gate |

## 2. Key Parameters

**Architecture:** obs_dim 64 · action_dim 50 · latent_dim 256 · hidden_dim 512 · graph_node_dim 128 · n_graph_nodes 20 · n_outcomes 5 · seq_len 8 · ensemble 5 · heads 8 · transformer_layers 3 · dropout 0.10
**Retrieval:** top_k 6 · embed_dim 384 · shap_perms 32 · mmr_λ 0.6 · RRF k 60 · HyDE sub-queries 3
**Planning:** plan_horizon 4 · plan_candidates 8 · causal_residual_scale 0.1
**LLM:** temperature 0.3 · max_tokens 2000 · top_p 0.9 · QLoRA r=16 / α=32 / dropout 0.05
**Training:** wm_epochs 15 · pred_epochs 10 · wm_lr 2e-4 · pred_lr 1e-3 · batch_size 64 · grad_clip 1.0 · weight_decay 1e-4 · AMP fp16 · seed 42 · device CUDA (RTX 4080 SUPER)
**Total trainable parameters:** **10,130,060** · **Training time:** **18.76 min**

## 3. Datasets

DDXPlus (80k train / 10k val / 10k test) · MedMCQA (50k docs) · MedQA (1k queries) — **100,000 total samples**.

## 4. Key Quantitative Results

### World-Model / Predictor Metrics
| Metric | Value |
|---|---|
| MAE | **0.0399** |
| RMSE | **0.0752** |
| 1σ coverage | **0.804** |
| ECE (calibration) | **0.0300** |
| Accuracy | **0.755** |
| F1 (macro) | 0.172 |
| Mean \|SHAP\| | 0.744 |

### GRAPES-SHAP vs Baseline RAG (10 clinical vignettes)
| Metric | Baseline RAG | GRAPES-SHAP |
|---|---|---|
| Clinical concept coverage | 0.700 | **0.967** |
| Answer structure completeness | 0.500 | **0.838** |
| Evidence citations (avg) | 2.00 | **5.10** |
| Stated confidence | 0.910 | 0.700 (calibrated) |
| SHAP attribution (mean \|SHAP\|) | — | **1.281** |
| Calibrated uncertainty | No | **Yes (deep ensemble)** |
| World-model planning | No | **Yes (Tree-of-Thought)** |

## 5. Takeaways

- GRAPES-SHAP improves **concept coverage (+38%)**, **answer completeness (+68%)**, and **evidence grounding (2.5× citations)** over a strong hybrid-RAG baseline.
- Lower but **better-calibrated confidence** (ECE 0.030, 1σ coverage 0.80) reflects honest uncertainty rather than overconfidence.
- Adds **interpretability (SHAP)** and **causal world-model planning** that the baseline lacks, at a compact **10.1M-parameter** footprint trained in **<19 min**.
