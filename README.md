# GRAPES-SHAP: Medical AI with SHAP Explainability

A comprehensive medical AI research pipeline combining **DDXPlus**, **MedMCQA**, and **MedQA** datasets with SHAP-based model explainability.

## 🎯 Overview

GRAPES-SHAP (Graph-based Retrieval-Augmented Pipeline for Explainable SHAP) is a research project that:

- **Integrates multiple medical datasets** (DDXPlus, MedMCQA, MedQA)
- **Combines classical ML and deep learning** (Causal GNN, Ensemble models, World model)
- **Provides explainability** using SHAP (SHapley Additive exPlanations)
- **Uses retrieval-augmented generation** for evidence-based predictions
- **Zero-barrier deployment** (all datasets are free & from HuggingFace)

## 📁 Project Structure

```
thesis/
├── src/grapes_shap/           # Main package
│   ├── config.py              # Configuration & hyperparameters
│   ├── data/                  # Data loading & preprocessing
│   ├── models/                # Neural network architectures
│   ├── training/              # Training & evaluation pipelines
│   ├── inference/             # Inference & SHAP attribution
│   └── visualization/         # Analysis & result visualization
├── scripts/                   # Entry point scripts
│   ├── main.py               # Full research pipeline
│   └── grapes_shap_full_research.py  # Alternative implementation
├── tests/                     # Unit tests
├── data/                      # Input datasets (auto-downloaded)
├── outputs/                   # Checkpoints & results
│   ├── checkpoints/          # Trained model weights
│   └── figures/              # Generated visualizations
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── .gitignore                # Git configuration
└── README.md                 # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/thesis.git
cd thesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .
```

### Running the Full Pipeline

```bash
cd scripts
python main.py
```

Or use the alternative research implementation:
```bash
python grapes_shap_full_research.py
```

## 📊 Datasets

| Dataset | Task | Samples | Source |
|---------|------|---------|--------|
| **DDXPlus** | Differential diagnosis | ~10k | HuggingFace |
| **MedMCQA** | Multiple choice QA | ~180k | HuggingFace |
| **MedQA** | USMLE exam questions | ~12k | HuggingFace |

All datasets are **automatically downloaded** during first run. No credentials required.

## 🧠 Model Architecture

- **Evidence Fusion Encoder**: Combines text embeddings from clinical evidence
- **Causal GNN**: Models relationships between symptoms, diseases, and treatments
- **Medical Knowledge Graph**: Structured medical domain knowledge
- **Latent World Model**: Learns disease progression patterns
- **Deep Ensemble**: Combines multiple model predictions
- **SHAP Attributor**: Explains model decisions via Shapley values

## ⚙️ Configuration

Edit `src/grapes_shap/config.py` to customize:
- Batch sizes & learning rates
- Dataset sample sizes
- Model architectures
- Hardware acceleration settings

## 📈 Training

The pipeline includes:
1. **Data Loading & Preprocessing** - Format diverse datasets consistently
2. **Model Training** - World model → Ensemble (with data parallelism)
3. **Evaluation** - Accuracy, F1, ROC-AUC, calibration metrics
4. **SHAP Attribution** - Feature importance analysis
5. **Visualization** - Comprehensive results dashboards

## 🔬 Inference

Hybrid retrieval-augmented inference:
- **BM25** for fast sparse retrieval
- **FAISS** for dense embedding search
- **Reranking** with cross-attention
- **SHAP** for prediction explanations

## 📚 Documentation

See `/docs/` for:
- Architecture details
- API reference
- Research methodology
- Results & benchmarks

## 💻 Hardware Requirements

| Component | Recommended | Minimum |
|-----------|-------------|---------|
| GPU | NVIDIA RTX 4090 (24GB) | RTX 3080 (10GB) |
| CPU | 16+ cores | 8 cores |
| RAM | 64GB | 32GB |
| Storage | 200GB | 100GB |

**Colab Compatible**: Runs on T4 GPU (15GB) with adjusted batch sizes.

## 🛠️ Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code quality
black src/
flake8 src/
mypy src/
```

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- DDXPlus dataset creators
- MedMCQA & MedQA from HuggingFace
- SHAP library (Lundberg et al.)
- PyTorch and HuggingFace communities

## 📮 Contact & Support

For questions or issues:
- Open a GitHub issue
- Contact: your.email@example.com

---

**Last Updated**: 2024
**Status**: Active Research
