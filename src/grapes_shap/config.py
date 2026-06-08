import os
import random
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup output directories (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SAVE_DIR = PROJECT_ROOT / "outputs"
SAVE_DIR.mkdir(exist_ok=True)
FIG_DIR = SAVE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR = SAVE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Detect GPU capabilities for optimization
def _detect_gpu_capabilities():
    """Detect GPU capabilities and return recommended settings."""
    capabilities = {
        "has_cuda": torch.cuda.is_available(),
        "has_fp16": torch.cuda.is_available(),
        "supports_tf32": torch.cuda.is_available(),
        "supports_flash_attention": False,
        "gpu_memory_gb": 0
    }
    
    if torch.cuda.is_available():
        # Check for Flash Attention 2 support (A100, H100, RTX 4090, RTX 4080, etc.)
        device_name = torch.cuda.get_device_name(0).lower()
        capabilities["supports_flash_attention"] = any(
            arch in device_name for arch in ["a100", "h100", "4090", "4080", "4070"]
        )
        
        # Get GPU memory in GB
        total_memory = torch.cuda.get_device_properties(0).total_memory
        capabilities["gpu_memory_gb"] = total_memory / (1024**3)
    
    return capabilities

GPU_CAPABILITIES = _detect_gpu_capabilities()


@dataclass
class Config:
    # Dataset
    ddxplus_n_train: int    = 80_000
    ddxplus_n_val: int      = 10_000
    ddxplus_n_test: int     = 10_000
    medmcqa_n_train: int    = 50_000
    medqa_n_test: int       = 1_000

    # Architecture
    obs_dim: int            = 64
    action_dim: int         = 50
    latent_dim: int         = 256
    hidden_dim: int         = 512
    graph_node_dim: int     = 128
    n_graph_nodes: int      = 20
    n_outcomes: int         = 5
    seq_len: int            = 8
    n_ensemble: int         = 5
    n_heads: int            = 8
    n_transformer_layers: int = 3
    dropout: float          = 0.10

    # Retrieval & Query Expansion (GRAPES Step 2-4)
    top_k: int              = 6
    embed_dim: int          = 384
    shap_perms: int         = 32
    
    # Query expansion
    use_hyde: bool          = True
    hyde_n_subqueries: int  = 3
    mmr_lambda: float       = 0.6  # MMR: 60% relevance, 40% diversity
    fusion_k: int           = 60   # RRF k parameter
    
    # Re-ranking & MMR
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    mmr_enabled: bool       = True

    # Planning & World Model (GRAPES Step 6-7)
    plan_horizon: int       = 4
    plan_candidates: int    = 8
    causal_residual_scale: float = 0.1

    # LLM Configuration (GRAPES Step 9)
    llm_model: str          = "deepseek-chat"
    llm_api_provider: str   = "deepseek"  # "deepseek" or "openai"
    llm_temperature: float  = 0.3
    llm_max_tokens: int     = 2000
    llm_top_p: float        = 0.9
    use_qlora: bool         = True
    qlora_rank: int         = 16
    qlora_alpha: int        = 32
    qlora_dropout: float    = 0.05

    # Hallucination Detection (GRAPES Step 10)
    use_self_rag: bool      = True
    use_nli_verification: bool = True
    hallucination_threshold: float = 0.15

    # SHAP Interpretation (GRAPES Step 11)
    use_pairwise_interactions: bool = True
    shap_top_k_interactions: int = 6

    # Training
    wm_epochs: int          = 15
    pred_epochs: int        = 10
    wm_lr: float            = 2e-4
    pred_lr: float          = 1e-3
    batch_size: int         = 64
    grad_clip: float        = 1.0
    weight_decay: float     = 1e-4
    amp_dtype: torch.dtype  = torch.float16

    # GPU Optimization Features
    use_mixed_precision: bool = GPU_CAPABILITIES["has_fp16"]
    use_flash_attention: bool = GPU_CAPABILITIES["supports_flash_attention"]
    use_gradient_checkpointing: bool = True
    use_tf32: bool          = GPU_CAPABILITIES["supports_tf32"]
    
    # Memory optimization based on GPU size
    adaptive_batch_size: bool = True
    target_gpu_memory_gb: float = max(8, GPU_CAPABILITIES["gpu_memory_gb"] * 0.8)

    # System
    device: str             = DEVICE
    seed: int               = 42
    
    # API Keys & External Services
    deepseek_api_key: str   = os.getenv("DEEPSEEK_API_KEY", "")
    openai_api_key: str     = os.getenv("OPENAI_API_KEY", "")
    
    def print_config(self):
        """Print configuration for debugging."""
        print("\n" + "="*60)
        print("GRAPES-SHAP Configuration")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"GPU Memory: {GPU_CAPABILITIES['gpu_memory_gb']:.1f} GB")
        print(f"Flash Attention 2: {self.use_flash_attention}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"\nGRAPES Pipeline Settings:")
        print(f"  - Query Expansion (HyDE): {self.use_hyde}")
        print(f"  - Cross-Encoder Re-ranking: {self.use_cross_encoder}")
        print(f"  - MMR Diversity: {self.mmr_enabled}")
        print(f"  - LLM: {self.llm_model} (QLoRA: {self.use_qlora})")
        print(f"  - Hallucination Detection: Self-RAG={self.use_self_rag}, NLI={self.use_nli_verification}")
        print(f"  - SHAP Interactions: {self.use_pairwise_interactions}")
        print("="*60 + "\n")

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

CFG = Config()
set_seed(CFG.seed)
