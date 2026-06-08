#!/usr/bin/env python3
"""
Environment configuration loader
Reads from .env file if present, otherwise uses defaults
"""

import os
from pathlib import Path
from typing import Optional

# Try to import dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_env():
    """Load environment variables from .env file if it exists."""
    # Path: src/grapes_shap/env_config.py → go up 2 levels to project root
    env_path = Path(__file__).parent.parent.parent / ".env"
    
    if env_path.exists():
        if DOTENV_AVAILABLE:
            load_dotenv(env_path, override=False)
            print(f"✓ Loaded .env from {env_path}")
        else:
            print(f"⚠ .env file found but python-dotenv not installed")
            print(f"  Install: pip install python-dotenv")
    else:
        print(f"ℹ No .env file found at {env_path}")
        print(f"  Using environment defaults/hardcoded values.")


class Config:
    """Configuration object with environment variable fallbacks."""
    
    # API Keys (Optional)
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    
    # Model config
    DEVICE: str = os.getenv("DEVICE", "cpu")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistral-7b-instruct-v0.2")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Training
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
    EPOCHS: int = int(os.getenv("EPOCHS", "15"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0002"))
    MAX_GRAD_NORM: float = float(os.getenv("MAX_GRAD_NORM", "1.0"))
    
    # RAG
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "6"))
    RAG_MAX_TOKENS: int = int(os.getenv("RAG_MAX_TOKENS", "2048"))
    
    # SHAP
    SHAP_PERMUTATIONS: int = int(os.getenv("SHAP_PERMUTATIONS", "32"))
    
    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    CHECKPOINT_DIR: Path = Path(os.getenv("CHECKPOINT_DIR", "outputs/checkpoints"))
    FIGURE_DIR: Path = Path(os.getenv("FIGURE_DIR", "outputs/figures"))
    
    # System
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    SEED: int = int(os.getenv("SEED", "42"))
    
    @classmethod
    def validate(cls) -> dict:
        """Validate configuration and report missing optional keys."""
        missing = {}
        
        if not cls.HF_TOKEN:
            missing["HF_TOKEN"] = "Optional - for higher HuggingFace rate limits"
        
        if not cls.DEEPSEEK_API_KEY:
            missing["DEEPSEEK_API_KEY"] = "Optional - needed for LLM reasoning stage (production only)"
        
        return missing
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("GRAPES-SHAP Configuration")
        print("="*60)
        print(f"\nDevice:          {cls.DEVICE}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Batch Size:      {cls.BATCH_SIZE}")
        print(f"Epochs:          {cls.EPOCHS}")
        print(f"Learning Rate:   {cls.LEARNING_RATE}")
        print(f"RAG Top-K:       {cls.RAG_TOP_K}")
        print(f"SHAP Perms:      {cls.SHAP_PERMUTATIONS}")
        print(f"\nPaths:")
        print(f"  Data:          {cls.DATA_DIR}")
        print(f"  Output:        {cls.OUTPUT_DIR}")
        print(f"  Checkpoints:   {cls.CHECKPOINT_DIR}")
        print(f"  Figures:       {cls.FIGURE_DIR}")
        
        missing = cls.validate()
        if missing:
            print(f"\n⚠ Optional API Keys Not Set:")
            for key, desc in missing.items():
                print(f"  • {key}: {desc}")
        else:
            print(f"\n✓ All optional API keys configured")
        print("="*60 + "\n")


# Auto-load on import
load_env()


if __name__ == "__main__":
    Config.print_config()
