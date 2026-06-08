"""
Project structure and organization documentation.

Directory Layout
================

src/grapes_shap/
    Core package containing all functionality:
    - config.py: Global configuration and paths
    - data/: Dataset loading, preprocessing, and loaders
    - models/: Neural network architectures
    - training/: Model training and evaluation pipelines
    - inference/: Inference engines and SHAP attribution
    - visualization/: Analysis and result visualization

scripts/
    Entry point scripts for running the full pipeline:
    - main.py: Full research pipeline orchestration
    - grapes_shap_full_research.py: Alternative implementation

tests/
    Unit tests and integration tests for all modules.

data/
    Input datasets (auto-downloaded on first run):
    - raw/: Original dataset files
    - processed/: Preprocessed data

outputs/
    Results, checkpoints, and visualizations:
    - checkpoints/: Trained model weights (.pt files)
    - figures/: Generated plots and visualizations

docs/
    Project documentation:
    - Architecture overview
    - API reference
    - Research methodology
    - Results and benchmarks

Root Files
==========

setup.py: Package installation configuration
    Install with: pip install -e .

requirements.txt: Direct Python dependency specification

README.md: Project overview and quick start guide

.gitignore: Git exclusion rules

LICENSE: MIT License

.env (optional): Environment variables (not in git)


Running the Code
================

Option 1: Using setup.py (Recommended)
    pip install -e .
    cd scripts
    python main.py

Option 2: Direct script execution
    cd scripts
    python main.py

Option 3: In Python
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('src')))
    from grapes_shap.config import CFG
    # ... rest of code
"""
