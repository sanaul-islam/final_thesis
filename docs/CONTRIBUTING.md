# Contributing Guidelines

## Development Setup

### 1. Clone and Install
```bash
git clone https://github.com/yourusername/thesis.git
cd thesis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Verify Installation
```bash
python run.py
```

## Project Structure for Developers

### Where to Put Code

| Task | Location |
|------|----------|
| New data loaders | `src/grapes_shap/data/` |
| New model architectures | `src/grapes_shap/models/` |
| New training pipelines | `src/grapes_shap/training/` |
| Inference engines | `src/grapes_shap/inference/` |
| Visualizations | `src/grapes_shap/visualization/` |
| Entry point scripts | `scripts/` |
| Unit tests | `tests/` |
| Documentation | `docs/` |

### Module Organization

Each submodule should have:
```
module/
├── __init__.py      # Exports public API
├── core.py          # Main implementation
├── utils.py         # Helper functions
└── __pycache__/     # (auto-generated)
```

### Example: Adding a New Module

1. Create `src/grapes_shap/new_module/`:
```bash
mkdir src/grapes_shap/new_module
touch src/grapes_shap/new_module/__init__.py
touch src/grapes_shap/new_module/core.py
```

2. Write code in `core.py`:
```python
def my_function():
    """My new function."""
    pass
```

3. Export in `__init__.py`:
```python
from .core import my_function

__all__ = ['my_function']
```

4. Use in scripts:
```python
from grapes_shap.new_module import my_function
```

## Code Style

### Python Style Guide (PEP 8)

Format code with Black:
```bash
pip install black
black src/
```

Check with Flake8:
```bash
pip install flake8
flake8 src/ --max-line-length=100
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `MedicalKG`)
- **Functions**: `snake_case` (e.g., `load_dataset`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_TOKENS`)
- **Private**: `_leading_underscore` (e.g., `_internal_helper`)

### Documentation

Add docstrings to all public functions:
```python
def train_model(config: Dict, data: DataLoader) -> Model:
    """Train a model on the given data.
    
    Args:
        config: Training configuration dictionary
        data: DataLoader with training samples
        
    Returns:
        Trained model object
        
    Raises:
        ValueError: If config is invalid
    """
    pass
```

## Testing

### Write Tests

Tests go in `tests/` with matching module structure:
```
tests/
├── test_data.py           # Tests for src/grapes_shap/data/
├── test_models.py         # Tests for src/grapes_shap/models/
├── test_inference.py      # Tests for src/grapes_shap/inference/
└── ...
```

Example test:
```python
# tests/test_models.py
import pytest
from grapes_shap.models import MedicalKG

def test_medical_kg_initialization():
    """Test MedicalKG initializes correctly."""
    kg = MedicalKG()
    assert kg is not None
    
def test_medical_kg_forward():
    """Test MedicalKG forward pass."""
    kg = MedicalKG()
    # ... test forward pass
    assert output.shape == expected_shape
```

### Run Tests
```bash
pytest tests/ -v
pytest tests/ --cov=src/grapes_shap
```

## Git Workflow

### Commit Messages

Use clear, descriptive messages:
```
✨ feat: Add SHAP attribution for ensemble models
🐛 fix: Resolve data loader batch size issue
📝 docs: Update API reference
🧹 refactor: Reorganize model architecture
⚡ perf: Optimize embedding computation
```

### Before Committing

1. Format code: `black src/`
2. Check style: `flake8 src/`
3. Run tests: `pytest tests/`
4. Update docs if needed

### Push to Remote

```bash
git add .
git commit -m "✨ feat: Brief description"
git push origin feature-branch
```

## Configuration

### Edit Config

Update `src/grapes_shap/config.py` for:
- Batch sizes
- Learning rates
- Dataset sizes
- Model hyperparameters
- Hardware settings

### Environment Variables

Create `.env` (not tracked by git):
```
CUDA_VISIBLE_DEVICES=0,1
WANDB_API_KEY=your_key_here
DEBUG=True
```

Load in code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
debug = os.getenv('DEBUG', False)
```

## Documentation

### Updating README
Edit top-level `README.md` for:
- Major changes
- New datasets
- New models
- Installation changes

### Updating Docs
Add details to `docs/` for:
- Architecture explanations
- Algorithm descriptions
- Experimental results
- Implementation notes

## Performance & Profiling

### Memory Profiling
```bash
pip install memory-profiler
python -m memory_profiler scripts/main.py
```

### Time Profiling
```bash
pip install line-profiler
kernprof -l -v scripts/main.py
```

### GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

## Debugging Tips

### Print Debugging
```python
print(f"DEBUG: variable = {variable}, shape = {variable.shape}")
```

### Interactive Debugging
```python
import pdb; pdb.set_trace()
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Checkpoint saved")
```

## Dependencies

### Adding New Package

1. Install: `pip install package-name`
2. Add to `requirements.txt`
3. Update `setup.py` if needed
4. Commit: `git add requirements.txt && git commit`

### Version Pinning
Specify versions in `requirements.txt`:
```
torch==2.0.0
numpy>=1.21.0,<2.0.0
```

## Review Checklist

Before submitting code:

- [ ] Code follows PEP 8 style
- [ ] Docstrings added to new functions
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No large files added (use Git LFS for >100MB)
- [ ] Sensitive data not committed
- [ ] Imports organized (stdlib, third-party, local)
- [ ] No print statements (use logging)
- [ ] Performance checked if applicable

## Questions?

- Check existing code for examples
- Read docstrings and comments
- Open an issue on GitHub
- Contact: your.email@example.com

---

Happy coding! 🚀
