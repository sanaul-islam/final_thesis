# GRAPES-SHAP Quick Start Guide

## 📋 Project Structure Overview

Your thesis project is now organized as a professional Python package:

```
thesis/
├── 📂 src/grapes_shap/        ← Main package code
├── 📂 scripts/                ← Entry point scripts
├── 📂 tests/                  ← Unit tests (add tests here)
├── 📂 data/                   ← Input datasets
├── 📂 outputs/                ← Results & checkpoints
├── 📂 docs/                   ← Documentation
├── 🔧 setup.py               ← Package installation config
├── 📄 requirements.txt         ← Dependencies
├── 📄 README.md               ← Project overview
├── 📄 .gitignore              ← Git exclusions
├── 📄 LICENSE                 ← MIT License
└── ▶️  run.py                 ← Quick launcher
```

## 🚀 How to Run

### Option 1: Quick Start (Recommended)
```bash
cd thesis
python run.py
```

### Option 2: Direct Script
```bash
cd scripts
python main.py
```

### Option 3: Install as Package
```bash
cd thesis
pip install -e .
python scripts/main.py
```

## 📦 Dependencies

All required packages are listed in `requirements.txt`:
```
numpy, pandas, torch, matplotlib, seaborn, scikit-learn,
tqdm, datasets, sentence-transformers, faiss-cpu, rank-bm25
```

Install with:
```bash
pip install -r requirements.txt
```

## 🗂️ Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/grapes_shap/` | Core package - don't move! |
| `scripts/` | Entry points (main.py, research scripts) |
| `tests/` | Add unit tests here |
| `data/` | Input datasets (auto-downloaded) |
| `outputs/` | Model checkpoints & visualizations |
| `docs/` | Project documentation |

## 💻 Development Tips

### Adding New Modules
Place them in `src/grapes_shap/`:
```python
# In src/grapes_shap/mymodule.py
def my_function():
    pass

# Import it in scripts
from grapes_shap.mymodule import my_function
```

### Using Config
```python
from grapes_shap.config import CFG
print(CFG.ddxplus_n_train)
```

### Accessing Outputs
Models saved to:
- Checkpoints: `outputs/checkpoints/`
- Figures: `outputs/figures/`

## 🔄 Project Workflow

1. **Modify code** → Edit files in `src/grapes_shap/`
2. **Run pipeline** → `python run.py`
3. **Check outputs** → View `outputs/checkpoints/` and `outputs/figures/`
4. **Write tests** → Add to `tests/`
5. **Document** → Update `docs/`

## 📝 Next Steps

- [ ] Configure `src/grapes_shap/config.py` with your hyperparameters
- [ ] Test the pipeline: `python run.py`
- [ ] Review outputs in `outputs/` folder
- [ ] Add unit tests to `tests/`
- [ ] Update author info in `setup.py`
- [ ] Initialize git: `git init && git add . && git commit -m "Initial commit"`

## 🆘 Troubleshooting

### Import errors?
Make sure you're running from the project root or using `python run.py`

### Missing data?
First run auto-downloads from HuggingFace. Check your internet connection.

### GPU issues?
Install CPU version: `pip install faiss-cpu`
Or GPU version: `pip install faiss-gpu` (requires CUDA)

## 📚 Documentation

See `docs/PROJECT_STRUCTURE.md` for detailed architecture information.
