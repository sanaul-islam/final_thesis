# GRAPES-SHAP Environment Setup

## Quick Start

### 1. Install Dependencies

```bash
cd /path/to/thesis
pip install -r requirements.txt
```

### 2. Set DeepSeek API Key

```bash
# Option 1: Set environment variable
export DEEPSEEK_API_KEY="sk-..."

# Option 2: Create .env file (not recommended for production)
echo "DEEPSEEK_API_KEY=sk-..." > .env

# Option 3: Set in code
import os
os.environ["DEEPSEEK_API_KEY"] = "sk-..."
```

### 3. Verify GPU Setup

```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"
```

### 4. Run Example

```bash
python example_grapes_usage.py
```

---

## Detailed Setup Guide

### System Requirements

**Minimum:**
- Python 3.9+
- 8GB RAM
- 16GB GPU memory (for LLM with QLoRA)

**Recommended:**
- Python 3.10+
- 32GB RAM
- 24GB GPU memory (RTX 4090 / A100)
- CUDA 11.8+
- cuDNN 8.6+

### GPU Compatibility

#### Automatically Optimized GPUs
These GPUs automatically enable Flash Attention 2:
- **NVIDIA:**
  - RTX 4090, 4080, 4070, 4060
  - RTX 6000, 5880, 5000, 4500
  - A100, H100
  - L40, L40S
- **Other:**
  - AMD MI300X
  - Intel Arc A770

#### Standard Compatibility (no Flash Attention)
- RTX 3090, 3080, 3070
- RTX 2080, 2070
- V100, T4

### Installation Steps

#### 1. Python Environment

```bash
# Create virtual environment
python3 -m venv thesis_venv
source thesis_venv/bin/activate  # On Windows: thesis_venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 2. Install PyTorch (GPU version)

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (fallback)
pip install torch torchvision torchaudio
```

#### 3. Install GRAPES-SHAP Dependencies

```bash
pip install -r requirements.txt

# Optional: For better performance
pip install --upgrade transformers
pip install flash-attn  # For Flash Attention 2 (optional, CUDA only)
```

#### 4. Verify Installation

```bash
python -c "
from grapes_shap.config import Config
from grapes_shap.inference.grapes_pipeline import GRAPESPipeline

cfg = Config()
cfg.print_config()
print('✓ GRAPES-SHAP installation verified')
"
```

---

## DeepSeek API Setup

### Get API Key

1. Visit https://platform.deepseek.com
2. Sign up or log in
3. Navigate to API Keys section
4. Create new API key
5. Copy the key (starts with `sk-`)

### Configure API Key

#### Option 1: Environment Variable (Recommended)

```bash
# Linux/Mac
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxx"

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxx"

# Windows (Command Prompt)
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx

# Permanent (Linux/Mac) - add to ~/.bashrc or ~/.zshrc
echo 'export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

#### Option 2: Configuration File

Create `.env` file in project root:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxx  # Optional
```

Then load in code:
```python
from dotenv import load_dotenv
load_dotenv()

import os
api_key = os.getenv("DEEPSEEK_API_KEY")
```

#### Option 3: Direct Configuration

```python
from grapes_shap.config import Config

cfg = Config()
cfg.deepseek_api_key = "sk-xxxxxxxxxxxxx"
```

### Test API Connection

```python
from grapes_shap.inference.deepseek_llm import DeepSeekLLMClient
from grapes_shap.config import Config
import os

cfg = Config()
cfg.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

llm = DeepSeekLLMClient(cfg)

# Test call
output = llm.generate_medical_recommendation(
    query="What is EGFR mutation?",
    evidence_docs=["EGFR mutations are..."],
    world_model_results={"best_score": 0.8}
)

print(f"✓ API working: {output.recommendation[:50]}...")
```

---

## Troubleshooting

### 1. CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# 1. Install NVIDIA GPU drivers
# 2. Install CUDA Toolkit
# 3. Install cuDNN
# 4. Reinstall PyTorch with cuda support
```

### 2. DeepSeek API Error

```
Error: Could not authenticate with DeepSeek API
```

**Solutions:**
1. Verify API key is correct
2. Check API key is active on DeepSeek dashboard
3. Check internet connection
4. Verify API endpoint is reachable: `https://api.deepseek.com`

### 3. Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Enable QLoRA (4-bit):
   ```python
   cfg.use_qlora = True
   ```
2. Enable gradient checkpointing:
   ```python
   cfg.use_gradient_checkpointing = True
   ```
3. Reduce batch size:
   ```python
   cfg.batch_size = 32  # from 64
   ```
4. Use CPU for some models:
   ```python
   cfg.device = "cpu"  # or "mps" for Mac
   ```

### 4. Slow Performance

**Optimization checklist:**
```python
cfg = Config()

# ✓ Enable Flash Attention 2 (auto-detected)
# ✓ Enable mixed precision
cfg.use_mixed_precision = True

# ✓ Enable gradient checkpointing
cfg.use_gradient_checkpointing = True

# ✓ Use TF32 (A100+ only)
cfg.use_tf32 = True

# ✓ Check GPU is actually being used
import torch
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
```

---

## Advanced Configuration

### Enable All GPU Features

```python
from grapes_shap.config import Config

cfg = Config()

# GPU optimization
cfg.use_mixed_precision = True
cfg.use_flash_attention = True
cfg.use_gradient_checkpointing = True
cfg.use_tf32 = True

# GRAPES features
cfg.use_hyde = True
cfg.use_cross_encoder = True
cfg.mmr_enabled = True
cfg.use_self_rag = True
cfg.use_nli_verification = True
cfg.use_pairwise_interactions = True

# LLM optimization
cfg.use_qlora = True
cfg.qlora_rank = 16

print("✓ Advanced optimizations enabled")
cfg.print_config()
```

### Memory Profiling

```python
import torch
import pytorch_memlab
from pytorch_memlab import profile

# Profile GRAPES pipeline
@profile
def infer_with_profiling():
    from grapes_shap.inference.grapes_pipeline import GRAPESPipeline
    from grapes_shap.config import Config
    
    cfg = Config()
    pipeline = GRAPESPipeline(cfg)
    output = pipeline.infer("query", ["doc1", "doc2"])
    return output

# Run with profiling
mem_report = infer_with_profiling()
print(mem_report)
```

### Performance Benchmarking

```python
import time
from grapes_shap.config import Config
from grapes_shap.inference.grapes_pipeline import GRAPESPipeline

cfg = Config()
pipeline = GRAPESPipeline(cfg)

# Warm up
_ = pipeline.infer("test query", ["doc"])

# Benchmark
times = []
for _ in range(5):
    start = time.time()
    _ = pipeline.infer("test query", ["doc"])
    times.append(time.time() - start)

print(f"Mean time: {sum(times)/len(times):.2f}s")
print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")
```

---

## Docker Setup (Optional)

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install flash-attn

# Copy code
COPY . .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "example_grapes_usage.py"]
```

### Build and Run

```bash
# Build
docker build -t grapes-shap:latest .

# Run
docker run --gpus all -v $(pwd):/app grapes-shap:latest
```

---

## Performance Optimization Tips

### For RTX 4090 (24GB)
```python
cfg.batch_size = 64
cfg.use_flash_attention = True
cfg.use_mixed_precision = True
cfg.use_tf32 = False  # Not available
```

### For RTX 4080 (16GB)
```python
cfg.batch_size = 32
cfg.use_flash_attention = True
cfg.use_mixed_precision = True
cfg.use_qlora = True
```

### For A100 (80GB)
```python
cfg.batch_size = 256
cfg.use_flash_attention = True
cfg.use_mixed_precision = True
cfg.use_tf32 = True  # A100 supports TF32
cfg.use_qlora = False  # Not needed
```

### For CPU Only
```python
cfg.device = "cpu"
cfg.batch_size = 8
cfg.use_mixed_precision = False
cfg.use_qlora = False
```

---

## References

- [NVIDIA CUDA Installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [DeepSeek API Docs](https://platform.deepseek.com/docs)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)

---

## Support

For issues:
1. Check troubleshooting section above
2. Verify all dependencies installed: `pip list | grep -E "torch|transformers|peft"`
3. Test GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Check API key: `echo $DEEPSEEK_API_KEY`
5. Review logs in `outputs/` directory
