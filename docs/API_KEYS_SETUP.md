# API Keys & Environment Setup Guide

## 🎯 Quick Answer: Do You Need API Keys?

**Short answer**: No, not for current implementation. The system works completely without API keys.

| Component | Current Status | API Key Needed? |
|-----------|---|---|
| Data Loading (DDXPlus, MedMCQA, MedQA) | ✅ Working | ❌ No |
| Training (World Model + Ensemble) | ✅ Complete | ❌ No |
| RAG Retrieval (FAISS + BM25) | ✅ Working | ❌ No |
| SHAP Attribution | ✅ Working | ❌ No |
| **Advanced: LLM Reasoning (Stage 8)** | ⏳ Optional | ✅ Yes (Deepseek) |
| **Advanced: HuggingFace Token** | ✅ Works without | ⚠️ Optional |

---

## 📋 Current Implementation Status

Your system is **fully functional** without any API keys:

```
✅ Stage 1: Query Expansion (HyDE) — Using local LLM or heuristics
✅ Stage 2-3: Hybrid Retrieval — FAISS + BM25 (local)
✅ Stage 4: MMR Filtering — Local computation
✅ Stage 5: Causal GNN — Trained, checkpoints saved
✅ Stage 6: World Model — Trained, checkpoints saved
✅ Stage 7: Deep Ensemble — Trained, checkpoints saved
✅ Stage 8: LLM Reasoning — ⏳ Can use local quantized models OR Deepseek API
✅ Stage 9: SHAP + Hallucination Detection — Local computation
```

---

## 🔧 When You Would Need API Keys

### **Deepseek API** (Optional)
**When needed**: If you want to use Deepseek's LLM for Stage 8 (LLM Reasoning) instead of running a local model

**Current alternative**: Run Mistral-7B locally (quantized 4-bit with LoRA)

**Cost**: ~$0.002 per 1M tokens

**How to get**: https://platform.deepseek.com

### **HuggingFace Token** (Optional)
**When needed**: 
- If HF rate limits are exceeded (>20K requests/day without token)
- For private HuggingFace models
- Recommended but not required

**Current status**: Works fine without it

**How to get**: https://huggingface.co/settings/tokens (free)

---

## 📝 Setup Instructions

### Option 1: No API Keys (Recommended for local development)

1. **Create `.env` file** from template:
```bash
cp .env.example .env
```

2. **Verify** that all required paths exist:
```bash
python -m grapes_shap.env_config
```

3. **Start using the system** — no additional setup needed!

### Option 2: With API Keys (For production/Deepseek)

1. **Create `.env` file**:
```bash
cp .env.example .env
```

2. **Get your API keys**:
   - Deepseek: https://platform.deepseek.com/account/api-keys
   - HuggingFace (optional): https://huggingface.co/settings/tokens

3. **Edit `.env`** and add your keys:
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxx
DEVICE=cuda  # Use GPU if available
```

4. **Verify configuration**:
```bash
python -m grapes_shap.env_config
```

---

## 🔑 Getting API Keys

### Deepseek API Key

1. Go to: https://platform.deepseek.com
2. Sign up (free account)
3. Navigate to: Account → API Keys
4. Create new key → Copy to `.env`

**Example `.env` entry:**
```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_MODEL=deepseek-chat
```

### HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" access
4. Copy to `.env`

**Example `.env` entry:**
```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Usage Examples

### Example 1: Run without any API keys
```bash
python scripts/prepare_data.py
python scripts/train_world_model.py
python run.py
# All work perfectly!
```

### Example 2: Add Deepseek for advanced reasoning
```bash
# 1. Update .env
echo "DEEPSEEK_API_KEY=sk-your-key-here" >> .env

# 2. Run with LLM stage enabled
python scripts/inference_with_llm.py --use-deepseek

# 3. System will use Deepseek for Stage 8 (LLM Reasoning)
```

### Example 3: With HuggingFace token (higher rate limits)
```bash
# 1. Update .env
echo "HF_TOKEN=hf_your-token-here" >> .env

# 2. Run normally - faster HF downloads
python run.py
```

---

## 📦 Installing Required Packages

The `.env` loading uses optional `python-dotenv` package:

```bash
# Optional but recommended
pip install python-dotenv
```

If not installed, the system still works but reads only from environment variables or uses hardcoded defaults.

---

## ✅ Verification Checklist

Run this to verify your setup:

```bash
python -c "from grapes_shap.env_config import Config; Config.print_config()"
```

**Expected output:**
```
============================================================
GRAPES-SHAP Configuration
============================================================

Device:          cpu
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Batch Size:      64
Epochs:          15
Learning Rate:   0.0002
RAG Top-K:       6
SHAP Perms:      32

Paths:
  Data:          data
  Output:        outputs
  Checkpoints:   outputs/checkpoints
  Figures:       outputs/figures

⚠ Optional API Keys Not Set:
  • HF_TOKEN: Optional - for higher HuggingFace rate limits
  • DEEPSEEK_API_KEY: Optional - needed for LLM reasoning stage

============================================================
```

---

## 🔒 Security Best Practices

### Do NOT commit `.env` file to git!

1. **Add to `.gitignore`**:
```bash
echo ".env" >> .gitignore
```

2. **Use `.env.example`** for templates (no real keys)

3. **Never log API keys**:
```python
# ✅ Good
logger.info(f"Using API: {api_key[:8]}...hidden")

# ❌ Bad
logger.info(f"API key: {api_key}")  # NEVER DO THIS!
```

---

## 🐛 Troubleshooting

### "No .env file found"
→ This is fine! The system uses defaults. Create one if you want to customize.

### "DEEPSEEK_API_KEY is not set"
→ If not using Deepseek, this is fine. System uses local models.

### "HuggingFace rate limited"
→ Add `HF_TOKEN` to `.env` for higher limits.

### ".env file not loading"
→ Install `python-dotenv`:
```bash
pip install python-dotenv
```

---

## 📄 Sample .env Files

### Minimal (Local development)
```env
DEVICE=cpu
BATCH_SIZE=64
DEBUG=false
```

### With GPU
```env
DEVICE=cuda
BATCH_SIZE=128
LEARNING_RATE=0.0005
```

### With Deepseek
```env
DEVICE=cuda
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_MODEL=deepseek-chat
```

### Production
```env
DEVICE=cuda
HF_TOKEN=hf_your-token
DEEPSEEK_API_KEY=sk-your-key
BATCH_SIZE=256
EPOCHS=20
LOG_LEVEL=INFO
```

---

## 🎓 Summary

| Task | API Needed? | Instructions |
|------|---|---|
| Train models | ❌ No | Just run `python run.py` |
| RAG retrieval | ❌ No | Works out of box |
| SHAP attribution | ❌ No | Already integrated |
| Advanced LLM reasoning | ⚠️ Optional | Get Deepseek key (takes 5 min) |
| Faster downloads | ⚠️ Optional | Get HF token (takes 2 min) |

**→ Start without any keys. Add keys only if you need LLM reasoning!**
