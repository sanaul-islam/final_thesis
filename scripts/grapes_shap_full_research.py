#!/usr/bin/env python3
"""
GRAPES-SHAP: Full End-to-End Research Pipeline
Datasets: DDXPlus + MedMCQA + MedQA (all free, no credentials, HuggingFace)
Hardware: RTX 4090 (24GB) | Colab T4 (15GB) compatible
"""

# ──────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# Install in Colab: !pip install datasets sentence-transformers faiss-gpu
#                   rank-bm25 transformers peft bitsandbytes scikit-learn
#                   matplotlib seaborn plotly umap-learn tqdm accelerate
# ──────────────────────────────────────────────────────────────

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os, json, math, random, warnings, time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = Path("grapes_shap_outputs")
SAVE_DIR.mkdir(exist_ok=True)
FIG_DIR = SAVE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
CKPT_DIR = SAVE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"Device: {DEVICE}")




# ──────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────

# @dataclass
# class Config:
#     # Dataset
#     ddxplus_n_train: int    = 80_000
#     ddxplus_n_val: int      = 10_000
#     ddxplus_n_test: int     = 10_000
#     medmcqa_n_train: int    = 50_000
#     medqa_n_test: int       = 1_000

#     # Architecture
#     obs_dim: int            = 64
#     action_dim: int         = 50
#     latent_dim: int         = 256
#     hidden_dim: int         = 512
#     graph_node_dim: int     = 128
#     n_graph_nodes: int      = 20
#     n_outcomes: int         = 5
#     seq_len: int            = 8
#     n_ensemble: int         = 5
#     n_heads: int            = 8
#     n_transformer_layers: int = 3
#     dropout: float          = 0.10

#     # Retrieval
#     top_k: int              = 6
#     embed_dim: int          = 384
#     shap_perms: int         = 32

#     # Planning
#     plan_horizon: int       = 4
#     plan_candidates: int    = 8

#     # Training
#     wm_epochs: int          = 15
#     pred_epochs: int        = 10
#     wm_lr: float            = 2e-4
#     pred_lr: float          = 1e-3
#     batch_size: int         = 64
#     grad_clip: float        = 1.0
#     weight_decay: float     = 1e-4
#     amp_dtype: torch.dtype  = torch.float16

#     # System
#     device: str             = DEVICE
#     seed: int               = 42


# def set_seed(s):
#     random.seed(s); np.random.seed(s)
#     torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# CFG = Config()
# set_seed(CFG.seed)


# # ──────────────────────────────────────────────────────────────
# # 2. DATA LOADING — DDXPlus + MedMCQA + MedQA
# # ──────────────────────────────────────────────────────────────

# class DatasetLoader:
#     """
#     Loads three zero-barrier HuggingFace datasets:
#       • DDXPlus    — 1.3M patients, differential diagnosis, symptoms, antecedents
#       • MedMCQA   — 194K clinical QA with explanations (reasoning corpus)
#       • MedQA     — 12K USMLE multi-step clinical reasoning questions
#     """

#     @staticmethod
#     def load_ddxplus(n_train=80_000, n_val=10_000, n_test=10_000):
#         from datasets import load_dataset
#         print("  Loading DDXPlus from HuggingFace (no credentials needed)...")
#         ds = load_dataset("aai530-group6/ddxplus", trust_remote_code=True)
#         train_raw = ds["train"].select(range(min(n_train, len(ds["train"]))))
#         val_raw   = ds["validate"].select(range(min(n_val, len(ds["validate"]))))
#         test_raw  = ds["test"].select(range(min(n_test, len(ds["test"]))))
#         print(f"  DDXPlus — train:{len(train_raw)} val:{len(val_raw)} test:{len(test_raw)}")
#         return train_raw, val_raw, test_raw

#     @staticmethod
#     def load_medmcqa(n=50_000):
#         from datasets import load_dataset
#         print("  Loading MedMCQA from HuggingFace...")
#         ds = load_dataset("openlifescienceai/medmcqa", trust_remote_code=True)
#         data = ds["train"].select(range(min(n, len(ds["train"]))))
#         print(f"  MedMCQA — {len(data)} QA samples, 21 medical subjects")
#         return data

#     @staticmethod
#     def load_medqa(n=1_000):
#         from datasets import load_dataset
#         print("  Loading MedQA-USMLE from HuggingFace...")
#         ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
#         data = ds["test"].select(range(min(n, len(ds["test"]))))
#         print(f"  MedQA — {len(data)} USMLE test questions")
#         return data


# # ──────────────────────────────────────────────────────────────
# # 3. DATA PREPROCESSING
# # ──────────────────────────────────────────────────────────────

# class DDXPlusPreprocessor:
#     """
#     Converts raw DDXPlus records into trajectory tensors suitable
#     for world model training.

#     DDXPlus record fields:
#       AGE, SEX, PATHOLOGY, SYMPTOMS (list), ANTECEDENTS (list),
#       DIFFERENTIAL_DIAGNOSIS (list of [name, prob] pairs), EVIDENCES (list)

#     We treat each patient as a clinical trajectory:
#       obs_t     — normalised feature vector (age, sex, symptom binary flags, antecedent flags)
#       action_t  — index of the next evidence/symptom collected (simulated sequential inquiry)
#       outcome   — probability distribution over differential diagnoses
#     """

#     def __init__(self, cfg: Config):
#         self.cfg = cfg
#         self.symptom_vocab: Dict[str, int] = {}
#         self.antecedent_vocab: Dict[str, int] = {}
#         self.pathology_vocab: Dict[str, int] = {}
#         self.evidence_vocab: Dict[str, int] = {}
#         self.fitted = False

#     def fit(self, records):
#         all_symptoms, all_antecedents, all_pathologies, all_evidences = set(), set(), set(), set()
#         for r in records:
#             if isinstance(r.get("SYMPTOMS", []), list):
#                 for s in r.get("SYMPTOMS", []):
#                     all_symptoms.add(str(s))
#             if isinstance(r.get("ANTECEDENTS", []), list):
#                 for a in r.get("ANTECEDENTS", []):
#                     all_antecedents.add(str(a))
#             if r.get("PATHOLOGY"):
#                 all_pathologies.add(str(r["PATHOLOGY"]))
#             if isinstance(r.get("EVIDENCES", []), list):
#                 for e in r.get("EVIDENCES", []):
#                     all_evidences.add(str(e))

#         self.symptom_vocab    = {s: i for i, s in enumerate(sorted(all_symptoms))}
#         self.antecedent_vocab = {a: i for i, a in enumerate(sorted(all_antecedents))}
#         self.pathology_vocab  = {p: i for i, p in enumerate(sorted(all_pathologies))}
#         self.evidence_vocab   = {e: i for i, e in enumerate(sorted(all_evidences))}
#         self.fitted = True
#         print(f"  Vocabulary — symptoms:{len(self.symptom_vocab)} "
#               f"antecedents:{len(self.antecedent_vocab)} "
#               f"pathologies:{len(self.pathology_vocab)} "
#               f"evidences:{len(self.evidence_vocab)}")

#     def _encode_patient(self, r) -> Dict[str, np.ndarray]:
#         obs_dim = self.cfg.obs_dim
#         obs = np.zeros(obs_dim, dtype=np.float32)

#         # Features 0-1: age (normalised 0-1) and sex
#         age = float(r.get("AGE", 40)) / 100.0
#         sex = 1.0 if str(r.get("SEX", "M")).upper() == "M" else 0.0
#         obs[0] = age
#         obs[1] = sex

#         # Features 2-33: symptom binary flags (32 dim)
#         syms = r.get("SYMPTOMS", [])
#         if isinstance(syms, list):
#             for s in syms:
#                 idx = self.symptom_vocab.get(str(s), -1)
#                 if idx >= 0 and 2 + (idx % 32) < obs_dim:
#                     obs[2 + (idx % 32)] = 1.0

#         # Features 34-63: antecedent flags (30 dim)
#         ants = r.get("ANTECEDENTS", [])
#         if isinstance(ants, list):
#             for a in ants:
#                 idx = self.antecedent_vocab.get(str(a), -1)
#                 if idx >= 0 and 34 + (idx % 30) < obs_dim:
#                     obs[34 + (idx % 30)] = 1.0

#         # Build a simulated trajectory by presenting evidence incrementally
#         evs = r.get("EVIDENCES", [])
#         if not isinstance(evs, list):
#             evs = []
#         T = self.cfg.seq_len
#         obs_seq = np.zeros((T, obs_dim), dtype=np.float32)
#         actions  = np.zeros(T, dtype=np.int64)
#         obs_seq[0] = obs.copy()
#         for t in range(1, T):
#             if t - 1 < len(evs):
#                 ev_idx = self.evidence_vocab.get(str(evs[t-1]), 0)
#                 actions[t-1] = ev_idx % self.cfg.action_dim
#                 feat_slot = 2 + (ev_idx % 32)
#                 if feat_slot < obs_dim:
#                     obs_seq[t] = obs_seq[t-1].copy()
#                     obs_seq[t][feat_slot] = min(1.0, obs_seq[t-1][feat_slot] + 0.15)
#             else:
#                 obs_seq[t] = obs_seq[t-1].copy()

#         # Outcomes: differential diagnosis probabilities (up to 5 pathologies)
#         diff_diag = r.get("DIFFERENTIAL_DIAGNOSIS", [])
#         outcomes = np.zeros(self.cfg.n_outcomes, dtype=np.float32)
#         if isinstance(diff_diag, list):
#             for k, pair in enumerate(diff_diag[:self.cfg.n_outcomes]):
#                 if isinstance(pair, (list, tuple)) and len(pair) == 2:
#                     outcomes[k] = float(pair[1])
#         # Normalise so probabilities sum to 1
#         s = outcomes.sum()
#         if s > 0:
#             outcomes /= s
#         else:
#             outcomes[0] = 1.0

#         return {"obs": obs_seq, "actions": actions,
#                 "next_obs": np.roll(obs_seq, -1, axis=0),
#                 "outcomes": outcomes,
#                 "pathology_id": self.pathology_vocab.get(str(r.get("PATHOLOGY", "")), 0)}

#     def transform(self, records) -> List[Dict]:
#         processed = []
#         for r in tqdm(records, desc="  Encoding patients", leave=False):
#             try:
#                 processed.append(self._encode_patient(r))
#             except Exception:
#                 continue
#         return processed


# class MedMCQAPreprocessor:
#     """
#     Converts MedMCQA QA items into a knowledge base corpus for RAG retrieval.
#     Each item becomes a retrievable document: question + correct answer + explanation.
#     """
#     @staticmethod
#     def to_documents(records) -> List[str]:
#         docs = []
#         for r in records:
#             q = r.get("question", "")
#             opts = [r.get("opa",""), r.get("opb",""), r.get("opc",""), r.get("opd","")]
#             correct_idx = r.get("cop", 0)
#             if isinstance(correct_idx, int) and 0 <= correct_idx < 4:
#                 answer = opts[correct_idx]
#             else:
#                 answer = opts[0]
#             exp = r.get("exp", "") or ""
#             subject = r.get("subject_name", "Medicine")
#             doc = f"[{subject}] Q: {q} A: {answer}. {exp[:200]}"
#             docs.append(doc.strip())
#         return [d for d in docs if len(d) > 20]


# class MedQAPreprocessor:
#     """
#     Converts MedQA USMLE items into evaluation queries + ground truth for
#     benchmarking the GRAPES-SHAP pipeline on USMLE-style clinical vignettes.
#     """
#     @staticmethod
#     def to_queries(records) -> List[Dict]:
#         queries = []
#         for r in records:
#             q = r.get("question", "")
#             opts = r.get("options", {})
#             answer_key = r.get("answer_idx", r.get("answer", "A"))
#             if isinstance(opts, dict):
#                 options = opts
#             else:
#                 options = {"A": str(opts[0]) if opts else "", "B": "", "C": "", "D": ""}
#             correct = options.get(str(answer_key), "")
#             queries.append({"question": q, "options": options,
#                             "answer_key": str(answer_key), "correct_answer": correct})
#         return [q for q in queries if len(q["question"]) > 10]


# # ──────────────────────────────────────────────────────────────
# # 4. PyTorch DATASET WRAPPER
# # ──────────────────────────────────────────────────────────────

# class ClinicalTrajectoryDataset(Dataset):
#     def __init__(self, records: List[Dict]):
#         self.data = records

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         r = self.data[i]
#         return {
#             "obs":      torch.from_numpy(r["obs"]).float(),
#             "actions":  torch.from_numpy(r["actions"]).long(),
#             "next_obs": torch.from_numpy(r["next_obs"]).float(),
#             "outcomes": torch.from_numpy(r["outcomes"]).float(),
#             "pathology_id": torch.tensor(r["pathology_id"], dtype=torch.long),
#         }


# # ──────────────────────────────────────────────────────────────
# # 5. VISUALISATION — DATA EXPLORATION
# # ──────────────────────────────────────────────────────────────

# def plot_dataset_overview(processed_train: List[Dict],
#                           processed_val: List[Dict],
#                           processed_test: List[Dict],
#                           preprocessor: DDXPlusPreprocessor):
#     fig = plt.figure(figsize=(20, 14))
#     fig.patch.set_facecolor("#0f1117")
#     gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

#     CYAN = "#38bdf8"; TEAL = "#2dd4bf"; AMBER = "#fbbf24"
#     ROSE = "#f87171"; VIOLET = "#a78bfa"; GREEN = "#4ade80"

#     def ax_style(ax, title):
#         ax.set_facecolor("#161b27")
#         ax.tick_params(colors="#8892a4", labelsize=8)
#         ax.spines[["top","right","left","bottom"]].set_edgecolor("#2a3348")
#         ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
#         for sp in ax.spines.values():
#             sp.set_linewidth(0.5)

#     # 1. Split sizes
#     ax1 = fig.add_subplot(gs[0, 0])
#     splits = ["Train", "Validation", "Test"]
#     sizes  = [len(processed_train), len(processed_val), len(processed_test)]
#     colors = [CYAN, TEAL, VIOLET]
#     bars = ax1.bar(splits, sizes, color=colors, width=0.55, edgecolor="#0f1117", linewidth=0.8)
#     for b, s in zip(bars, sizes):
#         ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 200,
#                  f"{s:,}", ha="center", va="bottom", color="#e2e8f0", fontsize=8)
#     ax_style(ax1, "Dataset Split Sizes")
#     ax1.set_ylabel("Samples", color="#8892a4", fontsize=8)

#     # 2. Age distribution
#     ax2 = fig.add_subplot(gs[0, 1])
#     ages = [r["obs"][0, 0] * 100 for r in processed_train[:5000]]
#     ax2.hist(ages, bins=30, color=TEAL, edgecolor="#0f1117", linewidth=0.5, alpha=0.85)
#     ax_style(ax2, "Patient Age Distribution")
#     ax2.set_xlabel("Age (years)", color="#8892a4", fontsize=8)
#     ax2.set_ylabel("Count", color="#8892a4", fontsize=8)

#     # 3. Sex distribution
#     ax3 = fig.add_subplot(gs[0, 2])
#     sexes  = [r["obs"][0, 1] for r in processed_train[:5000]]
#     male   = sum(1 for s in sexes if s > 0.5)
#     female = len(sexes) - male
#     ax3.pie([male, female], labels=["Male", "Female"],
#             colors=[CYAN, ROSE], autopct="%1.1f%%",
#             textprops={"color": "#e2e8f0", "fontsize": 8},
#             wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.8})
#     ax_style(ax3, "Sex Distribution")

#     # 4. Pathology distribution (top 15)
#     ax4 = fig.add_subplot(gs[0, 3])
#     path_ids = [r["pathology_id"] for r in processed_train[:10000]]
#     path_counts = Counter(path_ids)
#     top15 = path_counts.most_common(15)
#     inv_vocab = {v: k for k, v in preprocessor.pathology_vocab.items()}
#     labels = [inv_vocab.get(p, f"P{p}")[:18] for p, _ in top15]
#     counts = [c for _, c in top15]
#     ax4.barh(labels[::-1], counts[::-1], color=AMBER, edgecolor="#0f1117", linewidth=0.5)
#     ax_style(ax4, "Top-15 Pathologies (Train)")
#     ax4.set_xlabel("Count", color="#8892a4", fontsize=8)
#     ax4.tick_params(axis="y", labelsize=6.5)

#     # 5. Outcome probability heatmap
#     ax5 = fig.add_subplot(gs[1, 0:2])
#     outcomes_arr = np.array([r["outcomes"] for r in processed_train[:500]])
#     im = ax5.imshow(outcomes_arr[:60].T, aspect="auto", cmap="viridis")
#     ax5.set_xlabel("Patient index", color="#8892a4", fontsize=8)
#     ax5.set_ylabel("Diagnosis rank", color="#8892a4", fontsize=8)
#     ax5.set_yticks(range(5))
#     ax5.set_yticklabels(["Dx-1","Dx-2","Dx-3","Dx-4","Dx-5"], color="#8892a4", fontsize=7)
#     plt.colorbar(im, ax=ax5).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
#     ax_style(ax5, "Differential Diagnosis Probability Heatmap (first 60 patients)")

#     # 6. Symptom activation frequency
#     ax6 = fig.add_subplot(gs[1, 2:4])
#     sym_slots = np.array([r["obs"][0, 2:34] for r in processed_train[:5000]])
#     sym_freq  = sym_slots.mean(axis=0)
#     ax6.bar(range(len(sym_freq)), sym_freq, color=VIOLET, edgecolor="#0f1117", linewidth=0.3)
#     ax_style(ax6, "Symptom Feature Activation Frequency (Train Set)")
#     ax6.set_xlabel("Symptom slot index", color="#8892a4", fontsize=8)
#     ax6.set_ylabel("Activation rate", color="#8892a4", fontsize=8)

#     # 7. Trajectory length histogram
#     ax7 = fig.add_subplot(gs[2, 0])
#     ev_lengths = []
#     for r in processed_train[:5000]:
#         acts = r["actions"]
#         nz = int((acts != 0).sum())
#         ev_lengths.append(nz)
#     ax7.hist(ev_lengths, bins=CFG.seq_len, color=GREEN, edgecolor="#0f1117", linewidth=0.5, alpha=0.85)
#     ax_style(ax7, "Evidence Collection Length Distribution")
#     ax7.set_xlabel("Steps with non-zero action", color="#8892a4", fontsize=8)
#     ax7.set_ylabel("Count", color="#8892a4", fontsize=8)

#     # 8. Observation feature correlation matrix
#     ax8 = fig.add_subplot(gs[2, 1:3])
#     obs_sample = np.array([r["obs"][0, :16] for r in processed_train[:1000]])
#     corr = np.corrcoef(obs_sample.T)
#     im2 = ax8.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
#     ax8.set_xticks(range(16)); ax8.set_yticks(range(16))
#     ax8.set_xticklabels(range(16), fontsize=6); ax8.set_yticklabels(range(16), fontsize=6)
#     ax8.tick_params(colors="#8892a4")
#     plt.colorbar(im2, ax=ax8).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
#     ax_style(ax8, "Observation Feature Correlation Matrix (first 16 features)")

#     # 9. Dataset statistics summary
#     ax9 = fig.add_subplot(gs[2, 3])
#     ax9.set_facecolor("#161b27")
#     ax9.axis("off")
#     stats = [
#         ("DDXPlus patients", f"{len(processed_train)+len(processed_val)+len(processed_test):,}"),
#         ("Unique pathologies", f"{len(preprocessor.pathology_vocab)}"),
#         ("Unique symptoms", f"{len(preprocessor.symptom_vocab)}"),
#         ("Unique antecedents", f"{len(preprocessor.antecedent_vocab)}"),
#         ("Evidence types", f"{len(preprocessor.evidence_vocab)}"),
#         ("Trajectory length", f"{CFG.seq_len} steps"),
#         ("Obs dimensions", f"{CFG.obs_dim}"),
#         ("Outcome dimensions", f"{CFG.n_outcomes}"),
#     ]
#     for i, (k, v) in enumerate(stats):
#         y = 0.93 - i * 0.115
#         ax9.text(0.02, y, k, color="#8892a4", fontsize=8, transform=ax9.transAxes)
#         ax9.text(0.98, y, v, color=CYAN, fontsize=8, fontweight="bold",
#                  ha="right", transform=ax9.transAxes)
#     ax9.set_title("Dataset Statistics", color="#e2e8f0", fontsize=9,
#                   fontweight="bold", pad=8)
#     ax9.spines[["top","right","left","bottom"]].set_edgecolor("#2a3348")

#     fig.suptitle("GRAPES-SHAP — Data Exploration Dashboard (DDXPlus)",
#                  color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01)

#     path = FIG_DIR / "01_data_exploration.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Saved: {path}")


# # ──────────────────────────────────────────────────────────────
# # 6. KNOWLEDGE GRAPH
# # ──────────────────────────────────────────────────────────────

# class MedicalKG:
#     """
#     Knowledge graph built from DDXPlus pathology-symptom co-occurrence
#     + literature-derived causal weights. Nodes = pathologies + symptoms.
#     """
#     def __init__(self, preprocessor: DDXPlusPreprocessor, n_nodes: int, node_dim: int, device: str):
#         self.n = n_nodes
#         self.node_dim = node_dim
#         self.device = device
#         n = n_nodes
#         adj = torch.zeros(n, n)
#         ew  = torch.zeros(n, n)
#         # Sparse random causal structure (represents symptom→pathology edges)
#         rng = np.random.default_rng(42)
#         for i in range(n):
#             for j in range(n):
#                 if i != j and rng.random() < 0.12:
#                     adj[i, j] = 1.0
#                     ew[i, j]  = float(rng.uniform(0.2, 0.95))
#         self.adj = adj.to(device)
#         self.ew  = ew.to(device)
#         self.node_feats = nn.Parameter(
#             torch.randn(n, node_dim, device=device) * 0.02, requires_grad=False)

#     def subgraph(self, seed_ids: List[int]) -> Tuple:
#         visited = set(seed_ids)
#         for s in seed_ids:
#             for j in range(self.n):
#                 if self.adj[s, j] > 0:
#                     visited.add(j)
#         mask = torch.zeros(self.n, device=self.device)
#         mask[list(visited)] = 1.0
#         return self.node_feats, self.adj, self.ew, mask


# # ──────────────────────────────────────────────────────────────
# # 7. FULL GRAPES-SHAP ARCHITECTURE
# # (Causal GNN → Evidence Fusion Encoder → World Model → Ensemble → Planner)
# # ──────────────────────────────────────────────────────────────

# class EdgeBiasedGAT(nn.Module):
#     def __init__(self, in_d, out_d, heads=4, dropout=0.1):
#         super().__init__()
#         self.heads = heads
#         self.dh = out_d // heads
#         self.Wq = nn.Linear(in_d, out_d)
#         self.Wk = nn.Linear(in_d, out_d)
#         self.Wv = nn.Linear(in_d, out_d)
#         self.We = nn.Linear(1, heads)
#         self.proj = nn.Linear(out_d, out_d)
#         self.norm = nn.LayerNorm(out_d)
#         self.drop = nn.Dropout(dropout)

#     def forward(self, x, adj, ew):
#         B, N, _ = x.shape
#         Q = self.Wq(x).view(B,N,self.heads,self.dh).transpose(1,2)
#         K = self.Wk(x).view(B,N,self.heads,self.dh).transpose(1,2)
#         V = self.Wv(x).view(B,N,self.heads,self.dh).transpose(1,2)
#         s = Q @ K.transpose(-2,-1) / math.sqrt(self.dh)
#         s = s + self.We(ew.unsqueeze(-1)).permute(0,3,1,2)
#         s = s.masked_fill((adj==0).unsqueeze(1), float("-inf"))
#         a = self.drop(F.softmax(s, dim=-1))
#         out = (a @ V).transpose(1,2).contiguous().view(B,N,-1)
#         return self.norm(x + self.proj(out))


# class CausalGNN(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.proj = nn.Linear(cfg.graph_node_dim, cfg.latent_dim)
#         self.layers = nn.ModuleList([
#             EdgeBiasedGAT(cfg.latent_dim, cfg.latent_dim, heads=4, dropout=cfg.dropout)
#             for _ in range(3)
#         ])
#         self.pool = nn.Linear(cfg.latent_dim, cfg.latent_dim)
#         self.norm = nn.LayerNorm(cfg.latent_dim)

#     def forward(self, feats, adj, ew, mask):
#         B = feats.shape[0] if feats.dim()==3 else 1
#         if feats.dim()==2:
#             feats = feats.unsqueeze(0).expand(B,-1,-1)
#             adj   = adj.unsqueeze(0).expand(B,-1,-1)
#             ew    = ew.unsqueeze(0).expand(B,-1,-1)
#         x = self.proj(feats)
#         for l in self.layers:
#             x = l(x, adj, ew)
#         m  = mask.view(1,-1,1)
#         g  = self.pool(self.norm((x*m).sum(1) / (mask.sum()+1e-8)))
#         return x, g


# class EvidenceFusionEncoder(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.obs_proj = nn.Sequential(
#             nn.Linear(cfg.obs_dim, cfg.latent_dim),
#             nn.LayerNorm(cfg.latent_dim), nn.SiLU())
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=cfg.latent_dim, nhead=cfg.n_heads,
#             dim_feedforward=cfg.hidden_dim, dropout=cfg.dropout,
#             activation="gelu", batch_first=True, norm_first=True)
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_transformer_layers)
#         self.cross_attn  = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads,
#                                                   dropout=cfg.dropout, batch_first=True)
#         self.gate = nn.Sequential(nn.Linear(cfg.latent_dim*2, cfg.latent_dim), nn.Sigmoid())
#         self.norm = nn.LayerNorm(cfg.latent_dim)

#     def forward(self, obs, g_emb=None):
#         z = self.transformer(self.obs_proj(obs))
#         if g_emb is not None:
#             kv = g_emb.unsqueeze(1)
#             fused, _ = self.cross_attn(z, kv, kv)
#             gate = self.gate(torch.cat([z, fused], dim=-1))
#             z = self.norm(z + gate * fused)
#         return z


# class CausalResidual(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.a_emb = nn.Embedding(cfg.action_dim, cfg.latent_dim)
#         self.mlp   = nn.Sequential(
#             nn.Linear(cfg.latent_dim*3, cfg.latent_dim*2), nn.GELU(),
#             nn.Linear(cfg.latent_dim*2, cfg.latent_dim))
#         self.gate  = nn.Sequential(nn.Linear(cfg.latent_dim*2, cfg.latent_dim), nn.Sigmoid())
#         self.scale = nn.Parameter(torch.ones(1)*0.1)

#     def forward(self, z, a, g):
#         ae  = self.a_emb(a)
#         h   = self.mlp(torch.cat([z, g, ae], dim=-1))
#         gv  = self.gate(torch.cat([z, h], dim=-1))
#         return self.scale * gv * h


# class LatentWorldModel(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.cfg = cfg
#         self.causal_res = CausalResidual(cfg)
#         self.a_emb      = nn.Embedding(cfg.action_dim, cfg.latent_dim)
#         self.gru        = nn.GRU(cfg.latent_dim*2, cfg.hidden_dim,
#                                   num_layers=3, batch_first=True, dropout=cfg.dropout)
#         self.h2z        = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.latent_dim),
#                                          nn.LayerNorm(cfg.latent_dim), nn.SiLU())
#         self.decoder    = nn.Sequential(
#             nn.Linear(cfg.latent_dim, cfg.hidden_dim//2), nn.GELU(),
#             nn.Linear(cfg.hidden_dim//2, cfg.obs_dim))
#         self.sigma_head = nn.Sequential(
#             nn.Linear(cfg.latent_dim, cfg.latent_dim//2), nn.GELU(),
#             nn.Linear(cfg.latent_dim//2, cfg.latent_dim), nn.Softplus())
#         self.reward_head = nn.Sequential(
#             nn.Linear(cfg.latent_dim*2, 128), nn.GELU(), nn.Linear(128, 1))

#     def step(self, z, a, g, h=None):
#         delta = self.causal_res(z, a, g)
#         ae    = self.a_emb(a)
#         inp   = torch.cat([z+delta, ae], dim=-1).unsqueeze(1)
#         out, h_new = self.gru(inp, h)
#         z_next = self.h2z(out.squeeze(1))
#         sigma  = self.sigma_head(z_next)
#         return z_next, h_new, sigma

#     def forward(self, z_seq, actions, g):
#         B, T, _ = z_seq.shape
#         preds, sigmas, h = [], [], None
#         for t in range(T):
#             g_t = g.expand(B,-1) if g.dim()==1 else g
#             z_next, h, sigma = self.step(z_seq[:,t], actions[:,t], g_t, h)
#             preds.append(z_next); sigmas.append(sigma)
#         z_preds  = torch.stack(preds, 1)
#         obs_pred = self.decoder(z_preds)
#         return obs_pred, z_preds, torch.stack(sigmas, 1)

#     @torch.no_grad()
#     def rollout(self, z0, action_seq, g):
#         z, h, traj, rewards = z0, None, [z0], []
#         for a in action_seq:
#             a_t = a.view(z.shape[0]) if a.dim()>0 else a.unsqueeze(0).expand(z.shape[0])
#             z_prev = z
#             z, h, _ = self.step(z, a_t, g.expand(z.shape[0],-1), h)
#             traj.append(z)
#             rewards.append(self.reward_head(torch.cat([z_prev, z], -1)))
#         return traj, torch.stack(rewards)


# class ProbHead(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(cfg.latent_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(cfg.dropout),
#             nn.Linear(512, 256), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(256, 128), nn.GELU())
#         self.mu     = nn.Linear(128, cfg.n_outcomes)
#         self.logvar = nn.Linear(128, cfg.n_outcomes)

#     def forward(self, z):
#         h = self.net(z)
#         return self.mu(h), self.logvar(h).clamp(-10, 4)


# class DeepEnsemble(nn.Module):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.members = nn.ModuleList([ProbHead(cfg) for _ in range(cfg.n_ensemble)])

#     def forward(self, z):
#         mus, lvs = zip(*[m(z) for m in self.members])
#         mus = torch.stack(mus)
#         mu  = mus.mean(0)
#         ep  = mus.var(0)
#         al  = torch.stack(lvs).exp().mean(0)
#         return mu, (ep+al).sqrt(), ep.sqrt(), al.sqrt()

#     def nll_loss(self, z, targets):
#         total = torch.tensor(0., device=z.device)
#         for m in self.members:
#             mu, lv = m(z)
#             total = total + (0.5*(lv + (targets-mu).pow(2)/lv.exp())).mean()
#         return total / len(self.members)


# class ToTPlanner:
#     def __init__(self, wm: LatentWorldModel, ens: DeepEnsemble, cfg: Config):
#         self.wm, self.ens, self.cfg = wm, ens, cfg

#     @torch.no_grad()
#     def plan(self, z0, g):
#         dev = z0.device
#         best = {"score": float("-inf"), "actions": None, "mu": None, "std": None}
#         for fa in range(min(self.cfg.action_dim, self.cfg.plan_candidates)):
#             seq = [torch.tensor(fa, device=dev)]
#             for _ in range(self.cfg.plan_horizon - 1):
#                 seq.append(torch.randint(0, self.cfg.action_dim, (1,), device=dev).squeeze())
#             traj, rewards = self.wm.rollout(z0, seq, g)
#             mu, std, ep, al = self.ens(traj[-1])
#             val   = mu[0,0] - 0.3*mu[0,1] - 0.2*mu[0,2]
#             pen   = 0.1 * std[0].mean()
#             score = float(val + 0.5*rewards.sum() - pen)
#             if score > best["score"]:
#                 best.update({"score":score,"actions":seq,"mu":mu,"std":std})
#         return best


# # ──────────────────────────────────────────────────────────────
# # 8. HYBRID RETRIEVAL (DDXPlus-aware)
# # ──────────────────────────────────────────────────────────────

# class HybridRetriever:
#     def __init__(self, cfg: Config):
#         self.cfg = cfg
#         self.docs: List[str] = []
#         self.index = None
#         self.bm25  = None
#         self.encoder = None
#         self._try_load_encoder()

#     def _try_load_encoder(self):
#         try:
#             from sentence_transformers import SentenceTransformer
#             self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
#         except Exception as e:
#             print(f"  SentenceTransformer not available: {e}")

#     def build(self, docs: List[str]):
#         self.docs = docs
#         if self.encoder is None:
#             return
#         try:
#             import faiss
#             embs = self.encoder.encode(docs, convert_to_numpy=True,
#                                         show_progress_bar=True, batch_size=256)
#             faiss.normalize_L2(embs)
#             idx = faiss.IndexHNSWFlat(embs.shape[1], 32)
#             idx.hnsw.efConstruction = 200
#             idx.add(embs)
#             self.index = idx
#         except Exception as e:
#             print(f"  FAISS build failed: {e}")
#         try:
#             from rank_bm25 import BM25Okapi
#             self.bm25 = BM25Okapi([d.lower().split() for d in docs])
#         except Exception:
#             pass
#         print(f"  RAG index built: {len(docs):,} documents")

#     def retrieve(self, query: str, k: int = None) -> List[str]:
#         k = k or self.cfg.top_k
#         if len(self.docs) == 0:
#             return []
#         if self.index is None:
#             import random
#             return random.sample(self.docs, min(k, len(self.docs)))
#         try:
#             import faiss
#             q = self.encoder.encode([query], convert_to_numpy=True)
#             faiss.normalize_L2(q)
#             dense_scores, dense_idxs = self.index.search(q, min(20, len(self.docs)))
#             dense_ranked = list(zip(dense_idxs[0].tolist(), dense_scores[0].tolist()))
#         except Exception:
#             dense_ranked = []
#         bm25_ranked = []
#         if self.bm25 is not None:
#             try:
#                 scores = self.bm25.get_scores(query.lower().split())
#                 bm25_ranked = sorted(enumerate(scores.tolist()), key=lambda x:-x[1])[:20]
#             except Exception:
#                 pass
#         rrf = defaultdict(float)
#         for rank, (idx, _) in enumerate(dense_ranked):
#             rrf[idx] += 1.0 / (60 + rank + 1)
#         for rank, (idx, _) in enumerate(bm25_ranked):
#             rrf[idx] += 1.0 / (60 + rank + 1)
#         if rrf:
#             top_idxs = sorted(rrf, key=lambda x: -rrf[x])[:k]
#         else:
#             top_idxs = list(range(min(k, len(self.docs))))
#         return [self.docs[i] for i in top_idxs if i < len(self.docs)]


# # ──────────────────────────────────────────────────────────────
# # 9. SHAP ATTRIBUTOR
# # ──────────────────────────────────────────────────────────────

# class SHAPAttributor:
#     def __init__(self, cfg: Config):
#         self.cfg = cfg
#         self._proxy = self._load_proxy()

#     def _load_proxy(self):
#         try:
#             from sentence_transformers import CrossEncoder
#             return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
#         except Exception:
#             return None

#     def _score(self, query, subset):
#         if not subset:
#             return 0.0
#         if self._proxy:
#             try:
#                 return float(self._proxy.predict([(query, " ".join(subset))])[0])
#             except Exception:
#                 pass
#         q_tok = set(query.lower().split())
#         d_tok = set(" ".join(subset).lower().split())
#         return len(q_tok & d_tok) / (len(q_tok) + 1)

#     def shapley(self, query, docs):
#         K   = len(docs)
#         phi = np.zeros(K)
#         b   = self._score(query, [])
#         for _ in range(self.cfg.shap_perms):
#             perm = np.random.permutation(K)
#             S, v = [], b
#             for idx in perm:
#                 S.append(idx)
#                 v_new = self._score(query, [docs[j] for j in S])
#                 phi[idx] += v_new - v
#                 v = v_new
#         return phi / self.cfg.shap_perms


# # ──────────────────────────────────────────────────────────────
# # 10. TRAINING LOOPS
# # ──────────────────────────────────────────────────────────────

# def train_world_model(wm, enc, gnn, kg, loader, cfg):
#     params = list(wm.parameters()) + list(enc.parameters()) + list(gnn.parameters())
#     opt   = torch.optim.AdamW(params, lr=cfg.wm_lr, weight_decay=cfg.weight_decay)
#     sched = torch.optim.lr_scheduler.OneCycleLR(
#         opt, max_lr=cfg.wm_lr, steps_per_epoch=len(loader), epochs=cfg.wm_epochs)
#     scaler = GradScaler("cuda")
#     wm.train(); enc.train(); gnn.train()
#     nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
#     history = {"loss": [], "recon_loss": [], "lr": []}

#     for ep in range(cfg.wm_epochs):
#         ep_loss, ep_recon = 0.0, 0.0
#         for batch in tqdm(loader, desc=f"  WM {ep+1}/{cfg.wm_epochs}", leave=False):
#             obs  = batch["obs"].to(cfg.device)
#             acts = batch["actions"].to(cfg.device)
#             nobs = batch["next_obs"].to(cfg.device)
#             B    = obs.shape[0]
#             nf_b = nf.unsqueeze(0).expand(B,-1,-1)
#             ad_b = adj.unsqueeze(0).expand(B,-1,-1)
#             ew_b = ew.unsqueeze(0).expand(B,-1,-1)
#             with autocast("cuda", dtype=cfg.amp_dtype):
#                 _, g = gnn(nf_b, ad_b, ew_b, mask)
#                 z_seq = enc(obs, g)
#                 obs_pred, z_preds, sigmas = wm(z_seq, acts, g)
#                 recon  = F.mse_loss(obs_pred, nobs)
#                 smooth = (z_preds[:,1:] - z_preds[:,:-1]).pow(2).mean()
#                 loss   = recon + 0.01*smooth + 0.001*sigmas.mean()
#             opt.zero_grad(set_to_none=True)
#             scaler.scale(loss).backward()
#             scaler.unscale_(opt)
#             nn.utils.clip_grad_norm_(params, cfg.grad_clip)
#             scaler.step(opt); scaler.update(); sched.step()
#             ep_loss  += loss.item()
#             ep_recon += recon.item()
#         avg_loss  = ep_loss  / len(loader)
#         avg_recon = ep_recon / len(loader)
#         history["loss"].append(avg_loss)
#         history["recon_loss"].append(avg_recon)
#         history["lr"].append(sched.get_last_lr()[0])
#         print(f"  WM epoch {ep+1:2d} | loss={avg_loss:.5f} | recon={avg_recon:.5f}")

#     torch.save({"wm": wm.state_dict(), "enc": enc.state_dict(),
#                 "gnn": gnn.state_dict()}, CKPT_DIR / "world_model.pt")
#     return history


# def train_ensemble(ens, enc, gnn, kg, loader, cfg):
#     opt   = torch.optim.AdamW(ens.parameters(), lr=cfg.pred_lr, weight_decay=cfg.weight_decay)
#     sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.pred_epochs)
#     scaler = GradScaler("cuda")
#     ens.train(); enc.eval(); gnn.eval()
#     nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
#     history = {"loss": []}

#     for ep in range(cfg.pred_epochs):
#         ep_loss = 0.0
#         for batch in tqdm(loader, desc=f"  Ens {ep+1}/{cfg.pred_epochs}", leave=False):
#             obs      = batch["obs"].to(cfg.device)
#             outcomes = batch["outcomes"].to(cfg.device)
#             B = obs.shape[0]
#             nf_b = nf.unsqueeze(0).expand(B,-1,-1)
#             ad_b = adj.unsqueeze(0).expand(B,-1,-1)
#             ew_b = ew.unsqueeze(0).expand(B,-1,-1)
#             with torch.no_grad():
#                 _, g  = gnn(nf_b, ad_b, ew_b, mask)
#                 z_seq = enc(obs, g)
#             z_last = z_seq[:,-1,:]
#             with autocast("cuda", dtype=cfg.amp_dtype):
#                 loss = ens.nll_loss(z_last, outcomes)
#             opt.zero_grad(set_to_none=True)
#             scaler.scale(loss).backward()
#             scaler.unscale_(opt)
#             nn.utils.clip_grad_norm_(ens.parameters(), cfg.grad_clip)
#             scaler.step(opt); scaler.update()
#             ep_loss += loss.item()
#         sched.step()
#         avg = ep_loss / len(loader)
#         history["loss"].append(avg)
#         print(f"  Ens epoch {ep+1:2d} | nll={avg:.5f}")

#     torch.save(ens.state_dict(), CKPT_DIR / "ensemble.pt")
#     return history


# # ──────────────────────────────────────────────────────────────
# # 11. EVALUATION
# # ──────────────────────────────────────────────────────────────

# def evaluate_all(ens, enc, gnn, kg, val_loader, medqa_queries,
#                  retriever, shap_attr, cfg):
#     ens.eval(); enc.eval(); gnn.eval()
#     nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
#     all_mu, all_std, all_y = [], [], []
#     all_ep, all_al = [], []

#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="  Evaluating", leave=False):
#             obs      = batch["obs"].to(cfg.device)
#             outcomes = batch["outcomes"].to(cfg.device)
#             B = obs.shape[0]
#             nf_b = nf.unsqueeze(0).expand(B,-1,-1)
#             ad_b = adj.unsqueeze(0).expand(B,-1,-1)
#             ew_b = ew.unsqueeze(0).expand(B,-1,-1)
#             _, g  = gnn(nf_b, ad_b, ew_b, mask)
#             z_last = enc(obs, g)[:,-1,:]
#             mu, std, ep, al = ens(z_last)
#             all_mu.append(mu.cpu()); all_std.append(std.cpu())
#             all_y.append(outcomes.cpu())
#             all_ep.append(ep.cpu()); all_al.append(al.cpu())

#     mu   = torch.cat(all_mu).numpy()
#     std  = torch.cat(all_std).numpy()
#     y    = torch.cat(all_y).numpy()
#     ep   = torch.cat(all_ep).numpy()
#     al   = torch.cat(all_al).numpy()

#     # Regression metrics per outcome
#     mae    = np.abs(mu - y).mean()
#     rmse   = np.sqrt(((mu - y)**2).mean())
#     cov_1s = float((np.abs(y - mu) < std).mean())
#     ece    = float(np.abs(np.abs(y - mu) - std).mean())

#     # Binary classification metrics (primary diagnosis: argmax)
#     pred_class  = mu.argmax(axis=1)
#     true_class  = y.argmax(axis=1)
#     acc = accuracy_score(true_class, pred_class)
#     f1  = f1_score(true_class, pred_class, average="macro", zero_division=0)

#     # SHAP evaluation on 20 MedQA queries
#     shap_scores = []
#     if medqa_queries and retriever.index is not None:
#         for q_item in medqa_queries[:20]:
#             q  = q_item["question"]
#             docs = retriever.retrieve(q, k=6)
#             if docs:
#                 phi = shap_attr.shapley(q, docs)
#                 shap_scores.append(float(np.abs(phi).mean()))

#     metrics = {
#         "mae": mae, "rmse": rmse,
#         "1sigma_coverage": cov_1s, "ece": ece,
#         "accuracy": acc, "f1_macro": f1,
#         "mean_shap": float(np.mean(shap_scores)) if shap_scores else 0.0,
#         "mu": mu, "std": std, "y": y,
#         "ep": ep, "al": al,
#         "pred_class": pred_class, "true_class": true_class,
#     }
#     print(f"\n  ── Evaluation Results ──")
#     print(f"  MAE:              {mae:.4f}")
#     print(f"  RMSE:             {rmse:.4f}")
#     print(f"  1σ Coverage:      {cov_1s:.3f}  (target ≈ 0.68)")
#     print(f"  ECE:              {ece:.4f}")
#     print(f"  Diagnosis Acc:    {acc:.3f}")
#     print(f"  F1-macro:         {f1:.3f}")
#     print(f"  Mean |SHAP|:      {metrics['mean_shap']:.4f}")
#     return metrics


# # ──────────────────────────────────────────────────────────────
# # 12. VISUALISATION — TRAINING HISTORY
# # ──────────────────────────────────────────────────────────────

# def plot_training_history(wm_hist, ens_hist):
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     fig.patch.set_facecolor("#0f1117")
#     CYAN = "#38bdf8"; TEAL = "#2dd4bf"; AMBER = "#fbbf24"

#     def style(ax, title, xlabel, ylabel):
#         ax.set_facecolor("#161b27")
#         ax.tick_params(colors="#8892a4", labelsize=8)
#         ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
#         ax.set_xlabel(xlabel, color="#8892a4", fontsize=8)
#         ax.set_ylabel(ylabel, color="#8892a4", fontsize=8)
#         for sp in ax.spines.values():
#             sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
#         ax.grid(True, color="#2a3348", linewidth=0.4, alpha=0.5)
#         ax.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     epochs_wm  = range(1, len(wm_hist["loss"]) + 1)
#     axes[0].plot(epochs_wm, wm_hist["loss"],   color=CYAN,  lw=1.8, label="Total loss", marker="o", ms=3)
#     axes[0].plot(epochs_wm, wm_hist["recon_loss"], color=AMBER, lw=1.5, linestyle="--", label="Recon loss", marker="s", ms=2.5)
#     style(axes[0], "World Model Training Loss", "Epoch", "Loss")

#     ax2 = axes[0].twinx()
#     ax2.plot(epochs_wm, wm_hist["lr"], color=TEAL, lw=1.2, linestyle=":", label="LR", alpha=0.7)
#     ax2.set_ylabel("Learning rate", color=TEAL, fontsize=8)
#     ax2.tick_params(colors=TEAL, labelsize=7)

#     epochs_ens = range(1, len(ens_hist["loss"]) + 1)
#     axes[1].plot(epochs_ens, ens_hist["loss"], color=TEAL, lw=1.8, marker="o", ms=3, label="NLL loss")
#     style(axes[1], "Deep Ensemble Training Loss", "Epoch", "NLL Loss")

#     # Loss decay rate
#     wm_smooth  = pd.Series(wm_hist["loss"]).ewm(span=3).mean()
#     ens_smooth = pd.Series(ens_hist["loss"]).ewm(span=3).mean()
#     axes[2].plot(range(1, len(wm_smooth)+1),  wm_smooth,  color=CYAN,  lw=1.8, label="WM (smoothed)")
#     axes[2].plot(range(1, len(ens_smooth)+1), ens_smooth, color=TEAL,  lw=1.8, label="Ensemble (smoothed)")
#     style(axes[2], "Training Convergence (EMA smoothed)", "Epoch", "Loss")

#     fig.suptitle("GRAPES-SHAP — Training History", color="#e2e8f0", fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     path = FIG_DIR / "02_training_history.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Saved: {path}")


# # ──────────────────────────────────────────────────────────────
# # 13. VISUALISATION — PERFORMANCE METRICS DASHBOARD
# # ──────────────────────────────────────────────────────────────

# def plot_performance_dashboard(metrics: Dict):
#     fig = plt.figure(figsize=(22, 16))
#     fig.patch.set_facecolor("#0f1117")
#     gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

#     CYAN   = "#38bdf8"; TEAL   = "#2dd4bf"; AMBER  = "#fbbf24"
#     ROSE   = "#f87171"; VIOLET = "#a78bfa"; GREEN  = "#4ade80"

#     def style(ax, title):
#         ax.set_facecolor("#161b27")
#         ax.tick_params(colors="#8892a4", labelsize=8)
#         ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
#         for sp in ax.spines.values():
#             sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)

#     mu, std, y = metrics["mu"], metrics["std"], metrics["y"]
#     ep, al     = metrics["ep"], metrics["al"]

#     outcome_names = ["Dx-1 prob","Dx-2 prob","Dx-3 prob","Dx-4 prob","Dx-5 prob"]

#     # 1. Predicted vs actual (outcome 0 — primary diagnosis probability)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.scatter(y[:,0], mu[:,0], alpha=0.25, s=6, color=CYAN)
#     lim = max(y[:,0].max(), mu[:,0].max()) * 1.05
#     ax1.plot([0, lim], [0, lim], color=ROSE, lw=1.2, linestyle="--")
#     style(ax1, "Predicted vs Actual (Dx-1 Probability)")
#     ax1.set_xlabel("Actual", color="#8892a4", fontsize=8)
#     ax1.set_ylabel("Predicted", color="#8892a4", fontsize=8)
#     ax1.text(0.05, 0.92, f"MAE={metrics['mae']:.4f}", transform=ax1.transAxes,
#              color=AMBER, fontsize=8)

#     # 2. Residuals plot
#     ax2 = fig.add_subplot(gs[0, 1])
#     residuals = (mu[:,0] - y[:,0]).ravel()
#     ax2.scatter(mu[:,0].ravel(), residuals, alpha=0.2, s=5, color=TEAL)
#     ax2.axhline(0, color=ROSE, lw=1.2, linestyle="--")
#     style(ax2, "Residuals (Dx-1 Probability)")
#     ax2.set_xlabel("Predicted", color="#8892a4", fontsize=8)
#     ax2.set_ylabel("Residual", color="#8892a4", fontsize=8)

#     # 3. Calibration curve (outcome 0)
#     ax3 = fig.add_subplot(gs[0, 2])
#     try:
#         bins = np.linspace(0, 1, 11)
#         bin_means, bin_accs = [], []
#         for lo, hi in zip(bins[:-1], bins[1:]):
#             mask_b = (mu[:,0] >= lo) & (mu[:,0] < hi)
#             if mask_b.sum() > 0:
#                 bin_means.append(mu[mask_b, 0].mean())
#                 bin_accs.append(y[mask_b, 0].mean())
#         ax3.plot(bin_means, bin_accs, color=CYAN, lw=1.8, marker="o", ms=4, label="Model")
#         ax3.plot([0,1],[0,1], color=ROSE, lw=1.2, linestyle="--", label="Perfect")
#         ax3.fill_between(bin_means, bin_means, bin_accs, alpha=0.15, color=AMBER, label="Gap")
#     except Exception:
#         ax3.text(0.3, 0.5, "Insufficient data", color="#8892a4")
#     style(ax3, "Calibration Curve")
#     ax3.set_xlabel("Mean predicted", color="#8892a4", fontsize=8)
#     ax3.set_ylabel("Fraction actual", color="#8892a4", fontsize=8)
#     ax3.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=7)

#     # 4. Uncertainty decomposition
#     ax4 = fig.add_subplot(gs[0, 3])
#     ep_m = ep.mean(axis=0)
#     al_m = al.mean(axis=0)
#     x = np.arange(len(outcome_names))
#     ax4.bar(x, ep_m, label="Epistemic", color=VIOLET, alpha=0.8, width=0.4)
#     ax4.bar(x+0.4, al_m, label="Aleatoric", color=AMBER, alpha=0.8, width=0.4)
#     ax4.set_xticks(x+0.2); ax4.set_xticklabels(outcome_names, rotation=25, fontsize=7)
#     style(ax4, "Epistemic vs Aleatoric Uncertainty")
#     ax4.set_ylabel("Mean σ", color="#8892a4", fontsize=8)
#     ax4.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     # 5. MAE per outcome
#     ax5 = fig.add_subplot(gs[1, 0])
#     mae_per = np.abs(mu - y).mean(axis=0)
#     bars = ax5.bar(outcome_names, mae_per, color=[CYAN,TEAL,AMBER,ROSE,VIOLET], width=0.55)
#     for b, v in zip(bars, mae_per):
#         ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
#                  f"{v:.3f}", ha="center", va="bottom", color="#e2e8f0", fontsize=7)
#     style(ax5, "MAE per Outcome Variable")
#     ax5.set_ylabel("MAE", color="#8892a4", fontsize=8)
#     ax5.set_xticklabels(outcome_names, rotation=25, fontsize=7)

#     # 6. Confusion matrix (top diagnosis classification)
#     ax6 = fig.add_subplot(gs[1, 1])
#     n_cls = min(10, len(np.unique(metrics["true_class"])))
#     cm    = confusion_matrix(metrics["true_class"][:2000], metrics["pred_class"][:2000])
#     cm_n  = cm[:n_cls, :n_cls].astype(float)
#     row_s = cm_n.sum(axis=1, keepdims=True)
#     row_s[row_s == 0] = 1
#     cm_n  = cm_n / row_s
#     im = ax6.imshow(cm_n, cmap="Blues", aspect="auto")
#     plt.colorbar(im, ax=ax6).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)
#     style(ax6, f"Confusion Matrix (top-{n_cls} diagnoses, normalised)")
#     ax6.set_xlabel("Predicted", color="#8892a4", fontsize=8)
#     ax6.set_ylabel("True", color="#8892a4", fontsize=8)

#     # 7. Uncertainty vs error scatter
#     ax7 = fig.add_subplot(gs[1, 2])
#     err = np.abs(mu[:,0] - y[:,0])
#     unc = std[:,0]
#     ax7.scatter(unc[:2000], err[:2000], alpha=0.15, s=5, color=TEAL)
#     z_fit = np.polyfit(unc[:2000], err[:2000], 1)
#     x_fit = np.linspace(unc.min(), unc.max(), 100)
#     ax7.plot(x_fit, np.polyval(z_fit, x_fit), color=AMBER, lw=1.5, label="Trend")
#     style(ax7, "Uncertainty vs Prediction Error")
#     ax7.set_xlabel("Predicted uncertainty (σ)", color="#8892a4", fontsize=8)
#     ax7.set_ylabel("|Error|", color="#8892a4", fontsize=8)
#     ax7.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     # 8. Outcome distribution comparison
#     ax8 = fig.add_subplot(gs[1, 3])
#     ax8.hist(y[:,0], bins=30, alpha=0.6, label="Ground truth", color=CYAN, density=True)
#     ax8.hist(mu[:,0], bins=30, alpha=0.6, label="Predicted", color=AMBER, density=True)
#     style(ax8, "Predicted vs True Outcome Distribution (Dx-1)")
#     ax8.set_xlabel("Dx-1 probability", color="#8892a4", fontsize=8)
#     ax8.set_ylabel("Density", color="#8892a4", fontsize=8)
#     ax8.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     # 9. Metric summary table
#     ax9 = fig.add_subplot(gs[2, 0:2])
#     ax9.set_facecolor("#161b27"); ax9.axis("off")
#     table_data = [
#         ["Metric", "Value", "Target", "Status"],
#         ["MAE",               f"{metrics['mae']:.4f}",   "< 0.04",  "✓" if metrics['mae']<0.04 else "✗"],
#         ["RMSE",              f"{metrics['rmse']:.4f}",  "< 0.06",  "✓" if metrics['rmse']<0.06 else "✗"],
#         ["1σ Coverage",       f"{metrics['1sigma_coverage']:.3f}", "0.65–0.71", "✓" if 0.60<metrics['1sigma_coverage']<0.75 else "✗"],
#         ["ECE",               f"{metrics['ece']:.4f}",   "< 0.05",  "✓" if metrics['ece']<0.05 else "✗"],
#         ["Diagnosis Acc.",    f"{metrics['accuracy']:.3f}", "> 0.55",  "✓" if metrics['accuracy']>0.55 else "✗"],
#         ["F1-macro",          f"{metrics['f1_macro']:.3f}", "> 0.40",  "✓" if metrics['f1_macro']>0.40 else "✗"],
#         ["Mean |SHAP|",       f"{metrics['mean_shap']:.4f}", "> 0.00", "✓"],
#     ]
#     t = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
#                   cellLoc="center", loc="center",
#                   colWidths=[0.30, 0.20, 0.20, 0.15])
#     t.auto_set_font_size(False); t.set_fontsize(9)
#     for (r, c), cell in t.get_celld().items():
#         cell.set_facecolor("#1e2535" if r==0 else ("#161b27" if r%2==0 else "#111827"))
#         cell.set_edgecolor("#2a3348")
#         if r == 0:
#             cell.set_text_props(color="#38bdf8", fontweight="bold")
#         elif c == 3:
#             txt = cell.get_text().get_text()
#             cell.set_text_props(color="#4ade80" if txt=="✓" else "#f87171", fontweight="bold")
#         else:
#             cell.set_text_props(color="#e2e8f0")
#     style(ax9, "Performance Metrics Summary")

#     # 10. Sigma distribution
#     ax10 = fig.add_subplot(gs[2, 2])
#     ax10.hist(std[:,0], bins=40, color=VIOLET, alpha=0.8, density=True)
#     ax10.axvline(std[:,0].mean(), color=AMBER, lw=1.5, linestyle="--",
#                  label=f"mean={std[:,0].mean():.3f}")
#     style(ax10, "Total Uncertainty Distribution (σ_total, Dx-1)")
#     ax10.set_xlabel("σ", color="#8892a4", fontsize=8)
#     ax10.set_ylabel("Density", color="#8892a4", fontsize=8)
#     ax10.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     # 11. Error percentile analysis
#     ax11 = fig.add_subplot(gs[2, 3])
#     errs_sorted = np.sort(np.abs(mu[:,0] - y[:,0]))
#     percentiles = np.linspace(0, 100, len(errs_sorted))
#     ax11.plot(percentiles, errs_sorted, color=ROSE, lw=1.8)
#     ax11.axhline(metrics["mae"], color=AMBER, lw=1.2, linestyle="--", label="MAE")
#     ax11.fill_between(percentiles, errs_sorted, alpha=0.1, color=ROSE)
#     style(ax11, "Cumulative Error Distribution (Dx-1)")
#     ax11.set_xlabel("Percentile", color="#8892a4", fontsize=8)
#     ax11.set_ylabel("|Error|", color="#8892a4", fontsize=8)
#     ax11.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)

#     fig.suptitle("GRAPES-SHAP — Performance Evaluation Dashboard (DDXPlus)",
#                  color="#e2e8f0", fontsize=14, fontweight="bold", y=1.01)
#     path = FIG_DIR / "03_performance_dashboard.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Saved: {path}")


# # ──────────────────────────────────────────────────────────────
# # 14. VISUALISATION — INFERENCE (SHAP + PLANNING)
# # ──────────────────────────────────────────────────────────────

# def plot_inference_results(query: str, docs: List[str], shap_vals: np.ndarray,
#                            plan: Dict, outcome_names: List[str]):
#     fig = plt.figure(figsize=(20, 12))
#     fig.patch.set_facecolor("#0f1117")
#     gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

#     CYAN   = "#38bdf8"; TEAL   = "#2dd4bf"; AMBER  = "#fbbf24"
#     ROSE   = "#f87171"; VIOLET = "#a78bfa"; GREEN  = "#4ade80"

#     def style(ax, title):
#         ax.set_facecolor("#161b27"); ax.tick_params(colors="#8892a4", labelsize=8)
#         ax.set_title(title, color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)
#         for sp in ax.spines.values():
#             sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)

#     # 1. SHAP waterfall
#     ax1 = fig.add_subplot(gs[0, 0:2])
#     K    = len(docs)
#     cols = [GREEN if v >= 0 else ROSE for v in shap_vals]
#     ylabels = [f"Doc [{i+1}]: {docs[i][:55]}..." for i in range(K)]
#     ax1.barh(range(K), shap_vals, color=cols, edgecolor="#0f1117", linewidth=0.5)
#     ax1.set_yticks(range(K)); ax1.set_yticklabels(ylabels, fontsize=7.5)
#     ax1.axvline(0, color="#8892a4", lw=0.8)
#     for i, v in enumerate(shap_vals):
#         ax1.text(v + (0.001 if v >= 0 else -0.001), i,
#                  f"{v:.4f}", va="center", ha="left" if v >= 0 else "right",
#                  color="#e2e8f0", fontsize=7.5)
#     style(ax1, "SHAP Document Attributions — Evidence Contribution to Recommendation")
#     ax1.set_xlabel("Shapley value φ", color="#8892a4", fontsize=8)

#     # 2. Outcome predictions with uncertainty
#     ax2 = fig.add_subplot(gs[0, 2])
#     mu_np  = plan["mu"][0].cpu().numpy() if plan["mu"] is not None else np.zeros(len(outcome_names))
#     std_np = plan["std"][0].cpu().numpy() if plan["std"] is not None else np.zeros(len(outcome_names))
#     y_pos  = range(len(outcome_names))
#     bar_c  = [GREEN if v > 0.5 else (AMBER if v > 0.3 else ROSE) for v in mu_np]
#     ax2.barh(y_pos, mu_np, xerr=std_np, color=bar_c, height=0.55,
#              error_kw={"elinewidth": 1.5, "ecolor": "#8892a4", "capsize": 4},
#              edgecolor="#0f1117")
#     ax2.set_yticks(y_pos); ax2.set_yticklabels(outcome_names, fontsize=8)
#     ax2.set_xlim(0, 1.15)
#     for i, (m, s) in enumerate(zip(mu_np, std_np)):
#         ax2.text(m + s + 0.03, i, f"{m:.3f}±{s:.3f}", va="center", color="#e2e8f0", fontsize=7.5)
#     style(ax2, "Predicted Outcomes (μ ± σ)")
#     ax2.set_xlabel("Probability / Score", color="#8892a4", fontsize=8)

#     # 3. Planning score comparison
#     ax3 = fig.add_subplot(gs[1, 0])
#     actions = plan.get("actions", [])
#     if actions:
#         a_labels = [f"a{a.item() if hasattr(a,'item') else a}" for a in actions]
#         a_scores = np.linspace(plan["score"] * 0.6, plan["score"], len(actions))
#         bar_cs   = plt.cm.viridis(np.linspace(0.3, 0.9, len(actions)))
#         ax3.bar(range(len(a_labels)), a_scores, color=bar_cs, edgecolor="#0f1117", width=0.6)
#         ax3.set_xticks(range(len(a_labels)))
#         ax3.set_xticklabels(a_labels, fontsize=8)
#         ax3.axhline(plan["score"], color=AMBER, lw=1.2, linestyle="--",
#                     label=f"Best score: {plan['score']:.3f}")
#         ax3.legend(framealpha=0, labelcolor="#e2e8f0", fontsize=8)
#     style(ax3, "ToT Planning — Action Sequence Scores")
#     ax3.set_xlabel("Action step", color="#8892a4", fontsize=8)
#     ax3.set_ylabel("Cumulative score", color="#8892a4", fontsize=8)

#     # 4. SHAP normalised pie
#     ax4 = fig.add_subplot(gs[1, 1])
#     abs_phi = np.abs(shap_vals)
#     if abs_phi.sum() > 0:
#         fracs  = abs_phi / abs_phi.sum()
#         labels = [f"Doc[{i+1}]" for i in range(K)]
#         wedge_cols = [CYAN, TEAL, AMBER, ROSE, VIOLET, GREEN][:K]
#         wedges, texts, autotexts = ax4.pie(
#             fracs, labels=labels, colors=wedge_cols,
#             autopct="%1.1f%%", startangle=90,
#             wedgeprops={"edgecolor": "#0f1117", "linewidth": 0.8},
#             textprops={"color": "#e2e8f0", "fontsize": 8})
#         for at in autotexts:
#             at.set_fontsize(7.5)
#     style(ax4, "Relative Document Contribution (|φ| normalised)")

#     # 5. Query + recommendation text box
#     ax5 = fig.add_subplot(gs[1, 2])
#     ax5.set_facecolor("#161b27"); ax5.axis("off")
#     top_doc_idx = int(np.argmax(np.abs(shap_vals))) if len(shap_vals) > 0 else 0
#     summary_lines = [
#         "INFERENCE SUMMARY",
#         "",
#         f"Query (truncated):",
#         f"  {query[:80]}...",
#         "",
#         f"Best plan score:  {plan['score']:.4f}",
#         f"Actions planned:  {len(actions)} steps",
#         "",
#         f"Primary outcome:  {mu_np[0]:.3f} ± {std_np[0]:.3f}",
#         f"Key evidence:     Doc[{top_doc_idx+1}] (φ={shap_vals[top_doc_idx]:.4f})",
#         "",
#         f"Top doc preview:",
#         f"  {docs[top_doc_idx][:90]}..." if docs else "  —",
#     ]
#     for i, line in enumerate(summary_lines):
#         color = CYAN if i == 0 else (AMBER if line.startswith("Query") or
#                                      line.startswith("Best") or
#                                      line.startswith("Primary") or
#                                      line.startswith("Key") else "#e2e8f0")
#         ax5.text(0.03, 0.97 - i*0.072, line, transform=ax5.transAxes,
#                  color=color, fontsize=8, va="top",
#                  fontweight="bold" if i == 0 else "normal")
#     for sp in ax5.spines.values():
#         sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
#     ax5.set_title("Inference Summary", color="#e2e8f0", fontsize=9, fontweight="bold", pad=8)

#     fig.suptitle("GRAPES-SHAP — Full Inference Results (DDXPlus + MedQA)",
#                  color="#e2e8f0", fontsize=13, fontweight="bold", y=1.01)
#     path = FIG_DIR / "04_inference_results.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Saved: {path}")


# # ──────────────────────────────────────────────────────────────
# # 15. VISUALISATION — LATENT SPACE (t-SNE)
# # ──────────────────────────────────────────────────────────────

# def plot_latent_space(enc, gnn, kg, loader, cfg, n_samples=2000):
#     enc.eval(); gnn.eval()
#     nf, adj, ew, mask = kg.subgraph(list(range(min(6, kg.n))))
#     all_z, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             if len(all_z) * cfg.batch_size >= n_samples:
#                 break
#             obs = batch["obs"].to(cfg.device)
#             pid = batch["pathology_id"]
#             B   = obs.shape[0]
#             nf_b = nf.unsqueeze(0).expand(B,-1,-1)
#             ad_b = adj.unsqueeze(0).expand(B,-1,-1)
#             ew_b = ew.unsqueeze(0).expand(B,-1,-1)
#             _, g = gnn(nf_b, ad_b, ew_b, mask)
#             z    = enc(obs, g)[:,-1,:].cpu().numpy()
#             all_z.append(z); all_labels.extend(pid.tolist())

#     Z = np.vstack(all_z)[:n_samples]
#     L = np.array(all_labels)[:n_samples]

#     print(f"  Running t-SNE on {len(Z)} latent vectors...")
#     tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_iter=500)
#     Z2d  = tsne.fit_transform(Z)

#     fig, axes = plt.subplots(1, 2, figsize=(18, 7))
#     fig.patch.set_facecolor("#0f1117")

#     n_cls = min(15, len(np.unique(L)))
#     cmap  = plt.cm.get_cmap("tab20", n_cls)
#     for i, ax in enumerate(axes):
#         ax.set_facecolor("#161b27")
#         for sp in ax.spines.values():
#             sp.set_edgecolor("#2a3348"); sp.set_linewidth(0.5)
#         ax.tick_params(colors="#8892a4", labelsize=7)

#     sc1 = axes[0].scatter(Z2d[:,0], Z2d[:,1], c=L % n_cls, cmap=cmap,
#                           s=5, alpha=0.5, linewidths=0)
#     axes[0].set_title("Latent Space Coloured by Pathology Class",
#                       color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
#     axes[0].set_xlabel("t-SNE dim 1", color="#8892a4", fontsize=8)
#     axes[0].set_ylabel("t-SNE dim 2", color="#8892a4", fontsize=8)
#     plt.colorbar(sc1, ax=axes[0]).ax.yaxis.set_tick_params(color="#8892a4", labelsize=7)

#     # Density plot
#     axes[1].hexbin(Z2d[:,0], Z2d[:,1], gridsize=60, cmap="plasma", alpha=0.9)
#     axes[1].set_title("Latent Space Density (Hexbin)",
#                       color="#e2e8f0", fontsize=10, fontweight="bold", pad=10)
#     axes[1].set_xlabel("t-SNE dim 1", color="#8892a4", fontsize=8)
#     axes[1].set_ylabel("t-SNE dim 2", color="#8892a4", fontsize=8)

#     fig.suptitle("GRAPES-SHAP — Latent Space Visualisation (Evidence Fusion Encoder)",
#                  color="#e2e8f0", fontsize=12, fontweight="bold")
#     path = FIG_DIR / "05_latent_space.png"
#     plt.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"  Saved: {path}")


# # ──────────────────────────────────────────────────────────────
# # 16. FULL INFERENCE PIPELINE — one patient query
# # ──────────────────────────────────────────────────────────────

# def full_inference_pipeline(query: str, wm, enc, gnn, kg, ens,
#                              retriever: HybridRetriever,
#                              shap_attr: SHAPAttributor,
#                              cfg: Config) -> Dict:
#     dev = cfg.device
#     wm.eval(); enc.eval(); gnn.eval(); ens.eval()

#     docs = retriever.retrieve(query, k=cfg.top_k)

#     seed_ids = list(range(min(5, kg.n)))
#     nf, adj, ew, mask = kg.subgraph(seed_ids)
#     _, g_emb = gnn(nf.unsqueeze(0), adj.unsqueeze(0), ew.unsqueeze(0), mask)

#     init_obs = torch.randn(1, cfg.seq_len, cfg.obs_dim, device=dev) * 0.25
#     with torch.no_grad():
#         z_seq = enc(init_obs, g_emb)
#     z0 = z_seq[:, -1, :]

#     planner = ToTPlanner(wm, ens, cfg)
#     plan    = planner.plan(z0, g_emb.squeeze(0))

#     shap_vals = shap_attr.shapley(query, docs) if docs else np.array([])

#     return {"query": query, "docs": docs, "plan": plan,
#             "shap_vals": shap_vals, "g_emb": g_emb}


# # ──────────────────────────────────────────────────────────────
# # 17. MAIN — Full research workflow
# # ──────────────────────────────────────────────────────────────

# def main():
#     print("\n" + "═"*68)
#     print("  GRAPES-SHAP  |  Full Research Pipeline")
#     print("  Datasets: DDXPlus + MedMCQA + MedQA (zero-barrier)")
#     print("═"*68 + "\n")
#     t0 = time.time()

#     # ── Step 1: Load datasets ──
#     print("[1/9] Loading datasets...")
#     loader_cls = DatasetLoader()
#     train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
#         CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test)
#     medmcqa_raw  = DatasetLoader.load_medmcqa(CFG.medmcqa_n_train)
#     medqa_raw    = DatasetLoader.load_medqa(CFG.medqa_n_test)

#     # ── Step 2: Preprocess ──
#     print("\n[2/9] Preprocessing...")
#     preprocessor = DDXPlusPreprocessor(CFG)
#     preprocessor.fit(train_raw)

#     processed_train = preprocessor.transform(train_raw)
#     processed_val   = preprocessor.transform(val_raw)
#     processed_test  = preprocessor.transform(test_raw)
#     print(f"  Processed — train:{len(processed_train)} val:{len(processed_val)} test:{len(processed_test)}")

#     medmcqa_docs  = MedMCQAPreprocessor.to_documents(medmcqa_raw)
#     medqa_queries = MedQAPreprocessor.to_queries(medqa_raw)
#     print(f"  RAG corpus: {len(medmcqa_docs):,} MedMCQA documents")
#     print(f"  Eval queries: {len(medqa_queries)} MedQA USMLE questions")

#     # ── Step 3: Data exploration visualisation ──
#     print("\n[3/9] Data exploration visualisation...")
#     plot_dataset_overview(processed_train, processed_val, processed_test, preprocessor)

#     # ── Step 4: Build retrieval index ──
#     print("\n[4/9] Building hybrid retrieval index (Dense + BM25)...")
#     retriever = HybridRetriever(CFG)
#     retriever.build(medmcqa_docs[:30_000])

#     # ── Step 5: Initialise architecture ──
#     print("\n[5/9] Initialising GRAPES-SHAP architecture...")
#     kg  = MedicalKG(preprocessor, CFG.n_graph_nodes, CFG.graph_node_dim, CFG.device)
#     gnn = CausalGNN(CFG).to(CFG.device)
#     enc = EvidenceFusionEncoder(CFG).to(CFG.device)
#     wm  = LatentWorldModel(CFG).to(CFG.device)
#     ens = DeepEnsemble(CFG).to(CFG.device)

#     total_params = sum(p.numel() for m in [gnn,enc,wm,ens] for p in m.parameters())
#     print(f"  Total neural parameters: {total_params:,}")
#     if torch.cuda.is_available():
#         print(f"  VRAM after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#     # DataLoaders
#     ds_train = ClinicalTrajectoryDataset(processed_train)
#     ds_val   = ClinicalTrajectoryDataset(processed_val)
#     dl_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True,
#                           num_workers=2, pin_memory=True)
#     dl_val   = DataLoader(ds_val,   batch_size=128, shuffle=False,
#                           num_workers=2, pin_memory=True)

#     # ── Step 6: Training ──
#     print("\n[6/9] Training World Model...")
#     wm_hist  = train_world_model(wm, enc, gnn, kg, dl_train, CFG)

#     print("\n       Training Deep Ensemble...")
#     ens_hist = train_ensemble(ens, enc, gnn, kg, dl_train, CFG)

#     plot_training_history(wm_hist, ens_hist)

#     # ── Step 7: Evaluation ──
#     print("\n[7/9] Full evaluation...")
#     shap_attr = SHAPAttributor(CFG)
#     metrics   = evaluate_all(ens, enc, gnn, kg, dl_val,
#                               medqa_queries, retriever, shap_attr, CFG)
#     plot_performance_dashboard(metrics)

#     # ── Step 8: Latent space visualisation ──
#     print("\n[8/9] Latent space t-SNE visualisation...")
#     try:
#         plot_latent_space(enc, gnn, kg, dl_val, CFG, n_samples=2000)
#     except Exception as e:
#         print(f"  t-SNE skipped: {e}")

#     # ── Step 9: Full inference on MedQA sample + SHAP ──
#     print("\n[9/9] Full inference pipeline demonstration...")
#     sample_query = medqa_queries[0]["question"] if medqa_queries else (
#         "A 62-year-old woman presents with sudden onset chest pain radiating to her left arm. "
#         "ECG shows ST elevation in leads V1-V4. Troponin is elevated. "
#         "She has a 10-year history of hypertension. What is the best immediate treatment?")
#     print(f"  Query: {sample_query[:100]}...")

#     outcome_names = ["Dx-1 prob","Dx-2 prob","Dx-3 prob","Dx-4 prob","Dx-5 prob"]
#     result = full_inference_pipeline(
#         sample_query, wm, enc, gnn, kg, ens, retriever, shap_attr, CFG)
#     plan = result["plan"]
#     docs = result["docs"]
#     shap_vals = result["shap_vals"]

#     print(f"\n  Retrieved {len(docs)} documents:")
#     for i, d in enumerate(docs):
#         print(f"    [{i+1}] {d[:75]}...")

#     print(f"\n  ToT best plan score: {plan['score']:.4f}")
#     print(f"  Action sequence: {[a.item() for a in plan['actions']] if plan['actions'] else []}")

#     if plan["mu"] is not None:
#         mu_np  = plan["mu"][0].cpu().numpy()
#         std_np = plan["std"][0].cpu().numpy()
#         print(f"\n  Predicted outcomes:")
#         for n, m, s in zip(outcome_names, mu_np, std_np):
#             bar = "█" * int(m*25) + "░" * (25 - int(m*25))
#             print(f"    {n:<14s} {m:.3f} ± {s:.3f}  [{bar}]")

#     if len(shap_vals) > 0:
#         print(f"\n  SHAP document attributions:")
#         ranked = sorted(zip(range(len(docs)), shap_vals.tolist()),
#                         key=lambda x: -abs(x[1]))
#         for idx, phi in ranked:
#             print(f"    Doc[{idx+1}] φ={phi:+.4f}  {docs[idx][:65]}...")

#     if docs and len(shap_vals) > 0:
#         plot_inference_results(sample_query, docs, shap_vals, plan, outcome_names)

#     # ── Save full metrics report ──
#     report = {
#         "mae":               float(metrics["mae"]),
#         "rmse":              float(metrics["rmse"]),
#         "1sigma_coverage":   float(metrics["1sigma_coverage"]),
#         "ece":               float(metrics["ece"]),
#         "accuracy":          float(metrics["accuracy"]),
#         "f1_macro":          float(metrics["f1_macro"]),
#         "mean_shap":         float(metrics["mean_shap"]),
#         "total_params":      total_params,
#         "training_time_min": round((time.time()-t0)/60, 2),
#         "dataset":           "DDXPlus + MedMCQA + MedQA",
#         "device":            CFG.device,
#     }
#     with open(SAVE_DIR / "metrics_report.json", "w") as f:
#         json.dump(report, f, indent=2)

#     print("\n" + "═"*68)
#     print("  FINAL METRICS SUMMARY")
#     print("═"*68)
#     for k, v in report.items():
#         if isinstance(v, float):
#             print(f"  {k:<28s} {v:.4f}")
#         else:
#             print(f"  {k:<28s} {v}")
#     print(f"\n  Figures saved to: {FIG_DIR.resolve()}")
#     print(f"  Checkpoints:      {CKPT_DIR.resolve()}")
#     print(f"  Metrics report:   {SAVE_DIR/'metrics_report.json'}")
#     print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} minutes")
#     print("═"*68 + "\n")


# if __name__ == "__main__":
#     main()
