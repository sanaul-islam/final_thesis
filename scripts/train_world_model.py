#!/usr/bin/env python3
"""
World Model Training Script
Trains the Latent World Model on preprocessed clinical trajectory data.
Run from project root: python scripts/train_world_model.py
"""

import sys
import time
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grapes_shap.config import CFG, SAVE_DIR, FIG_DIR, CKPT_DIR
from grapes_shap.data import DatasetLoader, DDXPlusPreprocessor, ClinicalTrajectoryDataset
from grapes_shap.models import MedicalKG, CausalGNN, EvidenceFusionEncoder, LatentWorldModel
from grapes_shap.training import train_world_model
from torch.utils.data import DataLoader


def main():
    """Train the world model."""
    
    print("\n" + "=" * 70)
    print("  GRAPES-SHAP  |  World Model Training")
    print("=" * 70 + "\n")
    
    t_start = time.time()
    
    # ─────────────────────────────────────────────────────────────
    # Step 1: Load and preprocess data
    # ─────────────────────────────────────────────────────────────
    print("[1/5] Loading datasets...")
    
    train_raw, val_raw, _ = DatasetLoader.load_ddxplus(
        CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test
    )
    print(f"  ✓ DDXPlus: {len(train_raw)} train, {len(val_raw)} val")
    
    print("\n[2/5] Preprocessing data...")
    preprocessor = DDXPlusPreprocessor(CFG)
    preprocessor.fit(train_raw)
    
    processed_train = preprocessor.transform(train_raw)
    processed_val = preprocessor.transform(val_raw)
    print(f"  ✓ Train: {len(processed_train)} samples")
    print(f"  ✓ Val: {len(processed_val)} samples")
    
    # ─────────────────────────────────────────────────────────────
    # Step 3: Create PyTorch datasets and dataloaders
    # ─────────────────────────────────────────────────────────────
    print("\n[3/5] Creating datasets and dataloaders...")
    
    ds_train = ClinicalTrajectoryDataset(processed_train)
    ds_val = ClinicalTrajectoryDataset(processed_val)
    
    dl_train = DataLoader(
        ds_train, 
        batch_size=CFG.batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    
    dl_val = DataLoader(
        ds_val, 
        batch_size=128, 
        shuffle=False,
        num_workers=2, 
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"  ✓ Train batches: {len(dl_train)} ({CFG.batch_size} per batch)")
    print(f"  ✓ Val batches: {len(dl_val)}")
    
    # ─────────────────────────────────────────────────────────────
    # Step 4: Initialize models
    # ─────────────────────────────────────────────────────────────
    print("\n[4/5] Initializing models...")
    
    # Medical Knowledge Graph
    kg = MedicalKG(
        preprocessor, 
        CFG.n_graph_nodes, 
        CFG.graph_node_dim, 
        CFG.device
    )
    print(f"  ✓ MedicalKG initialized")
    
    # Causal GNN
    gnn = CausalGNN(CFG).to(CFG.device)
    print(f"  ✓ CausalGNN initialized ({sum(p.numel() for p in gnn.parameters()):,} params)")
    
    # Evidence Fusion Encoder
    enc = EvidenceFusionEncoder(CFG).to(CFG.device)
    print(f"  ✓ EvidenceFusionEncoder initialized ({sum(p.numel() for p in enc.parameters()):,} params)")
    
    # World Model
    wm = LatentWorldModel(CFG).to(CFG.device)
    print(f"  ✓ LatentWorldModel initialized ({sum(p.numel() for p in wm.parameters()):,} params)")
    
    total_params = sum(p.numel() for m in [gnn, enc, wm] for p in m.parameters())
    print(f"\n  Total trainable parameters: {total_params:,}")
    
    if torch.cuda.is_available():
        print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ─────────────────────────────────────────────────────────────
    # Step 5: Train world model
    # ─────────────────────────────────────────────────────────────
    print("\n[5/5] Training world model...")
    print(f"  Configuration:")
    print(f"    - Epochs: {CFG.wm_epochs}")
    print(f"    - Learning rate: {CFG.wm_lr}")
    print(f"    - Batch size: {CFG.batch_size}")
    print(f"    - Optimizer: Adam with weight decay {CFG.weight_decay}")
    print(f"    - Device: {CFG.device}")
    
    t_train_start = time.time()
    
    wm_history = train_world_model(
        wm=wm,
        enc=enc,
        gnn=gnn,
        kg=kg,
        loader=dl_train,
        cfg=CFG
    )
    
    elapsed_train = time.time() - t_train_start
    
    print(f"\n  ✓ Training complete in {elapsed_train:.1f}s")
    
    # ─────────────────────────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────────────────────────
    print("\n  Saving model...")
    
    # Save world model checkpoint
    wm_checkpoint = {
        'model_state_dict': wm.state_dict(),
        'config': CFG.__dict__ if hasattr(CFG, '__dict__') else str(CFG),
        'epoch': CFG.wm_epochs,
        'history': wm_history,
    }
    
    wm_path = CKPT_DIR / "world_model.pt"
    torch.save(wm_checkpoint, wm_path)
    print(f"  ✓ World model saved: {wm_path}")
    
    # Save training history
    history_path = SAVE_DIR / "world_model_history.json"
    with open(history_path, "w") as f:
        json.dump(wm_history, f, indent=2)
    print(f"  ✓ History saved: {history_path}")
    
    # ─────────────────────────────────────────────────────────────
    # Print summary
    # ─────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    
    print("\n" + "=" * 70)
    print("✓ World Model Training Complete!")
    print("=" * 70)
    
    print(f"\n📊 Training Summary:")
    print(f"   Total time: {elapsed_total:.1f}s")
    print(f"   Epochs: {CFG.wm_epochs}")
    
    if wm_history and len(wm_history) > 0:
        last_epoch = wm_history[-1]
        print(f"\n   Final Metrics (Epoch {CFG.wm_epochs}):")
        for key, val in last_epoch.items():
            if isinstance(val, (int, float)):
                if 'loss' in key.lower():
                    print(f"     - {key}: {val:.4f}")
                else:
                    print(f"     - {key}: {val:.6f}")
    
    print(f"\n💾 Checkpoints:")
    print(f"   - World model: {wm_path}")
    print(f"   - History: {history_path}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Train ensemble: python scripts/train_ensemble.py")
    print(f"   2. Full pipeline: python run.py")
    print(f"   3. Evaluate: python scripts/evaluate.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
