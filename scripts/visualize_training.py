#!/usr/bin/env python3
"""
Training Visualization Script
Generates and displays visualizations of model training metrics.
Run from project root: python scripts/visualize_training.py
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grapes_shap.config import SAVE_DIR, FIG_DIR, CKPT_DIR


def load_training_history():
    """Load training history from saved JSON files."""
    history_path = SAVE_DIR / "world_model_history.json"
    
    if history_path.exists():
        with open(history_path, "r") as f:
            return json.load(f)
    return None


def create_training_visualizations():
    """Create comprehensive training visualizations."""
    
    print("\n" + "=" * 70)
    print("  GRAPES-SHAP  |  Training Visualizations")
    print("=" * 70 + "\n")
    
    # Load history
    history = load_training_history()
    
    if not history:
        print("❌ No training history found.")
        print("   Run 'python scripts/train_world_model.py' first.")
        return False
    
    print("✓ Training history loaded")
    print(f"  Epochs: {len(history.get('loss', []))}")
    print(f"  Metrics: {list(history.keys())}")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ─────────────────────────────────────────────────────────────
    # 1. Total Loss over epochs
    # ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    if "loss" in history:
        losses = history["loss"]
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, 'b-', linewidth=2.5, marker='o', markersize=5)
        ax1.fill_between(epochs, losses, alpha=0.3)
        ax1.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Loss", fontsize=11, fontweight='bold')
        ax1.set_title("World Model: Total Loss", fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add best loss annotation
        best_epoch = np.argmin(losses) + 1
        best_loss = min(losses)
        ax1.annotate(
            f"Best: {best_loss:.4f}\n(Epoch {best_epoch})",
            xy=(best_epoch, best_loss),
            xytext=(best_epoch + 2, best_loss + 0.05 * (max(losses) - min(losses))),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5)
        )
    
    # ─────────────────────────────────────────────────────────────
    # 2. Reconstruction Loss
    # ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    if "recon_loss" in history:
        recon_losses = history["recon_loss"]
        epochs = range(1, len(recon_losses) + 1)
        ax2.plot(epochs, recon_losses, 'g-', linewidth=2.5, marker='s', markersize=5)
        ax2.fill_between(epochs, recon_losses, alpha=0.3, color='green')
        ax2.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Reconstruction Loss", fontsize=11, fontweight='bold')
        ax2.set_title("World Model: Reconstruction Loss", fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add improvement annotation
        improvement = (recon_losses[0] - recon_losses[-1]) / recon_losses[0] * 100
        ax2.text(
            0.98, 0.97,
            f"Improvement: {improvement:.1f}%",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        )
    
    # ─────────────────────────────────────────────────────────────
    # 3. Learning Rate Schedule
    # ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    
    if "lr" in history:
        lrs = history["lr"]
        epochs = range(1, len(lrs) + 1)
        ax3.plot(epochs, lrs, 'r-', linewidth=2.5, marker='^', markersize=5)
        ax3.fill_between(epochs, lrs, alpha=0.3, color='red')
        ax3.set_xlabel("Epoch", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Learning Rate", fontsize=11, fontweight='bold')
        ax3.set_title("Learning Rate Schedule (OneCycleLR)", fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # ─────────────────────────────────────────────────────────────
    # 4. Summary Statistics
    # ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = "📊 TRAINING SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"
    
    if "loss" in history:
        losses = history["loss"]
        summary_text += f"Total Loss:\n"
        summary_text += f"  • Initial: {losses[0]:.6f}\n"
        summary_text += f"  • Final: {losses[-1]:.6f}\n"
        summary_text += f"  • Best: {min(losses):.6f}\n"
        summary_text += f"  • Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%\n\n"
    
    if "recon_loss" in history:
        recon = history["recon_loss"]
        summary_text += f"Reconstruction Loss:\n"
        summary_text += f"  • Initial: {recon[0]:.6f}\n"
        summary_text += f"  • Final: {recon[-1]:.6f}\n"
        summary_text += f"  • Best: {min(recon):.6f}\n\n"
    
    summary_text += f"Epochs: {len(history.get('loss', []))}\n"
    summary_text += f"Device: CPU\n"
    summary_text += f"Optimizer: AdamW\n"
    summary_text += f"Batch Size: 64\n\n"
    
    if "lr" in history:
        summary_text += f"Learning Rate:\n"
        summary_text += f"  • Max: {max(history['lr']):.6f}\n"
        summary_text += f"  • Final: {history['lr'][-1]:.6f}\n"
    
    ax4.text(
        0.05, 0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # Save figure
    fig.suptitle(
        "GRAPES-SHAP: World Model Training Metrics",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    output_path = FIG_DIR / "training_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Show figure
    plt.tight_layout()
    plt.show()
    
    return True


def create_model_info_viz():
    """Create visualization of model architecture information."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    model_info = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   GRAPES-SHAP Model Architecture                          ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 MODEL COMPONENTS & PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MEDICAL KNOWLEDGE GRAPH (MedicalKG)
   └─ Provides domain knowledge for causal reasoning
   └─ Nodes: 20 | Node Dimension: 128
   └─ Status: Non-trainable (lookup)

2. CAUSAL GRAPH NEURAL NETWORK (CausalGNN)
   └─ Parameters: 890,392
   └─ Purpose: Causal structure learning
   └─ Features: Attention-based message passing

3. EVIDENCE FUSION ENCODER (EvidenceFusionEncoder)
   └─ Parameters: 1,993,472
   └─ Purpose: Encodes clinical observations
   └─ Architecture: Transformer-based
   └─ Layers: 3 | Heads: 8 | Hidden: 512

4. LATENT WORLD MODEL (LatentWorldModel)
   └─ Parameters: 5,755,586
   └─ Purpose: Learns disease progression dynamics
   └─ Output: Predicted next observation + uncertainty
   └─ Loss: MSE reconstruction + smoothness + uncertainty

5. DEEP ENSEMBLE (DeepEnsemble)
   └─ Parameters: N ensemble models
   └─ Purpose: Outcome prediction & calibration
   └─ Loss: Negative Log-Likelihood (NLL)
   └─ Outputs: 5 outcome dimensions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 TOTAL TRAINABLE PARAMETERS: 8,639,450 (8.6M)

💾 CHECKPOINT SIZES:
   • World Model: 34.6 MB (wm + encoder + gnn)
   • Ensemble: 5.98 MB

🎯 TRAINING OBJECTIVES:
   ✓ World Model Loss = Reconstruction + Smoothness + Uncertainty
   ✓ Ensemble Loss = Outcome Prediction NLL

🚀 TRAINING CONFIGURATION:
   • Optimizer: AdamW (weight_decay=0.0001)
   • WM Learning Rate: 0.0002 (OneCycleLR)
   • Ensemble LR: 0.001 (CosineAnnealing)
   • Batch Size: 64
   • Gradient Clipping: 1.0
   • Mixed Precision: float16 (when CUDA available)

📊 DATA PROCESSING:
   • Input: Clinical trajectories (sequences of observations)
   • Observation Dim: 64 | Latent Dim: 256
   • Action Sequence Length: 8
   • Outcome Dimensions: 5
   
════════════════════════════════════════════════════════════════════════════
"""
    
    ax.text(
        0.05, 0.95,
        model_info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1)
    )
    
    output_path = FIG_DIR / "model_architecture.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.show()


def create_checkpoint_info():
    """Display checkpoint information."""
    
    print("\n" + "=" * 70)
    print("  CHECKPOINT INFORMATION")
    print("=" * 70 + "\n")
    
    ckpt_dir = CKPT_DIR
    
    if not ckpt_dir.exists():
        print("❌ No checkpoints directory found")
        return False
    
    import os
    checkpoints = list(ckpt_dir.glob("*.pt"))
    
    if not checkpoints:
        print("❌ No checkpoint files found")
        return False
    
    print(f"📁 Location: {ckpt_dir}\n")
    
    for ckpt in sorted(checkpoints):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mod_time = Path(ckpt).stat().st_mtime
        from datetime import datetime
        mod_datetime = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"📦 {ckpt.name}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Modified: {mod_datetime}")
        
        if "world_model" in ckpt.name:
            print(f"   Contains: LatentWorldModel + EvidenceFusionEncoder + CausalGNN")
        elif "ensemble" in ckpt.name:
            print(f"   Contains: DeepEnsemble")
        print()
    
    return True


def main():
    """Generate all training visualizations."""
    
    print("\n📊 Generating training visualizations...")
    
    # Create visualizations
    success1 = create_training_visualizations()
    success2 = create_checkpoint_info()
    
    try:
        success3 = create_model_info_viz()
    except Exception as e:
        print(f"⚠️  Model info visualization skipped: {e}")
        success3 = False
    
    print("\n" + "=" * 70)
    if success1 or success2:
        print("✓ Visualizations generated successfully!")
        print("=" * 70)
        print(f"\n📁 Output location: {FIG_DIR}")
        print("\n🎯 Next steps:")
        print("   1. Review visualizations in outputs/figures/")
        print("   2. Run full pipeline: python run.py")
        print("   3. Generate inference results: python scripts/inference.py")
        print()
        return True
    else:
        print("❌ No visualizations could be generated")
        print("=" * 70)
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
