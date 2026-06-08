#!/usr/bin/env python3
"""
Data Preparation Pipeline
Loads and preprocesses all datasets (DDXPlus, MedMCQA, MedQA)
Run from project root: python scripts/prepare_data.py
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grapes_shap.config import CFG, SAVE_DIR, FIG_DIR, CKPT_DIR
from grapes_shap.data import (
    DatasetLoader,
    DDXPlusPreprocessor,
    MedMCQAPreprocessor,
    MedQAPreprocessor,
    ClinicalTrajectoryDataset
)


def main():
    """Run complete data preparation pipeline."""
    
    print("\n" + "=" * 70)
    print("  GRAPES-SHAP  |  Data Preparation Pipeline")
    print("  Datasets: DDXPlus + MedMCQA + MedQA")
    print("=" * 70 + "\n")
    
    t_start = time.time()
    
    # ─────────────────────────────────────────────────────────────
    # Step 1: Load raw datasets
    # ─────────────────────────────────────────────────────────────
    print("[1/4] Loading raw datasets from HuggingFace...")
    print(f"      DDXPlus: train={CFG.ddxplus_n_train}, val={CFG.ddxplus_n_val}, test={CFG.ddxplus_n_test}")
    print(f"      MedMCQA: {CFG.medmcqa_n_train} samples")
    print(f"      MedQA:   {CFG.medqa_n_test} samples")
    
    try:
        train_raw, val_raw, test_raw = DatasetLoader.load_ddxplus(
            CFG.ddxplus_n_train, CFG.ddxplus_n_val, CFG.ddxplus_n_test
        )
        print(f"  ✓ DDXPlus loaded: {len(train_raw)} train, {len(val_raw)} val, {len(test_raw)} test")
    except Exception as e:
        print(f"  ✗ Error loading DDXPlus: {e}")
        return False
    
    try:
        medmcqa_raw = DatasetLoader.load_medmcqa(CFG.medmcqa_n_train)
        print(f"  ✓ MedMCQA loaded: {len(medmcqa_raw)} samples")
    except Exception as e:
        print(f"  ✗ Error loading MedMCQA: {e}")
        return False
    
    try:
        medqa_raw = DatasetLoader.load_medqa(CFG.medqa_n_test)
        print(f"  ✓ MedQA loaded: {len(medqa_raw)} samples")
    except Exception as e:
        print(f"  ✗ Error loading MedQA: {e}")
        return False
    
    # ─────────────────────────────────────────────────────────────
    # Step 2: Preprocess datasets
    # ─────────────────────────────────────────────────────────────
    print("\n[2/4] Preprocessing datasets...")
    
    # DDXPlus preprocessing
    print("  Preprocessing DDXPlus...")
    preprocessor = DDXPlusPreprocessor(CFG)
    preprocessor.fit(train_raw)
    
    processed_train = preprocessor.transform(train_raw)
    processed_val = preprocessor.transform(val_raw)
    processed_test = preprocessor.transform(test_raw)
    
    print(f"    ✓ DDXPlus: {len(processed_train)} train, {len(processed_val)} val, {len(processed_test)} test")
    
    # MedMCQA preprocessing
    print("  Preprocessing MedMCQA...")
    medmcqa_docs = MedMCQAPreprocessor.to_documents(medmcqa_raw)
    print(f"    ✓ MedMCQA: {len(medmcqa_docs):,} documents")
    
    # MedQA preprocessing
    print("  Preprocessing MedQA...")
    medqa_queries = MedQAPreprocessor.to_queries(medqa_raw)
    print(f"    ✓ MedQA: {len(medqa_queries)} queries")
    
    # ─────────────────────────────────────────────────────────────
    # Step 3: Create torch datasets
    # ─────────────────────────────────────────────────────────────
    print("\n[3/4] Creating PyTorch datasets...")
    
    try:
        ds_train = ClinicalTrajectoryDataset(processed_train)
        ds_val = ClinicalTrajectoryDataset(processed_val)
        ds_test = ClinicalTrajectoryDataset(processed_test)
        
        print(f"  ✓ Train dataset: {len(ds_train)} samples")
        print(f"  ✓ Val dataset: {len(ds_val)} samples")
        print(f"  ✓ Test dataset: {len(ds_test)} samples")
        
        # Test a batch
        sample = ds_train[0]
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        for key, val in sample.items():
            if hasattr(val, 'shape'):
                print(f"    - {key}: {val.shape} {val.dtype}")
            else:
                print(f"    - {key}: {type(val)}")
    except Exception as e:
        print(f"  ✗ Error creating datasets: {e}")
        return False
    
    # ─────────────────────────────────────────────────────────────
    # Step 4: Data summary and statistics
    # ─────────────────────────────────────────────────────────────
    print("\n[4/4] Data summary and statistics...")
    
    # DDXPlus statistics
    print("  DDXPlus Statistics:")
    print(f"    - Train samples: {len(processed_train)}")
    print(f"    - Val samples: {len(processed_val)}")
    print(f"    - Test samples: {len(processed_test)}")
    if processed_train:
        sample = processed_train[0]
        print(f"    - Sample keys: {list(sample.keys())}")
    
    # RAG corpus
    print("\n  RAG Corpus:")
    print(f"    - MedMCQA documents: {len(medmcqa_docs):,}")
    print(f"    - MedQA queries: {len(medqa_queries)}")
    
    # Dataset sizes
    total_samples = len(processed_train) + len(processed_val) + len(processed_test)
    print(f"\n  Total Samples: {total_samples:,}")
    
    # Save summary
    summary = {
        "ddxplus": {
            "train": len(processed_train),
            "val": len(processed_val),
            "test": len(processed_test),
        },
        "medmcqa": {
            "documents": len(medmcqa_docs),
        },
        "medqa": {
            "queries": len(medqa_queries),
        },
        "total_samples": total_samples,
        "preprocessor": {
            "type": "DDXPlusPreprocessor",
            "config": str(CFG),
        }
    }
    
    summary_path = SAVE_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  ✓ Summary saved to: {summary_path}")
    
    # ─────────────────────────────────────────────────────────────
    # Completion
    # ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"✓ Data preparation complete in {elapsed:.1f} seconds")
    print("=" * 70 + "\n")
    
    print("📊 Next steps:")
    print("   1. Review data_summary.json in outputs/")
    print("   2. Run full pipeline: python run.py")
    print("   3. Or train individual models: python scripts/train_*.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Data preparation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
