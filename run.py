#!/usr/bin/env python3
"""
Simple runner script for GRAPES-SHAP pipeline.
Execute from project root: python run.py
"""

import sys
from pathlib import Path

# Add src to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

if __name__ == "__main__":
    # Import and run the main pipeline
    from scripts.main import main
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
