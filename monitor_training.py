#!/usr/bin/env python3
"""
Real-time Training Monitor
Continuously tracks and updates training metrics
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
import signal
import sys

PROJECT_ROOT = Path(__file__).parent

def run_tracker():
    """Run the training tracker"""
    tracker_script = PROJECT_ROOT / "training_tracker.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(tracker_script)],
            capture_output=False,
            timeout=600  # 10 minute timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠️  Tracker timeout (training still in progress)")
        return False
    except Exception as e:
        print(f"Error running tracker: {e}")
        return False

def monitor_training():
    """Monitor training in real-time"""
    log_file = PROJECT_ROOT / "training_log.txt"
    last_size = 0
    check_interval = 300  # Check every 5 minutes
    
    print("\n" + "="*70)
    print("  REAL-TIME TRAINING MONITOR")
    print("="*70 + "\n")
    
    while True:
        try:
            if log_file.exists():
                current_size = log_file.stat().st_size
                
                if current_size > last_size:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Log file updated "
                          f"({last_size} -> {current_size} bytes)")
                    
                    # Show last few lines
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            print("\nLatest training output:")
                            for line in lines[-5:]:
                                print(f"  {line.rstrip()}")
                    except:
                        pass
                    
                    last_size = current_size
                
                # Run tracker periodically
                if current_size % 1000000 == 0 or current_size > 5000000:
                    print(f"\n📊 Running metrics analysis...")
                    run_tracker()
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for log file...")
            
            time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitor stopped by user")
            print("Running final metrics analysis...")
            run_tracker()
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\nMonitor terminated.")
        sys.exit(0)
