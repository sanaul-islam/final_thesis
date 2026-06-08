#!/usr/bin/env python3
"""
Extract training metrics from tqdm progress bars
Analyzes World Model and Ensemble training performance
"""

import re
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent
LOG_FILE = PROJECT_ROOT / "training_log.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True, parents=True)


def extract_speed_from_tqdm(log_file: Path) -> Dict[str, List]:
    """
    Extract speed metrics from tqdm progress bars
    Format: "WM 14/15:  82%|████| 1021/1250 [00:39<00:07, 31.70it/s]"
    """
    
    if not log_file.exists():
        print(f"[ERROR] Log file not found: {log_file}")
        return {}
    
    wm_metrics = []
    ens_metrics = []
    
    # Pattern to extract: phase, epoch, total_epochs, percent, current, total, time, speed
    pattern = r'(WM|Ens)\s+(\d+)/(\d+):\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[[\w:]+<[\w:]+,\s+([\d.]+)it/s'
    
    print(f"[INFO] Reading {log_file}...")
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Extract progress lines
        for line in lines:
            match = re.search(pattern, line)
            if match:
                phase = match.group(1)
                epoch = int(match.group(2))
                total_epochs = int(match.group(3))
                percent = int(match.group(4))
                current_iter = int(match.group(5))
                total_iters = int(match.group(6))
                speed = float(match.group(7))
                
                metric = {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "percent": percent,
                    "iteration": current_iter,
                    "total_iterations": total_iters,
                    "speed_it_s": speed
                }
                
                if phase == "WM":
                    wm_metrics.append(metric)
                else:
                    ens_metrics.append(metric)
    
    except Exception as e:
        print(f"[ERROR] Failed to read log: {e}")
        return {}
    
    return {
        "world_model": wm_metrics,
        "ensemble": ens_metrics
    }


def analyze_metrics(metrics: Dict[str, List]) -> Dict:
    """Analyze collected metrics"""
    
    analysis = {
        "world_model": {},
        "ensemble": {}
    }
    
    for phase_name in ["world_model", "ensemble"]:
        phase_metrics = metrics.get(phase_name, [])
        
        if not phase_metrics:
            print(f"[SKIP] No metrics found for {phase_name}")
            continue
        
        # Extract unique epochs
        epochs = {}
        for metric in phase_metrics:
            epoch = metric["epoch"]
            if epoch not in epochs:
                epochs[epoch] = []
            epochs[epoch].append(metric["speed_it_s"])
        
        # Calculate per-epoch statistics
        epoch_stats = []
        for epoch in sorted(epochs.keys()):
            speeds = epochs[epoch]
            epoch_stats.append({
                "epoch": epoch,
                "samples": len(speeds),
                "avg_speed": sum(speeds) / len(speeds),
                "min_speed": min(speeds),
                "max_speed": max(speeds)
            })
        
        # Overall statistics
        all_speeds = [m["speed_it_s"] for m in phase_metrics]
        
        analysis[phase_name] = {
            "total_records": len(phase_metrics),
            "total_epochs": max([m["epoch"] for m in phase_metrics]) if phase_metrics else 0,
            "epoch_stats": epoch_stats,
            "overall_avg_speed": sum(all_speeds) / len(all_speeds) if all_speeds else 0,
            "overall_min_speed": min(all_speeds) if all_speeds else 0,
            "overall_max_speed": max(all_speeds) if all_speeds else 0,
            "speed_samples": len(all_speeds)
        }
    
    return analysis


def save_results(metrics: Dict, analysis: Dict):
    """Save metrics and analysis to files"""
    
    # Save raw metrics as JSON
    metrics_json = METRICS_DIR / "tqdm_metrics.json"
    with open(metrics_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {metrics_json}")
    
    # Save analysis as JSON
    analysis_json = METRICS_DIR / "training_analysis.json"
    with open(analysis_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"[SAVED] {analysis_json}")
    
    # Save as CSV
    csv_file = METRICS_DIR / "training_metrics.csv"
    rows = []
    
    for record in metrics.get("world_model", []):
        rows.append({
            "phase": "WorldModel",
            "epoch": record["epoch"],
            "iteration": record["iteration"],
            "total_iterations": record["total_iterations"],
            "speed_it_s": record["speed_it_s"],
            "percent": record["percent"]
        })
    
    for record in metrics.get("ensemble", []):
        rows.append({
            "phase": "Ensemble",
            "epoch": record["epoch"],
            "iteration": record["iteration"],
            "total_iterations": record["total_iterations"],
            "speed_it_s": record["speed_it_s"],
            "percent": record["percent"]
        })
    
    if rows:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SAVED] {csv_file}")
    
    # Print summary
    print("\n" + "="*70)
    print(" TRAINING METRICS SUMMARY")
    print("="*70)
    
    for phase in ["world_model", "ensemble"]:
        stats = analysis.get(phase, {})
        if not stats:
            continue
        
        print(f"\n{phase.upper().replace('_', ' ')}:")
        print(f"  Total Records: {stats.get('total_records', 0)}")
        print(f"  Epochs Completed: {stats.get('total_epochs', 0)}")
        print(f"  Avg Speed: {stats.get('overall_avg_speed', 0):.2f} it/s")
        print(f"  Min/Max Speed: {stats.get('overall_min_speed', 0):.2f} / {stats.get('overall_max_speed', 0):.2f} it/s")
        
        # Per-epoch breakdown
        print(f"\n  Per-epoch breakdown:")
        for epoch_info in stats.get("epoch_stats", [])[-5:]:  # Last 5 epochs
            print(f"    Epoch {epoch_info['epoch']:2d}: "
                  f"avg={epoch_info['avg_speed']:6.2f} it/s, "
                  f"range={epoch_info['min_speed']:6.2f}-{epoch_info['max_speed']:6.2f}")


if __name__ == "__main__":
    print("[START] Training Metrics Extraction")
    print("="*70)
    
    # Extract metrics
    metrics = extract_speed_from_tqdm(LOG_FILE)
    
    if not metrics or (not metrics.get("world_model") and not metrics.get("ensemble")):
        print("\n[ERROR] No metrics extracted from log file")
        print("[INFO] Verifying log file content...")
        
        # Debug: Check what's in the file
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if "WM" in content or "Ens" in content:
                print(f"[INFO] Log contains phase indicators, file size: {len(content)} bytes")
                # Try to find lines
                for line in content.split('\n')[-10:]:
                    if line.strip():
                        print(f"  {repr(line[:80])}")
    else:
        # Analyze metrics
        analysis = analyze_metrics(metrics)
        
        # Save results
        save_results(metrics, analysis)
        print("\n[DONE] Metrics extraction complete")
