#!/usr/bin/env python3
"""
Simple Training Metrics Tracker
Extracts speed and loss info from training log for research analysis
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent
LOG_FILE = PROJECT_ROOT / "training_log.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True, parents=True)


def parse_training_log() -> Dict[str, Any]:
    """Extract metrics from training log"""
    
    if not LOG_FILE.exists():
        print(f"[ERROR] Log file not found: {LOG_FILE}")
        return {}
    
    metrics = {
        "world_model": [],
        "ensemble": [],
        "summary": {}
    }
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"[INFO] Parsing {len(lines)} lines from {LOG_FILE}")
        
        # Parse epoch results - pattern: "  WM epoch  1 | loss=nan | recon=nan"
        epoch_pattern = r'(WM|ENS)\s+epoch\s+(\d+)\s*\|\s*loss=([^\s|]+)\s*\|\s*recon=([^\s|]+)'
        epoch_data = {}
        
        for line in lines:
            match = re.search(epoch_pattern, line)
            if match:
                phase = match.group(1)
                epoch = int(match.group(2))
                loss_str = match.group(3)
                recon_str = match.group(4)
                
                # Try to parse values
                loss = None if loss_str.lower() == 'nan' else (float(loss_str) if loss_str.replace('.', '', 1).replace('-', '', 1).isdigit() else None)
                recon = None if recon_str.lower() == 'nan' else (float(recon_str) if recon_str.replace('.', '', 1).replace('-', '', 1).isdigit() else None)
                
                key = f"{phase}_{epoch}"
                epoch_data[key] = {
                    "phase": phase,
                    "epoch": epoch,
                    "loss": loss,
                    "recon": recon
                }
        
        print(f"[INFO] Found {len(epoch_data)} epoch records")
        
        # Parse progress/speed data - note: leading spaces in log
        progress_pattern = r'\s+(WM|ENS)\s+(\d+)/(\d+):\s+\d+%\|[^\|]*\|\s+(\d+)/(\d+)\s+\[[\w:]+<[\w:]+,\s+([\d.]+)it/s'
        
        speed_by_epoch = {}
        for line in lines:
            match = re.search(progress_pattern, line)
            if match:
                phase = match.group(1)
                epoch = int(match.group(2))
                current = int(match.group(4))
                total = int(match.group(5))
                speed = float(match.group(6))
                
                key = f"{phase}_{epoch}"
                if key not in speed_by_epoch:
                    speed_by_epoch[key] = []
                speed_by_epoch[key].append(speed)
        
        print(f"[INFO] Extracted speed data from {len(speed_by_epoch)} epochs")
        
        # Combine data
        for key, epoch_info in epoch_data.items():
            speeds = speed_by_epoch.get(key, [])
            avg_speed = sum(speeds) / len(speeds) if speeds else None
            
            record = {
                "epoch": epoch_info["epoch"],
                "loss": epoch_info["loss"],
                "recon_loss": epoch_info["recon"],
                "speed_it_s": avg_speed,
                "speed_samples": len(speeds)
            }
            
            if epoch_info["phase"] == "WM":
                metrics["world_model"].append(record)
            else:
                metrics["ensemble"].append(record)
        
        # Calculate summary
        if metrics["world_model"]:
            wm_speeds = [m["speed_it_s"] for m in metrics["world_model"] if m["speed_it_s"]]
            wm_losses = [m["loss"] for m in metrics["world_model"] if m["loss"] is not None]
            
            metrics["summary"]["world_model"] = {
                "epochs": len(metrics["world_model"]),
                "avg_speed": sum(wm_speeds) / len(wm_speeds) if wm_speeds else None,
                "min_speed": min(wm_speeds) if wm_speeds else None,
                "max_speed": max(wm_speeds) if wm_speeds else None,
                "avg_loss": sum(wm_losses) / len(wm_losses) if wm_losses else None,
                "min_loss": min(wm_losses) if wm_losses else None,
                "max_loss": max(wm_losses) if wm_losses else None
            }
        
        if metrics["ensemble"]:
            ens_speeds = [m["speed_it_s"] for m in metrics["ensemble"] if m["speed_it_s"]]
            ens_losses = [m["loss"] for m in metrics["ensemble"] if m["loss"] is not None]
            
            metrics["summary"]["ensemble"] = {
                "epochs": len(metrics["ensemble"]),
                "avg_speed": sum(ens_speeds) / len(ens_speeds) if ens_speeds else None,
                "min_speed": min(ens_speeds) if ens_speeds else None,
                "max_speed": max(ens_speeds) if ens_speeds else None,
                "avg_loss": sum(ens_losses) / len(ens_losses) if ens_losses else None,
                "min_loss": min(ens_losses) if ens_losses else None,
                "max_loss": max(ens_losses) if ens_losses else None
            }
        
    except Exception as e:
        print(f"[ERROR] Failed to parse log: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics


def save_metrics(metrics: Dict[str, Any]):
    """Save metrics to JSON and CSV"""
    
    # Save JSON
    json_file = METRICS_DIR / "training_metrics.json"
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {json_file}")
    
    # Save CSV
    csv_file = METRICS_DIR / "training_metrics.csv"
    rows = []
    
    for record in metrics.get("world_model", []):
        rows.append({
            "phase": "WorldModel",
            "epoch": record["epoch"],
            "loss": record["loss"],
            "recon_loss": record["recon_loss"],
            "speed_it_s": record["speed_it_s"]
        })
    
    for record in metrics.get("ensemble", []):
        rows.append({
            "phase": "Ensemble",
            "epoch": record["epoch"],
            "loss": record["loss"],
            "recon_loss": record["recon_loss"],
            "speed_it_s": record["speed_it_s"]
        })
    
    if rows:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["phase", "epoch", "loss", "recon_loss", "speed_it_s"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SAVED] {csv_file}")
    
    # Print summary
    print("\n[SUMMARY] Training Metrics")
    print("=" * 60)
    
    for phase in ["world_model", "ensemble"]:
        if phase in metrics["summary"]:
            summary = metrics["summary"][phase]
            print(f"\n{phase.upper().replace('_', ' ')}:")
            print(f"  Epochs: {summary['epochs']}")
            print(f"  Avg Speed: {summary['avg_speed']:.2f} it/s" if summary['avg_speed'] else "  Avg Speed: N/A")
            print(f"  Speed Range: {summary['min_speed']:.2f} - {summary['max_speed']:.2f} it/s" if summary['min_speed'] else "  Speed Range: N/A")
            print(f"  Avg Loss: {summary['avg_loss']:.6f}" if summary['avg_loss'] else "  Avg Loss: N/A")
            print(f"  Loss Range: {summary['min_loss']:.6f} - {summary['max_loss']:.6f}" if summary['min_loss'] else "  Loss Range: N/A")


if __name__ == "__main__":
    print("[START] Training Metrics Tracker")
    metrics = parse_training_log()
    save_metrics(metrics)
    print("[DONE] Metrics saved")
