#!/usr/bin/env python3
import re
from pathlib import Path

LOG_FILE = Path("training_log.txt")

with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")

# Check if epoch strings exist
if "epoch" in content:
    print("Found 'epoch' in file")
else:
    print("'epoch' NOT found in file")

# Try to find patterns
lines = content.split('\n')
print(f"Total lines: {len(lines)}")

# Look for WM epoch pattern
epoch_count = 0
for i, line in enumerate(lines):
    if " epoch " in line or "epoch " in line:
        epoch_count += 1
        print(f"Line {i}: {repr(line[:100])}")
        if epoch_count >= 5:
            break

print(f"\nTotal lines with 'epoch': {epoch_count}")

# Check for loss pattern
loss_count = sum(1 for line in lines if "loss=" in line)
print(f"Total lines with 'loss=': {loss_count}")

# Print a sample of lines
print("\nFirst 50 lines with 'WM' in them:")
wm_count = 0
for line in lines:
    if " WM " in line or line.strip().startswith("WM"):
        wm_count += 1
        print(f"{wm_count}: {repr(line[:80])}")
        if wm_count >= 10:
            break
