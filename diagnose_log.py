#!/usr/bin/env python3
"""
Diagnostic script to understand log file format
"""

from pathlib import Path
import re

LOG_FILE = Path("training_log.txt")

print(f"Reading {LOG_FILE}...")
with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

print(f"File size: {len(content)} bytes")
print(f"'WM' count: {content.count('WM')}")
print(f"'Ens' count: {content.count('Ens')}")
print(f"'it/s' count: {content.count('it/s')}")

# Find lines with WM or Ens and speed
lines = content.split('\n')
speed_lines = [l for l in lines if ('it/s' in l) and ('WM' in l or 'Ens' in l)]
print(f"\nTotal lines with WM/Ens + 'it/s': {len(speed_lines)}")

print("\nFirst 5 lines:")
for i, line in enumerate(speed_lines[:5]):
    print(f"{i}: {repr(line[:100])}")

print("\nLast 5 lines:")
for i, line in enumerate(speed_lines[-5:]):
    print(f"{i}: {repr(line[:100])}")

# Try different patterns
print("\n\nTrying different patterns:")

patterns = [
    r'(WM|Ens)\s+(\d+)/(\d+).*?(\d+\.\d+)it/s',  # Simple
    r'(WM|Ens)\s+(\d+)/(\d+):.*?(\d+\.\d+)it/s',  # With colon
    r'(WM|Ens)\s+(\d+)/(\d+):.*\|.*\|\s+(\d+)/(\d+).*?(\d+\.\d+)it/s',  # Full
    r'(WM|Ens).*?(\d+\.\d+)it/s',  # Very loose
]

for pattern_idx, pattern in enumerate(patterns):
    matches = [re.search(pattern, l) for l in speed_lines[:3]]
    print(f"\nPattern {pattern_idx}: {pattern[:50]}...")
    print(f"  Matches: {sum(1 for m in matches if m)}")
    for i, m in enumerate(matches):
        if m:
            print(f"    Line {i}: groups={m.groups()}")
