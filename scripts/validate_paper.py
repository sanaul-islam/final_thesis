"""Validate the thesis LaTeX before Overleaf compilation."""
import re
import sys
from pathlib import Path

paper = Path("paper/main.tex")
bib = Path("paper/references.bib")
figdir = Path("outputs/figures/paper")
tex = paper.read_text(encoding="utf-8")
bibtext = bib.read_text(encoding="utf-8")

ok = True

# 1. citations resolve
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bibtext))
cited = set()
for m in re.findall(r"\\cite\{([^}]+)\}", tex):
    for k in m.split(","):
        cited.add(k.strip())
missing = sorted(cited - bibkeys)
unused = sorted(bibkeys - cited)
print(f"[cite] {len(cited)} cited, {len(bibkeys)} in bib")
if missing:
    ok = False
    print("  ! MISSING bib entries:", missing)
if unused:
    print("  (unused bib entries, harmless):", unused)

# 2. ref/label
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
dangling = sorted(refs - labels)
print(f"[ref] {len(labels)} labels, {len(refs)} refs")
if dangling:
    ok = False
    print("  ! DANGLING refs:", dangling)

# 3. begin/end balance
begins = re.findall(r"\\begin\{([^}]+)\}", tex)
ends = re.findall(r"\\end\{([^}]+)\}", tex)
print(f"[env] {len(begins)} begin, {len(ends)} end")
if len(begins) != len(ends):
    ok = False
    print("  ! begin/end count mismatch")
from collections import Counter
cb, ce = Counter(begins), Counter(ends)
for env in set(begins) | set(ends):
    if cb[env] != ce[env]:
        ok = False
        print(f"  ! unbalanced {env}: {cb[env]} begin / {ce[env]} end")

# 4. includegraphics files exist
imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
print(f"[fig] {len(imgs)} includegraphics")
for img in imgs:
    name = img.strip()
    pdf = (figdir / (name + ".pdf")).exists()
    png = (figdir / (name + ".png")).exists()
    if not (pdf or png):
        ok = False
        print(f"  ! MISSING figure: {name}")

# 5. brace balance (rough)
nopen = tex.count("{")
nclose = tex.count("}")
print(f"[brace] {{={nopen} }}={nclose}")
if nopen != nclose:
    ok = False
    print("  ! brace mismatch")

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
