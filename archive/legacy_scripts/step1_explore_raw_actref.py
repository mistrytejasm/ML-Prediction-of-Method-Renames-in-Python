# ARCHIVED 2026-10-31: moved from src/ to archive/legacy_scripts/ and renamed. Only file paths were
# updated for the new folder layout; the logic is unchanged. See reports/audit/.
"""
Act_Explore.py
Looks inside the ActRef download (data/raw_act) and reports:
  1. Which file types are there
  2. Any files outside the Before/After folders (possible label files)
  3. Any CSV / JSON / Excel / TXT / MD files (possible label files)
  4. How many commit folders and Python files there are
  5. A peek at one commit folder

Run from anywhere:
    python Src\Act_Explore.py
"""

from pathlib import Path
from collections import Counter

# ---------- Paths (work no matter where you run the script from) ----------
PROJECT = Path(__file__).resolve().parents[2]      # D:\Refactoring_Project
ROOT = PROJECT / "data" / "raw" / "actref"

print("Looking in:", ROOT)
print("Exists:", ROOT.exists())
if not ROOT.exists():
    raise SystemExit("data/raw/actref not found. Check the folder name and location.")

# Find the folder that holds the commit folders.
# It may be raw_act/commits or raw_act/commits/commits (zip unpacked twice).
commits_dir = ROOT / "commits"
if (commits_dir / "commits").exists():
    commits_dir = commits_dir / "commits"
if not commits_dir.exists():
    commits_dir = ROOT          # commit folders sit directly in raw_act
print("Commit folders are in:", commits_dir)

all_files = [f for f in ROOT.rglob("*") if f.is_file()]

# ---------- 1. File types ----------
print("\n=== 1. File types ===")
types = Counter(f.suffix.lower() or "(no extension)" for f in all_files)
for ext, n in types.most_common():
    print(f"  {ext:15} {n}")

# ---------- 2. Files outside Before/After ----------
print("\n=== 2. Files outside Before/After folders (possible labels) ===")
outside = [
    f for f in all_files
    if "Before" not in f.parts and "After" not in f.parts and f.name != ".DS_Store"
]
if outside:
    for f in outside:
        print(f"  {f.relative_to(ROOT)}   {f.stat().st_size // 1024} KB")
else:
    print("  (none found)")

# ---------- 3. Label-like files anywhere ----------
print("\n=== 3. CSV / JSON / Excel / TXT / MD files anywhere ===")
label_ext = {".csv", ".json", ".xlsx", ".xls", ".txt", ".md"}
label_files = [f for f in all_files if f.suffix.lower() in label_ext]
if label_files:
    for f in label_files[:50]:
        print(f"  {f.relative_to(ROOT)}   {f.stat().st_size // 1024} KB")
    if len(label_files) > 50:
        print(f"  ... and {len(label_files) - 50} more")
else:
    print("  (none found)")

# ---------- 4. Counts ----------
print("\n=== 4. Counts ===")
commits = [d for d in commits_dir.iterdir() if d.is_dir()]
print("  Commit folders:", len(commits))

before_py = after_py = no_before = no_after = 0
for d in commits:
    b, a = d / "Before", d / "After"
    if b.exists():
        before_py += len(list(b.glob("*.py")))
    else:
        no_before += 1
    if a.exists():
        after_py += len(list(a.glob("*.py")))
    else:
        no_after += 1

print("  Python files in Before folders:", before_py)
print("  Python files in After folders: ", after_py)
print("  Commits missing a Before folder:", no_before)
print("  Commits missing an After folder:", no_after)

# ---------- 5. Peek at one commit ----------
print("\n=== 5. Example commit ===")
if commits:
    ex = commits[0]
    print("  Commit:", ex.name)
    for sub in sorted(ex.iterdir()):
        print("   ", sub.name + ("/" if sub.is_dir() else ""))
        if sub.is_dir():
            for f in sorted(sub.iterdir())[:5]:
                real_path = f.name.replace("#", "/")
                print(f"      {f.name}   ->   {real_path}")