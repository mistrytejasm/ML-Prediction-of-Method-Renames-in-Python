# ARCHIVED 2026-10-31: moved from src/ to archive/legacy_scripts/ and renamed. Only file paths were
# updated for the new folder layout; the logic is unchanged. See reports/audit/.
"""
Label_Renames.py
Finds Rename Method refactorings in ActRef's Before/After code.

Rule: in the same file and same class, a method name disappears and a new
name appears whose body is very similar (>= SIM_THRESHOLD).

Output: data/interim/actref_auto_renames.csv
Run:    python Src\Label_Renames.py
"""
from pathlib import Path
import ast, difflib, re
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
COMMITS = PROJECT / "data" / "raw" / "actref" / "commits"
OUT     = PROJECT / "data" / "interim" / "rename_candidates" / "detected_rename_candidates.csv"
SIM_THRESHOLD = 0.70


def body_text(func):
    """Body source without the docstring, for comparing two methods."""
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    try:
        return "\n".join(ast.unparse(s) for s in body)
    except Exception:
        return ""


def collect_methods(tree):
    """Return {(class_path, name): info} for module functions and class methods."""
    out = {}

    def visit(node, class_path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_path}.{child.name}" if class_path else child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in child.args.args if a.arg not in ("self", "cls")]
                out[(class_path, child.name)] = {
                    "args": args,
                    "body": body_text(child),
                    "lineno": child.lineno,
                    "loc": (child.end_lineno or child.lineno) - child.lineno + 1,
                }

    visit(tree, "")
    return out


def parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return None


def is_dunder(name):
    return name.startswith("__") and name.endswith("__")


rows, files_seen, parse_errors = [], 0, 0

for commit_dir in sorted(d for d in COMMITS.iterdir() if d.is_dir()):
    before_dir, after_dir = commit_dir / "Before", commit_dir / "After"
    for bfile in before_dir.glob("*.py"):
        afile = after_dir / bfile.name
        if not afile.exists():
            continue
        files_seen += 1
        bt, at = parse(bfile), parse(afile)
        if bt is None or at is None:
            parse_errors += 1
            continue

        before, after = collect_methods(bt), collect_methods(at)
        removed = [k for k in before if k not in after and not is_dunder(k[1])]
        added   = [k for k in after if k not in before and not is_dunder(k[1])]

        # Score every (removed, added) pair in the same class
        pairs = []
        for r in removed:
            for a in added:
                if r[0] != a[0]:
                    continue
                sim = difflib.SequenceMatcher(None, before[r]["body"], after[a]["body"]).ratio()
                if sim >= SIM_THRESHOLD:
                    pairs.append((sim, r, a))

        # Greedy one-to-one matching, best similarity first
        used_r, used_a = set(), set()
        for sim, r, a in sorted(pairs, reverse=True):
            if r in used_r or a in used_a:
                continue
            used_r.add(r); used_a.add(a)
            rows.append({
                "commit": commit_dir.name,
                "file": bfile.name.replace("#", "/"),
                "class": r[0],
                "old_name": r[1],
                "new_name": a[1],
                "similarity": round(sim, 3),
                "same_args": before[r]["args"] == after[a]["args"],
                "old_lineno": before[r]["lineno"],
                "old_loc": before[r]["loc"],
            })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Files compared:        ", files_seen)
print("Files with parse errors:", parse_errors)
print("Renames found:          ", len(df))
if len(df):
    print("Commits with a rename:  ", df["commit"].nunique())
    print("Test-method renames:    ", int(df["old_name"].str.startswith("test").sum()))
    print("\nSample:")
    print(df.sample(min(10, len(df)), random_state=1)[
        ["commit", "file", "class", "old_name", "new_name", "similarity"]].to_string())
print("\nSaved:", OUT)


# ---------- Use 2: compare with PyRef's labels ----------
pyref_path = next((p for p in [PROJECT / "data" / "external" / "pyref" / "pyref_rename_method.csv",
                               PROJECT / "pyref_rename_method.csv"] if p.exists()), None)
if pyref_path is not None and len(df):
    py = pd.read_csv(pyref_path)
    py["hash"] = py["commit"].astype(str).str.rstrip("/").str.split("/").str[-1].str.lower()
    # description looks like: The method "JSD" is renamed to "jsd" ...
    names = py["description"].astype(str).str.extract(r'method\s+"([^"]+)"[^"]*"([^"]+)"')
    py["old_name"], py["new_name"] = names[0], names[1]

    act_hashes = {d.name.lower() for d in COMMITS.iterdir() if d.is_dir()}
    py_in = py[py["hash"].isin(act_hashes)]
    found = py_in.merge(df.assign(hash=df["commit"].str.lower()),
                        on=["hash", "old_name"], how="left", indicator=True)
    hit = (found["_merge"] == "both").groupby(found.index).any().sum() if len(found) else 0

    print("\n=== Check against PyRef ===")
    print("PyRef renames whose commit is in ActRef:", len(py_in))
    print("Of those, found by this labeler:        ", int(hit))
    if len(py_in):
        print(f"Recall vs PyRef: {hit / len(py_in):.1%}")