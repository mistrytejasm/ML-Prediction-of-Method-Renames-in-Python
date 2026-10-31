# ARCHIVED 2026-10-31: moved from src/ to archive/legacy_scripts/ and renamed. Only file paths were
# updated for the new folder layout; the logic is unchanged. See reports/audit/.
"""
Make_Review.py
Splits auto-detected renames into 'confident' and 'review', drops test methods,
and writes a review sheet with the old and new code side by side.
Run: python Src\Make_Review_Sheet.py
"""
from pathlib import Path
import ast
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
COMMITS = PROJECT / "data" / "raw" / "actref" / "commits"
INTERIM = PROJECT / "data" / "interim"

df = pd.read_csv(INTERIM / "rename_candidates" / "detected_rename_candidates.csv")
df["class"] = df["class"].fillna("")

# 1. Drop test methods
df = df[~df["old_name"].str.startswith("test")].copy()

# 2. Confident vs review
df["confident"] = (df["similarity"] >= 0.9) & (df["same_args"])


def method_source(commit, file, cls, name, side):
    """Get the source code of one method from the Before/After file."""
    path = COMMITS / commit / side / file.replace("/", "#")
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(text)
    except Exception:
        return ""
    target = cls.split(".") if cls else []

    def find(node, path_left):
        for child in ast.iter_child_nodes(node):
            if path_left and isinstance(child, ast.ClassDef) and child.name == path_left[0]:
                r = find(child, path_left[1:])
                if r:
                    return r
            if not path_left and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and child.name == name:
                return ast.get_source_segment(text, child) or ""
        return ""

    code = find(tree, target)
    return code[:1500]   # keep cells readable


review = df[~df["confident"]].copy()
review["old_code"] = [method_source(r.commit, r.file, r["class"], r.old_name, "Before")
                      for _, r in review.iterrows()]
review["new_code"] = [method_source(r.commit, r.file, r["class"], r.new_name, "After")
                      for _, r in review.iterrows()]
review["is_rename"] = ""    # fill in: Y or N
review["note"] = ""

cols = ["commit", "file", "class", "old_name", "new_name", "similarity",
        "same_args", "old_code", "new_code", "is_rename", "note"]
review[cols].to_excel(INTERIM / "label_review" / "uncertain_candidates_review_blank.xlsx", index=False)
df[df["confident"]].to_csv(INTERIM / "rename_candidates" / "auto_accepted_rename_candidates.csv", index=False)

print("Non-test renames:", len(df))
print("Confident (auto-accepted):", int(df["confident"].sum()))
print("To review by hand:", len(review))
print("Saved: data/interim/review_sheet.xlsx and confident_renames.csv")