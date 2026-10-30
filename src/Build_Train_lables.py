r"""
Build_Train_Labels.py
Combines confident renames + hand-checked review rows into the final
training positives. Caps bulk-rename commits so one commit can't dominate.
Run: python Src\Build_Train_Labels.py
"""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
INTERIM = PROJECT / "data" / "interim"
MAX_PER_COMMIT = 5

conf = pd.read_csv(INTERIM / "confident_renames.csv")
conf["label_source"] = "auto_confident"

rev = pd.read_excel(INTERIM / "review_sheet_done.xlsx")
rev["is_rename"] = rev["is_rename"].astype(str).str.strip().str.upper()
left = rev[~rev["is_rename"].isin(["Y", "N"])]
if len(left):
    print(f"WARNING: {len(left)} rows still not Y/N (they are skipped):")
    print(left[["old_name", "new_name", "is_rename"]].to_string())
rev_yes = rev[rev["is_rename"] == "Y"].copy()
rev_yes["label_source"] = "hand_checked"

cols = ["commit", "file", "class", "old_name", "new_name", "similarity", "label_source"]
pos = pd.concat([conf[cols], rev_yes[cols]], ignore_index=True)
pos["class"] = pos["class"].fillna("")
pos = pos.drop_duplicates(subset=["commit", "file", "class", "old_name"])
print("Renames before cap:", len(pos))

# Cap bulk-rename commits (e.g. the host.py commit with 40+ renames)
pos = (pos.sample(frac=1, random_state=42)
          .groupby("commit", group_keys=False)
          .head(MAX_PER_COMMIT)
          .sort_values(["commit", "file", "old_name"]))
pos["renamed"] = 1

out = INTERIM / "train_positives.csv"
pos.to_csv(out, index=False)

print("Renames after cap: ", len(pos))
print("Commits:           ", pos["commit"].nunique())
print(pos["label_source"].value_counts().to_string())
print("Saved:", out)