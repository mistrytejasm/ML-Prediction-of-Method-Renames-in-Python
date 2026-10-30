r"""
Ablation_v2.py  (for train_features_v2.csv)
Which feature groups really matter? Trains Logistic Regression, Random Forest
and LightGBM on different feature sets, with the same 5-fold grouped CV as
Train_Models.py.

Feature sets:
  1. All features
  2. Without file-level features (file_loc, file_n_methods)   <- the honest version
  3. Name features only
  4. Code features only
  5. Context features only

Output: results/ablation_v2_results.csv
Run on your PC:  python Src\Ablation.py
Run in Colab:    upload train_features_v2.csv, run  !pip install -q lightgbm
                 in one cell, then paste this whole file into the next cell.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef, recall_score, precision_score

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed. Skipping it.")

try:
    PROJECT = Path(__file__).resolve().parent.parent
    DATA    = PROJECT / "data" / "processed" / "train_features_v2.csv"
    RESULTS = PROJECT / "results"
except NameError:
    DATA    = Path("train_features_v2.csv")
    RESULTS = Path("results")
if not DATA.exists():
    raise SystemExit(f"Can't find {DATA}. In Colab, upload train_features_v2.csv first.")
RESULTS.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(DATA)
y = df["renamed"].values

# ---------- Feature groups ----------
NAME = ["name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit",
        "name_short_tokens", "name_no_vowel_tokens", "name_starts_verb", "name_single_char",
        "name_body_overlap", "style_mismatch"]
CODE = ["loc", "n_params", "n_defaults", "has_varargs", "has_kwargs", "n_returns", "n_calls",
        "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub", "is_async",
        "n_decorators", "is_property", "is_static_or_class"]
CONTEXT = ["in_class", "class_depth", "class_n_methods", "file_n_methods", "file_loc",
           "is_test_file", "call_sites"]
FILE_LEVEL = ["file_loc", "file_n_methods"]
RELATIVE = ["name_token_sharing", "first_token_shared", "style_share_in_file",
            "rel_name_len", "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls",
            "rel_cyclomatic", "rel_doc_len", "rel_call_sites", "rel_name_body_overlap"]
ALL = NAME + CODE + CONTEXT + RELATIVE

feature_sets = {
    "1. All features":              ALL,
    "2. Without file-level (old 34 minus 2)": [f for f in NAME + CODE + CONTEXT if f not in FILE_LEVEL],
    "3. Name only":                 NAME,
    "4. Code only":                 CODE,
    "5. Context only":              CONTEXT,
    "6. Relative-to-file only":     RELATIVE,
    "7. Method-level (2 + 6, no file size)": [f for f in ALL if f not in FILE_LEVEL],
}

# ---------- Same project grouping as Train_Models.py ----------
GENERIC = {"src", "lib", "python", "tests", "test", "testing", "scripts", "examples", "codes", ""}


def project_guess(path):
    parts = str(path).split("/")[:-1]
    for p in parts:
        if p.lower() not in GENERIC:
            return p.lower()
    return str(path).split("/")[-1].lower()


commit_project = (df.assign(p=df["file"].map(project_guess))
                    .groupby("commit")["p"].agg(lambda s: s.value_counts().index[0]))
groups = df["commit"].map(commit_project).values

ratio = (y == 0).sum() / max(1, y.sum())


def make_models():
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    if HAS_LGBM:
        m["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                       min_child_samples=10, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, scale_pos_weight=ratio,
                                       random_state=SEED, verbose=-1)
    return m


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
splits = list(cv.split(df, y, groups))      # same folds for every feature set
rows = []

for set_name, cols in feature_sets.items():
    X = df[cols].fillna(0).astype(float).values
    for model_name, model in make_models().items():
        pr, f1, mcc, rec, prec = [], [], [], [], []
        for tr, va in splits:
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[va])[:, 1]
            pred = (prob >= 0.5).astype(int)
            pr.append(average_precision_score(y[va], prob))
            f1.append(f1_score(y[va], pred, zero_division=0))
            mcc.append(matthews_corrcoef(y[va], pred))
            rec.append(recall_score(y[va], pred, zero_division=0))
            prec.append(precision_score(y[va], pred, zero_division=0))
        rows.append({
            "feature_set": set_name, "n_features": len(cols), "model": model_name,
            "pr_auc": np.mean(pr), "pr_auc_std": np.std(pr),
            "f1": np.mean(f1), "mcc": np.mean(mcc),
            "precision": np.mean(prec), "recall": np.mean(rec),
        })
        print(f"done: {set_name:24} {model_name}")

res = pd.DataFrame(rows).round(3)
res.to_csv(RESULTS / "ablation_v2_results.csv", index=False)

print("\n=== PR-AUC by feature set (random guess ≈ {:.3f}) ===".format(y.mean()))
print(res.pivot(index="feature_set", columns="model", values="pr_auc").to_string())
print("\n=== F1 by feature set ===")
print(res.pivot(index="feature_set", columns="model", values="f1").to_string())
print("\nSaved: results/ablation_v2_results.csv")
