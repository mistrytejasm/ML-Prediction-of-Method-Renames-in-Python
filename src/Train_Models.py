r"""
Train_Models.py
Trains baselines, Logistic Regression, Random Forest and LightGBM on
data/processed/train_features.csv with 5-fold cross-validation.

Folds are grouped so that methods from the same PROJECT never appear in both
the training and the validation part of a fold (no leakage between folds).

Outputs (results/):
  cv_results.csv          mean and std of every metric per model
  cv_folds.csv            every fold's scores (for the Wilcoxon test later)
  feature_importance.csv  LightGBM and Random Forest importances
Run on your PC:  python Src\Train_Models.py
Run in Colab:    upload train_features.csv, run  !pip install -q lightgbm
                 in one cell, then paste this whole file into the next cell.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, matthews_corrcoef, roc_auc_score)

warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed (pip install lightgbm). Skipping it.")

# Works both on your PC (as a .py file) and in Google Colab (pasted into a cell)
try:
    PROJECT = Path(__file__).resolve().parent.parent          # running Src\Train_Models.py
    DATA    = PROJECT / "data" / "processed" / "train_features.csv"
    RESULTS = PROJECT / "results"
except NameError:                                              # running in Colab / Jupyter
    DATA    = Path("train_features.csv")                      # upload it to the Colab files panel
    RESULTS = Path("results")
if not DATA.exists():
    raise SystemExit(f"Can't find {DATA}. In Colab, upload train_features.csv first.")
RESULTS.mkdir(exist_ok=True)
SEED = 42

df = pd.read_csv(DATA)
ID_COLS = ["commit", "file", "class", "name", "renamed"]
FEATURES = [c for c in df.columns if c not in ID_COLS]
X = df[FEATURES].fillna(0).astype(float).values
y = df["renamed"].values


# ---------- Group by project (approximate: ActRef gives no repo names) ----------
GENERIC = {"src", "lib", "python", "tests", "test", "testing", "scripts", "examples", "codes", ""}


def project_guess(path):
    """First meaningful folder of the file path, e.g. 'lib/streamlit/x.py' -> 'streamlit'."""
    parts = str(path).split("/")[:-1]
    for p in parts:
        if p.lower() not in GENERIC:
            return p.lower()
    return str(path).split("/")[-1].lower()   # single-file repos: use the file name


# A commit belongs to one project, so give every commit one project label
commit_project = (df.assign(p=df["file"].map(project_guess))
                    .groupby("commit")["p"].agg(lambda s: s.value_counts().index[0]))
groups = df["commit"].map(commit_project).values
print(f"Rows: {len(df)}  |  renamed: {y.sum()}  |  features: {len(FEATURES)}")
print(f"Commits: {df['commit'].nunique()}  |  project groups (approx.): {len(set(groups))}\n")


# ---------- Simple rule baseline: 'the name looks bad' ----------
class BadNameRule:
    """Predict renamed if the name is camelCase in a snake_case file, has a
    vowel-less token, is a single letter, or shares no words with its body."""
    def fit(self, X, y):
        return self

    def _rule(self, X):
        f = {n: X[:, i] for i, n in enumerate(FEATURES)}
        return ((f["style_mismatch"] > 0) | (f["name_no_vowel_tokens"] > 0) |
                (f["name_single_char"] > 0) | (f["name_short_tokens"] > 0)).astype(int)

    def predict(self, X):
        return self._rule(X)

    def predict_proba(self, X):
        p = self._rule(X).astype(float)
        return np.c_[1 - p, p]


ratio = (y == 0).sum() / max(1, y.sum())
models = {
    "Majority (always 0)": DummyClassifier(strategy="most_frequent"),
    "Random (stratified)": DummyClassifier(strategy="stratified", random_state=SEED),
    "Bad-name rule":       BadNameRule(),
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=2000)),
    "Random Forest":       RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                  class_weight="balanced", n_jobs=-1,
                                                  random_state=SEED),
}
if HAS_LGBM:
    models["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                        min_child_samples=10, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, scale_pos_weight=ratio,
                                        random_state=SEED, verbose=-1)


def scores(y_true, y_pred, y_prob):
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
        "pr_auc":    average_precision_score(y_true, y_prob),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = np.nan
    return out


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_rows = []
importances = {"feature": FEATURES}

for name, model in models.items():
    imp_sum = np.zeros(len(FEATURES))
    for k, (tr, va) in enumerate(cv.split(X, y, groups)):
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[va])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fold_rows.append({"model": name, "fold": k, **scores(y[va], pred, prob)})
        if hasattr(model, "feature_importances_"):
            imp_sum += model.feature_importances_
    if imp_sum.any():
        importances[name] = imp_sum / imp_sum.sum()

folds = pd.DataFrame(fold_rows)
folds.to_csv(RESULTS / "cv_folds.csv", index=False)

metrics = ["precision", "recall", "f1", "mcc", "pr_auc", "roc_auc"]
summary = folds.groupby("model", sort=False)[metrics].agg(["mean", "std"]).round(3)
summary.to_csv(RESULTS / "cv_results.csv")

# Readable table: mean ± std
table = pd.DataFrame({m: summary[(m, "mean")].map("{:.3f}".format) + " ± " +
                         summary[(m, "std")].map("{:.3f}".format) for m in metrics})
print("=== 5-fold cross-validation (grouped by project) ===")
print(table.to_string())
print(f"\nReference: a random guess gets PR-AUC ≈ {y.mean():.3f} (share of renamed rows).")

imp = pd.DataFrame(importances)
sort_col = "LightGBM" if "LightGBM" in imp else "Random Forest"
imp = imp.sort_values(sort_col, ascending=False)
imp.to_csv(RESULTS / "feature_importance.csv", index=False)
print(f"\n=== Top 10 features ({sort_col}) ===")
print(imp.head(10).round(3).to_string(index=False))
print("\nSaved: results/cv_results.csv, cv_folds.csv, feature_importance.csv")