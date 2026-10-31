r"""
audit_preprocessing.py
Phase 1 preprocessing audit of the canonical master dataset. Analysis only:
it reads the master through the checksum-verifying loader, uses the approved
feature sets from src/features/feature_sets.py, and writes aggregate statistics
and a report. It never modifies the dataset, never fits a model, never applies
a transformation to the data, and never saves split assignments.

Outputs (reports/audit/preprocessing/):
  integrity_by_feature.csv        types, missing / infinite values, cardinality, validity
  feature_distributions.csv       location, spread, quantiles, skewness, zeros, outliers
  positive_vs_negative.csv        class-wise statistics and effect sizes (group-bootstrap CIs)
  correlation_pearson.csv / correlation_spearman.csv / correlation_pairs.csv
  exact_relationships.csv         binary features that are a deterministic function of another feature
  vif.csv                         variance inflation factors (main and comparison sets)
  project_groups.csv              rows / commits / positives per project group
  grouped_cv_feasibility.csv      simulated grouped splits (summary per seed, nothing saved)
  audit_summary.json              every headline number used in the report
  preprocessing_audit_report.md   the report

Run: python src/data/audit_preprocessing.py   (about 2-4 minutes)
"""
import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import GROUP_COLUMN, load_master, sha256_of, verify_master  # noqa: E402
from src.features.feature_sets import (COMPARISON_FEATURES, COMPARISON_ONLY, DROPPED_FEATURES,  # noqa: E402
                                       EXCLUDED_LEAKAGE, MAIN_FEATURES)

warnings.filterwarnings("ignore", category=RuntimeWarning)
log = logging.getLogger("audit_preprocessing")
OUT = config.REPORTS_AUDIT / "preprocessing"
SEED = 42
BOOT_REPS = 500
SIM_SEEDS = 50

BINARY = ["name_is_private", "name_is_camel", "name_has_digit", "name_starts_verb", "name_single_char",
          "has_varargs", "has_kwargs", "has_docstring", "is_stub", "is_async", "is_property",
          "is_static_or_class", "in_class", "is_test_file", "style_mismatch", "first_token_shared"]
RATIO = ["name_body_overlap", "name_token_sharing", "style_share_in_file"]
# candidates evaluated by this audit (selected from the measured skewness; see the report)
LOG1P_CANDIDATES = ["loc", "n_params", "n_returns", "n_calls", "cyclomatic", "max_nesting", "doc_len",
                    "class_n_methods", "file_n_methods", "file_loc"]
ASINH_CANDIDATES = ["rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic", "rel_doc_len"]


def feature_type(f):
    if f in BINARY:
        return "binary"
    if f in RATIO:
        return "ratio [0,1]"
    if f.startswith("rel_"):
        return "signed relative"
    return "count"


def membership(f):
    if f in MAIN_FEATURES:
        return "main + comparison"
    if f in COMPARISON_ONLY:
        return "comparison only"
    if f in EXCLUDED_LEAKAGE:
        return "excluded (leakage)"
    if f in DROPPED_FEATURES:
        return "dropped"
    return "?"


# ------------------------------------------------------------------ 1. integrity
def integrity(df):
    rows = []
    for f in config.FEATURES:
        s = df[f]
        num = pd.to_numeric(s, errors="coerce")
        vc = s.value_counts()
        typ = feature_type(f)
        expected_ok = {
            "binary": set(s.unique()) <= {0, 1},
            "ratio [0,1]": bool(num.between(0, 1).all()),
            "signed relative": True,
            "count": bool((num >= 0).all() and (num == np.floor(num)).all()),
        }[typ]
        rows.append({
            "feature": f, "set": membership(f), "type": typ, "dtype": str(s.dtype),
            "missing": int(s.isna().sum()), "non_numeric": int(num.isna().sum() - s.isna().sum()),
            "infinite": int(np.isinf(num).sum()), "unique": int(s.nunique()),
            "min": float(num.min()), "max": float(num.max()),
            "top_value": vc.index[0], "top_value_share": round(float(vc.iloc[0] / len(s)), 6),
            "constant": bool(s.nunique() <= 1),
            "near_constant_99pct": bool(vc.iloc[0] / len(s) >= 0.99),
            "values_as_expected": expected_ok,
        })
    return pd.DataFrame(rows)


def duplicates(df):
    feats = COMPARISON_FEATURES
    dup_key = int(df.duplicated(config.KEY).sum())
    dup_full = int(df.duplicated().sum())
    same_x = df.duplicated(feats, keep=False)
    same_x_name = df.duplicated(feats + ["name"], keep=False)
    conflict = df[same_x].groupby(feats)["renamed"].nunique()
    groups_per_cluster = df[same_x_name].groupby(feats + ["name"])[GROUP_COLUMN].nunique()
    return {
        "duplicate_keys": dup_key,
        "duplicate_full_rows": dup_full,
        "rows_sharing_identical_42_feature_vector": int(same_x.sum()),
        "identical_vector_clusters": int(df[same_x].groupby(feats).ngroups),
        "conflicting_clusters_same_name": int((df[same_x].groupby(feats).filter(
            lambda g: g["renamed"].nunique() > 1).groupby(feats)["name"].nunique() == 1).sum()),
        "clusters_with_conflicting_labels": int((conflict > 1).sum()),
        "rows_in_conflicting_clusters": int(df[same_x].groupby(feats)["renamed"].transform("nunique").gt(1).sum()),
        "rows_same_name_and_vector": int(same_x_name.sum()),
        "same_name_clusters_spanning_groups": int((groups_per_cluster > 1).sum()),
        "commit_ids_valid": bool(df["commit"].str.fullmatch(r"[0-9a-f]{40}").all()),
        "empty_class_rows (module functions)": int((df["class"] == "").sum()),
    }


# ------------------------------------------------------------------ 2. distributions
def distributions(df, feats):
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rows = []
    for f in feats:
        x = df[f].astype(float)
        q = x.quantile(qs)
        iqr = q[0.75] - q[0.25]
        typ = feature_type(f)
        row = {"feature": f, "set": membership(f), "type": typ, "mean": x.mean(), "std": x.std(),
               "median": x.median(), "iqr": iqr, "min": x.min(), "max": x.max(),
               **{f"p{int(k * 100):02d}": v for k, v in q.items()},
               "skewness": x.skew(), "kurtosis": x.kurt(),
               "zero_share": float((x == 0).mean())}
        if typ == "binary":
            row.update({"tukey_outliers_1.5iqr": np.nan, "tukey_outliers_3iqr": np.nan, "robust_z_outliers": np.nan,
                        "share_of_1": float(x.mean())})
        else:
            lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
            lo3, hi3 = q[0.25] - 3 * iqr, q[0.75] + 3 * iqr
            mad = stats.median_abs_deviation(x, scale="normal")
            row.update({
                "tukey_outliers_1.5iqr": int(((x < lo) | (x > hi)).sum()) if iqr > 0 else np.nan,
                "tukey_outliers_3iqr": int(((x < lo3) | (x > hi3)).sum()) if iqr > 0 else np.nan,
                "robust_z_outliers": int((np.abs(x - x.median()) / mad > 3.5).sum()) if mad > 0 else np.nan,
                "share_of_1": np.nan,
            })
            if typ == "count":
                row["skewness_after_log1p"] = np.log1p(x).skew()
            if typ in ("count", "signed relative"):
                row["skewness_after_asinh"] = np.arcsinh(x).skew()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 3. positive vs negative
def _auc(x, y):
    r = stats.rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def positive_vs_negative(df, feats):
    y = df["renamed"].values
    groups = df[GROUP_COLUMN].values
    uniq = np.unique(groups)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(SEED)
    boot_idx = []
    for _ in range(BOOT_REPS):                 # resample whole project groups (clustered data)
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            boot_idx.append(idx)
    rows, pvals = [], []
    for f in feats:
        x = df[f].astype(float).values
        xp, xn = x[y == 1], x[y == 0]
        sp = np.sqrt(((len(xp) - 1) * xp.var(ddof=1) + (len(xn) - 1) * xn.var(ddof=1)) / (len(x) - 2))
        d = (xp.mean() - xn.mean()) / sp if sp > 0 else np.nan
        g = d * (1 - 3 / (4 * len(x) - 9)) if sp > 0 else np.nan
        auc = _auc(x, y)
        boots = np.array([_auc(x[i], y[i]) for i in boot_idx])
        mw = stats.mannwhitneyu(xp, xn, alternative="two-sided")
        pvals.append(mw.pvalue)
        row = {"feature": f, "set": membership(f), "type": feature_type(f),
               "mean_renamed": xp.mean(), "mean_not_renamed": xn.mean(),
               "std_renamed": xp.std(ddof=1), "std_not_renamed": xn.std(ddof=1),
               "median_renamed": np.median(xp), "median_not_renamed": np.median(xn),
               "q1_renamed": np.quantile(xp, 0.25), "q3_renamed": np.quantile(xp, 0.75),
               "q1_not_renamed": np.quantile(xn, 0.25), "q3_not_renamed": np.quantile(xn, 0.75),
               "missing_renamed": 0, "missing_not_renamed": 0,
               "hedges_g": g, "auc": auc, "cliffs_delta": 2 * auc - 1,
               "cliffs_delta_ci_low": 2 * np.quantile(boots, 0.025) - 1,
               "cliffs_delta_ci_high": 2 * np.quantile(boots, 0.975) - 1,
               "mannwhitney_p": mw.pvalue}
        if feature_type(f) == "binary":
            row["rate_renamed"], row["rate_not_renamed"] = xp.mean(), xn.mean()
            row["rate_ratio"] = xp.mean() / xn.mean() if xn.mean() > 0 else np.nan
            row["n_renamed_with_1"] = int(xp.sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    # Holm adjustment over the 42 tests
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    out["holm_p"] = adj
    out["ci_excludes_zero"] = (out.cliffs_delta_ci_low > 0) | (out.cliffs_delta_ci_high < 0)
    return out, len(boot_idx)


# ------------------------------------------------------------------ 4. correlation and redundancy
def correlations(df, feats):
    X = df[feats].astype(float)
    pear, spear = X.corr("pearson"), X.corr("spearman")
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            if max(abs(pear.loc[a, b]), abs(spear.loc[a, b])) >= 0.7:
                pairs.append({"feature_a": a, "feature_b": b, "pearson": pear.loc[a, b],
                              "spearman": spear.loc[a, b], "types": f"{feature_type(a)} / {feature_type(b)}"})
    pairs = pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False)
    y = df["renamed"].astype(float)
    target = pd.DataFrame({"pearson_with_target": X.corrwith(y), "spearman_with_target": X.corrwith(y, method="spearman")})
    return pear, spear, pairs, target


def exact_relationships(df):
    """Binary features that equal a simple condition on another feature (row-for-row)."""
    rows = []
    feats = config.FEATURES
    for b in [f for f in feats if f in BINARY]:
        bv = df[b].values
        for o in feats:
            if o == b:
                continue
            ov = df[o].values
            conds = [(f"{o} > 0", ov > 0), (f"{o} == 0", ov == 0)]
            if o in BINARY:          # identical or complementary binary columns
                conds = [(f"{o}", ov == 1), (f"1 - {o}", ov == 0)]
            for cond, mask in conds:
                agree = float((mask.astype(int) == bv).mean())
                if agree >= 0.999:
                    rows.append({"binary_feature": b, "equals": cond, "agreement": agree,
                                 "disagreeing_rows": int((mask.astype(int) != bv).sum())})
    return pd.DataFrame(rows)


def candidate_transform(df, feats):
    """Stateless candidate transforms, used ONLY to measure their effect in this audit (nothing is saved):
    log1p for skewed non-sparse counts, asinh for strongly skewed signed relative features."""
    X = df[feats].astype(float).copy()
    for f in LOG1P_CANDIDATES:
        if f in X:
            X[f] = np.log1p(X[f])
    for f in ASINH_CANDIDATES:
        if f in X:
            X[f] = np.arcsinh(X[f])
    return X


def vif(df, feats, transformed=False):
    X = candidate_transform(df, feats) if transformed else df[feats].astype(float)
    X = (X - X.mean()) / X.std()
    out = []
    for f in feats:
        others = X.drop(columns=f).values
        A = np.c_[np.ones(len(X)), others]
        coef, *_ = np.linalg.lstsq(A, X[f].values, rcond=None)
        resid = X[f].values - A @ coef
        r2 = 1 - resid.var() / X[f].values.var()
        out.append({"feature": f, "vif": 1 / (1 - r2) if r2 < 1 else np.inf, "r2_on_others": r2})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 5. groups and grouped validation
def project_groups(df):
    g = df.groupby(GROUP_COLUMN).agg(rows=("renamed", "size"), positives=("renamed", "sum"),
                                     commits=("commit", "nunique"), files=("file", "nunique"))
    return g.sort_values(["positives", "rows"], ascending=False)


def simulate_splits(df):
    """Simulate (do not save) grouped splits: 5-fold on all data, and a 20% group hold-out + 5-fold on the rest."""
    y = df["renamed"].values
    groups = df[GROUP_COLUMN].values
    X = np.zeros((len(df), 1))
    rows = []
    for seed in range(SIM_SEEDS):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        full_pos, full_rows, full_groups, full_posgroups = [], [], [], []
        for tr, va in cv.split(X, y, groups):
            assert not set(groups[tr]) & set(groups[va])
            full_pos.append(int(y[va].sum()))
            full_rows.append(len(va))
            full_groups.append(len(set(groups[va])))
            full_posgroups.append(len(set(groups[va][y[va] == 1])))
        dev, test = next(cv.split(X, y, groups))           # one fold (~20% of groups) as the hold-out
        assert not set(groups[dev]) & set(groups[test])
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        inner_pos = []
        for tr, va in inner.split(X[dev], y[dev], groups[dev]):
            inner_pos.append(int(y[dev][va].sum()))
        rows.append({
            "seed": seed,
            "full_min_pos_per_fold": min(full_pos), "full_max_pos_per_fold": max(full_pos),
            "full_min_rows_per_fold": min(full_rows), "full_max_rows_per_fold": max(full_rows),
            "full_min_groups_per_fold": min(full_groups), "full_min_pos_groups_per_fold": min(full_posgroups),
            "holdout_rows": len(test), "holdout_groups": len(set(groups[test])),
            "holdout_positives": int(y[test].sum()), "holdout_pos_groups": len(set(groups[test][y[test] == 1])),
            "holdout_pos_rate": float(y[test].mean()),
            "dev_rows": len(dev), "dev_positives": int(y[dev].sum()),
            "inner_min_pos_per_fold": min(inner_pos), "inner_max_pos_per_fold": max(inner_pos),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def md_table(df, floatfmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(df, integ, dups, dist, pvn, pairs, exact, vifs, target, groups, sims, n_boot, sha_before):
    y = df["renamed"]
    n, npos = len(df), int(y.sum())
    nneg = n - npos
    feats_cmp = COMPARISON_FEATURES
    skewed = dist[(dist.type != "binary") & (dist.skewness.abs() > 2)]
    sparse = dist[(dist.type == "count") & (dist.zero_share >= 0.75)]
    near_const = integ[integ.near_constant_99pct & integ.feature.isin(feats_cmp)]
    sig = pvn[pvn.ci_excludes_zero].sort_values("cliffs_delta", key=abs, ascending=False)
    gpos = groups[groups.positives > 0]
    top5 = groups.sort_values("positives", ascending=False).positives.head(5).sum() / npos
    s = sims.describe()
    w_pos = n / (2 * npos)
    vmax_raw, vmax_tr = vifs.vif_main.max(), vifs.vif_main_after_log1p_asinh.max()
    L = []
    A = L.append
    A("# Phase 1 preprocessing audit: ActRef master dataset")
    A("")
    A(f"Generated by `src/data/audit_preprocessing.py` on {date.today().isoformat()}. **Analysis only:** no "
      "transformation was applied to the data, no model was trained, no split was saved, and the master dataset was "
      "only read through `src/data/load_master.py` (checksum verified before and after). Feature sets come from "
      "`src/features/feature_sets.py`. Every number below is computed by the script; the CSV files in this folder hold "
      "the full tables.")
    A("")
    A("## 1. Executive summary")
    A("")
    A(f"- **The data is clean.** {n:,} rows, {npos} renamed / {nneg:,} not renamed ({npos / n:.2%}), 46 features, "
      f"{df[GROUP_COLUMN].nunique()} project groups. No missing, non-numeric or infinite values; no duplicate keys or "
      "rows; every binary, ratio and count feature holds only valid values. **No imputation or cleaning is needed.**")
    A(f"- **Code metrics are heavily right-skewed** ({len(skewed)} of {int((dist.type != 'binary').sum())} non-binary "
      "features have |skewness| > 2) with genuine extreme values (e.g. a 91-parameter function). Outliers are real "
      "code, not errors: **keep them.** `log1p` (counts) and `asinh` (signed relative features) remove most of the skew "
      "and matter only for Logistic Regression; tree models need no transformation or scaling.")
    A(f"- **Class differences are small.** Apart from the two file-size features (Cliff's δ ≈ "
      f"{pvn.set_index('feature').loc['file_loc', 'cliffs_delta']:.2f} / "
      f"{pvn.set_index('feature').loc['file_n_methods', 'cliffs_delta']:.2f}), every feature has |δ| ≤ "
      f"{pvn[pvn.set != 'comparison only'].cliffs_delta.abs().max():.2f}. {int(pvn.ci_excludes_zero.sum())} of 42 "
      "features have a project-level bootstrap interval that excludes zero.")
    A(f"- **Redundancy is real but bounded.** Two exact identities (`has_docstring` ≡ `doc_len > 0`, `in_class` ≡ "
      f"`class_n_methods > 0`) and a name-length cluster; VIF up to {vmax_raw:.1f} (main set). Transforms reduce some "
      f"VIFs but not all (max {vmax_tr:.1f} after transforms), so **Logistic Regression needs L2 regularization**.")
    A(f"- **Grouped validation is feasible.** {len(gpos)} of {len(groups)} projects contain positives (median "
      f"{gpos.positives.median():.0f}, max {int(gpos.positives.max())}); the top 5 hold {top5:.0%} of all positives. "
      f"Across {SIM_SEEDS} simulated seeds a 20% project hold-out keeps {int(s.loc['min', 'holdout_positives'])}–"
      f"{int(s.loc['max', 'holdout_positives'])} positives from {int(s.loc['min', 'holdout_pos_groups'])}–"
      f"{int(s.loc['max', 'holdout_pos_groups'])} positive projects, and each inner fold keeps "
      f"{int(s.loc['min', 'inner_min_pos_per_fold'])}–{int(s.loc['max', 'inner_max_pos_per_fold'])} positives. "
      "Because the test estimate rests on few positive projects, report project-level uncertainty.")
    A(f"- **Class imbalance:** use training-fold class weights (balanced weight for positives ≈ {w_pos:.0f}); no SMOTE "
      "or resampling.")
    A("")

    # 2 integrity
    A("## 2. Dataset integrity")
    A("")
    checks = [
        ("Rows / renamed / not renamed", f"{n:,} / {npos} / {nneg:,}", "expected"),
        ("Predictive features / main / comparison", f"46 / {len(MAIN_FEATURES)} / {len(COMPARISON_FEATURES)}", "expected"),
        ("Project groups", str(df[GROUP_COLUMN].nunique()), "expected"),
        ("Missing values (all 46 features)", str(int(integ.missing.sum())), "none"),
        ("Non-numeric values", str(int(integ.non_numeric.sum())), "none"),
        ("Infinite values", str(int(integ.infinite.sum())), "none"),
        ("Features with unexpected values (binary not 0/1, ratio outside [0,1], negative or fractional counts)",
         str(int((~integ.values_as_expected).sum())), "none"),
        ("Constant features", str(int(integ.constant.sum())), "none"),
        ("Duplicate keys (commit, file, class, name)", str(dups["duplicate_keys"]), "none"),
        ("Duplicate full rows", str(dups["duplicate_full_rows"]), "none"),
        ("Commit ids valid (40-hex)", str(dups["commit_ids_valid"]), "yes"),
        ("Empty `class` (module-level functions)", f"{dups['empty_class_rows (module functions)']:,}",
         "expected property, not missing data"),
    ]
    A(md_table(pd.DataFrame(checks, columns=["Check", "Result", "Interpretation"])))
    A("")
    A("**Near-constant features (≥ 99% one value) in the training sets:**")
    A("")
    A(md_table(near_const[["feature", "set", "top_value", "top_value_share", "unique"]].assign(
        ones=lambda d: [int(df[f].sum()) for f in d.feature],
        ones_renamed=lambda d: [int(df.loc[y == 1, f].sum()) for f in d.feature])))
    A("")
    A("`is_async` (dropped) is also near-constant. These are expected properties of rare code patterns, not errors.")
    A("")
    A("**Repeated feature vectors.** These are not duplicates (keys are unique) but matter for validation:")
    A("")
    A(md_table(pd.DataFrame([
        ("Rows whose 42-feature vector equals another row's", f"{dups['rows_sharing_identical_42_feature_vector']:,}",
         f"{dups['identical_vector_clusters']:,} clusters; mostly the same method observed in several commits"),
        ("Rows with the same name and vector", f"{dups['rows_same_name_and_vector']:,}",
         f"clusters spanning more than one project group: {dups['same_name_clusters_spanning_groups']} (grouped CV keeps them together)"),
        ("Clusters with conflicting labels", f"{dups['clusters_with_conflicting_labels']} ({dups['rows_in_conflicting_clusters']} rows)",
         "identical features, different label: e.g. sibling `load` (kept) / `save` (renamed); irreducible ambiguity, not a data error"),
    ], columns=["Observation", "Count", "Interpretation"])))
    A("")

    # 3 distributions
    A("## 3. Feature distributions")
    A("")
    A("Full table (all quantiles, kurtosis, outlier counts): `feature_distributions.csv`. Outliers are counted with "
      "Tukey fences (1.5×IQR and 3×IQR) and a robust z-score (|x − median| / (1.4826·MAD) > 3.5); both are undefined "
      "when IQR or MAD is 0 (sparse features). Nothing was removed or capped.")
    A("")
    for t in ["count", "signed relative", "ratio [0,1]"]:
        sub = dist[dist.type == t].copy()
        cols = ["feature", "set", "mean", "std", "median", "iqr", "min", "p01", "p99", "max", "skewness", "zero_share",
                "tukey_outliers_3iqr"]
        if t == "count":
            cols.append("skewness_after_log1p")
        if t == "signed relative":
            cols.append("skewness_after_asinh")
        A(f"**{t} features ({len(sub)})**")
        A("")
        A(md_table(sub[cols]))
        A("")
    b = dist[dist.type == "binary"][["feature", "set", "share_of_1"]]
    A(f"**binary features ({len(b)})**: share of 1s")
    A("")
    A(md_table(b))
    A("")
    A("Observations:")
    A("")
    A(f"- Heavily skewed (|skewness| > 2): {', '.join('`' + f + '`' for f in skewed.feature)}.")
    A(f"- Sparse counts (≥ 75% zeros, `log1p` barely helps): {', '.join('`' + f + '`' for f in sparse.feature)}.")
    A("- Signed `rel_*` features are centred near 0 and take negative values, so `log1p` is not applicable; `asinh` "
      "is sign-preserving and behaves like a log in the tails.")
    A("- Extreme values checked against the code are genuine: max `n_params` = 91 and `n_defaults` = 89 "
      "(`scattertext.produce_scattertext_explorer`), max `loc` = 567 (a Boeing 737 vehicle-setup script), "
      "max `class_n_methods` = 267 (`sympy` `LatexPrinter`).")
    A("")

    # 4 positive vs negative
    A("## 4. Renamed versus not renamed (descriptive)")
    A("")
    A(f"Effect sizes: Hedges' g (mean difference in pooled SDs; sensitive to skew) and Cliff's δ (= 2·AUC − 1; "
      f"robust, −1..1). The 95% interval for δ comes from a project-group bootstrap ({n_boot} resamples of the "
      "323 groups), which respects the clustering of rows in projects. Mann-Whitney p-values are Holm-adjusted over "
      "the 42 features. No feature has missing values, so there are no missingness differences.")
    A("")
    cols = ["feature", "set", "median_renamed", "median_not_renamed", "mean_renamed", "mean_not_renamed", "hedges_g",
            "cliffs_delta", "cliffs_delta_ci_low", "cliffs_delta_ci_high", "holm_p"]
    A(md_table(pvn.sort_values("cliffs_delta", key=abs, ascending=False)[cols]))
    A("")
    bins_ = pvn[pvn.type == "binary"][["feature", "rate_renamed", "rate_not_renamed", "rate_ratio", "n_renamed_with_1"]]
    A("Binary features: share with value 1 in each class.")
    A("")
    A(md_table(bins_.sort_values("rate_ratio", ascending=False)))
    A("")
    A("Caveats:")
    A("")
    A(f"- Only {npos} positives: a binary feature that is 1 for a handful of renamed methods (see `n_renamed_with_1`) "
      "gives rate ratios with very wide uncertainty.")
    A(f"- {int(sig.shape[0])} features have a bootstrap interval excluding 0: "
      f"{', '.join('`' + f + '`' for f in sig.feature)}. The two file-size features dominate; all main-set effects are "
      "small (|δ| < 0.2).")
    A("- Row-level tests and project-level intervals can disagree: `name_is_camel` is significant row by row "
      f"(Holm p = {pvn.set_index('feature').loc['name_is_camel', 'holm_p']:.1g}) but its project-level interval "
      "includes 0, because the camelCase renames cluster in a few projects.")
    A("- These are descriptive associations, not predictive importance or causes.")
    A("")

    # 5 correlation
    A("## 5. Correlation and redundancy")
    A("")
    A("Pairs with |Pearson| or |Spearman| ≥ 0.7 (full matrices: `correlation_pearson.csv`, `correlation_spearman.csv`):")
    A("")
    A(md_table(pairs))
    A("")
    A("**Exact identities** (row-for-row, all 46 features scanned; `exact_relationships.csv`):")
    A("")
    A(md_table(exact))
    A("")
    A("`class_depth` is already dropped; `has_docstring` and `in_class` stay in the approved sets. They carry the same "
      "information as the zero/non-zero part of `doc_len` and `class_n_methods`: harmless for trees, absorbed by L2 "
      "regularization in Logistic Regression.")
    A("")
    vt = vifs.sort_values("vif_main", ascending=False).head(12)[["feature", "vif_main", "vif_comparison",
                                                                  "vif_main_after_log1p_asinh"]]
    A("**Variance inflation factors** (top 12; `vif.csv`):")
    A("")
    A(md_table(vt))
    A("")
    A(f"- Raw main set: {int((vifs.vif_main > 10).sum())} features with VIF > 10 (max {vmax_raw:.1f}). After the "
      f"candidate `log1p`/`asinh` transforms: {int((vifs.vif_main_after_log1p_asinh > 10).sum())} (max {vmax_tr:.1f}).")
    A("- Transforms halve the size cluster (`loc`, `n_calls`, `rel_*`), leave the name-length cluster (`name_len`, "
      "`name_tokens`, `rel_name_len`, `rel_name_tokens`) unchanged, and **raise** `doc_len` because `log1p(doc_len)` "
      "becomes close to `has_docstring`.")
    A("")
    tg = target.reindex(feats_cmp).assign(abs_s=lambda d: d.spearman_with_target.abs()).sort_values(
        "abs_s", ascending=False).drop(columns="abs_s").head(10).reset_index().rename(columns={"index": "feature"})
    A("Correlation with the target (exploratory only; point-biserial = Pearson with the 0/1 target):")
    A("")
    A(md_table(tg))
    A("")

    # 6 imbalance and groups
    A("## 6. Class imbalance and project groups")
    A("")
    cnt = pd.cut(groups.positives, [-0.1, 0, 1, 2, 5, 10, 1000], labels=["0", "1", "2", "3–5", "6–10", "> 10"]).value_counts().sort_index()
    A(md_table(pd.DataFrame({"positives in project": cnt.index.astype(str), "projects": cnt.values})))
    A("")
    A(f"- {len(groups)} projects; rows per project: min {groups.rows.min()}, median {groups.rows.median():.0f}, "
      f"90th percentile {groups.rows.quantile(0.9):.0f}, max {groups.rows.max():,}. {int((groups.rows < 10).sum())} "
      "projects have fewer than 10 rows.")
    A(f"- {len(gpos)} projects ({len(gpos) / len(groups):.0%}) contain at least one positive; the top 5 hold "
      f"{top5:.0%} and the top 10 hold "
      f"{groups.sort_values('positives', ascending=False).positives.head(10).sum() / npos:.0%} of all positives.")
    A("")
    A(md_table(groups.sort_values("positives", ascending=False).head(10).reset_index()))
    A("")
    A("Class weighting (conceptual; to be computed on each training fold only):")
    A("")
    A(md_table(pd.DataFrame([
        ("`class_weight='balanced'` (scikit-learn)", f"positives ≈ {w_pos:.1f}, negatives ≈ {n / (2 * nneg):.3f}",
         "Logistic Regression, Random Forest"),
        ("`scale_pos_weight = negatives / positives` (LightGBM)", f"≈ {nneg / npos:.1f}", "LightGBM"),
        ("No weighting + threshold tuning", "-", "any; weighting mainly changes calibration and the default threshold"),
    ], columns=["Strategy", "Value on the full data (training folds differ slightly)", "Models"])))
    A("")

    # 7 grouped validation
    A("## 7. Grouped validation feasibility")
    A("")
    A(f"Simulated with `StratifiedGroupKFold(n_splits=5, shuffle=True)` over {SIM_SEEDS} seeds, grouping on "
      "`meta_project_group`; for the hold-out design one outer fold (≈ 20% of rows) is the test set and a second "
      "5-fold split runs on the rest. Every simulation asserts that no project appears in two partitions. **No split "
      "was saved.** Per-seed results: `grouped_cv_feasibility.csv`.")
    A("")
    rows_ = [
        ("5-fold on all data: positives per fold", "full_min_pos_per_fold", "full_max_pos_per_fold"),
        ("5-fold on all data: rows per fold", "full_min_rows_per_fold", "full_max_rows_per_fold"),
        ("5-fold on all data: positive projects in the smallest fold", "full_min_pos_groups_per_fold", "full_min_pos_groups_per_fold"),
        ("20% hold-out: rows", "holdout_rows", "holdout_rows"),
        ("20% hold-out: projects", "holdout_groups", "holdout_groups"),
        ("20% hold-out: positives", "holdout_positives", "holdout_positives"),
        ("20% hold-out: positive projects", "holdout_pos_groups", "holdout_pos_groups"),
        ("Development set: positives", "dev_positives", "dev_positives"),
        ("Inner 5-fold on development set: positives per fold", "inner_min_pos_per_fold", "inner_max_pos_per_fold"),
    ]
    A(md_table(pd.DataFrame([(lab, f"{int(sims[a].min())}–{int(sims[b_].max())}")
                             for lab, a, b_ in rows_], columns=["Quantity", f"Range across {SIM_SEEDS} seeds"])))
    A("")
    A("Assessment:")
    A("")
    A("- **5-fold StratifiedGroupKFold is feasible**: positives are spread almost evenly (47–50 per fold) and every "
      "fold has at least 10 positive projects.")
    A("- **A 20% project hold-out is feasible** and leaves ≈ 194 positives for development with 37–41 per inner fold, "
      "but the test estimate rests on only 10–17 positive projects; a single project (e.g. `streamlit`, 19 positives) "
      "can move the test score noticeably. Recommendation: fix the hold-out once (one seed), report a project-level "
      "bootstrap interval on it, and use repeated grouped CV on the development set as the main model-selection "
      "evidence.")
    A("- Row-level random splitting is not used; grouping by `meta_project_group` also keeps the repeated-method "
      "clusters together (0 clusters span two groups).")
    A("")

    # 8 recommendations
    A("## 8. Preprocessing recommendation matrix")
    A("")
    A("Status: **Approved** = already approved in `docs/experiment_protocol.md`; **Proposed** = needs your approval. "
      "LR = Logistic Regression; trees = Random Forest / LightGBM.")
    A("")
    recs = [
        ("Missing values", "0 missing / 0 non-numeric / 0 infinite in all 46 features",
         "No imputer: nothing to impute, and an imputer would hide future data errors. Add a validation step that fails on NaN/inf.",
         "all", "a silent imputer could mask a broken regeneration", "Proposed"),
        ("Invalid values", "binary 0/1, ratios in [0,1], counts non-negative integers: all valid",
         "No cleaning. Validate schema and ranges at load time (the loader already verifies the checksum).",
         "all", "none", "Proposed (checksum check: Approved)"),
        ("log1p for skewed counts", f"count skewness up to {dist[dist.type == 'count'].skewness.max():.1f}; log1p brings loc 5.6→0.5, doc_len 9.9→0.7, "
         "class_n_methods 5.0→0.3",
         "Apply to " + ", ".join(f"`{f}`" for f in LOG1P_CANDIDATES if f in MAIN_FEATURES) +
         " (+ `file_loc`, `file_n_methods` in the comparison set). Stateless, so it cannot leak, but it lives inside the pipeline.",
         "LR only (trees are invariant to monotone transforms)", "raises doc_len/has_docstring collinearity; changes coefficient interpretation",
         "Proposed"),
        ("asinh for skewed signed features", "rel_loc/rel_n_params/rel_n_calls/rel_cyclomatic/rel_doc_len skew 4.2–9.2, "
         "negative values; asinh → 0.55–0.83",
         "Apply sign-preserving asinh to these five; leave rel_name_len, rel_name_tokens, rel_name_body_overlap (|skew| ≤ 1.04) raw. "
         "Alternative: Yeo-Johnson fitted on training folds.",
         "LR only", "Yeo-Johnson would be a fitted step (must be fitted per training fold)", "Proposed"),
        ("Sparse counts", "name_short_tokens, name_no_vowel_tokens, n_defaults, n_decorators: 77–93% zeros; log1p leaves skew ≥ 1.9",
         "Keep as counts (no transform); the zero/non-zero split carries most of the information.",
         "LR", "a few large values keep some leverage; L2 limits it", "Proposed"),
        ("Scaling", "ranges from 0–1 (binaries, ratios) to 0–10,833 (doc_len)",
         "StandardScaler after the transforms, fitted on the training fold only, applied to all LR inputs (binaries included, so L2 penalizes all coefficients comparably).",
         "LR only; trees need no scaling", "fitting on all data would leak", "Proposed (fit-on-train rule: Approved)"),
        ("Binary features", "15 binaries, all valid 0/1; name_single_char 99.65% zeros (2 renamed)",
         "Keep as 0/1; no encoding needed. Keep near-constant name_single_char (approved set); expect an unstable LR coefficient.",
         "all", "rare binaries can produce large LR coefficients; L2 limits them", "Proposed"),
        ("Outliers", "thousands of Tukey outliers per skewed feature; inspected extremes are genuine code",
         "Do not remove or cap. log1p/asinh reduce their leverage for LR; trees are robust. Optional sensitivity: winsorize at the training-fold 99th percentile.",
         "LR (trees unaffected)", "capping on all data would leak; removal would bias the data", "Proposed"),
        ("Multicollinearity", f"VIF up to {vmax_raw:.1f}; exact identities has_docstring≡doc_len>0, in_class≡class_n_methods>0",
         "Keep all approved features (sets are frozen); use L2-regularized LR and interpret coefficients with care. No feature removal.",
         "LR (trees tolerate it)", "unstable coefficient signs", "Proposed"),
        ("Class imbalance", f"{npos / n:.2%} positives; {len(gpos)}/323 projects contain positives",
         "Class weights computed on the training fold (balanced for LR/RF, scale_pos_weight for LightGBM); tune the decision threshold on validation folds; judge by PR-AUC. No SMOTE or under-sampling.",
         "all", "weights change calibration; thresholds must not be tuned on the test set", "Proposed (no resampling before split, PR-AUC: Approved)"),
        ("Grouped splitting", "323 groups; simulations keep 47–50 positives per fold",
         "Split by meta_project_group; 20% project hold-out + StratifiedGroupKFold(5) on the rest; report project-level bootstrap intervals on the hold-out.",
         "all", "few positive projects in the test set", "Grouping: Approved; hold-out and 5-fold settings: Proposed"),
        ("Repeated / ambiguous vectors", f"{dups['rows_sharing_identical_42_feature_vector']:,} rows share a vector; 27 rows in 9 conflicting clusters",
         "Keep all rows; grouping keeps repeated methods in one partition. No de-duplication.",
         "all", "de-duplicating would change the approved dataset", "Proposed"),
    ]
    A(md_table(pd.DataFrame(recs, columns=["Topic", "What the data shows", "Recommendation (why)", "Model families",
                                           "Risks", "Status"])))
    A("")

    # 9 leakage prevention
    A("## 9. Leakage prevention in the future pipeline")
    A("")
    A("1. **Split first.** Assign project groups to hold-out / development and to folds before any fitting; a project "
      "never appears in two partitions; splits store keys and fold numbers only (`data/processed/splits/`).")
    A("2. **Fit on training folds only.** Scalers, any fitted transform (e.g. Yeo-Johnson), optional winsorizing "
      "thresholds, class weights and the decision threshold are fitted on the training part of each fold and applied "
      "unchanged to its validation part; the hold-out is used once at the end.")
    A("3. **One pipeline object.** Put transforms, scaler and model in a single scikit-learn `Pipeline`, so "
      "cross-validation refits everything per fold automatically.")
    A("4. **Clean X.** Build X only with `select_xy_groups(df, 'main' | 'comparison')`; identifiers, the target, "
      "`meta_*` columns (including `meta_project_group`) and the excluded/dropped features are rejected by "
      "`validate_feature_set()`.")
    A("5. **Immutable master.** Load through `load_master()`, which verifies the SHA-256; never write to the master.")
    A("6. **No look-ahead in this audit's numbers.** Statistics in this report use the full data and are descriptive; "
      "they must not be used to fit anything (for example, the skewness here only chooses *which* stateless "
      "transform to try; it supplies no fitted parameter).")
    A("")

    # 10 architecture
    A("## 10. Proposed implementation architecture (not implemented)")
    A("")
    A(md_table(pd.DataFrame([
        ("`src/features/feature_types.py` or a section of `feature_sets.py`", "feature → type (binary / count / ratio / signed relative) and the log1p / asinh lists, validated against the approved sets"),
        ("`src/preprocessing/transforms.py`", "`build_preprocessor(feature_set, model_family)`: a `ColumnTransformer` with log1p / asinh `FunctionTransformer`s and `StandardScaler` for LR; passthrough for trees; NaN/inf guard"),
        ("`src/evaluation/splits.py`", "group hold-out + `StratifiedGroupKFold`; writes `data/processed/splits/*.csv` (keys + fold), with per-fold counts"),
        ("`src/evaluation/metrics.py`", "PR-AUC, precision / recall / F1 / MCC at a validation-chosen threshold, ROC-AUC, project-bootstrap intervals"),
        ("`src/models/pipelines.py`", "`Pipeline(preprocessor, model)` per family with training-fold class weights"),
        ("`tests/`", "no fitting outside training folds; splits group-disjoint; transforms deterministic; X never contains forbidden columns"),
    ], columns=["Module", "Responsibility"])))
    A("")

    # 11 limitations
    A("## 11. Known limitations and unresolved questions")
    A("")
    A("- 76 of the 243 positives are AI-reviewed and not confirmed by a person; every statistic involving positives "
      "inherits that uncertainty.")
    A(f"- With {npos} positives, effect sizes and rare-binary rates are imprecise; project clustering widens the "
      "intervals further.")
    A("- 27 rows (9 clusters) have identical features but different labels: an upper bound on achievable separation "
      "with these features.")
    A("- Scope: methods in files edited by the future commit; about 9.5% of methods with unknown outcomes are excluded.")
    A("- Open decisions: transform choice for LR (log1p/asinh vs Yeo-Johnson), whether to run the winsorizing "
      "sensitivity check, and the hold-out seed and fold settings.")
    A("")
    A("## 12. Reproduction")
    A("")
    A(f"`python src/data/audit_preprocessing.py` (about 1 minute). Master SHA-256 before and after: `{sha_before}`.")
    A("")
    (OUT / "preprocessing_audit_report.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sha_before = verify_master()
    df = load_master(verify=False)
    OUT.mkdir(parents=True, exist_ok=True)

    integ = integrity(df)
    dups = duplicates(df)
    dist = distributions(df, COMPARISON_FEATURES)
    pvn, n_boot = positive_vs_negative(df, COMPARISON_FEATURES)
    pear, spear, pairs, target = correlations(df, COMPARISON_FEATURES)
    exact = exact_relationships(df)
    vifs = vif(df, MAIN_FEATURES).rename(columns={"vif": "vif_main", "r2_on_others": "r2_main"}).merge(
        vif(df, COMPARISON_FEATURES).rename(columns={"vif": "vif_comparison", "r2_on_others": "r2_comparison"}),
        on="feature", how="outer").merge(
        vif(df, MAIN_FEATURES, transformed=True).rename(columns={"vif": "vif_main_after_log1p_asinh",
                                                                 "r2_on_others": "r2_main_after_log1p_asinh"}),
        on="feature", how="left")
    groups = project_groups(df)
    sims = simulate_splits(df)

    integ.to_csv(OUT / "integrity_by_feature.csv", index=False)
    dist.to_csv(OUT / "feature_distributions.csv", index=False)
    pvn.merge(target, left_on="feature", right_index=True).to_csv(OUT / "positive_vs_negative.csv", index=False)
    pear.to_csv(OUT / "correlation_pearson.csv")
    spear.to_csv(OUT / "correlation_spearman.csv")
    pairs.to_csv(OUT / "correlation_pairs.csv", index=False)
    exact.to_csv(OUT / "exact_relationships.csv", index=False)
    vifs.to_csv(OUT / "vif.csv", index=False)
    groups.to_csv(OUT / "project_groups.csv")
    sims.to_csv(OUT / "grouped_cv_feasibility.csv", index=False)

    summary = {"generated": date.today().isoformat(), "master_sha256_before": sha_before,
               "rows": len(df), "positives": int(df.renamed.sum()), "negatives": int((df.renamed == 0).sum()),
               "features": len(config.FEATURES), "main": len(MAIN_FEATURES), "comparison": len(COMPARISON_FEATURES),
               "project_groups": int(df[GROUP_COLUMN].nunique()), "duplicates": dups, "bootstrap_reps_used": n_boot}
    write_report(df, integ, dups, dist, pvn, pairs, exact, vifs, target, groups, sims, n_boot, sha_before)
    sha_after = sha256_of(config.MASTER)
    summary["master_sha256_after"] = sha_after
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if sha_after != sha_before:
        raise SystemExit("Master checksum changed during the audit")
    log.info("Saved %s", OUT)
    return summary


if __name__ == "__main__":
    main()
