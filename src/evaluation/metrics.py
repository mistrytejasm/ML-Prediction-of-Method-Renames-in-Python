"""
Stage 6: evaluation metrics, maximum-F1 threshold selection, cross-validation
diagnostics and the stratified project-level bootstrap. Infrastructure only:
nothing here trains a model or reads the held-out test set by itself.

Metrics (explicit names; METRICS lists them in report order)
  average_precision  PRIMARY. sklearn.metrics.average_precision_score:
                     AP = sum_n (R_n - R_{n-1}) * P_n, a step-wise sum over the
                     precision-recall curve. NOT the trapezoidal area
                     auc(recall, precision) ("trapezoidal PR-AUC"), which joins the
                     curve's points with straight lines and can overstate rare-class
                     performance; trapezoidal_pr_auc() exists only to show the
                     difference and is never reported. Random baseline AP = positive rate.
  roc_auc            supplementary (roc_auc_score)
  -> SCORE_METRICS: both computed from the continuous probabilities; threshold-independent
  precision, recall, f1, mcc   THRESHOLD_METRICS: from 0 / 1 predictions at a fixed threshold
                     (predicted positive = probability >= threshold)

Reporting: evaluate() returns all six metrics (AP first) plus the confusion counts.
ThresholdSelection holds only the threshold-specific precision / recall / F1 of the
selected operating point; AP for the same data comes from evaluate() or
fold_average_precision(), never from a threshold.

Undefined values are NaN, never 0 (counts TP, FP, FN, TN):
  precision  TP / (TP + FP)       undefined when nothing is predicted positive
  recall     TP / (TP + FN)       undefined when there is no positive
  f1         2TP / (2TP + FP + FN)  undefined only when there is no positive AND
                                     nothing is predicted positive; with positives
                                     but no positive prediction it is 0 (well defined)
  mcc        (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
             undefined when any factor is 0 (scikit-learn returns 0.0 there; we do not)
  average_precision  undefined without a positive; roc_auc undefined without both classes

Threshold selection: select_max_f1_threshold() on DEVELOPMENT out-of-fold scores
only. Its precision / recall / F1 are measured on the same scores that chose the
threshold, so they describe the development operating point and are optimistic;
they are not a performance estimate.

Development CV AP = arithmetic mean (and sample standard deviation, ddof=1) of the
per-fold validation APs (fold_average_precision). The AP of the pooled OOF scores is
not a substitute; pooled OOF scores are used only to choose the threshold.

Bootstrap (project_bootstrap): stratified by project. Projects containing a
positive and negative-only projects are resampled separately, with replacement,
each to its observed count; a drawn project contributes all its rows, a project
drawn twice contributes them twice. Predictions and probabilities are fixed (no
refit). Percentile intervals. The intervals are CONDITIONAL on the observed
numbers of positive-containing and negative-only projects (e.g. 12 / 50 in the
planned test set): they do not include uncertainty in that composition. With so
few positive projects they are approximate and may cover less than 95%.
"""
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PRIMARY_METRIC = "average_precision"
SCORE_METRICS = ("average_precision", "roc_auc")
THRESHOLD_METRICS = ("precision", "recall", "f1", "mcc")
METRICS = SCORE_METRICS + THRESHOLD_METRICS
N_BOOT = 2000
BOOT_SEED = 42


class EvaluationInputError(ValueError):
    """Invalid labels, probabilities, predictions, folds or project identifiers."""


# ---------------------------------------------------------------------------- input validation
def _as_1d(values, name):
    arr = values.to_numpy() if isinstance(values, (pd.Series, pd.Index)) else np.asarray(values)
    if arr.ndim != 1:
        raise EvaluationInputError(f"{name} must be 1-D, got shape {arr.shape}")
    if len(arr) == 0:
        raise EvaluationInputError(f"{name} is empty")
    return arr


def validate_labels(y, require_both_classes=False):
    """Return y as an int array of 0 / 1 (positional); bools and other values are rejected."""
    arr = _as_1d(y, "labels")
    if arr.dtype == bool or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise EvaluationInputError(f"labels must be numeric 0 / 1, got dtype {arr.dtype}")
    if not np.isin(arr, (0, 1)).all():
        raise EvaluationInputError(f"labels must be 0 or 1, got {sorted(set(np.unique(arr)) - {0, 1})[:5]}")
    arr = arr.astype(int)
    if require_both_classes and len(np.unique(arr)) < 2:
        raise EvaluationInputError("labels must contain both classes")
    return arr


def validate_probabilities(p, n=None):
    """Return p as a float array; every value must be finite and in [0, 1]."""
    arr = _as_1d(p, "probabilities")
    if arr.dtype == bool or not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise EvaluationInputError(f"probabilities must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        raise EvaluationInputError("probabilities contain NaN or infinite values")
    if (arr < 0).any() or (arr > 1).any():
        raise EvaluationInputError("probabilities must lie in [0, 1]")
    if n is not None and len(arr) != n:
        raise EvaluationInputError(f"probabilities have length {len(arr)}, labels {n}")
    return arr


def validate_predictions(pred, n=None):
    arr = validate_labels(pred)
    if n is not None and len(arr) != n:
        raise EvaluationInputError(f"predictions have length {len(arr)}, labels {n}")
    return arr


def _check_series_alignment(*series):
    idx = [s.index for s in series if isinstance(s, pd.Series)]
    if len(idx) > 1 and not all(i.equals(idx[0]) for i in idx[1:]):
        raise EvaluationInputError("pandas inputs have different indices; align them before evaluating")


# ---------------------------------------------------------------------------- metrics
def confusion_counts(y, pred):
    y, pred = validate_labels(y), validate_predictions(pred)
    if len(y) != len(pred):
        raise EvaluationInputError(f"labels have length {len(y)}, predictions {len(pred)}")
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def threshold_metrics_from_counts(tp, fp, fn, tn):
    """Precision, recall, F1 and MCC from counts; NaN where a metric is undefined (never 0)."""
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else nan
    recall = tp / (tp + fn) if tp + fn else nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else nan
    denom = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else nan
    return {"precision": precision, "recall": recall, "f1": f1, "mcc": float(mcc)}


def threshold_metrics(y, pred):
    return threshold_metrics_from_counts(**confusion_counts(y, pred))


def average_precision(y, proba):
    """Average Precision (AP, the primary metric); NaN when there is no positive."""
    y = validate_labels(y)
    proba = validate_probabilities(proba, len(y))
    return float(average_precision_score(y, proba)) if y.any() else float("nan")


def roc_auc(y, proba):
    y = validate_labels(y)
    proba = validate_probabilities(proba, len(y))
    return float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan")


def trapezoidal_pr_auc(y, proba):
    """Trapezoidal area under the PR curve. For comparison only: never the reported metric."""
    y = validate_labels(y, require_both_classes=True)
    precision, recall, _ = precision_recall_curve(y, validate_probabilities(proba, len(y)))
    return float(auc(recall, precision))


def predict_at_threshold(proba, threshold):
    if not np.isfinite(threshold):
        raise EvaluationInputError("threshold must be finite")
    return (validate_probabilities(proba) >= threshold).astype(int)


def evaluate(y, proba, threshold):
    """All six metrics plus counts for fixed probabilities and a fixed threshold."""
    _check_series_alignment(y, proba)
    y = validate_labels(y)
    proba = validate_probabilities(proba, len(y))
    pred = predict_at_threshold(proba, threshold)
    counts = confusion_counts(y, pred)
    return {"average_precision": average_precision(y, proba), "roc_auc": roc_auc(y, proba),
            **threshold_metrics_from_counts(**counts), **counts,
            "n": int(len(y)), "n_positive": int(y.sum()), "threshold": float(threshold)}


# ---------------------------------------------------------------------------- threshold selection
@dataclass(frozen=True)
class ThresholdSelection:
    """Maximum-F1 operating point chosen on development OOF scores (optimistic on those scores)."""
    threshold: float
    precision: float
    recall: float
    f1: float
    n_tied: int                 # thresholds sharing the maximum F1; the lowest was chosen
    n_candidates: int
    objective: str = "max_f1"
    scope: str = "development OOF scores; metrics are on the selection data and are optimistic"


def select_max_f1_threshold(y_dev, oof_proba):
    """Choose the threshold with maximum F1 on development OOF scores; ties -> the lowest threshold.

    precision_recall_curve returns precision[i], recall[i] for thresholds[i] (i < len(thresholds));
    its last precision / recall pair (1, 0) has no threshold and is dropped. F1 is compared as an
    exact fraction 2TP / (2TP + FP + FN) from the counts, so ties are detected exactly.
    Never call this with held-out test labels or scores.
    """
    _check_series_alignment(y_dev, oof_proba)
    y = validate_labels(y_dev, require_both_classes=True)
    proba = validate_probabilities(oof_proba, len(y))
    precision, recall, thresholds = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]               # aligned with thresholds
    n_pos = int(y.sum())
    pos_sorted, all_sorted = np.sort(proba[y == 1]), np.sort(proba)
    tp = n_pos - np.searchsorted(pos_sorted, thresholds, side="left")       # exact counts of score >= t
    fp = (len(proba) - np.searchsorted(all_sorted, thresholds, side="left")) - tp
    fn = n_pos - tp
    if not (np.allclose(tp / n_pos, recall) and np.allclose(tp / (tp + fp), precision)):
        raise RuntimeError("precision / recall misaligned with thresholds")
    f1 = [Fraction(2 * a, 2 * a + b + c) for a, b, c in zip(tp, fp, fn)]
    best = max(f1)
    tied = [i for i, v in enumerate(f1) if v == best]
    i = min(tied, key=lambda k: thresholds[k])                    # lowest threshold among ties
    m = threshold_metrics(y, (proba >= thresholds[i]).astype(int))
    return ThresholdSelection(threshold=float(thresholds[i]), precision=m["precision"], recall=m["recall"],
                              f1=float(best), n_tied=len(tied), n_candidates=len(thresholds))


# ---------------------------------------------------------------------------- OOF and fold diagnostics
def validate_folds(folds, n_rows):
    """Every row must be in exactly one validation fold; train and validation of a fold must not overlap."""
    seen = np.zeros(n_rows, dtype=int)
    for k, (train, val) in enumerate(folds):
        train, val = np.asarray(train), np.asarray(val)
        for name, idx in (("train", train), ("validation", val)):
            if idx.size and (idx.min() < 0 or idx.max() >= n_rows or not np.issubdtype(idx.dtype, np.integer)):
                raise EvaluationInputError(f"fold {k} {name} indices outside 0..{n_rows - 1}")
            if len(np.unique(idx)) != len(idx):
                raise EvaluationInputError(f"fold {k} {name} indices repeat")
        if np.intersect1d(train, val).size:
            raise EvaluationInputError(f"fold {k}: a row is in both training and validation")
        seen[val] += 1
    if not (seen == 1).all():
        raise EvaluationInputError(f"{int((seen == 0).sum())} rows in no validation fold, "
                                   f"{int((seen > 1).sum())} in more than one")


def assemble_oof(fold_probabilities, folds, n_rows, index=None):
    """Place each fold's validation probabilities at its rows; returns OOF scores in development order."""
    if len(fold_probabilities) != len(folds):
        raise EvaluationInputError("one probability array per fold is required")
    validate_folds(folds, n_rows)
    oof = np.full(n_rows, np.nan)
    for (_, val), p in zip(folds, fold_probabilities):
        oof[np.asarray(val)] = validate_probabilities(p, len(val))
    return pd.Series(oof, index=index, name="oof_probability") if index is not None else oof


def validate_oof(oof, y_dev, folds):
    """OOF scores must have one finite probability per development row, in development-row order."""
    if isinstance(oof, pd.Series) and isinstance(y_dev, pd.Series) and not oof.index.equals(y_dev.index):
        raise EvaluationInputError("OOF index differs from the development rows (order or rows)")
    y = validate_labels(y_dev)
    validate_probabilities(oof, len(y))
    validate_folds(folds, len(y))


def fold_average_precision(y_dev, oof, folds):
    """Per-fold validation AP and the development CV AP = their arithmetic mean (std with ddof=1)."""
    validate_oof(oof, y_dev, folds)
    y, p = validate_labels(y_dev), validate_probabilities(oof)
    per_fold = [average_precision(y[np.asarray(val)], p[np.asarray(val)]) for _, val in folds]
    values = np.asarray(per_fold)
    return {"per_fold": per_fold, "mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "std_ddof": 1, "n_folds": len(per_fold)}


def fold_threshold_diagnostics(y_dev, oof, folds):
    """Threshold each validation fold would choose on its own (diagnostic only; never used to select)."""
    validate_oof(oof, y_dev, folds)
    y, p = validate_labels(y_dev), validate_probabilities(oof)
    per_fold = [select_max_f1_threshold(y[np.asarray(val)], p[np.asarray(val)]).threshold for _, val in folds]
    return {"per_fold": per_fold, "min": float(min(per_fold)), "max": float(max(per_fold)),
            "range": float(max(per_fold) - min(per_fold))}


# ---------------------------------------------------------------------------- project bootstrap
def _project_rows(y, projects):
    codes, uniques = pd.factorize(projects, sort=True)
    rows = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    has_pos = np.array([y[r].any() for r in rows])
    return rows, np.flatnonzero(has_pos), np.flatnonzero(~has_pos)


def validate_projects(projects, n):
    arr = _as_1d(projects, "projects")
    if len(arr) != n:
        raise EvaluationInputError(f"projects have length {len(arr)}, labels {n}")
    if pd.isna(arr).any() or (pd.Series(arr).astype(str).str.strip() == "").any():
        raise EvaluationInputError("projects contain missing or empty identifiers")
    return arr


def bootstrap_indices(y, projects, n_boot=N_BOOT, seed=BOOT_SEED):
    """Yield row-index arrays of stratified project resamples (positive-containing / negative-only)."""
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise EvaluationInputError("n_boot must be a positive integer")
    y = validate_labels(y)
    projects = validate_projects(projects, len(y))
    rows, pos, neg = _project_rows(y, projects)
    if len(pos) == 0 or len(neg) == 0:
        raise EvaluationInputError("need at least one positive-containing and one negative-only project")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        picks = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        yield np.concatenate([rows[k] for k in picks])


def project_bootstrap(y, proba, pred, projects, n_boot=N_BOOT, seed=BOOT_SEED, confidence=0.95,
                      return_replicates=False, threshold=None):
    """Percentile CIs for every metric from a stratified project bootstrap of FIXED predictions.

    AP and ROC-AUC come from `proba`; precision, recall, F1 and MCC from `pred`. If
    `threshold` is given, `pred` must equal (proba >= threshold), so the two cannot drift apart.

    Returns a DataFrame indexed by metric: point, ci_low, ci_high, n_valid, n_undefined, with
    attrs n_boot, seed, confidence, n_positive_projects, n_negative_projects. Undefined
    replicate values (e.g. precision with no positive prediction) are excluded from the
    percentiles and counted in n_undefined. Pass labels and scores of ONE evaluation set.
    """
    _check_series_alignment(y, proba, pred, projects)
    y = validate_labels(y, require_both_classes=True)
    proba = validate_probabilities(proba, len(y))
    pred = validate_predictions(pred, len(y))
    projects = validate_projects(projects, len(y))
    if not 0 < confidence < 1:
        raise EvaluationInputError("confidence must lie in (0, 1)")
    if threshold is not None and not np.array_equal(pred, predict_at_threshold(proba, threshold)):
        raise EvaluationInputError("pred differs from (proba >= threshold)")

    def metrics(idx):
        yy = y[idx]
        return {"average_precision": average_precision(yy, proba[idx]), "roc_auc": roc_auc(yy, proba[idx]),
                **threshold_metrics_from_counts(**confusion_counts(yy, pred[idx]))}

    reps = pd.DataFrame([metrics(idx) for idx in bootstrap_indices(y, projects, n_boot, seed)], columns=list(METRICS))
    point = pd.Series(metrics(np.arange(len(y))))
    alpha = (1 - confidence) / 2 * 100
    out = pd.DataFrame({
        "point": point,
        "ci_low": [np.percentile(reps[m].dropna(), alpha) if reps[m].notna().any() else np.nan for m in METRICS],
        "ci_high": [np.percentile(reps[m].dropna(), 100 - alpha) if reps[m].notna().any() else np.nan for m in METRICS],
        "n_valid": reps.notna().sum(),
        "n_undefined": reps.isna().sum()}, index=list(METRICS))
    _, pos, neg = _project_rows(y, projects)
    out.attrs.update({"n_boot": n_boot, "seed": seed, "confidence": confidence, "method": "stratified project percentile",
                      "n_positive_projects": int(len(pos)), "n_negative_projects": int(len(neg)),
                      "conditional_on": "the observed numbers of positive-containing and negative-only projects"})
    return (out, reps) if return_replicates else out
