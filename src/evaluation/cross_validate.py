"""
Stage 8: leakage-safe development cross-validation.

  load_development_data(feature_set)  development rows only, from prepare_dataset()
                                      and the saved split (load_splits); verified counts,
                                      no held-out project, folds checked against the
                                      split metadata. Test rows are dropped before
                                      anything is returned.
  select_development(...)             the same selection for any (X, y, groups, assignment);
                                      used by load_development_data and by synthetic tests
  run_development_cv(spec, data)      for each saved fold: clone the unfitted pipeline,
                                      fit on the fold's training rows, score its
                                      validation rows; then OOF assembly and the Stage 6
                                      metrics
  check_baseline_equivalence(...)     baselines must give identical OOF scores for main
                                      40 and comparison 42 (they ignore file-size columns)
  select_stage9_candidate(...)        the fixed Stage 9 selection rule

Rules enforced here (docs/experiment_protocol.md, Stage 8):
  - every learned step (scalers, Yeo-Johnson lambdas, class weights, the random
    baseline's rate) lives inside the pipeline and is refitted on each training fold
  - probability column 1 is used only after checking classes_ == [0, 1]; scores must
    be finite and in [0, 1]; every development row gets exactly one OOF score
  - a ConvergenceWarning fails the run; all other warnings are recorded with their
    fold; non-finite transformed values fail through FeatureOutputCheck
  - CV AP = mean and sample std (ddof=1) of the 5 validation-fold APs; the pooled-OOF
    AP is not computed. Threshold = max F1 on pooled development OOF scores; the
    precision / recall / F1 / MCC measured there are labelled optimistic.
Nothing here reads held-out test rows, and nothing is saved here (see run_store.py).
"""
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import GROUP_COLUMN  # noqa: E402
from src.data.prepare_dataset import prepare_dataset  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation.splits import KEY, inner_folds, load_splits, outer_indices  # noqa: E402
from src.models.pipelines import BASE_PARAMS, build_model_pipeline, resolve_model_params  # noqa: E402

EXPECTED_DEVELOPMENT = {"rows": 24967, "projects": 261, "positives": 193}
EXPLORATORY = "exploratory_post_hoc"
OPTIMISTIC_NOTE = ("optimistic: measured on the same pooled development OOF scores that were used to "
                   "select the threshold; not a performance estimate")
# Stage 9 candidates, simplest first (exact-tie rule); the naming heuristic is never a candidate
STAGE9_CANDIDATES = [("logistic_regression", "log1p_asinh"), ("logistic_regression", "yeo_johnson"),
                     ("random_forest", "none"), ("lightgbm", "none")]
BASELINES = ("dummy_prior", "stratified_random", "naming_heuristic")


class DevelopmentDataError(ValueError):
    """The development data or its folds do not match the saved split."""


class RunFailedError(RuntimeError):
    """A run violated a Stage 8 rule (convergence, probabilities, OOF integrity) and has no result."""

    def __init__(self, message, fold=None, warnings_log=None):
        super().__init__(message if fold is None else f"fold {fold}: {message}")
        self.fold = fold
        self.warnings_log = list(warnings_log or [])


@dataclass(frozen=True)
class RunSpec:
    """One configured experiment; model_params are the resolved, approved settings."""
    run_id: str
    model: str
    feature_set: str
    transform: str
    model_params: dict
    role: str
    research_status: str


@dataclass
class DevelopmentData:
    """Development rows only (index = master row), their keys and the 5 saved folds."""
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    keys: pd.DataFrame
    folds: list
    fold_of_row: np.ndarray
    split_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- development data
def _fold_counts(y, groups, folds):
    out = []
    for train, val in folds:
        yv, gv, yt, gt = y.iloc[val], groups.iloc[val], y.iloc[train], groups.iloc[train]
        out.append({"validation": {"rows": int(len(val)), "projects": int(gv.nunique()),
                                   "positives": int(yv.sum()), "negatives": int((yv == 0).sum()),
                                   "positive_projects": int(gv[yv == 1].nunique())},
                    "training": {"rows": int(len(train)), "projects": int(gt.nunique()),
                                 "positives": int(yt.sum()), "negatives": int((yt == 0).sum()),
                                 "positive_projects": int(gt[yt == 1].nunique())}})
    return out


def select_development(X, y, groups, assign, meta):
    """Keep only development rows (by the saved assignment) and verify them against the split metadata."""
    if len(X) != len(assign) or not (X.index.equals(y.index) and X.index.equals(groups.index)):
        raise DevelopmentDataError("X, y, groups and the split assignment must cover the same rows")
    if not np.array_equal(groups.astype(str).to_numpy(), assign[GROUP_COLUMN].astype(str).to_numpy()) or \
            not np.array_equal(y.to_numpy(), assign[config.TARGET].to_numpy()):
        raise DevelopmentDataError("groups or labels differ from the saved split assignment")
    dev_pos, test_pos = outer_indices(assign)
    test_projects = set(assign[GROUP_COLUMN].iloc[test_pos])
    dev_projects = set(assign[GROUP_COLUMN].iloc[dev_pos])
    if dev_projects & test_projects:
        raise DevelopmentDataError(f"held-out projects in the development data: {sorted(dev_projects & test_projects)[:5]}")

    Xd, yd, gd = X.iloc[dev_pos], y.iloc[dev_pos], groups.iloc[dev_pos]     # test rows dropped here
    keys = assign.iloc[dev_pos][["master_row"] + KEY].reset_index(drop=True)
    keys.index = Xd.index
    folds = inner_folds(assign)
    M.validate_folds(folds, len(Xd))
    expected = meta["counts"]["development"]
    got = {"rows": len(Xd), "projects": int(gd.nunique()), "positives": int(yd.sum())}
    if any(got[k] != expected[k] for k in got):
        raise DevelopmentDataError(f"development counts {got} differ from the split metadata {expected}")
    counts = _fold_counts(yd, gd, folds)
    for k, (c, m) in enumerate(zip(counts, meta["counts"]["inner_folds"])):
        if c["validation"] != m["validation"] or c["training"] != m["training"]:
            raise DevelopmentDataError(f"fold {k} counts differ from the split metadata")
    for k, (train, val) in enumerate(folds):
        if set(gd.iloc[train]) & set(gd.iloc[val]):
            raise DevelopmentDataError(f"fold {k}: a project is in both training and validation")
    fold_of_row = np.full(len(Xd), -1)
    for k, (_, val) in enumerate(folds):
        fold_of_row[val] = k
    info = {"split_hash": meta["split_hash"], "assignment_file_sha256": meta.get("assignment_file_sha256"),
            "dataset_sha256": meta["dataset_sha256"], "development": got, "folds": counts}
    return DevelopmentData(Xd, yd, gd, keys, folds, fold_of_row, info)


def load_development_data(feature_set):
    """Development rows of the real master for `feature_set`; reads no model and fits nothing."""
    X, y, groups = prepare_dataset(feature_set)
    assign, meta = load_splits()
    data = select_development(X, y, groups, assign, meta)
    if data.split_info["development"] != EXPECTED_DEVELOPMENT:
        raise DevelopmentDataError(f"development counts {data.split_info['development']} differ from the "
                                   f"approved {EXPECTED_DEVELOPMENT}")
    return data


# ---------------------------------------------------------------------------- one run
def build_spec_pipeline(spec):
    """Unfitted pipeline for a RunSpec. Only values that differ from the approved settings are passed
    as overrides (so locked parameters stay locked), and the resolved settings must equal spec.model_params."""
    base = BASE_PARAMS[spec.model]
    missing = sorted(set(base) - set(spec.model_params))
    if missing:
        raise ValueError(f"{spec.run_id}: model_params lacks approved settings {missing}")
    overrides = {k: v for k, v in spec.model_params.items() if k not in base or base[k] != v}
    if resolve_model_params(spec.model, overrides) != spec.model_params:
        raise ValueError(f"{spec.run_id}: model_params do not resolve to themselves")
    return build_model_pipeline(spec.model, spec.feature_set, spec.transform, overrides or None)


def _warning_record(w, fold, stage):
    return {"fold": fold, "stage": stage, "category": w.category.__name__, "message": str(w.message),
            "source": f"{Path(w.filename).name}:{w.lineno}"}


def run_development_cv(spec, data, pipeline_factory=None, fold_callback=None):
    """Cross-validate one RunSpec on DevelopmentData; returns a result dict (raises RunFailedError).

    pipeline_factory(spec) -> unfitted Pipeline (default: build_model_pipeline); it is called
    once and cloned per fold. fold_callback(fold, fitted_pipeline, train_idx, val_idx) is for
    inspection in tests; fitted models are never stored.
    """
    template = (pipeline_factory or build_spec_pipeline)(spec)
    log, fold_rows, parts = [], [], []
    for k, (train, val) in enumerate(data.folds):
        if set(data.groups.iloc[train]) & set(data.groups.iloc[val]):
            raise RunFailedError("a project is in both training and validation", k, log)
        pipe = clone(template)
        Xt, yt, Xv = data.X.iloc[train], data.y.iloc[train], data.X.iloc[val]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                t0 = time.perf_counter()
                pipe.fit(Xt, yt)
                t1 = time.perf_counter()
                fit_warnings = list(caught)
                proba = pipe.predict_proba(Xv)
                t2 = time.perf_counter()
            except Exception as exc:                     # non-finite transform, invalid input, ...
                log += [_warning_record(w, k, "fit/predict") for w in caught]
                raise RunFailedError(f"{type(exc).__name__}: {exc}", k, log) from exc
        log += [_warning_record(w, k, "fit") for w in fit_warnings]
        log += [_warning_record(w, k, "predict") for w in caught[len(fit_warnings):]]
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RunFailedError("ConvergenceWarning: the model did not converge (approved settings are not "
                                 "changed to force convergence)", k, log)
        classes = list(getattr(pipe, "classes_", []))
        if classes != [0, 1]:
            raise RunFailedError(f"classes_ is {classes}, expected [0, 1]", k, log)
        proba = np.asarray(proba)
        if proba.ndim != 2 or proba.shape != (len(val), 2):
            raise RunFailedError(f"predict_proba returned shape {proba.shape}, expected ({len(val)}, 2)", k, log)
        try:
            p = M.validate_probabilities(proba[:, 1], len(val))
        except M.EvaluationInputError as exc:
            raise RunFailedError(f"invalid probabilities: {exc}", k, log) from exc
        if fold_callback is not None:
            fold_callback(k, pipe, train, val)
        yv = data.y.iloc[val].to_numpy()
        fold_rows.append({"fold": k, "n_train": int(len(train)), "n_train_positive": int(yt.sum()),
                          "n_train_projects": int(data.groups.iloc[train].nunique()),
                          "n_val": int(len(val)), "n_val_positive": int(yv.sum()),
                          "n_val_projects": int(data.groups.iloc[val].nunique()),
                          "val_prevalence": float(yv.mean()), "average_precision": M.average_precision(yv, p),
                          "roc_auc": M.roc_auc(yv, p), "fit_seconds": round(t1 - t0, 4),
                          "predict_seconds": round(t2 - t1, 4),
                          "n_warnings": sum(1 for r in log if r["fold"] == k)})
        parts.append(p)

    try:
        oof = M.assemble_oof(parts, data.folds, len(data.X), index=data.X.index)
        M.validate_oof(oof, data.y, data.folds)
    except M.EvaluationInputError as exc:
        raise RunFailedError(f"OOF integrity: {exc}", None, log) from exc
    cv = M.fold_average_precision(data.y, oof, data.folds)
    selection = M.select_max_f1_threshold(data.y, oof)
    at_threshold = M.evaluate(data.y, oof, selection.threshold)
    summary = {
        "run_id": spec.run_id, "model": spec.model, "feature_set": spec.feature_set, "transform": spec.transform,
        "role": spec.role, "research_status": spec.research_status,
        "primary_metric": "average_precision",
        "cv_average_precision": {"mean": cv["mean"], "std": cv["std"], "std_ddof": cv["std_ddof"],
                                 "per_fold": cv["per_fold"], "n_folds": cv["n_folds"],
                                 "definition": "arithmetic mean and sample std of the 5 validation-fold APs"},
        "cv_roc_auc_per_fold": [r["roc_auc"] for r in fold_rows],
        "threshold_selection": {"objective": selection.objective, "threshold": selection.threshold,
                                "n_tied_max_f1": selection.n_tied, "n_candidates": selection.n_candidates,
                                "data": "pooled development OOF scores"},
        "development_threshold_metrics": {"note": OPTIMISTIC_NOTE,
                                          **{m: at_threshold[m] for m in M.THRESHOLD_METRICS},
                                          **{c: at_threshold[c] for c in ("tp", "fp", "fn", "tn")}},
        "fold_threshold_diagnostics": {**M.fold_threshold_diagnostics(data.y, oof, data.folds),
                                       "note": "diagnostic only; never used to select the threshold"},
        "n_warnings": len(log),
    }
    oof_table = data.keys.copy()
    oof_table[GROUP_COLUMN] = data.groups.to_numpy()
    oof_table["fold"] = data.fold_of_row
    oof_table[config.TARGET] = data.y.to_numpy()
    oof_table["oof_probability"] = np.asarray(oof)
    return {"summary": summary, "fold_metrics": pd.DataFrame(fold_rows), "oof": oof_table.reset_index(drop=True),
            "warnings": log}


# ---------------------------------------------------------------------------- matrix-level checks
def check_baseline_equivalence(results):
    """Baselines ignore file-size columns: their main and comparison OOF scores must be identical."""
    checked = []
    for model in BASELINES:
        a, b = results.get(f"{model}__main"), results.get(f"{model}__comparison")
        if a is None or b is None:
            continue
        if not np.array_equal(a["oof"]["oof_probability"].to_numpy(), b["oof"]["oof_probability"].to_numpy()):
            raise RunFailedError(f"{model}: main and comparison OOF scores differ")
        checked.append(model)
    return checked


def select_stage9_candidate(summaries, feature_set):
    """Fixed rule: highest CV AP mean among LR / RF / LightGBM; exact tie -> the simpler model."""
    rows = []
    for rank, (model, transform) in enumerate(STAGE9_CANDIDATES):
        s = next((s for s in summaries if s["model"] == model and s["transform"] == transform
                  and s["feature_set"] == feature_set), None)
        if s is None:
            raise ValueError(f"missing candidate {model} / {transform} for {feature_set}")
        if s["research_status"] == EXPLORATORY:
            raise ValueError("an exploratory run cannot be a Stage 9 candidate")
        rows.append((-s["cv_average_precision"]["mean"], rank, s["run_id"]))
    return min(rows)[2]
