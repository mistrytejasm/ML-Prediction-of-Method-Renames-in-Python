"""
Stage 5: model pipelines and fold-aware class weighting. Builds unfitted
pipelines only; nothing here trains on the master dataset.

  build_model_pipeline(model_name, feature_set, transform=None, model_params=None)

  model_name            transform (None = the model's default)     pipeline
  logistic_regression   "log1p_asinh" (default) | "yeo_johnson"    check -> pre -> output_check -> model
  random_forest         "none"                                     check -> model   (raw features)
  lightgbm              "none"                                     check -> model   (raw features)
  dummy_prior           "none"                                     check -> model   (raw features, ignored)

Every pipeline starts with the Stage 4 input check (FeatureFrameCheck); the
Logistic Regression steps are the Stage 4 preprocessing pipeline itself.

Baseline settings (explicit, not tuned; documented in BASE_PARAMS):
  logistic_regression  L2 (l1_ratio=0.0; `penalty` is deprecated since scikit-learn
                       1.8), C=1.0, solver="lbfgs", max_iter=5000,
                       class_weight="balanced", random_state=42
  random_forest        500 trees, max_features="sqrt", min_samples_leaf=1,
                       class_weight="balanced", random_state=42, n_jobs=-1
  lightgbm             FoldWeightedLGBM: binary objective, 300 trees,
                       learning_rate=0.05, num_leaves=31, min_child_samples=20,
                       random_state=42, n_jobs=1, deterministic=True;
                       scale_pos_weight = negatives / positives of each fit's labels
  dummy_prior          DummyClassifier(strategy="prior"): predicts the training
                       positive rate for every row (the class-imbalance reference)

model_params (dict) overrides baseline hyperparameters. Unknown parameters and the
parameters that define the approved design (LOCKED_PARAMS: class weighting, the
L2 penalty, the dummy strategy, LightGBM's objective and weighting) are rejected.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_sets import get_feature_set  # noqa: E402
from src.preprocessing.transforms import LR_TRANSFORMS, FeatureFrameCheck, build_lr_preprocessing_pipeline  # noqa: E402


class ModelConfigError(ValueError):
    """An unknown model, transform or parameter, or an unsupported combination."""


class FoldWeightedLGBM(ClassifierMixin, BaseEstimator):
    """LightGBM binary classifier whose scale_pos_weight comes from the labels of each fit.

    fit(X, y) sets scale_pos_weight_ = n_negative / n_positive of THIS y, so under
    cross-validation every training fold gets its own ratio and no global ratio
    exists. `lgbm_params` holds hyperparameters only; the wrapper manages
    objective, scale_pos_weight and verbosity and rejects them (and is_unbalance,
    class_weight) if supplied.
    """

    MANAGED = ("objective", "scale_pos_weight", "is_unbalance", "unbalance", "class_weight",
               "verbose", "verbosity")

    def __init__(self, lgbm_params=None):
        self.lgbm_params = lgbm_params

    def _checked_params(self):
        params = dict(self.lgbm_params or {})
        managed = sorted(set(params) & set(self.MANAGED))
        if managed:
            raise ModelConfigError(f"FoldWeightedLGBM manages these LightGBM parameters itself: {managed}")
        known = set(LGBMClassifier().get_params()) - {"kwargs"} | {"deterministic", "force_row_wise", "force_col_wise"}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ModelConfigError(f"unknown LightGBM parameters: {unknown}")
        return params

    def fit(self, X, y):
        params = self._checked_params()
        y = np.asarray(y)
        if y.ndim != 1 or len(y) == 0 or not set(np.unique(y)) <= {0, 1}:
            raise ModelConfigError("FoldWeightedLGBM needs a non-empty 1-D binary target with values 0 / 1")
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        if n_pos == 0:
            raise ModelConfigError("training labels contain no positive (renamed) example")
        if n_neg == 0:
            raise ModelConfigError("training labels contain no negative example")
        self.n_positive_, self.n_negative_ = n_pos, n_neg
        self.scale_pos_weight_ = n_neg / n_pos                 # from the labels of this fit only
        self.model_ = LGBMClassifier(objective="binary", scale_pos_weight=self.scale_pos_weight_, verbose=-1, **params)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return self.model_.feature_importances_


MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "lightgbm": FoldWeightedLGBM,
    "dummy_prior": DummyClassifier,
}
BASE_PARAMS = {
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 5000,
                            "class_weight": "balanced", "random_state": 42},
    "random_forest": {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1,
                      "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
                 "random_state": 42, "n_jobs": 1, "deterministic": True},
    "dummy_prior": {"strategy": "prior"},
}
LOCKED_PARAMS = {
    "logistic_regression": {"class_weight", "l1_ratio", "penalty"},
    "random_forest": {"class_weight"},
    "lightgbm": set(FoldWeightedLGBM.MANAGED),
    "dummy_prior": {"strategy"},
}
TRANSFORMS = {
    "logistic_regression": tuple(LR_TRANSFORMS),           # ("log1p_asinh", "yeo_johnson")
    "random_forest": ("none",),
    "lightgbm": ("none",),
    "dummy_prior": ("none",),
}
DEFAULT_TRANSFORM = {"logistic_regression": "log1p_asinh", "random_forest": "none",
                     "lightgbm": "none", "dummy_prior": "none"}


def resolve_model_params(model_name, overrides=None):
    """Baseline hyperparameters for `model_name` merged with validated overrides."""
    overrides = dict(overrides or {})
    locked = sorted(set(overrides) & LOCKED_PARAMS[model_name])
    if locked:
        raise ModelConfigError(f"{model_name}: these parameters are fixed by the approved design: {locked}")
    params = {**BASE_PARAMS[model_name], **overrides}
    if model_name == "lightgbm":
        FoldWeightedLGBM(params)._checked_params()
        return params
    allowed = set(MODELS[model_name]().get_params())
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ModelConfigError(f"{model_name}: unknown parameters {unknown}")
    return params


def build_model_pipeline(model_name, feature_set, transform=None, model_params=None):
    """Return an unfitted sklearn Pipeline for one model family, feature set and transform.

    transform=None selects the model's default (DEFAULT_TRANSFORM); an explicit
    value must be supported by that model (TRANSFORMS). Nothing is fitted.
    """
    if model_name not in MODELS:
        raise ModelConfigError(f"unknown model {model_name!r}; choose from {sorted(MODELS)}")
    get_feature_set(feature_set)                             # FeatureSetError (a ValueError) if unknown
    transform = DEFAULT_TRANSFORM[model_name] if transform is None else transform
    all_transforms = {t for ts in TRANSFORMS.values() for t in ts}
    if transform not in all_transforms:
        raise ModelConfigError(f"unknown transform {transform!r}; choose from {sorted(all_transforms)}")
    if transform not in TRANSFORMS[model_name]:
        raise ModelConfigError(f"{model_name} does not support transform {transform!r}; "
                               f"supported: {list(TRANSFORMS[model_name])}")
    params = resolve_model_params(model_name, model_params)
    model = FoldWeightedLGBM(params) if model_name == "lightgbm" else MODELS[model_name](**params)

    if model_name == "logistic_regression":
        steps = build_lr_preprocessing_pipeline(feature_set, transform).steps
    else:
        steps = [("check", FeatureFrameCheck(feature_set))]
    return Pipeline(steps + [("model", model)]).set_output(transform="pandas")
