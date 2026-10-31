"""
Stage 4: input validation and Logistic Regression preprocessing.

  FeatureFrameCheck(feature_set)         stateless input check; learns nothing
  make_lr_preprocessor(feature_set)      ColumnTransformer over the Stage 2 groups
  FeatureOutputCheck(feature_set)        checks names, order and finiteness of the output
  build_lr_preprocessing_pipeline(fs, transform="log1p_asinh")
                                         Pipeline(check -> pre -> output_check), pandas output
  make_lr_preprocessor_yeo_johnson(fs)   Yeo-Johnson variant (Proposed experiment, Stage 5)

ColumnTransformer branches (each column is scaled exactly once):

  log1p  checked log1p (rejects negative / non-finite) -> StandardScaler   8 | 10 columns
  asinh  arcsinh                                       -> StandardScaler   5 columns
  raw                                                     StandardScaler   27 columns

Output order = TransformGroups.output_order (log1p + asinh + raw), with the
original feature names (verbose_feature_names_out=False). The classifier is
deliberately not part of this stage. Every learned value (the scalers' mean_ and
scale_) comes only from the rows passed to fit(): in cross-validation, the
training fold. Nothing here is fitted on the master dataset.
Yeo-Johnson (transform="yeo_johnson") is a separate experiment: one
PowerTransformer(standardize=True) over the log1p + asinh columns, no second
scaler on them, StandardScaler on the raw columns; same output order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.prepare_dataset import validate_features  # noqa: E402
from src.features.feature_sets import get_feature_set  # noqa: E402
from src.features.transform_groups import transform_groups  # noqa: E402


class PreprocessingInputError(ValueError):
    """Values a transform is not defined for, or output that breaks the expected layout."""


def checked_log1p(X):
    """log1p for counts: refuses negative or non-finite values instead of returning NaN or a silent value."""
    values = np.asarray(X, dtype=float)
    ok = np.isfinite(values) & (values >= 0)
    if not ok.all():
        cols = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(values.shape[1]))
        bad = [c for c, good in zip(cols, ok.all(axis=0)) if not good]
        raise PreprocessingInputError(f"log1p columns need finite values >= 0: {bad}")
    return np.log1p(X)


class FeatureFrameCheck(TransformerMixin, BaseEstimator):
    """Pipeline step that rejects anything but the approved feature frame; returns X unchanged.

    fit() checks X against the approved configuration and stores feature_names_in_;
    transform() checks every later X against those names. Nothing is imputed or learned.
    Errors are DatasetValidationError / FeatureSetError (both ValueError).
    """

    def __init__(self, feature_set):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        validate_features(X, self.feature_set)
        self.feature_names_in_ = np.asarray(get_feature_set(self.feature_set), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        validate_features(X, self.feature_set)
        if list(X.columns) != list(self.feature_names_in_):
            raise PreprocessingInputError("X columns differ from the columns seen in fit()")
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()


class FeatureOutputCheck(TransformerMixin, BaseEstimator):
    """Pipeline step after the ColumnTransformer: exact names, order and count, all values finite."""

    def __init__(self, feature_set):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(transform_groups(self.feature_set).output_order, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self._check(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._check(X)
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_.copy()

    def _check(self, X):
        names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        if names is not None and names != list(self.feature_names_in_):
            raise PreprocessingInputError(f"output columns differ from {list(self.feature_names_in_)}: {names}")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise PreprocessingInputError(f"output has shape {values.shape}, expected {self.n_features_in_} columns")
        if not np.isfinite(values).all():
            raise PreprocessingInputError("output contains NaN or infinite values")


def make_lr_preprocessor(feature_set):
    """Unfitted ColumnTransformer for Logistic Regression: log1p / asinh / raw, each scaled once."""
    groups = transform_groups(feature_set)
    pre = ColumnTransformer(
        [("log1p", make_pipeline(FunctionTransformer(checked_log1p, feature_names_out="one-to-one"),
                                 StandardScaler()), list(groups.log1p)),
         ("asinh", make_pipeline(FunctionTransformer(np.arcsinh, feature_names_out="one-to-one"),
                                 StandardScaler()), list(groups.asinh)),
         ("raw", StandardScaler(), list(groups.raw))],
        remainder="drop", verbose_feature_names_out=False)
    return pre


def make_lr_preprocessor_yeo_johnson(feature_set):
    """Unfitted ColumnTransformer for the Yeo-Johnson experiment (Proposed configuration, not the default).

    The log1p and asinh columns go through one PowerTransformer(method="yeo-johnson",
    standardize=True), which learns a lambda plus a mean and std per column, so no
    second StandardScaler is applied to them; the raw columns get a StandardScaler.
    Output order is the same as the default: log1p + asinh + raw.
    """
    groups = transform_groups(feature_set)
    pre = ColumnTransformer(
        [("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True),
          list(groups.log1p) + list(groups.asinh)),
         ("raw", StandardScaler(), list(groups.raw))],
        remainder="drop", verbose_feature_names_out=False)
    return pre


LR_TRANSFORMS = {"log1p_asinh": make_lr_preprocessor, "yeo_johnson": make_lr_preprocessor_yeo_johnson}


def build_lr_preprocessing_pipeline(feature_set, transform="log1p_asinh"):
    """Unfitted preprocessing Pipeline for `feature_set` ('main' or 'comparison'); no classifier.

    transform: 'log1p_asinh' (primary, default) or 'yeo_johnson' (separate experiment).
    Steps: check (FeatureFrameCheck) -> pre (ColumnTransformer) -> output_check
    (FeatureOutputCheck). Outputs a pandas DataFrame whose columns are
    transform_groups(feature_set).output_order.
    """
    transform_groups(feature_set)                     # TransformGroupError (a ValueError) if unsupported
    if transform not in LR_TRANSFORMS:
        raise PreprocessingInputError(f"unknown transform {transform!r}; choose from {sorted(LR_TRANSFORMS)}")
    pipe = Pipeline([("check", FeatureFrameCheck(feature_set)),
                     ("pre", LR_TRANSFORMS[transform](feature_set)),
                     ("output_check", FeatureOutputCheck(feature_set))])
    return pipe.set_output(transform="pandas")
