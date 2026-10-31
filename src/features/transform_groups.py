"""
Transformation groups for Logistic Regression preprocessing (Stage 2).

Definitions only: nothing here fits or applies a transform. Every selected
feature belongs to exactly one group:

  log1p  non-negative, long-tailed counts
           main 8: loc, n_params, n_returns, n_calls, cyclomatic, max_nesting,
                   doc_len, class_n_methods
           comparison 10: the same 8 + file_n_methods, file_loc
  asinh  signed, long-tailed relative features (5): rel_loc, rel_n_params,
         rel_n_calls, rel_cyclomatic, rel_doc_len
  raw    every other selected feature (27 in both configurations)

Groups are derived from the approved feature list (src/features/feature_sets.py),
never from column positions. Within each group features keep the configuration's
order (= the master dataset's column order). The output order of a later
ColumnTransformer is TransformGroups.output_order: log1p + asinh + raw.
See the Phase 1 audit (reports/audit/preprocessing/) for the skewness evidence.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_sets import (COMPARISON_FEATURES, COMPARISON_ONLY, FEATURE_SETS,  # noqa: E402
                                       get_feature_set)

LOG1P_BASE = ["loc", "n_params", "n_returns", "n_calls", "cyclomatic", "max_nesting", "doc_len", "class_n_methods"]
LOG1P_FEATURES = LOG1P_BASE + COMPARISON_ONLY      # file_loc / file_n_methods occur only in comparison
ASINH_FEATURES = ["rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic", "rel_doc_len"]
GROUP_NAMES = ("log1p", "asinh", "raw")
EXPECTED_SIZES = {"main": (8, 5, 27), "comparison": (10, 5, 27)}


class TransformGroupError(ValueError):
    """An unsupported feature configuration or an invalid group definition."""


@dataclass(frozen=True)
class TransformGroups:
    feature_set: str
    log1p: tuple
    asinh: tuple
    raw: tuple

    @property
    def output_order(self):
        """Column order after a ColumnTransformer with branches log1p, asinh, raw."""
        return list(self.log1p + self.asinh + self.raw)

    def as_dict(self):
        return {"log1p": list(self.log1p), "asinh": list(self.asinh), "raw": list(self.raw)}


def validate_groups(groups, features):
    """Raise TransformGroupError unless `groups` (name -> list) partitions `features` exactly.

    Every feature must be in exactly one group; no group may hold a duplicate, a
    feature outside `features`, or a feature its transform is not defined for.
    """
    features = list(features)
    if set(groups) != set(GROUP_NAMES):
        raise TransformGroupError(f"groups must be exactly {list(GROUP_NAMES)}, got {sorted(groups)}")
    members = [f for name in GROUP_NAMES for f in groups[name]]
    dup = sorted({f for f in members if members.count(f) > 1})
    if dup:
        raise TransformGroupError(f"features in more than one group or listed twice: {dup}")
    extra = [f for f in members if f not in features]
    if extra:
        raise TransformGroupError(f"features not in the selected configuration: {extra}")
    missing = [f for f in features if f not in members]
    if missing:
        raise TransformGroupError(f"features in no group: {missing}")
    wrong_log = [f for f in groups["log1p"] if f not in LOG1P_FEATURES]
    wrong_asinh = [f for f in groups["asinh"] if f not in ASINH_FEATURES]
    if wrong_log or wrong_asinh:
        raise TransformGroupError(f"transform not defined for: log1p {wrong_log}, asinh {wrong_asinh}")
    return groups


def transform_groups(feature_set):
    """Return the TransformGroups of an approved configuration: 'main' (40) or 'comparison' (42)."""
    if feature_set not in FEATURE_SETS:
        raise TransformGroupError(f"unsupported feature configuration {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    features = get_feature_set(feature_set)
    log1p = [f for f in features if f in LOG1P_FEATURES]
    asinh = [f for f in features if f in ASINH_FEATURES]
    raw = [f for f in features if f not in log1p and f not in asinh]
    validate_groups({"log1p": log1p, "asinh": asinh, "raw": raw}, features)
    sizes = (len(log1p), len(asinh), len(raw))
    if sizes != EXPECTED_SIZES[feature_set]:
        raise TransformGroupError(f"{feature_set}: group sizes {sizes}, expected {EXPECTED_SIZES[feature_set]}")
    return TransformGroups(feature_set, tuple(log1p), tuple(asinh), tuple(raw))


def groups_for_features(features):
    """TransformGroups for an explicit feature list; it must equal an approved configuration exactly."""
    features = list(features)
    for name in FEATURE_SETS:
        if features == get_feature_set(name):
            return transform_groups(name)
    raise TransformGroupError("feature list is not an approved configuration (main 40 or comparison 42, "
                              f"in the approved order); got {len(features)} features")


def check_log1p_domain(X, groups):
    """Raise TransformGroupError if a log1p column of X has a negative or non-finite value (read-only check)."""
    cols = list(groups.log1p)
    values = X[cols].to_numpy(dtype=float)
    bad = [c for c, ok in zip(cols, (np.isfinite(values) & (values >= 0)).all(axis=0)) if not ok]
    if bad:
        raise TransformGroupError(f"log1p columns need finite values >= 0: {bad}")


def _check_definitions():
    for name, feature_list in (("log1p", LOG1P_FEATURES), ("asinh", ASINH_FEATURES)):
        outside = [f for f in feature_list if f not in COMPARISON_FEATURES]
        if outside:
            raise TransformGroupError(f"{name} lists features outside the approved configurations: {outside}")
    if set(LOG1P_FEATURES) & set(ASINH_FEATURES):
        raise TransformGroupError("a feature is listed for both log1p and asinh")
    for name in FEATURE_SETS:
        transform_groups(name)


_check_definitions()
