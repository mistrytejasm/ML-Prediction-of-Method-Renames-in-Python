r"""
prepare_dataset.py
Stage 1: validated (X, y, groups) for an approved feature configuration.

  prepare_dataset()    load_master() (SHA-256 checked) -> select_xy_groups()
                       -> validate_xy_groups(); returns (X, y, groups)
  validate_xy_groups() checks an (X, y, groups) triple and raises
                       DatasetValidationError on the first problem

select_xy_groups() in load_master.py selects the approved columns but does not
look at the values: NaN, infinity, a label other than 0 / 1 or an empty project
group pass through it. This module adds those checks without changing the
loader. Nothing is imputed, fitted or written; the master is only read.

Run as a check:  python src/data/prepare_dataset.py
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import GROUP_COLUMN, load_master, select_xy_groups  # noqa: E402
from src.features.feature_sets import EXCLUDED_LEAKAGE, DROPPED_FEATURES, get_feature_set  # noqa: E402

log = logging.getLogger("prepare_dataset")


class DatasetValidationError(ValueError):
    """X, y or groups break the rules for model input."""


def validate_features(X, feature_set):
    """Raise DatasetValidationError unless X alone is valid model input for `feature_set`.

    Checks: X is a non-empty DataFrame with exactly the approved columns in the
    approved order (no duplicates, no identifier, target, meta_*, leakage or
    dropped column), and every column is numeric (not bool) and finite.
    Also used by the pipeline's input check (src/preprocessing/transforms.py).
    """
    features = get_feature_set(feature_set)          # FeatureSetError (a ValueError) if unsupported
    if not isinstance(X, pd.DataFrame):
        raise DatasetValidationError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    if len(X) == 0:
        raise DatasetValidationError("X has no rows")

    columns = list(X.columns)
    dup = sorted({str(c) for c in columns if columns.count(c) > 1})
    if dup:
        raise DatasetValidationError(f"duplicate columns in X: {dup}")
    forbidden = set(config.ID_COLS) | {config.TARGET} | set(EXCLUDED_LEAKAGE) | set(DROPPED_FEATURES)
    leaked = [c for c in columns if c in forbidden or str(c).startswith("meta_")]
    if leaked:
        raise DatasetValidationError(f"non-feature columns in X: {leaked}")
    if columns != features:
        missing = [f for f in features if f not in columns]
        extra = [c for c in columns if c not in features]
        detail = f"missing {missing}, unexpected {extra}" if missing or extra else "same names, different order"
        raise DatasetValidationError(f"X columns differ from the {feature_set!r} configuration: {detail}")

    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    if non_numeric:
        raise DatasetValidationError(f"non-numeric feature columns: {non_numeric}")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = [c for c, ok in zip(columns, np.isfinite(values).all(axis=0)) if not ok]
        raise DatasetValidationError(f"NaN or infinite values in: {bad}")


def validate_xy_groups(X, y, groups, feature_set):
    """Raise DatasetValidationError unless (X, y, groups) is valid input for `feature_set`.

    Checks: validate_features(X); y is named `renamed` and holds only 0 / 1;
    groups is named meta_project_group and has no missing or empty values;
    X, y and groups have the same length and the same, unique index.
    """
    if not isinstance(y, pd.Series) or not isinstance(groups, pd.Series):
        raise DatasetValidationError("y and groups must be pandas Series")
    validate_features(X, feature_set)

    if not (len(X) == len(y) == len(groups)):
        raise DatasetValidationError(f"row counts differ: X {len(X)}, y {len(y)}, groups {len(groups)}")
    if not (X.index.equals(y.index) and X.index.equals(groups.index)):
        raise DatasetValidationError("X, y and groups do not share the same index")
    if not X.index.is_unique:
        raise DatasetValidationError("the index has duplicate labels")

    if y.name != config.TARGET:
        raise DatasetValidationError(f"y is named {y.name!r}, expected {config.TARGET!r}")
    if not pd.api.types.is_integer_dtype(y) or not set(y.unique()) <= {0, 1}:
        raise DatasetValidationError(f"y must be integer 0 / 1, got values {sorted(map(str, y.unique()))[:5]}")

    if groups.name != GROUP_COLUMN:
        raise DatasetValidationError(f"groups is named {groups.name!r}, expected {GROUP_COLUMN!r}")
    if groups.isna().any() or (groups.astype(str).str.strip() == "").any():
        raise DatasetValidationError("groups has missing or empty project identifiers")


def prepare_dataset(feature_set="main", path=config.MASTER):
    """Return validated (X, y, groups) from the checksum-verified master dataset."""
    df = load_master(path)                            # MasterChecksumError if the file changed
    X, y, groups = select_xy_groups(df, feature_set)  # KeyError if a required column is missing
    validate_xy_groups(X, y, groups, feature_set)
    return X, y, groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("main", "comparison"):
        X, y, groups = prepare_dataset(name)
        log.info("%-10s X %s | positives %d | negatives %d | groups %d",
                 name, X.shape, int(y.sum()), int((y == 0).sum()), groups.nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
