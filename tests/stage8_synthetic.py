"""
Synthetic data for the Stage 8 tests: a small "master" with project groups, split by the
approved StratifiedGroupKFold (src.evaluation.splits.assign_partitions) and passed through the
real select_development(). No master-dataset row is used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402
from src.evaluation.cross_validate import select_development  # noqa: E402
from src.features.feature_sets import get_feature_set  # noqa: E402

BINARY = ["name_is_private", "name_is_camel", "name_has_digit", "name_starts_verb", "name_single_char",
          "has_varargs", "has_kwargs", "has_docstring", "is_stub", "is_property", "is_static_or_class",
          "in_class", "is_test_file", "style_mismatch", "first_token_shared"]


def synthetic_master(n_projects=60, rows_per_project=30, seed=0):
    """Comparison-42 columns with realistic types, a learnable signal, and 0.08 positive rate."""
    rng = np.random.default_rng(seed)
    n = n_projects * rows_per_project
    features = get_feature_set("comparison")
    X = pd.DataFrame(rng.poisson(4, (n, len(features))).astype(float), columns=features)
    for c in BINARY:
        X[c] = rng.integers(0, 2, n).astype(float)
    for c in [f for f in features if f.startswith("rel_")]:
        X[c] = rng.normal(0, 5, n)
    logit = -3.2 + 0.35 * (X["loc"] - 4) + 0.8 * X["name_short_tokens"].clip(0, 2) - 0.6 * X["has_docstring"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name=config.TARGET)
    groups = pd.Series([f"p{i // rows_per_project:03d}" for i in range(n)], name="meta_project_group")
    keys = pd.DataFrame({"commit": [f"c{i % 97:03d}" for i in range(n)], "file": [f"f{i % 13}.py" for i in range(n)],
                         "class": ["" if i % 3 else "K" for i in range(n)], "name": [f"m{i}" for i in range(n)]})
    return X, y, groups, keys


def synthetic_split(X, y, groups, keys):
    partition, inner_fold = sp.assign_partitions(X, y, groups)
    assign = keys.copy()
    assign.insert(0, "master_row", np.arange(len(X)))
    assign["meta_project_group"] = groups.to_numpy()
    assign[config.TARGET] = y.to_numpy()
    assign["partition"] = partition
    assign["inner_fold"] = inner_fold
    assign = assign[sp.COLUMNS]
    meta = {"counts": sp.summarize(assign), "split_hash": sp.split_hash(assign),
            "dataset_sha256": "synthetic", "assignment_file_sha256": "synthetic"}
    return assign, meta


def synthetic_development(feature_set="main", seed=0, transform_X=None, **kw):
    X, y, groups, keys = synthetic_master(seed=seed, **kw)
    if transform_X is not None:
        X = transform_X(X)
    assign, meta = synthetic_split(X, y, groups, keys)
    return select_development(X[get_feature_set(feature_set)], y, groups, assign, meta)


def synthetic_loader(seed=0, transform_X=None):
    """A data_loader for run_stage8.main(): synthetic development data per feature set."""
    return lambda fs: synthetic_development(fs, seed=seed, transform_X=transform_X)
