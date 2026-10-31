"""
Tests for Stage 1, validated dataset preparation (src/data/prepare_dataset.py).
Validation rules are tested on synthetic DataFrames; the master dataset is only read.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data import load_master as lm  # noqa: E402
from src.data.prepare_dataset import DatasetValidationError, prepare_dataset, validate_xy_groups  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, FeatureSetError, MAIN_FEATURES  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
APPROVED = {"main": MAIN_FEATURES, "comparison": COMPARISON_FEATURES}


def synthetic_master(n=60, seed=0):
    """A small frame with the master's column roles: identifiers, all 46 features, target, meta_*."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"commit": [f"c{i % 7}" for i in range(n)], "file": "a.py", "class": "", "name": [f"m{i}" for i in range(n)]})
    for f in config.FEATURES:
        df[f] = rng.integers(0, 20, n)
    df["rel_loc"] = rng.normal(0, 5, n)
    df[config.TARGET] = (np.arange(n) % 10 == 0).astype(int)
    df["meta_project_group"] = [f"p{i % 6}" for i in range(n)]
    df["meta_new_name"] = ""
    return df


class ValidateSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_master()

    def xyg(self, feature_set="main", df=None):
        return lm.select_xy_groups(self.df if df is None else df, feature_set)

    def assertRejects(self, X, y, g, feature_set="main", text=None):
        with self.assertRaises(DatasetValidationError) as ctx:
            validate_xy_groups(X, y, g, feature_set)
        if text:
            self.assertIn(text, str(ctx.exception))

    def test_both_configurations_pass(self):
        for name, features in APPROVED.items():
            X, y, g = self.xyg(name)
            validate_xy_groups(X, y, g, name)
            self.assertEqual(list(X.columns), features)
            self.assertEqual((y.name, g.name), ("renamed", "meta_project_group"))
            self.assertTrue(X.index.equals(y.index) and X.index.equals(g.index))

    def test_unsupported_configuration(self):
        X, y, g = self.xyg()
        with self.assertRaises(FeatureSetError):
            validate_xy_groups(X, y, g, "all")
        self.assertTrue(issubclass(FeatureSetError, ValueError))

    def test_wrong_configuration_for_columns(self):
        X, y, g = self.xyg("comparison")
        self.assertRejects(X, y, g, "main", "unexpected ['file_n_methods', 'file_loc']")

    def test_missing_extra_and_reordered_columns(self):
        X, y, g = self.xyg()
        self.assertRejects(X.drop(columns="loc"), y, g, text="missing ['loc']")
        self.assertRejects(X[list(reversed(MAIN_FEATURES))], y, g, text="different order")
        self.assertRejects(X.assign(file_loc=1), y, g, text="unexpected ['file_loc']")

    def test_non_feature_columns_cannot_enter_X(self):
        X, y, g = self.xyg()
        for col in ["commit", "name", "renamed", "meta_project_group", "meta_new_name", "call_sites", "is_async"]:
            self.assertRejects(X.assign(**{col: self.df[col]}), y, g, text=f"non-feature columns in X: ['{col}']")

    def test_invalid_numeric_values(self):
        for value in (np.nan, np.inf, -np.inf):
            X, y, g = self.xyg()
            X = X.astype({"doc_len": float})
            X.iloc[3, X.columns.get_loc("doc_len")] = value
            self.assertRejects(X, y, g, text="NaN or infinite values in: ['doc_len']")
        X, y, g = self.xyg()
        self.assertRejects(X.astype({"loc": str}), y, g, text="non-numeric")
        self.assertRejects(X.astype({"in_class": bool}), y, g, text="non-numeric")

    def test_invalid_target(self):
        X, y, g = self.xyg()
        self.assertRejects(X, y.replace({1: 2}), g, text="0 / 1")
        self.assertRejects(X, y.astype(float), g, text="0 / 1")
        self.assertRejects(X, y.rename("label"), g, text="named")

    def test_invalid_groups(self):
        X, y, g = self.xyg()
        g2 = g.copy(); g2.iloc[0] = ""
        self.assertRejects(X, y, g2, text="empty project")
        g3 = g.astype(object); g3.iloc[0] = None
        self.assertRejects(X, y, g3, text="empty project")
        self.assertRejects(X, y, g.rename("project"), text="named")

    def test_row_counts_and_indices(self):
        X, y, g = self.xyg()
        self.assertRejects(X, y.iloc[:-1], g, text="row counts differ")
        self.assertRejects(X, y.set_axis(y.index + 1), g, text="same index")
        dup = [0] * len(X)
        self.assertRejects(X.set_axis(dup), y.set_axis(dup), g.set_axis(dup), text="duplicate")
        self.assertRejects(X.iloc[:0], y.iloc[:0], g.iloc[:0], text="no rows")

    def test_missing_required_column_in_source(self):
        for col in ["loc", "renamed", "meta_project_group"]:
            with self.assertRaises(KeyError, msg=col):
                lm.select_xy_groups(self.df.drop(columns=col), "main")


@unittest.skipUnless(config.MASTER.exists(), "master dataset not present")
class PrepareMasterTests(unittest.TestCase):
    def test_both_configurations_on_the_master(self):
        before = lm.sha256_of(config.MASTER)
        self.assertEqual(before, CANONICAL)
        for name, features in APPROVED.items():
            X, y, g = prepare_dataset(name)
            self.assertEqual(list(X.columns), features)
            self.assertEqual(X.shape, (31209, len(features)))
            self.assertEqual((int(y.sum()), int((y == 0).sum())), (243, 30966))
            self.assertEqual(g.nunique(), 323)
            self.assertEqual(int(g[y == 1].nunique()), 79)
        self.assertEqual(lm.sha256_of(config.MASTER), CANONICAL)

    def test_main_is_comparison_without_file_size(self):
        Xm, ym, gm = prepare_dataset("main")
        Xc, yc, gc = prepare_dataset("comparison")
        pd.testing.assert_frame_equal(Xm, Xc[MAIN_FEATURES])
        pd.testing.assert_series_equal(ym, yc)
        pd.testing.assert_series_equal(gm, gc)


if __name__ == "__main__":
    unittest.main()
