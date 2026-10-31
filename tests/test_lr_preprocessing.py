"""
Tests for Stage 4, input validation and Logistic Regression preprocessing
(src/preprocessing/transforms.py). Synthetic data for almost everything; the
real-data tests only read the saved split, and one fits the preprocessing on a
single development TRAINING fold (never on the whole master). No model is trained.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.data.prepare_dataset import DatasetValidationError, prepare_dataset  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402
from src.features.feature_sets import FeatureSetError, get_feature_set  # noqa: E402
from src.features.transform_groups import transform_groups  # noqa: E402
from src.preprocessing import transforms as tr  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
CONFIGS = ("main", "comparison")


def synthetic(feature_set, n=300, seed=0, shift=0.0, scale=1.0):
    """Counts >= 0 everywhere, signed rel_* columns, binaries; `shift`/`scale` move the distribution."""
    rng = np.random.default_rng(seed)
    features = get_feature_set(feature_set)
    X = pd.DataFrame(rng.poisson(6, (n, len(features))) * scale + shift, columns=features).astype(float)
    for c in transform_groups(feature_set).asinh:
        X[c] = rng.normal(0, 40, n) * scale + shift
    X["in_class"] = rng.integers(0, 2, n)                 # an int binary column stays int
    return X


def scaler_params(pipe):
    pre = pipe.named_steps["pre"]
    out = {}
    for name in ("log1p", "asinh", "raw"):
        step = pre.named_transformers_[name]
        sc = step[-1] if hasattr(step, "steps") else step
        out[name] = {k: np.array(getattr(sc, k), copy=True) for k in ("mean_", "scale_", "var_", "n_samples_seen_")}
    return out


class InputCheckTests(unittest.TestCase):
    def test_clone_and_params(self):
        for fs in CONFIGS:
            check = tr.FeatureFrameCheck(fs).fit(synthetic(fs))
            copy = clone(check)
            self.assertEqual(copy.get_params(), {"feature_set": fs})
            self.assertFalse(hasattr(copy, "feature_names_in_"))          # clone is unfitted
            self.assertEqual(list(check.feature_names_in_), get_feature_set(fs))
            self.assertEqual(check.n_features_in_, len(get_feature_set(fs)))

    def test_passes_frame_through_unchanged(self):
        for fs in CONFIGS:
            X = synthetic(fs)
            out = tr.FeatureFrameCheck(fs).fit(X).transform(X)
            pd.testing.assert_frame_equal(out, X)

    def test_rejects_bad_columns_at_fit_and_transform(self):
        for fs in CONFIGS:
            X = synthetic(fs)
            features = get_feature_set(fs)
            fitted = tr.FeatureFrameCheck(fs).fit(X)
            bad = {"missing": X.drop(columns="loc"), "extra": X.assign(extra=1.0),
                   "reordered": X[features[::-1]], "duplicate": pd.concat([X, X[["loc"]]], axis=1),
                   "identifier": X.assign(name="f"), "meta": X.assign(meta_project_group="p"),
                   "numpy": X.to_numpy()}
            for label, Xb in bad.items():
                with self.assertRaises(ValueError, msg=f"{fs} fit {label}"):
                    tr.FeatureFrameCheck(fs).fit(Xb)
                with self.assertRaises(ValueError, msg=f"{fs} transform {label}"):
                    fitted.transform(Xb)

    def test_rejects_nan_inf_non_numeric_and_bool(self):
        X = synthetic("main")
        fitted = tr.FeatureFrameCheck("main").fit(X)
        for value in (np.nan, np.inf, -np.inf):
            Xb = X.copy()
            Xb.loc[5, "doc_len"] = value
            with self.assertRaisesRegex(DatasetValidationError, "NaN or infinite"):
                tr.FeatureFrameCheck("main").fit(Xb)
            with self.assertRaisesRegex(DatasetValidationError, "NaN or infinite"):
                fitted.transform(Xb)
        for Xb in (X.astype({"loc": str}), X.astype({"in_class": bool})):
            with self.assertRaisesRegex(DatasetValidationError, "non-numeric"):
                fitted.transform(Xb)

    def test_unsupported_configuration_and_unfitted_use(self):
        with self.assertRaises(FeatureSetError):
            tr.FeatureFrameCheck("all").fit(synthetic("main"))
        with self.assertRaises(ValueError):
            tr.build_lr_preprocessing_pipeline("all")
        with self.assertRaises(NotFittedError):
            tr.FeatureFrameCheck("main").transform(synthetic("main"))
        for err in (DatasetValidationError, FeatureSetError, tr.PreprocessingInputError):
            self.assertTrue(issubclass(err, ValueError))


class LRPreprocessingTests(unittest.TestCase):
    def test_dimensions_names_and_order(self):
        for fs, n in (("main", 40), ("comparison", 42)):
            pipe = tr.build_lr_preprocessing_pipeline(fs)
            Z = pipe.fit_transform(synthetic(fs))
            order = transform_groups(fs).output_order
            self.assertIsInstance(Z, pd.DataFrame)
            self.assertEqual(Z.shape, (300, n))
            self.assertEqual(list(Z.columns), order)
            self.assertEqual(list(pipe.get_feature_names_out()), order)
            self.assertEqual(list(pipe.named_steps["pre"].get_feature_names_out()), order)
            self.assertEqual(len(set(Z.columns)), n)
            self.assertTrue(np.isfinite(Z.to_numpy()).all())
        comp = tr.build_lr_preprocessing_pipeline("comparison").fit(synthetic("comparison"))
        self.assertEqual(list(comp.get_feature_names_out()[8:10]), ["file_n_methods", "file_loc"])

    def test_values_follow_the_documented_formulas(self):
        X = synthetic("main")
        Z = tr.build_lr_preprocessing_pipeline("main").fit_transform(X)
        for col, f in (("loc", np.log1p), ("rel_loc", np.arcsinh), ("name_len", lambda v: v)):
            v = f(X[col].to_numpy(dtype=float))
            np.testing.assert_allclose(Z[col], (v - v.mean()) / v.std(), atol=1e-10)

    def test_asinh_handles_negative_values(self):
        X = synthetic("main")
        self.assertTrue((X["rel_loc"] < 0).any())
        Z = tr.build_lr_preprocessing_pipeline("main").fit_transform(X)
        self.assertTrue(np.isfinite(Z[list(transform_groups("main").asinh)].to_numpy()).all())

    def test_log1p_rejects_negative_values(self):
        for fs in CONFIGS:
            for col in transform_groups(fs).log1p:
                Xb = synthetic(fs)
                Xb.loc[0, col] = -0.5                    # np.log1p would quietly return a finite number
                with self.assertRaisesRegex(tr.PreprocessingInputError, col, msg=f"{fs} fit {col}"):
                    tr.build_lr_preprocessing_pipeline(fs).fit(Xb)
            fitted = tr.build_lr_preprocessing_pipeline(fs).fit(synthetic(fs))
            Xv = synthetic(fs, seed=1)
            Xv.loc[3, "loc"] = -2.0
            with self.assertRaises(tr.PreprocessingInputError):
                fitted.transform(Xv)

    def test_validation_data_never_refits(self):
        """Fit on training data, transform very different validation data: fitted parameters stay frozen."""
        for fs in CONFIGS:
            train = synthetic(fs, seed=1)
            val = synthetic(fs, seed=2, shift=50.0, scale=10.0)
            pipe = tr.build_lr_preprocessing_pipeline(fs).fit(train)
            before = scaler_params(pipe)
            Zv = pipe.transform(val)
            pipe.transform(val.iloc[:7])
            after = scaler_params(pipe)
            for branch in before:
                for key in before[branch]:
                    np.testing.assert_array_equal(before[branch][key], after[branch][key], err_msg=f"{fs} {branch} {key}")
            # validation output uses the TRAINING mean / std, so it is not re-centred
            v = np.log1p(val["loc"].to_numpy(dtype=float))
            t = np.log1p(train["loc"].to_numpy(dtype=float))
            np.testing.assert_allclose(Zv["loc"], (v - t.mean()) / t.std(), atol=1e-10)
            self.assertGreater(abs(Zv["loc"].mean()), 1.0)
            self.assertEqual(int(after["raw"]["n_samples_seen_"]), len(train))

    def test_pipeline_fit_transform_behaviour(self):
        X = synthetic("comparison")
        pipe = tr.build_lr_preprocessing_pipeline("comparison")
        with self.assertRaises(NotFittedError):
            pipe.transform(X)
        a = clone(pipe).fit_transform(X)
        b = clone(pipe).fit(X).transform(X)
        pd.testing.assert_frame_equal(a, b)
        unfitted = clone(pipe.fit(X))
        self.assertFalse(hasattr(unfitted.named_steps["check"], "feature_names_in_"))
        self.assertEqual([s for s, _ in pipe.steps], ["check", "pre", "output_check"])

    def test_each_column_scaled_exactly_once(self):
        """Double scaling is invisible in training output (a second scaler is the identity), so check the structure."""
        for fs in CONFIGS:
            pipe = tr.build_lr_preprocessing_pipeline(fs).fit(synthetic(fs))
            pre = pipe.named_steps["pre"]
            self.assertFalse(any(isinstance(s, StandardScaler) for _, s in pipe.steps))   # none outside the branches
            per_branch = {}
            for name, step, cols in pre.transformers_:
                if name == "remainder":
                    continue
                steps = [s for _, s in step.steps] if hasattr(step, "steps") else [step]
                per_branch[name] = (sum(isinstance(s, StandardScaler) for s in steps), len(cols))
            g = transform_groups(fs)
            self.assertEqual(per_branch, {"log1p": (1, len(g.log1p)), "asinh": (1, len(g.asinh)), "raw": (1, 27)})
            Z = pipe.transform(synthetic(fs))
            np.testing.assert_allclose(Z.mean().to_numpy(), 0, atol=1e-10)
            np.testing.assert_allclose(Z.std(ddof=0).to_numpy(), 1, atol=1e-10)

    def test_output_check_catches_a_broken_layout(self):
        check = tr.FeatureOutputCheck("main").fit(pd.DataFrame(np.zeros((3, 40)), columns=transform_groups("main").output_order))
        good = pd.DataFrame(np.zeros((3, 40)), columns=transform_groups("main").output_order)
        check.transform(good)
        for bad in (good[good.columns[::-1]], good.iloc[:, :39], good.replace(0.0, np.nan)):
            with self.assertRaises(tr.PreprocessingInputError):
                check.transform(bad)


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class SavedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assign, cls.meta = sp.load_splits()
        cls.data = {fs: prepare_dataset(fs) for fs in CONFIGS}

    def test_both_configurations_use_the_same_saved_split(self):
        self.assertEqual(self.meta["split_hash"], SPLIT_HASH)
        self.assertEqual(self.meta["feature_sets"], {"main": 40, "comparison": 42})
        dev, test = sp.outer_indices(self.assign)
        (Xm, ym, gm), (Xc, yc, gc) = self.data["main"], self.data["comparison"]
        self.assertTrue(Xm.index.equals(Xc.index))
        self.assertTrue(Xm.iloc[dev].index.equals(Xc.iloc[dev].index))
        self.assertTrue(gm.iloc[test].equals(gc.iloc[test]))
        for train, val in sp.inner_folds(self.assign):
            self.assertTrue(Xm.iloc[dev].iloc[train].index.equals(Xc.iloc[dev].iloc[train].index))

    def test_fit_on_one_training_fold_only(self):
        """Fitted on development fold-0 TRAINING rows, applied to its validation rows; never on all rows."""
        dev, _ = sp.outer_indices(self.assign)
        train, val = sp.inner_folds(self.assign)[0]
        for fs in CONFIGS:
            X = self.data[fs][0].iloc[dev]
            pipe = tr.build_lr_preprocessing_pipeline(fs).fit(X.iloc[train])
            self.assertEqual(int(scaler_params(pipe)["raw"]["n_samples_seen_"]), len(train))
            Zv = pipe.transform(X.iloc[val])
            self.assertEqual(Zv.shape, (len(val), len(get_feature_set(fs))))
            self.assertTrue(np.isfinite(Zv.to_numpy()).all())

    def test_inputs_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        self.assertEqual(sha256_of(sp.SPLIT_CSV), self.meta["assignment_file_sha256"])


if __name__ == "__main__":
    unittest.main()
