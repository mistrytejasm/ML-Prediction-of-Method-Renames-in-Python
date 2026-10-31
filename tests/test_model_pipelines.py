"""
Tests for Stage 5, model pipelines and fold-aware class weighting (src/models/pipelines.py).
Synthetic data only: no model is fitted on the master dataset. The only real-data
checks read file hashes.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import PowerTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402
from src.features.feature_sets import FeatureSetError, get_feature_set  # noqa: E402
from src.features.transform_groups import transform_groups  # noqa: E402
from src.models import pipelines as mp  # noqa: E402
from src.preprocessing.transforms import FeatureFrameCheck, FeatureOutputCheck  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
CONFIGS = ("main", "comparison")
COMBINATIONS = [(m, t) for m, ts in mp.TRANSFORMS.items() for t in ts]
SMALL = {"random_forest": {"n_estimators": 10}, "lightgbm": {"n_estimators": 10, "min_child_samples": 2}}


def synthetic(feature_set, n=240, seed=0, pos_rate=0.1):
    rng = np.random.default_rng(seed)
    features = get_feature_set(feature_set)
    X = pd.DataFrame(rng.poisson(5, (n, len(features))).astype(float), columns=features)
    for c in transform_groups(feature_set).asinh:
        X[c] = rng.normal(0, 20, n)
    y = pd.Series((rng.random(n) < pos_rate).astype(int), name="renamed")
    y.iloc[:3] = 1
    return X, y


def build(model, fs="main", transform=None, **params):
    return mp.build_model_pipeline(model, fs, transform, {**SMALL.get(model, {}), **params} or None)


class BuilderTests(unittest.TestCase):
    def test_every_combination_builds_unfitted_and_clones(self):
        for fs in CONFIGS:
            for model, transform in COMBINATIONS:
                pipe = mp.build_model_pipeline(model, fs, transform)
                for _, step in pipe.steps:                    # construction fits nothing
                    self.assertFalse(any(k.endswith("_") and not k.startswith("_") for k in vars(step)),
                                     f"{model} {step}")
                with self.assertRaises(NotFittedError):
                    pipe.predict_proba(synthetic(fs)[0])
                copy = clone(pipe)
                self.assertEqual(repr(copy.get_params()), repr(pipe.get_params()))
                self.assertIsNot(copy.steps[-1][1], pipe.steps[-1][1])

    def test_defaults(self):
        self.assertEqual(mp.DEFAULT_TRANSFORM["logistic_regression"], "log1p_asinh")
        lr = mp.build_model_pipeline("logistic_regression", "main")
        self.assertIsInstance(lr.named_steps["pre"].transformers[0][1].steps[-1][1], StandardScaler)

    def test_invalid_names_fail(self):
        with self.assertRaises(mp.ModelConfigError):
            mp.build_model_pipeline("svm", "main")
        with self.assertRaises(FeatureSetError):
            mp.build_model_pipeline("logistic_regression", "all")
        with self.assertRaises(mp.ModelConfigError):
            mp.build_model_pipeline("logistic_regression", "main", "log")
        for err in (mp.ModelConfigError, FeatureSetError):
            self.assertTrue(issubclass(err, ValueError))

    def test_incompatible_combinations_fail(self):
        for model in ("random_forest", "lightgbm", "dummy_prior"):
            for transform in ("log1p_asinh", "yeo_johnson"):
                with self.assertRaisesRegex(mp.ModelConfigError, "does not support", msg=f"{model} {transform}"):
                    mp.build_model_pipeline(model, "main", transform)
        with self.assertRaisesRegex(mp.ModelConfigError, "does not support"):
            mp.build_model_pipeline("logistic_regression", "main", "none")

    def test_parameters_explicit_and_validated(self):
        lr = mp.build_model_pipeline("logistic_regression", "main").named_steps["model"]
        self.assertEqual((lr.C, lr.l1_ratio, lr.solver, lr.max_iter, lr.class_weight, lr.random_state),
                         (1.0, 0.0, "lbfgs", 5000, "balanced", 42))
        rf = mp.build_model_pipeline("random_forest", "main").named_steps["model"]
        self.assertEqual((rf.n_estimators, rf.class_weight, rf.random_state), (500, "balanced", 42))
        self.assertEqual(mp.build_model_pipeline("random_forest", "main", model_params={"n_estimators": 50})
                         .named_steps["model"].n_estimators, 50)
        bad = [("logistic_regression", {"class_weight": None}), ("logistic_regression", {"penalty": "l1"}),
               ("logistic_regression", {"Cc": 1}), ("random_forest", {"class_weight": None}),
               ("random_forest", {"n_trees": 5}), ("dummy_prior", {"strategy": "most_frequent"}),
               ("lightgbm", {"num_leafs": 7})]
        for model, params in bad:
            with self.assertRaises(mp.ModelConfigError, msg=f"{model} {params}"):
                mp.build_model_pipeline(model, "main", model_params=params)
        self.assertEqual(mp.BASE_PARAMS["random_forest"]["n_estimators"], 500)   # overrides never leak back


class ModelFamilyTests(unittest.TestCase):
    def test_logistic_regression_gets_the_stage4_preprocessing(self):
        for fs in CONFIGS:
            pipe = mp.build_model_pipeline("logistic_regression", fs)
            self.assertEqual([n for n, _ in pipe.steps], ["check", "pre", "output_check", "model"])
            self.assertIsInstance(pipe.named_steps["check"], FeatureFrameCheck)
            self.assertIsInstance(pipe.named_steps["output_check"], FeatureOutputCheck)
            self.assertEqual([n for n, _, _ in pipe.named_steps["pre"].transformers], ["log1p", "asinh", "raw"])
            X, y = synthetic(fs)
            pipe.fit(X, y)
            self.assertIsInstance(pipe.named_steps["model"], LogisticRegression)
            self.assertEqual(list(pipe.named_steps["model"].feature_names_in_), transform_groups(fs).output_order)

    def test_yeo_johnson_variant(self):
        for fs in CONFIGS:
            pipe = mp.build_model_pipeline("logistic_regression", fs, "yeo_johnson")
            pre = pipe.named_steps["pre"]
            (n1, t1, c1), (n2, t2, c2) = pre.transformers
            g = transform_groups(fs)
            self.assertEqual((n1, n2), ("yeo_johnson", "raw"))
            self.assertIsInstance(t1, PowerTransformer)
            self.assertEqual((t1.method, t1.standardize), ("yeo-johnson", True))
            self.assertEqual(c1, list(g.log1p) + list(g.asinh))
            self.assertIsInstance(t2, StandardScaler)       # the transformed columns get no second scaler
            X, y = synthetic(fs)
            Xt = pipe[:-1].fit_transform(X, y)
            self.assertEqual(list(Xt.columns), g.output_order)
            self.assertTrue(np.isfinite(Xt.to_numpy()).all())
            np.testing.assert_allclose(Xt.std(ddof=0), 1, atol=1e-6)
            lambdas = pipe[:-1].named_steps["pre"].named_transformers_["yeo_johnson"].lambdas_
            self.assertEqual(len(lambdas), len(c1))

    def test_trees_and_dummy_get_raw_features(self):
        X, y = synthetic("comparison")
        for model, cls in (("random_forest", RandomForestClassifier), ("lightgbm", mp.FoldWeightedLGBM),
                           ("dummy_prior", DummyClassifier)):
            pipe = build(model, "comparison")
            self.assertEqual([n for n, _ in pipe.steps], ["check", "model"])
            self.assertFalse(any(isinstance(s, (ColumnTransformer, StandardScaler, PowerTransformer)) for _, s in pipe.steps))
            pipe.fit(X, y)
            self.assertIsInstance(pipe.named_steps["model"], cls)
            pd.testing.assert_frame_equal(pipe[:-1].transform(X), X)             # model sees the raw frame
            self.assertEqual(list(pipe.named_steps["model"].feature_names_in_), list(X.columns))

    def test_dummy_uses_the_prior(self):
        X, y = synthetic("main")
        pipe = mp.build_model_pipeline("dummy_prior", "main").fit(X, y)
        self.assertEqual(pipe.named_steps["model"].strategy, "prior")
        np.testing.assert_allclose(pipe.predict_proba(X)[:, 1], y.mean())

    def test_input_check_runs_before_every_model(self):
        X, y = synthetic("main")
        for model, transform in COMBINATIONS:
            with self.assertRaises(ValueError, msg=model):
                build(model, "main", transform).fit(X.assign(loc=np.nan), y)


class FoldWeightedLGBMTests(unittest.TestCase):
    def test_ratio_from_the_labels_of_each_fit(self):
        X, y = synthetic("main", n=300)
        m = mp.FoldWeightedLGBM({"n_estimators": 5, "min_child_samples": 2})
        m.fit(X, y)
        self.assertEqual((m.n_positive_, m.n_negative_), (int(y.sum()), int((y == 0).sum())))
        self.assertAlmostEqual(m.scale_pos_weight_, (y == 0).sum() / y.sum())
        self.assertEqual(m.model_.get_params()["scale_pos_weight"], m.scale_pos_weight_)
        sub = y.iloc[:100]
        m.fit(X.iloc[:100], sub)                                # refit on other labels -> new ratio
        self.assertAlmostEqual(m.scale_pos_weight_, (sub == 0).sum() / sub.sum())

    def test_each_cv_training_fold_has_its_own_ratio(self):
        X, y = synthetic("main", n=600, pos_rate=0.08)
        groups = pd.Series(np.arange(600) % 30)
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups))
        pipe = build("lightgbm")
        res = cross_validate(pipe, X, y, cv=folds, scoring="average_precision", return_estimator=True)
        got = [e.named_steps["model"].scale_pos_weight_ for e in res["estimator"]]
        expected = [float((y.iloc[tr] == 0).sum() / (y.iloc[tr] == 1).sum()) for tr, _ in folds]
        np.testing.assert_allclose(got, expected)
        self.assertGreater(len(set(np.round(got, 9))), 1)                        # folds differ
        global_ratio = (y == 0).sum() / y.sum()
        self.assertFalse(any(np.isclose(g, global_ratio) for g in got))         # no global ratio
        self.assertFalse(hasattr(pipe.named_steps["model"], "scale_pos_weight_"))  # the template stays unfitted

    def test_rejects_degenerate_labels(self):
        X, _ = synthetic("main", n=50)
        m = mp.FoldWeightedLGBM({"n_estimators": 5})
        for y, text in ((np.zeros(50, int), "no positive"), (np.ones(50, int), "no negative"),
                        (np.full(50, 2), "0 / 1")):
            with self.assertRaisesRegex(mp.ModelConfigError, text):
                m.fit(X, y)

    def test_rejects_managed_and_unknown_parameters(self):
        X, y = synthetic("main", n=50)
        for key in ("scale_pos_weight", "is_unbalance", "verbose", "verbosity", "objective", "class_weight"):
            with self.assertRaisesRegex(mp.ModelConfigError, "manages", msg=key):
                mp.FoldWeightedLGBM({key: 1}).fit(X, y)
            with self.assertRaises(mp.ModelConfigError, msg=key):
                mp.build_model_pipeline("lightgbm", "main", model_params={key: 1})
        with self.assertRaisesRegex(mp.ModelConfigError, "unknown"):
            mp.FoldWeightedLGBM({"num_leafs": 3}).fit(X, y)

    def test_clone_and_parameter_inspection(self):
        m = mp.FoldWeightedLGBM({"n_estimators": 5, "random_state": 1})
        X, y = synthetic("main", n=80)
        m.fit(X, y)
        c = clone(m)
        self.assertEqual(c.get_params(), {"lgbm_params": {"n_estimators": 5, "random_state": 1}})
        self.assertFalse(hasattr(c, "scale_pos_weight_"))
        c.set_params(lgbm_params={"n_estimators": 3})
        self.assertEqual(c.get_params()["lgbm_params"], {"n_estimators": 3})
        pipe = mp.build_model_pipeline("lightgbm", "main")
        self.assertIn("model__lgbm_params", pipe.get_params())
        self.assertNotIn("scale_pos_weight", pipe.get_params()["model__lgbm_params"])

    def test_deterministic_and_predicts(self):
        X, y = synthetic("main", n=200)
        a = build("lightgbm").fit(X, y).predict_proba(X)
        b = build("lightgbm").fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (200, 2))
        with self.assertRaises(NotFittedError):
            mp.FoldWeightedLGBM().predict_proba(X)


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class InputsUnchangedTests(unittest.TestCase):
    def test_master_and_split_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        _, meta = sp.load_splits()
        self.assertEqual(meta["split_hash"], SPLIT_HASH)


if __name__ == "__main__":
    unittest.main()
