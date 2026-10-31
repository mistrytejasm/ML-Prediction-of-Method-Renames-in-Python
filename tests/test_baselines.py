"""
Tests for Stage 7, the stratified-random baseline and the exploratory naming heuristic
(src/models/baselines.py, registered in src/models/pipelines.py).
Synthetic data only; nothing is fitted on the master dataset.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, MAIN_FEATURES, get_feature_set  # noqa: E402
from src.features.method_features import name_features  # noqa: E402
from src.models import baselines as bl  # noqa: E402
from src.models import pipelines as mp  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
BINARY = ["name_is_private", "name_is_camel", "name_has_digit", "name_starts_verb", "name_single_char",
          "has_varargs", "has_kwargs", "has_docstring", "is_stub", "is_property", "is_static_or_class",
          "in_class", "is_test_file", "style_mismatch", "first_token_shared"]


def synthetic(feature_set="main", n=400, rate=0.1, seed=0):
    """Approved columns with realistic types: binaries 0/1, counts >= 0, signed rel_* columns."""
    rng = np.random.default_rng(seed)
    features = get_feature_set(feature_set)
    X = pd.DataFrame(rng.poisson(3, (n, len(features))).astype(float), columns=features)
    for c in BINARY:
        X[c] = rng.integers(0, 2, n)
    for c in [f for f in features if f.startswith("rel_")]:
        X[c] = rng.normal(0, 5, n)
    y = pd.Series((rng.random(n) < rate).astype(int), name="renamed")
    y.iloc[:2] = [1, 0]
    return X, y


class StratifiedRandomTests(unittest.TestCase):
    def test_reproducible_and_seed_sensitive(self):
        X, y = synthetic()
        a = bl.StratifiedRandomClassifier(42).fit(X, y).predict_proba(X)
        b = bl.StratifiedRandomClassifier(42).fit(X, y).predict_proba(X)
        c = bl.StratifiedRandomClassifier(7).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        m = bl.StratifiedRandomClassifier(42).fit(X, y)
        np.testing.assert_array_equal(m.predict_proba(X), m.predict_proba(X))      # repeated calls agree

    def test_uses_the_training_prevalence(self):
        X, y = synthetic(n=20000, rate=0.05)
        m = bl.StratifiedRandomClassifier().fit(X, y)
        self.assertEqual(m.prevalence_, y.mean())
        s = m.predict_proba(X)[:, 1]
        self.assertAlmostEqual(s.mean(), y.mean(), delta=0.002)                   # mean score = pi
        self.assertAlmostEqual(m.predict(X).mean(), y.mean(), delta=0.005)        # flags a pi share
        self.assertLessEqual(s.max(), 2 * y.mean())
        Xh, yh = synthetic(n=20000, rate=0.7, seed=1)                             # pi > 0.5 branch
        sh = bl.StratifiedRandomClassifier().fit(Xh, yh).predict_proba(Xh)[:, 1]
        self.assertAlmostEqual(sh.mean(), yh.mean(), delta=0.005)

    def test_scores_are_valid_probabilities_and_a_random_ranking(self):
        X, y = synthetic(n=5000, rate=0.008)
        proba = bl.StratifiedRandomClassifier().fit(X, y).predict_proba(X)
        self.assertTrue(np.isfinite(proba).all() and (proba >= 0).all() and (proba <= 1).all())
        np.testing.assert_allclose(proba.sum(axis=1), 1)
        self.assertEqual(len(np.unique(proba[:, 1])), 5000)                        # no underflow ties
        M.validate_probabilities(proba[:, 1])

    def test_ignores_feature_values_and_validation_labels(self):
        X, y = synthetic()
        Xv, yv = synthetic(seed=5)
        m = bl.StratifiedRandomClassifier().fit(X, y)
        base = m.predict_proba(Xv)
        np.testing.assert_array_equal(base, m.predict_proba(Xv * 3 + 1))          # values do not matter
        refit = bl.StratifiedRandomClassifier().fit(X, y)                         # validation labels never seen
        np.testing.assert_array_equal(base, refit.predict_proba(Xv))
        other = bl.StratifiedRandomClassifier().fit(X, y.sample(frac=1, random_state=3).reset_index(drop=True))
        np.testing.assert_array_equal(base, other.predict_proba(Xv))              # same pi -> same scores

    def test_rejects_invalid_inputs(self):
        X, y = synthetic(n=50)
        for bad in (np.zeros(50, int), np.ones(50, int), np.full(50, 2), y.astype(bool), y.iloc[:-1]):
            with self.assertRaises(bl.BaselineInputError):
                bl.StratifiedRandomClassifier().fit(X, bad)
        for seed in (None, 1.5, True):
            with self.assertRaises(bl.BaselineInputError, msg=str(seed)):
                bl.StratifiedRandomClassifier(seed).fit(X, y)
        with self.assertRaises(NotFittedError):
            bl.StratifiedRandomClassifier().predict_proba(X)
        self.assertTrue(issubclass(bl.BaselineInputError, ValueError))


def named_methods(names, style_mismatch=None, feature_set="main"):
    """Rows whose name features come from the real extractor (name_features); other features neutral."""
    rows = []
    for i, n in enumerate(names):
        row = {f: 0.0 for f in get_feature_set(feature_set)}
        row.update({k: float(v) for k, v in name_features(n).items()})
        row["style_mismatch"] = float((style_mismatch or {}).get(n, 0))
        rows.append(row)
    return pd.DataFrame(rows, index=names)[get_feature_set(feature_set)]


FLAG_NAMES = ["short_token", "vowelless_token", "single_character_name", "contains_digit",
              "first_word_not_in_verb_list", "camel_case_in_snake_file"]


class NamingHeuristicTests(unittest.TestCase):
    def fitted(self, X):
        return bl.NamingHeuristicClassifier().fit(X, np.r_[1, np.zeros(len(X) - 1, int)])

    def test_flags_exactly_as_documented(self):
        X = named_methods(["get_value", "get_nb", "cnt_items", "x", "run2", "visit_node", "getValue", "_get_x"],
                          style_mismatch={"getValue": 1})
        f = self.fitted(X).flags(X)
        self.assertEqual(list(f.columns), FLAG_NAMES)
        expected = {
            "get_value": [0, 0, 0, 0, 0, 0],
            "get_nb": [1, 1, 0, 0, 0, 0],          # nb: short AND vowelless
            "cnt_items": [0, 1, 0, 0, 1, 0],       # cnt is not a listed verb
            "x": [0, 0, 1, 0, 1, 0],               # x is in SHORT_OK
            "run2": [0, 0, 0, 1, 0, 0],            # digit runs are not short / vowelless tokens
            "visit_node": [0, 0, 0, 0, 1, 0],      # a verb missing from the 68-word list
            "getValue": [0, 0, 0, 0, 0, 1],        # contextual: style_mismatch from the file
            "_get_x": [0, 0, 0, 0, 0, 0],          # leading underscore stripped; x allowed
        }
        for name, flags in expected.items():
            self.assertEqual(f.loc[name].tolist(), flags, name)
        for name, (feature, _) in zip(FLAG_NAMES, bl.NAMING_FLAGS.values()):
            raw = X[feature]
            want = (raw == 0) if feature == "name_starts_verb" else (raw > 0)
            self.assertEqual(f[name].tolist(), want.astype(int).tolist(), name)

    def test_correlated_flags_fire_together(self):
        X = named_methods(["f", "nb", "tf_idf", "md5_hash", "is_valid", "to_dict"])
        s = pd.Series(self.fitted(X).predict_proba(X)[:, 1], index=X.index) * 6
        flags = self.fitted(X).flags(X)
        self.assertEqual(round(s["f"]), 3)                                  # single + short + not-verb
        self.assertTrue(flags.loc["f", ["single_character_name", "short_token", "first_word_not_in_verb_list"]].all())
        self.assertTrue(flags.loc["nb", ["short_token", "vowelless_token"]].all())
        self.assertTrue(flags.loc["tf_idf", ["short_token", "vowelless_token"]].all())
        self.assertEqual(round(s["md5_hash"]), 4)                           # one token, four flags
        self.assertEqual(flags.loc["is_valid"].tolist(), [1, 0, 0, 0, 0, 0])  # 'is' is a verb and a short token
        self.assertEqual(flags.loc["to_dict"].tolist(), [1, 0, 0, 0, 0, 0])

    def test_name_is_private_is_not_a_flag(self):
        X = named_methods(["get_value", "_get_value"])
        self.assertEqual((X["name_is_private"].tolist()), [0.0, 1.0])
        s = self.fitted(X).predict_proba(X)[:, 1]
        self.assertEqual(s[0], s[1])
        self.assertNotIn("name_is_private", bl.NAMING_FEATURES)

    def test_deterministic_scores_in_unit_interval(self):
        X, y = synthetic()
        a = bl.NamingHeuristicClassifier().fit(X, y).predict_proba(X)
        b = bl.NamingHeuristicClassifier().fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(((a >= 0) & (a <= 1)).all() and np.isfinite(a).all())
        np.testing.assert_allclose(a.sum(axis=1), 1)
        self.assertTrue(set(np.round(a[:, 1] * 6, 9)) <= set(range(7)))
        self.assertEqual(bl.NamingHeuristicClassifier().get_params(), {})

    def test_never_reads_labels(self):
        X, y = synthetic()
        base = bl.NamingHeuristicClassifier().fit(X, y).predict_proba(X)
        for other in (1 - y, y.sample(frac=1, random_state=1).reset_index(drop=True), np.r_[1, np.zeros(len(y) - 1, int)]):
            np.testing.assert_array_equal(base, bl.NamingHeuristicClassifier().fit(X, other).predict_proba(X))

    def test_uses_only_approved_before_name_features(self):
        self.assertTrue(set(bl.NAMING_FEATURES) <= set(MAIN_FEATURES) <= set(COMPARISON_FEATURES))
        forbidden = set(config.ID_COLS) | {config.TARGET, "meta_new_name", "meta_project_group", "call_sites",
                                           "rel_call_sites", "name_is_private"}
        self.assertFalse(set(bl.NAMING_FEATURES) & forbidden)
        X, y = synthetic()
        m = bl.NamingHeuristicClassifier().fit(X, y)
        others = [c for c in X.columns if c not in bl.NAMING_FEATURES]
        changed = X.copy()
        changed[others] = changed[others] * 7 + 3                                # nothing else matters
        np.testing.assert_array_equal(m.predict_proba(X), m.predict_proba(changed))

    def test_default_predict_rule_is_not_the_evaluation_operating_point(self):
        # scores 0, 1/6, 2/6, 3/6, 4/6; renamed = the two methods with 1 and 2 flags
        X = named_methods(["get_value", "visit_node", "cnt_items", "f", "md5_hash"])
        y = np.array([0, 1, 1, 0, 0])
        m = bl.NamingHeuristicClassifier().fit(X, y)
        proba = m.predict_proba(X)[:, 1]
        self.assertEqual(bl.DEFAULT_RULE_MIN_FLAGS, 3)
        np.testing.assert_array_equal(m.predict(X), [0, 0, 0, 1, 1])               # >= 3 of 6 flags
        sel = M.select_max_f1_threshold(y, proba)                                  # Stage 6 procedure
        evaluated = M.evaluate(y, proba, sel.threshold)
        stage6_pred = M.predict_at_threshold(proba, sel.threshold)
        self.assertFalse(np.array_equal(stage6_pred, m.predict(X)))                # different operating points
        self.assertEqual(M.threshold_metrics(y, m.predict(X))["recall"], 0.0)      # default rule finds neither
        self.assertGreater(evaluated["recall"], 0)
        self.assertEqual(evaluated["average_precision"], M.average_precision(y, proba))   # AP ignores both rules

    def test_rejects_invalid_inputs(self):
        X, y = synthetic(n=30)
        cases = {"missing column": X.drop(columns="name_has_digit"), "numpy": X.to_numpy(),
                 "negative": X.assign(name_short_tokens=-1.0), "nan": X.assign(name_no_vowel_tokens=np.nan),
                 "non-binary": X.assign(style_mismatch=2)}
        for label, Xb in cases.items():
            with self.assertRaises(bl.BaselineInputError, msg=label):
                bl.NamingHeuristicClassifier().fit(Xb, y)
        with self.assertRaises(bl.BaselineInputError):
            bl.NamingHeuristicClassifier().fit(X, np.full(30, 3))
        with self.assertRaises(NotFittedError):
            bl.NamingHeuristicClassifier().predict_proba(X)

    def test_both_feature_configurations(self):
        for fs in ("main", "comparison"):
            X = named_methods(["get_nb", "run2"], feature_set=fs)
            pipe = mp.build_model_pipeline("naming_heuristic", fs).fit(X, [1, 0])
            np.testing.assert_allclose(pipe.predict_proba(X)[:, 1], [2 / 6, 1 / 6])


class IntegrationTests(unittest.TestCase):
    def test_builder_registration(self):
        for fs in ("main", "comparison"):
            for name, cls in (("stratified_random", bl.StratifiedRandomClassifier),
                              ("naming_heuristic", bl.NamingHeuristicClassifier)):
                pipe = mp.build_model_pipeline(name, fs)
                self.assertEqual([n for n, _ in pipe.steps], ["check", "model"])
                self.assertIsInstance(pipe.named_steps["model"], cls)
                self.assertEqual(repr(clone(pipe).get_params()), repr(pipe.get_params()))
                for t in ("log1p_asinh", "yeo_johnson"):
                    with self.assertRaises(mp.ModelConfigError):
                        mp.build_model_pipeline(name, fs, t)
        self.assertEqual(mp.build_model_pipeline("stratified_random", "main", model_params={"random_state": 9})
                         .named_steps["model"].random_state, 9)
        with self.assertRaises(mp.ModelConfigError):
            mp.build_model_pipeline("naming_heuristic", "main", model_params={"weight": 2})

    def test_input_check_excludes_identifiers_and_metadata(self):
        X, y = synthetic()
        for name in ("stratified_random", "naming_heuristic"):
            for extra in ("name", "meta_project_group", "renamed", "call_sites"):
                with self.assertRaises(ValueError, msg=f"{name} {extra}"):
                    mp.build_model_pipeline(name, "main").fit(X.assign(**{extra: 1}), y)

    def test_cross_validation_and_stage6_evaluation(self):
        for fs in ("main", "comparison"):
            X, y = synthetic(fs, n=600, rate=0.1, seed=2)
            groups = np.arange(600) % 30
            folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
            for name in ("stratified_random", "naming_heuristic", "dummy_prior"):
                oof = cross_val_predict(mp.build_model_pipeline(name, fs), X, y, cv=folds,
                                        method="predict_proba")[:, 1]
                M.validate_oof(oof, y, folds)
                cv = M.fold_average_precision(y, oof, folds)
                self.assertEqual(cv["n_folds"], 5)
                sel = M.select_max_f1_threshold(y, oof)
                e = M.evaluate(y, oof, sel.threshold)
                self.assertTrue(set(M.METRICS) <= set(e), name)
                if name == "dummy_prior":
                    # constant WITHIN each fold (that fold's training rate), but not across folds:
                    # per fold ROC-AUC = 0.5 and AP = the fold's positive rate
                    for (_, val), ap in zip(folds, cv["per_fold"]):
                        self.assertEqual(len(np.unique(oof[val])), 1)
                        self.assertEqual(M.roc_auc(y.iloc[val], oof[val]), 0.5)
                        self.assertAlmostEqual(ap, y.iloc[val].mean())
                    self.assertGreater(len(np.unique(oof)), 1)          # pooled OOF is not constant

    def test_random_scores_depend_on_the_training_fold_only(self):
        X, y = synthetic(n=500, rate=0.1, seed=4)
        train, val = np.arange(400), np.arange(400, 500)
        pipe = mp.build_model_pipeline("stratified_random", "main").fit(X.iloc[train], y.iloc[train])
        self.assertEqual(pipe.named_steps["model"].prevalence_, y.iloc[train].mean())
        s1 = pipe.predict_proba(X.iloc[val])
        y2 = y.copy(); y2.iloc[val] = 1 - y2.iloc[val]                    # flip every validation label
        s2 = mp.build_model_pipeline("stratified_random", "main").fit(X.iloc[train], y2.iloc[train]).predict_proba(X.iloc[val])
        np.testing.assert_array_equal(s1, s2)


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class InputsUnchangedTests(unittest.TestCase):
    def test_master_and_split_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        self.assertEqual(sp.load_splits()[1]["split_hash"], SPLIT_HASH)


if __name__ == "__main__":
    unittest.main()
