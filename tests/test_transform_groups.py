"""
Tests for Stage 2, transformation-group definitions (src/features/transform_groups.py).
Definitions only: no transform is fitted or applied.
Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.features import transform_groups as tg  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, MAIN_FEATURES  # noqa: E402

LOG1P_MAIN = ["loc", "n_params", "n_returns", "n_calls", "cyclomatic", "max_nesting", "doc_len", "class_n_methods"]
ASINH = ["rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic", "rel_doc_len"]


class TransformGroupTests(unittest.TestCase):
    def test_main_40(self):
        g = tg.transform_groups("main")
        self.assertEqual(list(g.log1p), LOG1P_MAIN)
        self.assertEqual(list(g.asinh), ASINH)
        self.assertEqual(list(g.raw), [f for f in MAIN_FEATURES if f not in LOG1P_MAIN + ASINH])
        self.assertEqual((len(g.log1p), len(g.asinh), len(g.raw)), (8, 5, 27))

    def test_comparison_42(self):
        g = tg.transform_groups("comparison")
        self.assertEqual(list(g.log1p), LOG1P_MAIN[:7] + ["class_n_methods", "file_n_methods", "file_loc"])
        self.assertEqual(list(g.asinh), ASINH)
        self.assertEqual(list(g.raw), list(tg.transform_groups("main").raw))
        self.assertEqual((len(g.log1p), len(g.asinh), len(g.raw)), (10, 5, 27))

    def test_partition_and_output_order(self):
        for name, features in (("main", MAIN_FEATURES), ("comparison", COMPARISON_FEATURES)):
            g = tg.transform_groups(name)
            out = g.output_order
            self.assertEqual(out, list(g.log1p) + list(g.asinh) + list(g.raw))
            self.assertEqual(sorted(out), sorted(features))
            self.assertEqual(len(out), len(set(out)))
            for group in (g.log1p, g.asinh, g.raw):           # each group keeps the configuration's order
                self.assertEqual(list(group), [f for f in features if f in group])
            self.assertEqual(tg.transform_groups(name), g)    # deterministic

    def test_derived_from_names_not_positions(self):
        self.assertEqual(tg.groups_for_features(list(MAIN_FEATURES)), tg.transform_groups("main"))
        self.assertEqual(tg.groups_for_features(tuple(COMPARISON_FEATURES)), tg.transform_groups("comparison"))

    def test_result_is_immutable(self):
        g = tg.transform_groups("main")
        with self.assertRaises(Exception):
            g.log1p = ()
        g.as_dict()["log1p"].append("x")
        self.assertEqual(len(tg.transform_groups("main").log1p), 8)

    def test_unsupported_configurations(self):
        for bad in ["all", "", None, "Main"]:
            with self.assertRaises(tg.TransformGroupError, msg=bad):
                tg.transform_groups(bad)
        self.assertTrue(issubclass(tg.TransformGroupError, ValueError))

    def test_invalid_and_incomplete_feature_lists(self):
        cases = {
            "incomplete": MAIN_FEATURES[:-1],
            "duplicate": MAIN_FEATURES + ["loc"],
            "reordered": list(reversed(MAIN_FEATURES)),
            "leakage": MAIN_FEATURES + ["call_sites"],
            "identifier": MAIN_FEATURES + ["name"],
            "half comparison": MAIN_FEATURES + ["file_loc"],
            "empty": [],
        }
        for label, features in cases.items():
            with self.assertRaises(tg.TransformGroupError, msg=label):
                tg.groups_for_features(features)

    def test_invalid_group_definitions(self):
        good = tg.transform_groups("main").as_dict()

        def broken(**change):
            return {**good, **change}

        cases = {
            "missing feature": broken(raw=good["raw"][1:]),
            "in two groups": broken(raw=good["raw"] + ["loc"]),
            "duplicate in group": broken(asinh=good["asinh"] + ["rel_loc"]),
            "not in configuration": broken(raw=good["raw"] + ["file_loc"]),
            "unknown name": broken(raw=good["raw"] + ["renamed"]),
            "signed feature in log1p": broken(log1p=good["log1p"] + ["rel_loc"], asinh=good["asinh"][1:]),
            "count in asinh": broken(asinh=good["asinh"] + ["loc"], log1p=good["log1p"][1:]),
            "extra group": {**good, "scale": []},
            "missing group": {k: v for k, v in good.items() if k != "raw"},
        }
        for label, groups in cases.items():
            with self.assertRaises(tg.TransformGroupError, msg=label):
                tg.validate_groups(groups, MAIN_FEATURES)
        tg.validate_groups(good, MAIN_FEATURES)

    def test_definitions_use_only_approved_features(self):
        self.assertTrue(set(tg.LOG1P_FEATURES) <= set(COMPARISON_FEATURES))
        self.assertTrue(set(tg.ASINH_FEATURES) <= set(MAIN_FEATURES))
        self.assertFalse(set(tg.LOG1P_FEATURES) & set(tg.ASINH_FEATURES))

    def test_log1p_domain_check(self):
        g = tg.transform_groups("main")
        X = pd.DataFrame(np.ones((4, 40)), columns=MAIN_FEATURES)
        X["rel_loc"] = -3.0                                   # asinh / raw columns may be negative
        tg.check_log1p_domain(X, g)
        for value in (-1.0, np.nan, np.inf):
            bad = X.copy()
            bad.loc[0, "doc_len"] = value
            with self.assertRaises(tg.TransformGroupError):
                tg.check_log1p_domain(bad, g)


@unittest.skipUnless(config.MASTER.exists(), "master dataset not present")
class MasterDomainTests(unittest.TestCase):
    def test_log1p_columns_are_non_negative_on_the_master(self):
        from src.data.prepare_dataset import prepare_dataset
        X, _, _ = prepare_dataset("comparison")             # read-only; nothing is transformed
        tg.check_log1p_domain(X, tg.transform_groups("comparison"))
        self.assertTrue((X[list(tg.ASINH_FEATURES)] < 0).any().all())   # why asinh, not log1p


if __name__ == "__main__":
    unittest.main()
