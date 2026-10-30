"""
Tests on the real master dataset (skipped if it has not been built).
Run: python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.validate_dataset import validate  # noqa: E402


@unittest.skipUnless(config.MASTER.exists(), "master dataset not built")
class MasterDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = pd.read_csv(config.MASTER, keep_default_na=False)
        cls.m["class"] = cls.m["class"].astype(str)
        cls.schema = json.loads(config.MASTER_SCHEMA.read_text(encoding="utf-8"))
        cls.pos = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)

    def test_row_counts(self):
        self.assertEqual(len(self.m), self.schema["rows"])
        self.assertEqual(int(self.m[config.TARGET].sum()), 243)
        self.assertEqual(int((self.m[config.TARGET] == 0).sum()), self.schema["negatives"])

    def test_binary_target(self):
        self.assertTrue(set(self.m[config.TARGET].unique()) <= {0, 1})

    def test_identifier_uniqueness(self):
        self.assertFalse(self.m.duplicated(config.KEY).any())

    def test_no_missing_or_infinite(self):
        X = self.m[config.FEATURES].astype(float)
        self.assertEqual(int(X.isna().sum().sum()), 0)
        self.assertEqual(int(np.isinf(X.values).sum()), 0)

    def test_feature_columns(self):
        self.assertEqual(self.schema["roles"]["feature"], config.FEATURES)
        self.assertEqual(len(config.FEATURES), 46)
        leaky = [c for c in config.FEATURES if c.startswith("meta_") or c in
                 ("new_name", "old_name", "similarity", "label_source", config.TARGET)]
        self.assertEqual(leaky, [])

    def test_positive_label_consistency(self):
        pos = set(map(tuple, self.pos[["commit", "file", "class", "old_name"]].astype(str).values))
        got = set(map(tuple, self.m.loc[self.m[config.TARGET] == 1, config.KEY].astype(str).values))
        self.assertEqual(got, pos)

    def test_full_validation_has_no_failures(self):
        v2 = pd.read_csv(config.V2_FEATURES, keep_default_na=False) if config.V2_FEATURES.exists() else None
        ck = validate(self.m, self.schema, self.pos, v2)
        failed = [c for c in ck.items if c["status"] == "FAIL"]
        self.assertEqual(failed, [], failed)


if __name__ == "__main__":
    unittest.main()
