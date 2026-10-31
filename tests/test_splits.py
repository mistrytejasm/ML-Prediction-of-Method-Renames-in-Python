"""
Tests for Stage 3, the saved project-level split (src/evaluation/splits.py).
Only y and groups are used; nothing is fitted or trained. The master is only read.
Run: python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.load_master import load_master, sha256_of  # noqa: E402
from src.data.prepare_dataset import prepare_dataset  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
# Reference values from the approved preprocessing guide (Steps 5-6)
EXPECTED_TEST = {"rows": 6242, "projects": 62, "positives": 50, "positive_projects": 12}
EXPECTED_DEV = {"rows": 24967, "projects": 261, "positives": 193, "positive_projects": 67}
EXPECTED_FOLDS = [  # validation projects, rows, positives, positive projects, training positives
    (50, 4999, 41, 13, 152), (54, 4991, 38, 13, 155), (53, 4991, 38, 12, 155),
    (52, 4995, 39, 13, 154), (52, 4991, 37, 16, 156)]


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class SavedSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = load_master()
        cls.assign, cls.meta = sp.load_splits(master=cls.master)      # validates against the master
        cls.dev = cls.assign[cls.assign["partition"] == "development"]
        cls.test = cls.assign[cls.assign["partition"] == "test"]

    def test_checksum_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        self.assertEqual(self.meta["dataset_sha256"], CANONICAL)

    def test_deterministic_regeneration_matches_saved_split(self):
        fresh = sp.generate_splits()
        self.assertEqual(sp.split_hash(fresh), SPLIT_HASH)
        self.assertEqual(self.meta["split_hash"], SPLIT_HASH)
        self.assertTrue(fresh.equals(self.assign))

    def test_exact_row_coverage_and_no_duplicates(self):
        self.assertEqual(len(self.assign), len(self.master))
        self.assertTrue(np.array_equal(self.assign["master_row"], np.arange(len(self.master))))
        self.assertFalse(self.assign.duplicated(sp.KEY).any())
        self.assertEqual(len(self.dev) + len(self.test), len(self.master))
        for col in sp.KEY + ["meta_project_group", "renamed"]:
            self.assertTrue((self.assign[col].astype(str).values == self.master[col].astype(str).values).all(), col)

    def test_no_project_overlap_between_development_and_test(self):
        self.assertFalse(set(self.dev["meta_project_group"]) & set(self.test["meta_project_group"]))
        self.assertEqual(self.dev["meta_project_group"].nunique() + self.test["meta_project_group"].nunique(), 323)

    def test_inner_folds(self):
        dev_pos, test_pos = sp.outer_indices(self.assign)
        groups = self.assign["meta_project_group"].to_numpy()[dev_pos]
        seen = np.zeros(len(dev_pos), dtype=int)
        for train, val in sp.inner_folds(self.assign):
            self.assertFalse(set(groups[train]) & set(groups[val]))        # project-disjoint
            self.assertEqual(len(train) + len(val), len(dev_pos))
            seen[val] += 1
        self.assertTrue((seen == 1).all())                                # every dev row validated exactly once
        self.assertTrue((self.test["inner_fold"] == -1).all())

    def test_helpers_match_the_splitter(self):
        X, y, g = prepare_dataset("main")
        outer = list(sp._splitter().split(X, y, g))
        dev_pos, test_pos = sp.outer_indices(self.assign)
        self.assertTrue(np.array_equal(dev_pos, outer[0][0]) and np.array_equal(test_pos, outer[0][1]))
        inner = list(sp._splitter().split(X.iloc[dev_pos], y.iloc[dev_pos], g.iloc[dev_pos]))
        for (t1, v1), (t2, v2) in zip(inner, sp.inner_folds(self.assign)):
            self.assertTrue(np.array_equal(t1, t2) and np.array_equal(v1, v2))

    def test_both_feature_configurations_get_identical_partitions(self):
        parts = {}
        for name in ("main", "comparison"):
            X, y, g = prepare_dataset(name)
            parts[name] = sp.assign_partitions(X, y, g)
        self.assertTrue(np.array_equal(parts["main"][0], parts["comparison"][0]))
        self.assertTrue(np.array_equal(parts["main"][1], parts["comparison"][1]))
        self.assertTrue(np.array_equal(parts["main"][0], self.assign["partition"].to_numpy()))
        self.assertTrue(np.array_equal(parts["main"][1], self.assign["inner_fold"].to_numpy()))

    def test_statistics_match_the_reference_values(self):
        counts = self.meta["counts"]
        self.assertEqual({k: counts["test"][k] for k in EXPECTED_TEST}, EXPECTED_TEST)
        self.assertEqual({k: counts["development"][k] for k in EXPECTED_DEV}, EXPECTED_DEV)
        got = [(f["validation"]["projects"], f["validation"]["rows"], f["validation"]["positives"],
                f["validation"]["positive_projects"], f["training"]["positives"]) for f in counts["inner_folds"]]
        self.assertEqual(got, EXPECTED_FOLDS)

    def test_acceptance_criteria(self):
        self.assertGreaterEqual(self.meta["counts"]["test"]["positives"], 40)
        self.assertGreaterEqual(self.meta["counts"]["test"]["positive_projects"], 10)
        self.assertIn("not prespecified", self.meta["acceptance_criteria"]["status"])

    def test_metadata(self):
        m = self.meta
        self.assertEqual((m["outer_split"]["random_state"], m["outer_split"]["test_fold"], m["inner_split"]["n_splits"]), (42, 0, 5))
        self.assertEqual(m["feature_sets"], {"main": 40, "comparison": 42})
        self.assertEqual(m["row_key"], ["commit", "file", "class", "name"])
        self.assertTrue({"python", "scikit-learn", "numpy", "pandas"} <= set(m["versions"]))
        self.assertEqual(m["assignment_file_sha256"], sha256_of(sp.SPLIT_CSV))

    # ---------------------------------------------------------------- tampered copies are rejected
    def assertInvalid(self, assign, text):
        with self.assertRaises(sp.SplitValidationError) as ctx:
            sp.validate_assignments(assign, self.master)
        self.assertIn(text, str(ctx.exception))

    def test_validation_rejects_broken_assignments(self):
        a = self.assign.copy()
        project = a.loc[a["partition"] == "test", "meta_project_group"].iloc[0]
        row = a.index[(a["meta_project_group"] == project)][0]
        b = a.copy(); b.loc[row, ["partition", "inner_fold"]] = ["development", 0]
        self.assertInvalid(b, "both development and test")
        dev_project = self.dev["meta_project_group"].iloc[0]
        rows = a.index[a["meta_project_group"] == dev_project]
        b = a.copy(); b.loc[rows[0], "inner_fold"] = (b.loc[rows[0], "inner_fold"] + 1) % 5
        self.assertInvalid(b, "more than one inner fold")
        self.assertInvalid(a.drop(index=5).reset_index(drop=True), "every master row")
        b = a.copy(); b.loc[3, "name"] = b.loc[4, "name"]
        self.assertInvalid(b, "does not match the master")
        b = a.copy(); b.loc[0, "renamed"] = 1 - b.loc[0, "renamed"]
        self.assertInvalid(b, "'renamed' does not match")
        b = a.copy(); b.loc[self.test.index[0], "inner_fold"] = 2
        self.assertInvalid(b, "inner_fold -1")

    def test_saved_files_are_protected(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            csv, meta = tmp / "s.csv", tmp / "s.json"
            shutil.copyfile(sp.SPLIT_CSV, csv); shutil.copyfile(sp.SPLIT_META, meta)
            sp.load_splits(csv, meta, self.master)
            with open(csv, "a", encoding="utf-8") as fh:
                fh.write("\n")
            with self.assertRaises(sp.SplitValidationError):
                sp.load_splits(csv, meta, self.master)                    # file changed after saving
            other = dict(self.meta, split_hash="0" * 64)
            with self.assertRaises(sp.SplitValidationError):
                sp.save_splits(self.assign, other, csv, meta)            # never overwritten
            self.assertFalse(sp.save_splits(self.assign, self.meta, sp.SPLIT_CSV, sp.SPLIT_META))  # same hash: no-op
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(sha256_of(sp.SPLIT_CSV), self.meta["assignment_file_sha256"])


if __name__ == "__main__":
    unittest.main()
