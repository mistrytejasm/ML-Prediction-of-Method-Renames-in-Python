"""
Tests for the read-only master loader (src/data/load_master.py).
Run: python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data import load_master as lm  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, MAIN_FEATURES  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"


@unittest.skipUnless(config.MASTER.exists(), "master dataset not present")
class LoadMasterTests(unittest.TestCase):
    def test_canonical_checksum(self):
        self.assertEqual(config.MASTER_SHA256, CANONICAL)
        self.assertEqual(lm.verify_master(), CANONICAL)
        self.assertEqual(lm.load_schema()["sources"]["output_sha256"], CANONICAL)

    def test_loading_does_not_modify_the_master(self):
        stat_before = config.MASTER.stat()
        df = lm.load_master()
        stat_after = config.MASTER.stat()
        self.assertEqual(lm.sha256_of(config.MASTER), CANONICAL)
        self.assertEqual((stat_before.st_size, stat_before.st_mtime_ns), (stat_after.st_size, stat_after.st_mtime_ns))
        self.assertEqual(df.shape, (31209, 59))
        self.assertEqual(int(df["renamed"].sum()), 243)

    def test_modified_copy_is_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            copy = tmp / "master.csv"
            shutil.copyfile(config.MASTER, copy)
            with open(copy, "ab") as fh:
                fh.write(b"\n")                      # one extra byte
            with self.assertRaises(lm.MasterChecksumError):
                lm.verify_master(copy)
            with self.assertRaises(lm.MasterChecksumError):
                lm.load_master(copy)
        finally:
            shutil.rmtree(tmp)
        self.assertEqual(lm.sha256_of(config.MASTER), CANONICAL)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(lm.MasterChecksumError):
            lm.verify_master(config.ROOT / "does_not_exist.csv")

    def test_select_xy_groups(self):
        df = lm.load_master()
        X, y, groups = lm.select_xy_groups(df, "main")
        self.assertEqual(list(X.columns), MAIN_FEATURES)
        X2, _, _ = lm.select_xy_groups(df, "comparison")
        self.assertEqual(list(X2.columns), COMPARISON_FEATURES)
        forbidden = set(config.ID_COLS) | {config.TARGET, "call_sites", "rel_call_sites", "class_depth", "is_async"}
        forbidden |= {c for c in df.columns if c.startswith("meta_")}
        self.assertFalse(forbidden & set(X2.columns))
        self.assertEqual(y.name, "renamed")
        self.assertEqual(groups.name, "meta_project_group")
        self.assertEqual(groups.nunique(), 323)


if __name__ == "__main__":
    unittest.main()
