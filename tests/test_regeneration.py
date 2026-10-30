"""
Regeneration consistency: re-extracting features from the raw ActRef code in
legacy_v2 mode must give exactly the stored v2 values, and the default
(Before-only) mode must differ from v2 only in call_sites and rel_* columns. Uses the first 60 commits by default; set
FULL_REGENERATION=1 to check all commits (about 2 minutes).
Skipped when the raw data is not present.
Run: python -m unittest tests.test_regeneration -v
"""
import os
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.extract_features import build  # noqa: E402


@unittest.skipUnless(config.RAW_COMMITS.exists() and config.V2_FEATURES.exists(), "raw data or v2 missing")
class RegenerationTests(unittest.TestCase):
    def test_features_regenerate_exactly(self):
        limit = None if os.environ.get("FULL_REGENERATION") == "1" else 60
        pos = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)
        new, missing = build(config.RAW_COMMITS, pos, limit=limit, legacy_v2=True)
        self.assertEqual(missing, set(), "a positive was not found in the Before code")

        old = pd.read_csv(config.V2_FEATURES, keep_default_na=False)
        old = old[old["commit"].isin(set(new["commit"]))]
        new["class"] = new["class"].astype(str)
        self.assertEqual(len(new), len(old))
        a = new.set_index(config.KEY).sort_index()
        b = old.set_index(config.KEY).loc[a.index, a.columns]
        diff = (a.astype(float) - b.astype(float)).abs().values.max()
        self.assertEqual(diff, 0.0)

    def test_before_only_mode_changes_only_leaky_columns(self):
        pos = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)
        new, _ = build(config.RAW_COMMITS, pos, limit=60)
        old = pd.read_csv(config.V2_FEATURES, keep_default_na=False)
        new["class"] = new["class"].astype(str)
        a = new.set_index(config.KEY).sort_index()
        b = old.set_index(config.KEY).loc[a.index, a.columns]
        changed = [c for c in a.columns if (a[c].astype(float) != b[c].astype(float)).any()]
        allowed = {"call_sites"} | {c for c in a.columns if c.startswith("rel_")}
        self.assertTrue(set(changed) <= allowed, f"unexpected changes: {set(changed) - allowed}")


if __name__ == "__main__":
    unittest.main()
