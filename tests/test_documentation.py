"""
Documentation reproducibility and consistency.
  - README code examples run and use only the approved 40 / 42 configurations
  - docs/feature_dataset_mapping.md regenerates identically (needs raw data)
  - docs/temporal_leakage_audit.md: every cited number is reproduced
    (slow, about 2-3 minutes; runs only with FULL_REGENERATION=1)
Run: python -m unittest discover -s tests -v
"""
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, MAIN_FEATURES  # noqa: E402

README = config.ROOT / "README.md"
PY = sys.executable


def python_blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


class ReadmeTests(unittest.TestCase):
    def test_no_example_uses_all_46_features_as_model_input(self):
        text = README.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"\bX\s*=\s*df\[schema\[.roles.\]\[.feature.\]\]")
        self.assertIn("40", text)
        self.assertIn("42", text)
        self.assertIn("meta_project_group", text)

    @unittest.skipUnless(config.MASTER.exists(), "master dataset not present")
    def test_readme_examples_run_and_select_approved_sets(self):
        blocks = [b for b in python_blocks(README) if "select_xy_groups" in b]
        self.assertEqual(len(blocks), 1)
        ns = {}
        cwd = os.getcwd()
        try:
            os.chdir(config.ROOT)
            exec(blocks[0], ns)
        finally:
            os.chdir(cwd)
        self.assertEqual(list(ns["X_main"].columns), MAIN_FEATURES)
        self.assertEqual(list(ns["X_comparison"].columns), COMPARISON_FEATURES)


@unittest.skipUnless(config.RAW_COMMITS.exists(), "raw ActRef data not present")
class GeneratedDocumentTests(unittest.TestCase):
    def run_check(self, script):
        return subprocess.run([PY, "-W", "ignore", str(config.ROOT / "src" / "data" / script), "--check"],
                              capture_output=True, text=True, cwd=config.ROOT)

    def test_feature_mapping_regenerates_identically(self):
        r = self.run_check("generate_feature_mapping.py")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("MATCHES", r.stdout)

    @unittest.skipUnless(os.environ.get("FULL_REGENERATION") == "1", "slow; set FULL_REGENERATION=1")
    def test_leakage_audit_evidence_is_reproduced(self):
        r = self.run_check("audit_temporal_leakage.py")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-2000:])
        self.assertIn("ALL CITED EVIDENCE REPRODUCED", r.stdout)


if __name__ == "__main__":
    unittest.main()
