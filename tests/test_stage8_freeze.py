"""
Integrity of the Stage 8 reproducibility freeze (results/runs/2026-10-31_stage8_freeze_manifest.json,
docs/stage8_results_freeze.md). Read-only: recomputes hashes and compares them with the manifest.
Source-file hashes are not asserted here (later stages may legitimately change code); they identify
the code that produced the frozen results.
Run: python -m unittest discover -s tests -v
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

MANIFEST = config.RUNS_DIR / "2026-10-31_stage8_freeze_manifest.json"
REPORT = config.DOCS / "stage8_results_freeze.md"
MATRIX = ["dummy_prior__main", "dummy_prior__comparison", "stratified_random__main", "stratified_random__comparison",
          "naming_heuristic__main", "naming_heuristic__comparison", "logistic_regression__main__log1p_asinh",
          "logistic_regression__comparison__log1p_asinh", "logistic_regression__main__yeo_johnson",
          "logistic_regression__comparison__yeo_johnson", "random_forest__main", "random_forest__comparison",
          "lightgbm__main", "lightgbm__comparison"]


def sha(path):
    return hashlib.sha256((config.ROOT / path).read_bytes()).hexdigest()


@unittest.skipUnless(MANIFEST.exists(), "Stage 8 freeze manifest not present")
class Stage8FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_report_names_this_manifest(self):
        text = REPORT.read_text(encoding="utf-8")
        self.assertIn(hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), text)
        self.assertIn("not final held-out test results", text)

    def test_inputs_unchanged(self):
        for item in self.m["inputs"].values():
            if "path" in item:
                self.assertEqual(sha(item["path"]), item["sha256"], item["path"])
        for path, digest in self.m["inputs"]["feature_set_configs"].items():
            self.assertEqual(sha(path), digest, path)
        self.assertEqual(self.m["inputs"]["master_dataset"]["sha256"], config.MASTER_SHA256)
        self.assertEqual(sha(self.m["environment"]["requirements_lock"]["path"]),
                         self.m["environment"]["requirements_lock"]["sha256"])

    def test_all_14_runs_and_every_file_unchanged(self):
        self.assertEqual([r["run_id"] for r in self.m["runs"]], MATRIX)
        for r in self.m["runs"]:
            folder = config.ROOT / r["path"]
            self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(r["files"]), r["run_id"])
            self.assertEqual(len(r["files"]), 6)
            for name, digest in r["files"].items():
                self.assertEqual(sha(f"{r['path']}/{name}"), digest, f"{r['run_id']}/{name}")
            self.assertEqual(r["status"], "complete")
        summary = self.m["summary"]
        self.assertEqual(sorted(p.name for p in (config.ROOT / summary["path"]).iterdir()), sorted(summary["files"]))
        for name, digest in summary["files"].items():
            self.assertEqual(sha(f"{summary['path']}/{name}"), digest, name)

    def test_recorded_claims(self):
        c = self.m["confirmations"]
        self.assertIs(c["held_out_test_evaluated"], False)
        self.assertIs(c["hyperparameter_tuning"], False)
        self.assertIn("exploratory_post_hoc", c["naming_heuristic_status"])
        statuses = {r["run_id"]: r["research_status"] for r in self.m["runs"]}
        self.assertEqual({k for k, v in statuses.items() if v == "exploratory_post_hoc"},
                         {"naming_heuristic__main", "naming_heuristic__comparison"})
        sel = self.m["stage9_selection_development_only"]["selected"]
        self.assertFalse(any("naming" in v for v in sel.values()))
        self.assertIn("does not authorize", self.m["stage9_selection_development_only"]["note"])
        self.assertTrue(self.m["validation_evidence"]["runtime_enforced"]["checks"])
        self.assertTrue(self.m["validation_evidence"]["reconstructable_from_artifacts"]["checks"])

    def test_source_addendum_matches_the_manifest(self):
        add = json.loads((config.RUNS_DIR / "2026-10-31_stage8_freeze_source_addendum.json").read_text(encoding="utf-8"))
        self.assertEqual(add["addendum_to"]["sha256"], hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
        manifest_sources = {s["path"]: s["sha256"] for s in self.m["code"]["source_files"]}
        self.assertEqual({r["path"]: r["manifest_sha256_working_tree_bytes"] for r in add["source_files"]},
                         manifest_sources)
        for r in add["source_files"]:
            self.assertRegex(r["git_blob_id"], r"^[0-9a-f]{40}$")
            self.assertRegex(r["sha256_lf_normalized"], r"^[0-9a-f]{64}$")

    def test_results_are_stored_byte_for_byte(self):
        import subprocess
        out = subprocess.run(["git", "check-attr", "text", "--", "results/runs/2026-10-31_stage8_summary/summary.md"],
                             cwd=config.ROOT, capture_output=True, text=True).stdout
        self.assertTrue(out.strip().endswith("unset"), out)
