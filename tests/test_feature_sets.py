"""
Tests for the approved feature configurations (src/features/feature_sets.py).
Run: python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.features import feature_sets as fs  # noqa: E402

EXPECTED_MAIN = [
    "name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit", "name_short_tokens",
    "name_no_vowel_tokens", "name_starts_verb", "name_single_char", "loc", "n_params", "n_defaults", "has_varargs",
    "has_kwargs", "n_returns", "n_calls", "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub",
    "n_decorators", "is_property", "is_static_or_class", "name_body_overlap", "in_class", "class_n_methods",
    "is_test_file", "style_mismatch", "name_token_sharing", "first_token_shared", "style_share_in_file",
    "rel_name_len", "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic", "rel_doc_len",
    "rel_name_body_overlap",
]


class FeatureSetTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(fs.MAIN_FEATURES), 40)
        self.assertEqual(len(fs.COMPARISON_FEATURES), 42)
        self.assertEqual(len(config.FEATURES), 46)

    def test_exact_main_list_and_order(self):
        self.assertEqual(fs.MAIN_FEATURES, EXPECTED_MAIN)

    def test_order_follows_master_columns(self):
        header = [c for c in config.FEATURES if c in fs.COMPARISON_FEATURES]
        self.assertEqual(fs.COMPARISON_FEATURES, header)
        self.assertEqual(fs.MAIN_FEATURES, [c for c in header if c not in fs.COMPARISON_ONLY])

    def test_comparison_is_main_plus_file_size(self):
        self.assertEqual(set(fs.COMPARISON_FEATURES) - set(fs.MAIN_FEATURES), {"file_loc", "file_n_methods"})
        self.assertTrue(set(fs.MAIN_FEATURES) < set(fs.COMPARISON_FEATURES))

    def test_excluded_and_dropped(self):
        self.assertEqual(fs.EXCLUDED_LEAKAGE, ["call_sites", "rel_call_sites"])
        self.assertEqual(fs.DROPPED_FEATURES, ["class_depth", "is_async"])
        for name in fs.EXCLUDED_LEAKAGE + fs.DROPPED_FEATURES:
            self.assertNotIn(name, fs.MAIN_FEATURES)
            self.assertNotIn(name, fs.COMPARISON_FEATURES)

    def test_every_master_feature_accounted_for(self):
        self.assertEqual(set(config.FEATURES), set(fs.COMPARISON_FEATURES) | set(fs.NOT_FOR_TRAINING))

    def test_validation_rejects_forbidden_features(self):
        for bad in fs.EXCLUDED_LEAKAGE + fs.DROPPED_FEATURES:
            with self.assertRaises(fs.FeatureSetError, msg=bad):
                fs.validate_feature_set(fs.MAIN_FEATURES + [bad])
        for bad in ["commit", "file", "class", "name", "renamed", "meta_project_group", "meta_new_name"]:
            with self.assertRaises(fs.FeatureSetError, msg=bad):
                fs.validate_feature_set(fs.MAIN_FEATURES + [bad])
        with self.assertRaises(fs.FeatureSetError):
            fs.validate_feature_set(fs.MAIN_FEATURES + ["loc"])            # duplicate
        with self.assertRaises(fs.FeatureSetError):
            fs.validate_feature_set(fs.MAIN_FEATURES + ["file_loc"], allow_file_size=False)

    def test_validation_accepts_approved_sets(self):
        self.assertEqual(fs.validate_feature_set(fs.MAIN_FEATURES, allow_file_size=False), fs.MAIN_FEATURES)
        self.assertEqual(fs.validate_feature_set(fs.COMPARISON_FEATURES), fs.COMPARISON_FEATURES)

    def test_check_configurations_catches_tampering(self):
        original = list(fs.MAIN_FEATURES)
        try:
            fs.MAIN_FEATURES.append("call_sites")
            with self.assertRaises(fs.FeatureSetError):
                fs.check_configurations()
        finally:
            fs.MAIN_FEATURES[:] = original
        fs.check_configurations()

    def test_get_feature_set_returns_copies(self):
        got = fs.get_feature_set("main")
        got.append("x")
        self.assertEqual(len(fs.MAIN_FEATURES), 40)
        with self.assertRaises(fs.FeatureSetError):
            fs.get_feature_set("all")

    def test_master_schema_and_header_contain_every_selected_feature(self):
        schema = json.loads(config.MASTER_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["roles"]["feature"], config.FEATURES)
        header = config.MASTER.read_text(encoding="utf-8").split("\n", 1)[0].strip().split(",")
        self.assertTrue(set(fs.COMPARISON_FEATURES) <= set(header))

    def test_report_generator_uses_the_same_objects(self):
        from src.data import generate_dataset_report as gen
        self.assertIs(gen.MAIN_FEATURES, fs.MAIN_FEATURES)
        self.assertIs(gen.COMPARISON_FEATURES, fs.COMPARISON_FEATURES)
        self.assertIs(gen.EXCLUDED_LEAKAGE, fs.EXCLUDED_LEAKAGE)

    def test_experiment_configs_reference_approved_sets(self):
        configs = sorted(config.EXPERIMENT_CONFIGS.glob("*.json"))
        self.assertEqual([c.name for c in configs], ["comparison_42.json", "main_40.json"])
        for path in configs:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            features = fs.get_feature_set(cfg["feature_set"])
            self.assertEqual(len(features), cfg["n_features"], path.name)
            self.assertEqual(cfg["target"], config.TARGET)
            self.assertEqual(cfg["group_column"], "meta_project_group")
            self.assertEqual(cfg["dataset"], config.MASTER.relative_to(config.ROOT).as_posix())
            self.assertNotIn("features", cfg, "configs must name a feature set, not list columns")


if __name__ == "__main__":
    unittest.main()
