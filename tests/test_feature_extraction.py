"""
Unit tests on small synthetic commits (no raw data needed):
  - features use only the Before code
  - label rules (kept -> 0, labelled rename -> 1, deleted/unknown -> skipped)
  - duplicate commit folders are skipped
  - rename type classification
Run: python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.extract_features import method_rows  # noqa: E402
from src.data.label_utils import label_provenance, rename_type  # noqa: E402
from src.data.raw_commits import list_commit_dirs  # noqa: E402

COMMIT = "a" * 40
FILE = "pkg/models.py"

BEFORE = textwrap.dedent('''
    class Model:
        def get_usr(self, x):
            """Return the user."""
            return self.load(x)

        def load(self, x):
            if x:
                return x
            return None

        def deleted_later(self):
            return 1

        def __init__(self):
            pass

        def test_something(self):
            pass
''')

AFTER_KEEP = BEFORE.replace("def get_usr(", "def get_user(")       # deleted_later still there
AFTER_SMALL = AFTER_KEEP.replace("    def deleted_later(self):\n        return 1\n", "")
assert "deleted_later" not in AFTER_SMALL

AFTER_BIG = textwrap.dedent('''
    class Model:
        def get_user(self, x, y=None, *args, **kwargs):
            for i in range(10):
                while True:
                    break
            return [self.load(i) for i in range(100)]

        def load(self, a, b, c):
            """A brand-new docstring that is very long."""
            return sum([a, b, c])

        def brand_new_method(self):
            return 2
''')


def make_commit(root, name, before, after):
    d = root / name
    (d / "Before").mkdir(parents=True)
    (d / "After").mkdir(parents=True)
    (d / "Before" / FILE.replace("/", "#")).write_text(before, encoding="utf-8")
    (d / "After" / FILE.replace("/", "#")).write_text(after, encoding="utf-8")
    return d


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pos_keys = {(COMMIT, FILE, "Model", "get_usr")}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def rows(self, after):
        d = make_commit(self.tmp / after[:5].strip().replace(" ", "_") / str(len(after)), COMMIT, BEFORE, after)
        return {r["name"]: r for r in method_rows(d, COMMIT, self.pos_keys, set())}

    def test_label_rules(self):
        rows = self.rows(AFTER_SMALL)
        self.assertEqual(rows["get_usr"]["renamed"], 1, "labelled rename must be positive")
        self.assertEqual(rows["load"]["renamed"], 0, "method kept with the same name must be negative")
        self.assertNotIn("deleted_later", rows, "deleted method has no known label and must be skipped")
        self.assertNotIn("__init__", rows, "dunder methods are excluded")
        self.assertNotIn("test_something", rows, "test methods are excluded")

    def test_features_use_only_before_code(self):
        # AFTER_KEEP keeps deleted_later; AFTER_BIG deletes it and rewrites every body.
        # Features (including rel_* file averages) must not change.
        keep, big = self.rows(AFTER_KEEP), self.rows(AFTER_BIG)
        self.assertIn("deleted_later", keep)
        self.assertNotIn("deleted_later", big)
        for name in ("get_usr", "load"):
            for col in config.FEATURES:
                self.assertEqual(keep[name][col], big[name][col],
                                 f"{name}.{col} changed when only the After code changed")

    def test_feature_values_from_before(self):
        r = self.rows(AFTER_BIG)["get_usr"]
        self.assertEqual(r["n_params"], 1)          # x only (self excluded)
        self.assertEqual(r["has_docstring"], 1)
        self.assertEqual(r["in_class"], 1)
        self.assertEqual(r["name_short_tokens"], 0)  # 'usr' has 3 letters
        self.assertEqual(set(config.FEATURES) - set(r), set(), "a documented feature is missing")

    def test_duplicate_commit_folder_skipped(self):
        make_commit(self.tmp, COMMIT, BEFORE, AFTER_SMALL)
        make_commit(self.tmp, COMMIT + "_copy", BEFORE, AFTER_SMALL)
        make_commit(self.tmp, "b" * 40 + "_different", BEFORE, AFTER_BIG)
        kept, skipped = list_commit_dirs(self.tmp)
        self.assertIn(COMMIT + "_copy", skipped)
        self.assertIn("b" * 40 + "_different", [d.name for d in kept], "non-duplicate folders are kept")

    def test_copy_with_different_content_is_kept(self):
        make_commit(self.tmp, COMMIT, BEFORE, AFTER_SMALL)
        make_commit(self.tmp, COMMIT + "_copy", BEFORE, AFTER_BIG)
        kept, skipped = list_commit_dirs(self.tmp)
        self.assertEqual(skipped, {})
        self.assertEqual(len(kept), 2)


class RenameTypeTests(unittest.TestCase):
    CASES = {
        ("init_model", "_init_model"): "visibility_only",
        ("_eval", "eval"): "visibility_only",
        ("compute_numerical_kl", "compute_numerical_KL"): "case_or_style_only",
        ("subsFactor", "subs_factor"): "case_or_style_only",
        ("euclidian_distance", "euclidean_distance"): "spelling_or_inflection",
        ("category_storage", "categories_storage"): "spelling_or_inflection",
        ("grad_of", "insert_grad_of"): "word_added",
        ("bbox_encode", "encode"): "word_removed",
        ("create_continuous_observation_encoder", "create_vector_observation_encoder"): "word_replaced",
        ("showXmlBoundingBoxes", "showBoundingBoxFromAnnotationFile"): "restructured",
    }

    def test_cases(self):
        for (old, new), expected in self.CASES.items():
            self.assertEqual(rename_type(old, new), expected, f"{old} -> {new}")

    def test_provenance(self):
        self.assertEqual(label_provenance("auto_confident"), "auto_rule_confident")
        self.assertEqual(label_provenance("hand_checked", ""), "ai_reviewed_unconfirmed")
        self.assertEqual(label_provenance("hand_checked", "RP 2026-11-02"), "human_confirmed")


if __name__ == "__main__":
    unittest.main()
