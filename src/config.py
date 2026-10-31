"""
Project paths and dataset constants. All paths are relative to the project
root, so the scripts work from any folder and on any machine.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW_COMMITS = DATA / "raw" / "actref" / "commits"
EXTERNAL = DATA / "external"

INTERIM = DATA / "interim"
AUTO_RENAMES = INTERIM / "rename_candidates" / "detected_rename_candidates.csv"
CONFIDENT_RENAMES = INTERIM / "rename_candidates" / "auto_accepted_rename_candidates.csv"
REVIEW_SHEET = INTERIM / "label_review" / "uncertain_candidates_review_blank.xlsx"
REVIEW_SHEET_DONE = INTERIM / "label_review" / "uncertain_candidates_review_ai_completed.xlsx"
TRAIN_POSITIVES = INTERIM / "final_labels" / "rename_positive_labels.csv"

V2_FEATURES = DATA / "processed" / "feature_tables" / "method_features_v2_all_commits.csv"
V1_FEATURES = ROOT / "archive" / "old_datasets" / "method_features_v1_rename_files_only.csv"
MASTER_DIR = DATA / "processed" / "master"
MASTER = MASTER_DIR / "actref_master_dataset.csv"
MASTER_SCHEMA = MASTER_DIR / "actref_master_dataset_schema.json"
# SHA-256 of the canonical, audited master dataset (also stored in its schema).
MASTER_SHA256 = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLITS_DIR = DATA / "processed" / "splits"
EXPERIMENT_CONFIGS = ROOT / "configs" / "experiments"
RUNS_DIR = ROOT / "results" / "runs"

REPORTS_AUDIT = ROOT / "reports" / "audit"
REGENERATED_FEATURES = REPORTS_AUDIT / "regenerated" / "method_features_v2_regenerated.csv"
DOCS = ROOT / "docs"

# Settings used by the original label pipeline (archive/legacy_scripts/)
SIM_THRESHOLD = 0.70          # Labels_Renames.py
CONFIDENT_SIM = 0.90          # Make_review.py
MAX_PER_COMMIT = 5            # Build_Train_lables.py
CAP_SEED = 42                 # Build_Train_lables.py

# Column roles in the feature tables
ID_COLS = ["commit", "file", "class", "name"]
TARGET = "renamed"
KEY = ["commit", "file", "class", "name"]

NAME_FEATURES = ["name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit",
                 "name_short_tokens", "name_no_vowel_tokens", "name_starts_verb", "name_single_char"]
CODE_FEATURES = ["loc", "n_params", "n_defaults", "has_varargs", "has_kwargs", "n_returns",
                 "n_calls", "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub",
                 "is_async", "n_decorators", "is_property", "is_static_or_class", "name_body_overlap"]
CONTEXT_FEATURES = ["in_class", "class_depth", "class_n_methods", "file_n_methods", "file_loc",
                    "is_test_file", "style_mismatch", "call_sites"]
RELATIVE_FEATURES = ["name_token_sharing", "first_token_shared", "style_share_in_file",
                     "rel_name_len", "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls",
                     "rel_cyclomatic", "rel_doc_len", "rel_call_sites", "rel_name_body_overlap"]
FEATURES = NAME_FEATURES + CODE_FEATURES + CONTEXT_FEATURES + RELATIVE_FEATURES
FEATURE_GROUPS = {"name": NAME_FEATURES, "code": CODE_FEATURES,
                  "context": CONTEXT_FEATURES, "relative": RELATIVE_FEATURES}
FILE_SIZE_FEATURES = ["file_loc", "file_n_methods"]   # v1 showed these can carry sampling bias

# Non-model metadata columns in the master dataset (all prefixed meta_)
META_COLS = ["meta_label_source", "meta_label_provenance", "meta_label_similarity",
             "meta_rename_type", "meta_new_name", "meta_project_guess", "meta_project_group"]
