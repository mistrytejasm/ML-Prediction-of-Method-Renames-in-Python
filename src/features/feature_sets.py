"""
Approved experimental feature configurations: the single source of truth.

The canonical master dataset (data/processed/master/actref_master_dataset.csv)
keeps all 46 audited features and is never changed. These lists only select
which of its columns a model may use. Approved 2026-10-31; see
docs/feature_dataset_mapping.md and docs/temporal_leakage_audit.md.

  MAIN_FEATURES        40  primary experiment
  COMPARISON_FEATURES  42  MAIN + file_loc, file_n_methods (file-size comparison)
  EXCLUDED_LEAKAGE      2  call_sites, rel_call_sites (search scope = the future
                           commit's changed-file list)
  DROPPED_FEATURES      2  class_depth (near-duplicate of in_class), is_async
                           (near-constant)

MAIN_FEATURES and COMPARISON_FEATURES keep the master dataset's column order. Do not edit these lists
without a new audit and approval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402

EXCLUDED_LEAKAGE = ["call_sites", "rel_call_sites"]
DROPPED_FEATURES = ["class_depth", "is_async"]
COMPARISON_ONLY = ["file_loc", "file_n_methods"]   # order as listed in the generated docs

MAIN_FEATURES = [
    # name (9)
    "name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit",
    "name_short_tokens", "name_no_vowel_tokens", "name_starts_verb", "name_single_char",
    # code (16; is_async dropped)
    "loc", "n_params", "n_defaults", "has_varargs", "has_kwargs", "n_returns", "n_calls",
    "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub", "n_decorators",
    "is_property", "is_static_or_class", "name_body_overlap",
    # context (4; class_depth dropped, call_sites excluded, file size comparison-only)
    "in_class", "class_n_methods", "is_test_file", "style_mismatch",
    # relative to file (11; rel_call_sites excluded)
    "name_token_sharing", "first_token_shared", "style_share_in_file", "rel_name_len",
    "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic",
    "rel_doc_len", "rel_name_body_overlap",
]

COMPARISON_FEATURES = [
    "name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit",
    "name_short_tokens", "name_no_vowel_tokens", "name_starts_verb", "name_single_char",
    "loc", "n_params", "n_defaults", "has_varargs", "has_kwargs", "n_returns", "n_calls",
    "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub", "n_decorators",
    "is_property", "is_static_or_class", "name_body_overlap",
    "in_class", "class_n_methods", "file_n_methods", "file_loc", "is_test_file", "style_mismatch",
    "name_token_sharing", "first_token_shared", "style_share_in_file", "rel_name_len",
    "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic",
    "rel_doc_len", "rel_name_body_overlap",
]

FEATURE_SETS = {"main": MAIN_FEATURES, "comparison": COMPARISON_FEATURES}
NOT_FOR_TRAINING = EXCLUDED_LEAKAGE + DROPPED_FEATURES


class FeatureSetError(ValueError):
    """Raised when a feature list breaks the approved configuration rules."""


def validate_feature_set(features, allow_file_size=True):
    """Raise FeatureSetError if `features` could not be a valid training input.

    Rejects: unknown columns (not one of the 46 master features, e.g. identifiers,
    the target or meta_* columns), duplicates, excluded leakage features, dropped
    features, and - when allow_file_size is False - file_loc / file_n_methods.
    """
    features = list(features)
    unknown = [f for f in features if f not in config.FEATURES]
    if unknown:
        raise FeatureSetError(f"not master predictive features: {unknown}")
    dup = sorted({f for f in features if features.count(f) > 1})
    if dup:
        raise FeatureSetError(f"duplicate features: {dup}")
    leak = [f for f in features if f in EXCLUDED_LEAKAGE]
    if leak:
        raise FeatureSetError(f"excluded for temporal leakage: {leak}")
    dropped = [f for f in features if f in DROPPED_FEATURES]
    if dropped:
        raise FeatureSetError(f"dropped features: {dropped}")
    if not allow_file_size:
        size = [f for f in features if f in COMPARISON_ONLY]
        if size:
            raise FeatureSetError(f"file-size features are comparison-only: {size}")
    return features


def check_configurations():
    """Fail loudly if the approved configurations were changed inconsistently."""
    validate_feature_set(MAIN_FEATURES, allow_file_size=False)
    validate_feature_set(COMPARISON_FEATURES)
    if (len(MAIN_FEATURES), len(COMPARISON_FEATURES)) != (40, 42):
        raise FeatureSetError(f"expected 40 / 42 features, got {len(MAIN_FEATURES)} / {len(COMPARISON_FEATURES)}")
    if set(COMPARISON_FEATURES) - set(MAIN_FEATURES) != set(COMPARISON_ONLY) or not set(MAIN_FEATURES) < set(COMPARISON_FEATURES):
        raise FeatureSetError("comparison must equal main + file_loc + file_n_methods")
    order = [f for f in config.FEATURES if f in COMPARISON_FEATURES]
    if COMPARISON_FEATURES != order or MAIN_FEATURES != [f for f in order if f not in COMPARISON_ONLY]:
        raise FeatureSetError("feature sets must keep the master dataset's column order")
    if set(config.FEATURES) - set(COMPARISON_FEATURES) != set(NOT_FOR_TRAINING):
        raise FeatureSetError("every master feature must be in a set, excluded or dropped")


def get_feature_set(name):
    """Return a copy of an approved feature list by name: 'main' (40) or 'comparison' (42)."""
    if name not in FEATURE_SETS:
        raise FeatureSetError(f"unknown feature set {name!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])


check_configurations()
