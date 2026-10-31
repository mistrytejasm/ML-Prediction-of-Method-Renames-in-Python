"""
Stage 7: a chance-reference baseline and an exploratory naming heuristic, in
addition to DummyClassifier(strategy="prior").

Both are sklearn classifiers used through build_model_pipeline() (input check ->
model, raw features, no transform), so they share the saved folds and the Stage 6
evaluation. Neither sees validation or test labels: fit() receives the training
rows only and predict_proba() receives X only.

StratifiedRandomClassifier ("stratified_random")
  fit(X, y)          learns only the training positive rate pi (prevalence_)
  predict_proba(X)   U_i ~ Uniform(0, 1) from np.random.default_rng(random_state);
                     score_i = 2 * pi * U_i when pi <= 0.5 (uniform on [0, 2 pi]),
                     else 1 - 2 * (1 - pi) * (1 - U_i): a random ranking whose
                     scores lie in [0, 1] with mean exactly pi (no underflow, no ties)
  predict(X)         1 where U_i >= 1 - pi, i.e. a random pi share of rows is
                     flagged, as in sklearn's "stratified" strategy
  The scores ignore the feature values and depend only on the number of rows and
  the seed, so the same call always returns the same scores.

  How it differs from DummyClassifier(strategy="prior"): the prior baseline gives
  every row the same score pi. Its AP equals the positive rate exactly (all rows
  tied), its ROC-AUC is 0.5, and at any threshold it flags either every row or none.
  The random baseline ranks rows randomly, so it shows the AP and ROC-AUC of a
  real but uninformative ranking (expected about pi and 0.5, with sampling spread),
  and it produces non-degenerate precision / recall / F1 / MCC at a selected
  threshold. Both are chance references and overlap in that purpose; the random
  one is the reference for the threshold metrics and for fold-to-fold variation.

NamingHeuristicClassifier ("naming_heuristic")
  RESEARCH STATUS: exploratory, post-hoc, rule-based naming heuristic. It is not a
  preregistered or independent confirmatory baseline: its flags were chosen after
  full-dataset, label-derived feature summaries (class means in
  docs/data_dictionary.md, among others) had been seen; see
  docs/naming_heuristic.md. Flags and weights were not tuned on labels.

  No training. Score = share of six flags raised by a method's Before name
  (equal weights; possible values 0, 1/6, ..., 1):

    flag                         condition                 what the feature actually measures
    short_token                  name_short_tokens > 0     a word of 1-2 letters that is not a digit run and
                                                           not in SHORT_OK = {id, io, ip, ok, db, os, ui, x, y,
                                                           z, fn, nn, lr}; includes English words such as
                                                           is, to, on, of, in, by, at (is_valid, to_dict)
    vowelless_token              name_no_vowel_tokens > 0  a word of 2+ letters with no a/e/i/o/u, not a digit
                                                           run (cnt, nb; also db, fn, lr, nn, tf)
    single_character_name        name_single_char == 1     the name without leading / trailing _ is one letter
    contains_digit               name_has_digit == 1       any digit in the name (run2, but also md5, utf8, v2)
    first_word_not_in_verb_list  name_starts_verb == 0     first word is not in a fixed list of 68 verbs
                                                           (VERBS in src/features/method_features.py, which
                                                           also contains to, from, is, has); verbs missing from
                                                           the list (visit, draw, emit) also raise it
    camel_case_in_snake_file     style_mismatch == 1       camelCase name in a file whose methods are mostly
                                                           snake_case / lower

  Conceptual rationale and what each flag is:
    naming defect (abbreviation / brevity):   short_token, vowelless_token,
                                              single_character_name
    naming convention (debatable defect):     contains_digit (numbered names; often
                                              legitimate), first_word_not_in_verb_list
                                              (method names as verb phrases; properties,
                                              factories and callbacks are often nouns)
    contextual style inconsistency:           camel_case_in_snake_file. The same name is
                                              fine in a camelCase file, so this is a
                                              property of the name IN its file, not of
                                              the name itself.

  The flags are NOT six independent defects:
    - short_token, vowelless_token and single_character_name all measure brevity /
      abbreviation, so that one characteristic carries up to 3/6 of the weight
    - a two-letter vowelless word (nb, tf) raises short_token AND vowelless_token
    - a one-letter name always raises single_character_name and
      first_word_not_in_verb_list (no listed verb has one letter), and short_token
      too unless it is x, y or z: score 3/6
    - md5_hash raises short_token, vowelless_token, contains_digit and
      first_word_not_in_verb_list (4/6) from one token
    - camel_case_in_snake_file can only fire for camelCase names

  name_is_private is excluded: a leading underscore marks visibility (an API
  decision), not a naming defect, and split_name() strips it before the other
  flags are computed.

  Operating points: predict_proba() gives the continuous score; the controlled
  evaluation uses it with the Stage 6 max-F1 threshold chosen on development OOF
  scores, like every other model. predict() is a separate, fixed convenience rule
  (at least DEFAULT_RULE_MIN_FLAGS = 3 of 6 flags) and must not replace that
  threshold selection.

  Other limitations: scores take only 7 values, so many methods tie; renames that
  only change visibility (a leading underscore) or meaning are invisible to the rule.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_sets import MAIN_FEATURES  # noqa: E402

NAMING_FLAGS = {
    "short_token": ("name_short_tokens", lambda v: v > 0),
    "vowelless_token": ("name_no_vowel_tokens", lambda v: v > 0),
    "single_character_name": ("name_single_char", lambda v: v == 1),
    "contains_digit": ("name_has_digit", lambda v: v == 1),
    "first_word_not_in_verb_list": ("name_starts_verb", lambda v: v == 0),
    "camel_case_in_snake_file": ("style_mismatch", lambda v: v == 1),
}
NAMING_FEATURES = [feature for feature, _ in NAMING_FLAGS.values()]
DEFAULT_RULE_MIN_FLAGS = 3                 # predict() only; not the evaluation operating point
BINARY_FEATURES = {"name_single_char", "name_has_digit", "name_starts_verb", "style_mismatch"}


class BaselineInputError(ValueError):
    """Invalid labels or feature values for a baseline."""


def _binary_labels(y):
    arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if arr.ndim != 1 or len(arr) == 0 or arr.dtype == bool or not np.isin(arr, (0, 1)).all():
        raise BaselineInputError("y must be a non-empty 1-D array of 0 / 1 labels")
    return arr.astype(int)


def _n_rows(X):
    if not hasattr(X, "shape") or len(X.shape) != 2 or X.shape[0] == 0:
        raise BaselineInputError("X must be a non-empty 2-D table")
    return X.shape[0]


class StratifiedRandomClassifier(ClassifierMixin, BaseEstimator):
    """Random ranking whose mean score equals the training positive rate; ignores feature values."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        y = _binary_labels(y)
        if len(y) != _n_rows(X):
            raise BaselineInputError(f"X has {_n_rows(X)} rows, y {len(y)}")
        if y.min() == y.max():
            raise BaselineInputError("training labels need both classes to define a positive rate in (0, 1)")
        if not isinstance(self.random_state, (int, np.integer)) or isinstance(self.random_state, bool):
            raise BaselineInputError("random_state must be a fixed integer")
        self.prevalence_ = float(y.mean())
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _uniform(self, X):
        check_is_fitted(self, "prevalence_")
        return np.random.default_rng(self.random_state).random(_n_rows(X))

    def predict_proba(self, X):
        u = self._uniform(X)
        pi = self.prevalence_
        score = 2 * pi * u if pi <= 0.5 else 1 - 2 * (1 - pi) * (1 - u)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (self._uniform(X) >= 1 - self.prevalence_).astype(int)


class NamingHeuristicClassifier(ClassifierMixin, BaseEstimator):
    """Exploratory naming heuristic: deterministic share of six name flags; learns nothing."""

    def fit(self, X, y):
        y = _binary_labels(y)                                # interface only; labels are not used
        if len(y) != _n_rows(X):
            raise BaselineInputError(f"X has {_n_rows(X)} rows, y {len(y)}")
        self._check_columns(X)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _check_columns(X):
        if not isinstance(X, pd.DataFrame):
            raise BaselineInputError("X must be a pandas DataFrame with named feature columns")
        missing = [f for f in NAMING_FEATURES if f not in X.columns]
        if missing:
            raise BaselineInputError(f"X lacks the name features the heuristic needs: {missing}")
        values = X[NAMING_FEATURES].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise BaselineInputError("name features must be finite and >= 0")
        binary = X[sorted(BINARY_FEATURES)].to_numpy(dtype=float)
        if not np.isin(binary, (0, 1)).all():
            raise BaselineInputError(f"{sorted(BINARY_FEATURES)} must be 0 / 1")

    def flags(self, X):
        """One 0 / 1 column per flag, in NAMING_FLAGS order (for inspection)."""
        check_is_fitted(self, "classes_")
        self._check_columns(X)
        return pd.DataFrame({name: rule(X[feature].to_numpy()).astype(int)
                             for name, (feature, rule) in NAMING_FLAGS.items()}, index=X.index)

    def predict_proba(self, X):
        score = self.flags(X).to_numpy().mean(axis=1)
        return np.column_stack([1 - score, score])

    def predict(self, X):
        """Fixed convenience rule: at least DEFAULT_RULE_MIN_FLAGS of 6 flags. Not the evaluation
        operating point, which is the Stage 6 max-F1 threshold applied to predict_proba()."""
        return (self.flags(X).to_numpy().sum(axis=1) >= DEFAULT_RULE_MIN_FLAGS).astype(int)


if not set(NAMING_FEATURES) <= set(MAIN_FEATURES):
    raise BaselineInputError("the naming heuristic must use approved main-40 features only")
