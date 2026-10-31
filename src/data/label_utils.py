"""
Helpers for describing labels: rename type, label provenance and project groups.
These produce metadata columns only (meta_*). None of them is a model feature.
"""
import difflib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.method_features import split_name  # noqa: E402

# ---------------------------------------------------------------- rename type
RENAME_TYPES = {
    "visibility_only": "Only a leading underscore added or removed (init_model -> _init_model)",
    "case_or_style_only": "Same words, different case/separator (compute_kl -> compute_KL, getX -> get_x)",
    "spelling_or_inflection": "One word respelled or inflected (Partioned -> Partitioned, tokens -> token)",
    "word_added": "Old words kept, new words added (grad_of -> insert_grad_of)",
    "word_removed": "Some old words dropped (bbox_encode -> encode)",
    "word_replaced": "Same number of words, at least one replaced (continuous -> vector)",
    "restructured": "Words added, removed and/or reordered together",
}


def _is_subsequence(short, long):
    it = iter(long)
    return all(t in it for t in short)


def rename_type(old, new):
    """Classify a rename from its old and new name only (descriptive, not a feature)."""
    if old == new:
        return "no_change"
    if old.lstrip("_") == new.lstrip("_"):
        return "visibility_only"
    a, b = split_name(old), split_name(new)
    if a == b or old.lower().replace("_", "") == new.lower().replace("_", ""):
        return "case_or_style_only"
    if len(a) == len(b):
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diff) == 1 and difflib.SequenceMatcher(None, *diff[0]).ratio() >= 0.75:
            return "spelling_or_inflection"
        if sorted(a) != sorted(b):
            return "word_replaced"
        return "restructured"               # same words, new order
    if len(b) > len(a) and _is_subsequence(a, b):
        return "word_added"
    if len(a) > len(b) and _is_subsequence(b, a):
        return "word_removed"
    return "restructured"


# ---------------------------------------------------------------- provenance
def label_provenance(label_source, confirmed_by_human=""):
    """
    A more accurate description than label_source:
      auto_confident -> auto_rule_confident   (similarity >= 0.9 and same parameters)
      hand_checked   -> ai_reviewed_unconfirmed, unless the review sheet's
                        confirmed_by_human column is filled in for that row
    """
    if label_source == "auto_confident":
        return "auto_rule_confident"
    if label_source == "hand_checked":
        return "human_confirmed" if str(confirmed_by_human).strip() else "ai_reviewed_unconfirmed"
    return "negative_name_kept" if label_source == "" else f"unknown:{label_source}"


# ---------------------------------------------------------------- project groups
GENERIC = {"src", "lib", "python", "tests", "test", "testing", "scripts", "examples", "codes", ""}


def project_guess(path):
    """Same rule as archive/legacy_scripts/Train_Models.py: first non-generic folder."""
    parts = str(path).split("/")[:-1]
    for p in parts:
        if p.lower() not in GENERIC:
            return p.lower()
    return str(path).split("/")[-1].lower()


def project_groups(df):
    """
    Returns (commit -> project_guess, commit -> project_group).

    project_guess is the legacy rule (majority first-folder name per commit).
    project_group joins commits that share a project_guess OR share a specific
    file path (a path with at least one non-generic folder). This merges cases
    where one repository was split across several guesses, e.g. 'imgaug' and
    'augmenters', so grouped cross-validation cannot leak a repository.
    """
    files = df[["commit", "file"]].drop_duplicates()
    guess = (files.assign(p=files["file"].map(project_guess))
             .merge(df[["commit", "file"]], on=["commit", "file"])
             .groupby("commit")["p"].agg(lambda s: s.value_counts().sort_index().idxmax()))

    parent = {c: c for c in guess.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    specific = files[files["file"].map(
        lambda p: any(d.lower() not in GENERIC for d in p.split("/")[:-1]))]
    for _, g in specific.groupby("file"):
        cs = list(g["commit"])
        for c in cs[1:]:
            union(cs[0], c)
    for _, g in guess.groupby(guess):
        cs = list(g.index)
        for c in cs[1:]:
            union(cs[0], c)

    root = pd.Series({c: find(c) for c in guess.index})
    # name each group after its most common project_guess (ties: alphabetical)
    names = {}
    for r, members in root.groupby(root):
        counts = guess[members.index].value_counts()
        names[r] = sorted(counts[counts == counts.max()].index)[0]
    group = root.map(names)
    return guess, group
