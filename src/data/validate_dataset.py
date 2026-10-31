r"""
validate_dataset.py
Checks the master dataset. Run it every time the dataset is regenerated.

Checks: schema and column roles, row counts, binary target, identifier
uniqueness, missing / infinite values, value ranges, constant and highly
correlated features, positive-label reconciliation with
rename_positive_labels.csv, metadata consistency, project groups, and that the
master differs from the original v2 only by the duplicate commit folder and
the Before-only recomputation of call_sites and rel_* (see extract_features.py).

Output: reports/audit/dataset/dataset_validation_results.json
Exit code 1 if any check FAILs.
Run: python src/data/validate_dataset.py
"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.label_utils import RENAME_TYPES  # noqa: E402
from src.data.raw_commits import HASH  # noqa: E402

log = logging.getLogger("validate_dataset")

BINARY = ["name_is_private", "name_is_camel", "name_has_digit", "name_starts_verb", "name_single_char",
          "has_varargs", "has_kwargs", "has_docstring", "is_stub", "is_async", "is_property",
          "is_static_or_class", "in_class", "is_test_file", "style_mismatch", "first_token_shared"]
RATIOS = ["name_body_overlap", "name_token_sharing", "style_share_in_file"]
AT_LEAST_ONE = ["name_len", "name_tokens", "loc", "cyclomatic", "file_n_methods", "file_loc"]
NON_NEGATIVE = [f for f in config.FEATURES if not f.startswith("rel_")]
FORBIDDEN_AS_FEATURE = {"old_name", "new_name", "similarity", "label_source", "same_args", "renamed"}
CORR_LIMIT = 0.95
# Features recomputed Before-only by extract_features.py (they depended on After code in v2)
LEAK_FIXED_FEATURES = ["call_sites"] + [f for f in config.FEATURES if f.startswith("rel_")]


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail, warn=False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.items.append({"check": name, "status": status, "detail": detail})
        getattr(log, "info" if ok else "warning")("%-4s %s: %s", status, name, detail)


def validate(master, schema, positives, v2=None, output_sha=None):
    ck = Checks()
    feats = schema["roles"]["feature"]
    meta = schema["roles"]["metadata_not_for_training"]

    # schema / columns
    expected = config.ID_COLS + [config.TARGET] + feats + meta
    ck.add("columns match the schema (names and order)", list(master.columns) == expected,
           f"{len(master.columns)} columns, schema {len(expected)}")
    ck.add("feature list is the 46 documented features", feats == config.FEATURES, f"{len(feats)} features")
    bad = [f for f in feats if f in FORBIDDEN_AS_FEATURE or f.startswith("meta_") or f in config.ID_COLS]
    ck.add("no identifier, target, label or metadata column is a feature", not bad, f"offending: {bad}")
    ck.add("row count matches the schema", len(master) == schema["rows"], f"{len(master)} rows")
    if output_sha:
        ck.add("file checksum matches the schema", output_sha == schema["sources"].get("output_sha256"),
               output_sha[:16])

    # target
    y = master[config.TARGET]
    ck.add("target is binary 0/1", set(y.unique()) <= {0, 1}, f"values {sorted(y.unique().tolist())}")
    ck.add("positives / negatives", int(y.sum()) == len(positives),
           f"{int(y.sum())} positives, {int((y == 0).sum())} negatives, rate {y.mean():.4%}")

    # identifiers
    dup = master.duplicated(config.KEY)
    ck.add("one row per (commit, file, class, name)", not dup.any(), f"{int(dup.sum())} duplicates")
    ck.add("commit ids are 40-character hashes", master["commit"].str.fullmatch(HASH.pattern).all(),
           f"{int((~master['commit'].str.fullmatch(HASH.pattern)).sum())} invalid")
    ck.add("file paths end with .py", master["file"].str.endswith(".py").all(), "")
    ck.add("method names are non-empty", (master["name"].str.len() > 0).all(), "")

    # feature values
    X = master[feats]
    non_numeric = [f for f in feats if not pd.api.types.is_numeric_dtype(X[f])]
    ck.add("all features are numeric", not non_numeric, f"non-numeric: {non_numeric}")
    Xn = X.apply(pd.to_numeric, errors="coerce")
    ck.add("no missing feature values", int(Xn.isna().sum().sum()) == 0, f"{int(Xn.isna().sum().sum())} missing")
    ck.add("no infinite feature values", int(np.isinf(Xn.values).sum()) == 0, f"{int(np.isinf(Xn.values).sum())}")
    bad_bin = [f for f in BINARY if not set(Xn[f].unique()) <= {0, 1}]
    ck.add("binary flags are 0/1", not bad_bin, f"bad: {bad_bin}")
    bad_ratio = [f for f in RATIOS if not Xn[f].between(0, 1).all()]
    ck.add("ratios are within [0, 1]", not bad_ratio, f"bad: {bad_ratio}")
    bad_neg = [f for f in NON_NEGATIVE if (Xn[f] < 0).any()]
    ck.add("count features are non-negative", not bad_neg, f"bad: {bad_neg}")
    bad_one = [f for f in AT_LEAST_ONE if (Xn[f] < 1).any()]
    ck.add("size features are at least 1", not bad_one, f"bad: {bad_one}")
    const = [f for f in feats if Xn[f].nunique() <= 1]
    ck.add("no constant features", not const, f"constant: {const}", warn=True)
    rare = {f: int(Xn[f].sum()) for f in BINARY if 0 < Xn[f].sum() < 30}
    ck.add("binary flags with fewer than 30 ones", not rare, str(rare), warn=True)
    corr = Xn.corr().abs()
    pairs = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(feats) for b in feats[i + 1:]
             if corr.loc[a, b] >= CORR_LIMIT]
    ck.add(f"feature pairs with |r| >= {CORR_LIMIT}", not pairs, str(pairs), warn=True)

    # labels
    pos_keys = set(map(tuple, positives[["commit", "file", "class", "old_name"]].astype(str).values))
    m_pos = set(map(tuple, master.loc[y == 1, config.KEY].astype(str).values))
    ck.add("renamed = 1 rows are exactly rename_positive_labels.csv", m_pos == pos_keys,
           f"missing {len(pos_keys - m_pos)}, extra {len(m_pos - pos_keys)}")
    ck.add("label_source values", set(master["meta_label_source"]) <=
           {"auto_confident", "hand_checked", "negative_name_kept"}, str(master["meta_label_source"].value_counts().to_dict()))
    ok_types = set(RENAME_TYPES) | {"not_renamed"}
    ck.add("rename types are valid and match the target",
           set(master["meta_rename_type"]) <= ok_types and
           ((master["meta_rename_type"] == "not_renamed") == (y == 0)).all(), "")
    ck.add("meta_new_name is set for positives only",
           ((master["meta_new_name"].astype(str) != "") == (y == 1)).all(), "")
    unconf = int((master["meta_label_provenance"] == "ai_reviewed_unconfirmed").sum())
    ck.add("positives without human confirmation", unconf == 0,
           f"{unconf} positives are ai_reviewed_unconfirmed", warn=True)

    # groups
    per_commit = master.groupby("commit")["meta_project_group"].nunique()
    ck.add("each commit belongs to exactly one project group", (per_commit == 1).all(),
           f"{master['meta_project_group'].nunique()} groups")
    ck.add("project group is never empty", (master["meta_project_group"].astype(str) != "").all(), "")
    feat_dup = master.duplicated(feats + ["name"], keep=False)
    cross = master[feat_dup].groupby(feats + ["name"])["meta_project_group"].nunique()
    ck.add("repeated observations (same name + features) stay inside one project group",
           (cross <= 1).all(), f"{int(feat_dup.sum())} rows in {len(cross)} clusters, "
                               f"{int((cross > 1).sum())} span several groups")

    # comparison with the original v2
    if v2 is not None:
        mk = set(map(tuple, master[config.KEY].astype(str).values))
        vk = set(map(tuple, v2[config.KEY].astype(str).values))
        only_v2 = vk - mk
        ok_dup = all(not HASH.match(k[0]) for k in only_v2)
        ck.add("rows dropped from v2 come only from duplicate commit folders",
               not (mk - vk) and ok_dup, f"{len(only_v2)} dropped, {len(mk - vk)} added")
        a = master.set_index(config.KEY)[feats + [config.TARGET]].sort_index()
        b = v2.set_index(config.KEY).loc[a.index, feats + [config.TARGET]].sort_index()
        changed = (a.astype(float) != b.astype(float))
        fixed = set(LEAK_FIXED_FEATURES)
        unexpected = [c for c in changed.columns if changed[c].any() and c not in fixed]
        ck.add("labels and all non-leak-fixed features equal the original v2 for every kept row",
               not unexpected, f"unexpected differences in: {unexpected}")
        per_col = {c: int(changed[c].sum()) for c in LEAK_FIXED_FEATURES if changed[c].any()}
        rows_changed = changed[list(fixed)].any(axis=1)
        ck.add("leak fix: rows whose call_sites / rel_* values differ from v2", True,
               f"{int(rows_changed.sum())} rows ({int(a.loc[rows_changed, config.TARGET].sum())} positives); "
               f"changed values per column: {per_col}")
    return ck


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path, default=config.MASTER)
    ap.add_argument("--out", type=Path, default=config.REPORTS_AUDIT / "dataset" / "dataset_validation_results.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    schema_path = args.master.with_name(args.master.stem + "_schema.json")
    for p in (args.master, schema_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Run: python src/data/build_master_dataset.py")
    master = pd.read_csv(args.master, keep_default_na=False)
    master["class"] = master["class"].astype(str)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    pos = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)
    v2 = pd.read_csv(config.V2_FEATURES, keep_default_na=False) if config.V2_FEATURES.exists() else None
    sha = hashlib.sha256(args.master.read_bytes()).hexdigest()

    ck = validate(master, schema, pos, v2, sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ck.items, indent=2, default=str), encoding="utf-8")
    failed = [c for c in ck.items if c["status"] == "FAIL"]
    warned = [c for c in ck.items if c["status"] == "WARN"]
    log.info("%d checks: %d passed, %d warnings, %d failed. Saved %s", len(ck.items),
             len(ck.items) - len(failed) - len(warned), len(warned), len(failed), args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
