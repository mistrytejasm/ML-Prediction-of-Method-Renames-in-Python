r"""
build_master_dataset.py
Creates the validated master dataset from the regenerated feature table
(the output of src/data/extract_features.py). Compared with the original v2 it
  - drops the 8 rows of the duplicate commit folder dd7f9fa8..._<suffix>
  - has call_sites and rel_* recomputed from the Before code only
validate_dataset.py checks that nothing else differs.

  rows      = one per method observation (no rows added)
  columns   = 4 identifiers + target + 46 Before-only features + meta_* columns

The meta_* columns describe where each label came from. They are NOT model
features: several (meta_new_name, meta_rename_type, meta_label_similarity)
come from the After code or from the labelling process and would leak the
target. Use the schema file to select the feature columns.

Outputs:
  data/processed/master/actref_master_dataset.csv
  data/processed/master/actref_master_dataset_schema.json
Run: python src/data/build_master_dataset.py
"""
import argparse
import ast
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.label_utils import RENAME_TYPES, label_provenance, project_groups, rename_type  # noqa: E402
from src.features.method_features import parse  # noqa: E402

log = logging.getLogger("build_master_dataset")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duplicate_definition_flags(df):
    """1 if the method's class+name is defined more than once in its Before file."""
    flags = {}
    for (commit, f), g in df.groupby(["commit", "file"]):
        _, tree = parse(config.RAW_COMMITS / commit / "Before" / f.replace("/", "#"))
        counts = Counter()

        def visit(node, cp):
            for c in ast.iter_child_nodes(node):
                if isinstance(c, ast.ClassDef):
                    visit(c, f"{cp}.{c.name}" if cp else c.name)
                elif isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    counts[(cp, c.name)] += 1
        if tree is not None:
            visit(tree, "")
        for idx, cls, name in zip(g.index, g["class"], g["name"]):
            flags[idx] = int(counts.get((cls, name), 0) > 1)
    return pd.Series(flags)


def build(v2, positives, review_done, with_raw=True):
    df = v2.copy()
    pos = positives.copy()
    pos["class"] = pos["class"].astype(str)
    review_done = review_done.fillna("")
    human = {tuple(r) for r in review_done.loc[review_done["confirmed_by_human"].astype(str).str.strip() != "",
                                               ["commit", "file", "class", "old_name"]].astype(str).values}
    meta = pos.rename(columns={"old_name": "name"})[
        config.KEY + ["new_name", "similarity", "label_source"]].copy()
    meta["meta_label_provenance"] = [
        label_provenance(s, "yes" if tuple(k) in human else "")
        for s, k in zip(meta["label_source"], meta[config.KEY].astype(str).values)]
    meta["meta_rename_type"] = [rename_type(o, n) for o, n in zip(meta["name"], meta["new_name"])]
    meta = meta.rename(columns={"new_name": "meta_new_name", "similarity": "meta_label_similarity",
                                "label_source": "meta_label_source"})

    df = df.merge(meta, on=config.KEY, how="left", validate="one_to_one")
    neg = df[config.TARGET] == 0
    df.loc[neg, "meta_label_source"] = "negative_name_kept"
    df.loc[neg, "meta_label_provenance"] = "negative_name_kept"
    df.loc[neg, "meta_rename_type"] = "not_renamed"
    df["meta_new_name"] = df["meta_new_name"].fillna("")

    guess, group = project_groups(df)
    df["meta_project_guess"] = df["commit"].map(guess)
    df["meta_project_group"] = df["commit"].map(group)
    if with_raw:
        df["meta_duplicate_definition_in_before"] = duplicate_definition_flags(df)

    meta_cols = [c for c in config.META_COLS if c in df] + \
                (["meta_duplicate_definition_in_before"] if with_raw else [])
    df = df[config.ID_COLS + [config.TARGET] + config.FEATURES + meta_cols]
    return df.sort_values(config.KEY).reset_index(drop=True), meta_cols


def schema(df, meta_cols, sources):
    return {
        "dataset": "actref_master_dataset",
        "created": date.today().isoformat(),
        "created_by": "src/data/build_master_dataset.py",
        "description": "One row per method observed in a Before file of an ActRef commit. "
                       "renamed = 1 if the method was renamed in that commit (validated positive label), "
                       "0 if the same class + name still exists in the After file.",
        "rows": len(df),
        "columns": len(df.columns),
        "positives": int(df[config.TARGET].sum()),
        "negatives": int((df[config.TARGET] == 0).sum()),
        "positive_rate": round(float(df[config.TARGET].mean()), 6),
        "commits": int(df["commit"].nunique()),
        "project_groups": int(df["meta_project_group"].nunique()),
        "key": config.KEY,
        "roles": {
            "identifier": config.ID_COLS,
            "target": config.TARGET,
            "feature": config.FEATURES,
            "feature_groups": config.FEATURE_GROUPS,
            "metadata_not_for_training": meta_cols,
            "group_column_for_cv": "meta_project_group",
            "file_size_features_to_ablate": config.FILE_SIZE_FEATURES,
        },
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "values": {
            "meta_label_source": df["meta_label_source"].value_counts().to_dict(),
            "meta_label_provenance": df["meta_label_provenance"].value_counts().to_dict(),
            "meta_rename_type": df["meta_rename_type"].value_counts().to_dict(),
        },
        "rename_type_definitions": RENAME_TYPES,
        "changes_from_original_v2": {
            "rows_removed": "8 rows of the byte-identical duplicate commit folder dd7f9fa8..._<suffix>",
            "features_recomputed_before_only": ["call_sites"] + [f for f in config.FEATURES
                                                                  if f.startswith("rel_")],
            "reason": "in v2 these depended on the After code (deleted methods / files without an After pair)",
        },
        "sources": sources,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=config.REGENERATED_FEATURES,
                    help="output of src/data/extract_features.py")
    ap.add_argument("--out", type=Path, default=config.MASTER)
    ap.add_argument("--no-raw", action="store_true", help="skip the duplicate-definition flag (needs raw data)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.features.exists():
        raise SystemExit(f"{args.features} not found. Run: python src/data/extract_features.py")
    v2 = pd.read_csv(args.features, keep_default_na=False)
    v2["class"] = v2["class"].astype(str)
    pos = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)
    rev = pd.read_excel(config.REVIEW_SHEET_DONE, dtype=str)
    with_raw = not args.no_raw and config.RAW_COMMITS.exists()
    df, meta_cols = build(v2, pos, rev, with_raw)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    sources = {str(p.relative_to(config.ROOT).as_posix()): sha256(p)
               for p in (args.features, config.V2_FEATURES, config.TRAIN_POSITIVES, config.REVIEW_SHEET_DONE)}
    sources["output_sha256"] = sha256(args.out)
    sch = schema(df, meta_cols, sources)
    args.out.with_name(args.out.stem + "_schema.json").write_text(json.dumps(sch, indent=2), encoding="utf-8")
    log.info("Rows %d | columns %d | positives %d | negatives %d | features %d | groups %d",
             sch["rows"], sch["columns"], sch["positives"], sch["negatives"], len(config.FEATURES),
             sch["project_groups"])
    log.info("Saved %s", args.out)
    return df


if __name__ == "__main__":
    main()
