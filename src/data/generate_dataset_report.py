r"""
generate_dataset_report.py
Profiles every dataset file and writes the documentation that depends on the
data, so the documents always match the files:

  docs/data_dictionary.md              every column of the master dataset
  docs/data_dictionary.csv             the same, as a table
  docs/data_quality_report.md          profile of every CSV / Excel file + master checks
  reports/audit/dataset/dataset_profiles.csv     one row per file x column
  reports/audit/dataset/feature_statistics.csv   per-feature stats by class
  reports/audit/dataset/feature_correlations.csv pairs with |r| >= 0.8

Read-only on all inputs. Run after validate_labels.py and validate_dataset.py:
    python src/data/generate_dataset_report.py
"""
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.features.feature_sets import (COMPARISON_FEATURES, COMPARISON_ONLY, DROPPED_FEATURES,  # noqa: E402
                                       EXCLUDED_LEAKAGE, MAIN_FEATURES, check_configurations)

log = logging.getLogger("generate_dataset_report")

# feature -> (meaning, how it is calculated, notes / concerns)
FEATURE_INFO = {
    "name_len": ("Length of the method name", "len(name)", ""),
    "name_tokens": ("Number of words in the name", "snake_case and camelCase split", ""),
    "name_is_private": ("Name starts with _", "name.startswith('_')",
                        "Visibility-only renames (43 positives) toggle exactly this flag"),
    "name_is_camel": ("Name is camelCase", "style_of(name) == 'camel'", ""),
    "name_has_digit": ("Name contains a digit", "any digit in name", ""),
    "name_short_tokens": ("Words of 1-2 letters that are not common abbreviations",
                          "len(token) <= 2 and token not in {id, io, ip, ok, db, os, ui, x, y, z, fn, nn, lr}", ""),
    "name_no_vowel_tokens": ("Words with no vowel (e.g. 'cnt')", "len(token) > 1 and no a/e/i/o/u", ""),
    "name_starts_verb": ("First word is a common verb", "first token in a fixed list of 72 verbs", ""),
    "name_single_char": ("Name is a single character", "len(name.strip('_')) == 1", "Very rare"),
    "loc": ("Lines of code of the method", "end_lineno - lineno + 1", ""),
    "n_params": ("Number of parameters (self/cls excluded)", "args + kw-only args", ""),
    "n_defaults": ("Parameters with a default value", "defaults + non-None kw_defaults", ""),
    "has_varargs": ("Has *args", "args.vararg is not None", ""),
    "has_kwargs": ("Has **kwargs", "args.kwarg is not None", ""),
    "n_returns": ("Number of return statements", "count of ast.Return (nested functions included)", ""),
    "n_calls": ("Number of function calls in the body", "count of ast.Call", ""),
    "cyclomatic": ("Cyclomatic complexity", "1 + branches (if/for/while/except/with/ifexp/comprehension/assert) + extra boolean operands", ""),
    "max_nesting": ("Deepest nesting of if/for/while/with/try", "recursive AST walk", ""),
    "has_docstring": ("Has a docstring", "ast.get_docstring", ""),
    "doc_len": ("Docstring length in characters", "len(docstring)", "Long right tail (max > 10,000)"),
    "is_stub": ("Body is only pass / NotImplementedError / a constant", "statement check", ""),
    "is_async": ("async def", "isinstance AsyncFunctionDef", "Rare"),
    "n_decorators": ("Number of decorators", "len(decorator_list)", ""),
    "is_property": ("Has a property decorator", "'property' in any decorator", ""),
    "is_static_or_class": ("@staticmethod or @classmethod", "decorator text match", ""),
    "name_body_overlap": ("Share of the name's words that appear in identifiers of the body",
                          "name tokens found among body Name/Attribute/arg tokens / name tokens", ""),
    # in_class / class_depth notes are computed from the master in main() (near_dup)
    "in_class": ("Method is inside a class", "class path is not empty", ""),
    "class_depth": ("Class nesting depth (0 = module function)", "depth while visiting ClassDef nodes", ""),
    "class_n_methods": ("Methods in the same class", "functions directly in the class (0 for module functions)", ""),
    "file_n_methods": ("Methods in the Before file", "len(collect(Before tree))",
                       "Showed sampling bias in v1; test with and without"),
    "file_loc": ("Lines in the Before file", "text.count('\\n') + 1",
                 "Showed sampling bias in v1; test with and without"),
    "is_test_file": ("'test' appears in the file path", "'test' in path.lower()", ""),
    "style_mismatch": ("camelCase name in a mostly snake_case file", "camel and snake share of the file > 0.5", "Rare"),
    "call_sites": ("Times the name is called in the commit's Before files",
                   "regex r'\\bname\\s*\\(' over all parseable Before files of the commit, minus 1",
                   "Excluded: search scope depends on the future commit's changed-file list. "
                   "Only files changed by the commit are searched, so this is a partial count. "
                   "Recomputed Before-only (v2 skipped Before files without an After pair)"),
    "name_token_sharing": ("Share of the name's words used by other methods in the file",
                           "tokens appearing in >1 Before method name / tokens", ""),
    "first_token_shared": ("Another method in the file starts with the same word", "sibling token count > 1", ""),
    "style_share_in_file": ("Share of methods in the file with the same naming style", "same style / all methods", ""),
}
for col in ["name_len", "name_tokens", "loc", "n_params", "n_calls", "cyclomatic", "doc_len", "call_sites",
            "name_body_overlap"]:
    rel_note = ("Recomputed Before-only (v2 averaged only over methods whose label was known, "
                "which depended on the After code)")
    if col == "call_sites":
        rel_note = "Excluded: inherits the commit-scope dependence of call_sites. " + rel_note
    else:
        rel_note += ". Passed the Before-only recomputation audit (0 mismatches)"
    FEATURE_INFO["rel_" + col] = (
        f"{col} minus the average {col} of the eligible methods in the same Before file",
        f"{col} - mean({col}) over non-dunder, non-test Before methods of the file",
        rel_note)

# ---- Approved experimental training configurations: defined once in src/features/feature_sets.py.
# The 46-feature master dataset is not changed; these lists only select columns for training.
LEAKY_BEFORE_ONLY = "Before text, but only of the files the future commit changed"


def use_in_model(f, async_rows, async_renamed, equal_pct, pearson):
    if f in EXCLUDED_LEAKAGE:
        return "no: excluded from both experiments (temporal leakage)"
    if f == "class_depth":
        return f"no: dropped (near-duplicate of in_class, {equal_pct} equal; Pearson r = {pearson})"
    if f == "is_async":
        return f"no: dropped (near-constant: {async_rows} methods, {async_renamed} renamed)"
    if f in COMPARISON_ONLY:
        return "comparison experiment only"
    return "main + comparison"

ID_INFO = {
    "commit": "ActRef commit folder name (40-character hash)",
    "file": "Path of the Python file in the repository",
    "class": "Class path (Outer.Inner); empty for module-level functions",
    "name": "Method name in the Before code",
    "renamed": "TARGET: 1 = renamed in this commit, 0 = same class + name still exists in the After code",
}
META_INFO = {
    "meta_label_source": "Original label_source from rename_positive_labels.csv (auto_confident / hand_checked); negative_name_kept for 0",
    "meta_label_provenance": "Accurate provenance: auto_rule_confident / ai_reviewed_unconfirmed / human_confirmed / negative_name_kept",
    "meta_label_similarity": "Body similarity of the rename pair (labelling process; empty for negatives)",
    "meta_rename_type": "Kind of rename from old and new name (see schema); not_renamed for 0",
    "meta_new_name": "Name after the commit (positives only) - After information",
    "meta_project_guess": "Legacy project guess from the file path (first non-generic folder)",
    "meta_project_group": "Project group for grouped cross-validation (project guess + shared file paths)",
    "meta_duplicate_definition_in_before": "1 if class + name is defined more than once in the Before file (e.g. property getter + setter); features describe the last definition",
}


def profile(path):
    if path.suffix == ".xlsx":
        df = pd.read_excel(path, dtype=str).fillna("")
    else:
        df = pd.read_csv(path, keep_default_na=False)
    rows = []
    for c in df.columns:
        s = df[c]
        empty = int((s.astype(str) == "").sum())
        num = pd.to_numeric(s.replace("", np.nan), errors="coerce")
        is_num = num.notna().sum() == (len(s) - empty) and len(s) > empty
        rows.append({"file": path.relative_to(config.ROOT).as_posix(), "column": c,
                     "dtype": "numeric" if is_num else "text", "empty": empty, "unique": int(s.nunique()),
                     "min": float(num.min()) if is_num else "", "max": float(num.max()) if is_num else "",
                     "inf": int(np.isinf(num).sum()) if is_num else 0})
    return df, rows


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = config.REPORTS_AUDIT / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    config.DOCS.mkdir(exist_ok=True)

    files = [config.AUTO_RENAMES, config.CONFIDENT_RENAMES, config.REVIEW_SHEET, config.REVIEW_SHEET_DONE,
             config.TRAIN_POSITIVES, config.V1_FEATURES, config.V2_FEATURES, config.MASTER]
    summary, all_rows, frames = [], [], {}
    for p in files:
        df, rows = profile(p)
        frames[p] = df
        all_rows += rows
        key = [c for c in (config.KEY if "name" in df else ["commit", "file", "class", "old_name"]) if c in df]
        tgt = df["renamed"].astype(int) if "renamed" in df else None
        summary.append({
            "file": p.relative_to(config.ROOT).as_posix(), "rows": len(df), "columns": len(df.columns),
            "duplicate keys": int(df.duplicated(key).sum()) if key else "",
            "exact duplicate rows": int(df.duplicated().sum()),
            "empty cells": int(sum(r["empty"] for r in rows)),
            "empty cells outside 'class'": int(sum(r["empty"] for r in rows if r["column"] not in
                                                   ("class", "note", "confirmed_by_human", "is_rename",
                                                    "meta_new_name", "meta_label_similarity"))),
            "renamed=1": int(tgt.sum()) if tgt is not None else "",
            "renamed=0": int((tgt == 0).sum()) if tgt is not None else "",
        })
    pd.DataFrame(all_rows).to_csv(out / "dataset_profiles.csv", index=False)
    summary = pd.DataFrame(summary)

    m = frames[config.MASTER]
    X = m[config.FEATURES].astype(float)
    y = m[config.TARGET].astype(int)
    stats = pd.DataFrame({
        "group": [next(g for g, fs in config.FEATURE_GROUPS.items() if f in fs) for f in config.FEATURES],
        "dtype": [str(m[f].dtype) for f in config.FEATURES],
        "min": X.min(), "median": X.median(), "p99": X.quantile(0.99), "max": X.max(),
        "unique": X.nunique(),
        "mean_renamed": X[y == 1].mean().round(3), "mean_not_renamed": X[y == 0].mean().round(3),
        "median_renamed": X[y == 1].median(), "median_not_renamed": X[y == 0].median(),
        "share_above_p99": ((X > X.quantile(0.99)).mean() * 100).round(2),
    })
    stats.index.name = "feature"
    stats.to_csv(out / "feature_statistics.csv")
    corr = X.corr()
    pairs = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(config.FEATURES)
             for b in config.FEATURES[i + 1:] if abs(corr.loc[a, b]) >= 0.8]
    pairs = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson_r"]).sort_values(
        "pearson_r", key=abs, ascending=False)
    pairs.to_csv(out / "feature_correlations.csv", index=False)

    # ------------------------------------------------ data dictionary
    check_configurations()
    equal_pct = f"{(m['class_depth'] == m['in_class']).mean():.4%}"
    pearson = f"{m['class_depth'].corr(m['in_class']):.6f}"
    near_dup = {"class_depth": f"Near-duplicate of in_class ({equal_pct} equal; Pearson r = {pearson})",
                "in_class": f"Near-duplicate of class_depth ({equal_pct} equal; Pearson r = {pearson})"}
    async_rows = int((m["is_async"] != 0).sum())
    async_renamed = int(m.loc[m["is_async"] != 0, config.TARGET].sum())
    dd = []
    for c in m.columns:
        if c in ID_INFO:
            role = "target" if c == config.TARGET else "identifier"
            dd.append({"column": c, "role": role, "group": "", "dtype": str(m[c].dtype), "meaning": ID_INFO[c],
                       "calculation": "", "before_only": "yes" if c != config.TARGET else "label",
                       "missing_possible": "class is empty for module functions" if c == "class" else "no",
                       "use_in_model": "no", "notes": ""})
        elif c in FEATURE_INFO:
            meaning, calc, note = FEATURE_INFO[c]
            g = stats.loc[c, "group"]
            dd.append({"column": c, "role": "feature", "group": g, "dtype": str(m[c].dtype), "meaning": meaning,
                       "calculation": calc,
                       "before_only": LEAKY_BEFORE_ONLY if c in EXCLUDED_LEAKAGE else "yes",
                       "missing_possible": "no",
                       "use_in_model": use_in_model(c, async_rows, async_renamed, equal_pct, pearson),
                       "notes": near_dup.get(c, note)})
        else:
            dd.append({"column": c, "role": "metadata", "group": "", "dtype": str(m[c].dtype),
                       "meaning": META_INFO.get(c, ""), "calculation": "", "before_only":
                       "no" if c in ("meta_new_name", "meta_rename_type", "meta_label_similarity") else "yes",
                       "missing_possible": "empty for negatives" if c in ("meta_new_name", "meta_label_similarity") else "no",
                       "use_in_model": "no (leaks the label or describes the labelling)", "notes": ""})
    dd = pd.DataFrame(dd)
    dd.to_csv(config.DOCS / "data_dictionary.csv", index=False)
    schema = json.loads(config.MASTER_SCHEMA.read_text(encoding="utf-8"))
    lines = [
        "# Data dictionary: actref_master_dataset.csv", "",
        f"Generated by `src/data/generate_dataset_report.py` on {date.today().isoformat()} from "
        f"`data/processed/master/actref_master_dataset.csv` ({schema['rows']:,} rows, {schema['columns']} columns).", "",
        f"Roles: **identifier** (never a model input), **target** (`renamed`), **feature** ({len(config.FEATURES)} "
        "audited predictive features, all computed from Before code), **metadata** (`meta_*`, labelling provenance, "
        "never a model input).", "",
        "## Master dataset and training configurations", "",
        f"The {len(config.FEATURES)}-feature master dataset is **unchanged** and remains the canonical, audited source "
        "dataset. The approved training configurations select columns from it; they are experimental settings, not "
        "changes to the master:", "",
        "| Configuration | Features | Definition |",
        "|---|---|---|",
        f"| Main experiment | **{len(MAIN_FEATURES)}** | {len(config.FEATURES)} minus "
        + ", ".join(f"`{f}`" for f in EXCLUDED_LEAKAGE + DROPPED_FEATURES + COMPARISON_ONLY) + " |",
        f"| Comparison experiment | **{len(COMPARISON_FEATURES)}** | Main + "
        + ", ".join(f"`{f}`" for f in COMPARISON_ONLY) + " |", "",
        "| Not used in either experiment | Reason |",
        "|---|---|",
        "| `call_sites` | Excluded: its search scope is every Before file the future commit changed, so it depends on "
        "the commit's changed-file list (`docs/temporal_leakage_audit.md`) |",
        "| `rel_call_sites` | Excluded: built from `call_sites`, so it inherits the same dependence |",
        f"| `class_depth` | Dropped: near-duplicate of `in_class` ({equal_pct} equal; Pearson r = {pearson}) |",
        f"| `is_async` | Dropped: near-constant ({async_rows} methods, "
        f"{'none' if async_renamed == 0 else async_renamed} renamed) |", "",
        "The other eight `rel_*` features passed the Before-only recomputation audit: 33,232 values recomputed from "
        "the Before file alone, 0 mismatches.", "",
        "**Research scope and data conditions**", "",
        "- The scope is methods belonging to files edited by the future commit (ActRef stores only changed files).",
        "- About 9.5% of eligible methods (3,263) have an unknown outcome (deleted, moved, or renamed undetected) and "
        "are excluded.",
        "- A file supplies rows only if both its Before and After versions parse.", "",
        "**Future work (not implemented, not approved):** an own-file-only call count as a replacement for "
        "`call_sites`, counting calls only inside the method's own Before file. It is not in the master dataset or "
        "the feature code.", "",
        "Select model inputs with the schema file, then apply the configuration:", "",
        "```python", "import json, pandas as pd",
        "schema = json.load(open('data/processed/master/actref_master_dataset_schema.json'))",
        "df = pd.read_csv('data/processed/master/actref_master_dataset.csv', keep_default_na=False)",
        "X_all = df[schema['roles']['feature']]                                        # 46 audited features",
        "X_comparison = X_all.drop(columns=['call_sites', 'rel_call_sites', 'class_depth', 'is_async'])   # 42",
        "X_main = X_comparison.drop(columns=['file_loc', 'file_n_methods'])                                # 40",
        "y, groups = df['renamed'], df['meta_project_group']",
        "```", "",
    ]
    for role, title in (("identifier", "Identifiers"), ("target", "Target"), ("feature", "Features"),
                        ("metadata", "Metadata (not for training)")):
        sub = dd[dd.role == role].drop(columns=["role"])
        if role != "feature":
            sub = sub.drop(columns=["group", "calculation"])
        lines += [f"## {title}", "", md_table(sub), ""]
    lines += ["## Feature value ranges and class means", "",
              "From the master dataset. p99 = 99th percentile.", "",
              md_table(stats.reset_index()[["feature", "group", "min", "median", "p99", "max", "unique",
                                            "mean_renamed", "mean_not_renamed"]]), ""]
    (config.DOCS / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------ data quality report
    lab = json.loads((config.REPORTS_AUDIT / "labels" / "label_validation_results.json").read_text(encoding="utf-8"))
    dsv = json.loads((config.REPORTS_AUDIT / "dataset" / "dataset_validation_results.json").read_text(encoding="utf-8"))
    inv_p = config.REPORTS_AUDIT / "raw" / "raw_data_inventory.json"
    inv = json.loads(inv_p.read_text(encoding="utf-8")) if inv_p.exists() else {}
    checks = pd.DataFrame(lab["checks"] + dsv)[["status", "check", "detail"]]
    size = stats.loc[config.FILE_SIZE_FEATURES, ["median_renamed", "median_not_renamed", "mean_renamed",
                                                 "mean_not_renamed"]].reset_index()
    q = [
        "# Data quality report", "",
        f"Generated by `src/data/generate_dataset_report.py` on {date.today().isoformat()}. "
        "All numbers are computed from the files; nothing is typed by hand.", "",
        "## Files profiled", "", md_table(summary), "",
        "Empty cells in `class` mean a module-level function. Empty `note` / `confirmed_by_human` / `is_rename` "
        "cells belong to the review sheets. Per-column details: `reports/audit/dataset/dataset_profiles.csv`.", "",
    ]
    if inv:
        q += ["## Raw ActRef data", "",
              md_table(pd.DataFrame([(k, v) for k, v in inv.items() if not isinstance(v, (list, dict))],
                                    columns=["item", "value"])), "",
              f"Duplicate commit folders skipped by the pipeline: {inv.get('duplicate_commit_folders (skipped by the pipeline)')}", ""]
    q += ["## Validation checks", "",
          "From `validate_labels.py --raw` and `validate_dataset.py`.", "", md_table(checks), "",
          "## Feature quality", "",
          f"- Missing values: {int(X.isna().sum().sum())}; infinite values: {int(np.isinf(X.values).sum())}.",
          f"- Constant features: {[f for f in config.FEATURES if X[f].nunique() <= 1] or 'none'}.",
          f"- Pairs with |r| >= 0.8: {len(pairs)} (see `feature_correlations.csv`); the strongest is "
          f"{pairs.iloc[0].feature_a} / {pairs.iloc[0].feature_b} (r = {pairs.iloc[0].pearson_r}).",
          "- Heavy right tails (max far above p99): " + ", ".join(
              f"{f} (p99 {stats.loc[f, 'p99']:.0f}, max {stats.loc[f, 'max']:.0f})"
              for f in ["doc_len", "file_loc", "loc", "n_calls", "call_sites"]) + ".", "",
          "## File-size features", "",
          "In v1 these separated the classes because of how negatives were sampled. In the master dataset:", "",
          md_table(size), "",
          "Renamed methods still sit in smaller files. Train every model with and without these two features "
          "and report both.", ""]
    (config.DOCS / "data_quality_report.md").write_text("\n".join(q), encoding="utf-8")
    log.info("Saved docs/data_dictionary.md, docs/data_dictionary.csv, docs/data_quality_report.md and %s", out)


if __name__ == "__main__":
    main()
