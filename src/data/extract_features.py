r"""
extract_features.py
Builds the v2 feature table: one row per method in a Before file, label
renamed = 1 (in train_positives.csv) or 0 (same class + name still in the After
file). Methods that disappeared without being a labelled rename are skipped,
because their label is unknown.

This is a refactor of archive/legacy_scripts/Build_Datasetv2.py with the same
logic and the same feature code (src/features/method_features.py). Changes:
  - paths come from src/config.py or the command line
  - files are processed in sorted order, so the row order is reproducible
    (the original iterated over a Python set)
  - it never overwrites the validated dataset unless --force is given
  - --limit N processes only the first N commits (used by the tests)
  - byte-identical duplicate commit folders are skipped (one exists:
    dd7f9fa8..._<garbled suffix>, which added 8 duplicate rows to v2)
  - two leaks of After information into features are fixed:
      call_sites  now searches every parseable Before file of the commit
                  (v2 skipped Before files that had no After counterpart)
      rel_*       now compare with all eligible Before methods of the file
                  (v2 averaged only over methods whose label was known, i.e.
                  it excluded methods that were deleted/moved in the After code)
  - --legacy-v2 restores all original behaviour and reproduces v2 exactly

Every feature is now computed from the Before AST/text only. The After tree is
used for exactly one thing: deciding the label (1 / 0 / unknown) of a method.

Run from the project root:
    python src/data/extract_features.py --out reports/audit/regenerated/method_features_v2_regenerated.csv
"""
import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.raw_commits import list_commit_dirs  # noqa: E402
from src.features.method_features import (REL_COLS, code_features, collect, is_dunder,  # noqa: E402
                                          name_features, parse, split_name, style_of)

log = logging.getLogger("extract_features")


def method_rows(commit_dir, commit, pos_keys, found, stats=None, legacy_v2=False):
    """Feature rows for every labelled Before method of one commit.

    legacy_v2=False (default, Before-only):
      - call_sites searches every parseable Before file of the commit
      - rel_* compares a method with all eligible methods of its Before file
        (eligible = not dunder, not test), whether or not their label is known
    legacy_v2=True reproduces the original v2 table, where both depended on the
    After code: call_sites ignored Before files without an After pair, and rel_*
    averaged only over methods whose label was known (After-dependent).
    """
    files = sorted(p.name.replace("#", "/") for p in (commit_dir / "Before").glob("*.py"))
    before_texts, parsed = {}, {}
    for f in files:
        btext, btree = parse(commit_dir / "Before" / f.replace("/", "#"))
        atext, atree = parse(commit_dir / "After" / f.replace("/", "#"))
        if btree is not None and (atree is not None or not legacy_v2):
            before_texts[f] = btext
        if btree is None or atree is None:
            if stats is not None:
                stats["files_skipped_missing_or_unparseable"] += 1
            continue
        parsed[f] = (btext, btree, atree)
    all_before_text = "\n".join(before_texts.values())

    rows = []
    for f, (btext, btree, atree) in parsed.items():
        before, after = collect(btree), collect(atree)
        file_loc = btext.count("\n") + 1
        file_styles = [style_of(k[1]) for k in before]
        snake_share = (sum(s in ("snake", "lower") for s in file_styles) / len(file_styles)) if file_styles else 1.0

        sibling_tokens = {}
        for (c2, n2) in before:
            for t in set(split_name(n2)):
                sibling_tokens[t] = sibling_tokens.get(t, 0) + 1

        block = []                              # rows used for the file averages (rel_*)
        for (cls, name), (func, cls_n, depth) in before.items():
            if is_dunder(name) or name.startswith("test"):
                if stats is not None:
                    stats["methods_excluded_dunder_or_test"] += 1
                continue
            key = (commit, f, cls, name)
            if key in pos_keys:
                label = 1
                found.add(key)
            elif (cls, name) in after:
                label = 0
            else:
                label = None                    # removed, moved or renamed below threshold
                if stats is not None:
                    stats["methods_excluded_unknown_label"] += 1
                if legacy_v2:
                    continue

            feat = {"commit": commit, "file": f, "class": cls, "name": name, "renamed": label}
            feat.update(name_features(name))
            feat.update(code_features(func, name))
            feat.update({
                "in_class": int(bool(cls)),
                "class_depth": depth,
                "class_n_methods": cls_n,
                "file_n_methods": len(before),
                "file_loc": file_loc,
                "is_test_file": int("test" in f.lower()),
                "style_mismatch": int(style_of(name) == "camel" and snake_share > 0.5),
                "call_sites": len(re.findall(r"\b" + re.escape(name) + r"\s*\(", all_before_text)) - 1,
            })
            toks = set(split_name(name))
            shared = [t for t in toks if sibling_tokens.get(t, 0) > 1]
            feat["name_token_sharing"] = round(len(shared) / len(toks), 3) if toks else 0.0
            first = split_name(name)[:1]
            feat["first_token_shared"] = int(bool(first) and sibling_tokens.get(first[0], 0) > 1)
            st = style_of(name)
            feat["style_share_in_file"] = round(
                sum(1 for s_ in file_styles if s_ == st) / len(file_styles), 3) if file_styles else 1.0
            block.append(feat)

        if block:
            for col in REL_COLS:
                mean = sum(r[col] for r in block) / len(block)
                for r in block:
                    r["rel_" + col] = round(r[col] - mean, 3)
        rows += [r for r in block if r["renamed"] is not None]
    return rows


def build(commits_dir, positives, limit=None, legacy_v2=False):
    """Return (feature DataFrame, set of positive keys not found in the Before code).

    Duplicate commit folders (byte-identical copies, see src/data/raw_commits.py)
    are skipped and features are strictly Before-only, unless legacy_v2=True,
    which reproduces the original v2 table exactly (duplicates included).
    """
    pos = positives.copy()
    pos["class"] = pos["class"].fillna("")
    pos_keys = {(r.commit, r.file, r["class"], r.old_name) for _, r in pos.iterrows()}
    if legacy_v2:
        commits, skipped = sorted(d.name for d in commits_dir.iterdir() if d.is_dir()), {}
    else:
        kept, skipped = list_commit_dirs(commits_dir)
        commits = [d.name for d in kept]
        for name, why in skipped.items():
            log.warning("Skipping commit folder %s: %s", name, why)
    if limit:
        commits = commits[:limit]
    rows, found = [], set()
    stats = Counter()
    for i, commit in enumerate(commits, 1):
        rows += method_rows(commits_dir / commit, commit, pos_keys, found, stats, legacy_v2)
        if i % 200 == 0:
            log.info("  %d/%d commits", i, len(commits))
    df = pd.DataFrame(rows)
    if limit:
        pos_keys = {k for k in pos_keys if k[0] in set(commits)}
    build.last_stats = dict(stats, commits_scanned=len(commits),
                            commit_folders_skipped_as_duplicates=skipped, rows=len(df),
                            commits_with_rows=int(df["commit"].nunique()) if len(df) else 0)
    return df, pos_keys - found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits", type=Path, default=config.RAW_COMMITS)
    ap.add_argument("--positives", type=Path, default=config.TRAIN_POSITIVES)
    ap.add_argument("--out", type=Path, default=config.REGENERATED_FEATURES)
    ap.add_argument("--limit", type=int, default=None, help="only the first N commits")
    ap.add_argument("--force", action="store_true", help="allow overwriting the validated dataset")
    ap.add_argument("--legacy-v2", action="store_true",
                    help="reproduce the original v2 exactly (duplicate commit included, "
                         "After-dependent call_sites and rel_* features)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.commits.exists():
        raise SystemExit(f"Raw commits not found: {args.commits}. See README.md for the download.")
    if args.out.resolve() == config.V2_FEATURES.resolve() and not args.force:
        raise SystemExit("Refusing to overwrite the validated v2 dataset without --force.")

    df, missing = build(args.commits, pd.read_csv(args.positives), args.limit, args.legacy_v2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Rows %d | renamed %d | not renamed %d | commits %d", len(df), int(df["renamed"].sum()),
             int((df["renamed"] == 0).sum()), df["commit"].nunique())
    log.info("Positives not found in the Before code: %d", len(missing))
    stats = dict(build.last_stats, positives_not_found=len(missing))
    log.info("Extraction stats: %s", stats)
    args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Saved %s", args.out)
    return df, missing


if __name__ == "__main__":
    main()
