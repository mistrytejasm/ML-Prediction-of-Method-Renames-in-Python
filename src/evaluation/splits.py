r"""
splits.py
Stage 3: the reproducible project-level held-out split and development folds.

Approved design (preprocessing guide, Steps 5-6):
  outer  StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on all
         rows; fold 0 = held-out test, the other 4 folds = development
  inner  StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on the
         development rows only (in master order); fold k = validation fold k
  groups meta_project_group (323 projects); never a row-level split

The splitters read only y and groups, so main 40 and comparison 42 get the same
partitions; generate_splits() builds both and checks they are identical.

Acceptance criteria for the test partition (>= 40 positives, >= 10 projects
with a positive) are audit-informed: they were set after the Phase 1
preprocessing audit and after the split's counts were known. They are a
minimum-information check, not prespecified or preregistered criteria. If they
fail, generation stops; the seed is never changed to make them pass.

Saved (data/processed/splits/):
  project_split_seed42.csv   one row per master row, in master order:
                             master_row, commit, file, class, name,
                             meta_project_group, renamed, partition
                             (development | test), inner_fold (0-4; -1 for test)
  project_split_seed42.json  splitter settings, master SHA-256, feature sets,
                             versions, counts per partition and fold, split hash

split hash = SHA-256 of the lines "commit<TAB>file<TAB>class<TAB>name<TAB>
partition<TAB>inner_fold" in master order, joined with "\n"; it changes if any
row moves to another partition or fold, and does not depend on file formatting.

Nothing is fitted or trained here; the master is only read (prepare_dataset()).

Generate once:   python src/evaluation/splits.py
Check saved:     python src/evaluation/splits.py --check
"""
import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import GROUP_COLUMN, load_master, sha256_of  # noqa: E402
from src.data.prepare_dataset import prepare_dataset  # noqa: E402
from src.features.feature_sets import FEATURE_SETS  # noqa: E402

log = logging.getLogger("splits")

SPLITTER = {"name": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": 42}
TEST_FOLD = 0
ACCEPTANCE = {"min_test_positives": 40, "min_test_positive_projects": 10}
SPLIT_CSV = config.SPLITS_DIR / "project_split_seed42.csv"
SPLIT_META = config.SPLITS_DIR / "project_split_seed42.json"
KEY = list(config.KEY)                                   # (commit, file, class, name): unique in the master
COLUMNS = ["master_row"] + KEY + [GROUP_COLUMN, config.TARGET, "partition", "inner_fold"]


class SplitValidationError(ValueError):
    """A split assignment breaks a partition rule or does not match the master."""


def _splitter():
    return StratifiedGroupKFold(n_splits=SPLITTER["n_splits"], shuffle=SPLITTER["shuffle"],
                                random_state=SPLITTER["random_state"])


def assign_partitions(X, y, groups):
    """Return (partition, inner_fold) arrays, one entry per row of X, from the approved splitters."""
    folds = list(_splitter().split(X, y, groups))
    dev_idx, test_idx = folds[TEST_FOLD]
    partition = np.full(len(X), "development", dtype=object)
    partition[test_idx] = "test"
    inner_fold = np.full(len(X), -1, dtype=int)
    Xd, yd, gd = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    for k, (_, val) in enumerate(_splitter().split(Xd, yd, gd)):
        inner_fold[dev_idx[val]] = k
    return partition, inner_fold


def generate_splits():
    """Build the split assignment table for the master; both feature configurations must agree."""
    df = load_master()
    results = {}
    for name in FEATURE_SETS:
        X, y, groups = prepare_dataset(name)
        results[name] = (assign_partitions(X, y, groups), y, groups)
    (part, fold), y, groups = results["main"]
    for name, ((p2, f2), y2, g2) in results.items():
        if not (np.array_equal(part, p2) and np.array_equal(fold, f2) and y.equals(y2) and groups.equals(g2)):
            raise SplitValidationError(f"feature set {name!r} produced different partitions")
    assign = df[KEY].copy()
    assign.insert(0, "master_row", np.arange(len(df)))
    assign[GROUP_COLUMN] = groups.to_numpy()
    assign[config.TARGET] = y.to_numpy()
    assign["partition"] = part
    assign["inner_fold"] = fold
    return assign[COLUMNS].reset_index(drop=True)


def split_hash(assign):
    lines = (assign["commit"].astype(str) + "\t" + assign["file"].astype(str) + "\t" + assign["class"].astype(str)
             + "\t" + assign["name"].astype(str) + "\t" + assign["partition"].astype(str)
             + "\t" + assign["inner_fold"].astype(str))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _counts(part):
    y, g = part[config.TARGET], part[GROUP_COLUMN]
    return {"rows": int(len(part)), "projects": int(g.nunique()), "positives": int(y.sum()),
            "negatives": int((y == 0).sum()), "positive_projects": int(g[y == 1].nunique())}


def summarize(assign):
    dev = assign[assign["partition"] == "development"]
    test = assign[assign["partition"] == "test"]
    folds = []
    for k in range(SPLITTER["n_splits"]):
        val, train = dev[dev["inner_fold"] == k], dev[dev["inner_fold"] != k]
        folds.append({"fold": k, "validation": _counts(val), "training": _counts(train)})
    return {"all": _counts(assign), "test": _counts(test), "development": _counts(dev), "inner_folds": folds}


def validate_assignments(assign, master=None):
    """Raise SplitValidationError unless `assign` is a complete, project-disjoint split of the master."""
    master = load_master() if master is None else master
    if list(assign.columns) != COLUMNS:
        raise SplitValidationError(f"columns {list(assign.columns)}, expected {COLUMNS}")
    if len(assign) != len(master) or not np.array_equal(assign["master_row"].to_numpy(), np.arange(len(master))):
        raise SplitValidationError("master_row must list every master row exactly once, in order")
    for col, src in [(c, c) for c in KEY] + [(GROUP_COLUMN, GROUP_COLUMN), (config.TARGET, config.TARGET)]:
        if not np.array_equal(assign[col].astype(str).to_numpy(), master[src].astype(str).to_numpy()):
            raise SplitValidationError(f"column {col!r} does not match the master row by row")
    if assign.duplicated(KEY).any():
        raise SplitValidationError("duplicate row keys")
    if not set(assign["partition"]) <= {"development", "test"}:
        raise SplitValidationError(f"unknown partitions: {set(assign['partition'])}")

    dev = assign[assign["partition"] == "development"]
    test = assign[assign["partition"] == "test"]
    if (test["inner_fold"] != -1).any():
        raise SplitValidationError("test rows must have inner_fold -1")
    if not set(dev["inner_fold"]) == set(range(SPLITTER["n_splits"])):
        raise SplitValidationError(f"development rows need inner folds 0-{SPLITTER['n_splits'] - 1}, "
                                   f"got {sorted(set(dev['inner_fold']))}")
    overlap = set(dev[GROUP_COLUMN]) & set(test[GROUP_COLUMN])
    if overlap:
        raise SplitValidationError(f"projects in both development and test: {sorted(overlap)[:5]}")
    folds_per_project = dev.groupby(GROUP_COLUMN)["inner_fold"].nunique()
    if (folds_per_project > 1).any():
        raise SplitValidationError(f"projects in more than one inner fold: {list(folds_per_project[folds_per_project > 1].index[:5])}")

    stats = summarize(assign)
    if stats["test"]["positives"] < ACCEPTANCE["min_test_positives"] or \
            stats["test"]["positive_projects"] < ACCEPTANCE["min_test_positive_projects"]:
        raise SplitValidationError(f"test partition fails the acceptance criteria {ACCEPTANCE}: {stats['test']}")
    for f in stats["inner_folds"]:
        if f["validation"]["positives"] == 0 or f["training"]["positives"] == 0:
            raise SplitValidationError(f"inner fold {f['fold']} has a partition without positives")
    return stats


def build_metadata(assign, stats):
    return {
        "name": "project_split_seed42",
        "created": date.today().isoformat(),
        "created_by": "src/evaluation/splits.py",
        "dataset": config.MASTER.relative_to(config.ROOT).as_posix(),
        "dataset_sha256": sha256_of(config.MASTER),
        "feature_sets": {name: len(features) for name, features in FEATURE_SETS.items()},
        "feature_sets_note": "the splitters read only y and groups; every feature set uses these partitions",
        "group_column": GROUP_COLUMN,
        "target": config.TARGET,
        "row_key": KEY,
        "outer_split": {**SPLITTER, "test_fold": TEST_FOLD, "split_on": "all master rows, master order"},
        "inner_split": {**SPLITTER, "split_on": "development rows only, master order"},
        "acceptance_criteria": {**ACCEPTANCE,
                                "status": "audit-informed, set after the Phase 1 preprocessing audit and after the "
                                          "split counts were known; not prespecified or preregistered",
                                "passed": True},
        "assignment_file": SPLIT_CSV.name,
        "assignment_columns": COLUMNS,
        "inner_fold_for_test_rows": -1,
        "split_hash": split_hash(assign),
        "split_hash_definition": "sha256 of 'commit\\tfile\\tclass\\tname\\tpartition\\tinner_fold' lines in "
                                 "master order joined with '\\n'",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "counts": stats,
    }


def save_splits(assign, meta, csv_path=SPLIT_CSV, meta_path=SPLIT_META):
    """Write the split once. An existing split with the same hash is left alone; a different one is an error."""
    csv_path, meta_path = Path(csv_path), Path(meta_path)
    if csv_path.exists() or meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if existing.get("split_hash") == meta["split_hash"] and csv_path.exists():
            log.info("Split already saved with the same hash; nothing written")
            return False
        raise SplitValidationError(f"a different split already exists at {csv_path.parent}; it is never overwritten")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    assign.to_csv(csv_path, index=False, lineterminator="\n")
    meta = {**meta, "assignment_file_sha256": sha256_of(csv_path)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return True


def load_splits(csv_path=SPLIT_CSV, meta_path=SPLIT_META, master=None):
    """Read the saved split and re-validate it against the master; returns (assign, meta)."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if sha256_of(config.MASTER) != meta["dataset_sha256"]:
        raise SplitValidationError("the master dataset's SHA-256 differs from the one the split was made from")
    if sha256_of(csv_path) != meta["assignment_file_sha256"]:
        raise SplitValidationError("the assignment file's SHA-256 differs from the metadata")
    assign = pd.read_csv(csv_path, keep_default_na=False, dtype={"class": str, "commit": str, "file": str, "name": str})
    stats = validate_assignments(assign, master)
    if split_hash(assign) != meta["split_hash"]:
        raise SplitValidationError("split hash differs from the metadata")
    if stats != meta["counts"]:
        raise SplitValidationError("partition counts differ from the metadata")
    return assign, meta


def outer_indices(assign):
    """Master row positions of the development and test partitions."""
    return (assign.index[assign["partition"] == "development"].to_numpy(),
            assign.index[assign["partition"] == "test"].to_numpy())


def inner_folds(assign):
    """(train, validation) positions into the development rows (master order), one pair per inner fold."""
    fold = assign.loc[assign["partition"] == "development", "inner_fold"].to_numpy()
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(SPLITTER["n_splits"])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only load and validate the saved split")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.check:
        _, meta = load_splits()
        log.info("Saved split is valid: hash %s", meta["split_hash"])
        return 0
    assign = generate_splits()
    stats = validate_assignments(assign)
    meta = build_metadata(assign, stats)
    written = save_splits(assign, meta)
    load_splits()
    log.info("%s split %s | test %s | development %s", "Saved" if written else "Kept", meta["split_hash"],
             stats["test"], stats["development"])
    for f in stats["inner_folds"]:
        log.info("inner fold %d: validation %s | training positives %d", f["fold"], f["validation"],
                 f["training"]["positives"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
