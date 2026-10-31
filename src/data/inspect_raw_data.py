r"""
inspect_raw_data.py
Inventory of the raw ActRef download (data/raw/actref/commits). Read-only.
Replaces archive/legacy_scripts/step1_explore_raw_actref.py and adds:
  - Before/After pairing per file, unparseable files, files only on one side
  - commits that yield no usable Before/After .py pair
  - a SHA-256 fingerprint of the whole raw tree, to detect any change later

Output: reports/audit/raw/raw_data_inventory.json
Run:    python src/data/inspect_raw_data.py
"""
import argparse
import ast
import hashlib
import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.raw_commits import HASH, list_commit_dirs  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)
log = logging.getLogger("inspect_raw_data")


def parses(path):
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def tree_fingerprint(root):
    """SHA-256 over (relative path, file hash) of every file, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def inventory(commits_dir, fingerprint=True):
    commits = sorted(d for d in commits_dir.iterdir() if d.is_dir())
    all_files = [f for f in commits_dir.rglob("*") if f.is_file()]
    ext = Counter(f.suffix.lower() or f.name for f in all_files)
    other = sorted(f.relative_to(commits_dir).as_posix() for f in all_files
                   if f.suffix != ".py" and f.name != ".DS_Store")

    stats = Counter()
    no_pair, unparseable, before_only, after_only = [], [], [], []
    for d in commits:
        b, a = d / "Before", d / "After"
        stats["missing_before_folder"] += not b.exists()
        stats["missing_after_folder"] += not a.exists()
        bn = {p.name for p in b.glob("*.py")} if b.exists() else set()
        an = {p.name for p in a.glob("*.py")} if a.exists() else set()
        stats["before_py"] += len(bn)
        stats["after_py"] += len(an)
        before_only += [f"{d.name}/{n}" for n in sorted(bn - an)]
        after_only += [f"{d.name}/{n}" for n in sorted(an - bn)]
        usable = 0
        for n in sorted(bn & an):
            ok_b, ok_a = parses(b / n), parses(a / n)
            if not ok_b:
                unparseable.append(f"{d.name}/Before/{n}")
            if not ok_a:
                unparseable.append(f"{d.name}/After/{n}")
            usable += ok_b and ok_a
        if not usable:
            no_pair.append(d.name)

    _, duplicates = list_commit_dirs(commits_dir)
    bad_names = [d.name for d in commits if not HASH.match(d.name)]
    return {
        "root": commits_dir.relative_to(config.ROOT).as_posix(),
        "commit_folders": len(commits),
        "unique_commits": len(commits) - len(duplicates),
        "folders_not_named_as_a_commit_hash": bad_names,
        "duplicate_commit_folders (skipped by the pipeline)": duplicates,
        "files_total": len(all_files),
        "files_by_extension": dict(ext.most_common()),
        "py_files_in_before": stats["before_py"],
        "py_files_in_after": stats["after_py"],
        "commits_missing_before_folder": stats["missing_before_folder"],
        "commits_missing_after_folder": stats["missing_after_folder"],
        "files_only_in_before (deleted/moved)": len(before_only),
        "files_only_in_after (added/moved)": len(after_only),
        "unparseable_py_files": len(unparseable),
        "commits_without_usable_pair": len(no_pair),
        "non_python_files (excluding .DS_Store)": other,
        "unparseable_examples": unparseable[:20],
        "commits_without_usable_pair_examples": no_pair[:20],
        "tree_sha256": tree_fingerprint(commits_dir) if fingerprint else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits", type=Path, default=config.RAW_COMMITS)
    ap.add_argument("--out", type=Path, default=config.REPORTS_AUDIT / "raw" / "raw_data_inventory.json")
    ap.add_argument("--no-fingerprint", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.commits.exists():
        raise SystemExit(f"Raw commits not found: {args.commits}. See README.md for the download.")
    inv = inventory(args.commits, not args.no_fingerprint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    for k, v in inv.items():
        if not isinstance(v, list):
            log.info("%-40s %s", k, v)
    log.info("Saved %s", args.out)
    return inv


if __name__ == "__main__":
    main()
