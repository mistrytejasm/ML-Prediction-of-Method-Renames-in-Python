r"""
generate_rename_labels.py
Detects Rename Method candidates in the raw ActRef Before/After code.

Rule (unchanged from archive/legacy_scripts/step2_detect_rename_candidates.py):
in the same file and the same class, a method name disappears and a new name
appears whose body (without docstring) is at least SIM_THRESHOLD similar.
Pairs are matched one-to-one, best similarity first. Dunder methods are ignored.

This only regenerates the candidate list (detected_rename_candidates.csv).
Choosing confident pairs, the review and the 5-per-commit cap are checked by
validate_labels.py; the reviewed labels themselves are never regenerated.

Run from the project root:
    python src/data/generate_rename_labels.py
"""
import argparse
import ast
import difflib
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.raw_commits import list_commit_dirs  # noqa: E402

log = logging.getLogger("generate_rename_labels")


def body_text(func):
    """Body source without the docstring, for comparing two methods."""
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    try:
        return "\n".join(ast.unparse(s) for s in body)
    except Exception:
        return ""


def collect_methods(tree):
    """{(class_path, name): info} for module functions and class methods."""
    out = {}

    def visit(node, class_path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_path}.{child.name}" if class_path else child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in child.args.args if a.arg not in ("self", "cls")]
                out[(class_path, child.name)] = {
                    "args": args,
                    "body": body_text(child),
                    "lineno": child.lineno,
                    "loc": (child.end_lineno or child.lineno) - child.lineno + 1,
                }

    visit(tree, "")
    return out


def parse(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, ValueError):
        return None


def is_dunder(name):
    return name.startswith("__") and name.endswith("__")


def detect(commits_dir, threshold=config.SIM_THRESHOLD):
    """Return (candidates DataFrame, stats dict)."""
    rows, files_seen, parse_errors = [], 0, 0
    commit_dirs, skipped = list_commit_dirs(commits_dir)
    for name, why in skipped.items():
        log.warning("Skipping commit folder %s: %s", name, why)
    for commit_dir in commit_dirs:
        before_dir, after_dir = commit_dir / "Before", commit_dir / "After"
        for bfile in sorted(before_dir.glob("*.py")):
            afile = after_dir / bfile.name
            if not afile.exists():
                continue
            files_seen += 1
            bt, at = parse(bfile), parse(afile)
            if bt is None or at is None:
                parse_errors += 1
                continue
            before, after = collect_methods(bt), collect_methods(at)
            removed = [k for k in before if k not in after and not is_dunder(k[1])]
            added = [k for k in after if k not in before and not is_dunder(k[1])]

            pairs = []
            for r in removed:
                for a in added:
                    if r[0] != a[0]:
                        continue
                    sim = difflib.SequenceMatcher(None, before[r]["body"], after[a]["body"]).ratio()
                    if sim >= threshold:
                        pairs.append((sim, r, a))

            used_r, used_a = set(), set()
            for sim, r, a in sorted(pairs, reverse=True):
                if r in used_r or a in used_a:
                    continue
                used_r.add(r)
                used_a.add(a)
                rows.append({
                    "commit": commit_dir.name,
                    "file": bfile.name.replace("#", "/"),
                    "class": r[0],
                    "old_name": r[1],
                    "new_name": a[1],
                    "similarity": round(sim, 3),
                    "same_args": before[r]["args"] == after[a]["args"],
                    "old_lineno": before[r]["lineno"],
                    "old_loc": before[r]["loc"],
                })
    return pd.DataFrame(rows), {"files_compared": files_seen, "files_with_parse_errors": parse_errors,
                                "commit_folders_skipped_as_duplicates": skipped}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commits", type=Path, default=config.RAW_COMMITS)
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_AUDIT / "regenerated" / "detected_rename_candidates_regenerated.csv")
    ap.add_argument("--force", action="store_true", help="allow overwriting the original candidate file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.commits.exists():
        raise SystemExit(f"Raw commits not found: {args.commits}. See README.md for the download.")
    if args.out.resolve() == config.AUTO_RENAMES.resolve() and not args.force:
        raise SystemExit("Refusing to overwrite the original candidate file without --force.")

    df, stats = detect(args.commits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("Files compared %d | parse errors %d | candidates %d | commits %d",
             stats["files_compared"], stats["files_with_parse_errors"], len(df), df["commit"].nunique())
    log.info("Saved %s", args.out)
    return df, stats


if __name__ == "__main__":
    main()
