r"""
audit_temporal_leakage.py
Reproduces the evidence behind docs/temporal_leakage_audit.md and checks that the
report still agrees with the data and the code.

The report itself is a hand-written analysis (2026-10-31). This script does not
write or rewrite it. It recomputes the measurements the report cites, read-only:
  - which files the raw commit folders contain (changed files only)
  - call_sites split into own-file and other-file calls, by class
  - the rename mechanism: renamed methods whose other-file callers were updated
    to the new name in the After snapshot
  - single-feature AUCs (call_sites, own/other-file calls, rel_call_sites, files per commit)
  - Before-only recomputation of the 8 kept rel_* features (first 400 files)
  - project groups per commit
and checks that the source lines the report cites still contain that code.

Needs the raw data in data/raw/actref/commits/ (about 1-2 minutes).

  python src/data/audit_temporal_leakage.py           write reports/audit/leakage/temporal_leakage_evidence.json
  python src/data/audit_temporal_leakage.py --check   recompute and verify every cited number; write nothing
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.raw_commits import list_commit_dirs  # noqa: E402
from src.features.method_features import code_features, collect, is_dunder, parse, split_name  # noqa: E402

warnings.filterwarnings("ignore")
REPORT = config.DOCS / "temporal_leakage_audit.md"
OUT = config.REPORTS_AUDIT / "leakage" / "temporal_leakage_evidence.json"
EXTRACT = config.ROOT / "src" / "data" / "extract_features.py"
REL_SAMPLE_FILES = 400
KEPT_REL = ["name_len", "name_tokens", "loc", "n_params", "n_calls", "cyclomatic", "doc_len", "name_body_overlap"]

# line in src/data/extract_features.py cited by the report -> code that must be on it
CITED_LINES = {
    61: "files = sorted(",
    67: "before_texts[f] = btext",
    73: "all_before_text =",
    77: "collect(btree), collect(atree)",
    87: "block = []",
    107: "name_features(name)",
    108: "code_features(func, name)",
    117: '"call_sites": len(re.findall(',
    129: "if block:",
    130: "for col in REL_COLS:",
    134: 'rows += [r for r in block if r["renamed"] is not None]',
}


def auc(scores, y):
    r = scores.rank().values
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evidence():
    m = pd.read_csv(config.MASTER, keep_default_na=False)
    m["class"] = m["class"].astype(str)
    kept, _ = list_commit_dirs(config.RAW_COMMITS)
    ev = {}

    # 1. raw folders hold only changed files
    pairs = identical = 0
    files_per_commit = {}
    for d in kept:
        before = sorted((d / "Before").glob("*.py"))
        files_per_commit[d.name] = len(before)
        for p in before:
            a = d / "After" / p.name
            if a.exists():
                pairs += 1
                identical += p.read_bytes() == a.read_bytes()
    ev["before_after_pairs"] = pairs
    ev["pairs_byte_identical"] = identical

    # 2. call_sites = own-file + other-file calls (same search space as extract_features.py)
    own, other = {}, {}
    for commit, g in m.groupby("commit"):
        d = config.RAW_COMMITS / commit
        texts = {}
        for p in sorted((d / "Before").glob("*.py")):
            t, tree = parse(p)
            if tree is not None:
                texts[p.name.replace("#", "/")] = t
        for idx, f, name in zip(g.index, g["file"], g["name"]):
            pat = re.compile(r"\b" + re.escape(name) + r"\s*\(")
            own[idx] = len(pat.findall(texts.get(f, ""))) - 1
            other[idx] = sum(len(pat.findall(t)) for ff, t in texts.items() if ff != f)
    m["own_file_calls"] = pd.Series(own)
    m["other_file_calls"] = pd.Series(other)
    ev["decomposition_matches_call_sites"] = bool((m.own_file_calls + m.other_file_calls == m.call_sites).all())
    y = m["renamed"].values
    pos, neg = m[m.renamed == 1], m[m.renamed == 0]
    ev["rows"] = len(m)
    ev["share_with_other_file_calls"] = {"renamed": float((pos.other_file_calls > 0).mean()),
                                         "not_renamed": float((neg.other_file_calls > 0).mean())}
    ev["mean_own_file_calls"] = {"renamed": float(pos.own_file_calls.mean()), "not_renamed": float(neg.own_file_calls.mean())}
    ev["mean_call_sites"] = {"renamed": float(pos.call_sites.mean()), "not_renamed": float(neg.call_sites.mean())}
    m["commit_files"] = m.commit.map(files_per_commit)
    ev["median_files_in_commit"] = {"renamed": float(pos.commit.map(files_per_commit).median()),
                                    "not_renamed": float(neg.commit.map(files_per_commit).median())}

    # 3. rename mechanism: other-file callers updated to the new name
    labels = pd.read_csv(config.TRAIN_POSITIVES, keep_default_na=False)
    new_name = {(r.commit, r.file, str(r["class"]), r.old_name): r.new_name for _, r in labels.iterrows()}
    with_callers = forced = 0
    for _, r in m[(m.renamed == 1) & (m.other_file_calls > 0)].iterrows():
        with_callers += 1
        d = config.RAW_COMMITS / r.commit
        new = new_name[(r.commit, r.file, r["class"], r["name"])]
        pat_old = re.compile(r"\b" + re.escape(r["name"]) + r"\s*\(")
        pat_new = re.compile(r"\b" + re.escape(new) + r"\s*\(")
        hit = False
        for p in (d / "Before").glob("*.py"):
            if p.name.replace("#", "/") == r.file:
                continue
            bt, _ = parse(p)
            at, _ = parse(d / "After" / p.name)
            if bt and pat_old.search(bt) and at and pat_new.search(at):
                hit = True
        forced += hit
    ev["renamed_with_other_file_callers"] = with_callers
    ev["of_which_callers_updated_to_new_name"] = forced

    # 4. single-feature AUCs
    ev["auc"] = {c: auc(m[c], y) for c in ["call_sites", "own_file_calls", "other_file_calls",
                                           "rel_call_sites", "commit_files"]}

    # 5. Before-only recomputation of the 8 kept rel_* features
    checked = mismatches = 0
    files_done = 0
    for (commit, f), g in list(m.groupby(["commit", "file"]))[:REL_SAMPLE_FILES]:
        files_done += 1
        _, bt = parse(config.RAW_COMMITS / commit / "Before" / f.replace("/", "#"))
        elig = {k: v for k, v in collect(bt).items() if not is_dunder(k[1]) and not k[1].startswith("test")}
        base = {}
        for (c, n), (fn, _, _) in elig.items():
            cf = code_features(fn, n)
            base[(c, n)] = {"name_len": len(n), "name_tokens": len(split_name(n)),
                            **{k: cf[k] for k in KEPT_REL[2:]}}
        for col in KEPT_REL:
            mean = sum(v[col] for v in base.values()) / len(base)
            for _, r in g.iterrows():
                checked += 1
                mismatches += round(base[(r["class"], r["name"])][col] - mean, 3) != r["rel_" + col]
    ev["rel_before_only_recompute"] = {"files": files_done, "values_checked": checked, "mismatches": mismatches}

    # 6. grouping
    ev["max_groups_per_commit"] = int(m.groupby("commit").meta_project_group.nunique().max())
    return ev


def cited_claims(ev):
    """Exact text fragments of docs/temporal_leakage_audit.md, rebuilt from the evidence."""
    s, mo, cs, med, a = (ev["share_with_other_file_calls"], ev["mean_own_file_calls"], ev["mean_call_sites"],
                         ev["median_files_in_commit"], ev["auc"])
    rc = ev["rel_before_only_recompute"]
    return {
        "Before/After file pairs": f"| Before/After file pairs | {ev['before_after_pairs']:,} |",
        "identical pairs": f"| {ev['pairs_byte_identical']} ({ev['pairs_byte_identical'] / ev['before_after_pairs']:.1%}) |",
        "rows decomposed": f"exactly for all {ev['rows']:,} rows",
        "other-file call share + AUC": f"**{s['renamed']:.1%}** | {s['not_renamed']:.1%} | {a['other_file_calls']:.3f} (other-file calls)",
        "own-file calls + AUC": f"| {mo['renamed']:.2f} | {mo['not_renamed']:.2f} | {a['own_file_calls']:.3f} (own-file calls)",
        "mean call_sites + AUC": f"| Mean `call_sites` | {cs['renamed']:.2f} | {cs['not_renamed']:.2f} | {a['call_sites']:.3f} |",
        "files per commit + AUC": f"| Files in the commit folder (median) | {med['renamed']:g} | {med['not_renamed']:g} | {a['commit_files']:.3f} |",
        "renamed with other-file callers": f"- {ev['renamed_with_other_file_callers']} renamed methods have callers in other files",
        "callers updated to new name": f"In **{ev['of_which_callers_updated_to_new_name']} of them**",
        "rel_call_sites AUC": f"Its single-feature AUC ({a['rel_call_sites']:.3f})",
        "rel_* recomputation": f"for {rc['files']} files ({rc['values_checked']:,} values)",
        "rel_* mismatches": f"**{rc['mismatches']} mismatches**",
        "groups per commit": f"max {ev['max_groups_per_commit']} group per commit",
    }


def check_report(ev):
    text = REPORT.read_text(encoding="utf-8")
    code = EXTRACT.read_text(encoding="utf-8").splitlines()
    ok = True
    for name, fragment in cited_claims(ev).items():
        found = fragment in text
        ok &= found
        print(f"{'PASS' if found else 'FAIL'} {name}: {fragment}")
    for line, snippet in CITED_LINES.items():
        found = len(code) >= line and snippet in code[line - 1]
        ok &= found
        print(f"{'PASS' if found else 'FAIL'} extract_features.py line {line} contains {snippet!r}")
    print("Not recomputed here: the 9.5% of eligible methods with an unknown label "
          "(3,263; from extract_features.py extraction statistics).")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the report's numbers and code lines; write nothing")
    args = ap.parse_args(argv)
    if not config.RAW_COMMITS.exists():
        raise SystemExit(f"Raw data not found: {config.RAW_COMMITS}")
    ev = evidence()
    if args.check:
        ok = check_report(ev)
        print("ALL CITED EVIDENCE REPRODUCED" if ok else "REPORT AND EVIDENCE DISAGREE")
        return 0 if ok else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(f"Saved {OUT.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
