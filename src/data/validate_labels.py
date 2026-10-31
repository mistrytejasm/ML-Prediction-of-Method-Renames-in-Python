r"""
validate_labels.py
Audits the whole label chain and writes a trace of every rename candidate.

  detected_rename_candidates.csv  (430, detector output)
    -> test methods dropped
    -> auto_accepted_rename_candidates.csv  (similarity >= 0.9 and same args)
    -> uncertain_candidates_review_*.xlsx   (the rest, reviewed Y/N)
    -> pool = auto-accepted + review Y, cap of 5 per commit (seed 42)
    -> rename_positive_labels.csv
    -> method_features_v2_all_commits.csv (renamed = 1 rows)

With --raw it also re-reads the raw Before/After code to check every positive
and every negative in v2 (takes about 2 minutes).

Outputs (reports/audit/labels/):
  label_validation_results.json     every check with PASS / WARN / FAIL
  candidate_label_trace.csv         all 430 candidates and what happened to each
  positive_labels_with_metadata.csv 243 positives + rename type + provenance
  excluded_by_commit_cap.csv        the candidates dropped by the 5-per-commit cap
  auto_accept_rule_mismatches.csv   rule-confident pairs missing from the file
  negative_body_change_summary.csv  (with --raw) negatives whose body changed

Exit code is 1 if any check FAILs.
Run: python src/data/validate_labels.py --raw
"""
import argparse
import ast
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.label_utils import label_provenance, rename_type  # noqa: E402
from src.features.method_features import collect, parse  # noqa: E402

log = logging.getLogger("validate_labels")
CAND_KEY = ["commit", "file", "class", "old_name"]


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail, warn=False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.items.append({"check": name, "status": status, "detail": detail})
        getattr(log, "info" if ok else "warning")("%-4s %s: %s", status, name, detail)

    @property
    def failed(self):
        return [c for c in self.items if c["status"] == "FAIL"]


def keyset(df, cols=CAND_KEY):
    return set(map(tuple, df[cols].astype(str).values))


def read_csv(path):
    df = pd.read_csv(path, keep_default_na=False)
    if "class" in df:
        df["class"] = df["class"].astype(str)
    return df


def read_review(path):
    df = pd.read_excel(path, dtype=str).fillna("")
    for c in ("similarity",):
        df[c] = df[c].astype(float)
    return df


# ---------------------------------------------------------------- raw checks
def method_defs(tree):
    """Count every def per (class_path, name), including duplicates that collect() collapses."""
    counts = Counter()

    def visit(node, class_path):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_path}.{child.name}" if class_path else child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts[(class_path, child.name)] += 1

    visit(tree, "")
    return counts


def raw_checks(v2, positives, checks):
    pos_rows = {tuple(r): None for r in positives[CAND_KEY + ["new_name"]].astype(str).values}
    by_file = v2.groupby(["commit", "file"])
    status = Counter()
    pos_problems, neg_problems, dup_rows = [], [], 0
    body_changed = []
    for (commit, f), g in by_file:
        cdir = config.RAW_COMMITS / commit
        _, bt = parse(cdir / "Before" / f.replace("/", "#"))
        _, at = parse(cdir / "After" / f.replace("/", "#"))
        if bt is None or at is None:
            neg_problems.append((commit, f, "", "", "file not parseable"))
            continue
        before, after = collect(bt), collect(at)
        defs = method_defs(bt)
        for r in g.itertuples(index=False):
            k = (r[2], r[3])                        # class, name
            if defs.get(k, 0) > 1:
                dup_rows += 1
            if k not in before:
                (pos_problems if r.renamed else neg_problems).append((commit, f, k[0], k[1], "not in Before"))
                continue
            if r.renamed == 1:
                if k in after:
                    pos_problems.append((commit, f, k[0], k[1], "old name still in After"))
                new = [p[4] for p in pos_rows if p[:4] == (commit, f, k[0], k[1])]
                if not new or (k[0], new[0]) not in after:
                    pos_problems.append((commit, f, k[0], k[1], "new name not in After"))
                status["positive_ok"] += 1
            else:
                if k not in after:
                    neg_problems.append((commit, f, k[0], k[1], "name missing in After"))
                    continue
                same = ast.dump(before[k][0]) == ast.dump(after[k][0])
                status["negative_unchanged" if same else "negative_body_or_signature_changed"] += 1
                if not same:
                    body_changed.append((commit, f, k[0], k[1]))
    checks.add("raw: positives present in Before, old name gone and new name present in After",
               not pos_problems, f"{status['positive_ok']} positives checked, {len(pos_problems)} problems")
    checks.add("raw: negatives keep the same class + name in After",
               not neg_problems, f"{status['negative_unchanged'] + status['negative_body_or_signature_changed']}"
                                 f" negatives checked, {len(neg_problems)} problems")
    checks.add("raw: negatives whose code changed while keeping their name",
               True, f"{status['negative_unchanged']} unchanged, "
                     f"{status['negative_body_or_signature_changed']} changed body/signature "
                     "(still valid negatives: the label means 'name kept', not 'code unchanged')")
    checks.add("raw: rows whose class+name is defined more than once in the Before file",
               dup_rows == 0, f"{dup_rows} rows (e.g. @property getter + @x.setter share a name; "
                              "the feature code keeps only the last definition)", warn=True)
    return status, pos_problems + neg_problems, body_changed, dup_rows


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="also re-check labels against the raw code")
    ap.add_argument("--out-dir", type=Path, default=config.REPORTS_AUDIT / "labels")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ck = Checks()

    cand = read_csv(config.AUTO_RENAMES)
    conf = read_csv(config.CONFIDENT_RENAMES)
    blank = read_review(config.REVIEW_SHEET)
    done = read_review(config.REVIEW_SHEET_DONE)
    pos = read_csv(config.TRAIN_POSITIVES)
    v2 = read_csv(config.V2_FEATURES)
    for df in (cand, conf):
        df["same_args"] = df["same_args"].astype(str).str.lower().isin(["true", "1"])

    # 1. candidates
    ck.add("candidates: row count", len(cand) == 430, f"{len(cand)} rows (expected 430)")
    ck.add("candidates: one row per (commit, file, class, old_name)",
           not cand.duplicated(CAND_KEY).any(), f"{cand.duplicated(CAND_KEY).sum()} duplicates")
    ck.add("candidates: one row per (commit, file, class, new_name)",
           not cand.duplicated(["commit", "file", "class", "new_name"]).any(),
           f"{cand.duplicated(['commit', 'file', 'class', 'new_name']).sum()} duplicates")
    ck.add("candidates: similarity within [0.70, 1.0]",
           cand["similarity"].between(config.SIM_THRESHOLD, 1.0).all(),
           f"min {cand['similarity'].min()}, max {cand['similarity'].max()}")
    dunder = cand["old_name"].str.match(r"^__.*__$") | cand["new_name"].str.match(r"^__.*__$")
    ck.add("candidates: no dunder methods", not dunder.any(), f"{int(dunder.sum())} dunder rows")
    is_test = cand["old_name"].str.startswith("test")
    ck.add("candidates: test methods (dropped in the next step)", True, f"{int(is_test.sum())} test methods")

    # 2. auto-accept rule vs file
    non_test = cand[~is_test]
    rule = non_test[(non_test["similarity"] >= config.CONFIDENT_SIM) & non_test["same_args"]]
    missing = keyset(rule) - keyset(conf)
    extra = keyset(conf) - keyset(rule)
    ck.add("auto-accepted: file matches the rule (similarity >= 0.9 and same args)",
           not extra and not missing,
           f"rule gives {len(rule)}, file has {len(conf)}; {len(missing)} rule pairs missing from the file "
           f"(removed by hand, not by code), {len(extra)} extra", warn=not extra)
    rule[rule[CAND_KEY].astype(str).apply(tuple, axis=1).isin(missing)].to_csv(
        out / "auto_accept_rule_mismatches.csv", index=False)

    # 3. review sheets
    review_rule = keyset(non_test) - keyset(rule)
    ck.add("review: blank sheet holds exactly the non-confident, non-test candidates",
           keyset(blank) == review_rule, f"{len(blank)} rows, rule gives {len(review_rule)}")
    ck.add("review: completed sheet has the same rows as the blank sheet",
           keyset(blank) == keyset(done), f"{len(done)} rows")
    dec = done["is_rename"].str.strip().str.upper()
    ck.add("review: every decision is Y or N", dec.isin(["Y", "N"]).all(),
           f"Y {int((dec == 'Y').sum())}, N {int((dec == 'N').sum())}, other {int((~dec.isin(['Y', 'N'])).sum())}")
    human = done["confirmed_by_human"].str.strip().ne("")
    note_src = done["note"].str.split(":").str[0].value_counts().to_dict()
    ck.add("review: decisions confirmed by a human", bool(human.all()),
           f"{int(human.sum())} of {len(done)} rows have confirmed_by_human filled in; note sources: {note_src}",
           warn=True)

    # 4. pool and cap
    yes = done[dec == "Y"].copy()
    yes["label_source"] = "hand_checked"
    c2 = conf.copy()
    c2["label_source"] = "auto_confident"
    cols = ["commit", "file", "class", "old_name", "new_name", "similarity", "label_source"]
    pool = pd.concat([c2[cols], yes[cols]], ignore_index=True).drop_duplicates(subset=CAND_KEY)
    ck.add("cap: candidate pool before the cap", len(pool) == 301, f"{len(pool)} (expected 301)")
    capped = (pool.sample(frac=1, random_state=config.CAP_SEED).groupby("commit", group_keys=False)
              .head(config.MAX_PER_COMMIT))
    ck.add("cap: re-running the cap (seed 42) reproduces rename_positive_labels.csv",
           keyset(capped) == keyset(pos), f"{len(keyset(capped) ^ keyset(pos))} differing keys")
    per_commit = pos.groupby("commit").size()
    ck.add("cap: at most 5 positives per commit", per_commit.max() <= config.MAX_PER_COMMIT,
           f"max {per_commit.max()}, commits at the cap {int((per_commit == config.MAX_PER_COMMIT).sum())}")
    excluded = pool[~pool[CAND_KEY].astype(str).apply(tuple, axis=1).isin(keyset(pos))].copy()
    pool_n = pool.groupby("commit").size()
    excluded["commit_pool_size"] = excluded["commit"].map(pool_n)
    ck.add("cap: every excluded candidate comes from a commit with more than 5 candidates",
           (excluded["commit_pool_size"] > config.MAX_PER_COMMIT).all(),
           f"{len(excluded)} excluded from {excluded['commit'].nunique()} commits")
    excluded["rename_type"] = [rename_type(o, n) for o, n in zip(excluded.old_name, excluded.new_name)]
    excluded.sort_values(["commit", "file", "old_name"]).to_csv(out / "excluded_by_commit_cap.csv", index=False)

    # 5. positives
    ck.add("positives: 243 rows, all renamed = 1",
           len(pos) == 243 and (pos["renamed"] == 1).all(), f"{len(pos)} rows, values {sorted(pos['renamed'].unique())}")
    ck.add("positives: unique key", not pos.duplicated(CAND_KEY).any(), f"{pos.duplicated(CAND_KEY).sum()} duplicates")
    ck.add("positives: label_source values", set(pos["label_source"]) <= {"auto_confident", "hand_checked"},
           str(pos["label_source"].value_counts().to_dict()))
    human_keys = keyset(done[human])
    pos["rename_type"] = [rename_type(o, n) for o, n in zip(pos.old_name, pos.new_name)]
    pos["label_provenance"] = [
        label_provenance(s, "yes" if k in human_keys else "")
        for s, k in zip(pos["label_source"], pos[CAND_KEY].astype(str).apply(tuple, axis=1))]
    ck.add("positives: 'hand_checked' rows are really human-confirmed",
           not (pos["label_provenance"] == "ai_reviewed_unconfirmed").any(),
           f"{int((pos['label_provenance'] == 'ai_reviewed_unconfirmed').sum())} of "
           f"{int((pos['label_source'] == 'hand_checked').sum())} hand_checked rows have no human confirmation "
           "-> recommended provenance value: ai_reviewed_unconfirmed", warn=True)
    pos.to_csv(out / "positive_labels_with_metadata.csv", index=False)

    # 6. v2 reconciliation
    v2_pos = v2[v2["renamed"] == 1]
    v2_keys = Counter(map(tuple, v2[config.KEY].astype(str).values))
    counts = [v2_keys.get(k, 0) for k in keyset(pos)]
    ck.add("v2: every positive appears exactly once", all(c == 1 for c in counts),
           f"once {counts.count(1)}, missing {counts.count(0)}, more than once {sum(c > 1 for c in counts)}")
    ck.add("v2: renamed = 1 rows are exactly the positives",
           keyset(v2_pos, config.KEY) == keyset(pos), f"{len(v2_pos)} renamed rows")
    ck.add("v2: unique key (commit, file, class, name)", not v2.duplicated(config.KEY).any(),
           f"{v2.duplicated(config.KEY).sum()} duplicates")
    not_positive = keyset(cand) - keyset(pos)
    leaked = not_positive & set(v2_keys)
    ck.add("v2: no rejected, capped or test candidate is labelled 0",
           not leaked, f"{len(leaked)} of {len(not_positive)} non-positive candidates appear in v2")

    # 7. trace of every candidate
    trace = cand[CAND_KEY + ["new_name", "similarity", "same_args"]].copy()
    k = trace[CAND_KEY].astype(str).apply(tuple, axis=1)
    yes_k, no_k, exc_k = keyset(yes), keyset(done[dec == "N"]), keyset(excluded)
    def outcome(key, test):
        if test:
            return "dropped_test_method"
        if key in missing:
            return "dropped_from_auto_accepted_by_hand"
        if key in keyset(pos):
            return "final_positive"
        if key in exc_k:
            return "dropped_by_commit_cap"
        if key in no_k:
            return "rejected_in_review"
        return "unexplained"
    trace["stage"] = ["auto_accepted" if kk in keyset(rule) else ("test" if t else "review") for kk, t in zip(k, is_test)]
    trace["review_decision"] = [("Y" if kk in yes_k else "N" if kk in no_k else "") for kk in k]
    trace["outcome"] = [outcome(kk, t) for kk, t in zip(k, is_test)]
    trace["rename_type"] = [rename_type(o, n) for o, n in zip(trace.old_name, trace.new_name)]
    trace.to_csv(out / "candidate_label_trace.csv", index=False)
    ck.add("trace: every candidate's outcome is explained", (trace["outcome"] != "unexplained").all(),
           str(trace["outcome"].value_counts().to_dict()))

    summary = {
        "candidates": len(cand), "test_methods": int(is_test.sum()), "non_test": len(non_test),
        "auto_accept_rule": len(rule), "auto_accepted_file": len(conf), "removed_by_hand": len(missing),
        "review_rows": len(done), "review_Y": int((dec == "Y").sum()), "review_N": int((dec == "N").sum()),
        "review_human_confirmed": int(human.sum()), "pool_before_cap": len(pool),
        "excluded_by_cap": len(excluded), "positives": len(pos),
        "positives_by_label_source": pos["label_source"].value_counts().to_dict(),
        "positives_by_provenance": pos["label_provenance"].value_counts().to_dict(),
        "positives_by_rename_type": pos["rename_type"].value_counts().to_dict(),
        "rename_type_by_provenance": pd.crosstab(pos["rename_type"], pos["label_provenance"]).to_dict(),
        "candidate_outcomes": trace["outcome"].value_counts().to_dict(),
        "candidate_rename_types": trace["rename_type"].value_counts().to_dict(),
    }

    if args.raw:
        status, problems, changed, dup_rows = raw_checks(v2, pos, ck)
        summary["raw"] = dict(status, rows_with_duplicate_definitions=dup_rows, problems=len(problems))
        pd.DataFrame(changed, columns=config.KEY).to_csv(out / "negative_body_change_summary.csv", index=False)

    (out / "label_validation_results.json").write_text(
        json.dumps({"summary": summary, "checks": ck.items}, indent=2, default=str), encoding="utf-8")
    log.info("Saved %s", out)
    if ck.failed:
        log.error("%d check(s) failed", len(ck.failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
