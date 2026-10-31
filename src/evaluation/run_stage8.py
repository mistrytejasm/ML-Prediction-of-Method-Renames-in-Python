r"""
run_stage8.py
Stage 8 command line: validate the fixed run matrix and run development
cross-validation. Nothing runs on import; only main() executes.

  python src/evaluation/run_stage8.py --dry-run          check config, data, folds; fit nothing
  python src/evaluation/run_stage8.py --only <run_id>    run one configured experiment
  python src/evaluation/run_stage8.py                    run the full 14-run matrix
  python src/evaluation/run_stage8.py --seed-spread      the optional 20-seed random baseline

The config (configs/runs/stage8_development_cv.json) is validated strictly: the run
ids must be exactly the approved matrix, every model_params block must equal the
approved settings (no tuning through the config), and the dataset and split hashes
must match. Each run is written once to results/runs/<YYYY-MM-DD>_stage8_<run_id>/
(see run_store.py); a failed run is recorded as <name>__FAILED. After a complete
full-matrix run, the baselines' main / comparison equivalence is checked and
results/runs/<YYYY-MM-DD>_stage8_summary/ is written. Only development rows are ever
loaded; this script has no way to read held-out test rows.
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation import run_store as rs  # noqa: E402
from src.evaluation.cross_validate import (EXPLORATORY, OPTIMISTIC_NOTE, RunFailedError, RunSpec,  # noqa: E402
                                           check_baseline_equivalence, load_development_data,
                                           run_development_cv, select_stage9_candidate)
from src.evaluation.splits import SPLIT_META  # noqa: E402
from src.features.feature_sets import FEATURE_SETS  # noqa: E402
from src.models.pipelines import MODELS, TRANSFORMS, resolve_model_params  # noqa: E402

DEFAULT_CONFIG = config.ROOT / "configs" / "runs" / "stage8_development_cv.json"
PREFIX = "stage8"
APPROVED_MATRIX = [
    "dummy_prior__main", "dummy_prior__comparison",
    "stratified_random__main", "stratified_random__comparison",
    "naming_heuristic__main", "naming_heuristic__comparison",
    "logistic_regression__main__log1p_asinh", "logistic_regression__comparison__log1p_asinh",
    "logistic_regression__main__yeo_johnson", "logistic_regression__comparison__yeo_johnson",
    "random_forest__main", "random_forest__comparison",
    "lightgbm__main", "lightgbm__comparison",
]
CONFIG_KEYS = {"name", "stage", "description", "dataset_sha256", "split_hash", "primary_metric",
               "primary_feature_set", "cv_result", "threshold_rule", "warning_policy", "stage9_selection_rule",
               "runs", "optional_experiments"}
RUN_KEYS = {"run_id", "model", "feature_set", "transform", "model_params", "role", "research_status"}
SPREAD_KEYS = {"purpose", "run_by_default", "model", "feature_set", "transform", "seeds", "role", "research_status"}
STATUSES = {"reference", "confirmatory", EXPLORATORY}


class ConfigError(ValueError):
    """The Stage 8 run configuration breaks the approved design."""


def _run_id(model, feature_set, transform):
    return f"{model}__{feature_set}" + (f"__{transform}" if model == "logistic_regression" else "")


def _validate_run(r):
    if set(r) != RUN_KEYS:
        raise ConfigError(f"run keys {sorted(r)} differ from {sorted(RUN_KEYS)}")
    model, fs, transform = r["model"], r["feature_set"], r["transform"]
    if model not in MODELS:
        raise ConfigError(f"unknown model {model!r}")
    if fs not in FEATURE_SETS:
        raise ConfigError(f"unknown feature set {fs!r}")
    if transform not in TRANSFORMS[model]:
        raise ConfigError(f"{model} does not support transform {transform!r}")
    if r["run_id"] != _run_id(model, fs, transform):
        raise ConfigError(f"run_id {r['run_id']!r} should be {_run_id(model, fs, transform)!r}")
    approved = resolve_model_params(model)
    if r["model_params"] != approved:
        raise ConfigError(f"{r['run_id']}: model_params differ from the approved settings {approved}")
    if r["research_status"] not in STATUSES:
        raise ConfigError(f"unknown research_status {r['research_status']!r}")
    if (model == "naming_heuristic") != (r["research_status"] == EXPLORATORY):
        raise ConfigError("naming_heuristic (and only it) must have research_status 'exploratory_post_hoc'")
    return RunSpec(**{k: r[k] for k in RUN_KEYS})


def validate_run_config(cfg):
    """Return the RunSpecs of the primary matrix; raise ConfigError on any deviation."""
    if set(cfg) != CONFIG_KEYS:
        raise ConfigError(f"config keys differ: missing {sorted(CONFIG_KEYS - set(cfg))}, "
                          f"unexpected {sorted(set(cfg) - CONFIG_KEYS)}")
    if cfg["dataset_sha256"] != config.MASTER_SHA256:
        raise ConfigError("dataset_sha256 differs from the canonical master")
    meta = json.loads(Path(SPLIT_META).read_text(encoding="utf-8"))
    if cfg["split_hash"] != meta["split_hash"]:
        raise ConfigError("split_hash differs from the saved split")
    if cfg["primary_metric"] != M.PRIMARY_METRIC or cfg["primary_feature_set"] != "main":
        raise ConfigError("primary metric must be average_precision and the primary feature set main")
    specs = [_validate_run(r) for r in cfg["runs"]]
    ids = [s.run_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate run ids")
    if ids != APPROVED_MATRIX:
        raise ConfigError(f"runs differ from the approved matrix: missing {sorted(set(APPROVED_MATRIX) - set(ids))}, "
                          f"unexpected {sorted(set(ids) - set(APPROVED_MATRIX))}" if set(ids) != set(APPROVED_MATRIX)
                          else "runs are not in the approved order")
    seed_spread_specs(cfg)                                     # validates the optional block too
    return specs


def seed_spread_specs(cfg):
    """RunSpecs of the optional 20-seed stratified-random experiment (never part of the primary matrix)."""
    opt = cfg["optional_experiments"]
    if set(opt) != {"stratified_random_seed_spread"}:
        raise ConfigError(f"unknown optional experiments {sorted(opt)}")
    e = opt["stratified_random_seed_spread"]
    if set(e) != SPREAD_KEYS:
        raise ConfigError(f"seed-spread keys {sorted(e)} differ from {sorted(SPREAD_KEYS)}")
    seeds = e["seeds"]
    if e["run_by_default"] is not False or e["model"] != "stratified_random" or e["transform"] != "none" \
            or e["feature_set"] not in FEATURE_SETS or e["research_status"] != "reference":
        raise ConfigError("the seed spread must be an opt-in stratified_random reference experiment")
    if len(seeds) != 20 or len(set(seeds)) != 20 or not all(isinstance(s, int) and not isinstance(s, bool)
                                                           for s in seeds):
        raise ConfigError("the seed spread needs 20 distinct integer seeds")
    if resolve_model_params("stratified_random")["random_state"] in seeds:
        raise ConfigError("the seed spread must not reuse the primary run's seed")
    return [RunSpec(run_id=f"stratified_random__{e['feature_set']}__seed{s:02d}", model="stratified_random",
                    feature_set=e["feature_set"], transform="none",
                    model_params=resolve_model_params("stratified_random", {"random_state": s}),
                    role=e["role"], research_status=e["research_status"]) for s in seeds]


def load_run_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_run_config(cfg)
    return cfg


# ---------------------------------------------------------------------------- summaries
def summary_table(summaries):
    rows = []
    for s in summaries:
        cv, thr, dev = s["cv_average_precision"], s["threshold_selection"], s["development_threshold_metrics"]
        rows.append({"run_id": s["run_id"], "model": s["model"], "feature_set": s["feature_set"],
                     "transform": s["transform"], "role": s["role"], "research_status": s["research_status"],
                     "cv_ap_mean": cv["mean"], "cv_ap_std_ddof1": cv["std"],
                     **{f"ap_fold{k}": v for k, v in enumerate(cv["per_fold"])},
                     "roc_auc_fold_mean": float(np.mean(s["cv_roc_auc_per_fold"])),
                     "threshold": thr["threshold"], "n_tied_max_f1": thr["n_tied_max_f1"],
                     **{f"dev_{m}_optimistic": dev[m] for m in M.THRESHOLD_METRICS},
                     "n_warnings": s["n_warnings"]})
    return pd.DataFrame(rows)


def summary_markdown(table, selection, baseline_check):
    def md(df):
        cols = ["run_id", "cv_ap_mean", "cv_ap_std_ddof1", "roc_auc_fold_mean", "threshold",
                "dev_precision_optimistic", "dev_recall_optimistic", "dev_f1_optimistic", "dev_mcc_optimistic"]
        df = df[cols]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(lines)

    confirm = table[table.research_status != EXPLORATORY]
    explore = table[table.research_status == EXPLORATORY]
    out = ["# Stage 8 development cross-validation: summary", "",
           "Development rows only (24,967 rows, 261 projects, 5 saved grouped folds). Held-out test rows were "
           "not used.", "",
           "- CV AP = mean and sample std (ddof=1) of the 5 validation-fold APs (primary metric).",
           f"- dev_*_optimistic: {OPTIMISTIC_NOTE}.", ""]
    for fs, title in (("main", "Main 40 (primary)"), ("comparison", "Comparison 42 (secondary)")):
        out += [f"## {title}", "", md(confirm[confirm.feature_set == fs]), ""]
    paired = confirm.pivot_table(index=["model", "transform"], columns="feature_set", values="cv_ap_mean")
    paired["comparison_minus_main"] = paired["comparison"] - paired["main"]
    out += ["## Comparison 42 minus Main 40 (CV AP mean)", "", "| model | transform | main | comparison | difference |",
            "|---|---|---|---|---|"]
    out += [f"| {m} | {t} | {r['main']:.4f} | {r['comparison']:.4f} | {r['comparison_minus_main']:+.4f} |"
            for (m, t), r in paired.iterrows()]
    out += [""]
    out += ["## Exploratory, post-hoc: naming heuristic (not ranked with the models above)", "",
            md(explore), "",
            "Its flags were proposed after full-dataset label-derived summaries had been seen "
            "(docs/naming_heuristic.md); it is not a confirmatory baseline and cannot be selected for Stage 9.", "",
            "## Stage 9 candidates (rule fixed before the results)", ""]
    out += [f"- {fs}: `{run_id}`" for fs, run_id in selection.items()]
    out += ["", f"Baselines identical for main and comparison: {', '.join(baseline_check)}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------- main
def main(argv=None, data_loader=load_development_data, runs_dir=config.RUNS_DIR, day=None, out=print):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="validate config, data and folds; fit nothing")
    ap.add_argument("--only", action="append", default=[], metavar="RUN_ID", help="run only this configured run")
    ap.add_argument("--seed-spread", action="store_true", help="run the optional 20-seed random baseline instead")
    args = ap.parse_args(argv)
    day = day or date.today().isoformat()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    specs = validate_run_config(cfg)
    if args.seed_spread:
        if args.only:
            raise ConfigError("--seed-spread and --only cannot be combined")
        specs = seed_spread_specs(cfg)
    elif args.only:
        unknown = sorted(set(args.only) - {s.run_id for s in specs})
        if unknown:
            raise ConfigError(f"unknown run ids: {unknown}")
        specs = [s for s in specs if s.run_id in args.only]
    full_matrix = not args.only and not args.seed_spread

    names = {s.run_id: rs.run_directory_name(PREFIX, s.run_id, day) for s in specs}
    out(f"{'DRY RUN: ' if args.dry_run else ''}{len(specs)} run(s) selected:")
    for s in specs:
        out(f"  {s.run_id:46} -> results/runs/{names[s.run_id]}")
    for stale in rs.stale_temporary_runs(runs_dir):
        out(f"WARNING: leftover temporary folder from an interrupted run (not a result): {stale.name}")
    existing = [n for n in names.values() if (Path(runs_dir) / n).exists()]
    if existing:
        raise rs.RunStoreError(f"run folders already exist (never overwritten): {existing}")

    data = {fs: data_loader(fs) for fs in sorted({s.feature_set for s in specs})}
    for fs, d in data.items():
        out(f"development data ({fs}): {d.split_info['development']}, {len(d.folds)} folds, "
            f"split hash {d.split_info['split_hash'][:12]}…")
    if args.dry_run:
        out("Dry run complete: nothing was fitted.")
        return 0

    config_sha = sha256_of(config_path)
    results, failures = {}, []
    for s in specs:
        prov = rs.provenance(data[s.feature_set].split_info,
                             {"started_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
                              "config_sha256": config_sha, "run_group": cfg["name"]})
        try:
            res = run_development_cv(s, data[s.feature_set])
        except RunFailedError as exc:
            folder = rs.write_failed_run(runs_dir, names[s.run_id], s, exc, exc.warnings_log, prov)
            out(f"FAILED {s.run_id}: {exc} -> {folder.name}")
            failures.append(s.run_id)
            continue
        folder = rs.write_run(runs_dir, names[s.run_id], s, res, prov)
        results[s.run_id] = res
        cv = res["summary"]["cv_average_precision"]
        out(f"done   {s.run_id}: CV AP {cv['mean']:.4f} ± {cv['std']:.4f} -> {folder.name}")

    if failures:
        out(f"{len(failures)} run(s) failed: {failures}. No matrix summary was written.")
        return 1
    if full_matrix or args.seed_spread:
        summaries = [r["summary"] for r in results.values()]
        table = summary_table(summaries)
        files = {"summary.csv": lambda p: table.to_csv(p, index=False, lineterminator="\n")}
        if full_matrix:
            checked = check_baseline_equivalence(results)
            selection = {fs: select_stage9_candidate(summaries, fs) for fs in ("main", "comparison")}
            files["summary.md"] = rs.text_file(summary_markdown(table, selection, checked))
            files["stage9_selection.json"] = rs.text_file(json.dumps(
                {"rule": cfg["stage9_selection_rule"], "selected": selection}, indent=2) + "\n")
            name = rs.run_directory_name(PREFIX, "summary", day)
        else:
            spread = table["cv_ap_mean"]
            files["spread.json"] = rs.text_file(json.dumps(
                {"purpose": cfg["optional_experiments"]["stratified_random_seed_spread"]["purpose"],
                 "n_seeds": int(len(spread)), "cv_ap_mean_min": float(spread.min()),
                 "cv_ap_mean_median": float(spread.median()), "cv_ap_mean_max": float(spread.max())}, indent=2) + "\n")
            name = rs.run_directory_name(PREFIX, "seed_spread_summary", day)
        folder = rs.write_folder(runs_dir, name, files)
        out(f"summary -> {folder.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
