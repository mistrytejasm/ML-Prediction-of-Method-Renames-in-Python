r"""
run_stage9.py
Stage 9: the one-time final held-out evaluation (docs/stage9_final_evaluation_protocol.md).

For each feature set (main 40 primary, comparison 42 secondary) the model selected by Stage 8
(results/runs/<day>_stage8_summary/stage9_selection.json) is refitted ONCE on all development rows
and scores the held-out test rows ONCE, at the threshold saved by Stage 8 (max F1 on pooled
development OOF scores, re-verified here from the saved OOF file). Nothing is tuned or selected on test
data; test labels are read only after the test probabilities exist.

  load_stage8_selection(fs)   selected RunSpec + saved threshold, verified against the OOF scores
  split_partitions(...)       development and test rows from the saved assignment; no shared project
  evaluate_final(...)         fit on development, predict test, metrics, bootstrap, error analysis

Artifacts (write-once; refuses to run if they exist):
  results/runs/<day>_stage9_final_test/     run_config.json, provenance.json, test_predictions.csv,
                                            test_metrics.json, bootstrap_metrics.json,
                                            error_analysis.csv, summary.md
  results/models/<day>_stage9_final_test/   <feature_set>_pipeline.joblib + model_metadata.json

Run (only with authorization; there is no dry-run of the test step):
    python src/evaluation/run_stage9.py
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.load_master import GROUP_COLUMN, sha256_of  # noqa: E402
from src.data.prepare_dataset import prepare_dataset  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation import run_store as rs  # noqa: E402
from src.evaluation.cross_validate import RunSpec, build_spec_pipeline, select_development  # noqa: E402
from src.evaluation.splits import KEY, load_splits, outer_indices  # noqa: E402

STAGE8_DAY = "2026-10-31"
FEATURE_SETS = ("main", "comparison")                    # primary first
EXPECTED = {"development": {"rows": 24967, "projects": 261, "positives": 193},
            "test": {"rows": 6242, "projects": 62, "positives": 50}}
EXPECTED_MODEL = ("logistic_regression", "yeo_johnson")
PROTOCOL = config.DOCS / "stage9_final_evaluation_protocol.md"
MODELS_DIR = config.ROOT / "results" / "models"
RUN_NAME = "stage9_final_test"
SOURCE_FILES = ["src/evaluation/run_stage9.py", "src/evaluation/cross_validate.py", "src/evaluation/metrics.py",
                "src/evaluation/splits.py", "src/evaluation/run_store.py", "src/models/pipelines.py",
                "src/models/baselines.py", "src/preprocessing/transforms.py", "src/features/transform_groups.py",
                "src/features/feature_sets.py", "src/data/prepare_dataset.py", "src/data/load_master.py",
                "src/config.py"]


class Stage9Error(RuntimeError):
    """An integrity check failed; the evaluation must not proceed."""


# ---------------------------------------------------------------------------- Stage 8 selection
def load_stage8_selection(feature_set, runs_dir=config.RUNS_DIR, day=STAGE8_DAY, dev_master_rows=None):
    """Return (RunSpec, threshold, stage8 summary, evidence) for the run Stage 8 selected for `feature_set`."""
    runs_dir = Path(runs_dir)
    sel_path = runs_dir / f"{day}_stage8_summary" / "stage9_selection.json"
    selected = json.loads(sel_path.read_text(encoding="utf-8"))["selected"]
    if feature_set not in selected:
        raise Stage9Error(f"no Stage 8 selection for {feature_set!r}")
    run_dir = runs_dir / f"{day}_stage8_{selected[feature_set]}"
    if not rs.is_complete_run(run_dir):
        raise Stage9Error(f"{run_dir.name} is not a complete Stage 8 run")
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    spec = RunSpec(**cfg)
    if spec.feature_set != feature_set or spec.research_status == "exploratory_post_hoc":
        raise Stage9Error(f"selected run {spec.run_id} does not fit {feature_set} / confirmatory")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(run_dir / "oof_predictions.csv", keep_default_na=False)
    scores = pd.to_numeric(oof["oof_probability"], errors="coerce")       # a blank or text score becomes NaN
    if not oof["master_row"].is_unique or scores.isna().any():
        raise Stage9Error("Stage 8 OOF predictions have duplicate rows or missing scores")
    if dev_master_rows is not None and not np.array_equal(oof["master_row"].to_numpy(), np.asarray(dev_master_rows)):
        raise Stage9Error("Stage 8 OOF rows differ from the development rows")
    saved = summary["threshold_selection"]["threshold"]
    try:
        recomputed = M.select_max_f1_threshold(oof[config.TARGET].to_numpy(), scores.to_numpy())
    except M.EvaluationInputError as exc:
        raise Stage9Error(f"Stage 8 OOF predictions are invalid: {exc}") from exc
    if recomputed.threshold != saved or summary["threshold_selection"]["data"] != "pooled development OOF scores":
        raise Stage9Error(f"saved threshold {saved!r} cannot be verified (recomputed {recomputed.threshold!r})")
    evidence = {"stage8_run": run_dir.name, "selection_file": sel_path.name,
                "selection_file_sha256": sha256_of(sel_path), "run_config_sha256": sha256_of(run_dir / "run_config.json"),
                "summary_sha256": sha256_of(run_dir / "summary.json"),
                "oof_sha256": sha256_of(run_dir / "oof_predictions.csv"),
                "threshold_recomputed_from_oof": recomputed.threshold, "threshold_n_tied": recomputed.n_tied}
    return spec, float(saved), summary, evidence


# ---------------------------------------------------------------------------- partitions
def split_partitions(X, y, groups, assign, meta, expected=None):
    """Development data (via select_development, with its fold checks) and the test rows of the saved split."""
    dev = select_development(X, y, groups, assign, meta)
    dev_pos, test_pos = outer_indices(assign)
    if len(np.intersect1d(dev_pos, test_pos)) or len(dev_pos) + len(test_pos) != len(assign):
        raise Stage9Error("rows are not in exactly one partition")
    test = {"X": X.iloc[test_pos], "y": y.iloc[test_pos], "groups": groups.iloc[test_pos],
            "keys": assign.iloc[test_pos][["master_row"] + KEY].reset_index(drop=True)}
    test["keys"].index = test["X"].index
    shared = set(dev.groups) & set(test["groups"])
    if shared:
        raise Stage9Error(f"projects in both development and test: {sorted(shared)[:5]}")
    counts = {"development": {"rows": len(dev.X), "projects": int(dev.groups.nunique()), "positives": int(dev.y.sum())},
              "test": {"rows": len(test["X"]), "projects": int(test["groups"].nunique()),
                       "positives": int(test["y"].sum())}}
    if counts["test"] != {k: meta["counts"]["test"][k] for k in ("rows", "projects", "positives")}:
        raise Stage9Error(f"test counts {counts['test']} differ from the split metadata")
    if expected is not None and counts != expected:
        raise Stage9Error(f"partition counts {counts} differ from the approved {expected}")
    counts["test"]["positive_projects"] = int(test["groups"][test["y"] == 1].nunique())
    return dev, test, counts


# ---------------------------------------------------------------------------- one feature set
def evaluate_final(spec, threshold, dev, test, n_boot=M.N_BOOT, seed=M.BOOT_SEED, pipeline_factory=None):
    """Fit on all development rows, score the test rows once, and evaluate at the saved threshold."""
    pipe = (pipeline_factory or build_spec_pipeline)(spec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        pipe.fit(dev.X, dev.y)                                   # development rows and labels only
        t1 = time.perf_counter()
        proba = pipe.predict_proba(test["X"])                    # test features only; labels unused here
        t2 = time.perf_counter()
    log = [{"stage": "fit/predict", "category": w.category.__name__, "message": str(w.message)} for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise Stage9Error("ConvergenceWarning while fitting the final model")
    if list(getattr(pipe, "classes_", [])) != [0, 1]:
        raise Stage9Error(f"classes_ is {list(getattr(pipe, 'classes_', []))}, expected [0, 1]")
    proba = np.asarray(proba)
    if proba.shape != (len(test["X"]), 2):
        raise Stage9Error(f"predict_proba returned shape {proba.shape}")
    try:
        p = M.validate_probabilities(proba[:, 1], len(test["X"]))
    except M.EvaluationInputError as exc:
        raise Stage9Error(f"invalid test probabilities: {exc}") from exc

    # ---- test labels are used only from here on
    y = test["y"].to_numpy()
    pred = M.predict_at_threshold(p, threshold)
    m = M.evaluate(y, p, threshold)
    n = len(y)
    metrics = {**{k: m[k] for k in M.METRICS}, **{k: m[k] for k in ("tp", "fp", "fn", "tn")},
               "n": n, "n_positive": int(y.sum()), "prevalence": float(y.mean()),
               "random_ap_baseline": float(y.mean()), "ap_lift_over_random": m["average_precision"] / float(y.mean()),
               "n_predicted_positive": int(pred.sum()), "share_predicted_positive": float(pred.mean()),
               "threshold": threshold, "fit_seconds": round(t1 - t0, 3), "predict_seconds": round(t2 - t1, 3)}
    boot = M.project_bootstrap(y, p, pred, test["groups"].to_numpy(), n_boot=n_boot, seed=seed, threshold=threshold)
    table = test["keys"].copy()
    table[GROUP_COLUMN] = test["groups"].to_numpy()
    table["feature_set"] = spec.feature_set
    table[config.TARGET] = y
    table["probability"] = p
    table["threshold"] = threshold
    table["predicted"] = pred
    table["outcome"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 1), (y == 1) & (pred == 0)],
                                 ["TP", "FP", "FN"], "TN")
    table["rank"] = pd.Series(p).rank(ascending=False, method="min").astype(int).to_numpy()
    oc = pd.get_dummies(table["outcome"]).reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0).astype(int)
    per_project = pd.concat([table[[GROUP_COLUMN, config.TARGET]], oc], axis=1).groupby(GROUP_COLUMN).agg(
        rows=(config.TARGET, "size"), positives=(config.TARGET, "sum"), TP=("TP", "sum"), FP=("FP", "sum"),
        FN=("FN", "sum"), TN=("TN", "sum")).reset_index()
    per_project["best_positive_rank"] = per_project[GROUP_COLUMN].map(
        table[table[config.TARGET] == 1].groupby(GROUP_COLUMN)["rank"].min())
    return {"pipeline": pipe, "metrics": metrics, "bootstrap": boot, "table": table, "per_project": per_project,
            "warnings": log}


# ---------------------------------------------------------------------------- main
def _source_hashes():
    out = []
    for s in SOURCE_FILES:
        b = (config.ROOT / s).read_bytes()
        out.append({"path": s, "sha256": hashlib.sha256(b).hexdigest(),
                    "sha256_lf_normalized": hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()})
    return out


def _boot_json(boot):
    return {"settings": dict(boot.attrs), "metrics": {m: {k: (None if pd.isna(v) else float(v) if k in ("point", "ci_low", "ci_high")
                                                          else int(v)) for k, v in boot.loc[m].items()}
                                                       for m in boot.index}}


def summary_markdown(results, counts, cfg):
    def f(v):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
    out = ["# Stage 9 final held-out evaluation: summary", "",
           "> Labels were constructed automatically (rule / AI review); **no person verified them**. Results measure "
           "agreement with these labels on unseen projects.", "",
           f"Test: {counts['test']['rows']:,} rows, {counts['test']['projects']} projects, "
           f"{counts['test']['positives']} positives from {counts['test']['positive_projects']} positive-containing projects "
           f"(prevalence {counts['test']['positives'] / counts['test']['rows']:.4f} = random AP baseline).", ""]
    for fs in FEATURE_SETS:
        r, d = results[fs], cfg["selected"][fs]
        m, b = r["metrics"], r["bootstrap"]
        out += [f"## {'Main 40 (primary)' if fs == 'main' else 'Comparison 42 (secondary)'}: {d['run_id']}", "",
                "| Metric | Held-out test | 95% project-bootstrap interval | Development (Stage 8) |", "|---|---|---|---|",
                f"| Average Precision (primary) | **{f(m['average_precision'])}** | {f(b.loc['average_precision', 'ci_low'])}–"
                f"{f(b.loc['average_precision', 'ci_high'])} | CV mean {f(d['dev_cv_ap_mean'])} ± {f(d['dev_cv_ap_std'])} |",
                f"| Random AP baseline (prevalence) | {f(m['random_ap_baseline'])} | | ≈ 0.0077 |",
                f"| ROC-AUC | {f(m['roc_auc'])} | {f(b.loc['roc_auc', 'ci_low'])}–{f(b.loc['roc_auc', 'ci_high'])} | "
                f"fold mean {f(d['dev_roc_auc_fold_mean'])} |"]
        for k in M.THRESHOLD_METRICS:
            out.append(f"| {k} at threshold | {f(m[k])} | {f(b.loc[k, 'ci_low'])}–{f(b.loc[k, 'ci_high'])} "
                       f"(undefined in {int(b.loc[k, 'n_undefined'])}) | OOF {f(d['dev_oof_threshold_metrics'][k])} (optimistic) |")
        out += ["", f"Threshold {m['threshold']!r} (Stage 8, pooled development OOF max F1). Confusion matrix: "
                f"TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']}; predicted positive {m['n_predicted_positive']} "
                f"({m['share_predicted_positive']:.2%} of test rows).", ""]
    out += ["Intervals: stratified project bootstrap, 2,000 replicates, seed 42; conditional on the observed 12 "
            "positive-containing / 50 negative-only test projects; approximate, no nominal-coverage guarantee.", ""]
    return "\n".join(out)


def main(argv=None, data_loader=None, runs_dir=config.RUNS_DIR, models_dir=MODELS_DIR, day=None, expected=EXPECTED,
         n_boot=M.N_BOOT, stage8_runs_dir=None, out=print):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    day = day or date.today().isoformat()
    name = f"{day}_{RUN_NAME}"
    runs_dir, models_dir = Path(runs_dir), Path(models_dir)
    stage8_runs_dir = Path(stage8_runs_dir or runs_dir)
    for d in (runs_dir / name, models_dir / name):
        if d.exists():
            raise rs.RunStoreError(f"{d} already exists; Stage 9 is evaluated once and never overwritten")
    started = datetime.now(timezone.utc).isoformat()

    # ---- integrity checks before any test prediction
    loader = data_loader or (lambda fs: (*prepare_dataset(fs), *load_splits()))
    results, cfg_selected, prov_selected, split_info = {}, {}, {}, None
    data = {fs: loader(fs) for fs in FEATURE_SETS}
    partitions = {}
    for fs in FEATURE_SETS:
        X, y, groups, assign, meta = data[fs]
        dev, test, counts = split_partitions(X, y, groups, assign, meta, expected)
        spec, thr, s8, ev = load_stage8_selection(fs, stage8_runs_dir, dev_master_rows=dev.X.index.to_numpy())
        if expected is not None and (spec.model, spec.transform) != EXPECTED_MODEL:
            raise Stage9Error(f"Stage 8 selected {spec.model}/{spec.transform}, the protocol fixes {EXPECTED_MODEL}")
        partitions[fs] = (dev, test, counts, spec, thr, s8, ev)
        split_info = dev.split_info
    if not partitions["main"][2] == partitions["comparison"][2]:
        raise Stage9Error("main and comparison partitions differ")
    if not partitions["main"][1]["X"].index.equals(partitions["comparison"][1]["X"].index):
        raise Stage9Error("main and comparison test rows differ")
    out(f"Integrity checks passed: {partitions['main'][2]}")

    # ---- the one-time evaluation
    for fs in FEATURE_SETS:
        dev, test, counts, spec, thr, s8, ev = partitions[fs]
        out(f"Fitting {spec.run_id} on {len(dev.X):,} development rows; scoring {len(test['X']):,} test rows once")
        results[fs] = evaluate_final(spec, thr, dev, test, n_boot=n_boot)
        cfg_selected[fs] = {"run_id": spec.run_id, "spec": asdict(spec), "threshold": thr,
                            "threshold_source": "Stage 8 max F1 on pooled development OOF scores (summary.json), "
                                                "re-verified from oof_predictions.csv",
                            "dev_cv_ap_mean": s8["cv_average_precision"]["mean"],
                            "dev_cv_ap_std": s8["cv_average_precision"]["std"],
                            "dev_cv_ap_per_fold": s8["cv_average_precision"]["per_fold"],
                            "dev_roc_auc_fold_mean": float(np.mean(s8["cv_roc_auc_per_fold"])),
                            "dev_oof_threshold_metrics": {k: s8["development_threshold_metrics"][k]
                                                          for k in M.THRESHOLD_METRICS},
                            "dev_oof_threshold_metrics_note": s8["development_threshold_metrics"]["note"]}
        prov_selected[fs] = ev
        m = results[fs]["metrics"]
        out(f"  {fs}: test AP {m['average_precision']:.4f} (random {m['random_ap_baseline']:.4f}), ROC-AUC {m['roc_auc']:.4f}")
    counts = partitions["main"][2]
    finished = datetime.now(timezone.utc).isoformat()

    run_config = {"stage": 9, "protocol": PROTOCOL.relative_to(config.ROOT).as_posix(),
                  "feature_sets": {"primary": "main", "secondary": "comparison"}, "selected": cfg_selected,
                  "bootstrap": {"n_boot": n_boot, "seed": M.BOOT_SEED, "method": "stratified project percentile",
                                "confidence": 0.95},
                  "label_limitation": "positive labels were constructed automatically (167 rule-accepted, 76 AI-reviewed); "
                                      "none is human-verified"}
    prov = rs.provenance(split_info, {"started_utc": started, "finished_utc": finished, "status": "complete",
                                      "protocol_sha256": sha256_of(PROTOCOL), "stage8_selection": prov_selected,
                                      "partition_counts": counts, "source_files": _source_hashes(),
                                      "python": platform.python_version()})
    metrics_json = {fs: {"test": results[fs]["metrics"], "development": {k: v for k, v in cfg_selected[fs].items()
                                                                         if k.startswith("dev_")},
                         "test_ap_minus_dev_cv_ap": results[fs]["metrics"]["average_precision"] - cfg_selected[fs]["dev_cv_ap_mean"],
                         "warnings": results[fs]["warnings"],
                         "per_project": results[fs]["per_project"].to_dict(orient="records")} for fs in FEATURE_SETS}
    predictions = pd.concat([results[fs]["table"] for fs in FEATURE_SETS], ignore_index=True)
    errors = predictions.sort_values(["feature_set", "rank"])[["feature_set", "master_row"] + KEY + [
        GROUP_COLUMN, config.TARGET, "probability", "rank", "threshold", "predicted", "outcome"]]
    files = {
        "run_config.json": rs.text_file(rs._json(run_config)),
        "provenance.json": rs.text_file(rs._json(prov)),
        "test_predictions.csv": lambda p: predictions.to_csv(p, index=False, lineterminator="\n"),
        "test_metrics.json": rs.text_file(rs._json(metrics_json)),
        "bootstrap_metrics.json": rs.text_file(rs._json({fs: _boot_json(results[fs]["bootstrap"]) for fs in FEATURE_SETS})),
        "error_analysis.csv": lambda p: errors.to_csv(p, index=False, lineterminator="\n"),
        "summary.md": rs.text_file(summary_markdown(results, counts, run_config)),
    }
    model_files = {f"{fs}_pipeline.joblib": (lambda pipe: (lambda p: joblib.dump(pipe, p)))(results[fs]["pipeline"])
                   for fs in FEATURE_SETS}
    model_meta = {"run": name, "fitted_on": "all 24,967 development rows" if expected else "development rows",
                  "feature_sets": {fs: {"run_id": cfg_selected[fs]["run_id"], "threshold": cfg_selected[fs]["threshold"],
                                        "feature_names_in": list(results[fs]["pipeline"].named_steps["check"].feature_names_in_)}
                                   for fs in FEATURE_SETS},
                  "load_with": "joblib.load (pickle: load only files you trust)",
                  "packages": rs.package_versions(), "python": platform.python_version()}
    model_files["model_metadata.json"] = rs.text_file(rs._json(model_meta))
    mfolder = rs.write_folder(models_dir, name, model_files)          # models first: their hashes go into provenance
    prov["model_artifacts"] = {"path": mfolder.relative_to(config.ROOT).as_posix() if mfolder.is_relative_to(config.ROOT)
                               else mfolder.name, "sha256": {q.name: sha256_of(q) for q in sorted(mfolder.iterdir())}}
    files["provenance.json"] = rs.text_file(rs._json(prov))
    folder = rs.write_folder(runs_dir, name, files)
    out(f"results -> {folder}\nmodels  -> {mfolder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
