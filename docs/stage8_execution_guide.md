# Stage 8 execution guide: development cross-validation

**Status:** the infrastructure is implemented and tested on synthetic data only. No real-data
dry run, cross-validation, OOF prediction or metric has been produced yet. Each real-data
step below needs explicit approval first.

## What Stage 8 does

Every configured run is cross-validated on the **development rows only**: 24,967 rows,
261 projects and 193 positives, using the 5 saved grouped folds in
`data/processed/splits/project_split_seed42.*`. For each fold, a fresh copy of the
unfitted pipeline is fitted on the 4 training folds and scores the validation fold.
Scalers, Yeo-Johnson lambdas, class weights and the random baseline's rate are all
learned inside that fit, so each training fold learns its own. Held-out test rows are
dropped by `load_development_data()` before anything else happens, and Stage 8 has no
way to read them.

## The fixed matrix (`configs/runs/stage8_development_cv.json`)

| Run id | Role | Research status |
|---|---|---|
| `dummy_prior__{main,comparison}` | chance reference | reference |
| `stratified_random__{main,comparison}` | chance reference | reference |
| `naming_heuristic__{main,comparison}` | exploratory heuristic | **exploratory_post_hoc** |
| `logistic_regression__{main,comparison}__log1p_asinh` | confirmatory model | confirmatory |
| `logistic_regression__{main,comparison}__yeo_johnson` | confirmatory model, transform experiment | confirmatory |
| `random_forest__{main,comparison}` | confirmatory model | confirmatory |
| `lightgbm__{main,comparison}` | confirmatory model | confirmatory |

The config is validated strictly. The run ids must be exactly this matrix, in this
order. Every `model_params` block must equal the approved settings in
`src/models/pipelines.py`, so nothing can be tuned through the config. The dataset
and split hashes must match.

The optional `stratified_random_seed_spread` experiment (20 seeds, 1-20) runs only
with `--seed-spread`. It shows chance-level spread and is never used for selection.

## Commands

```
python src/evaluation/run_stage8.py --dry-run          # config, data counts, checksums, folds; fits nothing
python src/evaluation/run_stage8.py --only lightgbm__main
python src/evaluation/run_stage8.py                    # the full 14-run matrix + summary
python src/evaluation/run_stage8.py --seed-spread      # optional chance-spread experiment
```

The selected run ids and their target folders are printed before anything runs.
Importing the module runs nothing. Running the same day again is refused, because run
folders are never overwritten.

## Rules applied to every run

- A `ConvergenceWarning` fails the run. Approved settings such as `max_iter` are
  never changed to force convergence.
- Every other warning (for example Yeo-Johnson overflow) is recorded with its fold.
- Non-finite transformed values fail the run.
- Probabilities are used only after checking that `classes_ == [0, 1]`. They must be
  finite and in [0, 1], and every development row must get exactly one out-of-fold
  score, in master order.
- A failed run is written as `<name>__FAILED`, containing `error.json` and
  `warnings.json` but no results. No matrix summary is written if any run failed.
- After a complete matrix run, the three baselines must give identical OOF scores
  for main and comparison, because they ignore the file-size columns.

## Artifacts (`results/runs/`)

```
<YYYY-MM-DD>_stage8_<run_id>/
  run_config.json      model, resolved parameters, seeds, feature set, transform, role, research_status
  provenance.json      git commit + dirty flag, Python / package versions, requirements-lock SHA-256,
                       master SHA-256, split hash, split CSV SHA-256, fold statistics, times, status
  fold_metrics.csv     per fold: rows, positives, projects, prevalence, AP, ROC-AUC, fit / predict seconds, warnings
  oof_predictions.csv  master_row, commit, file, class, name, meta_project_group, fold, renamed, oof_probability
  summary.json         CV AP (mean, std ddof=1, per fold), threshold, ties, optimistic dev metrics, diagnostics
  warnings.json        every captured warning with its fold
<YYYY-MM-DD>_stage8_summary/
  summary.csv, summary.md, stage9_selection.json
```

Folders are written to `.tmp_*` first and renamed only when complete. A `.tmp_*`
folder is never a result: the CLI warns about leftovers. Fitted models are not stored.

## Reading the results

- **Primary result:** `cv_average_precision.mean` ± `std`, the mean and sample std
  (ddof=1) of the 5 validation-fold APs. Compare each fold's AP with its prevalence
  (`fold_metrics.csv`), which is the chance-level AP for that fold. The pooled-OOF AP
  is not reported.
- **ROC-AUC per fold** is supplementary.
- **Threshold metrics:** `threshold_selection` gives the maximum-F1 threshold on the
  pooled development OOF scores, with the number of tied thresholds.
  `development_threshold_metrics` are **optimistic**, because they are measured on
  the scores that chose the threshold. They describe the operating point and are not
  a performance estimate. `fold_threshold_diagnostics` shows how stable the threshold
  is.
- **Main 40 is the primary table; comparison 42 is secondary.** `summary.md` also
  shows comparison minus main per model.
- **`naming_heuristic`** appears only in the summary's exploratory section, never
  ranked with the models.
- **`stage9_selection.json`** applies the rule fixed before any result: per feature
  set, the highest CV AP mean among LR (`log1p_asinh`, `yeo_johnson`), Random Forest
  and LightGBM, with an exact tie going to the simpler model.
- **Pooled-OOF caveat:** baseline scores differ between folds (each fold has its own
  training rate), so pooled-OOF scores carry fold-level artefacts. That is why the CV
  result is the fold mean.

## Order of execution (each step needs approval)

1. `--dry-run` on real data, and review its printed counts and folds.
2. The full matrix, once. No reruns with changes after seeing results, except
   documented bug fixes.
3. Review the summary and warnings, then commit the run folders.
4. Stage 9 (one held-out evaluation per feature set) is a separate stage with its
   own approval.
