# Stage 8 results freeze: development cross-validation

**These are development cross-validation results, not final held-out test results.** The
62 held-out test projects were not loaded or evaluated. Stage 9 (one held-out evaluation per
feature set) needs separate approval.

- **Frozen:** 2026-10-31. Runs executed 13:51:37–13:52:17 UTC.
- **Manifest:** `results/runs/2026-10-31_stage8_freeze_manifest.json`, SHA-256
  `d054c049f604e7408ad486463cfd4379f7b3841563575386c3189c80286cd783`.
- **Results:** `results/runs/2026-10-31_stage8_<run_id>/` (14 folders) and
  `results/runs/2026-10-31_stage8_summary/`.

## Design

| Item | Setting |
|---|---|
| Data | Development rows of the master dataset: 24,967 rows, 261 projects, 193 renamed (0.77%) |
| Folds | The 5 saved grouped (project) folds; train on 4, validate on 1; 37–41 renamed per validation fold |
| Learned steps | Scalers, Yeo-Johnson lambdas, class weights and the random baseline's rate, all fitted inside each training fold |
| Feature sets | **Main 40 (primary)**; comparison 42 = main + `file_loc`, `file_n_methods` (secondary) |
| Primary metric | Average Precision (AP): mean ± sample std (ddof=1) of the 5 validation-fold APs. Chance level = fold positive rate (≈ 0.0077) |
| Supplementary | ROC-AUC per fold; precision / recall / F1 / MCC at the max-F1 threshold on pooled development OOF scores, **optimistic** |
| Settings | Approved baseline settings, no tuning (`configs/runs/stage8_development_cv.json`) |

## The 14 experiments: CV AP, mean ± sd (ROC-AUC fold mean)

| Model | Main 40 (primary) | Comparison 42 |
|---|---|---|
| **Confirmatory models** | | |
| Logistic Regression, log1p + asinh | 0.0130 ± 0.0042 (0.62) | 0.0188 ± 0.0105 (0.67) |
| Logistic Regression, Yeo-Johnson | 0.0133 ± 0.0035 (0.63) | 0.0191 ± 0.0089 (0.68) |
| Random Forest | 0.0114 ± 0.0033 (0.55) | 0.0135 ± 0.0040 (0.59) |
| LightGBM | 0.0103 ± 0.0034 (0.49) | 0.0112 ± 0.0018 (0.56) |
| **Chance references** | | |
| dummy_prior | 0.0077 ± 0.0003 (0.50) | identical |
| stratified_random (seed 42) | 0.0090 ± 0.0015 (0.51) | identical |
| **Exploratory, post-hoc (not ranked)** | | |
| naming_heuristic | 0.0097 ± 0.0032 (0.52) | identical |

Per-fold values, thresholds and the optimistic threshold metrics are in each run's
`summary.json` and `fold_metrics.csv`.

## Findings

1. **The signal is weak.** On main 40, the best models (both Logistic Regression variants)
   reach about 1.7× chance-level AP (≈ 0.013 vs 0.0077), with ROC-AUC ≈ 0.63. At the
   optimistic development operating point, precision is about 2%.
2. **Logistic Regression is the only model family clearly above both chance references on
   main 40.** Random Forest is barely above the random baseline. LightGBM is within its range
   (ROC-AUC 0.49).
3. **The two Logistic Regression variants are practically equal.** Yeo-Johnson leads
   log1p/asinh by 0.0003 (main) and 0.0004 (comparison), far below the fold std of about
   0.004 and 0.009.
4. **File size adds AP, but unevenly.** For Logistic Regression it adds about +0.006 AP: AP
   improves in 4 of 5 folds, fold 2 provides 55–58% of the gain, and removing any single
   positive project from fold 2 still leaves a gain of at least +0.011. The comparison
   models raise the scores of methods in small files (Spearman of the score change with
   `file_loc` ≈ −0.72). Renamed development methods sit in smaller files (median
   `file_loc` 230 vs 414). This is descriptive, not causal, and may partly reflect how
   ActRef collects changed files.
5. **The naming heuristic is near chance** (exploratory; see `docs/naming_heuristic.md`).
6. **Folds vary a lot.** Fold std is 25–55% of the mean AP, so differences of a few
   thousandths between models are not meaningful.

## Stage 9 selection (development only; not an authorization)

By the rule fixed before the results (highest CV AP mean among LR, RF and LightGBM; exact
tie → simpler model):

- **main:** `logistic_regression__main__yeo_johnson`
- **comparison:** `logistic_regression__comparison__yeo_johnson`

These are recorded in `results/runs/2026-10-31_stage8_summary/stage9_selection.json`. Given
finding 3, the choice between the LR variants is a near-tie resolved mechanically by the rule.

## Limitations and open questions

- With 193 positives (37–41 per fold, 67 positive projects), CV estimates are noisy. The
  held-out test set has only 50 positives in 12 projects.
- 76 of the 243 positives are AI-reviewed and unconfirmed. Sensitivity runs (Stage 8b) and
  human label review are not done yet.
- The file-size association may reflect dataset construction (only changed files are stored).
- The thresholds are unstable across folds (see `fold_threshold_diagnostics`), and the
  threshold metrics are optimistic.
- The Stage 8 runner code was uncommitted when it ran (`dirty: true` in provenance). The
  manifest records the hash of every source file used.

## Provenance

| Item | Value |
|---|---|
| Git | branch `feature/stage-8-development-cv`, commit `669f07ed3c58e5678f963adb6a4cc0d98a748664`, working tree dirty (Stage 8 files untracked) |
| Master dataset | `data/processed/master/actref_master_dataset.csv`, `e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0` |
| Split | hash `16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c`; CSV `00644a426bf1d835c76101dd3aab4fe6613228ecc0888fd8913881122df7f7d1` |
| Config | `configs/runs/stage8_development_cv.json`, `f57af623c7b890c4c74ad06270c23dbbeb6bddbad2d8927095a9046ae9687c6d` |
| Environment | Python 3.12.0; numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0; `requirements-lock.txt` `a493f8db091d2b3fdab92806a8fd51ba8cf86571102e8d0ce3d8d4d10379d7ea` |
| Stage 8 code | `src/evaluation/cross_validate.py`, `run_store.py`, `run_stage8.py` (hashes in the manifest) |

**Validation evidence.** The manifest separates two kinds of check:

- **Runtime-enforced checks:** for example `classes_ == [0, 1]`, the probability validation,
  OOF coverage and order, and the held-out-project guard. These passed (all runs completed),
  but their per-fold details are not stored.
- **Checks reconstructable from the artifacts:** these were recomputed independently from
  the saved files, and all matched exactly. They include row coverage and order against the
  split, per-fold AP and ROC-AUC, the threshold and its metrics, and baseline byte-equality.

**Re-check.** Recompute the SHA-256 of every file listed in the manifest and compare it with
the recorded value. `tests/test_stage8_freeze.py` does this for the inputs and results.

**Line endings.** The manifest's source-file hashes are of working-tree bytes, which depend on
`core.autocrlf`. `results/runs/2026-10-31_stage8_freeze_source_addendum.json` adds, for each
source file, its LF-normalized SHA-256 and its git blob id. Those can be checked on any
checkout. Since the commit that preserves Stage 8, `.gitattributes` marks
`results/runs/** -text`, so every result file, including `summary.md`, is stored and checked
out byte-for-byte.
