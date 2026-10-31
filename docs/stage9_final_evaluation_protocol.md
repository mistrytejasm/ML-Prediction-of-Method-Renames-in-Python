# Stage 9 final held-out evaluation: protocol (frozen before any test access)

**Frozen:** 2026-10-31, before the Stage 9 runner was written and before any test prediction existed.
Its SHA-256 is recorded in the Stage 9 run's provenance. Changing it afterwards invalidates the run.

**Authorization:** the project owner explicitly authorized a one-time held-out evaluation on the original
canonical dataset and the existing Stage 8 results.

> **Label limitation (applies to every result):** the 243 positive labels were constructed
> automatically. 167 were accepted by a mechanical rule (body similarity ≥ 0.90 and an unchanged
> parameter list) and 76 by an AI first-pass review. **No person has verified any of them**, and that
> includes all 50 test positives. The human review package (`reports/label_review/`) is prepared but
> not yet completed. Results measure agreement with these automatic labels, not with human-verified
> ground truth.

## Inputs (verified before freezing)

| Item | Value |
|---|---|
| Master dataset | `data/processed/master/actref_master_dataset.csv`, SHA-256 `e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0`: 31,209 rows, 243 positives |
| Split | `data/processed/splits/project_split_seed42.{csv,json}`, split hash `16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c` |
| Development | 24,967 rows, 261 projects, 193 positives |
| Held-out test | 6,242 rows, 62 projects, 50 positives from 12 positive-containing projects; 0 projects shared with development |
| Code | Stage 8 commit `8052c6d`. All 20 source files equal the Stage 8 source addendum (git blob ids) |
| Stage 8 selection | `results/runs/2026-10-31_stage8_summary/stage9_selection.json` |

## What is evaluated (fixed by Stage 8; not changed by anything in Stage 9)

| | Primary | Secondary |
|---|---|---|
| Feature set | **main 40** | comparison 42 (main + `file_loc`, `file_n_methods`) |
| Model | `logistic_regression`, transform `yeo_johnson` | same |
| Settings | C=1.0, `l1_ratio=0.0` (L2), lbfgs, max_iter=5000, `class_weight="balanced"`, random_state=42 | same |
| Preprocessing | FeatureFrameCheck → PowerTransformer(Yeo-Johnson, standardize) on log1p + asinh columns, StandardScaler on raw columns → FeatureOutputCheck | same |
| Selection basis | Highest Stage 8 development CV AP among LR (both transforms), RF and LightGBM. Near-tie with LR log1p/asinh, resolved by the fixed rule | same |
| Stage 8 development CV AP (mean ± sd, ddof=1) | 0.013271 ± 0.003451 | 0.019126 ± 0.008937 |
| **Threshold** (max F1 on pooled development OOF scores; saved; recomputed bit-exact) | **0.859356439967941** | **0.8459416092065658** |
| Development OOF metrics at that threshold (optimistic) | P 0.0206, R 0.0570, F1 0.0303, MCC 0.0218 | P 0.0254, R 0.1244, F1 0.0421, MCC 0.0400 |

**No new model, feature set, transform, parameter or threshold.** Nothing is tuned, selected or chosen
on test data.

## Procedure (executed once)

1. Load the master through `load_master()` (checksum verified) and the saved split through
   `load_splits()` (hash and CSV verified). Build X, y and groups with `prepare_dataset()`.
2. Select development and test rows from the saved assignment. Assert:
   - the counts above;
   - 0 shared projects;
   - every row in exactly one partition.
3. For each feature set:
   1. Build the pipeline with `build_spec_pipeline` from the Stage 8 run's `run_config.json`.
   2. Fit it **once on all 24,967 development rows**. Every learned value (Yeo-Johnson lambdas, scaler
      means and standard deviations, balanced class weights, coefficients) comes from development rows
      only. A `ConvergenceWarning` or non-finite output fails the run.
   3. Predict test probabilities **once**.
   4. Validate `classes_ == [0, 1]` and that the probabilities are finite and in [0, 1].
4. Apply the saved threshold above (predicted positive = probability ≥ threshold). The threshold comes
   from the Stage 8 summary and is re-verified against the saved OOF scores before use.
5. Metrics (Stage 6 implementation):
   - **Average Precision (primary);**
   - ROC-AUC;
   - precision, recall, F1, MCC and the confusion matrix at the threshold;
   - test prevalence (= random AP baseline), and the number and share of rows predicted positive.
6. **Uncertainty:** the Stage 6 stratified project bootstrap, run on the fixed test predictions:
   - positive-containing and negative-only test projects resampled separately;
   - 2,000 replicates, seed 42, percentile 95% intervals;
   - valid and undefined replicate counts reported.

   With only 12 positive-containing test projects, the intervals are **approximate and conditional on
   the observed project mix**. They do not guarantee nominal coverage.
7. **Error analysis** (descriptive only): TP, FP, FN and TN per test row; probability distributions;
   per-project results. It is never used to change the model, features or threshold.
8. **Artifacts:**
   - `results/runs/2026-10-31_stage9_final_test/` is written once and refuses to overwrite.
   - The fitted pipelines go in `results/models/2026-10-31_stage9_final_test/`, with SHA-256 and metadata.

## Reporting rules

- **Kept separate in every report:**
  - development CV AP (mean of 5 folds);
  - held-out test AP;
  - development OOF threshold metrics (optimistic);
  - held-out test threshold metrics.

  Development numbers are never presented as test performance.
- **Main 40 is the primary result;** comparison 42 is the secondary feature-set comparison.
- **Known limitation of the threshold:** it was chosen on OOF scores from five fold models, but is
  applied to one model refitted on all development data. The final model's score scale can differ a
  little.
- **Weak or poor results are reported as they are.** Any rerun after seeing results is forbidden, except
  for a documented implementation bug, which would be disclosed.
