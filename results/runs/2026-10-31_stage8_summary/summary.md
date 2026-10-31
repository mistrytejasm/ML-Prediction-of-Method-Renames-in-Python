# Stage 8 development cross-validation: summary

Development rows only (24,967 rows, 261 projects, 5 saved grouped folds). Held-out test rows were not used.

- CV AP = mean and sample std (ddof=1) of the 5 validation-fold APs (primary metric).
- dev_*_optimistic: optimistic: measured on the same pooled development OOF scores that were used to select the threshold; not a performance estimate.

## Main 40 (primary)

| run_id | cv_ap_mean | cv_ap_std_ddof1 | roc_auc_fold_mean | threshold | dev_precision_optimistic | dev_recall_optimistic | dev_f1_optimistic | dev_mcc_optimistic |
|---|---|---|---|---|---|---|---|---|
| dummy_prior__main | 0.0077 | 0.0003 | 0.5000 | 0.0076 | 0.0077 | 1.0000 | 0.0153 | nan |
| stratified_random__main | 0.0090 | 0.0015 | 0.5076 | 0.0108 | 0.0092 | 0.3523 | 0.0180 | 0.0110 |
| logistic_regression__main__log1p_asinh | 0.0130 | 0.0042 | 0.6224 | 0.8654 | 0.0193 | 0.0570 | 0.0288 | 0.0201 |
| logistic_regression__main__yeo_johnson | 0.0133 | 0.0035 | 0.6313 | 0.8594 | 0.0206 | 0.0570 | 0.0303 | 0.0218 |
| random_forest__main | 0.0114 | 0.0033 | 0.5527 | 0.0319 | 0.0097 | 0.2383 | 0.0186 | 0.0109 |
| lightgbm__main | 0.0103 | 0.0034 | 0.4919 | 0.0334 | 0.0100 | 0.0881 | 0.0180 | 0.0071 |

## Comparison 42 (secondary)

| run_id | cv_ap_mean | cv_ap_std_ddof1 | roc_auc_fold_mean | threshold | dev_precision_optimistic | dev_recall_optimistic | dev_f1_optimistic | dev_mcc_optimistic |
|---|---|---|---|---|---|---|---|---|
| dummy_prior__comparison | 0.0077 | 0.0003 | 0.5000 | 0.0076 | 0.0077 | 1.0000 | 0.0153 | nan |
| stratified_random__comparison | 0.0090 | 0.0015 | 0.5076 | 0.0108 | 0.0092 | 0.3523 | 0.0180 | 0.0110 |
| logistic_regression__comparison__log1p_asinh | 0.0188 | 0.0105 | 0.6736 | 0.8368 | 0.0224 | 0.1244 | 0.0380 | 0.0355 |
| logistic_regression__comparison__yeo_johnson | 0.0191 | 0.0089 | 0.6814 | 0.8459 | 0.0254 | 0.1244 | 0.0421 | 0.0400 |
| random_forest__comparison | 0.0135 | 0.0040 | 0.5936 | 0.0640 | 0.0153 | 0.1088 | 0.0268 | 0.0208 |
| lightgbm__comparison | 0.0112 | 0.0018 | 0.5616 | 0.0220 | 0.0140 | 0.1347 | 0.0254 | 0.0204 |

## Comparison 42 minus Main 40 (CV AP mean)

| model | transform | main | comparison | difference |
|---|---|---|---|---|
| dummy_prior | none | 0.0077 | 0.0077 | +0.0000 |
| lightgbm | none | 0.0103 | 0.0112 | +0.0009 |
| logistic_regression | log1p_asinh | 0.0130 | 0.0188 | +0.0058 |
| logistic_regression | yeo_johnson | 0.0133 | 0.0191 | +0.0059 |
| random_forest | none | 0.0114 | 0.0135 | +0.0021 |
| stratified_random | none | 0.0090 | 0.0090 | +0.0000 |

## Exploratory, post-hoc: naming heuristic (not ranked with the models above)

| run_id | cv_ap_mean | cv_ap_std_ddof1 | roc_auc_fold_mean | threshold | dev_precision_optimistic | dev_recall_optimistic | dev_f1_optimistic | dev_mcc_optimistic |
|---|---|---|---|---|---|---|---|---|
| naming_heuristic__main | 0.0097 | 0.0032 | 0.5159 | 0.3333 | 0.0112 | 0.2021 | 0.0213 | 0.0161 |
| naming_heuristic__comparison | 0.0097 | 0.0032 | 0.5159 | 0.3333 | 0.0112 | 0.2021 | 0.0213 | 0.0161 |

Its flags were proposed after full-dataset label-derived summaries had been seen (docs/naming_heuristic.md); it is not a confirmatory baseline and cannot be selected for Stage 9.

## Stage 9 candidates (rule fixed before the results)

- main: `logistic_regression__main__yeo_johnson`
- comparison: `logistic_regression__comparison__yeo_johnson`

Baselines identical for main and comparison: dummy_prior, stratified_random, naming_heuristic
