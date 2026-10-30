# Project overview: ML Prediction of Method Renames in Python

## Goal

Predict, from the code **before** a commit, whether a Python method will be
renamed in that commit (Rename Method refactoring). A working predictor could
point developers to names that are likely to change.

## Prediction task

| | |
|---|---|
| Unit | One method in a Before file of an ActRef commit |
| Target | `renamed` = 1 if renamed in the commit, 0 if the same class + name still exists after it |
| Inputs | 46 features computed from the Before code only (name, code, context, relative-to-file) |
| Not predicted | What the new name is |
| Excluded | Test methods (`test*`), dunder methods (`__init__`, ...), methods whose label is unknown (deleted, moved, or renamed in a way the detector could not confirm) |

## Data

- **ActRef** (Wang et al., 2025): Before/After code of 1,160 commit folders (1,159 unique commits)
  from Python ML projects. The download has no labels, so labels are built by
  this project (see `dataset_documentation.md`).
- **PyRef** (Atwi et al., SCAM 2021) and **ICSE 2022** rename data: planned as external
  checks; their folders in `data/external/` are empty and are never mixed into ActRef training.

## Current state (2026-10-31)

- The master dataset is built and validated (31,209 rows, 243 positives).
- 76 positive labels are AI-reviewed and still need human confirmation.
- No models have been trained on the master dataset yet. Earlier results in
  `results/legacy_experiments/` come from v1/v2 and are kept for reference only.

## Next stages

Preprocessing → feature engineering → feature selection → grouped
train/validation/test split → class-imbalance experiments → baselines →
models → evaluation. See section H of
`reports/audit/project_cleanup_and_dataset_preparation_report.md`.
