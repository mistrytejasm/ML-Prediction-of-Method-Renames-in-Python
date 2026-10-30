# ML Prediction of Method Renames in Python

Predicts whether a Python method will be renamed in a commit, using only the
code before the commit. Training data is built from the
[ActRef](https://arxiv.org/abs/2505.06553) dataset.

## Setup

1. Install Python 3.10+ and: `pip install pandas scikit-learn lightgbm openpyxl`
2. Download `commits.zip` from the [ActRef replication package](https://figshare.com/s/984c7a39266137e29c37)
   and unzip it so the commit folders sit in `data/raw_act/commits/`.

## Pipeline (run from the project folder)

| Step | Script | Output |
|---|---|---|
| 1 | `src/Act_Explore.py` | Report on the raw ActRef folders |
| 2 | `src/Labels_Renames.py` | `data/interim/actref_auto_renames.csv` |
| 3 | `src/Make_review.py` | `data/interim/confident_renames.csv`, `data/interim/review_sheet.xlsx` |
| 4 | fill in `is_rename` (Y/N), save as `review_sheet_done.xlsx` | `data/interim/review_sheet_done.xlsx` |
| 5 | `src/Build_Train_lables.py` | `data/interim/train_positives.csv` |
| 6 | `src/Build_Dataset.py` / `src/Build_Datasetv2.py` | `data/processed/train_features.csv` / `train_features_v2.csv` |
| 7 | `src/Train_Models.py` | `results/cv_results.csv`, `cv_folds.csv`, `feature_importance.csv` |
| 8 | `src/Ablation.py` / `src/Ablation_v2.py` | `results/ablation_results.csv` / `ablation_v2_results.csv` |
