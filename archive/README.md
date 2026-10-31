# Archive

Kept so that earlier results stay reproducible. Nothing here is deleted.

## legacy_scripts/

The original pipeline, moved from `src/` on 2026-10-31. Only the file paths inside
each script were updated for the new folder layout; the logic is unchanged. They
need the raw data in `data/raw/actref/commits/`.

| Script | Original name | What it does |
|---|---|---|
| `step1_explore_raw_actref.py` | `Act_Explore.py` | Explores the raw download |
| `step2_detect_rename_candidates.py` | `Labels_Renames.py` | Detects rename candidates |
| `step3_split_confident_and_review.py` | `Make_review.py` | Splits auto-accepted / review pairs. **Overwrites** both label files if run |
| `step4_build_positive_labels.py` | `Build_Train_lables.py` | Builds the 243 positives. **Overwrites** `rename_positive_labels.csv` if run |
| `step5a_build_features_v1.py` | `Build_Dataset.py` | Builds the v1 table |
| `step5b_build_features_v2.py` | `Build_Datasetv2.py` | Builds the v2 table |
| `step6_train_models_v1.py` | `Train_Models.py` | Trains models on v1 |
| `step7a_ablation_v1.py` | `Ablation.py` | Feature-group ablation on v1 |
| `step7b_ablation_v2.py` | `Ablation_v2.py` | Feature-group ablation on v2 |

Replacements in `src/data/`: `inspect_raw_data.py` (step 1), `generate_rename_labels.py` (step 2),
`validate_labels.py` (checks steps 3–4 instead of re-running them), `extract_features.py` (step 5b).
Steps 3 and 4 depend on the review decisions, so they are verified rather than regenerated.

## old_datasets/

`method_features_v1_rename_files_only.csv` (was `data/processed/train_features.csv`): the v1
table. Its negatives come only from files containing a rename, which makes file size an artificial
signal. Do not use it for new experiments.
