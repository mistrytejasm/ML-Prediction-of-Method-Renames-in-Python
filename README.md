# ML Prediction of Method Renames in Python

Predicts whether a Python method will be renamed in a commit, using only the
code **before** the commit. Labels and features are built from the
[ActRef](https://arxiv.org/abs/2505.06553) dataset (1,159 unique commits).

**Main dataset:** `data/processed/master/actref_master_dataset.csv`:
31,209 methods, 243 renamed (0.78%), 46 Before-only features, 323 project
groups for grouped cross-validation. Read its schema
(`actref_master_dataset_schema.json`) to select feature columns.

> Label status: 76 of the 243 positive labels were reviewed by an AI first pass
> and have **not** been confirmed by a person (`meta_label_provenance =
> ai_reviewed_unconfirmed`). See `reports/audit/project_cleanup_and_dataset_preparation_report.md`.

## Project structure

```
├── README.md, requirements.txt, .gitignore
├── docs/                           project, dataset and pipeline documentation
├── data/
│   ├── raw/actref/commits/         raw ActRef Before/After code (not in git, see Setup)
│   ├── external/pyref/, icse/      other datasets (empty; never mixed into ActRef training)
│   ├── interim/
│   │   ├── rename_candidates/      detector output and auto-accepted pairs
│   │   ├── label_review/           review sheets for uncertain pairs
│   │   └── final_labels/           the 243 positive labels
│   └── processed/
│       ├── feature_tables/         original v2 feature table (unchanged)
│       └── master/                 validated master dataset + schema
├── src/
│   ├── config.py                   all paths and constants
│   ├── data/                       pipeline scripts (see below)
│   ├── features/method_features.py feature code (Before-only)
│   └── preprocessing/, models/, evaluation/   next stages (empty)
├── tests/                          unittest suite
├── reports/
│   ├── audit/                      audit outputs and the cleanup report
│   └── presentations/              proposal and progress presentations
├── results/legacy_experiments/     earlier model results (v1/v2), unchanged
└── archive/
    ├── legacy_scripts/             original scripts (paths updated only)
    └── old_datasets/               v1 feature table (biased; kept for reproducibility)
```

## Setup

1. Python 3.10+:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/macOS)
   pip install -r requirements.txt
   ```
2. Download `commits.zip` from the [ActRef replication package](https://figshare.com/s/984c7a39266137e29c37)
   and unzip it so the commit folders sit in `data/raw/actref/commits/`.

## Rebuild and validate the dataset (run from the project root)

| Step | Command | Output |
|---|---|---|
| 1 | `python src/data/inspect_raw_data.py` | `reports/audit/raw/raw_data_inventory.json` |
| 2 | `python src/data/generate_rename_labels.py` | regenerated rename candidates (for comparison) |
| 3 | `python src/data/validate_labels.py --raw` | `reports/audit/labels/` (label trace and checks) |
| 4 | `python src/data/extract_features.py` | `reports/audit/regenerated/method_features_v2_regenerated.csv` |
| 5 | `python src/data/build_master_dataset.py` | `data/processed/master/actref_master_dataset.csv` + schema |
| 6 | `python src/data/validate_dataset.py` | `reports/audit/dataset/dataset_validation_results.json` |
| 7 | `python src/data/generate_dataset_report.py` | `docs/data_dictionary.md`, `docs/data_quality_report.md` |
| Tests | `python -m unittest discover -s tests -v` | |

Scripts never overwrite the original label files or the original v2 table.
`validate_labels.py` and `validate_dataset.py` exit with code 1 if a check fails.

## Using the master dataset

```python
import json, pandas as pd
schema = json.load(open("data/processed/master/actref_master_dataset_schema.json"))
df = pd.read_csv("data/processed/master/actref_master_dataset.csv", keep_default_na=False)
X = df[schema["roles"]["feature"]]          # 46 features, Before-only
y = df["renamed"]
groups = df["meta_project_group"]           # use with StratifiedGroupKFold
```

Never train on `meta_*` columns: they describe the labelling and some contain After-commit information.

## Documentation

- `docs/project_overview.md`: goal, research questions, scope
- `docs/dataset_documentation.md`: every dataset file and how it was made
- `docs/data_pipeline.md`: scripts, inputs and outputs
- `docs/data_dictionary.md`: every column of the master dataset
- `docs/data_quality_report.md`: generated data-quality profile
- `reports/audit/project_cleanup_and_dataset_preparation_report.md`: full audit report
