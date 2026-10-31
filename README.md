# ML Prediction of Method Renames in Python

Predicts whether a Python method will be renamed in a commit, using only the
code **before** the commit. Labels and features are built from the
[ActRef](https://arxiv.org/abs/2505.06553) dataset (1,159 unique commits).

> Label status: 76 of the 243 positive labels were reviewed by an AI first pass
> and have **not** been confirmed by a person (`meta_label_provenance =
> ai_reviewed_unconfirmed`). See `reports/audit/project_cleanup_and_dataset_preparation_report.md`.

## Dataset and approved feature configurations

**Canonical master dataset:** `data/processed/master/actref_master_dataset.csv`:
31,209 methods, 243 renamed (0.78%), 46 audited Before-only features, 59 columns.
It is immutable: SHA-256 `e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0`
(checked by `src/data/load_master.py` before every load).

The master keeps all 46 features, but **models never use all 46**. The approved
training configurations are defined once, in `src/features/feature_sets.py`:

| Configuration | Features | Contents |
|---|---|---|
| **Main experiment** (`main`) | **40** | The primary experiment |
| **Comparison experiment** (`comparison`) | **42** | Main + `file_loc`, `file_n_methods`, to measure how much comes from file size |

| Never a model input | Why |
|---|---|
| `call_sites`, `rel_call_sites` | **Excluded, temporal leakage:** their search scope is the future commit's changed-file list (`docs/temporal_leakage_audit.md`) |
| `class_depth` | **Dropped:** near-duplicate of `in_class` (99.8718% equal; Pearson r = 0.996514) |
| `is_async` | **Dropped:** near-constant (37 methods, none renamed) |
| `commit`, `file`, `class`, `name`, `renamed` | Identifiers and the target |
| `meta_*` (8 columns) | Labelling provenance; some contain After-commit information |

**`meta_project_group`** (323 groups) is the grouping column for cross-validation:
split by it so that no repository appears in both training and validation/test
data. It is used to split rows, never as a feature.

Research scope: methods in files edited by the future commit; about 9.5% of
methods with unknown outcomes are excluded, and both Before and After files must parse.

### Using the master dataset

```python
from src.data.load_master import load_master, select_xy_groups

df = load_master()                                          # verifies the SHA-256, then reads (never writes)
X_main, y, groups = select_xy_groups(df, "main")            # 40 features
X_comparison, _, _ = select_xy_groups(df, "comparison")     # 42 features
# groups = df["meta_project_group"]: pass to StratifiedGroupKFold, not to the model
```

The feature lists themselves: `from src.features.feature_sets import MAIN_FEATURES, COMPARISON_FEATURES`.
`validate_feature_set()` rejects any list that contains an excluded or dropped feature, an identifier,
the target or a `meta_*` column. Rules for splitting, metrics and results: `docs/experiment_protocol.md`.

## Project structure

```
├── README.md, requirements.txt, requirements-lock.txt, .gitignore, .gitattributes
├── configs/experiments/            main_40.json, comparison_42.json (name a feature set, never list columns)
├── docs/                           project, dataset, feature and pipeline documentation, experiment protocol
├── data/
│   ├── raw/actref/commits/         raw ActRef Before/After code (not in git, see Setup)
│   ├── external/pyref/, icse/      other datasets (empty; never mixed into ActRef training)
│   ├── interim/
│   │   ├── rename_candidates/      detector output and auto-accepted pairs
│   │   ├── label_review/           review sheets for uncertain pairs
│   │   └── final_labels/           the 243 positive labels
│   └── processed/
│       ├── feature_tables/         original v2 feature table (unchanged, used by checks)
│       ├── master/                 canonical master dataset + schema (immutable)
│       └── splits/                 grouped split assignments (phase 1; keys + folds only)
├── src/
│   ├── config.py                   all paths and constants, including the master checksum
│   ├── data/                       dataset generation, validation, loading and documentation generators
│   ├── features/
│   │   ├── method_features.py      feature code (Before-only)
│   │   └── feature_sets.py         approved 40 / 42 feature configurations (single source of truth)
│   ├── preprocessing/              phase 1 (not implemented yet)
│   ├── evaluation/                 splits and metrics (not implemented yet)
│   └── models/                     baselines and models (not implemented yet)
├── tests/                          unittest suite
├── reports/
│   ├── audit/                      audit outputs, leakage evidence and the cleanup report
│   └── presentations/              proposal and progress presentations
├── results/
│   ├── runs/                       one folder per experiment run (training phase)
│   └── legacy_experiments/         earlier model results (v1/v2), unchanged
└── archive/
    ├── legacy_scripts/             original scripts (paths updated only)
    └── old_datasets/               v1 feature table (biased; kept for reproducibility)
```

## Setup

1. Python 3.12 (reference environment). Exact versions:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)   |   source .venv/bin/activate   (Linux/macOS)
   pip install -r requirements-lock.txt
   ```
   `requirements.txt` declares what the project needs (minimum versions); `requirements-lock.txt`
   pins the exact versions the results were produced with.
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
| 7 | `python src/data/generate_dataset_report.py` | `docs/data_dictionary.md/.csv`, `docs/data_quality_report.md` |
| 8 | `python src/data/generate_feature_mapping.py` | `docs/feature_dataset_mapping.md` (`--check` compares without writing) |
| 9 | `python src/data/audit_temporal_leakage.py` | `reports/audit/leakage/temporal_leakage_evidence.json` (`--check` verifies the numbers cited in `docs/temporal_leakage_audit.md`) |
| Check | `python src/data/load_master.py` | verifies the master checksum and the 40 / 42 feature sets |
| Tests | `python -m unittest discover -s tests -v` | set `FULL_REGENERATION=1` to also run the slow raw-data checks |

Scripts never overwrite the original label files or the original v2 table.
`validate_labels.py`, `validate_dataset.py` and the `--check` modes exit with code 1 if a check fails.

## Documentation

- `docs/project_overview.md`: goal, research questions, scope
- `docs/experiment_protocol.md`: data, feature, splitting, metric and result rules for the next phases
- `docs/feature_dataset_mapping.md`: every feature, the 40 / 42 lists and their verification
- `docs/temporal_leakage_audit.md`: why `call_sites` and `rel_call_sites` are excluded
- `docs/dataset_documentation.md`: every dataset file and how it was made
- `docs/data_pipeline.md`: scripts, inputs and outputs
- `docs/data_dictionary.md`: every column of the master dataset
- `docs/data_quality_report.md`: generated data-quality profile
- `reports/audit/project_cleanup_and_dataset_preparation_report.md`: full dataset audit report
