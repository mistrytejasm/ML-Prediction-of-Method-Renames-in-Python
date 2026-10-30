# Data pipeline

All scripts read paths from `src/config.py` and run from the project root.

```
data/raw/actref/commits/  (1,160 folders; 1 byte-identical duplicate is skipped)
        │
        │ generate_rename_labels.py      same class, old name gone, new name added,
        ▼                                body similarity >= 0.70, one-to-one matching
interim/rename_candidates/detected_rename_candidates.csv          430
        │ drop 96 test methods
        ├──► similarity >= 0.90 and same args ──► auto_accepted_rename_candidates.csv   183 (+2 removed by hand)
        └──► the rest ──► label_review/uncertain_candidates_review_blank.xlsx           149
                              │ AI first-pass review (Y/N), not yet confirmed by a person
                              ▼
                         label_review/uncertain_candidates_review_ai_completed.xlsx     118 Y, 31 N
        │ pool = 183 + 118 = 301, cap 5 per commit (seed 42) drops 58
        ▼
interim/final_labels/rename_positive_labels.csv                    243
        │
        │ extract_features.py    every non-test, non-dunder Before method:
        │                        1 = in the positive list, 0 = class+name still in After,
        │                        unknown (3,263 methods) = skipped
        ▼
reports/audit/regenerated/method_features_v2_regenerated.csv       31,209 × 51
        │ build_master_dataset.py   + meta_* provenance columns, sorted by key
        ▼
data/processed/master/actref_master_dataset.csv                    31,209 × 59
        │ validate_dataset.py, generate_dataset_report.py
        ▼
reports/audit/dataset/, docs/data_dictionary.md, docs/data_quality_report.md
```

## Scripts

| Script | Reads | Writes | Notes |
|---|---|---|---|
| `src/data/inspect_raw_data.py` | raw commits | `reports/audit/raw/raw_data_inventory.json` | Read-only inventory + SHA-256 fingerprint of the raw tree |
| `src/data/generate_rename_labels.py` | raw commits | `reports/audit/regenerated/detected_rename_candidates_regenerated.csv` | Reproduces `detected_rename_candidates.csv` exactly; refuses to overwrite it without `--force` |
| `src/data/validate_labels.py` | all label files, v2, raw (`--raw`) | `reports/audit/labels/` | Traces all 430 candidates; reproduces the cap; checks every label in the raw code |
| `src/data/extract_features.py` | raw commits, positive labels | `reports/audit/regenerated/method_features_v2_regenerated.csv` | Before-only features; `--legacy-v2` reproduces the original v2 exactly |
| `src/data/build_master_dataset.py` | regenerated features, positives, review sheet | master CSV + schema JSON | No rows added; provenance kept in `meta_*` |
| `src/data/validate_dataset.py` | master, schema, positives, v2 | `reports/audit/dataset/dataset_validation_results.json` | 32 checks; exit code 1 on failure |
| `src/data/generate_dataset_report.py` | every dataset file | `docs/data_dictionary.*`, `docs/data_quality_report.md`, `reports/audit/dataset/*.csv` | Documents are generated from the data |
| `src/data/label_utils.py` | | | Rename type, provenance and project-group helpers |
| `src/data/raw_commits.py` | | | Lists commit folders, skips byte-identical duplicates |
| `src/features/method_features.py` | | | Feature code, copied unchanged from the original v2 builder |

## Differences from the original pipeline

| Change | Why | Effect |
|---|---|---|
| Skip byte-identical duplicate commit folders | `dd7f9fa8…_<garbled suffix>` is a copy of `dd7f9fa8…` | 8 duplicate negative rows removed |
| `call_sites` searches every parseable Before file | v2 skipped Before files that had no After pair, so the feature depended on the After code | 306 values changed |
| `rel_*` averages over all eligible Before methods | v2 averaged only over methods whose label was known, so deleting a method in the After code changed its siblings' features | 7,399 rows changed (104 positives) |
| Files processed in sorted order | v2 iterated a Python set, so row order varied between runs | Row order is reproducible; values unchanged |
| Outputs go to new files | The original scripts overwrote label files | Original files cannot be overwritten by accident |

The original scripts are kept in `archive/legacy_scripts/` (renamed `step1_…` to `step7b_…`, with only their file paths updated).
