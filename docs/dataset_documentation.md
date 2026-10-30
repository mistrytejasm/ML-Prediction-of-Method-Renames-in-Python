# Dataset documentation

Every dataset file in the project, with its old name, what one row means, and
whether it is used for training. Contents of all original files are unchanged
(SHA-256 verified, see `reports/audit/file_moves_log.csv`); only names and
folders changed. Column-level detail: `docs/data_dictionary.md` and
`reports/audit/dataset/dataset_profiles.csv`.

## Summary

| File | Old name | Rows × cols | One row is | Use |
|---|---|---|---|---|
| `data/raw/actref/commits/` | `data/raw_act/commits/` | 1,160 folders, 10,668 `.py` | Before/After copies of the files changed by a commit | Source of everything |
| `data/interim/rename_candidates/detected_rename_candidates.csv` | `actref_auto_renames.csv` | 430 × 9 | A detected rename pair | Label building |
| `data/interim/rename_candidates/auto_accepted_rename_candidates.csv` | `confident_renames.csv` | 183 × 10 | A pair accepted without review | Label building |
| `data/interim/label_review/uncertain_candidates_review_blank.xlsx` | `review_sheet.xlsx` | 149 × 11 | An uncertain pair with old/new code | Label building |
| `data/interim/label_review/uncertain_candidates_review_ai_completed.xlsx` | `review_sheet_done.xlsx` | 149 × 12 | The same pair with an AI Y/N decision | Label building |
| `data/interim/final_labels/rename_positive_labels.csv` | `train_positives.csv` | 243 × 8 | A final positive label | Label source for features |
| `data/processed/feature_tables/method_features_v2_all_commits.csv` | `train_features_v2.csv` | 31,217 × 51 | A method + 46 features | Historical; superseded by the master |
| `data/processed/master/actref_master_dataset.csv` | (new) | 31,209 × 59 | A method + 46 features + provenance | **Training dataset** |
| `archive/old_datasets/method_features_v1_rename_files_only.csv` | `train_features.csv` | 2,158 × 39 | A method + 34 features | Do not use (sampling bias) |

## Raw ActRef commits

- 1,160 folders but **1,159 unique commits**: `dd7f9fa8b43d42edf20483b30ff7d4591270eaf6_<garbled suffix>`
  is a byte-identical copy of `dd7f9fa8…`. The pipeline skips it; the folder itself is left untouched.
- 5,334 Before and 5,334 After `.py` files; 589 files exist only on one side (deleted, added or moved).
- 5 files do not parse as Python; 16 commits have no usable Before/After pair.
- Leftover files: 1,164 `.DS_Store` and SonarQube files (`.sonar_lock`, `report-task.txt`) in 3 commits. They are ignored.

## Label files

**detected_rename_candidates.csv**: `commit, file, class, old_name, new_name, similarity, same_args, old_lineno, old_loc`.
The rule: in the same file and class, a name disappears, a new one appears, and the
bodies are ≥ 70% similar. Pairs are matched one-to-one. The file is reproduced
exactly by `generate_rename_labels.py`. 96 rows are test methods.

**auto_accepted_rename_candidates.csv**: the same columns + `confident`. The rule
(similarity ≥ 0.9 and same args, not a test method) gives 185 pairs; the file
has 183. Two were removed by hand, not by code: `_on_train_end → _on_train_begin` and
`get_verbosity → get_validation_split` (the names mean different things).

**uncertain_candidates_review_*.xlsx**: `old_code` and `new_code` hold up to 1,500 characters
of each method. In the completed sheet, every `note` starts with "Claude first guess" (137)
or "Decided by Claude" (12), and `confirmed_by_human` is empty in all 149 rows.

**rename_positive_labels.csv**: `commit, file, class, old_name, new_name, similarity, label_source, renamed`.
301 candidates were capped to at most 5 per commit with seed 42, which drops 58 (39 of them from the
bulk `host_` commit `8ceeb6ed…`). Re-running the cap reproduces the file exactly.
`label_source = hand_checked` (76 rows) is inaccurate: those rows were AI-reviewed only.
The master dataset records them as `meta_label_provenance = ai_reviewed_unconfirmed`;
the original file is not changed.

## Feature tables

**v1, method_features_v1_rename_files_only.csv**: negatives taken only from files that contain a
rename. File size then separates the classes artificially. Archived.

**v2, method_features_v2_all_commits.csv**: negatives from all commits. Reproduced exactly from the raw
data (`extract_features.py --legacy-v2`). Three problems found:
1. 8 duplicate rows from the duplicate commit folder.
2. `call_sites` ignored Before files without an After pair (After-dependent).
3. `rel_*` averaged only over methods whose label was known (After-dependent).

**Master, actref_master_dataset.csv**: v2 with the three problems fixed in the generation code, plus
`meta_*` columns:

| Group | Columns |
|---|---|
| Identifiers | `commit, file, class, name` |
| Target | `renamed` (243 = 1, 30,966 = 0, 0.78%) |
| Features (46) | 9 name, 17 code, 8 context, 12 relative (see data dictionary) |
| Metadata (8, never train on) | `meta_label_source, meta_label_provenance, meta_label_similarity, meta_rename_type, meta_new_name, meta_project_guess, meta_project_group, meta_duplicate_definition_in_before` |

The schema file `actref_master_dataset_schema.json` lists the column roles, counts and the SHA-256 of
its inputs and of the master file itself.

## Rename types of the 243 positives (`meta_rename_type`)

| Type | Count | Example |
|---|---|---|
| word_replaced | 66 | `create_continuous_observation_encoder → create_vector_observation_encoder` |
| word_added | 48 | `grad_of → insert_grad_of` |
| visibility_only | 43 | `init_model → _init_model` |
| restructured | 28 | `showXmlBoundingBoxes → showBoundingBoxFromAnnotationFile` |
| word_removed | 22 | `bbox_encode → encode` |
| spelling_or_inflection | 19 | `euclidian_distance → euclidean_distance` |
| case_or_style_only | 17 | `subsFactor → subs_factor` |

The classification uses only the old and new names (`src/data/label_utils.py`), so it is descriptive
metadata and must not be used as a feature.
