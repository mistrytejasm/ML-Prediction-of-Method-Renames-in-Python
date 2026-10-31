# Project cleanup and dataset preparation report

Project: **ML Prediction of Method Renames in Python** · Branch `feature/actref-data-audit-and-master-dataset` · 2026-10-31

**Result:** one validated master dataset, `data/processed/master/actref_master_dataset.csv`,
with 31,209 methods, 243 positives (0.78%), 46 Before-only features and 323 project groups.
Every number below was produced by the scripts in `src/data/` and can be regenerated.
The dataset is **ready for preprocessing**, but **not fully research-ready**: 76 of the
243 positive labels have only an AI review and no human confirmation.

Three defects in the original v2 table were found and fixed in the generation code:
one duplicated commit folder, and two features (`call_sites`, `rel_*`) that leaked
After-commit information. No model was trained, and no existing dataset or Excel file was edited.

---

## A. Initial project assessment

### Original structure (before this branch)

```
Refactoring_Project/
├── 4 presentation files (.pdf, .pptx) in the root
├── data/raw_act/commits/        1,160 folders
├── data/raw_py/, data/raw_Icse/ empty
├── data/interim/                actref_auto_renames.csv, confident_renames.csv,
│                                review_sheet.xlsx, review_sheet_done.xlsx, train_positives.csv
├── data/processed/              train_features.csv (v1), train_features_v2.csv (v2)
├── results/                     cv_results, cv_folds, feature_importance, ablation_results, ablation_v2_results
└── src/                         9 scripts
```

### Script → data dependencies (verified by reading the code)

| Script | Reads | Writes |
|---|---|---|
| `Act_Explore.py` | raw commits | console only |
| `Labels_Renames.py` | raw commits, optional `pyref_rename_method.csv` | `actref_auto_renames.csv` |
| `Make_review.py` | `actref_auto_renames.csv`, raw commits | `confident_renames.csv`, `review_sheet.xlsx` (overwrites both) |
| `Build_Train_lables.py` | `confident_renames.csv`, `review_sheet_done.xlsx` | `train_positives.csv` (overwrites) |
| `Build_Dataset.py` | raw commits, `train_positives.csv` | `train_features.csv` (v1) |
| `Build_Datasetv2.py` | raw commits, `train_positives.csv` | `train_features_v2.csv` (v2) |
| `Train_Models.py` | **v1** | `cv_results.csv`, `cv_folds.csv`, `feature_importance.csv` |
| `Ablation.py` / `Ablation_v2.py` | v1 / v2 | `ablation_results.csv` / `ablation_v2_results.csv` |

### Inconsistencies between what scripts claim and what they do

| # | Finding |
|---|---|
| 1 | Run commands in the docstrings name files that do not exist (`Label_Renames.py`, `Make_Review_Sheet.py`, `Build_Train_Labels.py`); `Build_Datasetv2.py`'s docstring calls itself `Build_Dataset.py`. |
| 2 | `Make_review.py` produces 185 confident pairs; `confident_renames.csv` has 183. Two pairs were removed by hand, so re-running the script silently brings them back. |
| 3 | `Build_Train_lables.py` labels reviewed rows `hand_checked`, but every review decision is an AI first guess and `confirmed_by_human` is empty. |
| 4 | `Train_Models.py` trains on v1, not on v2. |
| 5 | `Build_Datasetv2.py` iterates over a Python set, so the row order changes between runs (values do not). |
| 6 | The v2 docstring says features come from the Before code only. Two features (`call_sites`, `rel_*`) depended on the After code (section D). |
| 7 | `Ablation*.py` put `name_body_overlap` and `style_mismatch` in the NAME group, while the builders treat them as code / context features. |
| 8 | The project grouping in `Train_Models.py` splits some repositories across groups (section H.4). |
| 9 | The project documentation mentions `pyref_rename_method.csv` (139 rows); the file is not in the project. |
| 10 | Three scripts have invalid escape sequences in their docstrings (`SyntaxWarning`, harmless). |

---

## B. Dataset audit

Full per-column profile: `reports/audit/dataset/dataset_profiles.csv`. Generated summary: `docs/data_quality_report.md`.

| File (new name) | Expected | Actual | Duplicate keys | Empty cells | Labels | Issues |
|---|---|---|---|---|---|---|
| `detected_rename_candidates.csv` | 430 × 9 | 430 × 9 | 0 | `class` only (133) | – | 96 test methods (dropped later); reproduced exactly from raw |
| `auto_accepted_rename_candidates.csv` | 183 | 183 × 10 | 0 | `class` only | – | Rule gives 185; 2 removed by hand |
| `uncertain_candidates_review_blank.xlsx` | 149 | 149 × 11 | 0 | review columns | – | Matches the rule exactly |
| `uncertain_candidates_review_ai_completed.xlsx` | 149, 118 Y / 31 N | 149 × 12, 118 Y / 31 N | 0 | `confirmed_by_human` empty in all 149 | Y/N only | All decisions AI-made |
| `rename_positive_labels.csv` | 243 (167 + 76) | 243 × 8 | 0 | `class` only | all 1 | `hand_checked` is inaccurate |
| v1 `method_features_v1_rename_files_only.csv` | 2,158 × 39 | 2,158 × 39 | 0 | `class` only | 243 / 1,915 | Sampling bias; `is_async` constant |
| v2 `method_features_v2_all_commits.csv` | 31,217 × 51 | 31,217 × 51 | 0 | `class` only | 243 / 30,974 | 8 duplicate rows; 2 After-dependent features |
| **master** `actref_master_dataset.csv` | – | **31,209 × 59** | 0 | `class` + by-design `meta_*` | **243 / 30,966** | 2 warnings (section G) |

No file has missing values outside `class` (empty = module-level function) and
the review columns. No feature has infinite values or values outside its
valid range. The only invalid identifiers were the 8 rows of the duplicate
commit folder, whose name is not a commit hash.

### Raw data

`reports/audit/raw/raw_data_inventory.json`: 1,160 folders, **1,159 unique commits**, 11,842 files
(10,668 `.py`, 1,164 `.DS_Store`, 10 SonarQube leftovers). 5 unparseable `.py` files. 589 files exist
on only one side. 16 commits have no usable Before/After pair. All 11,842 files have the same
SHA-256 before and after the move.

### Regeneration

| Output | Regenerated from raw | Result |
|---|---|---|
| `detected_rename_candidates.csv` | `generate_rename_labels.py` | identical (430 / 430 rows) |
| v2 table | `extract_features.py --legacy-v2` | identical (31,217 rows × 51 columns, max difference 0.0) |
| `rename_positive_labels.csv` | cap re-applied by `validate_labels.py` | identical (0 differing keys) |

---

## C. Label quality report

### Trace of all 430 candidates (`reports/audit/labels/candidate_label_trace.csv`)

| Outcome | Count |
|---|---|
| final_positive | 243 |
| dropped_test_method | 96 |
| dropped_by_commit_cap | 58 |
| rejected_in_review (N) | 31 |
| dropped_from_auto_accepted_by_hand | 2 |
| unexplained | **0** |

### Reconciliation with the feature tables

- All 243 positives appear exactly once in v2 and in the master (key `commit + file + class + name`); 0 missing, 0 extra.
- None of the 187 non-positive candidates (rejected, capped, test, hand-removed) appears as a negative.
  Their old name no longer exists in the After code, so the builder skips them.
- Raw re-check (`validate_labels.py --raw`):
  - **Positives:** all 243 exist in Before, their old name is gone in After, and their new name exists.
  - **Negatives:** all 30,966 keep their class + name in After. 26,802 are unchanged and 4,172 changed body or signature.
    Those are still correct negatives: the label means "name kept", not "code unchanged".

### AI-reviewed versus human-confirmed

| Provenance (new `meta_label_provenance`) | Original `label_source` | Count |
|---|---|---|
| auto_rule_confident (similarity ≥ 0.9, same args) | auto_confident | 167 |
| ai_reviewed_unconfirmed | hand_checked | 76 |
| human_confirmed | – | **0** |

`label_source` is kept unchanged in `meta_label_source`. The 31 rejected candidates are also AI decisions,
so some real renames may be missing from the positives.

### Rename categories (positives, `meta_rename_type`)

| Type | Positives | of which AI-reviewed | Meaning for the research |
|---|---|---|---|
| word_replaced | 66 | 24 | Terminology change; core naming signal |
| word_added | 48 | 22 | Name made more specific |
| visibility_only | 43 | 7 | `_` added/removed: an API visibility change, not a naming-quality change |
| restructured | 28 | 12 | Several words changed |
| word_removed | 22 | 6 | Name shortened |
| spelling_or_inflection | 19 | 3 | Typo fixed, or plural/singular changed |
| case_or_style_only | 17 | 2 | Case or separator change (`subsFactor → subs_factor`) |

**Recommendation:** keep all 243 positives in the master (no category is removed). In every
experiment, also report results with the 60 visibility- and case-only renames excluded
(183 positives). If the results differ, then "predicting a rename" mixes two tasks: visibility
refactoring and improving names.

### Five-per-commit cap

58 candidates from 7 commits were dropped, 39 of them from the bulk `host_` commit `8ceeb6ed…`;
the list is in `excluded_by_commit_cap.csv`. 11 commits sit at the cap. The capped candidates
are real renames that are labelled neither 1 nor 0. That is correct (no false negatives), but
they are lost positives.

### Other findings

- 308 rows (1 positive) have a class + name defined twice in the Before file (e.g. a property getter + setter);
  the features describe the last definition. They are flagged in `meta_duplicate_definition_in_before`.
- 3,263 eligible Before methods are excluded because their label is unknown (deleted, moved, or renamed
  below the 0.70 similarity threshold or across classes). That is about 9.5% of the eligible methods:
  a selection effect to state in the report.

---

## D. Feature audit

The dictionary for all 46 features (meaning, calculation, Before-only status, missing values, notes) is in `docs/data_dictionary.md` / `.csv`.

| Group | Count | Features |
|---|---|---|
| Name | 9 | name_len, name_tokens, name_is_private, name_is_camel, name_has_digit, name_short_tokens, name_no_vowel_tokens, name_starts_verb, name_single_char |
| Code | 17 | loc, n_params, n_defaults, has_varargs, has_kwargs, n_returns, n_calls, cyclomatic, max_nesting, has_docstring, doc_len, is_stub, is_async, n_decorators, is_property, is_static_or_class, name_body_overlap |
| Context | 8 | in_class, class_depth, class_n_methods, file_n_methods, file_loc, is_test_file, style_mismatch, call_sites |
| Relative | 12 | name_token_sharing, first_token_shared, style_share_in_file, rel_name_len, rel_name_tokens, rel_loc, rel_n_params, rel_n_calls, rel_cyclomatic, rel_doc_len, rel_call_sites, rel_name_body_overlap |

All 46 features are numeric (int64 or float64), with no missing or infinite values and no categorical columns.

### Leakage audit

| Candidate leak | Status |
|---|---|
| `new_name`, After code, After features | Not features. `new_name` is kept only as `meta_new_name` |
| Label similarity, `label_source`, `same_args` | Not features (only in `meta_*`) |
| `name_body_overlap` | Before only: computed from the Before function node |
| **`call_sites`** | **Leaked in v2:** only Before files that also exist in After were searched. **Fixed:** every parseable Before file is searched (306 values changed) |
| **`rel_*` (9 features)** | **Leaked in v2:** the file average used only methods whose label was known, so methods deleted in After changed their siblings' values. **Fixed:** averaged over all eligible Before methods. 7,399 rows changed, including **104 of 243 positives (43%) against 24% of all rows** |
| Row selection | Rows are kept only when the label is known (1, or same name in After). This is a property of the task definition, not a feature leak. It is stated as a limitation. |

Measured effect of the fix: the standardized difference between class means changed by at most
0.024 per feature (e.g. `rel_name_tokens` 0.284 → 0.266, `rel_n_params` −0.069 → −0.093). So the leak
was real but small. A unit test (`test_features_use_only_before_code`) now fails if any feature
changes when only the After code changes.

### Constant and correlated features

No constant features (`is_async` was constant only in v1). Pairs with |r| ≥ 0.8
(`reports/audit/dataset/feature_correlations.csv`):

| Pair | r |
|---|---|
| in_class / class_depth | 0.997 |
| file_n_methods / file_loc | 0.879 |
| cyclomatic / rel_cyclomatic | 0.859 |
| name_token_sharing / first_token_shared | 0.849 |
| name_len / name_tokens | 0.848 |
| name_body_overlap / rel_name_body_overlap | 0.848 |
| rel_name_len / rel_name_tokens | 0.828 |
| rel_loc / rel_n_calls | 0.812 |
| loc / n_calls | 0.807 |

`doc_len`, `file_loc`, `loc`, `n_calls` and `call_sites` have long right tails (max far above the 99th percentile).

### Recommended features for future experiments

Keep all 46 for the first baselines. Then:
- Drop one of `in_class` / `class_depth` for linear models.
- Log-transform the long-tailed counts, fitted inside the training folds.
- Run every model with and without `file_loc` and `file_n_methods`. Renamed methods still sit in smaller files
  (median 206 vs 454 lines), and v1 showed this can be a sampling artefact.

Feature selection should come from those experiments, not from correlation with the target.

---

## E. Project restructuring

### New structure

```
Refactoring_Project/
├── README.md, requirements.txt, .gitignore
├── docs/          project_overview, dataset_documentation, data_pipeline, data_dictionary(.md/.csv), data_quality_report
├── data/
│   ├── raw/actref/commits/                    (not in git)
│   ├── external/pyref/, external/icse/        (empty)
│   ├── interim/rename_candidates/             detected_rename_candidates.csv, auto_accepted_rename_candidates.csv
│   ├── interim/label_review/                  uncertain_candidates_review_blank.xlsx, uncertain_candidates_review_ai_completed.xlsx
│   ├── interim/final_labels/                  rename_positive_labels.csv
│   ├── processed/feature_tables/              method_features_v2_all_commits.csv
│   └── processed/master/                      actref_master_dataset.csv, actref_master_dataset_schema.json
├── src/config.py, src/data/ (8 modules), src/features/method_features.py, src/preprocessing/, src/models/, src/evaluation/
├── tests/         test_feature_extraction.py, test_master_dataset.py, test_regeneration.py
├── reports/audit/ (this report, file_moves_log.csv, raw/, labels/, dataset/), reports/presentations/
├── results/legacy_experiments/   v1_cv_summary_by_model.csv, v1_cv_scores_per_fold.csv, v1_feature_importance.csv,
│                                 v1_ablation_by_feature_group.csv, v2_ablation_by_feature_group.csv
└── archive/       README.md, legacy_scripts/ (step1_… to step7b_…), old_datasets/method_features_v1_rename_files_only.csv
```

### Files moved, renamed, archived

Every file's original path, final path and SHA-256 are in `reports/audit/file_moves_log.csv`.
- All 16 dataset, Excel, result and presentation files are **byte-identical** to the originals: only names and folders changed.
- The 9 scripts were archived and renamed. Only their file paths were edited.
- The raw folder was renamed `data/raw_act` → `data/raw/actref`, with its contents verified unchanged.
- Nothing was deleted.

### Files generated

| File | By |
|---|---|
| `data/processed/master/actref_master_dataset.csv` + `_schema.json` | build_master_dataset.py |
| `reports/audit/raw/raw_data_inventory.json` | inspect_raw_data.py |
| `reports/audit/labels/*` (results JSON, candidate trace, positives with metadata, cap exclusions, auto-accept mismatches, negative body changes) | validate_labels.py |
| `reports/audit/dataset/*` (validation JSON, profiles, feature statistics, correlations) | validate_dataset.py, generate_dataset_report.py |
| `docs/data_dictionary.md/.csv`, `docs/data_quality_report.md` | generate_dataset_report.py |
| `reports/audit/regenerated/*` (not in git; reproducible) | extract_features.py, generate_rename_labels.py |

### Not done: these need your approval

1. Deleting the duplicate raw folder `dd7f9fa8…_<suffix>`. It is skipped, not deleted.
2. Writing human confirmations into the review Excel file. Excel contents are not edited, as you instructed.
3. Restoring the 2 hand-removed pairs, or making their removal part of the code.

---

## F. Master dataset report

| Item | Value |
|---|---|
| Location | `data/processed/master/actref_master_dataset.csv` (+ `actref_master_dataset_schema.json`) |
| Rows × columns | 31,209 × 59 |
| Positives / negatives | 243 / 30,966 (positive rate 0.778%) |
| Commits / project groups | 1,063 / 323 (79 groups contain positives; largest group keras, 2,711 rows) |
| Identifier columns | commit, file, class, name |
| Target | renamed |
| Predictive features | 46, all Before-only, numeric, no missing / infinite values |
| Provenance columns (never train on) | meta_label_source, meta_label_provenance, meta_label_similarity, meta_rename_type, meta_new_name, meta_project_guess, meta_project_group, meta_duplicate_definition_in_before |
| Duplicates | 0 duplicate keys. 1,397 rows (578 clusters) repeat the same method with the same features in another commit; every cluster stays within one project group |
| Positive reconciliation | 243 / 243, each exactly once |
| Changes from v2 | −8 duplicate rows; `call_sites` and `rel_*` recomputed Before-only; nothing else differs (checked) |
| Validation | 32 checks: 30 pass, 2 warnings, 0 failures. 16 unit tests pass |
| Ready for preprocessing | **Yes**, subject to the label caveat in G |

---

## G. Remaining problems

| Problem | Impact | Needed |
|---|---|---|
| 76 positives (and the 31 rejections) are AI-reviewed only | Possible label errors in 31% of positives | A person reviews the 149 rows. Record the decisions in a separate file rather than editing the Excel, then rebuild the master |
| 2 auto-accepted pairs removed by hand, outside the code | Reproducibility | Approve the removal and encode it in code, or restore the pairs |
| 3,263 methods with unknown label excluded | Selection effect: deleted/moved methods are never seen | State as a limitation |
| Only files changed by a commit are present | `call_sites` and file-level context are partial; negatives are methods in changed files only | State as a scope limit |
| Project groups are inferred from paths | Two repositories could still share a group, or one could be split | ActRef gives no repository names. If the mapping can be recovered from the ActRef paper's data, use it |
| PyRef / ICSE data absent | No external validation yet | Add the files to `data/external/` |
| 308 rows with duplicate definitions | Features describe the last definition only | Flagged; exclude them in a sensitivity check |

---

## H. Next-stage recommendations

1. **Preprocessing.** Load features through the schema. Fit any transformation (log of long-tailed counts,
   scaling for linear models) on training folds only. Keep the master file unchanged.
2. **Feature engineering.** Keep it Before-only, and re-run `test_features_use_only_before_code` for any new feature.
   Candidates: name-token rarity across the project, and identifier quality scores.
3. **Feature selection.** Ablations per feature group, plus with/without `file_loc`/`file_n_methods`;
   drop `class_depth` or `in_class` for linear models. Decide from grouped-CV results.
4. **Splitting.**
   - Use `StratifiedGroupKFold(n_splits=5)` on `meta_project_group`. Hold out about 20% of groups as a test set first.
   - Do not use the legacy `project_guess`: it splits repositories such as imgaug/augmenters,
     chainercv/links_tests and deepvariant/third_party across groups, giving 333 groups instead of 323.
   - Only 79 groups contain positives, so report per-fold counts.
5. **Class imbalance.** Compare class weights, undersampling and SMOTE, applied inside the training folds only.
6. **Baselines.** Majority class, stratified random, and the bad-name rule.
7. **Models.** Logistic regression, random forest, LightGBM.
8. **Evaluation.**
   - Primary metric: PR-AUC (random level is 0.0078).
   - Also report precision, recall, F1 and MCC at a threshold chosen on validation data, plus ROC-AUC as a supplement.
   - Never report accuracy alone: always predicting 0 scores 99.2%.
   - Report mean ± std over folds, and run every experiment twice: all 243 positives, and without the 60 visibility- and case-only renames.
