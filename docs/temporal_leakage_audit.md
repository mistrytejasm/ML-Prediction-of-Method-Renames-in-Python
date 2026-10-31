# Temporal leakage audit: `call_sites` and the 9 `rel_*` features

Audit date: 2026-10-31 · Scope: the 10 features recalculated in `actref_master_dataset.csv`
· Inspection only: no code, dataset or master CSV was changed.

**Result:**

| Classification | Count | Features |
|---|---|---|
| **LEAKAGE** | 2 | `call_sites`, `rel_call_sites` |
| **SAFE** | 8 | the other `rel_*` features |

- **Why the 2 leak:** in the master, `call_sites` no longer reads the After code. But it counts calls across every file the *future commit changed*, and a rename forces its caller files into that set. So the search scope depends on the outcome being predicted. `rel_call_sites` is built from `call_sites` and inherits the leak.
- **Why the 8 are safe:** they are computed only from the method's own Before file. Their values were reproduced from that file alone with zero mismatches.
- **Recommendation:** remove `call_sites` and `rel_call_sites` from both experiments. That gives a 40-feature main set and a 42-feature comparison set.

Matching the master CSV values (as the feature documentation does) proves only that the code was
documented correctly. It does not prove the inputs are free of leakage, so this audit checks *which data*
each feature reads.

---

## 1. Prediction-time definition used in this audit

A prediction is made for a method **at the parent commit**, before the commit is written. At that moment we
have the method's file and the rest of the repository as they are at the parent commit. We do **not** know:

- which files the commit will change;
- which methods it will rename, delete or move;
- anything in the After snapshot.

A feature is SAFE only if it can be computed from information available at that moment, and its value does
not depend on the commit's outcome.

## 2. What the raw data contains

ActRef stores, for each commit, a `Before/` and an `After/` folder holding **only the files the commit changed**:

| Check (raw data, 1,159 unique commits) | Result |
|---|---|
| Before/After file pairs | 4,789 |
| Pairs whose Before and After bytes are identical | 47 (1.0%) |

So almost every file in a commit folder was modified by that commit. **Which files are present is itself
information about the future commit.** The audit treats the Before snapshot of a file as the file at the
parent commit, as ActRef defines it. This could not be checked against git history, because the repositories
are not in the project.

## 3. Where the 10 features are computed

All 10 are computed in `src/data/extract_features.py`, function `method_rows()`. References are to the
current file:

| Lines | What happens | Snapshot |
|---|---|---|
| 61 | `files` = every `.py` file in the commit's `Before/` folder, i.e. **the files the commit changed** | Before (scope chosen by the commit) |
| 63–67 | Each Before file that parses is added to `before_texts` (default mode). The After file is parsed too, but only used for the next line | Before text; After only checked to exist |
| 68–72 | A file becomes a row source (`parsed`) only if both Before and After parse | Row selection uses After |
| 73 | `all_before_text` = all of those Before texts joined, the search space for `call_sites` | Before, commit scope |
| 77 | `before = collect(btree)` (siblings) and `after = collect(atree)` (labels only) | Before for features, After for labels |
| 88–104 | For every eligible Before method: `label` = 1 (positive list), 0 (same name in After) or None (unknown). Unknown methods are **kept** in the default mode | After used **only** for `label` |
| 107–118 | Base features, from `func` (a Before AST node) and `name` | Before |
| 117 | `call_sites = len(re.findall(r"\bname\s*\(", all_before_text)) - 1` | Before text of the **changed files** |
| 129–133 | For each column in `REL_COLS`: `rel_col = value - mean(value over block)`; the block holds every eligible Before method of the file, regardless of label | Before |
| 134 | Only rows with a known label are written out, **after** the averages were computed | After (selection only) |

Helpers: `REL_COLS`, `collect()`, `parse()` and `code_features()` in `src/features/method_features.py`. `collect()`
walks only the AST it is given. Here that is the Before tree for every feature value.

## 4. `call_sites`: detailed analysis

**What it counts:** occurrences of `name(` in `all_before_text`, minus one for the definition. It is a regular
expression match on the bare name, so a call to another function that happens to have the same name also counts.

**Which files and snapshots are searched (line 61 → 66 → 73 → 117):**
- **Searched:** the Before version of every parseable file in the commit folder. That includes the method's own
  file, and Before files deleted or moved by the commit.
- **Not searched:** any file the commit did not change, and any After file.

**The problem.** The search covers *other* files only when the future commit changed them. A rename forces
every caller to change, so the callers of a renamed method are pulled into the commit and get counted. The
callers of a method that is not renamed usually sit in unchanged files, which are missing from the data and
are not counted.

Evidence from the data (`call_sites` = own-file calls + other-file calls; the split was checked to add up
exactly for all 31,209 rows):

| Measure | Renamed | Not renamed | Single-feature AUC |
|---|---|---|---|
| Methods with ≥ 1 call found in another file | **45.7%** | 38.5% | 0.521 (other-file calls) |
| Mean calls in the method's own file | 1.20 | 2.82 | 0.448 (own-file calls) |
| Mean `call_sites` | 2.91 | 6.00 | 0.470 |
| Files in the commit folder (median) | 4 | 5 | 0.433 |

**The mechanism, checked directly:**
- 111 renamed methods have callers in other files of the commit.
- In **92 of them**, those caller files call the **new** name in the After snapshot.
- That means those files are in the commit *because* of the rename.

The two parts of `call_sites` point in opposite directions:
- **Own-file calls** (Before-only, own file) are *lower* for renamed methods.
- **Other-file calls**, whose scope the commit chooses, are *more often present* for renamed methods.

The two parts are mixed in one number, so `call_sites` combines a legitimate signal with an
outcome-dependent one.

| Question | Answer |
|---|---|
| 1. Where calculated | `src/data/extract_features.py`, `method_rows()`, line 117; scope built at lines 61–73 |
| 2. Snapshot | Before text only (in the default mode) |
| 3. Accesses future information | **Yes, indirectly.** The set of files searched is the future commit's changed-file set |
| 4. Sibling information only from Before | Yes for text; but *which* other files are present is decided by the commit |
| 5. Computable at prediction time | **Not as defined.** At prediction time you would search the whole repository (or only the own file); the commit's file set is unknown |
| 6. Uses commit-level information | **Yes:** the commit's changed-file set |
| 7. Indirectly encodes the outcome | **Yes:** renaming forces caller files into the commit, so cross-file calls are found mainly for renamed methods (92 of 111 cases) |
| 8. Train/test or group leakage | Low: all rows of a commit share one project group (checked: max 1 group per commit), so grouped CV keeps them together. The leak is temporal, not between splits |
| **Classification** | **LEAKAGE** |

In v2 (`--legacy-v2`) the scope was even narrower: Before files whose After counterpart was missing were
skipped (line 66). That made the value depend on the After snapshot directly. The master removed that
dependence but not the commit-scope dependence described here.

## 5. `rel_call_sites`

Computed at lines 129–133 as `call_sites - mean(call_sites over the file's eligible methods)`. Every sibling's
`call_sites` comes from line 117, with the same commit-scope search. Its single-feature AUC (0.566) is higher
than `call_sites` (0.470): comparing against siblings removes some of the own-file effect and keeps the
cross-file part.

| Question | Answer |
|---|---|
| 1. Where calculated | `extract_features.py`, `method_rows()`, lines 129–133, from line 117 values |
| 2. Snapshot | Before text only |
| 3. Accesses future information | **Yes, indirectly** (through `call_sites`) |
| 4. Sibling information only from Before | Siblings come from the Before file (`collect(btree)`, line 77), but their values carry the commit scope |
| 5. Computable at prediction time | **Not as defined** |
| 6. Uses commit-level information | **Yes**, inherited |
| 7. Indirectly encodes the outcome | **Yes**, inherited |
| 8. Train/test or group leakage | As for `call_sites` |
| **Classification** | **LEAKAGE** |

## 6. The other 8 `rel_*` features

`rel_name_len`, `rel_name_tokens`, `rel_loc`, `rel_n_params`, `rel_n_calls`, `rel_cyclomatic`, `rel_doc_len`,
`rel_name_body_overlap`.

How they are computed:
- **Base value, method side:** from the method's own Before node (`code_features(func, name)` at line 108) or
  its name (`name_features(name)` at line 107).
- **The average they are compared with:** taken over the `block` (line 87). The block holds every eligible
  method of `collect(btree)`, the **Before** tree of the same file (line 77).
- **Methods whose label is unknown** (deleted, moved or renamed below the detection threshold) stay in the
  block in the default mode (lines 100–104), so the After snapshot no longer decides who is a sibling.
- **The label** is assigned at lines 94–102. It is used only to filter rows at line 134, after the averages
  are computed (lines 129–133). So no label value enters any average.

**Independent check:** for 400 files (33,232 values), I recomputed every one of these 8 features from the
Before file alone, using `collect()` on the Before tree only and ignoring the After file, the labels and other
files. **0 mismatches** with the master CSV.

| Question | Answer (same for all 8) |
|---|---|
| 1. Where calculated | `extract_features.py`, `method_rows()`, lines 129–133. Base values at line 107 (`rel_name_len`, `rel_name_tokens`) or line 108 (the other 6) |
| 2. Snapshot | Before only |
| 3. Accesses future information | No |
| 4. Sibling information only from Before | Yes: the block comes from `collect(btree)`, and labels do not decide membership |
| 5. Computable at prediction time | Yes, from the method's file at the parent commit |
| 6. Uses commit-level information | No: only the method's own file. Other files of the commit are not read |
| 7. Indirectly encodes the outcome | No: averages are computed before the label filter and include methods of every label status |
| 8. Train/test or group leakage | Rows of one file share the same average, so a row-level random split would place siblings on both sides. That shares feature values but **not labels**. Grouped CV on `meta_project_group` keeps a file's rows together (checked) |
| **Classification** | **SAFE** (subject to the dataset-level condition in section 7, which applies to all 46 features) |

## 7. Dataset-level conditions (all 46 features, not specific to these 10)

These do not make a single feature leak, but they must be stated with any result:

1. **Rows come only from files the commit changed** (section 2). The model is trained and evaluated on methods
   in files that are about to be edited. At prediction time this matches a tool that scores the methods of a
   file a developer is currently changing, not a tool that scans a whole repository. Results should be
   described in those terms.
2. **Rows exist only when the label is known** (line 134): methods deleted, moved, or renamed in a way the
   detector cannot confirm are left out. That is about 9.5% of eligible methods, and the choice depends on
   the After snapshot. This is part of the task definition, not a feature leak, but it means the test set is
   not every method a developer would see.
3. **A file must parse in both snapshots to supply rows** (lines 68–72). The same kind of selection as point 2.

## 8. Summary

| Feature | Snapshot read | Future / commit-level information | Classification |
|---|---|---|---|
| `call_sites` | Before text of every file the commit changed | Yes: search scope = the commit's changed files, which a rename forces to include its callers | **LEAKAGE** |
| `rel_call_sites` | Same, via `call_sites` of each sibling | Yes (inherited) | **LEAKAGE** |
| `rel_name_len` | Method's own Before file | No | **SAFE** |
| `rel_name_tokens` | Method's own Before file | No | **SAFE** |
| `rel_loc` | Method's own Before file | No | **SAFE** |
| `rel_n_params` | Method's own Before file | No | **SAFE** |
| `rel_n_calls` | Method's own Before file | No | **SAFE** |
| `rel_cyclomatic` | Method's own Before file | No | **SAFE** |
| `rel_doc_len` | Method's own Before file | No | **SAFE** |
| `rel_name_body_overlap` | Method's own Before file | No | **SAFE** |

## 9. Recommendation

| Experiment | Previous plan | Recommended now | Features |
|---|---|---|---|
| Main | 42 | **40** | 42 minus `call_sites`, `rel_call_sites` |
| Comparison (file size) | 44 | **42** | the 40 + `file_loc`, `file_n_methods` |

- **Do not train the main or comparison models with `call_sites` or `rel_call_sites`.** Their values depend on
  which files the future commit changed, and that scope is tied to the rename itself.
- If a call-count signal is wanted, a **SAFE replacement** would count calls only inside the method's own Before
  file. That is the own-file part above (AUC 0.448); a repository-wide count would also work if the full
  repositories were available. Either needs a code change and a regenerated dataset, so neither has been done
  here; it needs your approval.
- The master CSV, the schema and `src/config.py` are unchanged. The schema still lists 46 features. Select the
  40 / 42 sets in the training code, or update the feature-set definitions after approval.
- Documents that still describe `call_sites` as usable will need updating once you approve this:
  `docs/feature_dataset_mapping.md`, the data dictionary and the Feature Reference doc.
- To measure the leak's effect, an optional diagnostic is to train once with and once without the two features
  under grouped CV, and report both, never the leaky one as a result.

Evidence for sections 2, 4 and 6 was produced by a read-only analysis of the raw commits and the master CSV
(no files written in the project).
