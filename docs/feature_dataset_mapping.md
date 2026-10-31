# Feature-to-dataset mapping

Generated on 2026-10-31 by reading the CSV headers and data directly. No dataset or feature code was modified. Every code example in section 4 was executed and its output matched the value stored in the master CSV.

**The 46-feature master dataset is unchanged and remains the canonical, audited source dataset.** The approved training configurations select columns from it:

| Training configuration | Features | Definition |
| --- | --- | --- |
| Main experiment | **40** | 46 minus `call_sites`, `rel_call_sites` (temporal leakage), `class_depth`, `is_async` (dropped), `file_loc`, `file_n_methods` (comparison only) |
| Comparison experiment | **42** | Main + `file_loc`, `file_n_methods` |

These are experimental training configurations, not changes to the master dataset. Exact lists: section 4.

| Dataset | File | Rows | Columns | Predictive features |
| --- | --- | --- | --- | --- |
| v1 | `archive/old_datasets/method_features_v1_rename_files_only.csv` | 2,158 | 39 | 34 |
| v2 | `data/processed/feature_tables/method_features_v2_all_commits.csv` | 31,217 | 51 | 46 |
| Master | `data/processed/master/actref_master_dataset.csv` | 31,209 | 59 | 46 |

A column counts as predictive when it is not an identifier (`commit`, `file`, `class`, `name`), not the target (`renamed`) and not a metadata column (`meta_*`).

## 1. Complete feature inventory

| # | Feature name | Feature group | v1 | v2 | Master | Description |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `name_len` | Name | Present | Present | Present | Length of the method name in characters. |
| 2 | `name_tokens` | Name | Present | Present | Present | Number of words in the name. |
| 3 | `name_is_private` | Name | Present | Present | Present | Whether the name starts with an underscore. |
| 4 | `name_is_camel` | Name | Present | Present | Present | Whether the name is camelCase. |
| 5 | `name_has_digit` | Name | Present | Present | Present | Whether the name contains a digit. |
| 6 | `name_short_tokens` | Name | Present | Present | Present | Number of 1-2 letter words that are not common abbreviations. |
| 7 | `name_no_vowel_tokens` | Name | Present | Present | Present | Number of words without a vowel (`cnt`, `msg`). |
| 8 | `name_starts_verb` | Name | Present | Present | Present | Whether the first word is a common verb. |
| 9 | `name_single_char` | Name | Present | Present | Present | Whether the name is a single character. |
| 10 | `loc` | Code | Present | Present | Present | Lines of code of the method. |
| 11 | `n_params` | Code | Present | Present | Present | Number of parameters, excluding `self` and `cls`. |
| 12 | `n_defaults` | Code | Present | Present | Present | Number of parameters with a default value. |
| 13 | `has_varargs` | Code | Present | Present | Present | Whether the method accepts `*args`. |
| 14 | `has_kwargs` | Code | Present | Present | Present | Whether the method accepts `**kwargs`. |
| 15 | `n_returns` | Code | Present | Present | Present | Number of `return` statements. |
| 16 | `n_calls` | Code | Present | Present | Present | Number of function calls in the body. |
| 17 | `cyclomatic` | Code | Present | Present | Present | Cyclomatic complexity (number of decision paths). |
| 18 | `max_nesting` | Code | Present | Present | Present | Deepest nesting of if / for / while / with / try blocks. |
| 19 | `has_docstring` | Code | Present | Present | Present | Whether the method has a docstring. |
| 20 | `doc_len` | Code | Present | Present | Present | Length of the docstring in characters. |
| 21 | `is_stub` | Code | Present | Present | Present | Whether the body is only a placeholder. |
| 22 | `is_async` | Code | Present | Present | Present | Whether the method is `async def`. |
| 23 | `n_decorators` | Code | Present | Present | Present | Number of decorators. |
| 24 | `is_property` | Code | Present | Present | Present | Whether a decorator mentions `property`. |
| 25 | `is_static_or_class` | Code | Present | Present | Present | Whether the method is a `@staticmethod` or `@classmethod`. |
| 26 | `name_body_overlap` | Code | Present | Present | Present | Share of the name's words that also appear in identifiers of the body. |
| 27 | `in_class` | Context | Present | Present | Present | Whether the method is inside a class (1) or a module-level function (0). |
| 28 | `class_depth` | Context | Present | Present | Present | Class nesting depth (0 = module function, 1 = class, 2+ = nested class). |
| 29 | `class_n_methods` | Context | Present | Present | Present | Number of methods directly in the same class (0 for module functions). |
| 30 | `file_n_methods` | File-size | Present | Present | Present | Number of methods and functions in the Before file. |
| 31 | `file_loc` | File-size | Present | Present | Present | Number of lines in the Before file. |
| 32 | `is_test_file` | Context | Present | Present | Present | Whether the file path contains `test`. |
| 33 | `style_mismatch` | Context | Present | Present | Present | Whether a camelCase name sits in a mostly snake_case file. |
| 34 | `call_sites` | Context | Present | Present | Present | How often the name is called in the commit's Before files. |
| 35 | `name_token_sharing` | Relative-to-file | Absent | Present | Present | Share of the name's words that other methods in the file also use. |
| 36 | `first_token_shared` | Relative-to-file | Absent | Present | Present | Whether another method in the file uses the same first word. |
| 37 | `style_share_in_file` | Relative-to-file | Absent | Present | Present | Share of methods in the file that use the same naming style. |
| 38 | `rel_name_len` | Relative-to-file | Absent | Present | Present | Name length minus the average of the eligible methods in the file. |
| 39 | `rel_name_tokens` | Relative-to-file | Absent | Present | Present | Word count minus the file average. |
| 40 | `rel_loc` | Relative-to-file | Absent | Present | Present | Lines of code minus the file average. |
| 41 | `rel_n_params` | Relative-to-file | Absent | Present | Present | Parameter count minus the file average. |
| 42 | `rel_n_calls` | Relative-to-file | Absent | Present | Present | Call count minus the file average. |
| 43 | `rel_cyclomatic` | Relative-to-file | Absent | Present | Present | Complexity minus the file average. |
| 44 | `rel_doc_len` | Relative-to-file | Absent | Present | Present | Docstring length minus the file average. |
| 45 | `rel_call_sites` | Relative-to-file | Absent | Present | Present | Call count minus the file average. |
| 46 | `rel_name_body_overlap` | Relative-to-file | Absent | Present | Present | Name-body overlap minus the file average. |

Non-predictive columns in each file:

| Dataset | Identifiers | Target | Metadata |
| --- | --- | --- | --- |
| v1 | `commit`, `file`, `class`, `name` | `renamed` | none |
| v2 | `commit`, `file`, `class`, `name` | `renamed` | none |
| Master | `commit`, `file`, `class`, `name` | `renamed` | `meta_label_source`, `meta_label_provenance`, `meta_label_similarity`, `meta_rename_type`, `meta_new_name`, `meta_project_guess`, `meta_project_group`, `meta_duplicate_definition_in_before` |

## 2. Feature group classification

| Group | v1 | v2 | Master |
| --- | --- | --- | --- |
| Name | 9 | 9 | 9 |
| Code | 17 | 17 | 17 |
| Context | 6 | 6 | 6 |
| Relative-to-file | 0 | 12 | 12 |
| File-size | 2 | 2 | 2 |
| **Total** | **34** | **46** | **46** |

**Name (9)**: `name_len`, `name_tokens`, `name_is_private`, `name_is_camel`, `name_has_digit`, `name_short_tokens`, `name_no_vowel_tokens`, `name_starts_verb`, `name_single_char`

**Code (17)**: `loc`, `n_params`, `n_defaults`, `has_varargs`, `has_kwargs`, `n_returns`, `n_calls`, `cyclomatic`, `max_nesting`, `has_docstring`, `doc_len`, `is_stub`, `is_async`, `n_decorators`, `is_property`, `is_static_or_class`, `name_body_overlap`

**Context (6)**: `in_class`, `class_depth`, `class_n_methods`, `is_test_file`, `style_mismatch`, `call_sites`

**Relative-to-file (12)**: `name_token_sharing`, `first_token_shared`, `style_share_in_file`, `rel_name_len`, `rel_name_tokens`, `rel_loc`, `rel_n_params`, `rel_n_calls`, `rel_cyclomatic`, `rel_doc_len`, `rel_call_sites`, `rel_name_body_overlap`

**File-size (2)**: `file_n_methods`, `file_loc`

`file_loc` and `file_n_methods` are listed as their own File-size group here; earlier documents counted them under Context (8 = 6 + 2).

## 3. Changes between datasets

### v1 → v2

- **Added (12):** `name_token_sharing`, `first_token_shared`, `style_share_in_file`, `rel_name_len`, `rel_name_tokens`, `rel_loc`, `rel_n_params`, `rel_n_calls`, `rel_cyclomatic`, `rel_doc_len`, `rel_call_sites`, `rel_name_body_overlap` (the whole Relative-to-file group).
- **Removed:** none.
- **Rows:** v1 took negatives only from files that contain a rename (2,158 rows); v2 takes them from all commits (31,217 rows).

### v2 → Master

- **Predictive features added or removed:** none. Both have the same 46 features in the same order.
- **Columns added:** 8 metadata columns: `meta_label_source`, `meta_label_provenance`, `meta_label_similarity`, `meta_rename_type`, `meta_new_name`, `meta_project_guess`, `meta_project_group`, `meta_duplicate_definition_in_before`. They are not features.
- **Rows removed:** 8 rows of a byte-identical duplicate commit folder (`dd7f9fa8…_<garbled suffix>`).
- **Recalculated features (10):** values differ from v2 on 7,399 of 31,209 shared rows (104 of them renamed):

| Feature | Values changed |
| --- | --- |
| `call_sites` | 306 |
| `rel_name_len` | 6,373 |
| `rel_name_tokens` | 6,202 |
| `rel_loc` | 6,429 |
| `rel_n_params` | 6,246 |
| `rel_n_calls` | 6,389 |
| `rel_cyclomatic` | 6,225 |
| `rel_doc_len` | 4,858 |
| `rel_call_sites` | 7,193 |
| `rel_name_body_overlap` | 6,112 |

### Leakage problems and corrections

| Feature(s) | Problem in v2 | Correction in the master |
| --- | --- | --- |
| `call_sites` | Before files with no After counterpart (deleted or moved in the commit) were not searched, so the value depended on the After code. | Every parseable Before file of the commit is searched. |
| `rel_*` (9) | The file average used only methods whose label was known; methods deleted in the After code were left out, so they changed their siblings' values. | The average uses every eligible (non-dunder, non-test) Before method of the file. |

No other feature had leakage in this comparison: the remaining 36 are identical in v2 and the master and are computed from the Before code only.

### Temporal leakage audit of the 10 recalculated features

A later audit (`docs/temporal_leakage_audit.md`) checked which data each of the 10 recalculated features reads:

| Feature | Result | Reason |
| --- | --- | --- |
| `call_sites` | **LEAKAGE, excluded** | Its search scope is every Before file the future commit changed, i.e. it depends on the commit's changed-file list. A rename forces caller files into that list (92 of 111 renamed methods with other-file callers had them updated to the new name). |
| `rel_call_sites` | **LEAKAGE, excluded** | Built from `call_sites` of the method and its siblings, so it inherits the same dependence. |
| The other 8 `rel_*` | **SAFE** | Computed from the method's own Before file only; they passed the Before-only recomputation audit (33,232 values recomputed from the Before file alone, 0 mismatches). |

### Features that should not be used for training

The master contains 46 audited predictive features (its schema). The approved training configurations use fewer:

| Feature | Recommendation | Reason |
| --- | --- | --- |
| `call_sites` | Exclude | Temporal leakage: search scope depends on the future commit's changed-file list. |
| `rel_call_sites` | Exclude | Inherits the leakage of `call_sites`. |
| `class_depth` | Drop | Near-duplicate of `in_class` (r = 0.997). |
| `is_async` | Drop | 1 for only 37 methods, 0 of them renamed. |
| `file_loc`, `file_n_methods` | Comparison model only | Strong dilution effect, not a naming signal. Train with and without. |
| Every v2 feature | Do not train on v2 | 10 leaky features and 8 duplicate rows. |
| Every v1 feature | Do not train on v1 | Biased negative sampling; no relative features. |

Main experiment = 40 features (46 minus the six above). Comparison experiment = 42 (main + `file_loc`, `file_n_methods`).

## 4. Experimental feature sets

The master dataset is not changed. These lists are applied when building X for training.

### Main experiment: 40 features

- **Name (9):** `name_len`, `name_tokens`, `name_is_private`, `name_is_camel`, `name_has_digit`, `name_short_tokens`, `name_no_vowel_tokens`, `name_starts_verb`, `name_single_char`
- **Code (16):** `loc`, `n_params`, `n_defaults`, `has_varargs`, `has_kwargs`, `n_returns`, `n_calls`, `cyclomatic`, `max_nesting`, `has_docstring`, `doc_len`, `is_stub`, `n_decorators`, `is_property`, `is_static_or_class`, `name_body_overlap`
- **Context (4):** `in_class`, `class_n_methods`, `is_test_file`, `style_mismatch`
- **Relative-to-file (11):** `name_token_sharing`, `first_token_shared`, `style_share_in_file`, `rel_name_len`, `rel_name_tokens`, `rel_loc`, `rel_n_params`, `rel_n_calls`, `rel_cyclomatic`, `rel_doc_len`, `rel_name_body_overlap`

### Comparison experiment: 42 features

The 40 main features plus the File-size group: `file_loc`, `file_n_methods`.

### Not used in either experiment

| Feature | Reason |
| --- | --- |
| `call_sites` | Temporal leakage: search scope depends on the future commit's changed-file list |
| `rel_call_sites` | Temporal leakage inherited from `call_sites` |
| `class_depth` | Near-duplicate of `in_class` (r = 0.997) |
| `is_async` | Near-constant: 37 methods, 0 renamed |

### Research scope and data conditions

- **Scope:** methods that belong to files edited by the future commit. ActRef stores only the files a commit changed (4,742 of 4,789 Before/After file pairs differ), so every row comes from such a file. Results describe scoring the methods of files being edited, not every method of a repository.
- **Unknown outcomes excluded:** about 9.5% of eligible methods (3,263) were deleted, moved, or renamed in a way the detector could not confirm, so their label is unknown and they are not in the dataset.
- **Parsing:** a file supplies rows only if both its Before and After versions parse as Python.

### Future work (not implemented, not approved)

- **Own-file call count** as a replacement for `call_sites`: count calls to the method only inside its own Before file, which does not depend on the commit's changed-file list. It is a proposal only: it is not in the master dataset or the feature code, and it needs approval, a code change and a regenerated dataset.

## 5. Every master feature explained

The examples use a real renamed method: `create_continuous_observation_encoder` in class `LearningModel` of `python/unitytrainers/models.py`, commit `0022854…`, renamed to `create_vector_observation_encoder`. Run this setup once from the project root, then any example below. Each example prints exactly the value stored in the master CSV for this method.

```python
import ast, re
from pathlib import Path
from src.features.method_features import collect, split_name, style_of, code_features, VERBS, SHORT_OK

commit = Path("data/raw/actref/commits/0022854a64751f0e8387c01e79e1c0545c3ea6d2")
path = "python/unitytrainers/models.py"
src = (commit / "Before" / path.replace("/", "#")).read_text(encoding="utf-8", errors="ignore")
tree = ast.parse(src)
methods = collect(tree)              # (class_path, name) -> (node, class_n_methods, class_depth)
cls_path = "LearningModel"
func, cls_n, depth = methods[(cls_path, "create_continuous_observation_encoder")]
name = func.name
eligible = {k: v for k, v in methods.items()           # used by the rel_* features
            if not (k[1].startswith("__") and k[1].endswith("__")) and not k[1].startswith("test")}
texts = []                                             # every parseable Before file of the commit
for p in sorted((commit / "Before").glob("*.py")):
    t = p.read_text(encoding="utf-8", errors="ignore")
    try:
        ast.parse(t)
        texts.append(t)
    except SyntaxError:
        pass
commit_before_text = "\n".join(texts)
```

### Name features

#### 1. `name_len`

- **Group:** Name
- **What it measures:** Length of the method name in characters.
- **How it is calculated:** `len(name)`.
- **Why it may help predict a rename:** Long, over-specific names are often shortened or reworded.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with name_tokens (r = 0.848).

```python
print(len(name))
```

Example output: `37`

#### 2. `name_tokens`

- **Group:** Name
- **What it measures:** Number of words in the name.
- **How it is calculated:** Name split on underscores and camelCase boundaries (`split_name`), then counted.
- **Why it may help predict a rename:** More words give more chances that one word is wrong or out of date.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with name_len (r = 0.848).

```python
print(len(split_name(name)))
```

Example output: `4`

#### 3. `name_is_private`

- **Group:** Name
- **What it measures:** Whether the name starts with an underscore.
- **How it is calculated:** `name.startswith('_')` as 0/1.
- **Why it may help predict a rename:** Lets the model recognise visibility-only renames (`init_model -> _init_model`, 43 of the 243 positives).
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(name.startswith("_")))
```

Example output: `0`

#### 4. `name_is_camel`

- **Group:** Name
- **What it measures:** Whether the name is camelCase.
- **How it is calculated:** `style_of(name) == 'camel'` (a lower-case letter followed by an upper-case one, no underscore).
- **Why it may help predict a rename:** camelCase breaks Python convention (PEP 8) and tends to be converted to snake_case.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(style_of(name) == "camel"))
```

Example output: `0`

#### 5. `name_has_digit`

- **Group:** Name
- **What it measures:** Whether the name contains a digit.
- **How it is calculated:** `any(ch.isdigit() for ch in name)` as 0/1.
- **Why it may help predict a rename:** Digits often mark temporary or versioned names (`layer2`, `run_v1`).
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(any(ch.isdigit() for ch in name)))
```

Example output: `0`

#### 6. `name_short_tokens`

- **Group:** Name
- **What it measures:** Number of 1-2 letter words that are not common abbreviations.
- **How it is calculated:** Words of length <= 2, not digits, not in `SHORT_OK` (id, io, ip, ok, db, os, ui, x, y, z, fn, nn, lr).
- **Why it may help predict a rename:** Unclear abbreviations are a classic naming smell.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(sum(1 for t in split_name(name) if len(t) <= 2 and t not in SHORT_OK and not t.isdigit()))
```

Example output: `0`

#### 7. `name_no_vowel_tokens`

- **Group:** Name
- **What it measures:** Number of words without a vowel (`cnt`, `msg`).
- **How it is calculated:** Words longer than 1 letter, not digits, containing none of a/e/i/o/u.
- **Why it may help predict a rename:** Abbreviations like these are often expanded to full words.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(sum(1 for t in split_name(name) if len(t) > 1 and not set(t) & set("aeiou") and not t.isdigit()))
```

Example output: `0`

#### 8. `name_starts_verb`

- **Group:** Name
- **What it measures:** Whether the first word is a common verb.
- **How it is calculated:** First word of `split_name(name)` is in the fixed `VERBS` list (72 verbs: get, set, create, load, ...).
- **Why it may help predict a rename:** Action names are refined as behaviour changes (`grad_of -> insert_grad_of`).
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
words = split_name(name)
print(int(bool(words) and words[0] in VERBS))
```

Example output: `1`

#### 9. `name_single_char`

- **Group:** Name
- **What it measures:** Whether the name is a single character.
- **How it is calculated:** `len(name.strip('_')) == 1` as 0/1.
- **Why it may help predict a rename:** One-letter names carry no meaning and are candidates for renaming.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Very rare (see value counts); kept because it costs nothing.

```python
print(int(len(name.strip("_")) == 1))
```

Example output: `0`

### Code features

#### 10. `loc`

- **Group:** Code
- **What it measures:** Lines of code of the method.
- **How it is calculated:** `end_lineno - lineno + 1` of the function node (includes the docstring).
- **Why it may help predict a rename:** Very short methods are less settled and cheap to rename.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Long right tail; correlated with n_calls (r = 0.807).

```python
print(func.end_lineno - func.lineno + 1)
```

Example output: `17`

#### 11. `n_params`

- **Group:** Code
- **What it measures:** Number of parameters, excluding `self` and `cls`.
- **How it is calculated:** Positional and keyword-only arguments of the signature.
- **Why it may help predict a rename:** Signature size; unusual signatures attract refactoring.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
a = func.args
print(len([x for x in a.args + a.kwonlyargs if x.arg not in ("self", "cls")]))
```

Example output: `6`

#### 12. `n_defaults`

- **Group:** Code
- **What it measures:** Number of parameters with a default value.
- **How it is calculated:** `len(args.defaults)` + keyword-only defaults that are not None.
- **Why it may help predict a rename:** Mature, widely used functions collect optional parameters and are renamed less often.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
a = func.args
print(len(a.defaults) + sum(1 for d in a.kw_defaults if d is not None))
```

Example output: `0`

#### 13. `has_varargs`

- **Group:** Code
- **What it measures:** Whether the method accepts `*args`.
- **How it is calculated:** `args.vararg is not None` as 0/1.
- **Why it may help predict a rename:** Generic wrapper functions tend to be long-lived.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(func.args.vararg is not None))
```

Example output: `0`

#### 14. `has_kwargs`

- **Group:** Code
- **What it measures:** Whether the method accepts `**kwargs`.
- **How it is calculated:** `args.kwarg is not None` as 0/1.
- **Why it may help predict a rename:** Same idea as has_varargs.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(func.args.kwarg is not None))
```

Example output: `0`

#### 15. `n_returns`

- **Group:** Code
- **What it measures:** Number of `return` statements.
- **How it is calculated:** Count of `ast.Return` nodes anywhere inside the function (nested functions included).
- **Why it may help predict a rename:** Describes control-flow shape; useful in combination.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Counts returns of nested functions too.

```python
print(sum(isinstance(n, ast.Return) for n in ast.walk(func)))
```

Example output: `1`

#### 16. `n_calls`

- **Group:** Code
- **What it measures:** Number of function calls in the body.
- **How it is calculated:** Count of `ast.Call` nodes inside the function.
- **Why it may help predict a rename:** How much work the method delegates.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Long right tail; correlated with loc (r = 0.807).

```python
print(sum(isinstance(n, ast.Call) for n in ast.walk(func)))
```

Example output: `5`

#### 17. `cyclomatic`

- **Group:** Code
- **What it measures:** Cyclomatic complexity (number of decision paths).
- **How it is calculated:** 1 + branch nodes (if, for, while, except, with, if-expression, comprehension, assert) + extra operands of `and`/`or`.
- **Why it may help predict a rename:** Logic complexity; complex methods are refactored more.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_cyclomatic (r = 0.859).

```python
BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With,
            ast.AsyncWith, ast.IfExp, ast.comprehension, ast.Assert)
nodes = list(ast.walk(func))
print(1 + sum(isinstance(n, BRANCHES) for n in nodes)
        + sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp)))
```

Example output: `3`

#### 18. `max_nesting`

- **Group:** Code
- **What it measures:** Deepest nesting of if / for / while / with / try blocks.
- **How it is calculated:** Recursive walk that adds 1 at each of those block types.
- **Why it may help predict a rename:** Structural complexity.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
def depth(node, d=0):
    best = d
    for c in ast.iter_child_nodes(node):
        nested = isinstance(c, (ast.If, ast.For, ast.While, ast.With, ast.Try))
        best = max(best, depth(c, d + 1 if nested else d))
    return best
print(depth(func))
```

Example output: `2`

#### 19. `has_docstring`

- **Group:** Code
- **What it measures:** Whether the method has a docstring.
- **How it is calculated:** `ast.get_docstring(func)` is not empty, as 0/1.
- **Why it may help predict a rename:** Documented code tends to be more stable.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Redundant with doc_len > 0.

```python
print(int(bool(ast.get_docstring(func))))
```

Example output: `1`

#### 20. `doc_len`

- **Group:** Code
- **What it measures:** Length of the docstring in characters.
- **How it is calculated:** `len(ast.get_docstring(func) or '')` (indentation cleaned by `get_docstring`).
- **Why it may help predict a rename:** Renamed methods have shorter docstrings on average.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Long right tail (max above 10,000): log-transform for linear models.

```python
print(len(ast.get_docstring(func) or ""))
```

Example output: `390`

#### 21. `is_stub`

- **Group:** Code
- **What it measures:** Whether the body is only a placeholder.
- **How it is calculated:** Every statement after the docstring is `pass`, `raise NotImplementedError` or a bare constant.
- **Why it may help predict a rename:** Placeholder methods often receive their final name later.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
doc = ast.get_docstring(func) or ""
body = func.body[1:] if doc else func.body
print(int(all(isinstance(s, ast.Pass)
              or (isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s))
              or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
              for s in body) if body else 1))
```

Example output: `0`

#### 22. `is_async`

- **Group:** Code
- **What it measures:** Whether the method is `async def`.
- **How it is calculated:** `isinstance(func, ast.AsyncFunctionDef)` as 0/1.
- **Why it may help predict a rename:** Async methods form a distinct API style.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Dropped (near-constant)
- **Leakage or redundancy concerns:** Near-constant: 37 of 31,209 methods, 0 renamed. Recommended to drop.

```python
print(int(isinstance(func, ast.AsyncFunctionDef)))
```

Example output: `0`

#### 23. `n_decorators`

- **Group:** Code
- **What it measures:** Number of decorators.
- **How it is calculated:** `len(func.decorator_list)`.
- **Why it may help predict a rename:** Decorators mark special roles (property, static method, framework hooks).
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(len(func.decorator_list))
```

Example output: `1`

#### 24. `is_property`

- **Group:** Code
- **What it measures:** Whether a decorator mentions `property`.
- **How it is calculated:** Any decorator whose source text contains `property` (a `@x.setter` decorator does not count).
- **Why it may help predict a rename:** Properties are public interface; renaming one means changing every caller.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(any("property" in ast.unparse(d) for d in func.decorator_list)))
```

Example output: `0`

#### 25. `is_static_or_class`

- **Group:** Code
- **What it measures:** Whether the method is a `@staticmethod` or `@classmethod`.
- **How it is calculated:** A decorator whose source text is exactly `staticmethod` or `classmethod`.
- **Why it may help predict a rename:** Describes the method's role.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
print(int(any(ast.unparse(d) in ("staticmethod", "classmethod") for d in func.decorator_list)))
```

Example output: `1`

#### 26. `name_body_overlap`

- **Group:** Code
- **What it measures:** Share of the name's words that also appear in identifiers of the body.
- **How it is calculated:** Words of every Name, Attribute and argument inside the function; overlap = matched name words / name words.
- **Why it may help predict a rename:** Tests directly whether the name describes the code.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_name_body_overlap (r = 0.848).

```python
idents = set()
for n in ast.walk(func):
    if isinstance(n, ast.Name): idents.add(n.id)
    elif isinstance(n, ast.Attribute): idents.add(n.attr)
    elif isinstance(n, ast.arg): idents.add(n.arg)
body_words = {w for i in idents for w in split_name(i)}
words = split_name(name)
print(round(sum(w in body_words for w in words) / len(words), 3))
```

Example output: `0.25`

### Context features

#### 27. `in_class`

- **Group:** Context
- **What it measures:** Whether the method is inside a class (1) or a module-level function (0).
- **How it is calculated:** Class path from `collect()` is not empty.
- **Why it may help predict a rename:** Context the model needs to read the class features.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Redundant with class_depth (r = 0.997).

```python
print(int(bool(cls_path)))
```

Example output: `1`

#### 28. `class_depth`

- **Group:** Context
- **What it measures:** Class nesting depth (0 = module function, 1 = class, 2+ = nested class).
- **How it is calculated:** Depth counted while `collect()` walks nested ClassDef nodes.
- **Why it may help predict a rename:** Nested classes are unusual code structure.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Dropped (redundant)
- **Leakage or redundancy concerns:** Near-duplicate of in_class (r = 0.997); only 40 rows have depth 2 or 3. Recommended to drop.

```python
print(depth)
```

Example output: `1`

#### 29. `class_n_methods`

- **Group:** Context
- **What it measures:** Number of methods directly in the same class (0 for module functions).
- **How it is calculated:** Count of function definitions in the class body.
- **Why it may help predict a rename:** Class size and structure; renamed methods sit in smaller classes.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Partly a size effect, like the file-size features.

```python
cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_path)
print(sum(isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) for c in cls_node.body))
```

Example output: `12`

#### 30. `is_test_file`

- **Group:** Context
- **What it measures:** Whether the file path contains `test`.
- **How it is calculated:** `'test' in path.lower()` as 0/1.
- **Why it may help predict a rename:** Test code follows looser naming habits.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None (test methods themselves are excluded from the dataset).

```python
print(int("test" in path.lower()))
```

Example output: `0`

#### 31. `style_mismatch`

- **Group:** Context
- **What it measures:** Whether a camelCase name sits in a mostly snake_case file.
- **How it is calculated:** Name is camelCase and more than half of the file's methods are snake_case or single lower-case words.
- **Why it may help predict a rename:** An inconsistent style within a file is a strong rename trigger when present.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Rare.

```python
styles = [style_of(n) for (_, n) in methods]
snake_share = sum(s in ("snake", "lower") for s in styles) / len(styles)
print(int(style_of(name) == "camel" and snake_share > 0.5))
```

Example output: `0`

#### 32. `call_sites`

- **Group:** Context
- **What it measures:** How often the name is called in the commit's Before files.
- **How it is calculated:** Regex `\bname\s*\(` over every parseable Before file of the commit, minus 1 for the definition.
- **Why it may help predict a rename:** Widely used methods are costly to rename, so they change less.
- **Available before the commit:** Computed from Before code only, but over the files the future commit changed: not available in this form at prediction time.
- **Training use:** Excluded (temporal leakage)
- **Leakage or redundancy concerns:** **EXCLUDED from both experiments (temporal leakage).** The search scope is every Before file the future commit changed, so it depends on the commit's changed-file list; a rename forces its caller files into that list (92 of 111 renamed methods with other-file callers). See docs/temporal_leakage_audit.md. Leaked After information in v2; the master removed that, not the commit-scope dependence.

```python
print(len(re.findall(r"\b" + re.escape(name) + r"\s*\(", commit_before_text)) - 1)
```

Example output: `4`

### Relative-to-file features

#### 33. `name_token_sharing`

- **Group:** Relative-to-file
- **What it measures:** Share of the name's words that other methods in the file also use.
- **How it is calculated:** For each distinct word of the name: is it used by more than one method name in the Before file? Share of such words.
- **Why it may help predict a rename:** Names outside the file's vocabulary tend to be aligned with it.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with first_token_shared (r = 0.849).

```python
counts = {}
for (_, n) in methods:
    for w in set(split_name(n)):
        counts[w] = counts.get(w, 0) + 1
words = set(split_name(name))
print(round(sum(counts[w] > 1 for w in words) / len(words), 3))
```

Example output: `0.75`

#### 34. `first_token_shared`

- **Group:** Relative-to-file
- **What it measures:** Whether another method in the file uses the same first word.
- **How it is calculated:** The name's first word appears in more than one method name of the Before file.
- **Why it may help predict a rename:** Spots a method that breaks a naming pattern (`get_a`, `get_b`, `fetch_c`).
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with name_token_sharing (r = 0.849).

```python
counts = {}
for (_, n) in methods:
    for w in set(split_name(n)):
        counts[w] = counts.get(w, 0) + 1
print(int(counts[split_name(name)[0]] > 1))
```

Example output: `1`

#### 35. `style_share_in_file`

- **Group:** Relative-to-file
- **What it measures:** Share of methods in the file that use the same naming style.
- **How it is calculated:** Methods with the same `style_of` value / all methods in the Before file.
- **Why it may help predict a rename:** A low value means the odd one out in naming style.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** None.

```python
styles = [style_of(n) for (_, n) in methods]
print(round(styles.count(style_of(name)) / len(styles), 3))
```

Example output: `0.833`

#### 36. `rel_name_len`

- **Group:** Relative-to-file
- **What it measures:** Name length minus the average of the eligible methods in the file.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** An unusually long name for its file stands out.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_name_tokens (r = 0.828). Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [len(n) for (_, n) in eligible]
print(round(len(name) - sum(vals) / len(vals), 3))
```

Example output: `14.273`

#### 37. `rel_name_tokens`

- **Group:** Relative-to-file
- **What it measures:** Word count minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** A name wordier than its neighbours is a strong rename clue.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_name_len. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [len(split_name(n)) for (_, n) in eligible]
print(round(len(split_name(name)) - sum(vals) / len(vals), 3))
```

Example output: `0.818`

#### 38. `rel_loc`

- **Group:** Relative-to-file
- **What it measures:** Lines of code minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** Unusual size for its file.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_n_calls (r = 0.812). Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [f.end_lineno - f.lineno + 1 for (f, _, _) in eligible.values()]
print(round(func.end_lineno - func.lineno + 1 - sum(vals) / len(vals), 3))
```

Example output: `-6.182`

#### 39. `rel_n_params`

- **Group:** Relative-to-file
- **What it measures:** Parameter count minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** Unusual signature for its file.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [code_features(f, n)["n_params"] for (_, n), (f, _, _) in eligible.items()]
print(round(code_features(func, name)["n_params"] - sum(vals) / len(vals), 3))
```

Example output: `3.455`

#### 40. `rel_n_calls`

- **Group:** Relative-to-file
- **What it measures:** Call count minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** Unusual amount of delegation for its file.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with rel_loc. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [code_features(f, n)["n_calls"] for (_, n), (f, _, _) in eligible.items()]
print(round(code_features(func, name)["n_calls"] - sum(vals) / len(vals), 3))
```

Example output: `-5.364`

#### 41. `rel_cyclomatic`

- **Group:** Relative-to-file
- **What it measures:** Complexity minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** Unusual logic complexity for its file.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with cyclomatic. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [code_features(f, n)["cyclomatic"] for (_, n), (f, _, _) in eligible.items()]
print(round(code_features(func, name)["cyclomatic"] - sum(vals) / len(vals), 3))
```

Example output: `0.091`

#### 42. `rel_doc_len`

- **Group:** Relative-to-file
- **What it measures:** Docstring length minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** Unusual amount of documentation for its file.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Long tails. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [code_features(f, n)["doc_len"] for (_, n), (f, _, _) in eligible.items()]
print(round(code_features(func, name)["doc_len"] - sum(vals) / len(vals), 3))
```

Example output: `203.636`

#### 43. `rel_call_sites`

- **Group:** Relative-to-file
- **What it measures:** Call count minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** How central the method is within its file.
- **Available before the commit:** Computed from Before code only, but over the files the future commit changed: not available in this form at prediction time.
- **Training use:** Excluded (temporal leakage)
- **Leakage or redundancy concerns:** **EXCLUDED from both experiments (temporal leakage):** it is built from call_sites of the method and its siblings, so it inherits the dependence on the future commit's changed-file list.

```python
def calls(n):
    return len(re.findall(r"\b" + re.escape(n) + r"\s*\(", commit_before_text)) - 1
vals = [calls(n) for (_, n) in eligible]
print(round(calls(name) - sum(vals) / len(vals), 3))
```

Example output: `2.273`

#### 44. `rel_name_body_overlap`

- **Group:** Relative-to-file
- **What it measures:** Name-body overlap minus the file average.
- **How it is calculated:** Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.
- **Why it may help predict a rename:** A name that matches its body less well than its neighbours do.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Main + comparison
- **Leakage or redundancy concerns:** Correlated with name_body_overlap. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).

```python
vals = [code_features(f, n)["name_body_overlap"] for (_, n), (f, _, _) in eligible.items()]
print(round(code_features(func, name)["name_body_overlap"] - sum(vals) / len(vals), 3))
```

Example output: `-0.159`

### File-size features

#### 45. `file_n_methods`

- **Group:** File-size
- **What it measures:** Number of methods and functions in the Before file.
- **How it is calculated:** `len(collect(tree))`: every function directly in the module or in a class.
- **Why it may help predict a rename:** Renamed methods sit in files with fewer methods.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Comparison only
- **Leakage or redundancy concerns:** Strong but mostly a dilution effect (a file with a rename has ~1.4 renamed methods whatever its size); showed sampling bias in v1. Run with and without. Correlated with file_loc (r = 0.879).

```python
print(len(methods))
```

Example output: `12`

#### 46. `file_loc`

- **Group:** File-size
- **What it measures:** Number of lines in the Before file.
- **How it is calculated:** `src.count('\n') + 1`.
- **Why it may help predict a rename:** Renamed methods sit in smaller files.
- **Available before the commit:** Yes, computed from the Before code only.
- **Training use:** Comparison only
- **Leakage or redundancy concerns:** Same dilution effect as file_n_methods; strongest single feature. Run with and without.

```python
print(src.count("\n") + 1)
```

Example output: `299`

## 6. Final verification

| Check | Result | Detail |
| --- | --- | --- |
| Master header has exactly 46 predictive features | PASS | 46 |
| v2 header has 46 predictive features | PASS | 46 |
| v1 header has 34 predictive features | PASS | 34 |
| Schema feature list equals the master header features (same names, same order) | PASS | 46 in schema |
| Master and v2 have the same 46 feature names in the same order | PASS |  |
| Metadata columns excluded from X | PASS | 8 meta_* columns |
| Identifiers excluded from X | PASS | commit, file, class, name |
| Target excluded from X | PASS | renamed |
| Grouping column excluded from X | PASS | meta_project_group |
| Every master column is exactly one of: identifier, target, feature, metadata | PASS | 4 + 1 + 46 + 8 = 59 |
| v1: group counts sum to its feature count | PASS | 34 = 34 |
| v2: group counts sum to its feature count | PASS | 46 = 46 |
| Master: group counts sum to its feature count | PASS | 46 = 46 |
| Recalculated features are exactly call_sites + the 9 rel_* features | PASS | call_sites, rel_name_len, rel_name_tokens, rel_loc, rel_n_params, rel_n_calls, rel_cyclomatic, rel_doc_len, rel_call_sites, rel_name_body_overlap |
| Main experiment has exactly 40 features | PASS | 40 |
| Comparison experiment has exactly 42 features | PASS | 42 |
| Comparison = main + file_loc + file_n_methods | PASS |  |
| Both sets exclude call_sites and rel_call_sites | PASS |  |
| Both sets are subsets of the master header features | PASS |  |
| Both sets contain no identifier, target or metadata column | PASS |  |
| Master dataset still has all 46 features (unchanged) | PASS |  |
| No CSV was modified by this inspection | PASS | read-only (pandas.read_csv only) |
| All 46 code examples print the master value | PASS | 46 of 46 matched |

Selecting X the safe way:

```python
import json, pandas as pd
schema = json.load(open("data/processed/master/actref_master_dataset_schema.json"))
df = pd.read_csv("data/processed/master/actref_master_dataset.csv", keep_default_na=False)
X_all = df[schema["roles"]["feature"]]                   # 46 audited features (master, unchanged)
EXCLUDE = ["call_sites", "rel_call_sites", "class_depth", "is_async"]
X_comparison = X_all.drop(columns=EXCLUDE)                # 42
X_main = X_comparison.drop(columns=["file_loc", "file_n_methods"])   # 40
y = df["renamed"]
groups = df["meta_project_group"]                         # for StratifiedGroupKFold, not an input
```
