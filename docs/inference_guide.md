# Inference guide: local rename-risk web application (Stage 10)

A local web application that scores the methods of an uploaded Python file with the two Stage 9 models.

> **Experimental research prototype.** All 243 positive training labels and all 50 positive test labels were
> constructed automatically and have **not** been verified by a person. On 62 unseen projects (Stage 9),
> main 40 reached Average Precision 0.0160, against a random baseline of 0.0080. Comparison 42 reached 0.0442,
> but 5 of its 6 true positives came from one project. Scores are model outputs, not calibrated probabilities
> and not proof that a method will be renamed. No prospective real-world validation has been performed. This
> is **not** a production tool.

## Install and start

```
pip install -r requirements-lock.txt -r requirements-app.txt     # research env + FastAPI/Uvicorn
python run.py                                                     # opens http://127.0.0.1:8000
```

`run.py`:
1. Loads and verifies the models once.
2. Starts Uvicorn on **127.0.0.1:8000** (this machine only).
3. Polls `GET /health` until the server answers.
4. **Opens the default browser once.** Auto-reload is off, so no duplicate tabs.

The server keeps running until you press Ctrl+C. Options: `--no-browser`, `--port 8010`, `--review-cutoff 0.6`. `--host` changes the
bind address; leave it at the localhost default unless you understand the exposure.

## Workflow

1. Drop a `.py` file on the page, or click *browse*. The limit is 2 MB.
2. Optionally give its repository-relative path, e.g. `pkg/tests/util.py`. This only affects
   `is_test_file`.
3. Click **Analyze**.
4. Read the results:
   - methods scored, and how many are above each model's threshold;
   - a ranked, searchable, sortable table with the main 40 score (primary) and comparison 42 score
     (secondary), each labelled *Above / Below model threshold*.
5. Click a method to see:
   - its location, both scores, and its rank in the file;
   - its largest contributions to the score;
   - all 42 extracted feature values;
   - its source.

The uploaded file is **never executed**: it is only parsed with Python's `ast`. It is not stored. Features are
computed in a temporary folder that is deleted at once, and the in-memory method cache keeps no source.

## Architecture

```
run.py ──► uvicorn (127.0.0.1) ──► src/app/server.py  (FastAPI)
                                     ├─ GET /                    src/app/static/{index.html, app.js, styles.css}
                                     ├─ GET /health, /api/models
                                     ├─ POST /api/analyze ──► src/app/inference.py
                                     │                          ├─ extract_file_features ──► src/data/extract_features.method_rows (training code, unchanged)
                                     │                          ├─ model_input (approved names, order, numeric, finite)
                                     │                          └─ Stage 9 pipelines (loaded once, hash-verified)
                                     └─ GET /api/methods/{id}?analysis_id=…
```

## Feature extraction and parity

The application does **not** re-implement any feature. `extract_file_features` writes the upload, bytes
unchanged, into a temporary commit folder as `Before/<path>` and an identical `After/<path>`. It then calls the
training function `method_rows` unchanged. Because the After copy is identical, every method has the label
"name kept"; inference never uses that label.

- **File context exactly as in training.** `file_loc`, `file_n_methods`, `style_mismatch`, `style_share_in_file`,
  `name_token_sharing`, `first_token_shared` and all `rel_*` features are computed over the whole uploaded file.
  `rel_*` subtracts the mean over the file's eligible methods.
- **Verified exactly against the canonical master dataset.**
  - Recomputing features one file at a time from the raw Before files reproduced **all 31,209 rows across 3,181
    files**, with **0 mismatches in 1,310,778 feature values** (all 42 features).
  - `tests/test_app.py` repeats this on a sample of 150 files, or all files with `APP_PARITY_FILES=0`.
  - It also asserts frame-equality with a direct `method_rows` call.
- **Scope** (same as training):
  - **Scored:** module-level functions and class methods, including nested classes, static and class methods,
    and decorated and async methods.
  - **Listed with a reason but not scored:**
    - dunder methods and names starting with `test`. These still count in the file context, as in training.
    - functions inside functions or inside `if`/`try`/`with` blocks;
    - an earlier definition replaced by a later one with the same name (e.g. a property getter followed by its
      setter; training keeps the last).
- **`is_test_file`** is `"test" in <repository-relative path>`. Give the original relative path to reproduce
  training exactly; otherwise the file name alone is used.
- **`call_sites` and `rel_call_sites`** would need the commit's other files. They are excluded from both models
  and never used.
- **Text handling:** decoded as UTF-8 with errors ignored, as training's `parse()` does. Files with invalid syntax
  are rejected.

## Models and thresholds

| | Main 40 (primary) | Comparison 42 (secondary) |
|---|---|---|
| Artifact | `results/models/2026-10-31_stage9_final_test/main_pipeline.joblib` | `comparison_pipeline.joblib` |
| Model | LR + Yeo-Johnson, fitted on all 24,967 development rows | same, plus `file_loc`, `file_n_methods` |
| Threshold (from Stage 8 OOF max F1, via the model metadata) | 0.859356439967941 | 0.8459416092065658 |
| Stage 9 held-out AP / ROC-AUC | 0.0160 / 0.690 | 0.0442 / 0.847 |

**Checks at startup:**
- **Before unpickling:** each artifact's SHA-256 must equal the value in the committed Stage 9
  `provenance.json`.
- **Thresholds:** they must equal the Stage 9 `run_config.json`.
- **Inputs:** feature names must equal the approved feature sets, and the pipeline's own input check must
  match.

If anything fails, the app still starts, reports **degraded** on `/health`, and refuses analysis with 503.
Models are only loaded, never refitted. Joblib files are pickles: load only these verified artifacts.

**Threshold limitation.** The threshold was chosen on the out-of-fold scores of five fold models, but is applied
to one model refitted on all development data. In Stage 9 it flagged fewer test methods than on development
(main 1.3% vs 2.1%). "Above model threshold" is therefore only a rough operating point; the ranking within a file
is more informative than the cut.

**Review cutoff 0.70 (user-chosen, not validated).** The badges and the summary counts use a **review
cutoff of 0.70**. The project owner chose this value on 2026-10-31; it is **not** a validated threshold.

- **What it trades off,** measured on the development out-of-fold scores:
  - main 40: 10.8% of methods flagged, 38 of 193 renames found (recall 19.7%), precision 1.4%;
  - for comparison, the validated 0.859: 2.1% flagged, recall 5.7%, precision 2.1%;
  - comparison 42 at 0.70: 11.6% flagged, recall 24.4%, precision 1.6%.

  So the cutoff suggests where to look; most methods it flags will not be renamed. It has not been evaluated on
  the held-out test set.
- **The validated Stage 8 thresholds are kept.** They appear in each method's details and in the model info, and
  `above_threshold` in the API still refers to them. The saved models and the Stage 8/9 artifacts are unchanged.
- **Changing it:** `python run.py --review-cutoff 0.6` (any value strictly between 0 and 1).
- **In the API:** `review_cutoff` and `above_review_cutoff` are reported next to `threshold` and
  `above_threshold`.

**Explanations.** The final estimator is linear, so each feature's contribution to the logit is exactly
`coef × transformed value`. Contributions plus the intercept equal the logit, which tests confirm. They describe
the model's arithmetic, in its transformed feature space. They are not causal reasons for a rename, and they are
not SHAP values.

## API

| Method and path | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /health` | `{"status": "ok" or "degraded", "models_loaded", "error"}` |
| `GET /api/models` | Models, features, thresholds, artifact hashes, Stage 9 results, disclaimer, threshold note |
| `POST /api/analyze` | multipart `file` (.py, ≤ 2 MB), optional `relative_path`. Returns `analysis_id`, `n_scored`, `skipped`, `above_threshold`, `review_cutoff`, `above_review_cutoff`, and `methods[]`. Each method has `name`, `class`, `lineno`, `features`, `source`, and per model `score`, `threshold`, `above_threshold`, `review_cutoff`, `above_review_cutoff`, `rank`, `logit`, `top_contributions` |
| `GET /api/methods/{method_id}?analysis_id=…` | Details of a method from one of the last 20 analyses (in memory; no source) |
| `GET /api/docs` | Interactive OpenAPI documentation |

**Errors:**
- 400 for not `.py`, invalid syntax, binary content or an invalid path;
- 413 if the file is too large;
- 422 if no file is sent;
- 503 if the models are unavailable;
- 500 with a generic message otherwise.

Messages never include server paths or tracebacks.

## Limitations

- **Weak signal,** with labels no person has verified (see the disclaimer). Not suitable for practical decisions.
- **One file at a time.** Each file is scored using only its own context. There is no repository scanner yet: it
  would be straightforward to add (`analyse` per file), but it has not been tested on whole repositories.
- **Training scope:** only methods of files that developers changed in ActRef commits from 323 projects. Other
  code bases may differ.
- **Scores are not calibrated.** Compare methods within a file; do not read a score as a probability.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: fastapi` | `pip install -r requirements-app.txt` |
| `/health` says degraded | A model artifact is missing or altered. Restore `results/models/2026-10-31_stage9_final_test/` and `results/runs/2026-10-31_stage9_final_test/` from git |
| Port 8000 in use | `python run.py --port 8010` |
| Browser did not open | Open http://127.0.0.1:8000 manually, or use `--no-browser` deliberately |
| "not valid Python" | The file has a syntax error, or is Python 2 (the parser is Python 3.12's `ast`) |
| A method is missing from the table | See "Not scored" under the results. It is outside the training scope |

## Tests

`python -m unittest tests.test_app -v` covers:
- model loading and artifact validation (altered, missing and inconsistent artifacts);
- feature parity against the master and against `method_rows`;
- scope (single function, multiple classes, module functions, static and class methods, decorators, async,
  nested definitions, redefinitions);
- empty files and files with no methods; invalid syntax, extension, binary content, path traversal and size;
- missing features and wrong column order; score and threshold validity; exact contributions;
- uploaded code never executed and temporary files removed;
- the API, its errors and that they leak no paths; degraded mode; the frontend not using HTML sinks;
- the launcher's readiness polling and its localhost default.
