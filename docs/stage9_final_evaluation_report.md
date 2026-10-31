# Stage 9 final held-out evaluation: report

**Executed once:** 2026-10-31, 16:57:41–16:57:58 UTC, under `docs/stage9_final_evaluation_protocol.md`
(SHA-256 `3e3e8613…`, recorded in provenance). Artifacts are in
`results/runs/2026-10-31_stage9_final_test/` and the fitted pipelines in
`results/models/2026-10-31_stage9_final_test/`.

> **Label limitation.** All 243 positive labels, including **all 50 test positives** (39 rule-accepted,
> 11 AI-reviewed), were constructed automatically and **no person has verified them**. Every number
> below measures agreement with these automatic labels on unseen projects. It is not human-verified
> accuracy. The human review package is prepared in `reports/label_review/` but not yet completed.

**Test set:** 6,242 methods from 62 projects never seen in development. 50 positives come from **12
positive-containing projects**. Prevalence is 0.0080, which is also the random-ranking AP baseline.

## Results

| | **Main 40 (primary)** | Comparison 42 (secondary) |
|---|---|---|
| Model (selected on development CV only) | LR + Yeo-Johnson | LR + Yeo-Johnson |
| **Test AP** [95% project-bootstrap interval] | **0.0160** [0.0049, 0.0376] | **0.0442** [0.0145, 0.1197] |
| Random AP baseline (test prevalence) | 0.0080 | 0.0080 |
| AP / baseline | 2.0× | 5.5× |
| Test ROC-AUC [interval] | 0.690 [0.543, 0.758] | 0.847 [0.671, 0.923] |
| Development CV AP (Stage 8, mean ± sd over 5 folds) | 0.0133 ± 0.0035 | 0.0191 ± 0.0089 |
| Test AP minus development CV AP | +0.0028 | +0.0251 |
| Saved threshold (Stage 8, pooled development OOF max F1) | 0.859356 | 0.845942 |
| Test predicted positive | 80 (1.28%) | 125 (2.00%) |
| Test confusion matrix (TP / FP / FN / TN) | 1 / 79 / 49 / 6,113 | 6 / 119 / 44 / 6,073 |
| Test precision / recall | 0.0125 [0, 0.075] / 0.020 [0, 0.098] | 0.048 [0, 0.089] / 0.120 [0, 0.328] |
| Test F1 / MCC | 0.015 / 0.006 [−0.015, 0.065] | 0.069 / 0.064 [−0.017, 0.155] |
| Development OOF metrics at threshold (**optimistic**, not test performance) | P 0.021, R 0.057, F1 0.030, MCC 0.022 | P 0.025, R 0.124, F1 0.042, MCC 0.040 |

**About the intervals:**
- They come from the stratified project bootstrap (2,000 replicates, seed 42), with all 2,000
  replicates valid for every metric.
- They are **conditional on the observed 12 positive-containing and 50 negative-only projects** and
  are approximate. With so few positive projects, nominal 95% coverage is not guaranteed.
- No paired test of main vs comparison was specified, so none is reported.

## Answers

1. **Which model was selected?** Logistic Regression with Yeo-Johnson preprocessing, for both feature
   sets. It was chosen by the fixed Stage 8 rule on development CV AP alone. The margin over LR with
   log1p/asinh was a near-tie (0.0003 / 0.0004 AP).

2. **How well does it perform on unseen projects?** **Main 40 (primary): weakly.**
   - AP is 0.016, twice the random baseline, and ROC-AUC is 0.69.
   - The AP interval [0.005, 0.038] **includes the random baseline**, so main 40 is not reliably better
     than chance at ranking the rare renames.
   - The ROC-AUC interval stays above 0.5, which suggests a weak ordering signal overall.
   - At the saved threshold it flags 80 methods, of which **1** is a true rename; 49 of 50 renames are
     missed.
   - **No positive appears among the 50 highest-scored methods.** The median positive ranks 1,459th of
     6,242.

3. **Does it outperform the random AP baseline?**
   - **Main 40:** its point estimate is higher (2.0×), but the interval includes the baseline, so it
     **is not clearly better**.
   - **Comparison 42:** yes, by point estimate (5.5×), and its interval lower bound (0.0145) is above
     the baseline. Item 7 below explains why this needs caution.

4. **How much did performance change from development to test?** Test AP was slightly **higher** than
   development CV AP for both feature sets: +0.003 (main) and +0.025 (comparison).
   - Given the development fold spread and the wide test intervals, this is **consistent with noise**,
     not evidence of better generalization.
   - There is also no sign of a collapse on unseen projects.
   - **The threshold did not transfer well.** The share of methods flagged fell from 2.1% on the
     development OOF scores to 1.3% on test (main), and from 3.8% to 2.0% (comparison). This is the
     known limitation of applying an OOF-chosen threshold to a single model refitted on all development
     data.

5. **What was missed?** Almost everything at the threshold, for main 40.
   - **By rename type (main 40):**
     - visibility-only renames (11, e.g. adding or removing a leading underscore) rank poorly (median
       2,020);
     - restructured renames rank best (median 787).
   - **By project:** `streamlit` (19 of the 50 test positives) has no positive in the top 2.5% of
     methods under either model. Several single-positive projects rank near the bottom.
   - **By label source:** the rule-accepted and AI-reviewed positives behave similarly.
   - **False positives** are spread over 10 (main) and 17 (comparison) of the 50 negative-only
     projects.

   Error analysis per method is in `error_analysis.csv`; it was not used to change anything.

6. **How stable are the results?** Not very.
   - The intervals are wide. For example, comparison AP spans [0.015, 0.120] and main ROC-AUC spans
     [0.54, 0.76].
   - The threshold metrics' intervals include 0.
   - Results rest on 12 positive projects, and one project, `syft` (10 positives), dominates the
     comparison result (item 7).

7. **Does comparison 42 improve on main 40?** On these test projects, **yes, numerically**:
   - AP 0.044 vs 0.016; ROC-AUC 0.85 vs 0.69;
   - 21 vs 10 positives in the top 10% of methods.

   This matches the Stage 8 development direction. However, the gain is **concentrated**:
   - 5 of comparison's 6 true positives, and its best-ranked positive (rank 2), come from **one
     project, `syft`**;
   - the gain comes mainly from word-added renames, 5 of whose positives are in `syft`;
   - the added features are file size (`file_loc`, `file_n_methods`), which may partly reflect how
     ActRef collects changed files rather than how renaming works.

   Comparison 42 is the secondary experiment, and this result does not change the primary conclusion.

8. **Limitations from automatically constructed labels:**
   - The test labels are unverified: 39 come from a similarity rule that is known to have made errors,
     and 11 from an AI review that sometimes saw truncated code.
   - Mislabelled positives or missed renames among the negatives would bias AP, recall and precision in
     unknown directions.
   - Renames below the detector's 0.70 similarity, or combined with a move, are absent from the labels
     altogether.
   - Every result should be read as agreement with the automatic labelling, and **re-evaluated after
     the human review**. That re-evaluation would be a new, separately approved held-out evaluation on
     a new dataset version.

9. **Practical use?** **No. It remains an experimental research prototype.**
   - At its operating point, main 40 finds 1 of 50 renames while raising 79 false alarms.
   - Comparison 42 finds 6 while raising 119.
   - Neither is precise enough to rank or flag methods for developers.
   - The models show only a weak signal that pre-change code characteristics carry information about
     future renames.

## Conclusion

On 62 previously unseen projects, a Logistic Regression trained on pre-change method characteristics
ranks future renames only slightly better than chance with the primary 40 features (AP 0.016 vs 0.008;
ROC-AUC 0.69), and its AP interval includes chance. Adding file size (comparison 42) gives a clearer but
fragile improvement (AP 0.044, ROC-AUC 0.85), carried largely by one project. Performance on unseen
projects is in line with development CV.

**Pre-change code characteristics contain a weak, unstable signal about which methods will be renamed.
It is not strong enough for practical prediction, and it is measured against labels no person has yet
verified.**

## Provenance

| Item | Value |
|---|---|
| Master dataset | SHA-256 `e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0` (unchanged) |
| Split hash | `16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c` (unchanged); no project shared between development and test |
| Code | git commit `8052c6d` plus the uncommitted Stage 9 runner. SHA-256 of all source files is in `provenance.json` |
| Environment | Python 3.12.0; scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, lightgbm 4.7.0 |
| Fit | On all 24,967 development rows only. 0 warnings, no convergence problem |
| Threshold | Recomputed bit-exact from the saved Stage 8 OOF scores before use |
| Tests | 13 synthetic Stage 9 tests and 5 sabotage checks. Full suite 208 passed, 1 skipped, before execution |
