# Stage 9 final held-out evaluation: summary

> Labels were constructed automatically (rule / AI review); **no person verified them**. Results measure agreement with these labels on unseen projects.

Test: 6,242 rows, 62 projects, 50 positives from 12 positive-containing projects (prevalence 0.0080 = random AP baseline).

## Main 40 (primary): logistic_regression__main__yeo_johnson

| Metric | Held-out test | 95% project-bootstrap interval | Development (Stage 8) |
|---|---|---|---|
| Average Precision (primary) | **0.0160** | 0.0049–0.0376 | CV mean 0.0133 ± 0.0035 |
| Random AP baseline (prevalence) | 0.0080 | | ≈ 0.0077 |
| ROC-AUC | 0.6900 | 0.5425–0.7576 | fold mean 0.6313 |
| precision at threshold | 0.0125 | 0.0000–0.0750 (undefined in 0) | OOF 0.0206 (optimistic) |
| recall at threshold | 0.0200 | 0.0000–0.0976 (undefined in 0) | OOF 0.0570 (optimistic) |
| f1 at threshold | 0.0154 | 0.0000–0.0715 (undefined in 0) | OOF 0.0303 (optimistic) |
| mcc at threshold | 0.0057 | -0.0148–0.0648 (undefined in 0) | OOF 0.0218 (optimistic) |

Threshold 0.859356439967941 (Stage 8, pooled development OOF max F1). Confusion matrix: TP 1, FP 79, FN 49, TN 6113; predicted positive 80 (1.28% of test rows).

## Comparison 42 (secondary): logistic_regression__comparison__yeo_johnson

| Metric | Held-out test | 95% project-bootstrap interval | Development (Stage 8) |
|---|---|---|---|
| Average Precision (primary) | **0.0442** | 0.0145–0.1197 | CV mean 0.0191 ± 0.0089 |
| Random AP baseline (prevalence) | 0.0080 | | ≈ 0.0077 |
| ROC-AUC | 0.8470 | 0.6714–0.9232 | fold mean 0.6814 |
| precision at threshold | 0.0480 | 0.0000–0.0889 (undefined in 0) | OOF 0.0254 (optimistic) |
| recall at threshold | 0.1200 | 0.0000–0.3281 (undefined in 0) | OOF 0.1244 (optimistic) |
| f1 at threshold | 0.0686 | 0.0000–0.1379 (undefined in 0) | OOF 0.0421 (optimistic) |
| mcc at threshold | 0.0641 | -0.0173–0.1549 (undefined in 0) | OOF 0.0400 (optimistic) |

Threshold 0.8459416092065658 (Stage 8, pooled development OOF max F1). Confusion matrix: TP 6, FP 119, FN 44, TN 6073; predicted positive 125 (2.00% of test rows).

Intervals: stratified project bootstrap, 2,000 replicates, seed 42; conditional on the observed 12 positive-containing / 50 negative-only test projects; approximate, no nominal-coverage guarantee.
