"""
Tests for Stage 6, evaluation metrics, threshold selection, CV diagnostics and the
project bootstrap (src/evaluation/metrics.py). Synthetic and hand-built data only;
no model is trained and no real predictions exist.
Run: python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
nan = float("nan")


def random_case(n=500, rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    y[:2] = [1, 0]
    p = np.clip(rng.random(n) * 0.7 + y * rng.random(n) * 0.3, 0, 1)
    return y, p


class MetricTests(unittest.TestCase):
    def test_average_precision_by_hand_and_against_sklearn(self):
        y, p = [0, 1, 1, 0, 1], [0.1, 0.9, 0.8, 0.4, 0.3]
        # ranked: .9(1) .8(1) .4(0) .3(1) .1(0) -> AP = 1/3*1 + 1/3*1 + 1/3*3/4 = 11/12
        self.assertAlmostEqual(M.average_precision(y, p), 11 / 12)
        self.assertAlmostEqual(M.average_precision(y, p), average_precision_score(y, p))
        for seed in range(5):
            yy, pp = random_case(seed=seed)
            self.assertAlmostEqual(M.average_precision(yy, pp), average_precision_score(yy, pp))
            self.assertAlmostEqual(M.roc_auc(yy, pp), roc_auc_score(yy, pp))

    def test_ap_is_not_trapezoidal_pr_auc(self):
        y, p = [0, 1, 1, 0, 1], [0.1, 0.9, 0.8, 0.4, 0.3]
        precision, recall, _ = precision_recall_curve(y, p)
        self.assertAlmostEqual(M.trapezoidal_pr_auc(y, p), auc(recall, precision))
        self.assertNotAlmostEqual(M.trapezoidal_pr_auc(y, p), M.average_precision(y, p), places=3)
        self.assertEqual(M.PRIMARY_METRIC, "average_precision")
        self.assertNotIn("trapezoidal_pr_auc", M.METRICS)

    def test_threshold_metrics_by_hand_and_against_sklearn(self):
        y, pred = [1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0]        # TP 2, FP 1, FN 1, TN 2
        self.assertEqual(M.confusion_counts(y, pred), {"tp": 2, "fp": 1, "fn": 1, "tn": 2})
        m = M.threshold_metrics(y, pred)
        for key, value in {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "mcc": 1 / 3}.items():
            self.assertAlmostEqual(m[key], value)
        rng = np.random.default_rng(3)
        for _ in range(5):
            yy, pr = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
            m = M.threshold_metrics(yy, pr)
            self.assertAlmostEqual(m["precision"], precision_score(yy, pr))
            self.assertAlmostEqual(m["recall"], recall_score(yy, pr))
            self.assertAlmostEqual(m["f1"], f1_score(yy, pr))
            self.assertAlmostEqual(m["mcc"], matthews_corrcoef(yy, pr))

    def test_undefined_metrics_are_nan_not_zero(self):
        m = M.threshold_metrics([1, 0, 1, 0], [0, 0, 0, 0])      # positives, no positive prediction
        self.assertTrue(np.isnan(m["precision"]) and np.isnan(m["mcc"]))
        self.assertEqual((m["recall"], m["f1"]), (0.0, 0.0))     # well defined: 2TP / (2TP+FP+FN) = 0
        self.assertEqual(matthews_corrcoef([1, 0, 1, 0], [0, 0, 0, 0]), 0.0)   # what sklearn would hide
        m = M.threshold_metrics([0, 0, 0], [0, 0, 0])            # no positives at all
        self.assertTrue(all(np.isnan(m[k]) for k in ("precision", "recall", "f1", "mcc")))
        self.assertTrue(np.isnan(M.average_precision([0, 0, 0], [0.1, 0.2, 0.3])))
        self.assertTrue(np.isnan(M.roc_auc([1, 1], [0.1, 0.2])))

    def test_evaluate_uses_score_at_least_threshold(self):
        e = M.evaluate([0, 1, 1, 0], [0.2, 0.5, 0.7, 0.5], threshold=0.5)
        self.assertEqual((e["tp"], e["fp"], e["fn"], e["tn"]), (2, 1, 0, 1))
        self.assertEqual(set(M.METRICS) <= set(e), True)

    def test_invalid_labels_and_probabilities(self):
        for y in ([0, 2, 1], [True, False], ["0", "1"], [], [[0, 1]], [0.5, 1]):
            with self.assertRaises(M.EvaluationInputError, msg=str(y)):
                M.validate_labels(y)
        for p in ([0.1, nan], [0.1, np.inf], [-0.1, 0.5], [0.5, 1.1], [[0.1, 0.2]], ["a", "b"], [True, False]):
            with self.assertRaises(M.EvaluationInputError, msg=str(p)):
                M.validate_probabilities(p)
        with self.assertRaises(M.EvaluationInputError):
            M.average_precision([0, 1, 1], [0.1, 0.2])
        with self.assertRaises(M.EvaluationInputError):
            M.evaluate(pd.Series([0, 1], index=[5, 6]), pd.Series([0.1, 0.9], index=[6, 5]), 0.5)
        self.assertTrue(issubclass(M.EvaluationInputError, ValueError))


class ThresholdTests(unittest.TestCase):
    def test_hand_built_case(self):
        # thresholds .1 / .35 / .4 / .8 give F1 4/6, 4/5, 2/4, 2/3 -> .35
        s = M.select_max_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual((s.threshold, s.n_tied, s.n_candidates), (0.35, 1, 4))
        self.assertAlmostEqual(s.f1, 0.8)
        self.assertAlmostEqual(s.precision, 2 / 3)
        self.assertAlmostEqual(s.recall, 1.0)

    def test_ties_choose_the_lowest_threshold_and_are_counted(self):
        # .9(1) .7(0) .5(0) .3(1) .1(0): F1 at .9 = 2/3 and at .3 = 4/6 (exactly equal)
        s = M.select_max_f1_threshold([1, 0, 0, 1, 0], [0.9, 0.7, 0.5, 0.3, 0.1])
        self.assertEqual((s.threshold, s.n_tied), (0.3, 2))
        self.assertAlmostEqual(s.f1, 2 / 3)
        self.assertEqual((s.precision, s.recall), (0.5, 1.0))
        s2 = M.select_max_f1_threshold([1, 0, 0, 1, 0][::-1], [0.9, 0.7, 0.5, 0.3, 0.1][::-1])  # input order irrelevant
        self.assertEqual(s, s2)

    def test_matches_brute_force_and_alignment(self):
        for seed in range(6):
            y, p = random_case(n=300, seed=seed)
            p = np.round(p, 2)                                   # many repeated scores -> ties
            best, best_t, tied = -1.0, None, 0
            for t in np.unique(p):                               # ascending
                f = M.threshold_metrics(y, (p >= t).astype(int))["f1"]
                if f > best + 1e-12:
                    best, best_t, tied = f, t, 1
                elif abs(f - best) <= 1e-12:
                    tied += 1
            s = M.select_max_f1_threshold(y, p)
            self.assertEqual(s.threshold, best_t, seed)
            self.assertAlmostEqual(s.f1, best)
            self.assertEqual(s.n_tied, tied)
            direct = M.threshold_metrics(y, (p >= s.threshold).astype(int))
            self.assertAlmostEqual(s.precision, direct["precision"])
            self.assertAlmostEqual(s.recall, direct["recall"])
            precision, recall, thresholds = precision_recall_curve(y, p)
            i = int(np.flatnonzero(thresholds == s.threshold)[0])
            self.assertAlmostEqual((precision[i], recall[i]), (s.precision, s.recall))

    def test_input_validation(self):
        for y, p in (([0, 0, 0], [0.1, 0.2, 0.3]), ([1, 1], [0.1, 0.2]), ([0, 1], [0.1, nan]),
                     ([0, 1], [0.1, 1.5]), ([0, 1, 1], [0.1, 0.2]), ([0, 2], [0.1, 0.2])):
            with self.assertRaises(M.EvaluationInputError, msg=f"{y} {p}"):
                M.select_max_f1_threshold(y, p)
        self.assertIn("optimistic", M.ThresholdSelection.__dataclass_fields__["scope"].default)


class FoldDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.y, self.p = random_case(n=100, rate=0.2, seed=7)
        order = np.random.default_rng(1).permutation(100)
        vals = np.array_split(order, 5)
        self.folds = [(np.setdiff1d(np.arange(100), v), np.sort(v)) for v in vals]

    def test_assemble_oof_puts_scores_back_in_row_order(self):
        parts = [self.p[val] for _, val in self.folds]
        oof = M.assemble_oof(parts, self.folds, 100)
        np.testing.assert_array_equal(oof, self.p)
        idx = pd.Index(np.arange(100) * 3 + 11)
        s = M.assemble_oof(parts, self.folds, 100, index=idx)
        self.assertTrue(s.index.equals(idx))
        with self.assertRaises(M.EvaluationInputError):
            M.assemble_oof(parts[:-1], self.folds, 100)
        with self.assertRaises(M.EvaluationInputError):
            M.assemble_oof([p[:-1] for p in parts], self.folds, 100)

    def test_coverage_and_order_are_validated(self):
        M.validate_folds(self.folds, 100)
        broken = {"row missing": [(t, v[1:]) for t, v in self.folds],
                  "row twice": self.folds[:4] + [(self.folds[4][0], np.r_[self.folds[4][1], self.folds[0][1][:1]])],
                  "train overlaps validation": [(np.arange(100), v) for _, v in self.folds],
                  "out of range": self.folds[:4] + [(self.folds[4][0], np.r_[self.folds[4][1], 100])]}
        for label, folds in broken.items():
            with self.assertRaises(M.EvaluationInputError, msg=label):
                M.validate_folds(folds, 100)
        y = pd.Series(self.y, index=np.arange(100) + 500)
        M.validate_oof(pd.Series(self.p, index=y.index), y, self.folds)
        with self.assertRaises(M.EvaluationInputError):
            M.validate_oof(pd.Series(self.p, index=y.index[::-1]), y, self.folds)    # wrong order
        with self.assertRaises(M.EvaluationInputError):
            M.validate_oof(self.p[:-1], self.y, self.folds)
        bad = self.p.copy(); bad[3] = nan
        with self.assertRaises(M.EvaluationInputError):
            M.validate_oof(bad, self.y, self.folds)

    def test_cv_ap_is_the_mean_of_fold_aps(self):
        res = M.fold_average_precision(self.y, self.p, self.folds)
        per = [average_precision_score(self.y[v], self.p[v]) for _, v in self.folds]
        np.testing.assert_allclose(res["per_fold"], per)
        self.assertAlmostEqual(res["mean"], float(np.mean(per)))
        self.assertAlmostEqual(res["std"], float(np.std(per, ddof=1)))
        self.assertEqual((res["n_folds"], res["std_ddof"]), (5, 1))
        pooled = average_precision_score(self.y, self.p)
        self.assertNotAlmostEqual(res["mean"], pooled, places=6)                 # never substituted

    def test_fold_threshold_diagnostics(self):
        d = M.fold_threshold_diagnostics(self.y, self.p, self.folds)
        expected = [M.select_max_f1_threshold(self.y[v], self.p[v]).threshold for _, v in self.folds]
        self.assertEqual(d["per_fold"], expected)
        self.assertAlmostEqual(d["range"], max(expected) - min(expected))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        # 3 positive-containing projects (A, B, C) and 4 negative-only projects (D-G) of different sizes
        sizes = {"A": 3, "B": 5, "C": 2, "D": 4, "E": 6, "F": 1, "G": 3}
        self.projects = np.concatenate([[k] * n for k, n in sizes.items()])
        self.y = np.zeros(len(self.projects), int)
        self.y[[0, 3, 4, 8]] = 1                                  # rows in A, B, B, C
        rng = np.random.default_rng(0)
        self.p = np.clip(rng.random(len(self.y)) * 0.6 + self.y * 0.3, 0, 1)
        self.pred = (self.p >= 0.5).astype(int)
        self.sizes = sizes

    def test_resamples_whole_projects_stratified(self):
        rows = {k: set(np.flatnonzero(self.projects == k)) for k in self.sizes}
        for idx in M.bootstrap_indices(self.y, self.projects, n_boot=200, seed=1):
            drawn = pd.Series(self.projects[idx]).value_counts()
            for k, count in drawn.items():
                self.assertEqual(count % self.sizes[k], 0)        # whole projects only
                times = count // self.sizes[k]
                got = pd.Series(idx[self.projects[idx] == k]).value_counts()
                self.assertEqual(set(got.index), rows[k])         # every row of the project ...
                self.assertTrue((got == times).all())             # ... repeated as often as the project
            draws = {k: c // self.sizes[k] for k, c in drawn.items()}
            self.assertEqual(sum(draws.get(k, 0) for k in "ABC"), 3)   # positive projects keep their count
            self.assertEqual(sum(draws.get(k, 0) for k in "DEFG"), 4)
            self.assertTrue(self.y[idx].any() and (self.y[idx] == 0).any())

    def test_reproducible_and_seed_sensitive(self):
        a = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=300, seed=42)
        b = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=300, seed=42)
        c = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=300, seed=7)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(c))
        self.assertEqual(a.attrs["n_boot"], 300)
        self.assertEqual((a.attrs["n_positive_projects"], a.attrs["n_negative_projects"]), (3, 4))
        self.assertIn("observed numbers", a.attrs["conditional_on"])
        self.assertEqual((M.N_BOOT, M.BOOT_SEED), (2000, 42))

    def test_percentile_intervals_and_point_estimates(self):
        out, reps = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=400, return_replicates=True)
        self.assertEqual(list(out.index), list(M.METRICS))
        for m in M.METRICS:
            v = reps[m].dropna()
            self.assertAlmostEqual(out.loc[m, "ci_low"], np.percentile(v, 2.5))
            self.assertAlmostEqual(out.loc[m, "ci_high"], np.percentile(v, 97.5))
            self.assertEqual(out.loc[m, "n_valid"] + out.loc[m, "n_undefined"], 400)
        self.assertAlmostEqual(out.loc["average_precision", "point"], average_precision_score(self.y, self.p))
        out90 = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=400, confidence=0.9)
        self.assertGreaterEqual(out90.loc["roc_auc", "ci_low"], out.loc["roc_auc", "ci_low"])

    def test_single_project_per_stratum_gives_a_degenerate_interval(self):
        projects = np.array(["P"] * 4 + ["N"] * 3)
        y = np.array([1, 0, 1, 0, 0, 0, 0])
        p = np.array([0.9, 0.2, 0.6, 0.4, 0.3, 0.7, 0.1])
        out = M.project_bootstrap(y, p, (p >= 0.5).astype(int), projects, n_boot=50)
        np.testing.assert_allclose(out["ci_low"], out["point"])
        np.testing.assert_allclose(out["ci_high"], out["point"])

    def test_undefined_precision_is_counted_not_zeroed(self):
        pred = np.zeros(len(self.y), int)                         # nothing predicted positive
        out = M.project_bootstrap(self.y, self.p, pred, self.projects, n_boot=100)
        self.assertEqual(out.loc["precision", "n_undefined"], 100)
        self.assertTrue(np.isnan(out.loc["precision", "ci_low"]) and np.isnan(out.loc["precision", "point"]))
        self.assertEqual(out.loc["mcc", "n_undefined"], 100)
        self.assertEqual((out.loc["f1", "n_undefined"], out.loc["f1", "ci_high"]), (0, 0.0))
        self.assertEqual(out.loc["average_precision", "n_undefined"], 0)

    def test_input_validation(self):
        args = (self.y, self.p, self.pred, self.projects)
        bad_projects = self.projects.astype(object).copy(); bad_projects[2] = ""
        none_projects = self.projects.astype(object).copy(); none_projects[2] = None
        cases = {"length": (self.y[:-1], self.p, self.pred, self.projects),
                 "projects length": (self.y, self.p, self.pred, self.projects[:-1]),
                 "empty project": (self.y, self.p, self.pred, bad_projects),
                 "missing project": (self.y, self.p, self.pred, none_projects),
                 "bad label": (np.r_[self.y[:-1], 2], self.p, self.pred, self.projects),
                 "bad score": (self.y, np.r_[self.p[:-1], 1.2], self.pred, self.projects),
                 "bad prediction": (self.y, self.p, np.r_[self.pred[:-1], 3], self.projects),
                 "one class": (np.zeros_like(self.y), self.p, self.pred, self.projects),
                 "no negative-only project": (self.y, self.p, self.pred, np.where(self.y == 1, "A", "A"))}
        for label, a in cases.items():
            with self.assertRaises(M.EvaluationInputError, msg=label):
                M.project_bootstrap(*a, n_boot=10)
        for kw in ({"n_boot": 0}, {"n_boot": 2.5}, {"confidence": 1.0}):
            with self.assertRaises(M.EvaluationInputError, msg=str(kw)):
                M.project_bootstrap(*args, **kw)

    def test_pandas_inputs_are_positional_when_aligned(self):
        idx = np.arange(len(self.y)) * 7 + 100
        series = [pd.Series(v, index=idx) for v in (self.y, self.p, self.pred, self.projects)]
        a = M.project_bootstrap(*series, n_boot=50)
        b = M.project_bootstrap(self.y, self.p, self.pred, self.projects, n_boot=50)
        pd.testing.assert_frame_equal(a, b)
        shuffled = series[1].sample(frac=1, random_state=0)
        with self.assertRaises(M.EvaluationInputError):
            M.project_bootstrap(series[0], shuffled, series[2], series[3], n_boot=10)


class ReportingTests(unittest.TestCase):
    """AP and ROC-AUC come from continuous probabilities; only the threshold metrics depend on the threshold."""

    def setUp(self):
        self.y, self.p = random_case(n=400, rate=0.1, seed=11)

    def test_overall_result_contains_ap_first(self):
        e = M.evaluate(self.y, self.p, 0.5)
        self.assertEqual(list(M.METRICS)[0], "average_precision")
        self.assertEqual(M.SCORE_METRICS, ("average_precision", "roc_auc"))
        self.assertEqual(M.THRESHOLD_METRICS, ("precision", "recall", "f1", "mcc"))
        self.assertTrue(set(M.METRICS) <= set(e))
        self.assertAlmostEqual(e["average_precision"], average_precision_score(self.y, self.p))
        self.assertNotIn("average_precision", M.ThresholdSelection.__dataclass_fields__)   # threshold-specific only

    def test_roc_auc_and_ap_use_continuous_probabilities(self):
        e = M.evaluate(self.y, self.p, 0.5)
        pred = (self.p >= 0.5).astype(int)
        self.assertAlmostEqual(e["roc_auc"], roc_auc_score(self.y, self.p))
        self.assertNotAlmostEqual(e["roc_auc"], roc_auc_score(self.y, pred), places=6)     # not from 0/1
        self.assertNotAlmostEqual(e["average_precision"], average_precision_score(self.y, pred), places=6)

    def test_threshold_changes_only_the_threshold_metrics(self):
        results = [M.evaluate(self.y, self.p, t) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for r in results[1:]:
            self.assertEqual(r["average_precision"], results[0]["average_precision"])
            self.assertEqual(r["roc_auc"], results[0]["roc_auc"])
        predicted = [r["tp"] + r["fp"] for r in results]
        recalls = [r["recall"] for r in results]
        self.assertEqual(predicted, sorted(predicted, reverse=True))                       # fewer flagged
        self.assertEqual(recalls, sorted(recalls, reverse=True))                          # recall cannot rise
        self.assertGreater(len({round(r["f1"], 9) for r in results}), 1)
        self.assertGreater(len({round(r["precision"], 9) for r in results}), 1)

    def test_bootstrap_ap_and_roc_ignore_the_predictions(self):
        projects = np.arange(len(self.y)) % 20
        y = self.y.copy(); y[projects >= 15] = 0                                          # 5 negative-only projects
        y[np.flatnonzero(projects < 15)[:15]] = 1
        low = M.project_bootstrap(y, self.p, (self.p >= 0.2).astype(int), projects, n_boot=100, threshold=0.2)
        high = M.project_bootstrap(y, self.p, (self.p >= 0.8).astype(int), projects, n_boot=100, threshold=0.8)
        pd.testing.assert_frame_equal(low.loc[list(M.SCORE_METRICS)], high.loc[list(M.SCORE_METRICS)])
        self.assertFalse(low.loc[list(M.THRESHOLD_METRICS)].equals(high.loc[list(M.THRESHOLD_METRICS)]))
        with self.assertRaisesRegex(M.EvaluationInputError, "differs"):
            M.project_bootstrap(y, self.p, (self.p >= 0.2).astype(int), projects, n_boot=10, threshold=0.8)

    def test_fold_ap_mean_and_sample_std_by_hand(self):
        # fold 0 ranks its positive first (AP 1); fold 1 ranks it last of 2 (AP 1/2); fold 2: AP 1
        y = np.array([1, 0, 0, 1, 1, 0])
        p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])), (np.array([0, 1, 4, 5]), np.array([2, 3])),
                 (np.array([0, 1, 2, 3]), np.array([4, 5]))]
        res = M.fold_average_precision(y, p, folds)
        self.assertEqual(res["per_fold"], [1.0, 0.5, 1.0])
        self.assertAlmostEqual(res["mean"], 2.5 / 3)
        self.assertAlmostEqual(res["std"], np.sqrt(((1 / 6) ** 2 * 2 + (1 / 3) ** 2) / 2))   # ddof = 1
        self.assertNotAlmostEqual(res["std"], np.std([1.0, 0.5, 1.0]))                      # not ddof = 0
        self.assertEqual(res["std_ddof"], 1)

    def test_protocol_records_the_fold_summary_as_approved(self):
        protocol = (config.DOCS / "experiment_protocol.md").read_text(encoding="utf-8")
        self.assertIn("| Report mean ± sample standard deviation (ddof=1) over the 5 validation folds | Approved |", protocol)
        self.assertIn("ROC-AUC (threshold-independent", protocol)


class ConfigTests(unittest.TestCase):
    def test_configs_name_average_precision(self):
        for path in sorted(config.EXPERIMENT_CONFIGS.glob("*.json")):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["primary_metric"], M.PRIMARY_METRIC, path.name)
            self.assertTrue(set(cfg["secondary_metrics"]) <= set(M.METRICS), path.name)
        protocol = (config.DOCS / "experiment_protocol.md").read_text(encoding="utf-8")
        self.assertIn("Primary metric: Average Precision (AP", protocol)


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class InputsUnchangedTests(unittest.TestCase):
    def test_master_and_split_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        self.assertEqual(sp.load_splits()[1]["split_hash"], SPLIT_HASH)


if __name__ == "__main__":
    unittest.main()
