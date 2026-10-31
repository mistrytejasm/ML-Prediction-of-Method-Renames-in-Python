"""
Tests for the Stage 9 final evaluation runner (src/evaluation/run_stage9.py).
SYNTHETIC DATA ONLY: a synthetic master split by the approved splitter, with genuine synthetic Stage 8
run folders. The real held-out test set is never touched here.
Run: python -m unittest discover -s tests -v
"""
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage8_synthetic import synthetic_master, synthetic_split  # noqa: E402
from src.evaluation import cross_validate as cv  # noqa: E402
from src.evaluation import run_stage8 as r8  # noqa: E402
from src.evaluation import run_stage9 as r9  # noqa: E402
from src.evaluation import run_store as rs  # noqa: E402
from src.features.feature_sets import get_feature_set  # noqa: E402
from src.preprocessing.transforms import FeatureFrameCheck  # noqa: E402

SPECS = {s.run_id: s for s in r8.validate_run_config(r8.load_run_config())}
SELECTED = {"main": "logistic_regression__main__yeo_johnson", "comparison": "logistic_regression__comparison__yeo_johnson"}


def synthetic_world(seed=0):
    """Like the real data, a third of the projects contain no positive (the bootstrap needs both kinds)."""
    X, y, groups, keys = synthetic_master(seed=seed)
    y = y.where(groups.str[1:].astype(int) % 3 != 0, 0)
    assign, meta = synthetic_split(X, y, groups, keys)
    return X, y, groups, assign, meta


def loader_for(world):
    X, y, groups, assign, meta = world
    return lambda fs: (X[get_feature_set(fs)], y, groups, assign, meta)


def write_stage8(runs_dir, world, selected=SELECTED):
    """Genuine synthetic Stage 8 runs (LR Yeo-Johnson) + a selection file, as Stage 8 would write them."""
    load = loader_for(world)
    for fs, rid in selected.items():
        X, y, groups, assign, meta = load(fs)
        data = cv.select_development(X, y, groups, assign, meta)
        res = cv.run_development_cv(SPECS[rid], data)
        rs.write_run(runs_dir, f"{r9.STAGE8_DAY}_stage8_{rid}", SPECS[rid], res,
                     rs.provenance(data.split_info, {"started_utc": "t"}))
    rs.write_folder(runs_dir, f"{r9.STAGE8_DAY}_stage8_summary", {
        "stage9_selection.json": rs.text_file(json.dumps({"rule": "test", "selected": selected}))})


class Fake(ClassifierMixin, BaseEstimator):
    def __init__(self, mode="ok"):
        self.mode = mode

    def fit(self, X, y):
        self.classes_ = np.array([1, 0] if self.mode == "classes" else [0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), 0.5)
        if self.mode == "nan":
            p[0] = np.nan
        if self.mode == "above":
            p = p + 1
        return np.column_stack([1 - p, p])


class Stage9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        cls.world = synthetic_world()
        cls.runs = cls.dir / "runs"
        write_stage8(cls.runs, cls.world)
        X, y, groups, assign, meta = loader_for(cls.world)("main")
        cls.dev, cls.test, cls.counts = r9.split_partitions(X, y, groups, assign, meta)
        cls.spec, cls.thr, cls.s8, cls.ev = r9.load_stage8_selection("main", cls.runs, dev_master_rows=cls.dev.X.index)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    # ---- partitions
    def test_project_separation_and_counts(self):
        self.assertFalse(set(self.dev.groups) & set(self.test["groups"]))
        self.assertEqual(len(self.dev.X) + len(self.test["X"]), len(self.world[0]))
        self.assertTrue((self.test["keys"]["master_row"].to_numpy() == self.test["X"].index.to_numpy()).all())
        with self.assertRaises(r9.Stage9Error):
            r9.split_partitions(*loader_for(self.world)("main"), expected={"development": {}, "test": {}})

    def test_corrupted_split_is_refused(self):
        X, y, groups, assign, meta = loader_for(self.world)("main")
        a = assign.copy()
        row = a.index[a.partition == "test"][0]
        a.loc[row, ["partition", "inner_fold"]] = ["development", 0]       # a test project leaks into development
        with self.assertRaises(ValueError):
            r9.split_partitions(X, y, groups, a, meta)
        m = copy.deepcopy(meta); m["counts"]["test"]["positives"] += 1
        with self.assertRaisesRegex(r9.Stage9Error, "test counts"):
            r9.split_partitions(X, y, groups, assign, m)

    def test_second_layer_project_overlap_guard(self):
        """Even if select_development's guard were bypassed, a shared project stops Stage 9."""
        X, y, groups, assign, meta = loader_for(self.world)("main")
        leaky = copy.copy(self.dev)
        g = self.dev.groups.copy()
        g.iloc[0] = self.test["groups"].iloc[0]                          # a test project inside development
        leaky.groups = g
        with mock.patch.object(r9, "select_development", return_value=leaky):
            with self.assertRaisesRegex(r9.Stage9Error, "both development and test"):
                r9.split_partitions(X, y, groups, assign, meta)

    # ---- Stage 8 threshold
    def test_saved_threshold_is_loaded_and_verified(self):
        self.assertEqual(self.thr, self.s8["threshold_selection"]["threshold"])
        self.assertEqual(self.ev["threshold_recomputed_from_oof"], self.thr)
        self.assertEqual(self.spec.run_id, SELECTED["main"])

    def test_unverifiable_threshold_or_bad_oof_is_refused(self):
        for label, tamper in (
                ("threshold", lambda d: (d / "summary.json").write_text(json.dumps(
                    {**json.loads((d / "summary.json").read_text()), "threshold_selection": {
                        **json.loads((d / "summary.json").read_text())["threshold_selection"], "threshold": 0.5}}))),
                ("duplicate OOF row", lambda d: pd.concat([pd.read_csv(d / "oof_predictions.csv").iloc[:1],
                                                           pd.read_csv(d / "oof_predictions.csv")]).to_csv(
                    d / "oof_predictions.csv", index=False)),
                ("missing OOF score", lambda d: pd.read_csv(d / "oof_predictions.csv").assign(
                    oof_probability=lambda t: t.oof_probability.where(t.index != 3)).to_csv(d / "oof_predictions.csv", index=False))):
            runs = self.dir / f"tamper_{label.replace(' ', '_')}"
            shutil.copytree(self.runs, runs)
            tamper(runs / f"{r9.STAGE8_DAY}_stage8_{SELECTED['main']}")
            with self.assertRaises(r9.Stage9Error, msg=label):
                r9.load_stage8_selection("main", runs, dev_master_rows=self.dev.X.index)
        with self.assertRaisesRegex(r9.Stage9Error, "differ from the development rows"):
            r9.load_stage8_selection("main", self.runs, dev_master_rows=self.dev.X.index[::-1])

    def test_exploratory_selection_is_refused(self):
        runs = self.dir / "exploratory"
        runs.mkdir()
        X, y, groups, assign, meta = loader_for(self.world)("main")
        data = cv.select_development(X, y, groups, assign, meta)
        rid = "naming_heuristic__main"
        rs.write_run(runs, f"{r9.STAGE8_DAY}_stage8_{rid}", SPECS[rid], cv.run_development_cv(SPECS[rid], data),
                     rs.provenance(data.split_info))
        rs.write_folder(runs, f"{r9.STAGE8_DAY}_stage8_summary", {
            "stage9_selection.json": rs.text_file(json.dumps({"selected": {"main": rid}}))})
        with self.assertRaisesRegex(r9.Stage9Error, "confirmatory"):
            r9.load_stage8_selection("main", runs)

    # ---- fitting on development only
    def test_preprocessing_and_class_weights_fit_on_development_only(self):
        res = r9.evaluate_final(self.spec, self.thr, self.dev, self.test, n_boot=50)
        pipe = res["pipeline"]
        pre = pipe.named_steps["pre"]
        cols = list(pre.transformers_[0][2])
        ref = PowerTransformer(method="yeo-johnson").fit(self.dev.X[cols])
        np.testing.assert_allclose(pre.named_transformers_["yeo_johnson"].lambdas_, ref.lambdas_)
        self.assertEqual(int(pre.named_transformers_["raw"].n_samples_seen_), len(self.dev.X))
        # balanced weights from development labels: equal to an explicit dict computed from y_dev
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=self.dev.y.to_numpy())
        Z = pipe[:-1].transform(self.dev.X)
        explicit = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=5000, random_state=42,
                                      class_weight={0: w[0], 1: w[1]}).fit(Z, self.dev.y)
        np.testing.assert_allclose(pipe.named_steps["model"].coef_, explicit.coef_, rtol=1e-6, atol=1e-8)

    def test_test_labels_cannot_influence_the_fit(self):
        a = r9.evaluate_final(self.spec, self.thr, self.dev, self.test, n_boot=20)
        shuffled = self.test["y"].copy()
        shuffled[:] = np.random.default_rng(7).permutation(shuffled.to_numpy())   # different test labels
        self.assertFalse(shuffled.equals(self.test["y"]))
        b = r9.evaluate_final(self.spec, self.thr, self.dev, dict(self.test, y=shuffled), n_boot=20)
        np.testing.assert_array_equal(a["table"]["probability"], b["table"]["probability"])
        np.testing.assert_array_equal(a["pipeline"].named_steps["model"].coef_, b["pipeline"].named_steps["model"].coef_)

    # ---- metrics and bootstrap
    def test_metrics_and_bootstrap(self):
        res = r9.evaluate_final(self.spec, self.thr, self.dev, self.test, n_boot=100)
        m, t = res["metrics"], res["table"]
        y, p = self.test["y"].to_numpy(), t["probability"].to_numpy()
        self.assertAlmostEqual(m["average_precision"], average_precision_score(y, p))
        self.assertAlmostEqual(m["roc_auc"], roc_auc_score(y, p))
        np.testing.assert_array_equal(t["predicted"], (p >= self.thr).astype(int))
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]),
                         tuple(int((t["outcome"] == o).sum()) for o in ("TP", "FP", "FN", "TN")))
        self.assertEqual(m["random_ap_baseline"], y.mean())
        self.assertEqual(m["n_predicted_positive"], int(t["predicted"].sum()))
        b = res["bootstrap"]
        self.assertEqual(b.attrs["n_positive_projects"], int(self.test["groups"][self.test["y"] == 1].nunique()))
        self.assertEqual(b.attrs["n_boot"], 100)
        self.assertAlmostEqual(b.loc["average_precision", "point"], m["average_precision"])
        again = r9.evaluate_final(self.spec, self.thr, self.dev, self.test, n_boot=100)["bootstrap"]
        pd.testing.assert_frame_equal(b, again)                                  # seeded: reproducible
        self.assertEqual(int(res["per_project"]["rows"].sum()), len(t))

    def test_invalid_probabilities_and_class_order_fail(self):
        for mode in ("nan", "above", "classes"):
            factory = lambda s, mode=mode: Pipeline([("check", FeatureFrameCheck(s.feature_set)), ("model", Fake(mode))])
            with self.assertRaises(r9.Stage9Error, msg=mode):
                r9.evaluate_final(self.spec, self.thr, self.dev, self.test, n_boot=10, pipeline_factory=factory)

    def test_convergence_warning_fails(self):
        spec = cv.RunSpec(**{**self.spec.__dict__, "model_params": {**self.spec.model_params, "max_iter": 1}})
        with self.assertRaisesRegex(r9.Stage9Error, "ConvergenceWarning"):
            r9.evaluate_final(spec, self.thr, self.dev, self.test, n_boot=10)

    # ---- end to end and write-once
    def test_end_to_end_and_write_once(self):
        out_runs, out_models = self.dir / "e2e_runs", self.dir / "e2e_models"
        buf = io.StringIO()
        with redirect_stdout(buf):
            r9.main([], data_loader=loader_for(self.world), runs_dir=out_runs, models_dir=out_models, day="2026-01-01",
                    expected=None, n_boot=50, stage8_runs_dir=self.runs, out=print)
        folder = out_runs / "2026-01-01_stage9_final_test"
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         sorted(["run_config.json", "provenance.json", "test_predictions.csv", "test_metrics.json",
                                 "bootstrap_metrics.json", "error_analysis.csv", "summary.md"]))
        preds = pd.read_csv(folder / "test_predictions.csv", keep_default_na=False)
        self.assertEqual(len(preds), 2 * len(self.test["X"]))
        self.assertEqual(set(preds.feature_set), {"main", "comparison"})
        metrics = json.loads((folder / "test_metrics.json").read_text())
        for fs in ("main", "comparison"):
            sub = preds[preds.feature_set == fs]
            self.assertEqual(metrics[fs]["test"]["tp"], int((sub.outcome == "TP").sum()))
            self.assertIn("dev_cv_ap_mean", metrics[fs]["development"])
        self.assertIn("no person verified", (folder / "summary.md").read_text(encoding="utf-8"))
        prov = json.loads((folder / "provenance.json").read_text())
        self.assertEqual(prov["status"], "complete")
        self.assertIn("protocol_sha256", prov)
        self.assertEqual(set(prov["model_artifacts"]["sha256"]),
                         {"main_pipeline.joblib", "comparison_pipeline.joblib", "model_metadata.json"})
        model = joblib.load(out_models / "2026-01-01_stage9_final_test" / "main_pipeline.joblib")
        np.testing.assert_allclose(model.predict_proba(self.test["X"])[:, 1],
                                   preds[preds.feature_set == "main"]["probability"].to_numpy())
        with mock.patch.object(r9, "evaluate_final", side_effect=AssertionError("refit")):
            with self.assertRaises(rs.RunStoreError):                       # refused before any fitting
                r9.main([], data_loader=loader_for(self.world), runs_dir=out_runs, models_dir=out_models,
                        day="2026-01-01", expected=None, n_boot=50, stage8_runs_dir=self.runs, out=lambda *a: None)

    def test_protocol_is_frozen_with_the_saved_thresholds(self):
        text = r9.PROTOCOL.read_text(encoding="utf-8")
        for needle in ("0.859356439967941", "0.8459416092065658", "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c",
                       "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0", "No person has verified"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
