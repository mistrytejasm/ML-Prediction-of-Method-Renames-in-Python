"""
Tests for Stage 8: development cross-validation (src/evaluation/cross_validate.py), the run
store (run_store.py) and the CLI (run_stage8.py). SYNTHETIC DATA ONLY: every model is fitted on
tests/stage8_synthetic.py data; no master-dataset row is used for fitting, and the real-data
dry run is never executed here.
Run: python -m unittest discover -s tests -v
"""
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage8_synthetic import synthetic_development, synthetic_loader, synthetic_master, synthetic_split  # noqa: E402
from src import config  # noqa: E402
from src.data.load_master import sha256_of  # noqa: E402
from src.evaluation import cross_validate as cv  # noqa: E402
from src.evaluation import run_stage8 as r8  # noqa: E402
from src.evaluation import run_store as rs  # noqa: E402
from src.evaluation import splits as sp  # noqa: E402
from src.features.feature_sets import get_feature_set  # noqa: E402
from src.preprocessing.transforms import FeatureFrameCheck  # noqa: E402

CANONICAL = "e6a8fd476feb87406846327093f9d8157a1238e7bd6f9fb7a50b60e659b8a8f0"
SPLIT_HASH = "16b4fac620af776ab3b018187b4a67e4c1be27f9d009a7acbab2ef6a8039f14c"
CONFIG = r8.load_run_config()
SPECS = {s.run_id: s for s in r8.validate_run_config(CONFIG)}
DATA = {fs: synthetic_development(fs) for fs in ("main", "comparison")}


def spec(run_id, **params):
    s = SPECS[run_id]
    return cv.RunSpec(**{**s.__dict__, "model_params": {**s.model_params, **params}})


def put_value(column, value, row=5):
    def f(X):
        X = X.copy()
        X.loc[row, column] = value
        return X
    return f


class FakeModel(ClassifierMixin, BaseEstimator):
    """Test double: probability = loc / 100 (row identity), or a deliberately broken output."""

    def __init__(self, mode="identity", classes=(0, 1)):
        self.mode = mode
        self.classes = classes

    def fit(self, X, y):
        self.classes_ = np.array(self.classes)
        if self.mode == "warn":
            warnings.warn("synthetic fit warning", UserWarning)
        return self

    def predict_proba(self, X):
        p = (np.asarray(X["loc"], float) % 100) / 100
        if self.mode == "above_one":
            p = p + 1.5
        elif self.mode == "nan":
            p = p.copy(); p[0] = np.nan
        if self.mode == "one_column":
            return p[:, None]
        return np.column_stack([1 - p, p])


def fake_factory(mode="identity", classes=(0, 1)):
    return lambda s: Pipeline([("check", FeatureFrameCheck(s.feature_set)), ("model", FakeModel(mode, classes))])


# ---------------------------------------------------------------------------- configuration (22, 21, 19)
class ConfigTests(unittest.TestCase):
    def mutate(self, f):
        cfg = copy.deepcopy(CONFIG)
        f(cfg)
        return cfg

    def test_valid_config_is_the_approved_matrix(self):
        self.assertEqual([s.run_id for s in r8.validate_run_config(CONFIG)], r8.APPROVED_MATRIX)
        self.assertEqual(len(r8.APPROVED_MATRIX), 14)
        for s in SPECS.values():
            self.assertEqual(s.model_params, r8.resolve_model_params(s.model))

    def test_rejects_invalid_settings(self):
        def run(i):
            return lambda c: c["runs"][i]
        cases = {
            "unknown model": lambda c: c["runs"][0].update(model="svm"),
            "unknown feature set": lambda c: c["runs"][0].update(feature_set="all"),
            "unknown transform": lambda c: c["runs"][6].update(transform="log"),
            "transform for a tree": lambda c: c["runs"][10].update(transform="yeo_johnson"),
            "tuned parameter": lambda c: c["runs"][10]["model_params"].update(n_estimators=50),
            "unknown parameter": lambda c: c["runs"][12]["model_params"].update(num_leafs=7),
            "dropped parameter": lambda c: c["runs"][6]["model_params"].pop("max_iter"),
            "extra run key": lambda c: c["runs"][0].update(note="x"),
            "wrong run id": lambda c: c["runs"][0].update(run_id="dummy"),
            "duplicate run": lambda c: c["runs"].append(copy.deepcopy(c["runs"][0])),
            "missing run": lambda c: c["runs"].pop(),
            "reordered runs": lambda c: c["runs"].reverse(),
            "heuristic not exploratory": lambda c: c["runs"][4].update(research_status="confirmatory"),
            "model marked exploratory": lambda c: c["runs"][6].update(research_status="exploratory_post_hoc"),
            "unknown status": lambda c: c["runs"][0].update(research_status="final"),
            "wrong dataset hash": lambda c: c.update(dataset_sha256="0" * 64),
            "wrong split hash": lambda c: c.update(split_hash="0" * 64),
            "wrong primary metric": lambda c: c.update(primary_metric="roc_auc"),
            "unknown top-level key": lambda c: c.update(extra=1),
            "spread on by default": lambda c: c["optional_experiments"]["stratified_random_seed_spread"].update(run_by_default=True),
            "spread reuses seed 42": lambda c: c["optional_experiments"]["stratified_random_seed_spread"]["seeds"].__setitem__(0, 42),
            "spread with 19 seeds": lambda c: c["optional_experiments"]["stratified_random_seed_spread"]["seeds"].pop(),
            "spread other model": lambda c: c["optional_experiments"]["stratified_random_seed_spread"].update(model="lightgbm"),
        }
        for label, f in cases.items():
            with self.assertRaises((r8.ConfigError, ValueError), msg=label):
                r8.validate_run_config(self.mutate(f))

    def test_seed_spread_is_separate(self):
        spread = r8.seed_spread_specs(CONFIG)
        self.assertEqual(len(spread), 20)
        self.assertFalse({s.run_id for s in spread} & set(r8.APPROVED_MATRIX))
        self.assertEqual({s.model for s in spread}, {"stratified_random"})
        self.assertEqual(len({s.model_params["random_state"] for s in spread}), 20)
        self.assertNotIn(42, {s.model_params["random_state"] for s in spread})
        self.assertFalse(CONFIG["optional_experiments"]["stratified_random_seed_spread"]["run_by_default"])
        self.assertIn("chance", CONFIG["optional_experiments"]["stratified_random_seed_spread"]["purpose"].lower())

    def test_heuristic_labelled_exploratory(self):
        for fs in ("main", "comparison"):
            self.assertEqual(SPECS[f"naming_heuristic__{fs}"].research_status, cv.EXPLORATORY)
        self.assertEqual({s.research_status for s in SPECS.values() if s.model != "naming_heuristic"},
                         {"reference", "confirmatory"})


# ---------------------------------------------------------------------------- development data (5)
class DevelopmentDataTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.groups, self.keys = synthetic_master()
        self.assign, self.meta = synthetic_split(self.X, self.y, self.groups, self.keys)

    def test_only_development_rows_in_master_order(self):
        d = cv.select_development(self.X, self.y, self.groups, self.assign, self.meta)
        dev = self.assign.index[self.assign.partition == "development"]
        self.assertTrue(d.X.index.equals(pd.Index(dev)))
        self.assertTrue((np.diff(d.X.index) > 0).all())
        self.assertFalse(set(d.groups) & set(self.assign.loc[self.assign.partition == "test", "meta_project_group"]))
        self.assertTrue((d.keys["master_row"].to_numpy() == d.X.index.to_numpy()).all())
        self.assertEqual(len(d.X), self.meta["counts"]["development"]["rows"])
        self.assertTrue((d.fold_of_row >= 0).all())
        self.assertNotIn("test", json.dumps(d.split_info))           # no test counts passed on

    def test_rejects_a_test_project_in_development(self):
        a = self.assign.copy()
        row = a.index[a.partition == "test"][0]
        a.loc[row, ["partition", "inner_fold"]] = ["development", 0]
        with self.assertRaisesRegex(cv.DevelopmentDataError, "held-out projects"):
            cv.select_development(self.X, self.y, self.groups, a, self.meta)

    def test_rejects_counts_or_folds_that_differ_from_the_metadata(self):
        m = copy.deepcopy(self.meta); m["counts"]["development"]["positives"] += 1
        with self.assertRaisesRegex(cv.DevelopmentDataError, "development counts"):
            cv.select_development(self.X, self.y, self.groups, self.assign, m)
        m = copy.deepcopy(self.meta); m["counts"]["inner_folds"][2]["validation"]["rows"] += 1
        with self.assertRaisesRegex(cv.DevelopmentDataError, "fold 2"):
            cv.select_development(self.X, self.y, self.groups, self.assign, m)
        y2 = self.y.copy(); y2.iloc[0] = 1 - y2.iloc[0]
        with self.assertRaisesRegex(cv.DevelopmentDataError, "labels"):
            cv.select_development(self.X, y2, self.groups, self.assign, self.meta)

    def test_expected_real_counts_are_the_approved_ones(self):
        self.assertEqual(cv.EXPECTED_DEVELOPMENT, {"rows": 24967, "projects": 261, "positives": 193})


# ---------------------------------------------------------------------------- runner (1-15)
class RunnerTests(unittest.TestCase):
    def test_every_configured_model_runs(self):
        for s in SPECS.values():
            res = cv.run_development_cv(s, DATA[s.feature_set])
            summ = res["summary"]
            self.assertEqual(summ["cv_average_precision"]["n_folds"], 5, s.run_id)
            self.assertEqual(len(res["oof"]), len(DATA[s.feature_set].X))
            self.assertEqual(list(res["fold_metrics"]["fold"]), [0, 1, 2, 3, 4])
            self.assertIn("optimistic", summ["development_threshold_metrics"]["note"])
            self.assertNotIn("pooled", json.dumps(summ["cv_average_precision"]).lower().replace("pooled development", ""))

    def test_oof_coverage_and_row_order(self):
        d = DATA["main"]
        res = cv.run_development_cv(SPECS["dummy_prior__main"], d, pipeline_factory=fake_factory())
        oof = res["oof"]
        self.assertEqual(list(oof.columns), ["master_row", "commit", "file", "class", "name", "meta_project_group",
                                             "fold", "renamed", "oof_probability"])
        self.assertTrue((oof["master_row"].to_numpy() == d.X.index.to_numpy()).all())
        np.testing.assert_allclose(oof["oof_probability"], (d.X["loc"].to_numpy() % 100) / 100)
        np.testing.assert_array_equal(oof["fold"], d.fold_of_row)
        np.testing.assert_array_equal(oof["renamed"], d.y.to_numpy())
        self.assertEqual(sorted(oof["fold"].value_counts().to_dict()), [0, 1, 2, 3, 4])

    def test_group_separation_is_enforced(self):
        d = synthetic_development("main")
        train, val = d.folds[0]
        d.folds[0] = (np.r_[train, val[:3]], val)                    # project rows on both sides
        with self.assertRaises(cv.RunFailedError):
            cv.run_development_cv(SPECS["dummy_prior__main"], d)
        res = cv.run_development_cv(SPECS["dummy_prior__main"], DATA["main"])
        self.assertTrue((res["fold_metrics"]["n_train_projects"] + res["fold_metrics"]["n_val_projects"]
                         == DATA["main"].groups.nunique()).all())

    def test_missing_or_duplicated_oof_rows_fail(self):
        for label, change in (("missing", lambda f: (f[0][0], f[0][1][1:])),
                              ("duplicated", lambda f: (f[0][0], np.r_[f[0][1], f[1][1][:1]]))):
            d = synthetic_development("main")
            d.folds[0] = change(d.folds)
            with self.assertRaises(cv.RunFailedError, msg=label):
                cv.run_development_cv(SPECS["dummy_prior__main"], d, pipeline_factory=fake_factory())

    def test_preprocessing_and_weights_fitted_per_training_fold(self):
        d = DATA["comparison"]
        seen = {}

        def lr_check(k, pipe, train, val):
            pre = pipe.named_steps["pre"]
            raw_cols = list(pre.transformers_[2][2])
            np.testing.assert_allclose(pre.named_transformers_["raw"].mean_, d.X.iloc[train][raw_cols].mean())
            log_cols = list(pre.transformers_[0][2])
            np.testing.assert_allclose(pre.named_transformers_["log1p"][-1].mean_,
                                       np.log1p(d.X.iloc[train][log_cols]).mean())
            self.assertEqual(int(pre.named_transformers_["raw"].n_samples_seen_), len(train))
            seen.setdefault("lr", []).append(pre.named_transformers_["raw"].mean_[0])

        cv.run_development_cv(SPECS["logistic_regression__comparison__log1p_asinh"], d, fold_callback=lr_check)
        self.assertEqual(len(set(np.round(seen["lr"], 9))), 5)              # each fold its own mean

        def yj_check(k, pipe, train, val):
            pt = pipe.named_steps["pre"].named_transformers_["yeo_johnson"]
            cols = list(pipe.named_steps["pre"].transformers_[0][2])
            from sklearn.preprocessing import PowerTransformer
            ref = PowerTransformer(method="yeo-johnson").fit(d.X.iloc[train][cols])
            np.testing.assert_allclose(pt.lambdas_, ref.lambdas_)
        cv.run_development_cv(SPECS["logistic_regression__comparison__yeo_johnson"], d, fold_callback=yj_check)

        ratios = []

        def lgbm_check(k, pipe, train, val):
            yt = d.y.iloc[train]
            self.assertAlmostEqual(pipe.named_steps["model"].scale_pos_weight_, (yt == 0).sum() / yt.sum())
            ratios.append(pipe.named_steps["model"].scale_pos_weight_)
        cv.run_development_cv(spec("lightgbm__comparison", n_estimators=10), d, fold_callback=lgbm_check)
        self.assertGreater(len(set(np.round(ratios, 9))), 1)
        self.assertNotIn(round((d.y == 0).sum() / d.y.sum(), 9), set(np.round(ratios, 9)))

    def test_random_baseline_training_prevalence_and_reproducible(self):
        d = DATA["main"]
        prev = []
        cv.run_development_cv(SPECS["stratified_random__main"], d,
                              fold_callback=lambda k, p, tr, va: prev.append((p.named_steps["model"].prevalence_,
                                                                              d.y.iloc[tr].mean())))
        for got, want in prev:
            self.assertEqual(got, want)
        a = cv.run_development_cv(SPECS["stratified_random__main"], d)["oof"]
        b = cv.run_development_cv(SPECS["stratified_random__main"], d)["oof"]
        pd.testing.assert_frame_equal(a, b)

    def test_validation_labels_never_influence_a_folds_scores(self):
        for run_id in ("stratified_random__main", "logistic_regression__main__log1p_asinh", "naming_heuristic__main"):
            base = cv.run_development_cv(SPECS[run_id], DATA["main"])["oof"]
            d = synthetic_development("main")
            val = d.folds[2][1]
            y2 = d.y.copy()
            y2.iloc[val] = 1 - y2.iloc[val]                           # flip fold 2's validation labels only
            d.y = y2
            changed = cv.run_development_cv(SPECS[run_id], d)["oof"]
            np.testing.assert_array_equal(base["oof_probability"].to_numpy()[val],
                                          changed["oof_probability"].to_numpy()[val], err_msg=run_id)

    def test_invalid_probabilities_fail(self):
        for mode, classes in (("above_one", (0, 1)), ("nan", (0, 1)), ("one_column", (0, 1)), ("identity", (1, 0))):
            with self.assertRaises(cv.RunFailedError, msg=f"{mode} {classes}") as ctx:
                cv.run_development_cv(SPECS["dummy_prior__main"], DATA["main"],
                                      pipeline_factory=fake_factory(mode, classes))
            self.assertEqual(ctx.exception.fold, 0)

    def test_convergence_warning_fails_the_run(self):
        with self.assertRaisesRegex(cv.RunFailedError, "ConvergenceWarning") as ctx:
            cv.run_development_cv(spec("logistic_regression__main__log1p_asinh", max_iter=1), DATA["main"])
        self.assertEqual(ctx.exception.fold, 0)
        self.assertTrue(any(w["category"] == "ConvergenceWarning" for w in ctx.exception.warnings_log))

    def test_warnings_recorded_with_fold_ids(self):
        res = cv.run_development_cv(SPECS["dummy_prior__main"], DATA["main"], pipeline_factory=fake_factory("warn"))
        self.assertEqual(sorted(w["fold"] for w in res["warnings"]), [0, 1, 2, 3, 4])
        self.assertEqual({w["category"] for w in res["warnings"]}, {"UserWarning"})
        self.assertEqual(list(res["fold_metrics"]["n_warnings"]), [1] * 5)
        d = synthetic_development("main", transform_X=put_value("doc_len", 1e155))
        with self.assertRaises(cv.RunFailedError) as ctx:          # Yeo-Johnson warns, then LR cannot converge
            cv.run_development_cv(SPECS["logistic_regression__main__yeo_johnson"], d)
        yj = [w for w in ctx.exception.warnings_log if w["category"] == "RuntimeWarning"]
        self.assertTrue(yj and all(isinstance(w["fold"], int) for w in yj))
        self.assertTrue(any("overflow" in w["message"] for w in yj))

    def test_non_finite_transformed_values_fail(self):
        d = synthetic_development("main", transform_X=put_value("doc_len", 1e200))
        with self.assertRaisesRegex(cv.RunFailedError, "NaN or infinite"):
            cv.run_development_cv(SPECS["logistic_regression__main__yeo_johnson"], d)

    def test_metrics_reuse_stage6_definitions(self):
        from src.evaluation import metrics as M
        res = cv.run_development_cv(SPECS["naming_heuristic__main"], DATA["main"])
        d, oof = DATA["main"], res["oof"]["oof_probability"].to_numpy()
        fold = M.fold_average_precision(d.y.to_numpy(), oof, d.folds)
        self.assertEqual(res["summary"]["cv_average_precision"]["per_fold"], fold["per_fold"])
        self.assertEqual(res["summary"]["cv_average_precision"]["std"], fold["std"])
        sel = M.select_max_f1_threshold(d.y.to_numpy(), oof)
        self.assertEqual(res["summary"]["threshold_selection"]["threshold"], sel.threshold)
        self.assertEqual(res["summary"]["threshold_selection"]["n_tied_max_f1"], sel.n_tied)
        np.testing.assert_allclose(res["fold_metrics"]["val_prevalence"],
                                   [d.y.iloc[v].mean() for _, v in d.folds])


# ---------------------------------------------------------------------------- matrix rules (20, selection)
class MatrixRuleTests(unittest.TestCase):
    def test_baselines_identical_between_feature_sets(self):
        results = {rid: cv.run_development_cv(SPECS[rid], DATA[SPECS[rid].feature_set])
                   for rid in SPECS if SPECS[rid].model in cv.BASELINES}
        self.assertEqual(cv.check_baseline_equivalence(results), list(cv.BASELINES))
        bad = dict(results)
        tampered = copy.deepcopy(bad["dummy_prior__comparison"])
        tampered["oof"].loc[0, "oof_probability"] += 1e-9
        bad["dummy_prior__comparison"] = tampered
        with self.assertRaises(cv.RunFailedError):
            cv.check_baseline_equivalence(bad)

    def test_stage9_rule(self):
        def summ(model, transform, ap, fs="main", status="confirmatory"):
            return {"model": model, "transform": transform, "feature_set": fs, "research_status": status,
                    "run_id": f"{model}_{transform}_{fs}", "cv_average_precision": {"mean": ap}}
        base = [summ("logistic_regression", "log1p_asinh", 0.30), summ("logistic_regression", "yeo_johnson", 0.31),
                summ("random_forest", "none", 0.35), summ("lightgbm", "none", 0.33)]
        self.assertEqual(cv.select_stage9_candidate(base, "main"), "random_forest_none_main")
        tie = base[:2] + [summ("random_forest", "none", 0.33), summ("lightgbm", "none", 0.33)]
        self.assertEqual(cv.select_stage9_candidate(tie, "main"), "random_forest_none_main")      # simpler wins
        tie_lr = [summ("logistic_regression", "log1p_asinh", 0.4), summ("logistic_regression", "yeo_johnson", 0.4)] + base[2:]
        self.assertEqual(cv.select_stage9_candidate(tie_lr, "main"), "logistic_regression_log1p_asinh_main")
        best_heuristic = base + [summ("naming_heuristic", "none", 0.99, status=cv.EXPLORATORY)]
        self.assertEqual(cv.select_stage9_candidate(best_heuristic, "main"), "random_forest_none_main")
        with self.assertRaises(ValueError):
            cv.select_stage9_candidate(base[:3], "main")
        relabelled = [summ("logistic_regression", "log1p_asinh", 0.30, status=cv.EXPLORATORY)] + base[1:]
        with self.assertRaisesRegex(ValueError, "exploratory"):           # a candidate can never be exploratory
            cv.select_stage9_candidate(relabelled, "main")


# ---------------------------------------------------------------------------- run store (16-18)
class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.spec = SPECS["naming_heuristic__main"]
        self.result = cv.run_development_cv(self.spec, DATA["main"])
        self.prov = rs.provenance(DATA["main"].split_info, {"started_utc": "t0"})

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_complete_run_has_every_artifact(self):
        folder = rs.write_run(self.dir, "2026-01-01_stage8_x", self.spec, self.result, self.prov)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(rs.REQUIRED_FILES))
        self.assertTrue(rs.is_complete_run(folder))
        oof = pd.read_csv(folder / "oof_predictions.csv", keep_default_na=False)
        self.assertEqual(list(oof.columns)[:6], ["master_row", "commit", "file", "class", "name", "meta_project_group"])
        np.testing.assert_allclose(oof["oof_probability"], self.result["oof"]["oof_probability"])
        cfg = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["research_status"], cv.EXPLORATORY)
        self.assertFalse(list(folder.rglob("*.pkl")) + list(folder.rglob("*.joblib")))      # no fitted models

    def test_never_overwrites(self):
        folder = rs.write_run(self.dir, "run_a", self.spec, self.result, self.prov)
        before = sha256_of(folder / "summary.json")
        with self.assertRaises(rs.RunStoreError):
            rs.write_run(self.dir, "run_a", self.spec, self.result, self.prov)
        self.assertEqual(sha256_of(folder / "summary.json"), before)
        self.assertEqual(rs.stale_temporary_runs(self.dir), [])

    def test_interrupted_write_leaves_no_result(self):
        broken = dict(self.result, oof=None)                          # writing the OOF file fails
        with self.assertRaises(Exception):
            rs.write_run(self.dir, "run_b", self.spec, broken, self.prov)
        self.assertFalse((self.dir / "run_b").exists())
        self.assertEqual(rs.stale_temporary_runs(self.dir), [])
        tmp = self.dir / ".tmp_run_c_dead"                            # a crash that skipped clean-up
        tmp.mkdir()
        self.assertEqual(rs.stale_temporary_runs(self.dir), [tmp])
        self.assertFalse(rs.is_complete_run(tmp))

    def test_failed_and_incomplete_runs_are_not_results(self):
        err = cv.RunFailedError("boom", 3, [{"fold": 3, "category": "ConvergenceWarning"}])
        folder = rs.write_failed_run(self.dir, "run_d", self.spec, err, err.warnings_log, self.prov)
        self.assertTrue(folder.name.endswith("__FAILED"))
        self.assertFalse(rs.is_complete_run(folder))
        self.assertEqual(json.loads((folder / "error.json").read_text(encoding="utf-8"))["fold"], 3)
        self.assertFalse((folder / "oof_predictions.csv").exists())
        good = rs.write_run(self.dir, "run_e", self.spec, self.result, self.prov)
        (good / "summary.json").unlink()
        self.assertFalse(rs.is_complete_run(good))

    def test_provenance_fields(self):
        p = self.prov
        for key in ("git", "python", "platform", "packages", "requirements_lock_sha256", "master_sha256",
                    "split_hash", "split_csv_sha256", "development_counts", "fold_statistics", "command"):
            self.assertIn(key, p)
        self.assertEqual(set(p["git"]), {"commit", "dirty", "dirty_paths"})
        self.assertRegex(p["git"]["commit"] or "", r"^[0-9a-f]{40}$")
        self.assertTrue(set(rs.PACKAGES) <= set(p["packages"]))
        self.assertTrue(p["packages"]["scikit-learn"] and p["packages"]["lightgbm"])
        self.assertEqual(len(p["fold_statistics"]), 5)


# ---------------------------------------------------------------------------- CLI end to end (synthetic)
class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def main(self, *argv, loader=None):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = r8.main(list(argv), data_loader=loader or synthetic_loader(), runs_dir=self.dir,
                           day="2026-01-01", out=print)
        return code, buf.getvalue()

    def test_import_runs_nothing(self):
        self.assertFalse(any(self.dir.iterdir()))

    def test_dry_run_fits_nothing(self):
        with mock.patch.object(r8, "run_development_cv", side_effect=AssertionError("fitted")):
            code, out = self.main("--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("Dry run complete", out)
        for rid in r8.APPROVED_MATRIX:
            self.assertIn(rid, out)
        self.assertFalse(any(self.dir.iterdir()))

    def test_only_one_run(self):
        code, out = self.main("--only", "naming_heuristic__main")
        self.assertEqual(code, 0)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["2026-01-01_stage8_naming_heuristic__main"])
        with self.assertRaises(r8.ConfigError):
            self.main("--only", "svm__main")

    def test_full_matrix_and_summary(self):
        code, out = self.main()
        self.assertEqual(code, 0, out)
        folders = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(len(folders), 15)
        self.assertTrue(all(rs.is_complete_run(self.dir / f"2026-01-01_stage8_{rid}") for rid in r8.APPROVED_MATRIX))
        summ = self.dir / "2026-01-01_stage8_summary"
        md = (summ / "summary.md").read_text(encoding="utf-8")
        main_section = md.split("## Exploratory")[0]
        self.assertNotIn("naming_heuristic", main_section)                    # not ranked with the models
        self.assertIn("naming_heuristic__main", md.split("## Exploratory")[1])
        self.assertIn("dummy_prior, stratified_random, naming_heuristic", md)
        sel = json.loads((summ / "stage9_selection.json").read_text(encoding="utf-8"))["selected"]
        self.assertEqual(set(sel), {"main", "comparison"})
        self.assertFalse(any("naming" in v or "dummy" in v or "random__" in v for v in sel.values()))
        table = pd.read_csv(summ / "summary.csv")
        self.assertEqual(len(table), 14)
        self.assertNotIn("pooled", " ".join(table.columns))
        with self.assertRaises(rs.RunStoreError):                             # same day again: refused
            self.main()

    def test_seed_spread_runs_separately(self):
        code, out = self.main("--seed-spread")
        self.assertEqual(code, 0)
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(len(names), 21)
        self.assertTrue(all("seed" in n for n in names))
        self.assertFalse(any(rid in n for rid in r8.APPROVED_MATRIX for n in names if n.endswith(rid)))

    def test_failed_runs_are_recorded_and_block_the_summary(self):
        code, out = self.main(loader=synthetic_loader(transform_X=put_value("doc_len", 1e200)))
        self.assertEqual(code, 1)
        failed = {p.name.replace("2026-01-01_stage8_", "").replace("__FAILED", ""): p
                  for p in self.dir.iterdir() if p.name.endswith("__FAILED")}
        # Yeo-Johnson overflows to non-finite values; Random Forest casts X to float32 (max ~3.4e38)
        self.assertEqual(set(failed), {"logistic_regression__main__yeo_johnson",
                                       "logistic_regression__comparison__yeo_johnson",
                                       "random_forest__main", "random_forest__comparison"})
        for run_id, folder in failed.items():
            err = json.loads((folder / "error.json").read_text(encoding="utf-8"))["error"]
            self.assertIn("NaN or infinite" if "yeo" in run_id else "float32", err, run_id)
            self.assertFalse(rs.is_complete_run(folder))
        self.assertFalse((self.dir / "2026-01-01_stage8_summary").exists())


@unittest.skipUnless(config.MASTER.exists() and sp.SPLIT_CSV.exists(), "master dataset or saved split not present")
class InputsUnchangedTests(unittest.TestCase):
    def test_master_and_split_unchanged(self):
        self.assertEqual(sha256_of(config.MASTER), CANONICAL)
        self.assertEqual(sp.load_splits()[1]["split_hash"], SPLIT_HASH)
        self.assertEqual(CONFIG["dataset_sha256"], CANONICAL)
        self.assertEqual(CONFIG["split_hash"], SPLIT_HASH)


if __name__ == "__main__":
    unittest.main()
