"""
Tests for the Stage 10 local web application (src/app/, run.py).
Feature parity is checked against the canonical master dataset (exact equality) and against the training
extractor; models are only loaded, never refitted. No Stage 8/9 artifact is modified.
Run: python -m unittest discover -s tests -v
"""
import http.server
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from src import config  # noqa: E402
from src.app import inference as inf  # noqa: E402
from src.app.server import create_app  # noqa: E402
from src.data.load_master import load_master  # noqa: E402
from src.features.feature_sets import COMPARISON_FEATURES, MAIN_FEATURES  # noqa: E402

HAVE_MODELS = (inf.MODEL_DIR / "main_pipeline.joblib").exists() and (inf.RESULT_DIR / "provenance.json").exists()
SAMPLE = b'''import functools

def helper(x):
    return x + 1

class Outer:
    def get_value(self):
        return self._v

    @staticmethod
    def calc_cnt(a, b):
        return a + b

    @classmethod
    def from_cfg(cls, cfg):
        return cls()

    @functools.lru_cache()
    def tmp2(self):
        return 2

    async def fetch_data(self, url):
        return url

    def __init__(self):
        self._v = 1

    @property
    def size(self):
        return 1

    @size.setter
    def size(self, v):
        pass

    class Inner:
        def do_it(self):
            def nested():
                return 1
            return nested()

def test_helper():
    assert helper(1) == 2
'''


@unittest.skipUnless(HAVE_MODELS, "Stage 9 model artifacts not present")
class ModelLoadingTests(unittest.TestCase):
    def test_models_load_and_match_the_stage9_record(self):
        reg = inf.load_models()
        self.assertEqual(set(reg.models), {"main", "comparison"})
        cfg = json.loads((inf.RESULT_DIR / "run_config.json").read_text())
        for fs, m in reg.models.items():
            self.assertEqual(m.threshold, cfg["selected"][fs]["threshold"])
            self.assertEqual(m.features, MAIN_FEATURES if fs == "main" else COMPARISON_FEATURES)
            self.assertEqual(list(m.pipeline.classes_), [0, 1])

    def copy_artifacts(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(inf.MODEL_DIR, tmp / "models"); shutil.copytree(inf.RESULT_DIR, tmp / "results")
        return tmp / "models", tmp / "results"

    def test_altered_missing_or_inconsistent_artifacts_are_refused(self):
        models, results = self.copy_artifacts()
        p = models / "main_pipeline.joblib"
        p.write_bytes(p.read_bytes() + b"\0")                          # one byte appended
        with self.assertRaisesRegex(inf.ModelArtifactError, "does not match its recorded SHA-256"):
            inf.load_models(models, results)
        models, results = self.copy_artifacts()
        (models / "comparison_pipeline.joblib").unlink()
        with self.assertRaisesRegex(inf.ModelArtifactError, "missing"):
            inf.load_models(models, results)
        models, results = self.copy_artifacts()
        cfg = json.loads((results / "run_config.json").read_text()); cfg["selected"]["main"]["threshold"] = 0.5
        (results / "run_config.json").write_text(json.dumps(cfg))
        with self.assertRaisesRegex(inf.ModelArtifactError, "threshold differs"):
            inf.load_models(models, results)
        models, results = self.copy_artifacts()
        (results / "provenance.json").unlink()
        with self.assertRaisesRegex(inf.ModelArtifactError, "Stage 9 record unavailable"):
            inf.load_models(models, results)

    def test_original_artifacts_untouched(self):
        prov = json.loads((inf.RESULT_DIR / "provenance.json").read_text())
        for name, digest in prov["model_artifacts"]["sha256"].items():
            self.assertEqual(inf._sha256(inf.MODEL_DIR / name), digest)


@unittest.skipUnless(config.MASTER.exists() and config.RAW_COMMITS.exists(), "master or raw data not present")
class FeatureParityTests(unittest.TestCase):
    """Inference features (one file at a time) must equal the training features of the master dataset exactly."""

    def test_master_rows_reproduced_exactly(self):
        master = load_master()
        n_files = int(os.environ.get("APP_PARITY_FILES", "150"))      # set 0 for all 3,181 files (~80 s)
        groups = list(master.groupby(["commit", "file"], sort=True))
        if n_files:
            groups = groups[:: max(1, len(groups) // n_files)]
        compared = 0
        for (commit, file), g in groups:
            src = (config.RAW_COMMITS / commit / "Before" / file.replace("/", "#")).read_bytes()
            feats, _, _ = inf.extract_file_features(src, file)
            feats = feats.set_index(["class", "name"])
            for _, r in g.iterrows():
                got = feats.loc[(r["class"], r["name"])]
                for f in COMPARISON_FEATURES:
                    self.assertEqual(float(got[f]), float(r[f]), f"{commit[:8]} {file} {r['name']} {f}")
                compared += 1
        self.assertGreater(compared, 1000)

    def test_identical_to_training_extractor_call(self):
        from src.data.extract_features import method_rows
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, True)
        for side in ("Before", "After"):
            (tmp / "c" / side).mkdir(parents=True); (tmp / "c" / side / "pkg#mod.py").write_bytes(SAMPLE)
        ref = pd.DataFrame(method_rows(tmp / "c", "c", set(), set()))
        got, _, _ = inf.extract_file_features(SAMPLE, "pkg/mod.py")
        pd.testing.assert_frame_equal(got.drop(columns=["lineno", "commit"]), ref.drop(columns=["commit"]))


@unittest.skipUnless(HAVE_MODELS, "Stage 9 model artifacts not present")
class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = inf.load_models()

    def test_scope_matches_training(self):
        res = inf.analyse(SAMPLE, "sample.py", self.reg)
        scored = {(m["class"], m["name"]) for m in res["methods"]}
        self.assertEqual(scored, {(None, "helper"), ("Outer", "get_value"), ("Outer", "calc_cnt"), ("Outer", "from_cfg"),
                                  ("Outer", "tmp2"), ("Outer", "fetch_data"), ("Outer", "size"), ("Outer.Inner", "do_it")})
        reasons = {s["name"]: s["reason"] for s in res["skipped"]}
        self.assertIn("dunder", reasons["__init__"]); self.assertIn("test", reasons["test_helper"])
        self.assertIn("inside a function", reasons["nested"]); self.assertIn("replaced by a later one", reasons["size"])
        by = {m["name"]: m for m in res["methods"]}
        self.assertTrue(by["fetch_data"]["is_async"]); self.assertEqual(by["tmp2"]["decorators"], 1)
        import ast
        last_size = max(n.lineno for n in ast.walk(ast.parse(SAMPLE)) if getattr(n, "name", None) == "size")
        self.assertEqual(by["size"]["lineno"], last_size)               # the last definition, as in training
        self.assertEqual(res["n_scored"], 8)

    def test_single_function_module_and_multiple_classes(self):
        one = inf.analyse(b"def compute_total(xs):\n    return sum(xs)\n", "a.py", self.reg)
        self.assertEqual([m["name"] for m in one["methods"]], ["compute_total"])
        multi = inf.analyse(b"class A:\n    def f(self):\n        pass\nclass B:\n    def g(self):\n        return 1\n", "b.py", self.reg)
        self.assertEqual({(m["class"], m["name"]) for m in multi["methods"]}, {("A", "f"), ("B", "g")})

    def test_empty_and_no_methods(self):
        for src in (b"", b"X = 1\nimport os\n", b"class Empty:\n    pass\n"):
            res = inf.analyse(src, "e.py", self.reg)
            self.assertEqual((res["n_scored"], res["methods"]), (0, []))

    def test_invalid_inputs(self):
        cases = {"syntax": (b"def f(:\n", "a.py"), "extension": (b"x = 1", "a.txt"), "binary": (b"\x00\x01", "a.py"),
                 "traversal": (b"x = 1", "../../x.py"), "absolute": (b"x = 1", "C:/x.py"),
                 "too large": (b"#" * (inf.MAX_BYTES + 1), "a.py")}
        for label, (src, name) in cases.items():
            with self.assertRaises(inf.InferenceError, msg=label):
                inf.analyse(src, name, self.reg)

    def test_relative_path_sets_is_test_file_only(self):
        a = inf.analyse(SAMPLE, "util.py", self.reg)
        b = inf.analyse(SAMPLE, "util.py", self.reg, relative_path="pkg/tests/util.py")
        fa, fb = a["methods"][0]["features"], b["methods"][0]["features"]
        self.assertEqual((fa["is_test_file"], fb["is_test_file"]), (0, 1))
        self.assertEqual({k: v for k, v in fa.items() if k != "is_test_file"}, {k: v for k, v in fb.items() if k != "is_test_file"})

    def test_scores_thresholds_and_contributions(self):
        res = inf.analyse(SAMPLE, "sample.py", self.reg)
        feats, _, _ = inf.extract_file_features(SAMPLE, "sample.py")
        for fs, model in self.reg.models.items():
            direct = model.pipeline.predict_proba(inf.model_input(feats, model.features))[:, 1]
            by_name = dict(zip(zip(feats["class"], feats["name"]), direct))
            for m in res["methods"]:
                mm = m["models"][fs]
                self.assertAlmostEqual(mm["score"], by_name[(m["class"] or "", m["name"])], places=12)
                self.assertTrue(0 <= mm["score"] <= 1)
                self.assertEqual(mm["above_threshold"], mm["score"] >= model.threshold)
                self.assertAlmostEqual(inf.logit_to_score(mm["logit"]), mm["score"], places=9)   # exact decomposition
        mains = [m["models"]["main"]["score"] for m in res["methods"]]
        self.assertEqual(mains, sorted(mains, reverse=True))

    def test_review_cutoff_is_separate_from_the_validated_threshold(self):
        self.assertEqual(inf.REVIEW_CUTOFF, 0.70)
        res = inf.analyse(SAMPLE, "sample.py", self.reg)
        self.assertEqual(res["review_cutoff"], 0.70)
        for fs, model in self.reg.models.items():
            for m in res["methods"]:
                mm = m["models"][fs]
                self.assertEqual(mm["threshold"], model.threshold)                 # validated value unchanged
                self.assertEqual(mm["above_review_cutoff"], mm["score"] >= 0.70)
                self.assertEqual(mm["above_threshold"], mm["score"] >= model.threshold)
            self.assertEqual(res["above_review_cutoff"][fs], sum(m["models"][fs]["score"] >= 0.70 for m in res["methods"]))
        self.assertGreaterEqual(res["above_review_cutoff"]["main"], res["above_threshold"]["main"])
        low = inf.analyse(SAMPLE, "sample.py", self.reg, review_cutoff=0.3)
        self.assertGreaterEqual(low["above_review_cutoff"]["main"], res["above_review_cutoff"]["main"])
        for bad in (0, 1, -0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError, msg=str(bad)):
                inf.analyse(SAMPLE, "sample.py", self.reg, review_cutoff=bad)

    def test_missing_features_and_wrong_order(self):
        feats, _, _ = inf.extract_file_features(SAMPLE, "sample.py")
        with self.assertRaisesRegex(inf.InferenceError, "missing"):
            inf.model_input(feats.drop(columns=["loc"]), MAIN_FEATURES)
        X = inf.model_input(feats, MAIN_FEATURES)
        self.assertEqual(list(X.columns), MAIN_FEATURES)
        with self.assertRaises(ValueError):                              # the pipeline refuses reordered columns
            self.reg.models["main"].pipeline.predict_proba(X[MAIN_FEATURES[::-1]])
        bad = feats.copy(); bad["loc"] = np.inf
        with self.assertRaises(inf.InferenceError):
            inf.model_input(bad, MAIN_FEATURES)

    def test_uploaded_code_is_never_executed(self):
        marker = Path(tempfile.gettempdir()) / "rename_risk_pwned_marker.txt"
        marker.unlink(missing_ok=True)
        evil = f"open({str(marker)!r}, 'w').write('x')\nraise SystemExit(1)\ndef f():\n    return 1\n".encode()
        before = {p.name for p in Path(tempfile.gettempdir()).glob("rename_risk_*")}
        res = inf.analyse(evil, "evil.py", self.reg)
        self.assertEqual(res["n_scored"], 1)
        self.assertFalse(marker.exists())
        self.assertEqual({p.name for p in Path(tempfile.gettempdir()).glob("rename_risk_*")}, before)  # temp removed


@unittest.skipUnless(HAVE_MODELS, "Stage 9 model artifacts not present")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_health_models_and_dashboard(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        m = self.client.get("/api/models").json()
        self.assertEqual(m["primary"], "main")
        self.assertIn("not been verified by a person", m["disclaimer"])
        self.assertIn("threshold", m["threshold_note"])
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200); self.assertIn("Experimental research prototype", page.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)

    def test_review_cutoff_reported_and_configurable(self):
        m = self.client.get("/api/models").json()
        self.assertEqual(m["review_cutoff"], 0.70)
        self.assertIn("NOT validated", m["review_cutoff_note"])
        self.assertEqual(m["models"]["main"]["threshold"], 0.859356439967941)
        d = self.client.post("/api/analyze", files={"file": ("s.py", SAMPLE)}).json()
        self.assertEqual(d["review_cutoff"], 0.70)
        custom = TestClient(create_app(review_cutoff=0.5)).post("/api/analyze", files={"file": ("s.py", SAMPLE)}).json()
        self.assertEqual(custom["review_cutoff"], 0.5)
        with self.assertRaises(ValueError):
            create_app(review_cutoff=1.2)

    def test_upload_and_method_details(self):
        r = self.client.post("/api/analyze", files={"file": ("sample.py", SAMPLE)})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["n_scored"], 8)
        mid = d["methods"][0]["method_id"]
        det = self.client.get(f"/api/methods/{mid}", params={"analysis_id": d["analysis_id"]})
        self.assertEqual(det.status_code, 200)
        self.assertNotIn("source", det.json())                          # source is not kept server-side
        self.assertEqual(self.client.get("/api/methods/m1", params={"analysis_id": "nope"}).status_code, 404)

    def test_api_errors_are_clear_and_leak_no_paths(self):
        for name, body, code in (("bad.py", b"def f(:\n", 400), ("a.txt", b"x=1", 400), ("../x.py", b"x=1", 400),
                                 ("big.py", b"#" * (inf.MAX_BYTES + 1), 413)):
            r = self.client.post("/api/analyze", files={"file": (name, body)})
            self.assertEqual(r.status_code, code, name)
            text = r.text
            for leak in (str(config.ROOT), "Refactoring_Project", "Traceback", tempfile.gettempdir()):
                self.assertNotIn(leak, text, name)
        self.assertEqual(self.client.post("/api/analyze").status_code, 422)

    def test_missing_models_degrade_gracefully(self):
        def broken():
            raise inf.ModelArtifactError("model artifact missing: main_pipeline.joblib")
        c = TestClient(create_app(registry_loader=broken))
        self.assertEqual(c.get("/health").json()["status"], "degraded")
        self.assertEqual(c.post("/api/analyze", files={"file": ("a.py", b"x=1")}).status_code, 503)
        self.assertEqual(c.get("/api/models").status_code, 503)

    def test_frontend_never_uses_innerhtml(self):
        js = (Path(inf.__file__).parent / "static" / "app.js").read_text(encoding="utf-8")
        import re
        self.assertIsNone(re.search(r"\.innerHTML|\.outerHTML|insertAdjacentHTML|document\.write|\beval\(", js))


class LauncherTests(unittest.TestCase):
    def test_wait_until_ready(self):
        class Ok(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

            def log_message(self, *a):
                pass
        srv = http.server.HTTPServer(("127.0.0.1", 0), Ok)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        self.assertTrue(run.wait_until_ready(f"http://127.0.0.1:{srv.server_port}/health", timeout=5))
        self.assertFalse(run.wait_until_ready("http://127.0.0.1:9/health", timeout=0.5, interval=0.1))

    def test_refuses_a_port_that_is_already_in_use(self):
        import socket
        busy = socket.socket(); busy.bind(("127.0.0.1", 0)); busy.listen(1)
        self.addCleanup(busy.close)
        port = busy.getsockname()[1]
        self.assertFalse(run.port_is_free("127.0.0.1", port))
        self.assertEqual(run.main(["--port", str(port), "--no-browser"]), 1)     # exits before starting a server

    def test_defaults_bind_to_localhost(self):
        import argparse  # noqa: F401
        src = Path(run.__file__).read_text(encoding="utf-8")
        self.assertIn('default="127.0.0.1"', src); self.assertIn("reload=False", src)


if __name__ == "__main__":
    unittest.main()
