"""
Stage 10 inference: rename-risk scores for the methods of one Python file, using the Stage 9 models.

Feature parity by construction
  Features are computed by the training extractor itself, src/data/extract_features.method_rows, which is
  called unchanged. The uploaded file is written (bytes as uploaded) into a temporary commit folder as
  Before/<path> and an identical After/<path>, so that every method counts as "name kept". That label is
  never used. The folder is deleted straight after. Nothing is re-implemented. Consequently:
    - scope = training scope: module-level functions and class methods, including nested classes,
      static/class methods, decorated and async ones. Dunder methods and names starting with "test" are
      not scored, but still count in the file context. Functions nested in functions or in if/try blocks
      are not collected.
    - file-context features (file_loc, file_n_methods, style_*, name_token_sharing, first_token_shared,
      rel_*) use every method of the uploaded file, exactly as in training.
    - is_test_file depends on the file's repository-relative path ("test" in the path). Pass the original
      relative path to reproduce it; otherwise the file name alone is used.
    - call_sites / rel_call_sites (excluded from both models) would need the commit's other files; they are
      computed but never used.

Models
  results/models/2026-10-31_stage9_final_test/{main,comparison}_pipeline.joblib. Each file's SHA-256 must
  equal the value recorded in the committed Stage 9 provenance.json *before* it is unpickled, and the
  thresholds must equal the Stage 9 run_config. Models are only loaded, never refitted.

Scores are model outputs on a 0-1 scale, NOT calibrated probabilities that a rename will happen.
"""
import ast
import hashlib
import json
import math
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config  # noqa: E402
from src.data.extract_features import method_rows  # noqa: E402
from src.features.feature_sets import get_feature_set  # noqa: E402
from src.features.method_features import collect, is_dunder  # noqa: E402

MODEL_RUN = "2026-10-31_stage9_final_test"
MODEL_DIR = config.ROOT / "results" / "models" / MODEL_RUN
RESULT_DIR = config.RUNS_DIR / MODEL_RUN
FEATURE_SETS = ("main", "comparison")            # main 40 = primary model
MAX_BYTES = 2 * 1024 * 1024
# User-chosen review cutoff (project owner's decision, 2026-10-31). It is NOT a validated threshold:
# on the development OOF scores it flags ~11% of methods and finds ~20% of renames (main 40), against
# ~2% / ~6% for the validated Stage 8 threshold. The validated thresholds are kept and reported separately.
REVIEW_CUTOFF = 0.70
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.\- /]+$")
_FAKE_COMMIT = "upload"


class InferenceError(ValueError):
    """The input cannot be analysed (not Python, unparsable, too large, invalid path)."""


class ModelArtifactError(RuntimeError):
    """A model artifact is missing, altered, or inconsistent with the Stage 9 record."""


# ---------------------------------------------------------------------------- models
@dataclass
class LoadedModel:
    feature_set: str
    pipeline: object
    threshold: float
    features: list
    run_id: str
    sha256: str


@dataclass
class ModelRegistry:
    models: dict
    info: dict = field(default_factory=dict)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(model_dir=MODEL_DIR, result_dir=RESULT_DIR):
    """Verify every artifact against the committed Stage 9 record, then load the two pipelines once."""
    model_dir, result_dir = Path(model_dir), Path(result_dir)
    try:
        prov = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
        run_cfg = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((result_dir / "test_metrics.json").read_text(encoding="utf-8"))
        trusted = prov["model_artifacts"]["sha256"]
    except (OSError, KeyError, ValueError) as exc:
        raise ModelArtifactError(f"Stage 9 record unavailable or invalid: {type(exc).__name__}") from exc
    for name, digest in trusted.items():
        path = model_dir / name
        if not path.is_file():
            raise ModelArtifactError(f"model artifact missing: {name}")
        if _sha256(path) != digest:
            raise ModelArtifactError(f"model artifact {name} does not match its recorded SHA-256")
    meta = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    models = {}
    for fs in FEATURE_SETS:
        fs_meta = meta["feature_sets"][fs]
        threshold = float(fs_meta["threshold"])
        if threshold != float(run_cfg["selected"][fs]["threshold"]):
            raise ModelArtifactError(f"{fs}: threshold differs between model metadata and the Stage 9 run config")
        features = get_feature_set(fs)
        if list(fs_meta["feature_names_in"]) != features:
            raise ModelArtifactError(f"{fs}: model feature names differ from the approved feature set")
        pipeline = joblib.load(model_dir / f"{fs}_pipeline.joblib")      # hash verified above
        if list(pipeline.named_steps["check"].feature_names_in_) != features or list(pipeline.classes_) != [0, 1]:
            raise ModelArtifactError(f"{fs}: loaded pipeline does not match the approved inputs / classes")
        models[fs] = LoadedModel(fs, pipeline, threshold, features, fs_meta["run_id"], trusted[f"{fs}_pipeline.joblib"])
    info = {"run": MODEL_RUN, "fitted_on": meta["fitted_on"], "packages": meta["packages"],
            "test_results": {fs: {k: metrics[fs]["test"][k] for k in ("average_precision", "roc_auc", "precision",
                                                                       "recall", "tp", "fp", "fn", "tn", "prevalence",
                                                                       "n_predicted_positive")}
                             for fs in FEATURE_SETS},
            "development_cv_ap": {fs: {"mean": metrics[fs]["development"]["dev_cv_ap_mean"],
                                       "std": metrics[fs]["development"]["dev_cv_ap_std"]} for fs in FEATURE_SETS}}
    return ModelRegistry(models, info)


# ---------------------------------------------------------------------------- features
def clean_relative_path(filename, relative_path=None):
    """The repository-relative path used for is_test_file; only ever used as a string, never as a filesystem path."""
    raw = (relative_path or filename or "").replace("\\", "/").strip()
    if not raw.lower().endswith(".py"):
        raise InferenceError("only .py files can be analysed")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts) or ":" in raw or not _SAFE_PATH.match(raw) or len(raw) > 300:
        raise InferenceError("invalid file path (allowed: letters, digits, _ . - / and spaces; no '..')")
    return "/".join(parts)


def extract_file_features(source_bytes, rel_path):
    """Feature rows for every scored method of one file, computed by the training extractor (unchanged)."""
    if len(source_bytes) > MAX_BYTES:
        raise InferenceError(f"file too large (limit {MAX_BYTES // 1024 // 1024} MB)")
    if b"\x00" in source_bytes:
        raise InferenceError("binary content: not a Python source file")
    text = source_bytes.decode("utf-8", errors="ignore")               # what training's parse() reads

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as exc:
        line = getattr(exc, "lineno", None)
        raise InferenceError(f"the file is not valid Python{f' (line {line})' if line else ''}") from None
    flat = rel_path.replace("/", "#")
    with tempfile.TemporaryDirectory(prefix="rename_risk_") as tmp:
        cdir = Path(tmp) / _FAKE_COMMIT
        for side in ("Before", "After"):
            (cdir / side).mkdir(parents=True)
            (cdir / side / flat).write_bytes(source_bytes)
        rows = method_rows(cdir, _FAKE_COMMIT, pos_keys=set(), found=set())
    feats = pd.DataFrame(rows)
    all_defs = collect(tree)
    skipped = [{"class": c, "name": n, "lineno": node.lineno,
                "reason": "dunder method (outside training scope)" if is_dunder(n) else "name starts with 'test' (outside training scope)"}
               for (c, n), (node, _, _) in all_defs.items() if is_dunder(n) or n.startswith("test")]
    collected = {id(node) for node, _, _ in all_defs.values()}
    parent = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and id(node) not in collected:
            direct = isinstance(parent.get(id(node)), (ast.Module, ast.ClassDef))
            skipped.append({"class": None, "name": node.name, "lineno": node.lineno,
                            "reason": "earlier definition replaced by a later one with the same name (training keeps the last)"
                            if direct else "defined inside a function or an if/try/with block (not collected in training)"})
    skipped.sort(key=lambda s: s["lineno"])
    if feats.empty:
        return feats, all_defs, skipped
    feats["lineno"] = [all_defs[(c, n)][0].lineno for c, n in zip(feats["class"], feats["name"])]
    return feats, all_defs, skipped


def model_input(feats, features):
    """The exact frame the pipeline expects: approved columns, approved order, numeric, finite."""
    missing = [f for f in features if f not in feats.columns]
    if missing:
        raise InferenceError(f"features missing from extraction: {missing}")
    X = feats[features].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise InferenceError("non-finite feature values")
    return X


def contributions(model, X):
    """Exact per-feature contributions to the Logistic Regression logit: coef_j * transformed x_j.

    Valid only because the final estimator is linear; the sum over features plus the intercept equals the
    model's logit. Contributions are in the model's transformed (Yeo-Johnson / scaled) feature space.
    """
    pre = model.pipeline[:-1]
    lr = model.pipeline.named_steps["model"]
    Z = pre.transform(X)
    contrib = Z.to_numpy() * lr.coef_[0]
    return pd.DataFrame(contrib, columns=list(Z.columns), index=X.index), float(lr.intercept_[0])


def validate_review_cutoff(value):
    value = float(value)
    if not (0.0 < value < 1.0) or math.isnan(value):
        raise ValueError("the review cutoff must lie strictly between 0 and 1")
    return value


def analyse(source_bytes, filename, registry, relative_path=None, top_k=5, review_cutoff=None):
    """Scores, threshold positions and evidence for every scored method of one Python file.

    Two separate operating points are reported per model:
      above_threshold      the validated Stage 8 threshold (evaluated once in Stage 9)
      above_review_cutoff  the user-chosen review cutoff (default REVIEW_CUTOFF = 0.70), NOT validated
    """
    review_cutoff = validate_review_cutoff(REVIEW_CUTOFF if review_cutoff is None else review_cutoff)
    rel = clean_relative_path(filename, relative_path)
    feats, all_defs, skipped = extract_file_features(source_bytes, rel)
    methods = []
    if not feats.empty:
        per_model = {}
        for fs, model in registry.models.items():
            X = model_input(feats, model.features)
            scores = model.pipeline.predict_proba(X)[:, 1]
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise InferenceError("the model produced invalid scores")
            contrib, intercept = contributions(model, X)
            per_model[fs] = (scores, contrib, intercept)
        lines = source_bytes.decode("utf-8", errors="ignore").splitlines()
        for i, row in feats.reset_index(drop=True).iterrows():
            node = all_defs[(row["class"], row["name"])][0]
            m = {"method_id": f"m{i + 1}", "name": row["name"], "class": row["class"] or None,
                 "lineno": int(row["lineno"]), "end_lineno": int(node.end_lineno or node.lineno),
                 "is_async": bool(row["is_async"]), "decorators": len(node.decorator_list),
                 "source": "\n".join(lines[min(d.lineno for d in node.decorator_list) - 1 if node.decorator_list
                                           else node.lineno - 1: node.end_lineno]),
                 "features": {f: (float(row[f]) if isinstance(row[f], (float, np.floating)) else int(row[f]))
                              for f in registry.models["comparison"].features},
                 "models": {}}
            for fs, (scores, contrib, intercept) in per_model.items():
                model = registry.models[fs]
                c = contrib.iloc[i]
                order = c.abs().sort_values(ascending=False).index[:top_k]
                logit = float(c.sum() + intercept)
                m["models"][fs] = {"score": float(scores[i]), "threshold": model.threshold,
                                   "above_threshold": bool(scores[i] >= model.threshold),
                                   "review_cutoff": review_cutoff,
                                   "above_review_cutoff": bool(scores[i] >= review_cutoff),
                                   "logit": logit, "intercept": intercept,
                                   "top_contributions": [{"feature": f, "contribution": float(c[f]),
                                                          "raw_value": m["features"][f]} for f in order]}
            methods.append(m)
        for fs in FEATURE_SETS:
            ranks = pd.Series([m["models"][fs]["score"] for m in methods]).rank(ascending=False, method="min")
            for m, r in zip(methods, ranks):
                m["models"][fs]["rank"] = int(r)
        methods.sort(key=lambda m: -m["models"]["main"]["score"])
    return {"file": rel, "n_definitions": len(all_defs), "n_scored": len(methods), "skipped": skipped,
            "above_threshold": {fs: sum(m["models"][fs]["above_threshold"] for m in methods) for fs in FEATURE_SETS},
            "review_cutoff": review_cutoff,
            "above_review_cutoff": {fs: sum(m["models"][fs]["above_review_cutoff"] for m in methods)
                                    for fs in FEATURE_SETS},
            "methods": methods}


def logit_to_score(logit):
    return 1 / (1 + math.exp(-logit))
