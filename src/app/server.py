"""
Stage 10 web application: FastAPI backend + static dashboard (src/app/static/).

  GET  /                      dashboard (static HTML/CSS/JS)
  GET  /health                status and whether the models loaded
  GET  /api/models            model, threshold and Stage 9 evaluation information
  POST /api/analyze           multipart upload: file (.py, <= 2 MB) [+ relative_path]; returns scored methods
  GET  /api/methods/{id}      details of a method from one of the last analyses (in memory only; no source)

Uploaded source is analysed in memory and a temporary folder that is deleted immediately; it is never
stored. Errors never include server paths. Start with `python run.py` (localhost only).
"""
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.app.inference import (MAX_BYTES, REVIEW_CUTOFF, InferenceError, ModelArtifactError, analyse,  # noqa: E402
                               load_models, validate_review_cutoff)

STATIC = Path(__file__).resolve().parent / "static"
LABEL_DISCLAIMER = ("Experimental research prototype. All 243 positive training labels and all 50 positive test labels "
                    "were constructed automatically and have not been verified by a person. Scores are model outputs, "
                    "not calibrated probabilities and not proof that a method will be renamed. No prospective "
                    "real-world validation has been performed.")
CACHE_SIZE = 20


def create_app(registry_loader=load_models, review_cutoff=REVIEW_CUTOFF):
    review_cutoff = validate_review_cutoff(review_cutoff)
    app = FastAPI(title="Python method rename-risk (research prototype)", docs_url="/api/docs", redoc_url=None)
    state = {"registry": None, "error": None, "cache": OrderedDict()}
    try:
        state["registry"] = registry_loader()
    except ModelArtifactError as exc:              # the app still starts and reports the problem
        state["error"] = str(exc)
    except Exception as exc:                       # never leak paths or tracebacks
        state["error"] = f"models could not be loaded ({type(exc).__name__})"

    def registry():
        if state["registry"] is None:
            raise HTTPException(503, f"Models unavailable: {state['error']}")
        return state["registry"]

    @app.get("/health")
    def health():
        return {"status": "ok" if state["registry"] else "degraded", "models_loaded": state["registry"] is not None,
                "error": state["error"]}

    @app.get("/api/models")
    def models():
        reg = registry()
        return {"primary": "main", "secondary": "comparison", "disclaimer": LABEL_DISCLAIMER,
                "models": {fs: {"run_id": m.run_id, "n_features": len(m.features), "features": m.features,
                                "threshold": m.threshold, "artifact_sha256": m.sha256,
                                "threshold_source": "Stage 8: max F1 on pooled development out-of-fold scores"}
                           for fs, m in reg.models.items()},
                "evaluation": reg.info,
                "threshold_note": ("The threshold was chosen on development out-of-fold scores of five fold models "
                                   "and is applied to one model refitted on all development data. In Stage 9 it "
                                   "flagged fewer test methods than on development (main 1.3% vs 2.1%), so "
                                   "'above threshold' is only a rough operating point."),
                "review_cutoff": review_cutoff,
                "review_cutoff_note": (f"The {review_cutoff:.2f} review cutoff is chosen by the user and is NOT validated. "
                                       "Methods at or above it are suggested for review; most will not be renamed. On "
                                       "development out-of-fold scores, 0.70 flags about 11% of methods and finds about "
                                       "20% of renames (main 40). The validated Stage 8 thresholds (main 0.859, "
                                       "comparison 0.846) are shown separately."),
                "dataset": {"rows": 31209, "positives": 243, "projects": 323, "development_projects": 261,
                            "test_projects": 62}}

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), relative_path: str | None = Form(None)):
        reg = registry()
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).")
        try:
            result = analyse(data, file.filename or "", reg, relative_path=relative_path or None,
                             review_cutoff=review_cutoff)
        except InferenceError as exc:
            raise HTTPException(400, str(exc)) from None
        finally:
            del data                                # the upload is not kept
        analysis_id = uuid.uuid4().hex[:12]
        cache = state["cache"]
        cache[analysis_id] = {m["method_id"]: {k: v for k, v in m.items() if k != "source"} for m in result["methods"]}
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
        return {"analysis_id": analysis_id, "disclaimer": LABEL_DISCLAIMER, **result}

    @app.get("/api/methods/{method_id}")
    def method(method_id: str, analysis_id: str):
        entry = state["cache"].get(analysis_id, {}).get(method_id)
        if entry is None:
            raise HTTPException(404, "Unknown analysis or method (results are kept in memory for the last 20 analyses).")
        return entry

    @app.exception_handler(Exception)
    async def unexpected(request, exc):             # generic message; no traceback or paths
        return JSONResponse(status_code=500, content={"detail": "Internal error while processing the request."})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app
