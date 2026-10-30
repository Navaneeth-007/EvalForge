"""Single-service showcase with an explicit fixture mode and optional local inference."""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.examples import EXAMPLES, SCHEMA
from app.schemas.request import SQLGenerationRequest, SQLGenerationResponse
from train.dataset_prep import format_prompt

logger = logging.getLogger(__name__)
MODE = os.getenv("EVALFORGE_MODE", "walkthrough")
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
ADAPTER_MODEL = os.getenv("ADAPTER_MODEL")
_pipe = None
_model_info = {}
_inference_lock = threading.Lock()
STATIC = Path(__file__).parent / "static"


def load_model():
    global _pipe, _model_info
    if MODE == "walkthrough":
        _model_info = {"mode": MODE, "adapter": None, "training_status": "pending"}
        return
    if MODE != "model":
        raise ValueError("EVALFORGE_MODE must be walkthrough or model")
    if not ADAPTER_MODEL:
        raise ValueError("Model mode requires ADAPTER_MODEL; no silent base-model fallback")
    from eval.run_eval import load_pipeline
    _pipe = load_pipeline(BASE_MODEL, ADAPTER_MODEL, os.getenv("USE_4BIT", "false").lower() == "true")
    _model_info = {"mode": MODE, "base_model": BASE_MODEL, "adapter": ADAPTER_MODEL}


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(title="EvalForge", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    if request.headers.get("content-length", "").isdigit() and int(request.headers["content-length"]) > 24000:
        return JSONResponse({"detail": "Request too large"}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'" if request.url.path != "/docs" else "frame-ancestors 'none'"
    return response


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    ready = MODE == "walkthrough" or _pipe is not None
    return JSONResponse({"status": "ok" if ready else "unavailable", "ready": ready, "model": _model_info}, status_code=200 if ready else 503)


@app.get("/examples")
def examples():
    return {"examples": [{"name": "Employee analytics", "schema": SCHEMA,
                          "sample_questions": [item["question"] for item in EXAMPLES]}],
            "mode": MODE}


@app.get("/results")
def results():
    return {"status": "pending", "message": "No verified training run or benchmark results published yet.",
            "base_model": BASE_MODEL, "metrics": None}


@app.post("/generate-sql", response_model=SQLGenerationResponse)
def generate_sql(request: SQLGenerationRequest):
    started = time.monotonic()
    if MODE == "walkthrough":
        example = next((item for item in EXAMPLES if item["question"].casefold() == request.question.casefold()
                        and item["schema"].strip() == request.schema.strip()), None)
        if not example:
            raise HTTPException(422, "Walkthrough mode supports only the listed examples. A trained adapter is required for custom questions.")
        sql = example["sql"]
        model = "authored-example (no model inference)"
    else:
        if _pipe is None:
            raise HTTPException(503, "Model unavailable")
        if not _inference_lock.acquire(blocking=False):
            raise HTTPException(429, "Model is busy; try again shortly", headers={"Retry-After": "5"})
        try:
            prompt = format_prompt(request.question, request.schema)
            if len(_pipe.tokenizer.encode(prompt)) > 3800:
                raise HTTPException(422, "Prompt exceeds model context limit")
            output = _pipe(prompt, return_full_text=False)
            sql = output[0]["generated_text"].split("<|end|>")[0].strip()
            model = ADAPTER_MODEL
        except HTTPException:
            raise
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(503, "Inference unavailable; try again later") from None
        finally:
            _inference_lock.release()
    return SQLGenerationResponse(question=request.question, schema=request.schema, sql=sql,
                                 latency_ms=round((time.monotonic()-started)*1000), model=model, mode=MODE)
