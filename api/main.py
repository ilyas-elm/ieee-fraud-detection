from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from api.explain import explain_fraud
from api.predict import (
    ArtifactLoadError,
    InferenceError,
    PayloadValidationError,
    load_prediction_artifacts,
    predict_fraud,
    predict_fraud_batch,
)


API_KEY_ENV = "FRAUD_API_KEY"


def get_expected_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV)

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server misconfigured: {API_KEY_ENV} is not set.",
        )

    return api_key


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected_api_key = get_expected_api_key()

    if not x_api_key or not hmac.compare_digest(x_api_key, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artifacts = load_prediction_artifacts()
        app.state.model_version = artifacts["model_version"]
        app.state.threshold = artifacts["threshold"]
    except Exception as exc:
        raise RuntimeError("Failed to load model artifacts during startup.") from exc

    yield


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": getattr(app.state, "model_version", None),
        "threshold": getattr(app.state, "threshold", None),
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(payload: dict):
    try:
        return predict_fraud(payload)
    except PayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch", dependencies=[Depends(require_api_key)])
def predict_batch(payloads: list[dict]):
    try:
        return predict_fraud_batch(payloads)
    except PayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain", dependencies=[Depends(require_api_key)])
def explain(payload: dict):
    try:
        return explain_fraud(payload)
    except PayloadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ArtifactLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
