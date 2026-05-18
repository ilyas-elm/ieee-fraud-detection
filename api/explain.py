from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from api.predict import (
    InferenceError,
    PayloadValidationError,
    align_features_strict,
    build_raw_input_frame,
    load_prediction_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLAIN_LOG_PATH = PROJECT_ROOT / "logs" / "explanations.jsonl"

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

_explain_log_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)

    if hasattr(payload, "model_dump"):
        return payload.model_dump()

    if hasattr(payload, "dict"):
        return payload.dict()

    raise PayloadValidationError("Payload must be a JSON object or Pydantic model.")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if not np.isfinite(value) else value

    if isinstance(value, float):
        return None if not np.isfinite(value) else value

    if pd.isna(value):
        return None

    return value


def _extract_top_k(payload: dict[str, Any]) -> int:
    top_k = payload.pop("top_k", DEFAULT_TOP_K)

    try:
        top_k = int(top_k)
    except Exception as exc:
        raise PayloadValidationError("top_k must be an integer.") from exc

    if top_k < 1:
        raise PayloadValidationError("top_k must be >= 1.")

    return min(top_k, MAX_TOP_K)


def _ensure_feature_frame(transformed: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(transformed, pd.DataFrame):
        return transformed.copy()

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    arr = np.asarray(transformed)

    if arr.ndim != 2:
        raise InferenceError(f"Transformed features must be 2D, got shape={arr.shape}.")

    if arr.shape[1] != len(feature_names):
        raise InferenceError(
            f"Transformed feature count={arr.shape[1]}, expected={len(feature_names)}."
        )

    return pd.DataFrame(arr, columns=feature_names)


def _prepare_single_feature_row(payload: dict[str, Any]) -> pd.DataFrame:
    artifacts = load_prediction_artifacts()
    feature_pipeline = artifacts["feature_pipeline"]
    feature_names = artifacts["feature_names"]

    raw_input = build_raw_input_frame(
        [payload],
        expected_columns=list(feature_pipeline.input_columns_),
    )

    transformed = feature_pipeline.transform(raw_input)
    features = _ensure_feature_frame(
        transformed,
        feature_names=list(feature_pipeline.feature_names_out_),
    )

    return align_features_strict(features, feature_names)


def _compute_xgboost_shap_contributions(
    model: Any,
    features: pd.DataFrame,
) -> tuple[np.ndarray, float, float]:
    if not hasattr(model, "get_booster"):
        raise InferenceError("Model does not expose get_booster(); cannot compute SHAP.")

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        features,
        feature_names=list(features.columns),
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        validate_features=True,
    )

    if contributions.ndim != 2 or contributions.shape[1] != features.shape[1] + 1:
        raise InferenceError(
            "Unexpected SHAP contribution shape: "
            f"{contributions.shape}, expected (*, {features.shape[1] + 1})."
        )

    shap_values = contributions[0, :-1]
    base_value = float(contributions[0, -1])
    raw_margin = float(base_value + shap_values.sum())

    return shap_values, base_value, raw_margin


def _top_shap_features(
    features: pd.DataFrame,
    shap_values: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    row = features.iloc[0]
    ranked_idx = np.argsort(np.abs(shap_values))[::-1][:top_k]

    output = []

    for idx in ranked_idx:
        feature = features.columns[idx]
        shap_value = float(shap_values[idx])

        if shap_value > 0:
            direction = "increases_fraud_score"
        elif shap_value < 0:
            direction = "decreases_fraud_score"
        else:
            direction = "neutral"

        output.append(
            {
                "feature": feature,
                "feature_value": _json_safe(row.iloc[idx]),
                "shap_value": shap_value,
                "abs_shap_value": abs(shap_value),
                "direction": direction,
            }
        )

    return output


def _write_explanation_log(record: dict[str, Any]) -> None:
    try:
        EXPLAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with _explain_log_lock:
            with open(EXPLAIN_LOG_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def explain_fraud(payload: Any) -> dict[str, Any]:
    start = time.perf_counter()

    payload_dict = _payload_to_dict(payload)
    request_id = str(payload_dict.get("request_id") or uuid.uuid4())
    top_k = _extract_top_k(payload_dict)

    artifacts = load_prediction_artifacts()
    model = artifacts["model"]
    threshold = artifacts["threshold"]

    try:
        features = _prepare_single_feature_row(payload_dict)
        fraud_probability = float(model.predict_proba(features)[:, 1][0])
        is_fraud = bool(fraud_probability >= threshold)

        shap_values, base_value, raw_margin = _compute_xgboost_shap_contributions(
            model,
            features,
        )

        top_features = _top_shap_features(
            features=features,
            shap_values=shap_values,
            top_k=top_k,
        )

    except PayloadValidationError:
        raise
    except Exception as exc:
        raise InferenceError("Explanation pipeline failed.") from exc

    inference_time_ms = (time.perf_counter() - start) * 1000

    response = {
        "request_id": request_id,
        "fraud_probability": fraud_probability,
        "is_fraud": is_fraud,
        "threshold_used": threshold,
        "model_version": artifacts["model_version"],
        "score_is_calibrated": artifacts["score_is_calibrated"],
        "explanation_method": "xgboost_native_treeshap",
        "explanation_space": "raw_margin_log_odds",
        "base_value": base_value,
        "raw_margin": raw_margin,
        "top_features": top_features,
        "inference_time_ms": inference_time_ms,
    }

    _write_explanation_log(
        {
            "timestamp_utc": _utc_now(),
            "request_id": request_id,
            "model_version": artifacts["model_version"],
            "fraud_probability": fraud_probability,
            "is_fraud": is_fraud,
            "threshold_used": threshold,
            "top_k": top_k,
            "top_features": top_features,
            "inference_time_ms": inference_time_ms,
            "payload_logged": False,
        }
    )

    return response
