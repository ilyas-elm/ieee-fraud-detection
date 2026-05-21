from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "xgb_generalized_primary.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "fraud_model_v1_config.json"
PIPELINE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_pipeline.joblib"
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

SUPPORTED_SCHEMA_VERSION = "v1"
MAX_STRING_LENGTH = 256
MAX_ABS_NUMERIC_VALUE = 1e12
LATENCY_WARNING_MS = 500.0

META_FIELDS = {"request_id", "schema_version"}

KNOWN_CATEGORICAL_COLUMNS = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
}

_log_lock = Lock()


class ArtifactLoadError(RuntimeError):
    pass


class PayloadValidationError(ValueError):
    pass


class InferenceError(RuntimeError):
    pass


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load config: {path}") from exc


@lru_cache(maxsize=1)
def load_prediction_artifacts() -> dict[str, Any]:
    for path in [MODEL_PATH, CONFIG_PATH, PIPELINE_PATH]:
        if not path.exists():
            raise ArtifactLoadError(f"Required artifact is missing: {path}")

    try:
        model_artifact = joblib.load(MODEL_PATH)
        feature_pipeline = joblib.load(PIPELINE_PATH)
        config = _load_json(CONFIG_PATH)
    except Exception as exc:
        raise ArtifactLoadError("Failed to load prediction artifacts.") from exc

    required_keys = {"model", "feature_names"}
    missing_keys = required_keys - set(model_artifact)
    if missing_keys:
        raise ArtifactLoadError(f"Model artifact missing keys: {sorted(missing_keys)}")

    model = model_artifact["model"]
    feature_names = list(model_artifact["feature_names"])
    thresholds = model_artifact.get("thresholds", {})

    if not hasattr(model, "predict_proba"):
        raise ArtifactLoadError("Loaded model does not support predict_proba().")

    if not hasattr(feature_pipeline, "input_columns_"):
        raise ArtifactLoadError("Feature pipeline missing input_columns_.")

    if not hasattr(feature_pipeline, "feature_names_out_"):
        raise ArtifactLoadError("Feature pipeline missing feature_names_out_.")

    pipeline_feature_names = list(feature_pipeline.feature_names_out_)

    if pipeline_feature_names != feature_names:
        raise ArtifactLoadError(
            "Model feature names do not match feature pipeline output names."
        )

    if hasattr(model, "n_features_in_") and int(model.n_features_in_) != len(feature_names):
        raise ArtifactLoadError(
            f"Model expects {model.n_features_in_} features, "
            f"artifact provides {len(feature_names)}."
        )

    config_n_features = config.get("n_features")
    if config_n_features is not None and int(config_n_features) != len(feature_names):
        raise ArtifactLoadError(
            f"Config n_features={config_n_features}, expected {len(feature_names)}."
        )

    threshold = thresholds.get("deployed", config.get("threshold_deployed"))
    if threshold is None:
        raise ArtifactLoadError("No deployed threshold found in artifacts/config.")

    model_version = config.get("model_version", "v1")

    return {
        "model": model,
        "feature_pipeline": feature_pipeline,
        "feature_names": feature_names,
        "threshold": float(threshold),
        "model_version": model_version,
        "config": config,
        "model_metadata": model_artifact.get("metadata", {}),
        "score_is_calibrated": bool(config.get("score_is_calibrated", False)),
    }


def _validate_schema_version(payload: dict[str, Any]) -> None:
    schema_version = payload.get("schema_version", SUPPORTED_SCHEMA_VERSION)

    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise PayloadValidationError(
            f"Unsupported schema_version={schema_version!r}. "
            f"Expected {SUPPORTED_SCHEMA_VERSION!r}."
        )


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _validate_and_coerce_value(col: str, value: Any) -> Any:
    # Missing fields are allowed for all optional raw columns.
    # The feature pipeline was trained to handle NaNs.
    if _is_missing_value(value):
        return np.nan

    if isinstance(value, (list, dict, tuple, set)):
        raise PayloadValidationError(f"{col}: nested values are not allowed.")

    if col in KNOWN_CATEGORICAL_COLUMNS:
        value = str(value)

        if len(value) > MAX_STRING_LENGTH:
            raise PayloadValidationError(
                f"{col}: string too long. Max length={MAX_STRING_LENGTH}."
            )

        return value

    if isinstance(value, bool):
        raise PayloadValidationError(f"{col}: boolean is not a valid numeric value.")

    try:
        numeric_value = float(value)
    except Exception as exc:
        raise PayloadValidationError(f"{col}: expected numeric value.") from exc

    if not np.isfinite(numeric_value):
        raise PayloadValidationError(f"{col}: value must be finite.")

    if abs(numeric_value) > MAX_ABS_NUMERIC_VALUE:
        raise PayloadValidationError(f"{col}: numeric value is unreasonably large.")

    if col == "TransactionAmt" and numeric_value <= 0:
        raise PayloadValidationError("TransactionAmt must be positive.")

    if col == "TransactionDT" and numeric_value < 0:
        raise PayloadValidationError("TransactionDT must be non-negative.")

    return numeric_value


def build_raw_input_frame(
    payloads: list[dict[str, Any]],
    expected_columns: list[str],
) -> pd.DataFrame:
    allowed_keys = set(expected_columns) | META_FIELDS
    rows = []

    for i, payload in enumerate(payloads):
        _validate_schema_version(payload)

        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            raise PayloadValidationError(
                f"Request row {i}: unknown fields are not allowed: {unknown[:10]}"
            )

        if "TransactionAmt" not in payload or _is_missing_value(payload["TransactionAmt"]):
            raise PayloadValidationError(f"Request row {i}: TransactionAmt is required.")

        row = {}

        for col in expected_columns:
            if col in payload:
                row[col] = _validate_and_coerce_value(col, payload[col])
            else:
                row[col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows, columns=expected_columns)


def _ensure_feature_frame(
    transformed: Any,
    feature_names: list[str],
) -> pd.DataFrame:
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


def align_features_strict(
    features: pd.DataFrame,
    model_feature_names: list[str],
) -> pd.DataFrame:
    actual = list(features.columns)
    expected = list(model_feature_names)

    missing = [col for col in expected if col not in features.columns]
    extra = [col for col in actual if col not in set(expected)]

    if missing:
        raise InferenceError(f"Missing transformed model features: {missing[:10]}")

    if extra:
        raise InferenceError(f"Unexpected transformed model features: {extra[:10]}")

    aligned = features.loc[:, expected]

    if aligned.isna().any().any():
        bad_cols = aligned.columns[aligned.isna().any()].tolist()
        raise InferenceError(f"Transformed features contain NaNs: {bad_cols[:10]}")

    return aligned


def _write_prediction_log(record: dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with _log_lock:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Logging must never break fraud inference.
        pass


def predict_fraud_batch(payloads: list[Any]) -> list[dict[str, Any]]:
    request_start = time.perf_counter()
    artifacts = load_prediction_artifacts()

    normalized_payloads = [_payload_to_dict(payload) for payload in payloads]
    request_ids = [
        str(payload.get("request_id") or uuid.uuid4())
        for payload in normalized_payloads
    ]

    model = artifacts["model"]
    feature_pipeline = artifacts["feature_pipeline"]
    feature_names = artifacts["feature_names"]
    threshold = artifacts["threshold"]

    try:
        raw_input = build_raw_input_frame(
            normalized_payloads,
            expected_columns=list(feature_pipeline.input_columns_),
        )

        transform_start = time.perf_counter()
        transformed = feature_pipeline.transform(raw_input)
        transform_ms = (time.perf_counter() - transform_start) * 1000

        features = _ensure_feature_frame(transformed, feature_pipeline.feature_names_out_)
        features = align_features_strict(features, feature_names)

        predict_start = time.perf_counter()
        scores = model.predict_proba(features)[:, 1]
        predict_ms = (time.perf_counter() - predict_start) * 1000

    except PayloadValidationError:
        raise
    except Exception as exc:
        raise InferenceError("Prediction pipeline failed.") from exc

    total_ms = (time.perf_counter() - request_start) * 1000

    responses = []

    for request_id, score in zip(request_ids, scores):
        score = float(score)
        is_fraud = bool(score >= threshold)

        response = {
            "request_id": request_id,
            "fraud_probability": score,
            "is_fraud": is_fraud,
            "threshold_used": threshold,
            "model_version": artifacts["model_version"],
            "score_is_calibrated": artifacts["score_is_calibrated"],
            "score_type": "xgboost_predict_proba",
            "inference_time_ms": total_ms,
            "transform_time_ms": transform_ms,
            "predict_time_ms": predict_ms,
            "latency_warning": total_ms > LATENCY_WARNING_MS,
        }

        responses.append(response)

        _write_prediction_log(
            {
                "timestamp_utc": _utc_now(),
                "request_id": request_id,
                "model_version": artifacts["model_version"],
                "fraud_probability": score,
                "is_fraud": is_fraud,
                "threshold_used": threshold,
                "score_is_calibrated": artifacts["score_is_calibrated"],
                "inference_time_ms": total_ms,
                "transform_time_ms": transform_ms,
                "predict_time_ms": predict_ms,
                "payload_logged": False,
            }
        )

    return responses


def predict_fraud(payload: Any) -> dict[str, Any]:
    return predict_fraud_batch([payload])[0]
