import os
os.environ["FRAUD_API_KEY"] = "dev-secret-key-change-in-prod"

from fastapi.testclient import TestClient
from api.predict import load_prediction_artifacts
from api.main import app

# Clear any cached failed load from import time
load_prediction_artifacts.cache_clear()

client = TestClient(app, raise_server_exceptions=False)
API_KEY = "dev-secret-key-change-in-prod"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_version" in data
    assert "threshold" in data


def test_predict_valid():
    response = client.post(
        "/predict",
        json={"TransactionAmt": 150.0, "ProductCD": "W"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold_used" in data
    assert "model_version" in data


def test_predict_missing_amount():
    response = client.post(
        "/predict",
        json={"ProductCD": "W"},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_predict_wrong_key():
    response = client.post(
        "/predict",
        json={"TransactionAmt": 150.0},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_predict_no_key():
    response = client.post(
        "/predict",
        json={"TransactionAmt": 150.0},
    )
    assert response.status_code == 401