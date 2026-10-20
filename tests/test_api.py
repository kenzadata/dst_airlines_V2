from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.etl.db import get_engine
from src.ml.model import DelayModel, build_features

_PREDICT_PAYLOAD = {
    "flight_id": "abc123_20240101",
    "origin_airport": "ATL",
    "temperature": 10.0,
    "wind_speed": 15.0,
    "precipitation": 0.0,
    "visibility": 9000.0,
    "hour": 8,
}


def _trained_model() -> DelayModel:
    df = pd.DataFrame(
        {
            "observed_at": pd.Timestamp("2024-01-01T08:00", tz="UTC"),
            "origin_airport": ["ATL", "JFK", "JFK", "ATL"],
            "temperature": [10.0, 12.0, 15.0, 8.0],
            "wind_speed": [10.0, 30.0, 5.0, 40.0],
            "precipitation": [0.0, 5.0, 0.0, 8.0],
            "visibility": [9000.0, 3000.0, 9500.0, 2000.0],
            "delayed": [0, 1, 0, 1],
            "delay_minutes": [2.0, 30.0, 1.0, 45.0],
        }
    )
    features = build_features(df)
    return DelayModel.new("test-version").fit(features, df["delayed"], df["delay_minutes"])


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_model", lambda: None)
    monkeypatch.setattr(api_main, "get_engine", lambda: get_engine(f"sqlite:///{tmp_path}/test.db"))

    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": None}


def test_predict_returns_prediction(monkeypatch, tmp_path):
    model = _trained_model()
    monkeypatch.setattr(api_main, "get_model", lambda: model)
    monkeypatch.setattr(api_main, "get_engine", lambda: get_engine(f"sqlite:///{tmp_path}/test.db"))

    with TestClient(api_main.app) as client:
        response = client.post("/predict", json=_PREDICT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["flight_id"] == _PREDICT_PAYLOAD["flight_id"]
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["model_version"] == "test-version"
    assert body["feature_importances"]


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_model", lambda: None)
    monkeypatch.setattr(api_main, "get_engine", lambda: get_engine(f"sqlite:///{tmp_path}/test.db"))

    with TestClient(api_main.app) as client:
        response = client.post("/predict", json=_PREDICT_PAYLOAD)

    assert response.status_code == 503
