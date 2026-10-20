from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import main as api_main
from src.etl.db import get_engine
from src.ml.model import (
    DelayModel,
    build_features,
    build_targets,
    make_hgb_classifier,
    make_hgb_regressor,
)
from tests.test_model import make_flights

_PREDICT_PAYLOAD = {
    "flight_id": "DL1234_20240115",
    "fl_date": "2024-01-15",
    "crs_dep_time": 1745,
    "origin": "ATL",
    "dest": "LAX",
    "op_carrier": "DL",
    "distance": 1946.0,
    "temp": 4.5,
    "prcp": 1.2,
    "wspd": 25.0,
}


def _trained_model() -> DelayModel:
    df = make_flights(n=200)
    X = build_features(df)
    delayed, minutes = build_targets(df)
    return DelayModel(
        model_version="test-version",
        classifier=make_hgb_classifier(max_iter=10).fit(X, delayed),
        regressor=make_hgb_regressor(max_iter=10).fit(X, minutes),
        threshold=0.5,
        importances={"prcp": 0.3, "wspd": 0.1},
    )


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
    assert body["predicted_delayed"] == (body["delay_probability"] >= 0.5)
    assert body["predicted_delay_minutes"] >= 0
    assert body["model_version"] == "test-version"
    assert list(body["feature_importances"]) == ["prcp", "wspd"]


def test_predict_rejects_invalid_departure_time(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_model", _trained_model)
    monkeypatch.setattr(api_main, "get_engine", lambda: get_engine(f"sqlite:///{tmp_path}/test.db"))

    with TestClient(api_main.app) as client:
        response = client.post("/predict", json={**_PREDICT_PAYLOAD, "crs_dep_time": 2500})

    assert response.status_code == 422


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_model", lambda: None)
    monkeypatch.setattr(api_main, "get_engine", lambda: get_engine(f"sqlite:///{tmp_path}/test.db"))

    with TestClient(api_main.app) as client:
        response = client.post("/predict", json=_PREDICT_PAYLOAD)

    assert response.status_code == 503


def test_get_model_ignores_model_in_previous_format(monkeypatch, tmp_path):
    import joblib

    old_model = _trained_model()
    del old_model.__dict__["threshold"]
    joblib.dump(old_model, tmp_path / "delay_model_old.pkl")
    monkeypatch.setattr(api_main, "_model", None)
    monkeypatch.setattr(api_main, "latest_model_path", lambda: tmp_path / "delay_model_old.pkl")

    assert api_main.get_model() is None
