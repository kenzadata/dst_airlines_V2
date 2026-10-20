from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.model import DelayModel, build_features


def _sample_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "observed_at": pd.Timestamp("2024-01-01T08:00", tz="UTC"),
            "origin_airport": rng.choice(["ATL", "JFK"], size=n),
            "temperature": rng.normal(10, 5, n),
            "wind_speed": rng.normal(20, 5, n),
            "precipitation": rng.exponential(1, n),
            "visibility": rng.normal(9000, 1000, n),
            "delayed": rng.integers(0, 2, n),
            "delay_minutes": rng.uniform(0, 60, n),
        }
    )


def test_build_features_returns_expected_columns():
    features = build_features(_sample_df())

    assert list(features.columns) == [
        "temperature", "wind_speed", "precipitation", "visibility", "hour", "origin_airport",
    ]
    assert features["hour"].between(0, 23).all()


def test_delay_model_fit_predict_roundtrip():
    df = _sample_df(n=40)
    features = build_features(df)

    model = DelayModel.new("test-version").fit(features, df["delayed"], df["delay_minutes"])
    probabilities, minutes = model.predict(features)

    assert len(probabilities) == len(df)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert len(minutes) == len(df)


def test_delay_model_feature_importances_not_empty():
    df = _sample_df(n=40)
    features = build_features(df)
    model = DelayModel.new("test-version").fit(features, df["delayed"], df["delay_minutes"])

    importances = model.feature_importances()

    assert len(importances) > 0
    assert all(isinstance(v, float) for v in importances.values())


def test_delay_model_save_and_load_roundtrip(tmp_path):
    df = _sample_df(n=40)
    features = build_features(df)
    model = DelayModel.new("v1").fit(features, df["delayed"], df["delay_minutes"])

    saved_path = model.save(models_dir=tmp_path)
    loaded = DelayModel.load(saved_path)

    assert loaded.model_version == "v1"
    probabilities, _ = loaded.predict(features)
    assert len(probabilities) == len(df)
