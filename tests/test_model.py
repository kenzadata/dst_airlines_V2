from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import train as train_module
from src.ml.model import (
    FEATURE_COLUMNS,
    DelayModel,
    build_features,
    build_targets,
    make_hgb_classifier,
    make_hgb_regressor,
)


def make_flights(n: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Vols fictifs sur 6 mois, au format de `dataset.load_training_data`,
    où le retard dépend de la pluie, du vent et de l'heure de départ."""
    rng = np.random.default_rng(seed)
    fl_time_utc = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 181, n)), unit="D")
    crs_dep_time = rng.choice([600, 815, 1230, 1745, 2110], size=n)
    prcp = rng.exponential(1.0, n)
    wspd = rng.gamma(2.0, 8.0, n)
    dep_delay = rng.normal(-2, 8, n) + 6 * prcp + 0.4 * wspd + (crs_dep_time > 1700) * 10
    return pd.DataFrame(
        {
            "fl_date": fl_time_utc.tz_localize(None).normalize(),
            "fl_time_utc": fl_time_utc,
            "origin": rng.choice(["ATL", "JFK"], n),
            "dest": rng.choice(["LAX", "ORD", "BOS"], n),
            "op_carrier": rng.choice(["DL", "B6", "AA"], n),
            "tail_num": rng.choice(["N1", "N2", None], n),
            "crs_dep_time": crs_dep_time,
            "distance": rng.choice([760.0, 1946.0, 187.0], n),
            "crs_elapsed_time": rng.choice([120.0, 300.0, 75.0], n),
            "aircraft_age": np.where(rng.random(n) < 0.2, np.nan, rng.integers(1, 25, n)),
            "temp": rng.normal(12, 8, n),
            "prcp": prcp,
            "wspd": wspd,
            "pres": rng.normal(1015, 6, n),
            "coco": rng.choice([1.0, 3.0, 8.0, np.nan], n),
            "dep_delay": dep_delay,
        }
    )


def test_build_features_derives_calendar_and_handles_missing_inputs():
    raw = pd.DataFrame({"fl_date": ["2024-07-15"], "crs_dep_time": [2400], "origin": ["atl"], "coco": [8.0]})

    features = build_features(raw)

    assert list(features.columns) == FEATURE_COLUMNS
    row = features.iloc[0]
    assert (row["dep_hour"], row["day_of_week"], row["month"]) == (0, 0, 7)
    assert (row["origin"], row["coco"]) == ("ATL", "8")
    assert pd.isna(row["dest"]) and pd.isna(row["temp"])


def test_build_targets_uses_dot_threshold_and_clips_early_departures():
    delayed, minutes = build_targets(pd.DataFrame({"dep_delay": [-5.0, 14.0, 15.0, 90.0]}))

    assert delayed.tolist() == [0, 0, 1, 1]
    assert minutes.tolist() == [0.0, 14.0, 15.0, 90.0]


def test_split_train_test_is_chronological():
    df = make_flights()

    train_df, test_df = train_module.split_train_test(df, test_months=2)

    assert train_df["fl_date"].max() < test_df["fl_date"].min()
    assert test_df["fl_date"].dt.to_period("M").nunique() == 2


def test_split_train_test_rejects_too_short_period():
    df = make_flights().assign(fl_date=pd.Timestamp("2024-01-15"))

    with pytest.raises(ValueError):
        train_module.split_train_test(df, test_months=3)


def test_best_f1_threshold_separates_perfect_scores():
    y = pd.Series([0, 0, 1, 1])
    assert train_module.best_f1_threshold(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.8


def _fitted_model(df: pd.DataFrame) -> DelayModel:
    X = build_features(df)
    delayed, minutes = build_targets(df)
    return DelayModel(
        model_version="test-version",
        classifier=make_hgb_classifier(max_iter=20).fit(X, delayed),
        regressor=make_hgb_regressor(max_iter=20).fit(X, minutes),
        threshold=0.4,
        importances={"prcp": 0.2, "wspd": 0.1, "temp": 0.0},
    )


def test_delay_model_predicts_from_raw_rows_and_roundtrips(tmp_path):
    df = make_flights(n=300)
    model = _fitted_model(df)

    loaded = DelayModel.load(model.save(models_dir=tmp_path))
    raw = df.drop(columns=["dep_delay"]).head(5).assign(temp=np.nan, dest="UNSEEN")
    probability, minutes = loaded.predict(raw)

    assert ((probability >= 0) & (probability <= 1)).all()
    assert (minutes >= 0).all()
    assert loaded.is_delayed(np.array([0.39, 0.4])).tolist() == [False, True]
    assert list(loaded.feature_importances(top_n=2)) == ["prcp", "wspd"]


def test_train_selects_model_logs_to_mlflow_and_beats_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(DelayModel, "save", lambda self, models_dir=tmp_path: tmp_path / "model.pkl")

    model, metrics = train_module.train(
        make_flights(), model_version="test", test_months=1, n_iter=2, n_splits=2, register=False,
    )

    assert metrics["test_roc_auc"] > 0.7
    assert metrics["test_pr_auc"] > metrics["test_base_rate"]
    assert metrics["test_mae_minutes"] < metrics["baseline_mae_minutes"]
    assert 0 < model.threshold < 1
    assert set(model.importances) == set(FEATURE_COLUMNS)
    assert max(model.importances, key=model.importances.get) in {"prcp", "wspd", "dep_hour"}

    import mlflow

    runs = mlflow.search_runs(experiment_names=[train_module.EXPERIMENT_NAME])
    assert len(runs) == 4  # run principal + 3 candidats


def test_train_drops_entirely_empty_feature(tmp_path, monkeypatch):
    """Sans registre FAA, aircraft_age est vide partout : l'entraînement doit l'écarter, pas planter."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(DelayModel, "save", lambda self, models_dir=tmp_path: tmp_path / "model.pkl")

    model, _ = train_module.train(
        make_flights().assign(aircraft_age=np.nan), model_version="no-faa",
        test_months=1, n_iter=1, n_splits=2, register=False,
    )

    assert model.importances["aircraft_age"] == 0.0
    probability, _ = model.predict(make_flights(n=5).drop(columns=["dep_delay"]))
    assert len(probability) == 5


def test_load_rejects_model_saved_before_current_format(tmp_path):
    import joblib

    model = _fitted_model(make_flights(n=200))
    del model.__dict__["metrics"]  # modèle sérialisé avant l'ajout du champ
    path = tmp_path / "delay_model_old.pkl"
    joblib.dump(model, path)

    with pytest.raises(TypeError):
        DelayModel.load(path)


def test_predicted_minutes_are_capped_to_plausible_range():
    df = make_flights(n=300)
    model = _fitted_model(df)
    model.max_delay_minutes = 30.0
    extreme = df.drop(columns=["dep_delay"]).head(3).assign(prcp=50.0, wspd=150.0, crs_dep_time=2300)

    _, minutes = model.predict(extreme)

    assert (minutes <= 30.0).all() and (minutes >= 0).all()
