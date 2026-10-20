"""Entraînement du modèle de prédiction de retard, avec suivi MLflow
(README section 6, étape 3 : "mettre en place MLflow pour le versionning
des modèles").

Limite MVP (cf. `src/etl/pipeline.py`) : les données collectées via l'API
OpenSky gratuite ne portent pas d'horaires programmés/réels, donc pas de
retard observé à apprendre. En attendant de brancher une source d'horaires
(ex. AviationStack), ce script retombe sur un jeu de données synthétique
plausible (retard corrélé à la météo) afin que tout le pipeline
collecte → ETL → ML → API reste exécutable de bout en bout. Dès qu'un CSV
labellisé (`delayed`, `delay_minutes`) apparaît dans `data/processed/`, il
est utilisé à la place.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from src.ml.model import DelayModel, build_features

load_dotenv()

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
PILOT_AIRPORTS = os.environ.get("PILOT_AIRPORTS", "ATL,JFK").split(",")


def make_demo_training_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Génère un jeu de données synthétique mais réaliste : le retard est
    corrélé au vent, aux précipitations et à une visibilité réduite."""
    rng = np.random.default_rng(seed)
    airports = [code.strip().upper() for code in PILOT_AIRPORTS if code.strip()]

    hours = rng.integers(0, 24, size=n)
    wind_speed = rng.gamma(shape=2.0, scale=8.0, size=n)
    precipitation = rng.exponential(scale=1.2, size=n)
    visibility = rng.normal(loc=9000, scale=2500, size=n).clip(min=200)
    temperature = rng.normal(loc=14, scale=7, size=n)

    low_visibility_penalty = np.clip((6000 - visibility) / 1000, 0, None)
    delay_score = (
        0.5 * wind_speed
        + 4.0 * precipitation
        + 6.0 * low_visibility_penalty
        + 0.6 * np.abs(hours - 8)
        - 12.0
    )
    delay_minutes = np.clip(delay_score + rng.normal(0, 6, size=n), 0, None)
    delayed = (delay_minutes > 15).astype(int)

    return pd.DataFrame(
        {
            "observed_at": pd.Timestamp.now(tz="UTC") - pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
            "origin_airport": rng.choice(airports, size=n),
            "temperature": temperature,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
            "visibility": visibility,
            "hour": hours,
            "delayed": delayed,
            "delay_minutes": delay_minutes,
        }
    )


def load_training_data(processed_dir: Path = PROCESSED_DATA_DIR) -> pd.DataFrame:
    for path in sorted(processed_dir.glob("flights_weather_*.csv"), reverse=True):
        df = pd.read_csv(path)
        if {"delayed", "delay_minutes"}.issubset(df.columns):
            logger.info("Entraînement à partir des données labellisées de %s", path)
            return df

    logger.warning(
        "Aucun CSV labellisé trouvé dans %s : entraînement sur un jeu de "
        "démonstration synthétique (cf. docstring du module).",
        processed_dir,
    )
    return make_demo_training_data()


def train(df: pd.DataFrame, model_version: str | None = None) -> tuple[DelayModel, dict[str, float]]:
    model_version = model_version or datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    X = build_features(df)
    y_delayed = df["delayed"].astype(int)
    y_minutes = df["delay_minutes"].astype(float)

    stratify = y_delayed if y_delayed.nunique() > 1 else None
    X_train, X_test, y_delayed_train, y_delayed_test, y_minutes_train, y_minutes_test = train_test_split(
        X, y_delayed, y_minutes, test_size=0.2, random_state=42, stratify=stratify
    )

    model = DelayModel.new(model_version).fit(X_train, y_delayed_train, y_minutes_train)
    delay_probability_pred, minutes_pred = model.predict(X_test)

    metrics: dict[str, float] = {"mae_minutes": float(mean_absolute_error(y_minutes_test, minutes_pred))}
    if y_delayed_test.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_delayed_test, delay_probability_pred))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("dst-airlines-delay-prediction")
    with mlflow.start_run(run_name=model_version):
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_metrics(metrics)
        mlflow.log_dict(model.feature_importances(), "feature_importances.json")

    saved_path = model.save()
    logger.info("Modèle %s sauvegardé dans %s (métriques : %s)", model_version, saved_path, metrics)
    return model, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    training_data = load_training_data()
    train(training_data)
