"""Feature engineering + modèle de prédiction de retard.

Ce module ne fait qu'entraîner/prédire (README section 10, point 3) : la
lecture des données et l'orchestration de l'entraînement vivent dans
`train.py`, la lecture des requêtes API dans `src/api/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["temperature", "wind_speed", "precipitation", "visibility", "hour"]
CATEGORICAL_FEATURES = ["origin_airport"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les colonnes de features à partir des données jointes vols+météo."""
    features = df.copy()
    if "observed_at" in features.columns:
        observed_at = pd.to_datetime(features["observed_at"], utc=True)
        features["hour"] = observed_at.dt.hour
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].fillna(
        features[NUMERIC_FEATURES].median(numeric_only=True)
    )
    return features[FEATURE_COLUMNS]


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


@dataclass
class DelayModel:
    """Regroupe un classifieur (probabilité de retard) et un régresseur
    (minutes de retard estimées) partageant le même prétraitement de features.
    """

    model_version: str
    classifier: Pipeline
    regressor: Pipeline

    @classmethod
    def new(cls, model_version: str, random_state: int = 42) -> DelayModel:
        classifier = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor()),
                ("estimator", RandomForestClassifier(n_estimators=200, random_state=random_state)),
            ]
        )
        regressor = Pipeline(
            steps=[
                ("preprocess", _build_preprocessor()),
                ("estimator", RandomForestRegressor(n_estimators=200, random_state=random_state)),
            ]
        )
        return cls(model_version=model_version, classifier=classifier, regressor=regressor)

    def fit(self, X: pd.DataFrame, y_delayed: pd.Series, y_delay_minutes: pd.Series) -> DelayModel:
        self.classifier.fit(X, y_delayed)
        self.regressor.fit(X, y_delay_minutes)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        proba = self.classifier.predict_proba(X)
        classes = list(self.classifier.classes_)
        # Un jeu d'entraînement déséquilibré peut ne présenter qu'une seule
        # classe (aucun retard observé) : predict_proba n'a alors qu'une
        # colonne, il faut la retrouver par position plutôt qu'indexer [:, 1].
        delay_probability = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        predicted_delay_minutes = self.regressor.predict(X)
        return delay_probability, predicted_delay_minutes

    def feature_importances(self, top_n: int = 10) -> dict[str, float]:
        """Expose les facteurs principaux du régresseur (README section 10.8 :
        interprétabilité obligatoire pour toute fonctionnalité de prédiction)."""
        preprocess = self.regressor.named_steps["preprocess"]
        estimator = self.regressor.named_steps["estimator"]
        feature_names = preprocess.get_feature_names_out()
        importances = estimator.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
        return {name: float(score) for name, score in ranked[:top_n]}

    def save(self, models_dir: Path = MODELS_DIR) -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"delay_model_{self.model_version}.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> DelayModel:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Fichier modèle invalide : {path}")
        return model


def latest_model_path(models_dir: Path = MODELS_DIR) -> Path | None:
    candidates = sorted(models_dir.glob("delay_model_*.pkl"))
    return candidates[-1] if candidates else None
