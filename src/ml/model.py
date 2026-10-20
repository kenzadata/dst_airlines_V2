"""Feature engineering + modèle de prédiction du retard au départ.

Ce module ne fait qu'entraîner/prédire (README section 10, point 3) : la
lecture des données et l'orchestration de l'entraînement vivent dans
`dataset.py` et `train.py`, la lecture des requêtes API dans `src/api/`.

Un même `build_features` sert à l'entraînement (lignes de l'entrepôt) et à
l'inférence (requêtes API, dashboard) : aucune divergence possible entre les
features vues à l'entraînement et celles reçues en production.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Seuil officiel du DOT : un vol est "en retard" à partir de 15 minutes.
DELAY_THRESHOLD_MINUTES = 15

# Colonnes brutes attendues en entrée (lignes de l'entrepôt ou requête API).
# Toutes sont connues avant le départ ; la météo est celle de l'heure prévue.
RAW_INPUT_COLUMNS = [
    "fl_date", "crs_dep_time", "origin", "dest", "op_carrier",
    "distance", "crs_elapsed_time", "aircraft_age",
    "temp", "prcp", "wspd", "pres", "coco",
]

# `coco` (code condition Meteostat) est un code, pas une grandeur : catégoriel.
CATEGORICAL_FEATURES = ["origin", "dest", "op_carrier", "coco"]
NUMERIC_FEATURES = [
    "dep_hour", "day_of_week", "month",
    "distance", "crs_elapsed_time", "aircraft_age",
    "temp", "prcp", "wspd", "pres",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Destinations / compagnies vues moins souvent sont regroupées en "rare".
MIN_CATEGORY_FREQUENCY = 20

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features à partir des colonnes brutes `RAW_INPUT_COLUMNS`.

    Les colonnes optionnelles absentes (avion inconnu du registre FAA, météo
    manquante) deviennent des valeurs manquantes, gérées nativement par le modèle.
    """
    raw = df.reindex(columns=RAW_INPUT_COLUMNS)
    fl_date = pd.to_datetime(raw["fl_date"])
    hhmm = pd.to_numeric(raw["crs_dep_time"], errors="coerce") % 2400

    features = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_FEATURES:
        values = raw[col]
        if col == "coco":
            values = pd.to_numeric(values, errors="coerce").astype("Int64")
        values = values.astype("string").str.strip().str.upper()
        features[col] = values.astype(object).where(values.notna(), np.nan)

    features["dep_hour"] = hhmm // 100
    features["day_of_week"] = fl_date.dt.dayofweek
    features["month"] = fl_date.dt.month
    for col in ["distance", "crs_elapsed_time", "aircraft_age", "temp", "prcp", "wspd", "pres"]:
        features[col] = pd.to_numeric(raw[col], errors="coerce")
    features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].astype(float)
    return features[FEATURE_COLUMNS]


def build_targets(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cibles : retard >= 15 min (classification) et minutes de retard >= 0 (régression)."""
    dep_delay = pd.to_numeric(df["dep_delay"], errors="coerce")
    return (dep_delay >= DELAY_THRESHOLD_MINUTES).astype(int), dep_delay.clip(lower=0).astype(float)


def active_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Features (catégorielles, numériques) ayant au moins une valeur renseignée.

    Une feature entièrement vide (ex. `aircraft_age` sans registre FAA) n'apporte
    rien et fait échouer le gradient boosting : elle est écartée du modèle.
    """
    present = X.notna().any()
    return ([c for c in CATEGORICAL_FEATURES if present[c]],
            [c for c in NUMERIC_FEATURES if present[c]])


def _tree_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Encodage ordinal des catégories : les arbres de gradient boosting les
    traitent nativement comme catégorielles (pas d'explosion one-hot)."""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=np.nan,
        encoded_missing_value=np.nan, min_frequency=MIN_CATEGORY_FREQUENCY,
    )
    return ColumnTransformer(
        [("categorical", encoder, categorical), ("numeric", "passthrough", numeric)],
        verbose_feature_names_out=False,
    )


def make_hgb_classifier(
    random_state: int = 42, categorical: list[str] = CATEGORICAL_FEATURES,
    numeric: list[str] = NUMERIC_FEATURES, **params,
) -> Pipeline:
    return Pipeline([
        ("preprocess", _tree_preprocessor(categorical, numeric)),
        ("estimator", HistGradientBoostingClassifier(
            categorical_features=[True] * len(categorical) + [False] * len(numeric),
            random_state=random_state, **params)),
    ])


def make_hgb_regressor(
    random_state: int = 42, categorical: list[str] = CATEGORICAL_FEATURES,
    numeric: list[str] = NUMERIC_FEATURES, **params,
) -> Pipeline:
    # Perte de Poisson : cible positive, très asymétrique (beaucoup de 0, longue traîne)
    return Pipeline([
        ("preprocess", _tree_preprocessor(categorical, numeric)),
        ("estimator", HistGradientBoostingRegressor(
            loss="poisson", categorical_features=[True] * len(categorical) + [False] * len(numeric),
            random_state=random_state, **params)),
    ])


def make_logistic_classifier(
    categorical: list[str] = CATEGORICAL_FEATURES, numeric: list[str] = NUMERIC_FEATURES,
) -> Pipeline:
    """Modèle linéaire de référence, comparé au gradient boosting lors de l'entraînement."""
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY)),
        ]), categorical),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    return Pipeline([("preprocess", preprocess), ("estimator", LogisticRegression(max_iter=2000))])


@dataclass
class DelayModel:
    """Classifieur (probabilité de retard >= 15 min) + régresseur (minutes de
    retard attendues), entraînés sur les mêmes features.

    `threshold` est le seuil de décision choisi à l'entraînement ;
    `max_delay_minutes` plafonne les minutes prédites (99e centile des retards
    d'entraînement) : sur des combinaisons rares (neige + vent + soirée), le
    gradient boosting peut extrapoler bien au-delà de tout retard plausible ;
    `importances` l'importance par permutation de chaque feature (exigence
    d'interprétabilité, README section 10.8).
    """

    model_version: str
    classifier: Pipeline
    regressor: Pipeline
    threshold: float = 0.5
    max_delay_minutes: float = float("inf")
    importances: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def predict(self, raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prend des lignes brutes (`RAW_INPUT_COLUMNS`), renvoie (probabilité, minutes)."""
        X = build_features(raw)
        proba = self.classifier.predict_proba(X)
        classes = list(self.classifier.classes_)
        # Un jeu d'entraînement sans aucun retard n'a qu'une classe : pas de colonne 1
        delay_probability = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        return delay_probability, self.predict_minutes(X)

    def predict_minutes(self, X: pd.DataFrame) -> np.ndarray:
        """Minutes de retard attendues, bornées à [0, max_delay_minutes] (X : features)."""
        return np.clip(self.regressor.predict(X), 0, self.max_delay_minutes)

    def is_delayed(self, delay_probability: np.ndarray) -> np.ndarray:
        return np.asarray(delay_probability) >= self.threshold

    def feature_importances(self, top_n: int = 10) -> dict[str, float]:
        ranked = sorted(self.importances.items(), key=lambda item: item[1], reverse=True)
        return {name: float(score) for name, score in ranked[:top_n]}

    def save(self, models_dir: Path = MODELS_DIR) -> Path:
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f"delay_model_{self.model_version}.pkl"
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> DelayModel:
        model = joblib.load(path)
        # vars() et non hasattr() : les valeurs par défaut de la classe masqueraient
        # un champ absent d'un modèle sérialisé avant l'ajout de ce champ.
        if not isinstance(model, cls) or {f.name for f in fields(cls)} - set(vars(model)):
            raise TypeError(f"Fichier modèle invalide ou d'un format antérieur : {path}")
        return model


def latest_model_path(models_dir: Path = MODELS_DIR) -> Path | None:
    candidates = sorted(models_dir.glob("delay_model_*.pkl"))
    return candidates[-1] if candidates else None
