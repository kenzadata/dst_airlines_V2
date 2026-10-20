"""Schémas Pydantic de l'API, alignés sur les features du modèle (`src/ml/model.py`)."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Vol programmé + météo prévue à l'aéroport de départ.

    Les champs optionnels inconnus (avion hors registre FAA, météo indisponible)
    peuvent être omis : le modèle gère les valeurs manquantes.
    """

    flight_id: str
    fl_date: date = Field(..., description="Date du vol (heure locale de l'aéroport de départ)")
    crs_dep_time: int = Field(..., ge=0, le=2400, description="Heure de départ programmée HHMM, ex. 1745")
    origin: str = Field(..., min_length=3, max_length=3, description="Code IATA de l'aéroport pilote, ex. ATL")
    dest: str = Field(..., min_length=3, max_length=3, description="Code IATA de destination, ex. LAX")
    op_carrier: str = Field(..., min_length=2, max_length=3, description="Code IATA de la compagnie, ex. DL")
    distance: float | None = Field(None, ge=0, description="Distance (miles)")
    crs_elapsed_time: float | None = Field(None, ge=0, description="Durée de vol programmée (minutes)")
    aircraft_age: float | None = Field(None, ge=0, description="Âge de l'avion (années)")
    temp: float | None = Field(None, description="Température (°C)")
    prcp: float | None = Field(None, ge=0, description="Précipitations sur l'heure (mm)")
    wspd: float | None = Field(None, ge=0, description="Vent moyen (km/h)")
    pres: float | None = Field(None, description="Pression au niveau de la mer (hPa)")
    coco: int | None = Field(None, ge=1, le=27, description="Code condition météo Meteostat")


class PredictionResponse(BaseModel):
    flight_id: str
    delay_probability: float = Field(..., description="Probabilité d'un retard au départ >= 15 min")
    predicted_delayed: bool = Field(..., description="Probabilité >= seuil de décision du modèle")
    decision_threshold: float
    predicted_delay_minutes: float
    model_version: str
    feature_importances: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
