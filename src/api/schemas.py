"""Schémas Pydantic de l'API, alignés sur le modèle de données (README section 5)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    flight_id: str
    origin_airport: str = Field(..., description="Code IATA de l'aéroport pilote, ex. ATL")
    temperature: float = Field(..., description="°C")
    wind_speed: float = Field(..., description="km/h")
    precipitation: float = Field(..., description="mm")
    visibility: float = Field(..., description="m")
    hour: int = Field(..., ge=0, le=23, description="Heure prévue de départ (0-23)")


class PredictionResponse(BaseModel):
    flight_id: str
    delay_probability: float
    predicted_delay_minutes: float
    model_version: str
    feature_importances: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
