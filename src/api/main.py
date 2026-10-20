"""API FastAPI exposant la prédiction de retard (README section 6, étape 4).

Ce module ne fait qu'exposer le modèle déjà entraîné : pas de logique
d'entraînement ni de transformation ETL ici.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.exc import OperationalError

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.etl.db import Prediction, get_engine, get_session_factory, init_db
from src.ml.model import DelayModel, latest_model_path

logger = logging.getLogger(__name__)

_model: DelayModel | None = None


def get_model() -> DelayModel | None:
    global _model
    if _model is None:
        path = latest_model_path()
        if path is not None:
            _model = DelayModel.load(path)
    return _model


def _init_db_with_retry(*, attempts: int = 5, delay: float = 2.0) -> None:
    """Réessaie l'initialisation de la base au démarrage : au sein de
    docker-compose, `depends_on: service_healthy` garantit que Postgres
    accepte des connexions, mais pas que le DNS interne du conteneur `db`
    soit déjà propagé au tout premier instant du démarrage de l'API."""
    for attempt in range(1, attempts + 1):
        try:
            init_db(get_engine())
            return
        except OperationalError:
            if attempt == attempts:
                raise
            logger.warning(
                "Connexion à la base indisponible au démarrage (tentative %s/%s), nouvel essai dans %ss",
                attempt, attempts, delay,
            )
            time.sleep(delay)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _init_db_with_retry()
    get_model()
    yield


app = FastAPI(title="DST Airlines — API de prédiction de retard", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    return HealthResponse(status="ok", model_version=model.model_version if model else None)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle entraîné disponible. Lancer `python -m src.ml.train`.",
        )

    features = pd.DataFrame([request.model_dump(exclude={"flight_id"})])
    delay_probability, predicted_delay_minutes = model.predict(features)

    response = PredictionResponse(
        flight_id=request.flight_id,
        delay_probability=float(delay_probability[0]),
        predicted_delay_minutes=float(predicted_delay_minutes[0]),
        model_version=model.model_version,
        feature_importances=model.feature_importances(),
    )

    _persist_prediction(response)
    return response


def _persist_prediction(response: PredictionResponse) -> None:
    """Sauvegarde la prédiction en base ; un échec ici ne doit jamais faire
    échouer la réponse API (README section 10.9 : pas de crash silencieux,
    mais pas de crash court non plus sur une écriture secondaire)."""
    try:
        session_factory = get_session_factory(get_engine())
        with session_factory() as session:
            session.add(
                Prediction(
                    id=str(uuid4()),
                    flight_id=response.flight_id,
                    predicted_at=datetime.now(timezone.utc),
                    delay_probability=response.delay_probability,
                    predicted_delay_minutes=response.predicted_delay_minutes,
                    model_version=response.model_version,
                )
            )
            session.commit()
    except Exception:
        logger.exception("Échec de la sauvegarde de la prédiction pour %s", response.flight_id)
