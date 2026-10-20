"""Client de collecte de la météo temps réel via OpenWeather (Current Weather 2.5).

Sert à l'inférence : la normalisation vers le schéma Meteostat utilisé à
l'entraînement (mêmes colonnes, unités et codes `coco`) est faite par
`src/etl/dimensions.py`. Nécessite `OPENWEATHER_API_KEY` dans `.env`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from src.collector.airports_reference import get_airport, pilot_airport_codes

load_dotenv()

logger = logging.getLogger(__name__)

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def fetch_current_weather(
    iata_code: str, *, timeout: float = 15.0, max_retries: int = 3
) -> dict[str, Any] | None:
    """Récupère la météo courante d'un aéroport pilote.

    Aucune exception n'est propagée en cas d'échec réseau/quota ou de clé
    absente : on logue et on retourne None (README section 10.9).
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error("OpenWeather: OPENWEATHER_API_KEY absente de l'environnement")
        return None

    airport = get_airport(iata_code)
    params = {
        "lat": airport["latitude"],
        "lon": airport["longitude"],
        "units": "metric",  # temp en °C, vent en m/s
        "appid": api_key,
    }

    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPENWEATHER_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                logger.warning(
                    "OpenWeather: quota dépassé (tentative %s/%s) pour %s",
                    attempt, max_retries, iata_code,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            # Pas de `exc` brut dans le log : l'URL qu'il contient porte la clé API.
            logger.warning(
                "OpenWeather: erreur réseau/API pour %s (tentative %s/%s) : %s",
                iata_code, attempt, max_retries, type(exc).__name__,
            )
            time.sleep(backoff)
            backoff *= 2

    logger.error(
        "OpenWeather: échec de la collecte pour %s après %s tentatives", iata_code, max_retries
    )
    return None


def save_raw(
    iata_code: str, payload: dict[str, Any] | None, *, output_dir: Path = RAW_DATA_DIR
) -> Path | None:
    if not payload:
        logger.info("Aucune donnée météo temps réel à sauvegarder pour %s", iata_code)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"openweather_{iata_code}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"airport": iata_code, "collected_at": timestamp, "current": payload},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Météo temps réel sauvegardée pour %s -> %s", iata_code, output_path)
    return output_path


def collect_all_pilot_airports() -> dict[str, Path | None]:
    return {code: save_raw(code, fetch_current_weather(code)) for code in pilot_airport_codes()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all_pilot_airports()
