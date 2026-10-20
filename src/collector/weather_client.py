"""Client de collecte des données météo via l'API Open-Meteo (sans clé requise)."""
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

WEATHER_API_URL = os.environ.get("WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast")
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

HOURLY_VARIABLES = ["temperature_2m", "wind_speed_10m", "precipitation", "visibility"]


def fetch_weather_for_airport(
    iata_code: str, *, timeout: float = 15.0, max_retries: int = 3
) -> dict[str, Any] | None:
    """Récupère les prévisions météo horaires pour un aéroport pilote.

    Comme pour le collecteur de vols, aucune exception n'est propagée en cas
    d'échec réseau/quota : on logue et on retourne None (README section 10.9).
    """
    airport = get_airport(iata_code)
    params = {
        "latitude": airport["latitude"],
        "longitude": airport["longitude"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(WEATHER_API_URL, params=params, timeout=timeout)
            if response.status_code == 429:
                logger.warning(
                    "Open-Meteo: quota dépassé (tentative %s/%s) pour %s",
                    attempt, max_retries, iata_code,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning(
                "Open-Meteo: erreur réseau/API pour %s (tentative %s/%s) : %s",
                iata_code, attempt, max_retries, exc,
            )
            time.sleep(backoff)
            backoff *= 2

    logger.error(
        "Open-Meteo: échec de la collecte pour %s après %s tentatives", iata_code, max_retries
    )
    return None


def save_raw(
    iata_code: str, payload: dict[str, Any] | None, *, output_dir: Path = RAW_DATA_DIR
) -> Path | None:
    if not payload:
        logger.info("Aucune donnée météo à sauvegarder pour %s", iata_code)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"weather_{iata_code}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"airport": iata_code, "collected_at": timestamp, "forecast": payload},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Météo sauvegardée pour %s -> %s", iata_code, output_path)
    return output_path


def collect_all_pilot_airports() -> dict[str, Path | None]:
    results: dict[str, Path | None] = {}
    for code in pilot_airport_codes():
        payload = fetch_weather_for_airport(code)
        results[code] = save_raw(code, payload)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all_pilot_airports()
