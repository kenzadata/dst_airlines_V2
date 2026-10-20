"""Client de collecte des données de vols via l'API OpenSky Network.

Ce module ne fait que collecter : pas de nettoyage ni de jointure ici,
c'est le rôle de `src/etl/pipeline.py` (README section 10, point 3).
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

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Ordre des champs tel que documenté par l'API OpenSky pour /states/all
STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]


def _auth() -> tuple[str, str] | None:
    username = os.environ.get("OPENSKY_USERNAME")
    password = os.environ.get("OPENSKY_PASSWORD")
    if username and password:
        return (username, password)
    return None


def fetch_states_for_airport(
    iata_code: str, *, timeout: float = 15.0, max_retries: int = 3
) -> list[dict[str, Any]]:
    """Récupère les états de vols (positions courantes) dans la bbox d'un aéroport pilote.

    Ne lève jamais d'exception réseau/API vers l'appelant : en cas d'échec
    persistant (quota dépassé, timeout, 5xx), retourne une liste vide et logue
    un warning/error plutôt que de crasher (contrainte README section 10.9).
    """
    airport = get_airport(iata_code)
    bbox = airport["bbox"]
    params = {
        "lamin": bbox["lamin"],
        "lamax": bbox["lamax"],
        "lomin": bbox["lomin"],
        "lomax": bbox["lomax"],
    }

    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                OPENSKY_STATES_URL, params=params, auth=_auth(), timeout=timeout
            )
            if response.status_code == 429:
                logger.warning(
                    "OpenSky: quota dépassé (tentative %s/%s) pour %s",
                    attempt, max_retries, iata_code,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            payload = response.json()
            states = payload.get("states") or []
            return [dict(zip(STATE_FIELDS, state)) for state in states]
        except requests.RequestException as exc:
            logger.warning(
                "OpenSky: erreur réseau/API pour %s (tentative %s/%s) : %s",
                iata_code, attempt, max_retries, exc,
            )
            time.sleep(backoff)
            backoff *= 2

    logger.error(
        "OpenSky: échec de la collecte pour %s après %s tentatives", iata_code, max_retries
    )
    return []


def save_raw(
    iata_code: str, states: list[dict[str, Any]], *, output_dir: Path = RAW_DATA_DIR
) -> Path | None:
    """Écrit les états bruts collectés dans un fichier JSON horodaté."""
    if not states:
        logger.info("Aucune donnée à sauvegarder pour %s", iata_code)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"flights_{iata_code}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"airport": iata_code, "collected_at": timestamp, "states": states},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("%s états sauvegardés pour %s -> %s", len(states), iata_code, output_path)
    return output_path


def collect_all_pilot_airports() -> dict[str, Path | None]:
    """Collecte et sauvegarde les états courants pour tous les aéroports pilotes."""
    results: dict[str, Path | None] = {}
    for code in pilot_airport_codes():
        states = fetch_states_for_airport(code)
        results[code] = save_raw(code, states)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collect_all_pilot_airports()
