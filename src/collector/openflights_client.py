"""Client de collecte du référentiel aéroports OpenFlights (`airports.dat`).

Ne fait que télécharger le fichier brut ; la construction de DIM_AIRPORT
(filtrage sur les aéroports pilotes, sélection des colonnes) est faite par
`src/etl/dimensions.py` (README section 10, point 3).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OPENFLIGHTS_AIRPORTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
AIRPORTS_DAT_PATH = RAW_DATA_DIR / "airports.dat"


def fetch_airports_dat(
    *, output_path: Path = AIRPORTS_DAT_PATH, timeout: float = 30.0, max_retries: int = 3
) -> Path | None:
    """Télécharge `airports.dat` et l'écrit tel quel sur disque.

    Comme les autres collecteurs, aucune exception réseau n'est propagée :
    en cas d'échec persistant on logue et on retourne None (README section 10.9).
    """
    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(OPENFLIGHTS_AIRPORTS_URL, timeout=timeout)
            response.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(response.text, encoding="utf-8")
            logger.info("airports.dat sauvegardé -> %s", output_path)
            return output_path
        except requests.RequestException as exc:
            logger.warning(
                "OpenFlights: erreur réseau (tentative %s/%s) : %s", attempt, max_retries, exc
            )
            time.sleep(backoff)
            backoff *= 2

    logger.error("OpenFlights: échec du téléchargement après %s tentatives", max_retries)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_airports_dat()
