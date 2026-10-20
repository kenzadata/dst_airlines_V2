"""Référentiel statique des aéroports pilotes (IATA/ICAO/coordonnées).

Sert de source de vérité légère à la place d'un scraping en direct : les
codes IATA/ICAO et coordonnées d'un aéroport ne changent quasiment jamais,
un fichier JSON versionné est donc suffisant et plus fiable qu'un appel
réseau supplémentaire à chaque collecte.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

_REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "reference", "airports.json"
)


@lru_cache(maxsize=1)
def load_airports() -> dict[str, dict[str, Any]]:
    with open(_REFERENCE_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_airport(iata_code: str) -> dict[str, Any]:
    airports = load_airports()
    try:
        return airports[iata_code.upper()]
    except KeyError as exc:
        raise ValueError(f"Aéroport pilote inconnu : {iata_code!r}") from exc


def pilot_airport_codes() -> list[str]:
    raw = os.environ.get("PILOT_AIRPORTS", "ATL,JFK")
    return [code.strip().upper() for code in raw.split(",") if code.strip()]
