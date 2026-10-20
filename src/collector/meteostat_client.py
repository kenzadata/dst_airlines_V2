"""Client de collecte de la météo horaire historique via Meteostat (NOAA ISD/GHCN).

Sert de source d'entraînement : la météo est récupérée sur la même période
que les vols BTS, pour les aéroports pilotes. Les timestamps Meteostat sont
en UTC. La normalisation vers le schéma DIM_WEATHER est faite par
`src/etl/dimensions.py`.

API Meteostat 1.x (`pip install "meteostat<2.0"`).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from meteostat import Hourly, Point, Stations

from src.collector.airports_reference import get_airport, pilot_airport_codes

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
STATION_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "reference" / "station_map.parquet"

DEFAULT_START = pd.Timestamp("2023-01-01 00:00:00")
DEFAULT_END = pd.Timestamp("2025-12-31 23:59:59")

# Au-delà de cette distance (m), la station n'est plus représentative de
# la météo au départ : on la signale pour vérification manuelle.
MAX_STATION_DISTANCE_M = 15_000
MIN_STATION_COVERAGE = 0.90


def _airport_point(iata_code: str) -> Point:
    airport = get_airport(iata_code)
    return Point(airport["latitude"], airport["longitude"], airport.get("elevation_m"))


def fetch_hourly_history(
    iata_code: str, start: pd.Timestamp = DEFAULT_START, end: pd.Timestamp = DEFAULT_END
) -> pd.DataFrame:
    """Récupère l'historique horaire de la station la plus proche de l'aéroport.

    Retourne un DataFrame vide (et logue) en cas d'échec plutôt que de crasher
    (README section 10.9).
    """
    try:
        df = Hourly(_airport_point(iata_code), start, end).fetch()  # index = 'time' en UTC
    except (OSError, ValueError) as exc:  # erreurs réseau (URLError) ou fichier distant illisible
        logger.error("Meteostat: échec de la collecte pour %s : %s", iata_code, exc)
        return pd.DataFrame()

    if df.empty:
        logger.warning("Meteostat: aucune donnée pour %s entre %s et %s", iata_code, start, end)
        return df
    df = df.reset_index()
    logger.info(
        "Meteostat: %s lignes pour %s (%s -> %s)",
        len(df), iata_code, df["time"].min(), df["time"].max(),
    )
    return df


def save_raw(iata_code: str, df: pd.DataFrame, *, output_dir: Path = RAW_DATA_DIR) -> Path | None:
    if df.empty:
        logger.info("Aucune donnée météo historique à sauvegarder pour %s", iata_code)
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"weather_history_{iata_code}.parquet"
    df.to_parquet(output_path, index=False)
    logger.info("Météo historique sauvegardée pour %s -> %s", iata_code, output_path)
    return output_path


def resolve_station(
    iata_code: str, start: pd.Timestamp = DEFAULT_START, end: pd.Timestamp = DEFAULT_END
) -> dict:
    """Retient la station Meteostat la plus proche couvrant réellement la période.

    Parcourt les 5 stations les plus proches et garde la première dont la
    couverture horaire est >= 90 % et la distance <= 15 km. À défaut, renvoie
    la plus proche avec le statut "A_VERIFIER" pour inspection manuelle.
    """
    airport = get_airport(iata_code)
    found = Stations().nearby(airport["latitude"], airport["longitude"]).fetch(5)
    if found.empty:
        return {"IATA": iata_code, "station_id": None, "station_name": None,
                "distance_m": None, "coverage": None, "status": "AUCUNE_STATION"}

    for station_id, row in found.iterrows():
        coverage = Hourly(station_id, start, end).coverage()  # part de non-NaN, 0..1
        distance = float(row["distance"])
        if coverage >= MIN_STATION_COVERAGE and distance <= MAX_STATION_DISTANCE_M:
            return {"IATA": iata_code, "station_id": station_id, "station_name": row["name"],
                    "distance_m": round(distance, 1), "coverage": round(coverage, 3), "status": "OK"}

    best_id = found.index[0]
    best = found.iloc[0]
    return {"IATA": iata_code, "station_id": best_id, "station_name": best["name"],
            "distance_m": round(float(best["distance"]), 1),
            "coverage": round(Hourly(best_id, start, end).coverage(), 3), "status": "A_VERIFIER"}


def resolve_all_stations(output_path: Path = STATION_MAP_PATH) -> pd.DataFrame:
    """Construit le mapping IATA -> station Meteostat pour les aéroports pilotes."""
    mapping = pd.DataFrame.from_records([resolve_station(code) for code in pilot_airport_codes()])
    problems = mapping[mapping["status"] != "OK"]
    if not problems.empty:
        logger.warning("%s aéroport(s) à vérifier manuellement : %s",
                       len(problems), problems["IATA"].tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)
    logger.info("Mapping des stations écrit -> %s", output_path)
    return mapping


def collect_all_pilot_airports(
    start: pd.Timestamp = DEFAULT_START, end: pd.Timestamp = DEFAULT_END
) -> dict[str, Path | None]:
    return {code: save_raw(code, fetch_hourly_history(code, start, end)) for code in pilot_airport_codes()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Collecte la météo horaire historique (Meteostat)")
    parser.add_argument("--start", default=str(DEFAULT_START.date()))
    parser.add_argument("--end", default=str(DEFAULT_END.date()))
    parser.add_argument("--resolve-stations", action="store_true",
                        help="Résout et écrit aussi data/reference/station_map.parquet")
    args = parser.parse_args()

    if args.resolve_stations:
        resolve_all_stations()
    collect_all_pilot_airports(pd.Timestamp(args.start), pd.Timestamp(args.end) + pd.Timedelta(hours=23, minutes=59))
