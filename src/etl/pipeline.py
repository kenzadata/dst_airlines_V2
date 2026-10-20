"""Pipeline ETL : nettoyage des données brutes, jointure vols+météo,
construction des features (README section 10, point 3 : "l'ETL ne fait
que transformer").

Limite connue du MVP : l'API OpenSky (`/states/all`, gratuite) renvoie des
*positions courantes* d'aéronefs, pas des horaires programmés/réels par vol
(ces derniers nécessiteraient une API payante type AviationStack). On
construit donc un `flight_id` stable par aéronef+jour, et un statut dérivé
de `on_ground`/`velocity` plutôt que des colonnes `status` fournies par
l'API. `scheduled_*`/`actual_*` restent `NaT` tant qu'une source d'horaires
programmés n'est pas branchée.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.etl.db import Flight, Weather, get_engine, get_session_factory, init_db

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _load_json_files(pattern: str, raw_dir: Path) -> list[dict[str, Any]]:
    files = sorted(raw_dir.glob(pattern))
    records = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def load_raw_flight_records(raw_dir: Path = RAW_DATA_DIR) -> list[dict[str, Any]]:
    return _load_json_files("flights_*.json", raw_dir)


def load_raw_weather_records(raw_dir: Path = RAW_DATA_DIR) -> list[dict[str, Any]]:
    return _load_json_files("weather_*.json", raw_dir)


def flights_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Transforme les fichiers bruts `flights_client.save_raw` en DataFrame nettoyé."""
    rows = []
    for record in records:
        airport = record["airport"]
        for state in record.get("states", []):
            icao24 = state.get("icao24")
            last_contact = state.get("last_contact")
            if not icao24 or not last_contact:
                continue
            observed_at = datetime.fromtimestamp(last_contact, tz=timezone.utc)
            rows.append(
                {
                    "flight_id": f"{icao24}_{observed_at:%Y%m%d}",
                    "airline": (state.get("callsign") or "").strip()[:3] or None,
                    "origin_airport": airport,
                    "destination_airport": None,
                    "scheduled_departure": pd.NaT,
                    "actual_departure": pd.NaT,
                    "scheduled_arrival": pd.NaT,
                    "actual_arrival": pd.NaT,
                    "status": "au sol" if state.get("on_ground") else "en vol",
                    "observed_at": observed_at,
                }
            )
    columns = [
        "flight_id", "airline", "origin_airport", "destination_airport",
        "scheduled_departure", "actual_departure", "scheduled_arrival",
        "actual_arrival", "status", "observed_at",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["flight_id", "observed_at"]).sort_values("observed_at")


def weather_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Aplati les prévisions horaires Open-Meteo en une ligne par heure/aéroport."""
    rows = []
    for record in records:
        airport = record["airport"]
        hourly = record.get("forecast", {}).get("hourly", {})
        times = hourly.get("time", [])
        for i, time_str in enumerate(times):
            rows.append(
                {
                    "airport": airport,
                    "observed_at": pd.to_datetime(time_str, utc=True),
                    "temperature": hourly.get("temperature_2m", [None] * len(times))[i],
                    "wind_speed": hourly.get("wind_speed_10m", [None] * len(times))[i],
                    "precipitation": hourly.get("precipitation", [None] * len(times))[i],
                    "visibility": hourly.get("visibility", [None] * len(times))[i],
                }
            )
    columns = ["airport", "observed_at", "temperature", "wind_speed", "precipitation", "visibility"]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["airport", "observed_at"]).sort_values("observed_at")


def join_flights_weather(flights_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Associe à chaque observation de vol la météo la plus proche dans le temps,
    pour le même aéroport."""
    if flights_df.empty:
        return flights_df.assign(
            temperature=pd.Series(dtype=float),
            wind_speed=pd.Series(dtype=float),
            precipitation=pd.Series(dtype=float),
            visibility=pd.Series(dtype=float),
        )
    if weather_df.empty:
        return flights_df.assign(temperature=None, wind_speed=None, precipitation=None, visibility=None)

    joined_parts = []
    for airport, flights_group in flights_df.groupby("origin_airport"):
        weather_group = weather_df[weather_df["airport"] == airport]
        if weather_group.empty:
            part = flights_group.assign(temperature=None, wind_speed=None, precipitation=None, visibility=None)
        else:
            part = pd.merge_asof(
                flights_group.sort_values("observed_at"),
                weather_group.sort_values("observed_at")[
                    ["observed_at", "temperature", "wind_speed", "precipitation", "visibility"]
                ],
                on="observed_at",
                direction="nearest",
            )
        joined_parts.append(part)
    return pd.concat(joined_parts, ignore_index=True).sort_values("observed_at")


def write_processed(df: pd.DataFrame, output_dir: Path = PROCESSED_DATA_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"flights_weather_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    logger.info("%s lignes écrites dans %s", len(df), output_path)
    return output_path


def upsert_flights_and_weather(flights_df: pd.DataFrame, weather_df: pd.DataFrame, engine=None) -> None:
    engine = engine or get_engine()
    init_db(engine)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        for _, row in flights_df.iterrows():
            session.merge(
                Flight(
                    flight_id=row["flight_id"],
                    airline=row.get("airline"),
                    origin_airport=row["origin_airport"],
                    destination_airport=row.get("destination_airport"),
                    scheduled_departure=_none_if_nat(row.get("scheduled_departure")),
                    actual_departure=_none_if_nat(row.get("actual_departure")),
                    scheduled_arrival=_none_if_nat(row.get("scheduled_arrival")),
                    actual_arrival=_none_if_nat(row.get("actual_arrival")),
                    status=row.get("status"),
                )
            )
        for _, row in weather_df.iterrows():
            session.merge(
                Weather(
                    id=str(uuid4()),
                    airport=row["airport"],
                    observed_at=row["observed_at"],
                    temperature=row.get("temperature"),
                    wind_speed=row.get("wind_speed"),
                    precipitation=row.get("precipitation"),
                    visibility=row.get("visibility"),
                )
            )
        session.commit()


def _none_if_nat(value):
    return None if pd.isna(value) else value


def run_pipeline(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR, engine=None) -> pd.DataFrame:
    flights_df = flights_to_dataframe(load_raw_flight_records(raw_dir))
    weather_df = weather_to_dataframe(load_raw_weather_records(raw_dir))
    joined_df = join_flights_weather(flights_df, weather_df)
    if not joined_df.empty:
        write_processed(joined_df, processed_dir)
    upsert_flights_and_weather(flights_df, weather_df, engine)
    return joined_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
