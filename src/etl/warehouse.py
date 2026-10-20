"""Chargement de l'entrepôt historique (schéma en étoile `sql/schema.sql`) dans PostgreSQL.

Lit les fichiers produits par `src/etl/bts.py` et `src/etl/dimensions.py` et
les insère par lots (`execute_values`). Idempotent : `ON CONFLICT DO NOTHING`
sur la clé primaire de chaque table, on peut donc relancer sans doublons.

La connexion vient de `WAREHOUSE_DATABASE_URL`, sinon de `DATABASE_URL` s'il
pointe vers PostgreSQL (le schéma utilise TIMESTAMPTZ : SQLite n'est pas
supporté ici). Une base Neon fonctionne en y mettant son URL.

Usage :
    python -m src.etl.warehouse [schema|airports|airlines|aircraft|weather|flights|all]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from src.etl.bts import PROCESSED_BTS_DIR
from src.etl.dimensions import (
    DIM_AIRCRAFT_PATH,
    DIM_AIRLINE_PATH,
    DIM_AIRPORT_PATH,
    DIM_WEATHER_PATH,
    build_fl_time_utc,
    floor_to_hour,
)

load_dotenv()

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "schema.sql"
CONNECT_TIMEOUT = 15
PAGE_SIZE = 5_000

FACT_COLUMNS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID", "ORIGIN", "DEST_AIRPORT_ID", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "ARR_TIME", "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "FLIGHTS", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
    "FL_TIME_UTC", "FL_TIME_UTC_HOUR",
]


def get_dsn() -> str:
    """URL PostgreSQL au format libpq (sans suffixe de driver SQLAlchemy)."""
    for name in ("WAREHOUSE_DATABASE_URL", "DATABASE_URL"):
        url = os.environ.get(name, "")
        if url.startswith(("postgresql", "postgres://")):
            return url.replace("postgresql+psycopg2://", "postgresql://", 1)
    raise RuntimeError(
        "Aucune URL PostgreSQL : définir WAREHOUSE_DATABASE_URL "
        "(ex. postgresql://dst:dst@localhost:5432/dst_airlines)"
    )


def connect(dsn: str | None = None):
    return psycopg2.connect(dsn or get_dsn(), connect_timeout=CONNECT_TIMEOUT)


def apply_schema(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    logger.info("Schéma appliqué (%s)", SCHEMA_PATH.name)


def to_db_rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """DataFrame -> tuples pour psycopg2, avec NaN/NA/NaT -> None (NULL SQL).

    `astype(object)` avant `where` : sinon un dtype nullable réinjecte sa propre
    valeur manquante au lieu du None Python.
    """
    df = df[columns].astype(object)
    return list(df.where(pd.notna(df), None).itertuples(index=False, name=None))


def insert_rows(conn, table: str, db_columns: list[str], rows: list[tuple]) -> None:
    sql = f"INSERT INTO {table} ({', '.join(db_columns)}) VALUES %s ON CONFLICT DO NOTHING"
    with conn, conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=PAGE_SIZE)
    logger.info("%s : %s lignes soumises", table, len(rows))


def _load_table(conn, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    insert_rows(conn, table, [c.lower() for c in columns], to_db_rows(df, columns))


def load_dim_airport(conn, path: Path = DIM_AIRPORT_PATH) -> None:
    df = pd.read_csv(path).rename(columns={"IATA_CODE": "iata_code", "TIMEZONE": "timezone"})
    _load_table(conn, "dim_airport", df, ["iata_code", "name", "city", "latitude", "longitude", "timezone"])


def load_dim_airline(conn, path: Path = DIM_AIRLINE_PATH) -> None:
    _load_table(conn, "dim_airline", pd.read_csv(path), ["iata_code", "airline_name"])


def load_dim_aircraft(conn, path: Path = DIM_AIRCRAFT_PATH) -> None:
    df = pd.read_parquet(path)
    for col in ("YEAR_MFR", "AIRCRAFT_AGE"):
        df[col] = df[col].astype("Int64")  # SMALLINT nullable
    _load_table(conn, "dim_aircraft", df, ["TAIL_NUM", "MANUFACTURER", "MODEL", "YEAR_MFR", "AIRCRAFT_AGE"])


def load_dim_weather(conn, path: Path = DIM_WEATHER_PATH) -> None:
    df = pd.read_parquet(path)
    df["coco"] = df["coco"].astype("Int64")  # SMALLINT nullable
    _load_table(conn, "dim_weather", df,
                ["AIRPORT_IATA_CODE", "WEATHER_TIME_UTC", "temp", "prcp", "wspd", "pres", "coco"])


def load_tz_map(conn) -> dict[str, str]:
    """IATA -> fuseau IANA, depuis dim_airport (à charger avant les vols)."""
    with conn.cursor() as cur:
        cur.execute("SELECT iata_code, timezone FROM dim_airport WHERE timezone IS NOT NULL")
        return dict(cur.fetchall())


def prepare_fact_flights(df: pd.DataFrame, tz_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["FL_TIME_UTC"] = build_fl_time_utc(df, tz_map)
    df["FL_TIME_UTC_HOUR"] = floor_to_hour(df["FL_TIME_UTC"])
    n_missing = df["FL_TIME_UTC"].isna().sum()
    if n_missing:
        logger.warning("%s vols sans FL_TIME_UTC (fuseau manquant ou CRS_DEP_TIME nul)", n_missing)
    return df


def load_fact_flights(conn, flights_dir: Path = PROCESSED_BTS_DIR) -> None:
    tz_map = load_tz_map(conn)
    if not tz_map:
        raise RuntimeError("dim_airport est vide : charger les aéroports avant les vols")
    # Un Parquet par mois : on charge mois par mois pour borner la mémoire
    for path in sorted(flights_dir.glob("*.parquet")):
        logger.info("Chargement de %s", path.name)
        _load_table(conn, "fact_flights", prepare_fact_flights(pd.read_parquet(path), tz_map), FACT_COLUMNS)


LOADERS = {
    "airports": load_dim_airport,
    "airlines": load_dim_airline,
    "aircraft": load_dim_aircraft,
    "weather": load_dim_weather,
    "flights": load_fact_flights,
}


def run(target: str = "all", dsn: str | None = None) -> None:
    conn = connect(dsn)
    try:
        if target in ("schema", "all"):
            apply_schema(conn)
        for name, loader in LOADERS.items():
            if target in (name, "all"):
                loader(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Charge l'entrepôt historique dans PostgreSQL")
    parser.add_argument("target", nargs="?", default="all", choices=["schema", *LOADERS, "all"])
    run(parser.parse_args().target)
