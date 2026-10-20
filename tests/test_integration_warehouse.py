"""Test d'intégration de bout en bout sur un vrai PostgreSQL :
fichier BTS brut -> transform -> chargement de l'entrepôt -> jeu
d'entraînement ML -> requêtes du dashboard.

Ignoré sans `WAREHOUSE_TEST_DATABASE_URL`. Tout se passe dans un schéma
temporaire créé puis supprimé : on peut viser la base de développement
sans toucher à ses tables.

    WAREHOUSE_TEST_DATABASE_URL=postgresql://dst:dst@localhost:5432/dst_airlines pytest tests/test_integration_warehouse.py
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import psycopg2
import pytest

from src.dashboard import queries
from src.etl import bts, warehouse
from src.ml.dataset import load_training_data

TEST_URL = os.environ.get("WAREHOUSE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="WAREHOUSE_TEST_DATABASE_URL non défini")


@pytest.fixture
def dsn():
    schema = f"it_{uuid4().hex[:8]}"
    admin = psycopg2.connect(TEST_URL)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
    separator = "&" if "?" in TEST_URL else "?"
    yield f"{TEST_URL}{separator}options=-csearch_path%3D{schema}"
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


def _prezip_month(tmp_path) -> Path:
    """Fichier mensuel au format des ZIPs pré-zippés du BTS (noms CamelCase)."""
    flights = pd.DataFrame({
        "FlightDate": ["2024-01-15", "2024-01-15", "2024-01-15", "2024-01-16"],
        "Reporting_Airline": ["DL", "B6", "DL", "AA"],
        "IATA_CODE_Reporting_Airline": ["DL", "B6", "DL", "AA"],
        "Tail_Number": ["N123DL", "N999JB", "N123DL", "N1AA"],
        "Flight_Number_Reporting_Airline": [100, 200, 101, 300],
        "OriginAirportID": [10397, 12478, 10397, 13930],
        "Origin": ["ATL", "JFK", "ATL", "ORD"],
        "DestAirportID": [12892, 10397, 12478, 10397],
        "Dest": ["LAX", "ATL", "JFK", "ATL"],
        "CRSDepTime": [815, 1730, 1900, 600],
        "DepTime": [840, 1725, None, 600],
        "DepDelay": [25.0, -5.0, None, 0.0],
        "ArrDelay": [20.0, -10.0, None, 0.0],
        "Cancelled": [0.0, 0.0, 1.0, 0.0],
        "Diverted": [0.0, 0.0, 0.0, 0.0],
        "Distance": [1946.0, 760.0, 760.0, 606.0],
        "CRSElapsedTime": [300.0, 140.0, 130.0, 120.0],
        "Unused": ["x"] * 4,
    })
    path = tmp_path / "bts_raw_2024_01.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv",
                    flights.to_csv(index=False))
    return path


def _load_everything(conn, tmp_path):
    bts.process_file(_prezip_month(tmp_path), output_dir=tmp_path / "bts")
    pd.DataFrame({"IATA_CODE": ["ATL", "JFK"], "name": ["Atlanta", "JFK"], "city": ["Atlanta", "New York"],
                  "latitude": [33.6, 40.6], "longitude": [-84.4, -73.8],
                  "TIMEZONE": ["America/New_York"] * 2}).to_csv(tmp_path / "dim_airport.csv", index=False)
    pd.DataFrame({"iata_code": ["DL", "B6", "AA"], "airline_name": ["Delta", "JetBlue", "American"]}).to_csv(
        tmp_path / "dim_airline.csv", index=False)
    pd.DataFrame({"TAIL_NUM": ["N123DL"], "MANUFACTURER": ["BOEING"], "MODEL": ["737-932ER"],
                  "YEAR_MFR": [2010.0], "AIRCRAFT_AGE": [16.0]}).to_parquet(tmp_path / "dim_aircraft.parquet")
    pd.DataFrame({"AIRPORT_IATA_CODE": ["ATL"], "WEATHER_TIME_UTC": pd.to_datetime(["2024-01-15 13:00"], utc=True),
                  "temp": [4.0], "prcp": [1.5], "wspd": [20.0], "pres": [1012.0], "coco": [8.0]}).to_parquet(
        tmp_path / "dim_weather.parquet")

    warehouse.apply_schema(conn)
    warehouse.load_dim_airport(conn, tmp_path / "dim_airport.csv")
    warehouse.load_dim_airline(conn, tmp_path / "dim_airline.csv")
    warehouse.load_dim_aircraft(conn, tmp_path / "dim_aircraft.parquet")
    warehouse.load_dim_weather(conn, tmp_path / "dim_weather.parquet")
    warehouse.load_fact_flights(conn, tmp_path / "bts")


def test_warehouse_load_is_idempotent_and_feeds_ml_and_dashboard(dsn, tmp_path):
    conn = warehouse.connect(dsn)
    try:
        _load_everything(conn, tmp_path)
        _load_everything(conn, tmp_path)  # 2e passe : aucun doublon
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_flights")
            assert cur.fetchone()[0] == 3  # ORD n'est pas un aéroport pilote
    finally:
        conn.close()

    engine = warehouse.get_engine(dsn)

    training = load_training_data(engine)
    assert len(training) == 2  # le vol annulé est exclu
    atl = training.set_index("origin").loc["ATL"]
    assert (atl["dep_delay"], atl["aircraft_age"], atl["prcp"], atl["coco"]) == (25.0, 16, 1.5, 8)
    assert atl["fl_time_utc"] == pd.Timestamp("2024-01-15 13:15", tz="UTC")

    kpis = queries.delay_kpis(engine, origin="ATL")
    assert (kpis["flights"], kpis["delay_rate"], kpis["cancel_rate"]) == (1, 1.0, 0.5)
    assert set(queries.delay_by(engine, "carrier")["key"]) == {"DL", "B6"}
    for grouping in queries.GROUPINGS:
        assert not queries.delay_by(engine, grouping).empty
    assert queries.carriers(engine)["airline_name"].tolist() == ["Delta", "JetBlue"]
    aircraft = queries.aircraft_summary(engine, min_flights=1)
    assert aircraft.set_index("tail_num").loc["N123DL", "model"] == "737-932ER"
    assert len(queries.aircraft_flights(engine, "N123DL")) == 2
