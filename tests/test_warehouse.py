from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.etl import warehouse


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"WAREHOUSE_DATABASE_URL": "postgresql://neon/db", "DATABASE_URL": "postgresql://local/db"}, "postgresql://neon/db"),
        ({"DATABASE_URL": "postgresql+psycopg2://dst:dst@db:5432/dst_airlines"}, "postgresql://dst:dst@db:5432/dst_airlines"),
    ],
)
def test_get_dsn_prefers_warehouse_url_and_strips_driver(monkeypatch, env, expected):
    monkeypatch.delenv("WAREHOUSE_DATABASE_URL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert warehouse.get_dsn() == expected


def test_get_dsn_rejects_sqlite(monkeypatch):
    monkeypatch.delenv("WAREHOUSE_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/processed/dst_airlines.db")

    with pytest.raises(RuntimeError):
        warehouse.get_dsn()


def test_to_db_rows_converts_missing_values_to_none():
    df = pd.DataFrame(
        {
            "a": pd.array([1, None], dtype="Int64"),
            "b": [1.5, float("nan")],
            "c": pd.to_datetime(["2024-01-01", None]),
            "unused": ["x", "y"],
        }
    )

    rows = warehouse.to_db_rows(df, ["a", "b", "c"])

    assert rows[0][:2] == (1, 1.5)
    assert rows[1] == (None, None, None)


def test_prepare_fact_flights_adds_utc_time_keys():
    flights = pd.DataFrame(
        {"FL_DATE": pd.to_datetime(["2024-01-15"]), "CRS_DEP_TIME": [845], "ORIGIN": ["ATL"]}
    )

    df = warehouse.prepare_fact_flights(flights, {"ATL": "America/New_York"})

    assert df["FL_TIME_UTC"].iloc[0] == pd.Timestamp("2024-01-15 13:45", tz="UTC")
    assert df["FL_TIME_UTC_HOUR"].iloc[0] == pd.Timestamp("2024-01-15 13:00", tz="UTC")


def test_insert_rows_builds_idempotent_insert(monkeypatch):
    captured = {}
    monkeypatch.setattr(warehouse, "execute_values", lambda cur, sql, rows, page_size: captured.update(sql=sql, rows=rows))
    conn = MagicMock()

    warehouse.insert_rows(conn, "dim_airline", ["iata_code", "airline_name"], [("DL", "Delta")])

    assert captured["sql"] == "INSERT INTO dim_airline (iata_code, airline_name) VALUES %s ON CONFLICT DO NOTHING"
    assert captured["rows"] == [("DL", "Delta")]


def test_load_fact_flights_requires_airports_first(tmp_path, monkeypatch):
    monkeypatch.setattr(warehouse, "load_tz_map", lambda conn: {})

    with pytest.raises(RuntimeError):
        warehouse.load_fact_flights(MagicMock(), tmp_path)


def test_schema_file_is_replayable_without_data_loss():
    sql = warehouse.SCHEMA_PATH.read_text(encoding="utf-8").upper()

    assert "DELETE FROM" not in sql
    assert "DROP TABLE" not in sql
    assert sql.count("CREATE TABLE IF NOT EXISTS") == 5


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://dst:dst@db:5432/dst_airlines", "postgresql+psycopg2://dst:dst@db:5432/dst_airlines"),
        ("postgres://u:p@host/db", "postgresql+psycopg2://u:p@host/db"),
        ("postgresql+psycopg2://u:p@host/db", "postgresql+psycopg2://u:p@host/db"),
        ("sqlite:///./data/processed/dst_airlines.db", "sqlite:///./data/processed/dst_airlines.db"),
    ],
)
def test_sqlalchemy_url_pins_psycopg2_driver(url, expected):
    from src.etl.db import sqlalchemy_url

    assert sqlalchemy_url(url) == expected
    assert warehouse.get_engine("postgresql://u:p@localhost/db").dialect.driver == "psycopg2"
