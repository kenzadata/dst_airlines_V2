from __future__ import annotations

import zipfile

import pandas as pd

from src.etl import bts


def _raw_bts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FL_DATE": ["1/15/2024 12:00:00 AM", "1/15/2024 12:00:00 AM", "1/16/2024 12:00:00 AM", "1/15/2024 12:00:00 AM"],
            "OP_UNIQUE_CARRIER": ["DL", "DL", "B6", "AA"],
            "OP_CARRIER": [" dl", "DL", "B6", "AA"],
            "TAIL_NUM": [" n123dl", "N123DL", None, "N1AA"],
            "OP_CARRIER_FL_NUM": [100, 100, 200, 300],
            "ORIGIN_AIRPORT_ID": [10397, 10397, 12478, 13930],
            "ORIGIN": ["ATL", "ATL", "JFK", "ORD"],
            "DEST_AIRPORT_ID": [12478, 12478, 10397, 10397],
            "DEST": ["JFK", "JFK", "ATL", "ATL"],
            "CRS_DEP_TIME": [815, 815, 2359, 600],
            "DEP_TIME": [830, 830, None, 600],
            "CANCELLED": [0.0, 0.0, 1.0, 0.0],
            "DIVERTED": [0.0, 0.0, None, 0.0],
            "WEATHER_DELAY": [15.0, 15.0, None, None],
            "DISTANCE": ["760", "760", "760", "606"],
            "UNUSED_COLUMN": ["x", "x", "x", "x"],
        }
    )


def test_filter_flights_keeps_pilot_origins_and_known_columns():
    df = bts.filter_flights(_raw_bts(), ["ATL", "JFK"])

    assert set(df["ORIGIN"]) == {"ATL", "JFK"}
    assert "UNUSED_COLUMN" not in df.columns


def test_clean_types_normalizes_codes_and_fills_delay_causes():
    df = bts.clean_types(bts.filter_flights(_raw_bts(), ["ATL", "JFK"]))

    assert df["FL_DATE"].iloc[0] == pd.Timestamp("2024-01-15")
    assert df["OP_CARRIER"].iloc[0] == "DL"
    assert df["TAIL_NUM"].iloc[0] == "N123DL"
    assert str(df["DEP_TIME"].dtype) == "Int64"
    assert pd.isna(df["DEP_TIME"].iloc[2])
    assert df["DIVERTED"].tolist() == [0, 0, 0]
    assert df["WEATHER_DELAY"].tolist() == [15.0, 15.0, 0.0]


def test_finalize_drops_duplicates_sorts_and_keeps_full_schema():
    df = bts.finalize(bts.clean_types(bts.filter_flights(_raw_bts(), ["ATL", "JFK"])))

    assert len(df) == 2
    assert df["FL_DATE"].is_monotonic_increasing
    assert list(df.columns) == bts.FINAL_COLUMNS
    assert df["AIR_TIME"].isna().all()  # absente de la source -> colonne vide


def test_derive_parquet_name_uses_data_period_then_falls_back_to_stem():
    df = pd.DataFrame({"FL_DATE": pd.to_datetime(["2024-03-31", "2024-03-01"])})

    assert bts.derive_parquet_name(df, "T_ONTIME_REPORTING_123.zip") == "bts_2024_03.parquet"
    assert bts.derive_parquet_name(df.iloc[0:0], "raw/T_ONTIME.zip") == "T_ONTIME.parquet"


def test_process_file_reads_zip_and_writes_monthly_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("PILOT_AIRPORTS", "ATL,JFK")
    zip_path = tmp_path / "T_ONTIME_REPORTING_20260817_154907.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("T_ONTIME_REPORTING.csv", _raw_bts().to_csv(index=False))

    output_path = bts.process_file(zip_path, output_dir=tmp_path / "out", delete_source=True)

    assert output_path.name == "bts_2024_01.parquet"
    assert len(pd.read_parquet(output_path)) == 2
    assert not zip_path.exists()


def test_read_bts_file_maps_prezip_columns_and_skips_unused(tmp_path):
    csv_path = tmp_path / "prezip.csv"
    pd.DataFrame({"FlightDate": ["2024-01-15"], "IATA_CODE_Reporting_Airline": ["DL"], "Origin": ["ATL"],
                  "DepDelay": [12.0], "Div1Airport": ["XXX"]}).to_csv(csv_path, index=False)

    df = bts.read_bts_file(csv_path)

    assert list(df.columns) == ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEP_DELAY"]


def test_derive_dep_delay_matches_bts_definition_across_midnight():
    crs = pd.Series([815, 2350, 10, 1200], dtype="Int64")
    dep = pd.Series([830, 10, 2355, None], dtype="Int64")

    delay = bts.derive_dep_delay(crs, dep)

    assert delay.tolist()[:3] == [15.0, 20.0, -15.0]
    assert pd.isna(delay.iloc[3])
