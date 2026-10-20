from __future__ import annotations

import json

import pandas as pd

from src.etl import dimensions


def test_build_dim_airport_filters_openflights_on_pilot_airports(tmp_path):
    airports_dat = tmp_path / "airports.dat"
    airports_dat.write_text(
        '3682,"Hartsfield Jackson Atlanta International Airport","Atlanta","United States","ATL","KATL",'
        '33.6367,-84.428101,1026,-5,"A","America/New_York","airport","OurAirports"\n'
        '1382,"Charles de Gaulle International Airport","Paris","France","CDG","LFPG",'
        '49.012798,2.55,392,1,"E","Europe/Paris","airport","OurAirports"\n'
        '9999,"No IATA","Nowhere","Nowhere",\\N,"XXXX",0,0,0,0,"N",\\N,"airport","OurAirports"\n',
        encoding="utf-8",
    )

    dim = dimensions.build_dim_airport(airports_dat, ["ATL", "JFK"])

    assert dim["IATA_CODE"].tolist() == ["ATL"]
    assert dim.iloc[0]["TIMEZONE"] == "America/New_York"
    assert list(dim.columns) == ["IATA_CODE", "name", "city", "latitude", "longitude", "TIMEZONE"]


def test_build_dim_airline_renames_and_deduplicates(tmp_path):
    lookup = tmp_path / "L_UNIQUE_CARRIERS.csv"
    lookup.write_text('Code,Description\nDL,"Delta Air Lines Inc."\nB6,"JetBlue Airways"\nDL,"Delta bis"\n,"Vide"\n')

    dim = dimensions.build_dim_airline(lookup)

    assert dim["iata_code"].tolist() == ["B6", "DL"]
    assert dim.loc[dim["iata_code"] == "DL", "airline_name"].item() == "Delta Air Lines Inc."


def test_normalize_tail_adds_n_prefix():
    assert dimensions.normalize_tail(pd.Series([" 123dl ", "1AA"])).tolist() == ["N123DL", "N1AA"]


def test_build_dim_aircraft_joins_faa_registry_and_computes_age(tmp_path):
    master = tmp_path / "MASTER.txt"
    master.write_text(
        "N-NUMBER,SERIAL NUMBER,MFR MDL CODE,YEAR MFR,\n"
        "123DL   ,S1,1384401,2010,\n"
        "1AA     ,S2,1384401,    ,\n"
        "999ZZ   ,S3,1384401,2000,\n"
    )
    acftref = tmp_path / "ACFTREF.txt"
    acftref.write_text("CODE,MFR,MODEL,\n1384401,BOEING              ,737-932ER  ,\n")

    dim = dimensions.build_dim_aircraft({"N123DL", "N1AA", "N404NF"}, master, acftref, current_year=2026)

    row = dim.set_index("TAIL_NUM").loc["N123DL"]
    assert sorted(dim["TAIL_NUM"]) == ["N123DL", "N1AA"]
    assert (row["MANUFACTURER"], row["MODEL"], row["AIRCRAFT_AGE"]) == ("BOEING", "737-932ER", 16)
    assert pd.isna(dim.set_index("TAIL_NUM").loc["N1AA", "AIRCRAFT_AGE"])


def test_load_flight_tails_reads_distinct_tails(tmp_path):
    pd.DataFrame({"TAIL_NUM": ["n1", "N1", None, " "]}).to_parquet(tmp_path / "bts_2024_01.parquet")

    assert dimensions.load_flight_tails(tmp_path) == {"N1"}


def test_build_dim_weather_normalizes_meteostat_history(tmp_path):
    pd.DataFrame(
        {"time": pd.to_datetime(["2024-01-01 01:00", "2024-01-01 00:00"]), "temp": [5.0, 4.0], "prcp": [0.0, 0.2]}
    ).to_parquet(tmp_path / "weather_history_ATL.parquet")

    dim = dimensions.build_dim_weather(tmp_path)

    assert list(dim.columns) == dimensions.WEATHER_COLUMNS
    assert dim["AIRPORT_IATA_CODE"].unique().tolist() == ["ATL"]
    assert str(dim["WEATHER_TIME_UTC"].dt.tz) == "UTC"
    assert dim["temp"].tolist() == [4.0, 5.0]
    assert dim["coco"].isna().all()


def test_owm_id_to_coco_maps_groups():
    assert dimensions.owm_id_to_coco(800) == 1
    assert dimensions.owm_id_to_coco(502) == 9
    assert dimensions.owm_id_to_coco(211) == 25
    assert dimensions.owm_id_to_coco(741) == 5
    assert dimensions.owm_id_to_coco(None) is None


def test_build_weather_realtime_converts_units_to_meteostat_schema(tmp_path):
    payload = {
        "dt": 1704067200,
        "main": {"temp": 3.5, "pressure": 1012},
        "wind": {"speed": 10.0},
        "weather": [{"id": 501}],
        "rain": {"1h": 1.24},
    }
    (tmp_path / "openweather_JFK_20240101T000000Z.json").write_text(
        json.dumps({"airport": "JFK", "collected_at": "20240101T000000Z", "current": payload})
    )

    df = dimensions.build_weather_realtime(tmp_path)

    row = df.iloc[0]
    assert row["AIRPORT_IATA_CODE"] == "JFK"
    assert row["WEATHER_TIME_UTC"] == pd.Timestamp("2024-01-01", tz="UTC")
    assert (row["wspd"], row["prcp"], row["coco"]) == (36.0, 1.2, 8)


def test_build_fl_time_utc_converts_local_departure_to_utc():
    flights = pd.DataFrame(
        {
            "FL_DATE": pd.to_datetime(["2024-01-15", "2024-07-15", "2024-01-15", "2024-01-15", "2024-01-15"]),
            "CRS_DEP_TIME": pd.array([815, 815, 2400, None, 815], dtype="Int64"),
            "ORIGIN": ["ATL", "JFK", "ATL", "ATL", "ORD"],
        }
    )

    result = dimensions.build_fl_time_utc(flights, {"ATL": "America/New_York", "JFK": "America/New_York"})

    assert result.iloc[0] == pd.Timestamp("2024-01-15 13:15", tz="UTC")  # hiver : UTC-5
    assert result.iloc[1] == pd.Timestamp("2024-07-15 12:15", tz="UTC")  # été : UTC-4
    assert result.iloc[2] == pd.Timestamp("2024-01-15 05:00", tz="UTC")  # 2400 = minuit
    assert pd.isna(result.iloc[3])  # heure manquante
    assert pd.isna(result.iloc[4])  # fuseau inconnu


def test_floor_to_hour_truncates():
    s = pd.Series(pd.to_datetime(["2024-01-15 13:59"], utc=True))
    assert dimensions.floor_to_hour(s).iloc[0] == pd.Timestamp("2024-01-15 13:00", tz="UTC")
