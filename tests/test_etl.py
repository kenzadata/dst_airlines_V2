from __future__ import annotations

from src.etl import pipeline


def _flight_record(airport="ATL"):
    return {
        "airport": airport,
        "collected_at": "20240101T000000Z",
        "states": [
            {"icao24": "abc123", "callsign": "AF123", "on_ground": False, "last_contact": 1704067200},
            {"icao24": "abc123", "callsign": "AF123", "on_ground": False, "last_contact": None},
        ],
    }


def _weather_record(airport="ATL"):
    return {
        "airport": airport,
        "collected_at": "20240101T000000Z",
        "forecast": {
            "hourly": {
                "time": ["2024-01-01T00:00"],
                "temperature_2m": [5.0],
                "wind_speed_10m": [20.0],
                "precipitation": [0.5],
                "visibility": [9000.0],
            }
        },
    }


def test_flights_to_dataframe_drops_incomplete_and_builds_flight_id():
    df = pipeline.flights_to_dataframe([_flight_record()])

    assert len(df) == 1
    assert df.iloc[0]["flight_id"] == "abc123_20240101"
    assert df.iloc[0]["status"] == "en vol"


def test_flights_to_dataframe_empty_records_returns_empty_df():
    df = pipeline.flights_to_dataframe([])
    assert df.empty


def test_weather_to_dataframe_flattens_hourly_series():
    df = pipeline.weather_to_dataframe([_weather_record()])

    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 5.0
    assert df.iloc[0]["airport"] == "ATL"


def test_join_flights_weather_attaches_nearest_weather():
    flights_df = pipeline.flights_to_dataframe([_flight_record()])
    weather_df = pipeline.weather_to_dataframe([_weather_record()])

    joined = pipeline.join_flights_weather(flights_df, weather_df)

    assert joined.iloc[0]["temperature"] == 5.0


def test_join_flights_weather_no_weather_data_fills_none():
    flights_df = pipeline.flights_to_dataframe([_flight_record()])
    empty_weather_df = pipeline.weather_to_dataframe([])

    joined = pipeline.join_flights_weather(flights_df, empty_weather_df)

    assert joined.iloc[0]["temperature"] is None


def test_upsert_flights_and_weather_persists_to_db(tmp_path):
    from src.etl.db import Flight, get_engine, get_session_factory

    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    flights_df = pipeline.flights_to_dataframe([_flight_record()])
    weather_df = pipeline.weather_to_dataframe([_weather_record()])

    pipeline.upsert_flights_and_weather(flights_df, weather_df, engine=engine)

    with get_session_factory(engine)() as session:
        flights = session.query(Flight).all()
        assert len(flights) == 1
        assert flights[0].origin_airport == "ATL"


def test_positions_to_dataframe_keeps_located_states_only():
    record = {
        "airport": "ATL",
        "states": [
            {"icao24": "abc123", "callsign": "DAL123  ", "last_contact": 1704067200, "latitude": 33.6,
             "longitude": -84.4, "baro_altitude": 3000.0, "velocity": 150.0, "true_track": 90.0, "on_ground": False},
            {"icao24": "def456", "callsign": "JBU1", "last_contact": 1704067200, "latitude": None, "longitude": None},
        ],
    }

    df = pipeline.positions_to_dataframe([record])

    assert df["icao24"].tolist() == ["abc123"]
    assert df.iloc[0]["callsign"] == "DAL123"
    assert df.iloc[0]["id"] == "abc123_1704067200"


def test_upsert_positions_is_idempotent(tmp_path):
    from src.etl.db import AircraftPosition, get_engine, get_session_factory

    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    record = {"airport": "JFK", "states": [{"icao24": "abc123", "last_contact": 1704067200, "latitude": 40.6,
                                             "longitude": -73.8, "on_ground": True}]}
    df = pipeline.positions_to_dataframe([record])

    pipeline.upsert_positions(df, engine)
    pipeline.upsert_positions(df, engine)

    with get_session_factory(engine)() as session:
        assert session.query(AircraftPosition).count() == 1
