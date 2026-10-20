from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.collector import (
    flights_client,
    meteostat_client,
    openflights_client,
    openweather_client,
    weather_client,
)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(flights_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(weather_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(openweather_client.time, "sleep", lambda *_: None)
    monkeypatch.setattr(openflights_client.time, "sleep", lambda *_: None)


def test_fetch_states_for_airport_parses_fields(monkeypatch):
    raw_state = [
        "abc123", "AF123   ", "France", 1700000000, 1700000000,
        2.55, 49.0, 10000, False, 250, 90, 0, None, 10500, "1000", False, 0,
    ]
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"states": [raw_state]}
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(flights_client.requests, "get", lambda *a, **k: mock_response)

    states = flights_client.fetch_states_for_airport("ATL")

    assert len(states) == 1
    assert states[0]["icao24"] == "abc123"
    assert states[0]["on_ground"] is False


def test_fetch_states_retries_then_gives_up_on_rate_limit(monkeypatch):
    mock_response = MagicMock(status_code=429)
    calls = {"count": 0}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        return mock_response

    monkeypatch.setattr(flights_client.requests, "get", fake_get)

    states = flights_client.fetch_states_for_airport("ATL", max_retries=2)

    assert states == []
    assert calls["count"] == 2


def test_fetch_states_unknown_airport_raises():
    with pytest.raises(ValueError):
        flights_client.fetch_states_for_airport("XXX")


def test_save_raw_writes_json(tmp_path):
    states = [{"icao24": "abc123", "callsign": "AF123", "on_ground": False, "last_contact": 1700000000}]
    output_path = flights_client.save_raw("ATL", states, output_dir=tmp_path)

    assert output_path is not None
    assert output_path.exists()


def test_save_raw_no_states_returns_none(tmp_path):
    assert flights_client.save_raw("ATL", [], output_dir=tmp_path) is None


def test_fetch_weather_for_airport_success(monkeypatch):
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "hourly": {"time": ["2024-01-01T00:00"], "temperature_2m": [5.0]}
    }
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(weather_client.requests, "get", lambda *a, **k: mock_response)

    payload = weather_client.fetch_weather_for_airport("JFK")

    assert payload["hourly"]["temperature_2m"] == [5.0]


def test_fetch_weather_network_error_returns_none(monkeypatch):
    def fake_get(*args, **kwargs):
        raise weather_client.requests.RequestException("boom")

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    assert weather_client.fetch_weather_for_airport("JFK", max_retries=1) is None


def test_fetch_current_weather_without_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    monkeypatch.setattr(openweather_client.requests, "get", lambda *a, **k: pytest.fail("appel réseau inattendu"))

    assert openweather_client.fetch_current_weather("ATL") is None


def test_fetch_current_weather_retries_then_gives_up_on_rate_limit(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    calls = {"count": 0}

    def fake_get(*args, **kwargs):
        calls["count"] += 1
        return MagicMock(status_code=429)

    monkeypatch.setattr(openweather_client.requests, "get", fake_get)

    assert openweather_client.fetch_current_weather("JFK", max_retries=2) is None
    assert calls["count"] == 2


def test_openweather_save_raw_writes_json(tmp_path):
    output_path = openweather_client.save_raw("ATL", {"dt": 1704067200}, output_dir=tmp_path)

    assert output_path.name.startswith("openweather_ATL_")
    assert openweather_client.save_raw("ATL", None, output_dir=tmp_path) is None


def test_fetch_airports_dat_writes_file(tmp_path, monkeypatch):
    mock_response = MagicMock(status_code=200, text='1,"Airport","City","Country","ATL"\n')
    mock_response.raise_for_status.return_value = None
    monkeypatch.setattr(openflights_client.requests, "get", lambda *a, **k: mock_response)

    output_path = openflights_client.fetch_airports_dat(output_path=tmp_path / "airports.dat")

    assert output_path.read_text(encoding="utf-8").startswith("1,")


def test_fetch_airports_dat_network_error_returns_none(tmp_path, monkeypatch):
    def fake_get(*args, **kwargs):
        raise openflights_client.requests.RequestException("boom")

    monkeypatch.setattr(openflights_client.requests, "get", fake_get)

    assert openflights_client.fetch_airports_dat(output_path=tmp_path / "a.dat", max_retries=1) is None


def test_fetch_hourly_history_error_returns_empty_dataframe(monkeypatch):
    def failing_hourly(*args, **kwargs):
        raise OSError("meteostat indisponible")

    monkeypatch.setattr(meteostat_client, "Hourly", failing_hourly)

    assert meteostat_client.fetch_hourly_history("ATL").empty


def test_meteostat_save_raw_writes_parquet(tmp_path):
    df = pd.DataFrame({"time": pd.to_datetime(["2024-01-01"]), "temp": [5.0]})

    output_path = meteostat_client.save_raw("JFK", df, output_dir=tmp_path)

    assert output_path.name == "weather_history_JFK.parquet"
    assert meteostat_client.save_raw("JFK", pd.DataFrame(), output_dir=tmp_path) is None


def test_bts_download_month_streams_to_file_and_skips_existing(tmp_path, monkeypatch):
    from src.collector import bts_client

    calls = []
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_content.return_value = [b"PK", b"zip"]
    monkeypatch.setattr(bts_client.requests, "get", lambda url, **k: calls.append(url) or response)

    path = bts_client.download_month(2024, 1, output_dir=tmp_path)
    bts_client.download_month(2024, 1, output_dir=tmp_path)

    assert path.name == "bts_raw_2024_01.zip"
    assert path.read_bytes() == b"PKzip"
    assert calls == [bts_client.PREZIP_URL.format(year=2024, month=1)]


def test_bts_download_network_error_returns_none(tmp_path, monkeypatch):
    from src.collector import bts_client

    def fake_get(*args, **kwargs):
        raise bts_client.requests.RequestException("boom")

    monkeypatch.setattr(bts_client.requests, "get", fake_get)
    monkeypatch.setattr(bts_client.time, "sleep", lambda *_: None)

    assert bts_client.download_month(2024, 2, output_dir=tmp_path) is None
    assert list(tmp_path.iterdir()) == []
