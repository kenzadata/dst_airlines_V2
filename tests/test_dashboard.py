from __future__ import annotations

import pandas as pd
from sqlalchemy.exc import OperationalError

from src.dashboard import app as dashboard
from tests.test_model import make_flights


def _rates(keys) -> pd.DataFrame:
    return pd.DataFrame({"key": keys, "flights": 100, "delay_rate": 0.2, "avg_delay_minutes": 12.0})


def _fake_warehouse(monkeypatch):
    monkeypatch.setattr(dashboard, "warehouse_engine_or_none", lambda: object())
    monkeypatch.setattr(dashboard, "warehouse_engine", lambda: object())
    monkeypatch.setattr(dashboard.queries, "delay_kpis", lambda *a: {
        "flights": 1234, "delay_rate": 0.21, "avg_delay_minutes": 13.4, "cancel_rate": 0.01,
        "first_date": "2024-01-01", "last_date": "2025-12-31"})
    groupings = {"hour": [6, 7], "weekday": [1, 2], "month": ["2024-01-01", "2024-02-01"], "carrier": ["DL", "B6"]}
    monkeypatch.setattr(dashboard.queries, "delay_by", lambda engine, by, *a: _rates(groupings[by]))
    monkeypatch.setattr(dashboard.queries, "aircraft_summary", lambda *a: pd.DataFrame(
        {"tail_num": ["N123DL"], "model": ["737"], "flights": [40], "delay_rate": [0.25]}))


def test_update_history_builds_kpis_charts_and_aircraft_table(monkeypatch):
    _fake_warehouse(monkeypatch)

    message, cards, _by_hour, by_weekday, by_month, _by_carrier, aircraft = dashboard.update_history("ATL", None)

    assert message == ""
    assert len(cards) == 5
    assert list(by_weekday.data[0].x) == ["Lun", "Mar"]
    assert by_month.data[0].type == "scatter"  # courbe mensuelle
    assert aircraft == [{"tail_num": "N123DL", "model": "737", "flights": 40, "delay_rate": 0.25}]


def test_update_history_without_warehouse_shows_message(monkeypatch):
    monkeypatch.setattr(dashboard, "warehouse_engine_or_none", lambda: None)

    message, cards, *_figures, aircraft = dashboard.update_history(None, None)

    assert "WAREHOUSE_DATABASE_URL" in message
    assert cards == [] and aircraft == []


def test_update_history_database_error_is_reported(monkeypatch):
    _fake_warehouse(monkeypatch)

    def failing(*args):
        raise OperationalError("SELECT", {}, Exception("connexion refusée"))

    monkeypatch.setattr(dashboard.queries, "delay_kpis", failing)

    message, *_ = dashboard.update_history(None, None)

    assert "indisponibles" in message


def test_update_aircraft_flights_plots_selected_aircraft(monkeypatch):
    _fake_warehouse(monkeypatch)
    flights = pd.DataFrame({
        "fl_time_utc": pd.to_datetime(["2024-01-01 13:00", "2024-01-02 13:00"], utc=True),
        "origin": ["ATL", "ATL"], "dest": ["LAX", "LAX"], "op_carrier": ["DL", "DL"],
        "op_carrier_fl_num": [1, 1], "dep_delay": [5.0, 40.0], "cancelled": [0, 0], "prcp": [0.0, 3.0], "wspd": [10.0, 30.0],
    })
    captured = {}

    def fake_flights(engine, tail_num):
        captured["tail_num"] = tail_num
        return flights

    monkeypatch.setattr(dashboard.queries, "aircraft_flights", fake_flights)

    figure = dashboard.update_aircraft_flights([1], [{"tail_num": "N1"}, {"tail_num": "N123DL"}])

    assert captured["tail_num"] == "N123DL"
    assert list(figure.data[0].y) == [5.0, 40.0]


def test_update_map_plots_latest_positions(monkeypatch):
    positions = pd.DataFrame({
        "icao24": ["abc"], "callsign": ["DAL1"], "latitude": [33.6], "longitude": [-84.4], "altitude_m": [900.0],
        "velocity_ms": [120.0], "on_ground": [False], "observed_at": pd.to_datetime(["2026-10-20 21:36"]),
    })
    monkeypatch.setattr(dashboard, "realtime_engine", lambda: object())
    monkeypatch.setattr(dashboard.queries, "latest_positions", lambda engine, airport: positions)

    figure, message = dashboard.update_map("ATL")

    assert figure.data[0].type == "scattermap"
    assert "2026-10-20 21:36" in message


def test_predict_delay_uses_latest_model(monkeypatch):
    from tests.test_model import _fitted_model

    model = _fitted_model(make_flights(n=300))
    model.metrics = {"test_roc_auc": 0.7, "test_pr_auc": 0.4, "test_base_rate": 0.2, "test_f1": 0.45,
                     "test_mae_minutes": 14.0, "baseline_mae_minutes": 18.0}
    monkeypatch.setattr(dashboard, "load_latest_model", lambda: model)

    result, info, importances = dashboard.predict_delay(
        1, "ATL", "lax", "dl", "2024-07-15", 1745, 1946, 30, 5.0, 40, 1005, 25)

    assert "probabilité" in str(result)
    assert "ROC AUC 0.700" in str(info)
    assert importances.data[0].orientation == "h"


def test_predict_delay_without_model(monkeypatch):
    monkeypatch.setattr(dashboard, "load_latest_model", lambda: None)

    result, *_ = dashboard.predict_delay(1, "ATL", "LAX", "DL", "2024-07-15", 1745, 1946, 30, 5.0, 40, 1005)

    assert "Aucun modèle" in str(result)
