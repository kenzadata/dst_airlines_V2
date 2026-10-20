from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import requests

from src.ml import score


def _flights() -> pd.DataFrame:
    return pd.DataFrame({
        "flight_id": ["DL1_20251015_ATL", "B62_20251015_JFK", "AA3_20251015_ATL"],
        "fl_date": pd.to_datetime(["2025-10-15"] * 3),
        "crs_dep_time": np.array([1745, 700, 1200], dtype="int16"),
        "origin": ["ATL", "JFK", "ATL"], "dest": ["LAX", "BOS", "MIA"], "op_carrier": ["DL", "B6", "AA"],
        "tail_num": ["N1", "N2", None], "distance": [1946.0, 187.0, 595.0], "crs_elapsed_time": [300.0, 75.0, 110.0],
        "aircraft_age": [np.nan, 12.0, np.nan], "temp": [20.0, 5.0, 25.0], "prcp": [3.0, 0.0, np.nan],
        "wspd": [30.0, 10.0, 12.0], "pres": [1005.0, 1020.0, 1015.0], "coco": [8.0, 1.0, np.nan],
        "dep_delay": [45.0, -3.0, 10.0],
    })


class FakeSession:
    """Session HTTP simulée : renvoie une prédiction, ou une erreur pour les vols listés."""

    def __init__(self, failing=(), healthy=True):
        self.failing, self.healthy, self.payloads = set(failing), healthy, []

    def get(self, url, timeout):
        if not self.healthy:
            raise requests.ConnectionError("refusé")
        return FakeResponse({"status": "ok", "model_version": "v-test"})

    def post(self, url, json, timeout):
        self.payloads.append(json)
        if json["flight_id"] in self.failing:
            return FakeResponse({}, status=500)
        delayed = json["prcp"] is not None and json["prcp"] > 1
        return FakeResponse({"delay_probability": 0.6 if delayed else 0.1, "predicted_delayed": delayed,
                             "predicted_delay_minutes": 40.0 if delayed else 5.0, "model_version": "v-test",
                             "feature_importances": {"dep_hour": 0.07}})


class FakeResponse:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


def test_to_payload_matches_api_schema_and_nulls_missing_values():
    payload = score.to_payload(_flights().to_dict("records")[2])

    assert set(payload) == set(score.REQUEST_FIELDS)
    assert payload["fl_date"] == "2025-10-15"
    assert payload["crs_dep_time"] == 1200 and isinstance(payload["crs_dep_time"], int)
    assert payload["coco"] is None and payload["prcp"] is None and payload["aircraft_age"] is None


def test_score_flights_compares_predictions_with_actual_delays():
    session = FakeSession()

    scored = score.score_flights(_flights(), "http://api", session)

    assert len(session.payloads) == 3
    assert "dep_delay" not in session.payloads[0]  # la cible n'est jamais envoyée à l'API
    assert scored["predicted_delayed"].tolist() == [True, False, False]
    assert scored["actual_delayed"].tolist() == [True, False, False]
    assert scored["actual_delay_minutes"].tolist() == [45.0, 0.0, 10.0]
    summary = score.summarize(scored)
    assert (summary["scored"], summary["accuracy"], summary["recall"]) == (3, 1.0, 1.0)


def test_score_flights_records_isolated_errors_without_stopping():
    scored = score.score_flights(_flights(), "http://api", FakeSession(failing={"B62_20251015_JFK"}))

    assert scored["error"].notna().tolist() == [False, True, False]
    assert score.summarize(scored)["errors"] == 1


def test_check_api_reports_unreachable_api():
    with pytest.raises(RuntimeError, match="injoignable"):
        score.check_api("http://api", FakeSession(healthy=False))
    assert score.check_api("http://api", FakeSession()) == "v-test"


def test_save_writes_csv_named_after_model_version(tmp_path):
    scored = score.score_flights(_flights(), "http://api", FakeSession())

    path = score.save(scored, output_dir=tmp_path)

    assert path.name.startswith("scoring_v-test_") and path.suffix == ".csv"
    assert len(pd.read_csv(path)) == 3


def test_write_docs_report_publishes_summary_table_and_csv(tmp_path):
    scored = score.score_flights(_flights(), "http://api", FakeSession(failing={"AA3_20251015_ATL"}))
    csv_path = score.save(scored, output_dir=tmp_path / "out")

    report = score.write_docs_report(scored, csv_path, n=3, months=3, seed=42, api_url="http://api",
                                     docs_dir=tmp_path / "docs")

    text = report.read_text(encoding="utf-8")
    assert (tmp_path / "docs" / "scoring" / csv_path.name).exists()
    assert "| Vols scorés / en erreur | 2 / 1 |" in text
    assert "| DL1_20251015_ATL | ATL → LAX | 17:45 |" in text
    assert "| Pluie |" in text  # code Meteostat 8 affiché en clair
    assert "## Erreurs" in text and "AA3_20251015_ATL" in text
    assert text.index("DL1_20251015_ATL") < text.index("B62_20251015_JFK")  # trié par probabilité
