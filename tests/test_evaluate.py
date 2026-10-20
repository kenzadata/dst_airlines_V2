from __future__ import annotations

import json

from src.ml import evaluate
from tests.test_model import _fitted_model, make_flights


def test_evaluate_writes_figures_and_metrics(tmp_path):
    df = make_flights(n=1500)
    model = _fitted_model(df)
    model.max_delay_minutes = 60.0

    results = evaluate.evaluate(df, model, test_months=1, output_dir=tmp_path)

    assert {p.name for p in tmp_path.glob("*.png")} == set(results["figures"])
    assert len(results["figures"]) == 7  # dont l'âge des avions, renseigné dans les vols fictifs
    saved = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    confusion = saved["classifier"]["confusion"]
    assert sum(confusion.values()) == saved["data"]["test_flights"]
    assert 0 <= saved["classifier"]["roc_auc"] <= 1
    assert {row["segment"] for row in saved["by_origin"]} == {"ATL", "JFK"}
    assert len(saved["calibration"]) == 10
    assert saved["regressor"]["capped_share"] <= 1
    assert sum(row["flights"] for row in saved["aircraft"]["by_age"]) == int(df["aircraft_age"].notna().sum())


def test_aircraft_profiles_without_registry_returns_empty_analysis():
    df = make_flights(n=200).assign(aircraft_age=float("nan"))

    profiles = evaluate.aircraft_profiles(df)

    assert profiles["by_age"] == [] and profiles["coverage"] == 0.0
