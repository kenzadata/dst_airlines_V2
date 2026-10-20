"""Dashboard Dash : vols suivis + retards prédits (README section 6, étape 3).

Ce module ne fait que restituer les données déjà stockées/prédites — la
collecte, l'ETL et l'entraînement vivent dans leurs modules dédiés.
"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
from dash import Dash, dash_table, dcc, html
from sqlalchemy.exc import SQLAlchemyError

from src.etl.db import get_engine
from src.ml.model import DelayModel, latest_model_path

logger = logging.getLogger(__name__)


def load_flights_df() -> pd.DataFrame:
    engine = get_engine()
    try:
        return pd.read_sql("SELECT * FROM flights ORDER BY origin_airport", engine)
    except SQLAlchemyError:
        logger.warning("Impossible de lire la table flights (DB non initialisée ?)")
        return pd.DataFrame(columns=["flight_id", "airline", "origin_airport", "status"])


def load_latest_model() -> DelayModel | None:
    path = latest_model_path()
    return DelayModel.load(path) if path else None


def build_layout() -> html.Div:
    flights_df = load_flights_df()
    model = load_latest_model()

    status_counts = (
        flights_df["status"].value_counts().reset_index(name="count").rename(columns={"index": "status"})
        if not flights_df.empty
        else pd.DataFrame(columns=["status", "count"])
    )
    status_fig = px.bar(status_counts, x="status", y="count", title="Statuts des vols suivis")

    children = [
        html.H1("DST Airlines — Suivi de vols & retards prédits"),
        html.H2("Vols suivis (aéroports pilotes)"),
        dash_table.DataTable(
            data=flights_df.tail(50).to_dict("records"),
            columns=[{"name": c, "id": c} for c in flights_df.columns],
            page_size=10,
        ),
        dcc.Graph(figure=status_fig),
    ]

    if model is not None:
        importance_df = pd.DataFrame(
            model.feature_importances().items(), columns=["feature", "importance"]
        )
        importance_fig = px.bar(
            importance_df, x="feature", y="importance",
            title=f"Facteurs de retard — modèle {model.model_version}",
        )
        children.append(dcc.Graph(figure=importance_fig))
    else:
        children.append(html.P("Aucun modèle entraîné — lancer `python -m src.ml.train`."))

    return html.Div(children, style={"fontFamily": "sans-serif", "margin": "2rem"})


def create_app() -> Dash:
    app = Dash(__name__, title="DST Airlines")
    app.layout = build_layout
    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="0.0.0.0", port=8050)
