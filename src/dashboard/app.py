"""Dashboard Dash (README section 6, étape 3).

Trois onglets :
1. Carte temps réel : dernière position des avions autour des aéroports
   pilotes (OpenSky, table `aircraft_positions`).
2. Retards historiques & avions : indicateurs, retards par heure, mois et
   compagnie, bilan par avion et historique des vols d'un avion
   sélectionné (entrepôt BTS).
3. Prédiction : simulateur interrogeant le dernier modèle entraîné, avec
   ses métriques de test et les facteurs principaux du retard.

Ce module ne fait que restituer : collecte, ETL et entraînement vivent dans
leurs modules dédiés. Chaque onglet reste utilisable si une source manque.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.dash_table import FormatTemplate
from dash.dash_table.Format import Format, Scheme
from sqlalchemy.exc import SQLAlchemyError

from src.collector.airports_reference import get_airport, pilot_airport_codes
from src.dashboard import queries
from src.etl import db, warehouse
from src.ml.model import DelayModel, latest_model_path

logger = logging.getLogger(__name__)

WEEKDAYS = {1: "Lun", 2: "Mar", 3: "Mer", 4: "Jeu", 5: "Ven", 6: "Sam", 7: "Dim"}
# Codes condition Meteostat (`coco`) proposés dans le simulateur
WEATHER_CONDITIONS = {1: "Clair", 3: "Nuageux", 5: "Brouillard", 7: "Pluie légère", 8: "Pluie",
                      9: "Forte pluie", 14: "Neige légère", 15: "Neige", 25: "Orage"}
CARD_STYLE = {"padding": "1rem", "border": "1px solid #ddd", "borderRadius": "8px", "minWidth": "160px"}


# --- Sources de données ------------------------------------------------------

@lru_cache(maxsize=1)
def realtime_engine():
    return db.get_engine()


@lru_cache(maxsize=1)
def warehouse_engine():
    return warehouse.get_engine()


@lru_cache(maxsize=4)
def _load_model(path: Path) -> DelayModel:
    return DelayModel.load(path)


def load_latest_model() -> DelayModel | None:
    """Dernier modèle de `models/` ; un nouvel entraînement est pris en compte sans redémarrage."""
    path = latest_model_path()
    if path is None:
        return None
    try:
        return _load_model(path)
    except TypeError:
        logger.warning("Modèle %s d'un format antérieur : relancer `python -m src.ml.train`", path.name)
        return None


def safe_query(fn, *args, **kwargs):
    """Exécute une lecture ; renvoie (résultat, None) ou (None, message d'erreur lisible)."""
    try:
        return fn(*args, **kwargs), None
    except (SQLAlchemyError, RuntimeError) as exc:
        logger.warning("Lecture %s impossible : %s", fn.__name__, exc)
        return None, f"Données indisponibles ({type(exc).__name__}). Voir README section 8."


# --- Figures -----------------------------------------------------------------

def positions_figure(positions: pd.DataFrame, airport: str) -> go.Figure:
    center = get_airport(airport)
    df = positions.assign(statut=positions["on_ground"].map({True: "Au sol", False: "En vol"}).fillna("Inconnu"))
    fig = px.scatter_map(
        df, lat="latitude", lon="longitude", color="statut", hover_name="callsign",
        hover_data={"icao24": True, "altitude_m": ":.0f", "velocity_ms": ":.0f", "observed_at": True,
                    "latitude": False, "longitude": False, "statut": False},
        center={"lat": center["latitude"], "lon": center["longitude"]}, zoom=8.5, height=550,
        map_style="carto-positron",
    )
    fig.update_layout(margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      title=f"{len(df)} avions autour de {airport}")
    return fig


def delay_rate_figure(df: pd.DataFrame, x: str, title: str, kind: str = "bar") -> go.Figure:
    plot = px.line if kind == "line" else px.bar
    fig = plot(df, x=x, y="delay_rate", hover_data={"flights": True, "avg_delay_minutes": ":.1f"},
               labels={"delay_rate": "Taux de retard ≥ 15 min", "flights": "Vols",
                       "avg_delay_minutes": "Retard moyen (min)"}, title=title)
    fig.update_yaxes(tickformat=".0%")
    return fig


def aircraft_history_figure(flights: pd.DataFrame, tail_num: str) -> go.Figure:
    operated = flights[flights["cancelled"] == 0].sort_values("fl_time_utc")
    operated = operated.assign(route=operated["origin"] + " → " + operated["dest"])
    fig = px.scatter(operated, x="fl_time_utc", y="dep_delay", color="route",
                     hover_data=["op_carrier", "op_carrier_fl_num", "prcp", "wspd"],
                     labels={"fl_time_utc": "Départ programmé (UTC)", "dep_delay": "Retard au départ (min)"},
                     title=f"Vols de l'avion {tail_num}")
    fig.add_hline(y=15, line_dash="dot", annotation_text="Seuil retard (15 min)")
    return fig


def importances_figure(model: DelayModel) -> go.Figure:
    importances = pd.DataFrame(model.feature_importances(top_n=12).items(), columns=["feature", "importance"])
    fig = px.bar(importances.sort_values("importance"), x="importance", y="feature", orientation="h",
                 title=f"Facteurs du retard — modèle {model.model_version}",
                 labels={"importance": "Importance (perte de PR AUC si la feature est brouillée)", "feature": ""})
    return fig


def kpi_card(label: str, value: str) -> html.Div:
    return html.Div([html.Div(label, style={"color": "#666"}), html.H3(value, style={"margin": 0})], style=CARD_STYLE)


def _pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


# --- Mise en page --------------------------------------------------------------

def realtime_tab() -> html.Div:
    airports = pilot_airport_codes()
    return html.Div([
        html.Label("Aéroport"),
        dcc.Dropdown(airports, airports[0], id="rt-airport", clearable=False, style={"width": "200px"}),
        dcc.Interval(id="rt-refresh", interval=60_000),
        html.Div(id="rt-message"),
        dcc.Graph(id="rt-map"),
    ])


def history_tab() -> html.Div:
    engine = warehouse_engine_or_none()
    carriers = safe_query(queries.carriers, engine)[0] if engine is not None else None
    carrier_options = (
        [{"label": f"{r.airline_name} ({r.op_carrier})", "value": r.op_carrier} for r in carriers.itertuples()]
        if carriers is not None else []
    )
    return html.Div([
        html.Div([
            html.Div([html.Label("Aéroport de départ"),
                      dcc.Dropdown(pilot_airport_codes(), None, id="hist-origin", placeholder="Tous")],
                     style={"width": "220px"}),
            html.Div([html.Label("Compagnie"),
                      dcc.Dropdown(carrier_options, None, id="hist-carrier", placeholder="Toutes")],
                     style={"width": "320px"}),
        ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),
        html.Div(id="hist-message"),
        html.Div(id="hist-kpis", style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "margin": "1rem 0"}),
        html.Div([dcc.Graph(id="hist-by-hour"), dcc.Graph(id="hist-by-weekday")],
                 style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))"}),
        html.Div([dcc.Graph(id="hist-by-month"), dcc.Graph(id="hist-by-carrier")],
                 style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(420px, 1fr))"}),
        html.H3("Avions (au moins 20 vols) — cliquer une ligne pour voir ses vols"),
        dash_table.DataTable(
            id="hist-aircraft", page_size=10, sort_action="native", filter_action="native",
            row_selectable="single", style_table={"overflowX": "auto"},
            columns=[
                {"name": "Immatriculation", "id": "tail_num"},
                {"name": "Constructeur", "id": "manufacturer"},
                {"name": "Modèle", "id": "model"},
                {"name": "Âge (ans)", "id": "aircraft_age"},
                {"name": "Compagnie(s)", "id": "carriers"},
                {"name": "Vols", "id": "flights"},
                {"name": "Taux de retard", "id": "delay_rate", "type": "numeric",
                 "format": FormatTemplate.percentage(1)},
                {"name": "Retard moyen (min)", "id": "avg_delay_minutes", "type": "numeric",
                 "format": Format(precision=1, scheme=Scheme.fixed)},
            ],
        ),
        dcc.Graph(id="hist-aircraft-flights"),
    ])


def prediction_tab() -> html.Div:
    airports = pilot_airport_codes()

    def field(label, component):
        return html.Div([html.Label(label), component], style={"display": "flex", "flexDirection": "column"})

    return html.Div([
        html.Div([
            field("Départ", dcc.Dropdown(airports, airports[0], id="pred-origin", clearable=False)),
            field("Destination (IATA)", dcc.Input(id="pred-dest", value="LAX", maxLength=3)),
            field("Compagnie (IATA)", dcc.Input(id="pred-carrier", value="DL", maxLength=3)),
            field("Date", dcc.DatePickerSingle(id="pred-date", date=date.today())),
            field("Heure prévue (HHMM)", dcc.Input(id="pred-time", type="number", value=1745, min=0, max=2400)),
            field("Distance (miles)", dcc.Input(id="pred-distance", type="number", value=1946)),
            field("Température (°C)", dcc.Input(id="pred-temp", type="number", value=15)),
            field("Précipitations (mm/h)", dcc.Input(id="pred-prcp", type="number", value=0, min=0)),
            field("Vent (km/h)", dcc.Input(id="pred-wspd", type="number", value=15, min=0)),
            field("Pression (hPa)", dcc.Input(id="pred-pres", type="number", value=1015)),
            field("Conditions", dcc.Dropdown(
                [{"label": label, "value": code} for code, label in WEATHER_CONDITIONS.items()], 1,
                id="pred-coco", clearable=True)),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))", "gap": "1rem"}),
        html.Button("Prédire le retard", id="pred-button", n_clicks=0, style={"margin": "1rem 0"}),
        html.Div(id="pred-result"),
        html.Div(id="pred-model-info"),
        dcc.Graph(id="pred-importances"),
    ])


def warehouse_engine_or_none():
    try:
        return warehouse_engine()
    except RuntimeError:
        return None


def build_layout() -> html.Div:
    return html.Div([
        html.H1("DST Airlines — Suivi des vols & prédiction des retards"),
        dcc.Tabs([
            dcc.Tab(realtime_tab(), label="Carte temps réel"),
            dcc.Tab(history_tab(), label="Retards historiques & avions"),
            dcc.Tab(prediction_tab(), label="Prédiction"),
        ]),
    ], style={"fontFamily": "sans-serif", "margin": "1rem 2rem"})


# --- Callbacks -------------------------------------------------------------------

def update_map(airport: str, _n_intervals=None):
    positions, error = safe_query(queries.latest_positions, realtime_engine(), airport)
    if error:
        return go.Figure(), error
    if positions.empty:
        return go.Figure(), f"Aucune position pour {airport} : lancer `python -m src.collector.flights_client` puis `python -m src.etl.pipeline`."
    last_seen = pd.to_datetime(positions["observed_at"]).max()
    return positions_figure(positions, airport), f"Dernière observation : {last_seen:%Y-%m-%d %H:%M} UTC"


def update_history(origin: str | None, carrier: str | None):
    engine = warehouse_engine_or_none()
    empty = go.Figure()
    if engine is None:
        return ("Entrepôt non configuré (WAREHOUSE_DATABASE_URL).", [], empty, empty, empty, empty, [])

    kpis, error = safe_query(queries.delay_kpis, engine, origin, carrier)
    if error or not kpis["flights"]:
        return (error or "Entrepôt vide : lancer `just warehouse`.", [], empty, empty, empty, empty, [])

    by_hour = queries.delay_by(engine, "hour", origin, carrier)
    by_weekday = queries.delay_by(engine, "weekday", origin, carrier)
    by_weekday["key"] = by_weekday["key"].map(WEEKDAYS)
    by_month = queries.delay_by(engine, "month", origin, carrier)
    by_carrier = queries.delay_by(engine, "carrier", origin, carrier).sort_values("delay_rate", ascending=False)
    aircraft = queries.aircraft_summary(engine, origin, carrier)

    cards = [
        kpi_card("Vols opérés", f"{int(kpis['flights']):,}".replace(",", " ")),
        kpi_card("Taux de retard ≥ 15 min", _pct(kpis["delay_rate"])),
        kpi_card("Retard moyen", f"{kpis['avg_delay_minutes'] or 0:.1f} min"),
        kpi_card("Taux d'annulation", _pct(kpis["cancel_rate"])),
        kpi_card("Période", f"{kpis['first_date']} → {kpis['last_date']}"),
    ]
    return (
        "",
        cards,
        delay_rate_figure(by_hour, "key", "Retard selon l'heure de départ prévue").update_xaxes(title="Heure locale"),
        delay_rate_figure(by_weekday, "key", "Retard selon le jour de la semaine").update_xaxes(title=""),
        delay_rate_figure(by_month, "key", "Retard par mois", kind="line").update_xaxes(title=""),
        delay_rate_figure(by_carrier, "key", "Retard par compagnie").update_xaxes(title="Compagnie"),
        aircraft.to_dict("records"),
    )


def update_aircraft_flights(selected_rows: list[int] | None, rows: list[dict] | None):
    if not selected_rows or not rows:
        return go.Figure(layout={"title": "Sélectionner un avion dans le tableau"})
    tail_num = rows[selected_rows[0]]["tail_num"]
    flights, error = safe_query(queries.aircraft_flights, warehouse_engine(), tail_num)
    if error or flights.empty:
        return go.Figure(layout={"title": error or f"Aucun vol pour {tail_num}"})
    return aircraft_history_figure(flights, tail_num)


def predict_delay(n_clicks, origin, dest, carrier, fl_date, crs_dep_time, distance, temp, prcp, wspd, pres, coco=None):
    model = load_latest_model()
    if model is None:
        return html.P("Aucun modèle entraîné : lancer `python -m src.ml.train`."), "", go.Figure()

    metrics = model.metrics
    info = html.P(
        f"Modèle {model.model_version} — test : ROC AUC {metrics.get('test_roc_auc', float('nan')):.3f}, "
        f"PR AUC {metrics.get('test_pr_auc', float('nan')):.3f} (hasard : {metrics.get('test_base_rate', float('nan')):.3f}), "
        f"F1 {metrics.get('test_f1', float('nan')):.3f} au seuil {model.threshold:.2f} ; "
        f"erreur moyenne {metrics.get('test_mae_minutes', float('nan')):.1f} min "
        f"(référence {metrics.get('baseline_mae_minutes', float('nan')):.1f} min).",
        style={"color": "#555"},
    )
    if not n_clicks:
        return "", info, importances_figure(model)

    raw = pd.DataFrame([{
        "fl_date": fl_date, "crs_dep_time": crs_dep_time, "origin": origin,
        "dest": (dest or "").upper(), "op_carrier": (carrier or "").upper(), "distance": distance,
        "temp": temp, "prcp": prcp, "wspd": wspd, "pres": pres, "coco": coco,
    }])
    probability, minutes = model.predict(raw)
    delayed = bool(model.is_delayed(probability)[0])
    verdict = "Retard probable" if delayed else "Départ probablement à l'heure"
    result = html.Div([
        html.H2(f"{verdict} — probabilité {probability[0]:.0%}", style={"color": "#c0392b" if delayed else "#27ae60"}),
        html.P(f"Retard attendu : {minutes[0]:.0f} min (seuil de décision {model.threshold:.0%})"),
    ], style=CARD_STYLE)
    return result, info, importances_figure(model)


def create_app() -> Dash:
    app = Dash(__name__, title="DST Airlines", suppress_callback_exceptions=True)
    app.layout = build_layout
    app.callback(Output("rt-map", "figure"), Output("rt-message", "children"),
                 Input("rt-airport", "value"), Input("rt-refresh", "n_intervals"))(update_map)
    app.callback(
        Output("hist-message", "children"), Output("hist-kpis", "children"),
        Output("hist-by-hour", "figure"), Output("hist-by-weekday", "figure"),
        Output("hist-by-month", "figure"), Output("hist-by-carrier", "figure"),
        Output("hist-aircraft", "data"),
        Input("hist-origin", "value"), Input("hist-carrier", "value"),
    )(update_history)
    app.callback(Output("hist-aircraft-flights", "figure"),
                 Input("hist-aircraft", "selected_rows"), State("hist-aircraft", "data"))(
        update_aircraft_flights)
    app.callback(
        Output("pred-result", "children"), Output("pred-model-info", "children"), Output("pred-importances", "figure"),
        Input("pred-button", "n_clicks"),
        State("pred-origin", "value"), State("pred-dest", "value"), State("pred-carrier", "value"),
        State("pred-date", "date"), State("pred-time", "value"), State("pred-distance", "value"),
        State("pred-temp", "value"), State("pred-prcp", "value"), State("pred-wspd", "value"),
        State("pred-pres", "value"), State("pred-coco", "value"),
    )(predict_delay)
    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="0.0.0.0", port=8050)
