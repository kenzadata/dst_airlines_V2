"""Lectures en base du dashboard (lecture seule).

- Temps réel : table `aircraft_positions` (pipeline OpenSky, `DATABASE_URL`).
- Historique : entrepôt en étoile (`WAREHOUSE_DATABASE_URL`). Les agrégations
  sont faites en SQL : on ne rapatrie jamais les ~800 000 vols dans pandas.

Un vol est "en retard" s'il part avec 15 minutes ou plus de retard (seuil DOT),
calculé sur les vols non annulés.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

DELAYED = "CASE WHEN f.dep_delay >= 15 THEN 1.0 ELSE 0.0 END"
OPERATED = "f.cancelled = 0 AND f.dep_delay IS NOT NULL"

# Axes d'agrégation autorisés (liste blanche : jamais d'expression venant de l'utilisateur)
GROUPINGS = {
    "hour": "f.crs_dep_time / 100",
    "weekday": "EXTRACT(ISODOW FROM f.fl_date)::int",
    "month": "DATE_TRUNC('month', f.fl_date)::date",
    "carrier": "f.op_carrier",
    "dest": "f.dest",
}


def _filters(origin: str | None, carrier: str | None) -> tuple[str, dict]:
    clauses, params = ["TRUE"], {}
    if origin:
        clauses.append("f.origin = :origin")
        params["origin"] = origin
    if carrier:
        clauses.append("f.op_carrier = :carrier")
        params["carrier"] = carrier
    return " AND ".join(clauses), params


def latest_positions(engine, airport: str | None = None, limit: int = 5000) -> pd.DataFrame:
    """Dernière position connue de chaque aéronef (table `aircraft_positions`)."""
    where = "WHERE airport = :airport" if airport else ""
    sql = text(f"SELECT * FROM aircraft_positions {where} ORDER BY observed_at DESC LIMIT :limit")
    df = pd.read_sql(sql, engine, params={"airport": airport, "limit": limit})
    return df.drop_duplicates(subset=["icao24"]).reset_index(drop=True)


def carriers(engine) -> pd.DataFrame:
    sql = text("""
        SELECT c.op_carrier, COALESCE(a.airline_name, c.op_carrier) AS airline_name
        FROM (SELECT DISTINCT op_carrier FROM fact_flights) c
        LEFT JOIN dim_airline a ON a.iata_code = c.op_carrier
        ORDER BY airline_name
    """)
    return pd.read_sql(sql, engine)


def delay_kpis(engine, origin: str | None = None, carrier: str | None = None) -> dict:
    where, params = _filters(origin, carrier)
    sql = text(f"""
        SELECT
            COUNT(*) FILTER (WHERE {OPERATED})                     AS flights,
            AVG({DELAYED}) FILTER (WHERE {OPERATED})               AS delay_rate,
            AVG(GREATEST(f.dep_delay, 0)) FILTER (WHERE {OPERATED}) AS avg_delay_minutes,
            AVG(f.cancelled::float)                                AS cancel_rate,
            MIN(f.fl_date) AS first_date,
            MAX(f.fl_date) AS last_date
        FROM fact_flights f
        WHERE {where}
    """)
    return pd.read_sql(sql, engine, params=params).iloc[0].to_dict()


def delay_by(engine, by: str, origin: str | None = None, carrier: str | None = None) -> pd.DataFrame:
    """Taux de retard et retard moyen par heure, jour, mois, compagnie ou destination."""
    key = GROUPINGS[by]
    where, params = _filters(origin, carrier)
    sql = text(f"""
        SELECT {key} AS key, COUNT(*) AS flights,
               AVG({DELAYED}) AS delay_rate, AVG(GREATEST(f.dep_delay, 0)) AS avg_delay_minutes
        FROM fact_flights f
        WHERE {where} AND {OPERATED}
        GROUP BY 1
        ORDER BY 1
    """)
    return pd.read_sql(sql, engine, params=params)


def aircraft_summary(
    engine, origin: str | None = None, carrier: str | None = None, min_flights: int = 20, limit: int = 300
) -> pd.DataFrame:
    """Bilan par avion (immatriculation) : nombre de vols, taux et durée moyenne de retard."""
    where, params = _filters(origin, carrier)
    sql = text(f"""
        SELECT f.tail_num, a.manufacturer, a.model, a.aircraft_age,
               STRING_AGG(DISTINCT f.op_carrier, ', ') AS carriers,
               COUNT(*) AS flights,
               AVG({DELAYED}) AS delay_rate,
               AVG(GREATEST(f.dep_delay, 0)) AS avg_delay_minutes
        FROM fact_flights f
        LEFT JOIN dim_aircraft a ON a.tail_num = f.tail_num
        WHERE {where} AND {OPERATED} AND f.tail_num IS NOT NULL
        GROUP BY f.tail_num, a.manufacturer, a.model, a.aircraft_age
        HAVING COUNT(*) >= :min_flights
        ORDER BY flights DESC
        LIMIT :limit
    """)
    return pd.read_sql(sql, engine, params={**params, "min_flights": min_flights, "limit": limit})


def aircraft_flights(engine, tail_num: str, limit: int = 300) -> pd.DataFrame:
    """Derniers vols d'un avion, avec la météo au départ."""
    sql = text("""
        SELECT f.fl_date, f.fl_time_utc, f.origin, f.dest, f.op_carrier, f.op_carrier_fl_num,
               f.crs_dep_time, f.dep_delay, f.cancelled, w.prcp, w.wspd
        FROM fact_flights f
        LEFT JOIN dim_weather w
               ON w.airport_iata_code = f.origin AND w.weather_time_utc = f.fl_time_utc_hour
        WHERE f.tail_num = :tail_num
        ORDER BY f.fl_time_utc DESC
        LIMIT :limit
    """)
    return pd.read_sql(sql, engine, params={"tail_num": tail_num, "limit": limit})
