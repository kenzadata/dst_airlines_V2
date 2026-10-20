"""Jeu d'entraînement du modèle de retard, lu depuis l'entrepôt historique.

Une ligne par vol BTS non annulé, au départ d'un aéroport pilote, avec :
- les informations connues AVANT le départ (horaire programmé, route,
  compagnie, avion) : ce sont les seules utilisables comme features ;
- la météo observée à l'aéroport de départ à l'heure programmée ;
- la cible : le retard au départ observé (`dep_delay`, minutes).

Les colonnes postérieures au départ (heures réelles, causes de retard,
temps de vol réel) sont volontairement exclues pour éviter toute fuite de
la cible dans les features.
"""
from __future__ import annotations

import pandas as pd

TRAINING_QUERY = """
SELECT
    f.fl_date,
    f.fl_time_utc,
    f.origin,
    f.dest,
    f.op_carrier,
    f.tail_num,
    f.crs_dep_time,
    f.distance,
    f.crs_elapsed_time,
    a.aircraft_age,
    a.manufacturer,  -- analyse uniquement (src/ml/evaluate.py), pas une feature
    a.model AS aircraft_model,
    w.temp,
    w.prcp,
    w.wspd,
    w.pres,
    w.coco,
    f.dep_delay
FROM fact_flights f
LEFT JOIN dim_aircraft a
    ON a.tail_num = f.tail_num
LEFT JOIN dim_weather w
    ON w.airport_iata_code = f.origin
   AND w.weather_time_utc = f.fl_time_utc_hour
WHERE f.cancelled = 0
  AND f.dep_delay IS NOT NULL
  AND f.fl_time_utc IS NOT NULL
ORDER BY f.fl_time_utc
"""


def load_training_data(engine) -> pd.DataFrame:
    """Charge le jeu d'entraînement, trié chronologiquement (requis par la validation temporelle)."""
    df = pd.read_sql(TRAINING_QUERY, engine)
    df["fl_date"] = pd.to_datetime(df["fl_date"])
    df["fl_time_utc"] = pd.to_datetime(df["fl_time_utc"], utc=True)
    return df
