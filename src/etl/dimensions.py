"""Construction des dimensions du schéma en étoile (`sql/schema.sql`) et des
clés temporelles de FACT_FLIGHTS.

Chaque `build_*` lit une source brute (collectée par `src/collector/` ou
déposée manuellement) et renvoie un DataFrame au format de la table cible :

| Dimension    | Source brute                                              |
|--------------|-----------------------------------------------------------|
| dim_airport  | OpenFlights `data/raw/airports.dat` (openflights_client)  |
| dim_airline  | BTS `data/reference/L_UNIQUE_CARRIERS.csv` (manuel)       |
| dim_aircraft | FAA `data/raw/faa/MASTER.txt` + `ACFTREF.txt` (manuel)    |
| dim_weather  | Meteostat `data/raw/weather_history_*.parquet`            |

Usage :
    python -m src.etl.dimensions [airports|airlines|aircraft|weather|weather-realtime|all]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.collector.airports_reference import pilot_airport_codes
from src.etl.bts import PROCESSED_BTS_DIR

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
REFERENCE_DIR = ROOT_DIR / "data" / "reference"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

AIRPORTS_DAT_PATH = RAW_DATA_DIR / "airports.dat"
CARRIERS_LOOKUP_PATH = REFERENCE_DIR / "L_UNIQUE_CARRIERS.csv"
FAA_MASTER_PATH = RAW_DATA_DIR / "faa" / "MASTER.txt"
FAA_ACFTREF_PATH = RAW_DATA_DIR / "faa" / "ACFTREF.txt"

DIM_AIRPORT_PATH = PROCESSED_DATA_DIR / "dim_airport.csv"
DIM_AIRLINE_PATH = PROCESSED_DATA_DIR / "dim_airline.csv"
DIM_AIRCRAFT_PATH = PROCESSED_DATA_DIR / "dim_aircraft.parquet"
DIM_WEATHER_PATH = PROCESSED_DATA_DIR / "dim_weather.parquet"
WEATHER_REALTIME_PATH = PROCESSED_DATA_DIR / "weather_realtime.parquet"

# Variables météo à valeur prédictive causale sur le retard au départ :
# temp (°C), prcp (mm), wspd (km/h), pres (hPa), coco (code condition Meteostat).
WEATHER_COLUMNS = ["AIRPORT_IATA_CODE", "WEATHER_TIME_UTC", "temp", "prcp", "wspd", "pres", "coco"]


# --- DIM_AIRPORT -------------------------------------------------------------

# Format OpenFlights airports.dat (pas d'en-tête dans le fichier)
OPENFLIGHTS_COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "latitude", "longitude", "altitude", "timezone_offset",
    "dst", "tz_database", "type", "source",
]


def build_dim_airport(airports_dat: Path = AIRPORTS_DAT_PATH, iata_codes: list[str] | None = None) -> pd.DataFrame:
    iata_codes = iata_codes or pilot_airport_codes()
    df = pd.read_csv(
        airports_dat, header=None, names=OPENFLIGHTS_COLUMNS,
        na_values=["\\N", ""],  # OpenFlights encode les valeurs manquantes en \N
    )
    df = df[df["iata"].isin(iata_codes)]
    dim = df[["iata", "name", "city", "latitude", "longitude", "tz_database"]].rename(
        columns={"iata": "IATA_CODE", "tz_database": "TIMEZONE"}
    )
    missing = set(iata_codes) - set(dim["IATA_CODE"])
    if missing:
        logger.warning("Aéroports absents d'OpenFlights : %s", sorted(missing))
    return dim.reset_index(drop=True)


# --- DIM_AIRLINE -------------------------------------------------------------

def build_dim_airline(lookup_path: Path = CARRIERS_LOOKUP_PATH) -> pd.DataFrame:
    """BTS livre la table de décodage avec les colonnes `Code` et `Description`."""
    df = pd.read_csv(lookup_path, encoding="utf-8")
    return (
        df.rename(columns={"Code": "iata_code", "Description": "airline_name"})
        [["iata_code", "airline_name"]]
        .dropna()
        .drop_duplicates(subset=["iata_code"])
        .sort_values("iata_code")
        .reset_index(drop=True)
    )


# --- DIM_AIRCRAFT ------------------------------------------------------------

def load_flight_tails(flights_dir: Path = PROCESSED_BTS_DIR) -> set[str]:
    """TAIL_NUM distincts présents dans les vols BTS nettoyés."""
    files = sorted(flights_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Aucun Parquet de vols dans {flights_dir}")
    tails: set[str] = set()
    for path in files:
        s = pd.read_parquet(path, columns=["TAIL_NUM"])["TAIL_NUM"].dropna().astype(str).str.strip().str.upper()
        tails.update(s[s != ""])
    return tails


def normalize_tail(series: pd.Series) -> pd.Series:
    """FAA stocke le N-NUMBER sans le préfixe 'N' ('12345'), BTS l'inclut ('N12345')."""
    return "N" + series.astype(str).str.strip().str.upper()


def _read_faa_csv(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    # Les fichiers FAA ont des noms de colonnes avec espaces et des valeurs paddées
    df = pd.read_csv(path, dtype=str, usecols=lambda c: c.strip() in columns, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=columns)
    return df.apply(lambda s: s.str.strip())


def build_dim_aircraft(
    tails: set[str],
    master_path: Path = FAA_MASTER_PATH,
    acftref_path: Path = FAA_ACFTREF_PATH,
    current_year: int | None = None,
) -> pd.DataFrame:
    """Registre FAA (MASTER + ACFTREF) filtré aux seuls avions présents dans les vols."""
    master = _read_faa_csv(master_path, {"N-NUMBER": "N_NUMBER", "MFR MDL CODE": "MFR_MDL_CODE", "YEAR MFR": "YEAR_MFR"})
    master["TAIL_NUM"] = normalize_tail(master["N_NUMBER"])
    master = master[master["TAIL_NUM"].isin(tails)]

    missing = tails - set(master["TAIL_NUM"])
    if missing:
        logger.warning("%s TAIL_NUM absents du registre FAA (non-US, immatriculation changée ou invalide)", len(missing))

    acftref = _read_faa_csv(acftref_path, {"CODE": "MFR_MDL_CODE", "MFR": "MANUFACTURER", "MODEL": "MODEL"})
    dim = master.merge(acftref, on="MFR_MDL_CODE", how="left")

    dim["YEAR_MFR"] = pd.to_numeric(dim["YEAR_MFR"], errors="coerce")
    dim["AIRCRAFT_AGE"] = (current_year or datetime.now(UTC).year) - dim["YEAR_MFR"]
    # Années FAA parfois vides ou aberrantes -> âge inconnu plutôt que faux
    dim.loc[(dim["YEAR_MFR"] < 1930) | (dim["AIRCRAFT_AGE"] < 0), ["YEAR_MFR", "AIRCRAFT_AGE"]] = np.nan

    dim = dim[["TAIL_NUM", "MANUFACTURER", "MODEL", "YEAR_MFR", "AIRCRAFT_AGE"]]
    return dim.drop_duplicates(subset=["TAIL_NUM"]).reset_index(drop=True)


# --- DIM_WEATHER -------------------------------------------------------------

def meteostat_to_dim_weather(iata_code: str, raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise un historique Meteostat brut (colonne `time` en UTC naïf)."""
    df = raw.rename(columns={"time": "WEATHER_TIME_UTC"}).copy()
    df["WEATHER_TIME_UTC"] = pd.to_datetime(df["WEATHER_TIME_UTC"], utc=True)
    df["AIRPORT_IATA_CODE"] = iata_code
    # Selon la station certaines variables manquent : schéma stable garanti
    for col in WEATHER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[WEATHER_COLUMNS]


def build_dim_weather(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    frames = [
        meteostat_to_dim_weather(path.stem.removeprefix("weather_history_"), pd.read_parquet(path))
        for path in sorted(raw_dir.glob("weather_history_*.parquet"))
    ]
    if not frames:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    weather = pd.concat(frames, ignore_index=True)
    return weather.sort_values(["AIRPORT_IATA_CODE", "WEATHER_TIME_UTC"]).reset_index(drop=True)


def owm_id_to_coco(owm_id: int | None) -> int | None:
    """Code condition OpenWeather (`weather.id`) -> code `coco` Meteostat.

    coco : 1=Clair, 2=Peu nuageux, 3=Nuageux, 4=Couvert, 5=Brume, 7=Pluie légère,
    8=Pluie, 9=Pluie forte, 14=Neige légère, 15=Neige, 16=Neige forte,
    17=Averse pluie, 24=Grêle/pluie verglaçante, 25=Orage.
    """
    if owm_id is None:
        return None
    if 200 <= owm_id < 300:          # Thunderstorm
        return 25
    if 300 <= owm_id < 400:          # Drizzle
        return 7
    if 500 <= owm_id < 505:          # Rain (light -> extreme)
        return {500: 7, 501: 8}.get(owm_id, 9)
    if owm_id == 511:                # Freezing rain
        return 24
    if 520 <= owm_id < 600:          # Shower rain
        return 17
    if 600 <= owm_id < 700:          # Snow
        return {600: 14, 601: 15, 602: 16}.get(owm_id, 15)
    if 700 <= owm_id < 800:          # Atmosphere (mist, fog, haze...)
        return 5
    return {800: 1, 801: 2, 802: 3, 803: 4, 804: 4}.get(owm_id)


def openweather_to_weather_row(iata_code: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise une réponse OpenWeather vers le schéma Meteostat de l'entraînement.

    Le modèle doit recevoir à l'inférence exactement les mêmes features (colonnes,
    unités, encodage coco) qu'à l'entraînement.
    """
    main = payload.get("main", {})
    wind_ms = payload.get("wind", {}).get("speed")
    weather = payload.get("weather") or [{}]
    prcp = payload.get("rain", {}).get("1h", 0.0) + payload.get("snow", {}).get("1h", 0.0)
    return {
        "AIRPORT_IATA_CODE": iata_code,
        "WEATHER_TIME_UTC": pd.to_datetime(payload["dt"], unit="s", utc=True),
        "temp": main.get("temp"),
        "prcp": round(prcp, 1),
        "wspd": round(wind_ms * 3.6, 1) if wind_ms is not None else None,  # m/s -> km/h
        "pres": main.get("pressure"),
        "coco": owm_id_to_coco(weather[0].get("id")),
    }


def build_weather_realtime(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("openweather_*.json")):
        with path.open(encoding="utf-8") as f:
            record = json.load(f)
        rows.append(openweather_to_weather_row(record["airport"], record["current"]))
    return pd.DataFrame(rows, columns=WEATHER_COLUMNS)


# --- Clés temporelles de FACT_FLIGHTS ------------------------------------------

def build_fl_time_utc(df: pd.DataFrame, tz_map: dict[str, str]) -> pd.Series:
    """Heure de départ programmée en UTC, depuis FL_DATE + CRS_DEP_TIME (heure locale ORIGIN).

    `tz_map` : IATA -> fuseau IANA (colonne TIMEZONE de dim_airport). Renvoie NaT
    si le fuseau ou l'heure manque. Sert de clé de jointure avec dim_weather.
    """
    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    hhmm = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce") % 2400  # BTS encode parfois minuit en 2400
    local_naive = (
        pd.to_datetime(df["FL_DATE"]).dt.normalize()
        + pd.to_timedelta(hhmm // 100, unit="h")
        + pd.to_timedelta(hhmm % 100, unit="m")
    )
    for origin, tz in tz_map.items():
        mask = (df["ORIGIN"] == origin) & local_naive.notna()
        if mask.any():
            result[mask] = (
                local_naive[mask]
                # Heure ambiguë (passage à l'heure d'hiver) : on retient l'heure d'été
                .dt.tz_localize(tz, ambiguous=np.ones(mask.sum(), dtype=bool), nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )
    return result


def floor_to_hour(s: pd.Series) -> pd.Series:
    """Heure pleine tronquée (floor, pas round) : granularité de dim_weather."""
    return pd.to_datetime(s).dt.floor("h")


# --- CLI ---------------------------------------------------------------------

def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info("%s lignes écrites dans %s", len(df), path)


def run(target: str = "all") -> None:
    if target in ("airports", "all"):
        _write(build_dim_airport(), DIM_AIRPORT_PATH)
    if target in ("airlines", "all"):
        _write(build_dim_airline(), DIM_AIRLINE_PATH)
    if target in ("aircraft", "all"):
        _write(build_dim_aircraft(load_flight_tails()), DIM_AIRCRAFT_PATH)
    if target in ("weather", "all"):
        _write(build_dim_weather(), DIM_WEATHER_PATH)
    if target in ("weather-realtime", "all"):
        _write(build_weather_realtime(), WEATHER_REALTIME_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Construit les dimensions du schéma en étoile")
    parser.add_argument("target", nargs="?", default="all",
                        choices=["airports", "airlines", "aircraft", "weather", "weather-realtime", "all"])
    run(parser.parse_args().target)
