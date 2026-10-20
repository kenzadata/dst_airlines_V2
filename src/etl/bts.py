"""Transformation des vols historiques BTS (Reporting Carrier On-Time Performance).

Source : fichiers ZIP (ou CSV) téléchargés manuellement depuis transtats.bts.gov
et déposés dans `data/raw/bts/`. Pour chaque fichier : filtrage sur les
aéroports pilotes (ORIGIN), sélection des colonnes utiles, typage, puis
export d'un Parquet par mois dans `data/processed/bts/` (`bts_YYYY_MM.parquet`).

Contrairement à l'API OpenSky (`pipeline.py`), BTS fournit les horaires
programmés/réels et les causes de retard : c'est la source d'entraînement.

Usage :
    python -m src.etl.bts data/raw/bts/*.zip [--delete-source]
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd

from src.collector.airports_reference import pilot_airport_codes

logger = logging.getLogger(__name__)

RAW_BTS_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "bts"
PROCESSED_BTS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "bts"

FINAL_COLUMNS = [
    "FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID", "ORIGIN", "DEST_AIRPORT_ID", "DEST",
    "CRS_DEP_TIME", "DEP_TIME", "ARR_TIME", "CANCELLED", "DIVERTED",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "FLIGHTS", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]

INT_COLUMNS = [
    "ORIGIN_AIRPORT_ID", "DEST_AIRPORT_ID", "CRS_DEP_TIME", "DEP_TIME", "ARR_TIME",
    "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "FLIGHTS", "OP_CARRIER_FL_NUM",
]
DELAY_CAUSE_COLUMNS = [
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
DEDUP_COLUMNS = ["FL_DATE", "ORIGIN", "DEST", "OP_CARRIER", "OP_CARRIER_FL_NUM"]


def read_bts_file(filepath: str | Path) -> pd.DataFrame:
    """Lit un fichier BTS brut, ZIP (sans extraction sur disque) ou CSV."""
    path = Path(filepath)
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path, low_memory=False)

    with zipfile.ZipFile(path) as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"Aucun CSV trouvé dans {path.name}")
        # Les ZIPs BTS ne contiennent qu'un CSV, en général T_ONTIME_REPORTING.csv
        member = next((m for m in csv_members if m == "T_ONTIME_REPORTING.csv"), csv_members[0])
        with zf.open(member) as f:
            return pd.read_csv(f, low_memory=False)


def filter_flights(df: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    """Garde les colonnes utiles et les vols au départ des aéroports pilotes."""
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Colonnes absentes du fichier BTS (seront vides) : %s", missing)
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]
    filtered = df[df["ORIGIN"].isin(airports)]
    logger.info("%s lignes brutes -> %s après filtrage ORIGIN in %s", len(df), len(filtered), airports)
    return filtered.copy()


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et force le type de chaque colonne."""
    df = df.copy()
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], format="mixed", errors="coerce")

    for col in ["ORIGIN", "DEST", "OP_UNIQUE_CARRIER", "OP_CARRIER"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Heures BTS = entiers HHMM (ex. 815 pour 08h15)
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["AIR_TIME", "DISTANCE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CANCELLED", "DIVERTED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # BTS laisse les causes de retard vides quand elles ne s'appliquent pas : NaN -> 0
    for col in DELAY_CAUSE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "TAIL_NUM" in df.columns:
        df["TAIL_NUM"] = df["TAIL_NUM"].astype("string").str.strip().str.upper()

    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Déduplique, garantit le schéma de sortie et trie par date.

    Les colonnes absentes du fichier source sont créées vides : tous les
    Parquets ont ainsi le même schéma, attendu par le chargement de fact_flights.
    """
    dedup = [c for c in DEDUP_COLUMNS if c in df.columns]
    if dedup:
        df = df.drop_duplicates(subset=dedup)
    df = df.reindex(columns=FINAL_COLUMNS)
    if "FL_DATE" in df.columns:
        df = df.sort_values("FL_DATE")
    return df.reset_index(drop=True)


def derive_parquet_name(df: pd.DataFrame, filepath: str | Path) -> str:
    """Nomme le Parquet d'après la période réelle des données (`bts_YYYY_MM.parquet`).

    Les ZIPs BTS portent tous un nom du type T_ONTIME_REPORTING_<horodatage>.zip,
    sans année/mois exploitables : on lit donc FL_DATE. Repli sur le nom du
    fichier source si FL_DATE est absent ou vide.
    """
    if "FL_DATE" in df.columns and len(df) > 0:
        min_date = pd.to_datetime(df["FL_DATE"], errors="coerce").min()
        if pd.notna(min_date):
            return f"bts_{min_date.year}_{min_date.month:02d}.parquet"
    return f"{Path(filepath).stem}.parquet"


def transform_file(filepath: str | Path, airports: list[str] | None = None) -> pd.DataFrame:
    airports = airports or pilot_airport_codes()
    return finalize(clean_types(filter_flights(read_bts_file(filepath), airports)))


def process_file(
    filepath: str | Path, *, output_dir: Path = PROCESSED_BTS_DIR, delete_source: bool = False
) -> Path:
    """Transforme un fichier BTS brut et écrit son Parquet ; retourne le chemin produit."""
    df = transform_file(filepath)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / derive_parquet_name(df, filepath)
    df.to_parquet(output_path, index=False)
    logger.info("%s vols écrits dans %s", len(df), output_path)

    # Libère l'espace disque au fil de l'eau (un CSV BTS brut pèse 150-250 Mo/mois)
    if delete_source:
        os.remove(filepath)
        logger.info("Fichier source supprimé : %s", filepath)
    return output_path


def run(filepaths: list[str], delete_source: bool = False) -> list[Path]:
    return [process_file(fp, delete_source=delete_source) for fp in filepaths]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Nettoie les fichiers BTS bruts (ZIP ou CSV) vers des Parquets.")
    parser.add_argument("files", nargs="*", help=f"Fichiers ZIP/CSV (glob OK). Défaut : {RAW_BTS_DIR}/*.zip")
    parser.add_argument("--delete-source", action="store_true",
                        help="Supprime chaque fichier source après traitement.")
    args = parser.parse_args()

    patterns = args.files or [str(RAW_BTS_DIR / "*.zip")]
    filepaths = [match for pattern in patterns for match in (glob.glob(pattern) or [pattern])]
    run(filepaths, delete_source=args.delete_source)
