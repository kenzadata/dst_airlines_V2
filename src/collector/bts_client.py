"""Client de collecte des vols historiques BTS (Reporting Carrier On-Time Performance).

Télécharge les fichiers mensuels pré-zippés publiés par le BTS (toutes les
compagnies US, toutes les colonnes, y compris DepDelay/ArrDelay) et la table
de décodage des compagnies (L_UNIQUE_CARRIERS). Le filtrage sur les aéroports
pilotes et le typage sont faits par `src/etl/bts.py`.

Usage :
    python -m src.collector.bts_client --start 2024-01 --end 2025-12
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PREZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
CARRIERS_LOOKUP_URL = "https://www.transtats.bts.gov/Download_Lookup.asp?Y11x72=Y_haVdhR_PNeeVRef"

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_BTS_DIR = ROOT_DIR / "data" / "raw" / "bts"
CARRIERS_LOOKUP_PATH = ROOT_DIR / "data" / "reference" / "L_UNIQUE_CARRIERS.csv"


def _download(url: str, output_path: Path, *, timeout: float = 120.0, max_retries: int = 3) -> Path | None:
    """Télécharge `url` en streaming vers `output_path` (écriture atomique).

    Aucune exception réseau n'est propagée : on logue et on retourne None
    (README section 10.9).
    """
    tmp_path = output_path.with_name(output_path.name + ".part")
    backoff = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp_path.replace(output_path)
            return output_path
        except requests.RequestException as exc:
            logger.warning("BTS: échec du téléchargement de %s (tentative %s/%s) : %s",
                           url, attempt, max_retries, exc)
            tmp_path.unlink(missing_ok=True)
            time.sleep(backoff)
            backoff *= 2
    logger.error("BTS: abandon du téléchargement de %s après %s tentatives", url, max_retries)
    return None


def download_month(year: int, month: int, *, output_dir: Path = RAW_BTS_DIR) -> Path | None:
    """Télécharge le fichier BTS d'un mois ; ne retélécharge pas un fichier déjà présent."""
    output_path = output_dir / f"bts_raw_{year}_{month:02d}.zip"
    if output_path.exists():
        logger.info("Déjà présent : %s", output_path.name)
        return output_path
    path = _download(PREZIP_URL.format(year=year, month=month), output_path)
    if path:
        logger.info("BTS %s-%02d téléchargé -> %s (%.1f Mo)", year, month, path, path.stat().st_size / 1e6)
    return path


def download_period(start: str, end: str, *, output_dir: Path = RAW_BTS_DIR) -> list[Path]:
    """Télécharge tous les mois entre `start` et `end` inclus (format AAAA-MM)."""
    months = pd.period_range(start=start, end=end, freq="M")
    paths = [download_month(p.year, p.month, output_dir=output_dir) for p in months]
    return [p for p in paths if p is not None]


def download_carriers_lookup(*, output_path: Path = CARRIERS_LOOKUP_PATH) -> Path | None:
    """Table de décodage des codes compagnies (colonnes `Code`, `Description`)."""
    path = _download(CARRIERS_LOOKUP_URL, output_path, timeout=30.0)
    if path:
        logger.info("Table des compagnies sauvegardée -> %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Télécharge les vols BTS mensuels et la table des compagnies")
    parser.add_argument("--start", default="2024-01", help="Premier mois (AAAA-MM)")
    parser.add_argument("--end", default="2025-12", help="Dernier mois inclus (AAAA-MM)")
    args = parser.parse_args()

    download_carriers_lookup()
    download_period(args.start, args.end)
