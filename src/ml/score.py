"""Scoring par lots via l'API : un échantillon de vols de l'entrepôt est envoyé
à `POST /predict`, et les prédictions sont confrontées au retard réellement observé.

Sert à vérifier de bout en bout le modèle *tel qu'il est servi* (API +
modèle chargé), et non le modèle en mémoire comme `evaluate.py`.

- Échantillon : `n` vols opérés tirés au hasard (graine fixe) parmi les
  `months` derniers mois de l'entrepôt. Par défaut, les 3 derniers mois,
  soit la période de test que le modèle n'a pas vue à l'entraînement.
- Chaque vol est envoyé à l'API avec les seules informations connues avant le
  départ ; le retard réel n'est utilisé qu'ensuite, pour la comparaison.
- Résultat : un DataFrame affiché dans le terminal et enregistré en CSV dans
  `data/output_scoring/scoring_<version du modèle>_<horodatage>.csv`.
- Avec `--docs` : publication dans `docs/resultats_scoring.md` (résumé, matrice de
  confusion, tableau des vols) et copie du CSV dans `docs/scoring/`.

L'API enregistre chaque prédiction dans sa table `predictions`, comme pour
tout appel à `/predict`.

Usage :
    python -m src.ml.score [--n 100] [--months 3] [--seed 42] [--api-url http://localhost:8000] [--docs]
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

from src.ml.model import DELAY_THRESHOLD_MINUTES

load_dotenv()

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output_scoring"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

# Codes condition Meteostat (`coco`), pour un rapport lisible
COCO_LABELS = {
    1: "Clair", 2: "Beau", 3: "Nuageux", 4: "Couvert", 5: "Brouillard", 6: "Brouillard givrant",
    7: "Pluie légère", 8: "Pluie", 9: "Forte pluie", 10: "Pluie verglaçante", 11: "Forte pluie verglaçante",
    12: "Grésil", 13: "Fort grésil", 14: "Neige légère", 15: "Neige", 16: "Forte neige",
    17: "Averse", 18: "Forte averse", 19: "Averse de grésil", 20: "Forte averse de grésil",
    21: "Averse de neige", 22: "Forte averse de neige", 23: "Éclairs", 24: "Grêle",
    25: "Orage", 26: "Fort orage", 27: "Tempête",
}
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Champs de la requête /predict (src/api/schemas.py) ; les autres colonnes servent à la comparaison
REQUEST_FIELDS = [
    "flight_id", "fl_date", "crs_dep_time", "origin", "dest", "op_carrier",
    "distance", "crs_elapsed_time", "aircraft_age", "temp", "prcp", "wspd", "pres", "coco",
]

# Tirage reproductible : tri sur un hachage de la clé du vol et de la graine
SAMPLE_QUERY = text("""
    SELECT
        f.op_carrier || f.op_carrier_fl_num || '_' || TO_CHAR(f.fl_date, 'YYYYMMDD') || '_' || f.origin AS flight_id,
        f.fl_date, f.crs_dep_time, f.origin, f.dest, f.op_carrier, f.tail_num,
        f.distance, f.crs_elapsed_time, a.aircraft_age,
        w.temp, w.prcp, w.wspd, w.pres, w.coco,
        f.dep_delay
    FROM fact_flights f
    LEFT JOIN dim_aircraft a ON a.tail_num = f.tail_num
    LEFT JOIN dim_weather w
           ON w.airport_iata_code = f.origin AND w.weather_time_utc = f.fl_time_utc_hour
    WHERE f.cancelled = 0
      AND f.dep_delay IS NOT NULL
      AND f.fl_date >= (SELECT DATE_TRUNC('month', MAX(fl_date)) FROM fact_flights) - make_interval(months => :months - 1)
    ORDER BY MD5(f.fl_date::text || f.op_carrier || f.op_carrier_fl_num::text || f.origin || :seed)
    LIMIT :n
""")


def sample_flights(engine, n: int = 100, months: int = 3, seed: int = 42) -> pd.DataFrame:
    """Tire `n` vols opérés des `months` derniers mois de l'entrepôt, avec leur retard réel."""
    df = pd.read_sql(SAMPLE_QUERY, engine, params={"n": n, "months": months, "seed": str(seed)})
    if df.empty:
        raise RuntimeError("Aucun vol dans l'entrepôt : lancer `just warehouse`")
    return df


def to_payload(flight: dict) -> dict:
    """Ligne de l'entrepôt -> corps JSON de /predict (valeurs manquantes -> null)."""
    payload = {}
    for field in REQUEST_FIELDS:
        value = flight.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            payload[field] = None
        elif field == "fl_date":
            payload[field] = pd.Timestamp(value).date().isoformat()
        elif field in ("crs_dep_time", "coco"):
            payload[field] = int(value)
        elif isinstance(value, np.generic):
            payload[field] = value.item()
        else:
            payload[field] = value
    return payload


def check_api(api_url: str, session: requests.Session | None = None, timeout: float = 5.0) -> str:
    """Vérifie que l'API répond et qu'un modèle est chargé ; renvoie sa version."""
    session = session or requests.Session()
    try:
        response = session.get(f"{api_url}/health", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"API injoignable sur {api_url} : {exc}. Lancer `docker compose up -d api`.") from exc
    version = response.json().get("model_version")
    if not version:
        raise RuntimeError("L'API ne sert aucun modèle : lancer `python -m src.ml.train` puis redémarrer l'API.")
    return version


def score_flights(
    flights: pd.DataFrame, api_url: str = DEFAULT_API_URL, session: requests.Session | None = None,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Appelle /predict pour chaque vol ; un échec isolé est noté dans `error` sans interrompre le lot."""
    session = session or requests.Session()
    rows = []
    for flight in flights.to_dict("records"):
        row = {"error": None}
        try:
            response = session.post(f"{api_url}/predict", json=to_payload(flight), timeout=timeout)
            response.raise_for_status()
            prediction = response.json()
            row.update({
                "delay_probability": prediction["delay_probability"],
                "predicted_delayed": prediction["predicted_delayed"],
                "predicted_delay_minutes": prediction["predicted_delay_minutes"],
                "model_version": prediction["model_version"],
            })
        except (requests.RequestException, ValueError, KeyError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)

    result = pd.concat([flights.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    result["actual_delayed"] = result["dep_delay"] >= DELAY_THRESHOLD_MINUTES
    result["actual_delay_minutes"] = result["dep_delay"].clip(lower=0)
    return result.drop(columns=["dep_delay"])


def summarize(scored: pd.DataFrame) -> dict:
    """Indicateurs du lot : prédictions vs réalité, sur les vols scorés avec succès."""
    ok = scored[scored["error"].isna()]
    if ok.empty:
        return {"scored": 0, "errors": len(scored)}
    predicted, actual = ok["predicted_delayed"].astype(bool), ok["actual_delayed"].astype(bool)
    return {
        "scored": len(ok),
        "errors": int(scored["error"].notna().sum()),
        "predicted_delayed_rate": float(predicted.mean()),
        "actual_delayed_rate": float(actual.mean()),
        "accuracy": float((predicted == actual).mean()),
        "precision": float(actual[predicted].mean()) if predicted.any() else float("nan"),
        "recall": float(predicted[actual].mean()) if actual.any() else float("nan"),
        "mean_probability_delayed": float(ok.loc[actual, "delay_probability"].mean()) if actual.any() else float("nan"),
        "mean_probability_on_time": float(ok.loc[~actual, "delay_probability"].mean()) if (~actual).any() else float("nan"),
        "mae_minutes": float((ok["predicted_delay_minutes"] - ok["actual_delay_minutes"]).abs().mean()),
    }


def save(scored: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = scored["model_version"].dropna().unique() if "model_version" in scored else []
    version = versions[0] if len(versions) else "inconnu"
    path = output_dir / f"scoring_{version}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.csv"
    scored.to_csv(path, index=False)
    return path


DISPLAY_COLUMNS = [
    "flight_id", "dest", "crs_dep_time", "temp", "prcp", "wspd", "coco",
    "delay_probability", "predicted_delayed", "actual_delayed",
    "predicted_delay_minutes", "actual_delay_minutes",
]


def _fr(value, fmt: str = ".1%") -> str:
    """Nombre au format français pour le rapport ("—" si absent)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return format(value, fmt).replace(".", ",").replace("%", " %")


def _yes_no(value) -> str:
    return "—" if pd.isna(value) else ("oui" if bool(value) else "non")


def write_docs_report(
    scored: pd.DataFrame, csv_path: Path, *, n: int, months: int, seed: int, api_url: str,
    docs_dir: Path = DOCS_DIR,
) -> Path:
    """Publie le lot dans docs/ : page Markdown (contexte, résumé, tableau) + copie du CSV."""
    summary = summarize(scored)
    ok = scored[scored["error"].isna()]
    version = ok["model_version"].iloc[0] if not ok.empty else "inconnu"
    (docs_dir / "scoring").mkdir(parents=True, exist_ok=True)
    csv_copy = docs_dir / "scoring" / csv_path.name
    scored.to_csv(csv_copy, index=False)

    predicted, actual = ok["predicted_delayed"].astype(bool), ok["actual_delayed"].astype(bool)
    period = f"{pd.Timestamp(scored['fl_date'].min()):%d/%m/%Y} au {pd.Timestamp(scored['fl_date'].max()):%d/%m/%Y}"
    lines = [
        "# Résultats de scoring via l'API",
        "",
        ("Page générée par `python -m src.ml.score --docs` ([`src/ml/score.py`](../src/ml/score.py)) : "
        f"{len(scored)} vols tirés au hasard dans l'entrepôt sont envoyés à `POST /predict`, puis chaque "
        "prédiction est confrontée au retard réellement observé."),
        "",
        "| | |",
        "|---|---|",
        f"| Date du scoring | {datetime.now(UTC):%d/%m/%Y %H:%M} UTC |",
        f"| Modèle servi par l'API | `{version}` |",
        f"| API appelée | `{api_url}` |",
        (f"| Échantillon | {len(scored)} vols opérés, tirés parmi les {months} derniers mois de l'entrepôt "
        f"(vols du {period}), graine {seed} |"),
        f"| Fichier complet | [`scoring/{csv_copy.name}`](scoring/{csv_copy.name}) |",
        "",
        ("Ces vols appartiennent à la période de test : le modèle ne les a pas vus à l'entraînement. "
        "Le retard réel n'est jamais envoyé à l'API."),
        "",
        "## Résumé",
        "",
        "| Indicateur | Valeur |",
        "|---|---|",
        f"| Vols scorés / en erreur | {summary['scored']} / {summary['errors']} |",
        f"| Vols prédits en retard (≥ 15 min) | {_fr(summary.get('predicted_delayed_rate'))} |",
        f"| Vols réellement en retard | {_fr(summary.get('actual_delayed_rate'))} |",
        f"| Exactitude | {_fr(summary.get('accuracy'))} |",
        f"| Précision (vols signalés réellement en retard) | {_fr(summary.get('precision'))} |",
        f"| Rappel (retards réels détectés) | {_fr(summary.get('recall'))} |",
        f"| Probabilité moyenne, vols réellement en retard | {_fr(summary.get('mean_probability_delayed'))} |",
        f"| Probabilité moyenne, vols à l'heure | {_fr(summary.get('mean_probability_on_time'))} |",
        f"| Erreur moyenne sur les minutes de retard | {_fr(summary.get('mae_minutes'), '.1f')} min |",
        "",
        "## Matrice de confusion",
        "",
        "| | Prédit « à l'heure » | Prédit « en retard » |",
        "|---|---|---|",
        f"| **Réellement à l'heure** | {int((~predicted & ~actual).sum())} | {int((predicted & ~actual).sum())} |",
        f"| **Réellement en retard** | {int((~predicted & actual).sum())} | {int((predicted & actual).sum())} |",
        "",
        (f"Sur {len(ok)} vols, ces indicateurs varient sensiblement d'un tirage à l'autre. Les résultats de "
        "référence, calculés sur les 103 583 vols de la période de test, sont dans le "
        "[rapport de modélisation](rapport_modelisation.md)."),
        "",
        "## Détail des vols",
        "",
        ("Trié par probabilité de retard décroissante. Météo à l'aéroport de départ à l'heure prévue "
        "(température en °C, précipitations en mm/h, vent en km/h, conditions observées)."),
        "",
        ("| Vol | Départ → arrivée | Heure prévue | Temp. | Précip. | Vent | Conditions | Probabilité | "
        "Prédit en retard | Réellement en retard | Minutes prédites | Minutes réelles |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in scored.sort_values("delay_probability", ascending=False, na_position="last").itertuples():
        hhmm = int(row.crs_dep_time) % 2400
        lines.append(
            f"| {row.flight_id} | {row.origin} → {row.dest} | {hhmm // 100:02d}:{hhmm % 100:02d} | "
            f"{_fr(row.temp, '.1f')} | {_fr(row.prcp, '.1f')} | {_fr(row.wspd, '.0f')} | "
            f"{'—' if pd.isna(row.coco) else COCO_LABELS.get(int(row.coco), int(row.coco))} | "
            f"{_fr(row.delay_probability, '.0%')} | "
            f"{_yes_no(row.predicted_delayed)} | {_yes_no(row.actual_delayed)} | "
            f"{_fr(row.predicted_delay_minutes, '.0f')} | {_fr(row.actual_delay_minutes, '.0f')} |"
        )
    if scored["error"].notna().any():
        lines += ["", "## Erreurs", ""] + [f"- `{r.flight_id}` : {r.error}" for r in scored[scored["error"].notna()].itertuples()]

    report = docs_dir / "resultats_scoring.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run(
    n: int = 100, months: int = 3, seed: int = 42, api_url: str = DEFAULT_API_URL, docs: bool = False,
) -> tuple[pd.DataFrame, Path]:
    from src.etl.warehouse import get_engine

    session = requests.Session()
    version = check_api(api_url, session)
    flights = sample_flights(get_engine(), n=n, months=months, seed=seed)
    logger.info("%s vols tirés, scoring via %s (modèle %s)", len(flights), api_url, version)
    scored = score_flights(flights, api_url, session)
    path = save(scored)

    with pd.option_context("display.max_rows", None, "display.width", 200, "display.precision", 2):
        print(scored[DISPLAY_COLUMNS].to_string(index=False))
    print("\nRésumé :", {k: round(v, 3) if isinstance(v, float) else v for k, v in summarize(scored).items()})
    print(f"Résultat enregistré dans {path}")
    if docs:
        print(f"Rapport publié dans {write_docs_report(scored, path, n=n, months=months, seed=seed, api_url=api_url)}")
    return scored, path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Score un échantillon de vols via l'API /predict")
    parser.add_argument("--n", type=int, default=100, help="Nombre de vols (défaut : 100)")
    parser.add_argument("--months", type=int, default=3, help="Tirage dans les N derniers mois de l'entrepôt")
    parser.add_argument("--seed", type=int, default=42, help="Graine du tirage (même graine = mêmes vols)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="URL de l'API (défaut : $API_URL ou localhost:8000)")
    parser.add_argument("--docs", action="store_true",
                        help="Publie aussi le résultat dans docs/resultats_scoring.md (+ CSV dans docs/scoring/)")
    args = parser.parse_args()
    try:
        run(n=args.n, months=args.months, seed=args.seed, api_url=args.api_url, docs=args.docs)
    except RuntimeError as exc:
        raise SystemExit(f"Erreur : {exc}") from None
